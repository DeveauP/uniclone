"""
uniclone.simulate.bamsurgeon_wrap
==================================

Wrapper around BAMSurgeon for MetaRouter training corpus generation.

Spikes synthetic somatic SNVs into a template (normal) BAM at known VAFs,
then counts alt/ref reads back via pysam to produce training data with
realistic sequencing artefacts (mapping noise, strand bias, base-quality
variation) that QuantumCat's pure-binomial model cannot capture.

Environment variables
---------------------
UNICLONE_BAMSURGEON_NORMAL_BAM : str
    Path to the template normal BAM (must be indexed).
UNICLONE_BAMSURGEON_REFERENCE : str
    Path to the reference FASTA (must be indexed with .fai).
UNICLONE_BAMSURGEON_BIN : str, optional
    Directory containing BAMSurgeon scripts (addsnv.py, etc.).
    If unset, they must be on $PATH.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import pysam

    PYSAM_AVAILABLE = True
except ImportError:
    pysam = None
    PYSAM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class BAMSurgeonParams:
    """Parameters for a single BAMSurgeon spike-in tumour."""

    n_clones: int = 3
    n_mutations: int = 500
    n_samples: int = 1
    purity: float = 0.8
    depth: int = 100
    clone_fractions: np.ndarray | None = None
    cn_profile: np.ndarray | None = None
    seed: int | None = None

    # BAMSurgeon-specific
    normal_bam: str | None = None  # override env var
    reference_fasta: str | None = None  # override env var

    def __post_init__(self) -> None:
        if self.n_clones < 1:
            raise ValueError(f"n_clones must be >= 1, got {self.n_clones}")
        if self.n_mutations < 1:
            raise ValueError(f"n_mutations must be >= 1, got {self.n_mutations}")


@dataclass
class BAMSurgeonResult:
    """Output of a BAMSurgeon simulation — same interface as QuantumCatResult."""

    alt: np.ndarray  # (n_mut, n_samples)
    depth: np.ndarray  # (n_mut, n_samples)
    adj_factor: np.ndarray  # (n_mut, n_samples)
    true_assignments: np.ndarray  # (n_mut,)
    true_centers: np.ndarray  # (n_clones, n_samples)
    true_tree: np.ndarray | None  # (n_clones, n_clones) adjacency or None
    params: BAMSurgeonParams


# ---------------------------------------------------------------------------
# Main simulation entry-point
# ---------------------------------------------------------------------------


def simulate_bamsurgeon(params: BAMSurgeonParams) -> BAMSurgeonResult:
    """
    Spike synthetic SNVs into a normal BAM via BAMSurgeon and read back counts.

    Pipeline per tumour:
    1. Draw clone structure (cellularities, assignments) — same priors as QuantumCat
    2. Sample genomic positions from the reference .fai
    3. Write a BAMSurgeon variant file with target VAFs
    4. Run ``addsnv.py`` to produce a spiked BAM
    5. Count alt/ref reads at spiked loci with pysam
    6. Return arrays compatible with the training pipeline

    Parameters
    ----------
    params : BAMSurgeonParams

    Returns
    -------
    BAMSurgeonResult
    """
    if not PYSAM_AVAILABLE:
        raise ImportError("BAMSurgeon wrapper requires pysam: pip install pysam")

    normal_bam = _resolve_path(
        params.normal_bam,
        "UNICLONE_BAMSURGEON_NORMAL_BAM",
        "normal BAM",
    )
    reference = _resolve_path(
        params.reference_fasta,
        "UNICLONE_BAMSURGEON_REFERENCE",
        "reference FASTA",
    )
    addsnv = _find_addsnv()

    rng = np.random.default_rng(params.seed)
    K = params.n_clones
    N = params.n_mutations
    S = params.n_samples

    # ---- 1. Clone structure (mirrors QuantumCat logic) --------------------
    if params.clone_fractions is not None:
        fracs = np.asarray(params.clone_fractions, dtype=np.float64)
    else:
        fracs = rng.dirichlet(np.ones(K))

    counts = np.floor(fracs * N).astype(int)
    remainder = N - counts.sum()
    fractional = (fracs * N) - counts
    top_idx = np.argsort(fractional)[::-1][:remainder]
    counts[top_idx] += 1
    true_assignments = np.repeat(np.arange(K), counts)

    true_centers = np.zeros((K, S), dtype=np.float64)
    for s in range(S):
        if s == 0:
            ordered = np.sort(rng.beta(2, 2, size=K))[::-1]
            true_centers[:, s] = ordered * params.purity
        else:
            noise = rng.normal(0, 0.05, size=K)
            true_centers[:, s] = np.clip(
                true_centers[:, 0] + noise,
                0.01,
                params.purity,
            )
    true_centers[0, :] = params.purity

    # ---- 2. Copy-number / adj_factor (same as QuantumCat) -----------------
    if params.cn_profile is not None:
        cn = np.asarray(params.cn_profile, dtype=np.float64)
        normal_cn = 2.0
        total_cn = cn if cn.ndim == 1 else cn[:, 0]
        adj = normal_cn / (params.purity * total_cn + (1 - params.purity) * normal_cn)
        adj_factor = np.broadcast_to(adj[:, np.newaxis], (N, S)).copy()
    else:
        adj_factor = np.ones((N, S), dtype=np.float64)
        n_cn_altered = max(1, int(0.2 * N))
        cn_idx = rng.choice(N, size=n_cn_altered, replace=False)
        total_cn = rng.choice([1, 3, 4], size=n_cn_altered, p=[0.3, 0.5, 0.2])
        normal_cn = 2.0
        adj = normal_cn / (params.purity * total_cn + (1 - params.purity) * normal_cn)
        adj_factor[cn_idx, :] = adj[:, np.newaxis]

    # ---- 3. Target VAFs per mutation per sample ---------------------------
    mutation_cellularity = true_centers[true_assignments]  # (N, S)
    target_vaf = np.clip(mutation_cellularity * adj_factor, 0.001, 0.999)

    # ---- 4. Sample spike-in positions from reference ----------------------
    positions = _sample_positions(reference, N, rng)

    # ---- 5. Run BAMSurgeon per sample & read back counts ------------------
    alt_arr = np.zeros((N, S), dtype=np.float64)
    depth_arr = np.zeros((N, S), dtype=np.float64)

    for s in range(S):
        alt_s, depth_s = _run_spike_and_count(
            addsnv=addsnv,
            normal_bam=normal_bam,
            reference=reference,
            positions=positions,
            target_vafs=target_vaf[:, s],
            rng=rng,
        )
        alt_arr[:, s] = alt_s
        depth_arr[:, s] = depth_s

    # Ensure no zero-depth (same guard as QuantumCat)
    np.maximum(depth_arr, 1, out=depth_arr)

    # ---- 6. Tree (simple linear chain) ------------------------------------
    true_tree = np.zeros((K, K), dtype=int)
    for k in range(K - 1):
        true_tree[k, k + 1] = 1

    return BAMSurgeonResult(
        alt=alt_arr,
        depth=depth_arr,
        adj_factor=adj_factor,
        true_assignments=true_assignments,
        true_centers=true_centers,
        true_tree=true_tree if K > 1 else None,
        params=params,
    )


def sample_tumour_params(rng: np.random.Generator) -> BAMSurgeonParams:
    """
    Draw tumour simulation parameters from realistic priors.

    Same distributions as QuantumCat's ``sample_tumour_params`` but returns
    a ``BAMSurgeonParams`` instance.
    """
    n_clones = int(rng.choice([1, 2, 3, 4, 5], p=[0.1, 0.3, 0.3, 0.2, 0.1]))
    n_mutations = int(rng.choice([100, 200, 500, 1000, 2000], p=[0.1, 0.2, 0.3, 0.3, 0.1]))
    n_samples = int(rng.choice([1, 2, 3], p=[0.5, 0.3, 0.2]))
    purity = float(rng.beta(5, 2))
    depth = int(np.exp(rng.normal(np.log(100), 0.5)))
    depth = max(20, min(depth, 1000))
    seed = int(rng.integers(0, 2**31))

    return BAMSurgeonParams(
        n_clones=n_clones,
        n_mutations=n_mutations,
        n_samples=n_samples,
        purity=purity,
        depth=depth,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------


def is_available() -> bool:
    """Return True if BAMSurgeon and pysam are installed and paths are set."""
    if not PYSAM_AVAILABLE:
        return False
    try:
        _find_addsnv()
        _resolve_path(None, "UNICLONE_BAMSURGEON_NORMAL_BAM", "normal BAM")
        _resolve_path(None, "UNICLONE_BAMSURGEON_REFERENCE", "reference FASTA")
    except (OSError, FileNotFoundError):
        return False
    return True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_path(
    explicit: str | None,
    env_var: str,
    label: str,
) -> str:
    """Resolve a path from explicit value or environment variable."""
    path = explicit or os.environ.get(env_var)
    if not path:
        raise OSError(f"No {label} path provided. Set {env_var} or pass explicitly.")
    if not Path(path).exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _find_addsnv() -> str:
    """Locate BAMSurgeon's addsnv.py script."""
    bamsurgeon_bin = os.environ.get("UNICLONE_BAMSURGEON_BIN", "")
    if bamsurgeon_bin:
        candidate = Path(bamsurgeon_bin) / "addsnv.py"
        if candidate.exists():
            return str(candidate)

    # Try PATH
    found = shutil.which("addsnv.py")
    if found:
        return found

    raise FileNotFoundError(
        "Cannot find addsnv.py. Install BAMSurgeon and ensure addsnv.py is "
        "on $PATH, or set UNICLONE_BAMSURGEON_BIN to the BAMSurgeon bin dir."
    )


def _sample_positions(
    reference: str,
    n: int,
    rng: np.random.Generator,
) -> list[tuple[str, int, str, str]]:
    """
    Sample *n* random autosomal positions from the reference .fai index.

    Returns list of (chrom, pos_1based, ref_base, alt_base).
    """
    fai_path = reference + ".fai"
    if not Path(fai_path).exists():
        raise FileNotFoundError(f"Reference index not found: {fai_path}. Run samtools faidx.")

    # Parse .fai → list of (chrom, length)
    chroms: list[tuple[str, int]] = []
    with open(fai_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            chrom, length = parts[0], int(parts[1])
            # Autosomes only (chr1-22 or 1-22)
            name = chrom.replace("chr", "")
            if name.isdigit() and 1 <= int(name) <= 22:
                chroms.append((chrom, length))

    if not chroms:
        raise ValueError("No autosomal contigs found in reference index.")

    # Weight by chromosome length
    lengths = np.array([l for _, l in chroms], dtype=np.float64)
    probs = lengths / lengths.sum()
    chrom_idx = rng.choice(len(chroms), size=n, p=probs)

    complement = {"A": "T", "T": "A", "C": "G", "G": "C"}
    positions = []
    ref_fasta = pysam.FastaFile(reference)
    try:
        for ci in chrom_idx:
            chrom, chrom_len = chroms[ci]
            # Avoid telomeric regions (first/last 10kb)
            margin = 10_000
            pos = int(rng.integers(margin, max(chrom_len - margin, margin + 1)))
            ref_base = ref_fasta.fetch(chrom, pos - 1, pos).upper()
            if ref_base not in complement:
                # Skip ambiguous bases — pick a simple substitution
                ref_base = "C"
            alt_base = complement[ref_base]
            positions.append((chrom, pos, ref_base, alt_base))
    finally:
        ref_fasta.close()

    return positions


def _write_variant_file(
    positions: list[tuple[str, int, str, str]],
    target_vafs: np.ndarray,
    path: str,
) -> None:
    """
    Write BAMSurgeon addsnv.py variant input file.

    Format: chrom  start  end  vaf  ref>alt
    (1-based, tab-delimited)
    """
    with open(path, "w") as f:
        for i, (chrom, pos, ref_base, alt_base) in enumerate(positions):
            vaf = float(target_vafs[i])
            f.write(f"{chrom}\t{pos}\t{pos}\t{vaf:.4f}\t{ref_base}>{alt_base}\n")


def _run_spike_and_count(
    *,
    addsnv: str,
    normal_bam: str,
    reference: str,
    positions: list[tuple[str, int, str, str]],
    target_vafs: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run BAMSurgeon addsnv.py, then count alt/ref at spiked positions.

    Returns (alt_counts, depth_counts) arrays of shape (n_mutations,).
    """
    n = len(positions)

    with tempfile.TemporaryDirectory(prefix="uniclone_bamsurgeon_") as tmpdir:
        var_file = os.path.join(tmpdir, "variants.txt")
        _write_variant_file(positions, target_vafs, var_file)

        out_bam = os.path.join(tmpdir, "spiked.bam")
        seed_val = int(rng.integers(0, 2**31))

        cmd = [
            "python",
            addsnv,
            "--varfile",
            var_file,
            "--bamfile",
            normal_bam,
            "--reference",
            reference,
            "--outbam",
            out_bam,
            "--seed",
            str(seed_val),
            "--mindepth",
            "1",
            "--maxdepth",
            "10000",
            "--aligner",
            "mem",
            "--picardjar",
            _find_picard_jar(),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"BAMSurgeon addsnv.py failed (rc={result.returncode}):\n{result.stderr[:2000]}"
            )

        # Index the output BAM
        pysam.sort("-o", out_bam + ".sorted.bam", out_bam)
        sorted_bam = out_bam + ".sorted.bam"
        pysam.index(sorted_bam)

        # Count reads at spiked positions
        alt_counts = np.zeros(n, dtype=np.float64)
        depth_counts = np.zeros(n, dtype=np.float64)

        samfile = pysam.AlignmentFile(sorted_bam, "rb")
        try:
            for i, (chrom, pos, _ref_base, alt_base) in enumerate(positions):
                a, d = _count_alleles(samfile, chrom, pos, alt_base)
                alt_counts[i] = a
                depth_counts[i] = d
        finally:
            samfile.close()

    return alt_counts, depth_counts


def _count_alleles(
    samfile: pysam.AlignmentFile,
    chrom: str,
    pos: int,
    alt_base: str,
) -> tuple[int, int]:
    """Count alt and total reads at a single position via pileup."""
    alt_base_upper = alt_base.upper()
    alt_count = 0
    total_count = 0

    # pysam uses 0-based coordinates
    for pileup_col in samfile.pileup(
        chrom,
        pos - 1,
        pos,
        min_base_quality=0,
        truncate=True,
    ):
        if pileup_col.reference_pos == pos - 1:
            for read in pileup_col.pileups:
                if read.is_del or read.is_refskip:
                    continue
                total_count += 1
                base = read.alignment.query_sequence[read.query_position].upper()
                if base == alt_base_upper:
                    alt_count += 1
            break

    return alt_count, total_count


def _find_picard_jar() -> str:
    """Locate picard.jar from env or common locations."""
    env_path = os.environ.get("PICARD_JAR")
    if env_path and Path(env_path).exists():
        return env_path

    common: list[str] = [
        "/usr/local/share/picard/picard.jar",
        "/usr/share/picard/picard.jar",
        str(Path.home() / "picard.jar"),
    ]
    for p in common:
        if Path(p).exists():
            return str(p)

    # BAMSurgeon can work without picard for addsnv in some configurations
    return ""
