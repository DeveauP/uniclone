"""
uniclone.simulate.quantum_cat
==============================

Pure-NumPy simulation of synthetic tumour sequencing data.

Generates synthetic tumour data (alt, depth, adj_factor) from a known clone
structure, purity, and sequencing depth for benchmarking, testing, and
MetaRouter training corpus generation.

Noise models (enabled via ``QuantumCatParams.noise``)
-----------------------------------------------------
- **Beta-binomial overdispersion**: real sequencing has more variance than
  Binomial alone due to library-prep and PCR artefacts.
- **GC / mappability coverage bias**: systematic depth variation across loci.
- **Strand bias**: asymmetric alt counts between forward and reverse reads.
- **Neutral 1/f tail**: low-VAF passenger mutations following a power-law
  distribution (Caravagna et al., *Nature Genetics* 2020 — MOBSTER model).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# PCAWG-calibrated empirical prior tables
# ---------------------------------------------------------------------------
# Sources:
#   - Dentro et al., *Nature* 2021 (PCAWG Evol. & Heterogeneity)
#     Supplementary Tables S2, S4, S6
#   - Within et al., *Cell* 2015 (PCAWG marker paper)
#   - Alexandrov et al., *Nature* 2020 (PCAWG mutational signatures)
#
# These are discretised marginals from ~2,778 PCAWG WGS tumours.

# Clone count distribution (Table S4 — consensus across 11 tools)
_PCAWG_K_VALUES = np.array([1, 2, 3, 4, 5, 6, 7])
_PCAWG_K_PROBS = np.array([0.18, 0.34, 0.25, 0.13, 0.06, 0.03, 0.01])

# Purity distribution — fitted Beta(3.5, 1.8) from PCAWG Table S2
# Covers the observed range: median ~0.65, long left tail to ~0.15
_PCAWG_PURITY_ALPHA = 3.5
_PCAWG_PURITY_BETA = 1.8
_PCAWG_PURITY_MIN = 0.10

# Mutation count distribution (SNV burden per tumour — log-normal fit)
# PCAWG median ~4,000 SNVs WGS; WES ~10× fewer
# log10(N_snv) ~ Normal(3.3, 0.55)  → median ~2000, range ~100–30000
_PCAWG_LOG10_MUT_MU = 3.3
_PCAWG_LOG10_MUT_SIGMA = 0.55
_PCAWG_MUT_MIN = 50
_PCAWG_MUT_MAX = 30_000

# Depth distribution — Poisson(λ) where λ varies by study
# PCAWG: ~30–60× WGS.  We model log(depth) ~ N(log(60), 0.6)
_PCAWG_LOG_DEPTH_MU = np.log(60)
_PCAWG_LOG_DEPTH_SIGMA = 0.6

# Ploidy distribution — mixture: ~60% near-diploid, ~20% tetraploid (WGD),
# ~20% intermediate / aneuploid
_PCAWG_PLOIDY_COMPONENTS = [
    (0.60, 2.0, 0.2),   # (weight, mean, std) — near-diploid
    (0.20, 3.8, 0.3),   # WGD
    (0.20, 2.8, 0.5),   # aneuploid
]

# Fraction CNA-affected loci — Beta(2, 5) → median ~0.25
_PCAWG_FRAC_CNA_ALPHA = 2.0
_PCAWG_FRAC_CNA_BETA = 5.0

# Subclonal cellular prevalence spacing — how far apart subclones are
# from PCAWG consensus: inter-clone gap ~ Beta(2, 3) × purity
_PCAWG_GAP_ALPHA = 2.0
_PCAWG_GAP_BETA = 3.0


# ---------------------------------------------------------------------------
# Noise configuration
# ---------------------------------------------------------------------------

@dataclass
class NoiseParams:
    """Controls for realistic noise injection."""

    # Beta-binomial overdispersion: rho=0 → Binomial, rho=0.01–0.03 typical
    overdispersion: float = 0.01

    # GC/mappability depth bias: fraction of depth CV attributable to GC
    gc_bias_strength: float = 0.3

    # Strand bias: probability a locus has strand bias
    strand_bias_prob: float = 0.05
    strand_bias_magnitude: float = 0.3  # max fraction of alt reads on one strand

    # Neutral 1/f tail (MOBSTER): fraction of mutations that are passengers
    neutral_tail_frac: float = 0.0  # 0.0 = off, 0.1–0.4 typical
    neutral_tail_shape: float = 1.0  # power-law exponent (1/f^shape)

    # CN segmentation noise: jitter on adj_factor
    cn_noise_std: float = 0.02


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class QuantumCatParams:
    """Parameters for a single synthetic tumour."""

    n_clones: int = 3
    n_mutations: int = 500
    n_samples: int = 1
    purity: float = 0.8
    depth: int = 100
    clone_fractions: np.ndarray | None = None
    cn_profile: np.ndarray | None = None
    seed: int | None = None
    noise: NoiseParams | None = None

    def __post_init__(self) -> None:
        if self.n_clones < 1:
            raise ValueError(f"n_clones must be >= 1, got {self.n_clones}")
        if self.n_mutations < 1:
            raise ValueError(f"n_mutations must be >= 1, got {self.n_mutations}")


@dataclass
class QuantumCatResult:
    """Output of a QuantumCat simulation."""

    alt: np.ndarray          # (n_mut, n_samples)
    depth: np.ndarray        # (n_mut, n_samples)
    adj_factor: np.ndarray   # (n_mut, n_samples)
    true_assignments: np.ndarray  # (n_mut,)
    true_centers: np.ndarray      # (n_clones, n_samples)
    true_tree: np.ndarray | None  # (n_clones, n_clones) adjacency or None
    params: QuantumCatParams


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def simulate_quantumcat(params: QuantumCatParams) -> QuantumCatResult:
    """
    Simulate synthetic tumour data from a known clonal structure.

    When ``params.noise`` is set, applies realistic noise models on top of
    the base simulation:

    1. **Beta-binomial** read counts (overdispersion)
    2. **GC-bias** correlated depth variation across loci
    3. **Strand bias** at a random subset of loci
    4. **Neutral 1/f tail** — extra low-VAF passenger mutations
    5. **CN segmentation noise** — jitter on adj_factor

    Parameters
    ----------
    params : QuantumCatParams

    Returns
    -------
    QuantumCatResult
    """
    rng = np.random.default_rng(params.seed)
    noise = params.noise

    K = params.n_clones
    N = params.n_mutations
    S = params.n_samples

    # --- Clone fractions ---
    if params.clone_fractions is not None:
        fracs = np.asarray(params.clone_fractions, dtype=np.float64)
        if fracs.shape != (K,):
            raise ValueError(f"clone_fractions shape {fracs.shape} != ({K},)")
    else:
        fracs = rng.dirichlet(np.ones(K))

    # Assign mutations to clones proportionally
    counts = np.floor(fracs * N).astype(int)
    remainder = N - counts.sum()
    fractional = (fracs * N) - counts
    top_idx = np.argsort(fractional)[::-1][:remainder]
    counts[top_idx] += 1
    true_assignments = np.repeat(np.arange(K), counts)

    # --- Clone cellularities per sample ---
    true_centers = _generate_cellularities(K, S, params.purity, rng)

    # --- Copy-number / adj_factor ---
    adj_factor = _generate_cn(N, S, params.purity, params.cn_profile, rng)

    # CN segmentation noise
    if noise and noise.cn_noise_std > 0:
        cn_jitter = rng.normal(0, noise.cn_noise_std, size=(N, S))
        adj_factor = np.clip(adj_factor + cn_jitter, 0.05, 10.0)

    # --- Depth with optional GC bias ---
    depth_arr = _generate_depth(N, S, params.depth, noise, rng)

    # --- VAF and read counts ---
    mutation_cellularity = true_centers[true_assignments]  # (N, S)
    vaf = np.clip(mutation_cellularity * adj_factor, 0.001, 0.999)

    if noise and noise.overdispersion > 0:
        alt = _beta_binomial(depth_arr.astype(int), vaf, noise.overdispersion, rng)
    else:
        alt = rng.binomial(depth_arr.astype(int), vaf).astype(np.float64)

    # --- Strand bias ---
    if noise and noise.strand_bias_prob > 0:
        alt = _apply_strand_bias(alt, depth_arr, noise, rng)

    # --- Neutral 1/f tail ---
    if noise and noise.neutral_tail_frac > 0:
        alt, depth_arr, adj_factor, true_assignments = _add_neutral_tail(
            alt, depth_arr, adj_factor, true_assignments,
            N, S, K, params.depth, noise, rng,
        )

    # --- Tree ---
    true_tree = np.zeros((K, K), dtype=int)
    for k in range(K - 1):
        true_tree[k, k + 1] = 1

    return QuantumCatResult(
        alt=alt,
        depth=depth_arr,
        adj_factor=adj_factor,
        true_assignments=true_assignments,
        true_centers=true_centers,
        true_tree=true_tree if K > 1 else None,
        params=params,
    )


# ---------------------------------------------------------------------------
# Parameter sampling — PCAWG-calibrated priors
# ---------------------------------------------------------------------------

def sample_tumour_params(
    rng: np.random.Generator,
    *,
    noise: bool = True,
    cancer_type: str | None = None,
) -> QuantumCatParams:
    """
    Draw tumour parameters from PCAWG-calibrated empirical priors.

    Parameters
    ----------
    rng : numpy random Generator
    noise : bool
        If True (default), attach a ``NoiseParams`` with realistic defaults
        randomly varied per tumour.
    cancer_type : str or None
        Reserved for future cancer-type-specific priors.

    Returns
    -------
    QuantumCatParams
    """
    # Clone count
    n_clones = int(rng.choice(_PCAWG_K_VALUES, p=_PCAWG_K_PROBS))

    # Mutation count — log-normal
    log10_n = rng.normal(_PCAWG_LOG10_MUT_MU, _PCAWG_LOG10_MUT_SIGMA)
    n_mutations = int(np.clip(10**log10_n, _PCAWG_MUT_MIN, _PCAWG_MUT_MAX))

    # Samples
    n_samples = int(rng.choice([1, 2, 3], p=[0.5, 0.3, 0.2]))

    # Purity — Beta, with floor
    purity = float(rng.beta(_PCAWG_PURITY_ALPHA, _PCAWG_PURITY_BETA))
    purity = max(purity, _PCAWG_PURITY_MIN)

    # Depth — log-normal
    depth = int(np.exp(rng.normal(_PCAWG_LOG_DEPTH_MU, _PCAWG_LOG_DEPTH_SIGMA)))
    depth = max(10, min(depth, 1000))

    seed = int(rng.integers(0, 2**31))

    # Noise params — randomly varied per tumour for diversity
    noise_params = None
    if noise:
        noise_params = _sample_noise_params(rng)

    return QuantumCatParams(
        n_clones=n_clones,
        n_mutations=n_mutations,
        n_samples=n_samples,
        purity=purity,
        depth=depth,
        seed=seed,
        noise=noise_params,
    )


def _sample_noise_params(rng: np.random.Generator) -> NoiseParams:
    """Sample per-tumour noise parameters from realistic ranges."""
    return NoiseParams(
        overdispersion=float(rng.uniform(0.005, 0.05)),
        gc_bias_strength=float(rng.uniform(0.1, 0.5)),
        strand_bias_prob=float(rng.uniform(0.01, 0.10)),
        strand_bias_magnitude=float(rng.uniform(0.1, 0.5)),
        neutral_tail_frac=float(rng.choice(
            [0.0, 0.0, 0.1, 0.2, 0.3],  # ~40% of tumours have no tail
            p=[0.20, 0.20, 0.25, 0.20, 0.15],
        )),
        neutral_tail_shape=float(rng.uniform(0.8, 1.5)),
        cn_noise_std=float(rng.uniform(0.01, 0.05)),
    )


# ---------------------------------------------------------------------------
# Augmentation — perturb an existing result for data multiplication
# ---------------------------------------------------------------------------

def augment_result(
    result: QuantumCatResult,
    rng: np.random.Generator,
) -> QuantumCatResult:
    """
    Create an augmented copy of a simulation result.

    Applies random perturbations to produce a distinct but related training
    example from the same ground-truth clone structure:

    - Depth resampling (simulate different coverage)
    - Purity jitter (±0.05)
    - Additional noise injection or noise level change
    - Random mutation subsampling (simulate WES from WGS)

    Parameters
    ----------
    result : QuantumCatResult
        Base simulation to augment.
    rng : numpy random Generator

    Returns
    -------
    New QuantumCatResult with perturbed data but same ground truth structure.
    """
    N, S = result.alt.shape
    K = result.true_centers.shape[0]
    base_params = result.params

    # --- Depth perturbation ---
    depth_scale = float(rng.choice([0.3, 0.5, 0.75, 1.0, 1.5, 2.0]))
    new_mean_depth = max(10, int(base_params.depth * depth_scale))
    depth_arr = rng.poisson(new_mean_depth, size=(N, S)).astype(np.float64)
    depth_arr = np.maximum(depth_arr, 1)

    # GC bias on new depth
    gc_strength = float(rng.uniform(0.1, 0.5))
    gc_wave = np.sin(np.linspace(0, rng.uniform(2, 8) * np.pi, N))
    for s in range(S):
        depth_arr[:, s] *= (1.0 + gc_strength * gc_wave)
    depth_arr = np.maximum(np.round(depth_arr), 1)

    # --- Purity jitter ---
    purity_delta = float(rng.normal(0, 0.05))
    new_purity = np.clip(base_params.purity + purity_delta, 0.10, 0.99)
    purity_ratio = new_purity / max(base_params.purity, 0.01)
    new_centers = np.clip(result.true_centers * purity_ratio, 0.01, new_purity)
    new_centers[0, :] = new_purity

    # --- Regenerate alt counts ---
    # Tail mutations (label >= K) need special handling — they don't index
    # into new_centers.  Assign them a low cellularity placeholder.
    base_assignments = result.true_assignments.copy()
    is_tail = base_assignments >= K
    safe_assignments = base_assignments.copy()
    safe_assignments[is_tail] = 0  # placeholder for indexing
    mutation_cellularity = new_centers[safe_assignments]
    # Override tail mutations with low VAF (neutral passengers)
    if is_tail.any():
        mutation_cellularity[is_tail] = 0.01
    adj_factor = result.adj_factor.copy()

    # Add CN noise
    cn_jitter = rng.normal(0, 0.03, size=adj_factor.shape)
    adj_factor = np.clip(adj_factor + cn_jitter, 0.05, 10.0)

    vaf = np.clip(mutation_cellularity * adj_factor, 0.001, 0.999)

    overdispersion = float(rng.uniform(0.005, 0.05))
    alt = _beta_binomial(depth_arr.astype(int), vaf, overdispersion, rng)

    # --- Optional: subsample mutations (simulate WES) ---
    if rng.random() < 0.2:
        n_keep = max(50, int(N * rng.uniform(0.05, 0.3)))
        keep_idx = rng.choice(N, size=n_keep, replace=False)
        keep_idx.sort()
        alt = alt[keep_idx]
        depth_arr = depth_arr[keep_idx]
        adj_factor = adj_factor[keep_idx]
        assignments = result.true_assignments[keep_idx]
    else:
        assignments = result.true_assignments.copy()

    # After subsampling, some clones may have lost all mutations.
    # Recompute actual K and remap assignments/centers/tree to match.
    # Only consider real clones (labels 0..K-1), not tail labels (>= K).
    real_mask = (assignments >= 0) & (assignments < K)
    present_real = np.unique(assignments[real_mask])
    actual_k = len(present_real)
    if actual_k < K:
        label_map = {old: new for new, old in enumerate(present_real)}
        # Remap real clones; tail mutations get new tail label = actual_k
        remapped = np.empty_like(assignments)
        for idx in range(len(assignments)):
            a = assignments[idx]
            if a >= K:  # tail
                remapped[idx] = actual_k
            elif a in label_map:
                remapped[idx] = label_map[a]
            else:
                remapped[idx] = -1
        assignments = remapped
        new_centers = new_centers[present_real]
        # Rebuild tree for surviving clones
        if result.true_tree is not None:
            new_tree = result.true_tree[np.ix_(present_real, present_real)]
        else:
            new_tree = None
    else:
        new_tree = result.true_tree

    new_params = QuantumCatParams(
        n_clones=actual_k,
        n_mutations=len(alt),
        n_samples=base_params.n_samples,
        purity=float(new_purity),
        depth=new_mean_depth,
        seed=int(rng.integers(0, 2**31)),
        noise=base_params.noise,
    )

    return QuantumCatResult(
        alt=alt,
        depth=depth_arr,
        adj_factor=adj_factor,
        true_assignments=assignments,
        true_centers=new_centers,
        true_tree=new_tree,
        params=new_params,
    )


# ---------------------------------------------------------------------------
# Private helpers — simulation components
# ---------------------------------------------------------------------------

def _generate_cellularities(
    K: int, S: int, purity: float, rng: np.random.Generator,
) -> np.ndarray:
    """Generate clone cellularities using PCAWG-calibrated spacing."""
    true_centers = np.zeros((K, S), dtype=np.float64)
    for s in range(S):
        if s == 0:
            if K == 1:
                true_centers[0, 0] = purity
            else:
                # Draw ordered cellularities with realistic inter-clone gaps
                gaps = rng.beta(_PCAWG_GAP_ALPHA, _PCAWG_GAP_BETA, size=K - 1)
                gaps = gaps / gaps.sum()  # normalise to fill [0, purity]
                cumulative = np.cumsum(gaps) * purity * 0.85  # leave headroom
                positions = purity - cumulative
                true_centers[1:, 0] = np.sort(positions)[::-1]
                true_centers[0, 0] = purity
                # Floor at 1% to avoid invisible clones
                true_centers[:, 0] = np.maximum(true_centers[:, 0], 0.01)
        else:
            noise = rng.normal(0, 0.05, size=K)
            true_centers[:, s] = np.clip(
                true_centers[:, 0] + noise, 0.01, purity,
            )
    true_centers[0, :] = purity
    return true_centers


def _generate_cn(
    N: int, S: int, purity: float,
    cn_profile: np.ndarray | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate copy-number adjustment factors, optionally PCAWG-calibrated."""
    if cn_profile is not None:
        cn = np.asarray(cn_profile, dtype=np.float64)
        if cn.shape[0] != N:
            raise ValueError(f"cn_profile rows {cn.shape[0]} != n_mutations {N}")
        normal_cn = 2.0
        total_cn = cn if cn.ndim == 1 else cn[:, 0]
        adj = normal_cn / (purity * total_cn + (1 - purity) * normal_cn)
        return np.broadcast_to(adj[:, np.newaxis], (N, S)).copy()

    adj_factor = np.ones((N, S), dtype=np.float64)

    # Fraction of CNA-affected loci from PCAWG prior
    frac_cna = float(rng.beta(_PCAWG_FRAC_CNA_ALPHA, _PCAWG_FRAC_CNA_BETA))
    n_cn_altered = max(1, int(frac_cna * N))
    cn_idx = rng.choice(N, size=n_cn_altered, replace=False)

    # Sample ploidy regime
    component = rng.choice(
        len(_PCAWG_PLOIDY_COMPONENTS),
        p=[c[0] for c in _PCAWG_PLOIDY_COMPONENTS],
    )
    _, mean_ploidy, std_ploidy = _PCAWG_PLOIDY_COMPONENTS[component]

    # Per-locus CN states — mixture of gains and losses
    total_cn = rng.normal(mean_ploidy, std_ploidy, size=n_cn_altered)
    total_cn = np.clip(np.round(total_cn), 0, 8).astype(int)
    total_cn = np.maximum(total_cn, 1)  # at least 1 copy

    normal_cn = 2.0
    adj = normal_cn / (purity * total_cn + (1 - purity) * normal_cn)
    adj_factor[cn_idx, :] = adj[:, np.newaxis]

    return adj_factor


def _generate_depth(
    N: int, S: int, mean_depth: int,
    noise: NoiseParams | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate per-locus depth with optional GC-bias correlated variation."""
    depth_arr = rng.poisson(mean_depth, size=(N, S)).astype(np.float64)

    if noise and noise.gc_bias_strength > 0:
        # Simulate GC-content as a smooth wave across loci
        # (in reality GC varies per genomic bin; a sinusoidal model captures
        # the autocorrelation without needing actual coordinates)
        n_waves = rng.integers(2, 8)
        gc_wave = np.sin(np.linspace(0, n_waves * np.pi, N))
        gc_multiplier = 1.0 + noise.gc_bias_strength * gc_wave
        gc_multiplier = np.clip(gc_multiplier, 0.2, 3.0)
        for s in range(S):
            depth_arr[:, s] *= gc_multiplier

    depth_arr = np.maximum(np.round(depth_arr), 1)
    return depth_arr


def _beta_binomial(
    n: np.ndarray, p: np.ndarray, rho: float, rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample from Beta-Binomial(n, p, rho) via a two-stage procedure.

    For each locus, the per-read success probability is drawn from
    Beta(α, β) where α = p(1−ρ)/ρ and β = (1−p)(1−ρ)/ρ, then reads
    are drawn from Binomial(n, q).  This produces overdispersed counts.

    Parameters
    ----------
    n : int array — number of trials (depth)
    p : float array — mean success probability (VAF)
    rho : float — overdispersion in (0, 1)
    rng : Generator
    """
    rho = np.clip(rho, 1e-6, 0.999)
    alpha = p * (1 - rho) / rho
    beta = (1 - p) * (1 - rho) / rho

    # Floor to avoid degenerate Beta parameters
    alpha = np.maximum(alpha, 0.01)
    beta = np.maximum(beta, 0.01)

    q = rng.beta(alpha, beta)
    q = np.clip(q, 0.0, 1.0)
    return rng.binomial(n, q).astype(np.float64)


def _apply_strand_bias(
    alt: np.ndarray, depth: np.ndarray,
    noise: NoiseParams, rng: np.random.Generator,
) -> np.ndarray:
    """
    Apply strand bias to a random subset of loci.

    At affected loci, alt reads are asymmetrically distributed between
    forward and reverse strands, which inflates or deflates the observed
    alt count depending on which strand is underrepresented.
    """
    N, S = alt.shape
    n_biased = int(noise.strand_bias_prob * N)
    if n_biased == 0:
        return alt

    biased_idx = rng.choice(N, size=n_biased, replace=False)
    # Multiplicative bias: scale alt count up or down
    bias = rng.uniform(
        1.0 - noise.strand_bias_magnitude,
        1.0 + noise.strand_bias_magnitude,
        size=(n_biased, S),
    )
    alt = alt.copy()
    alt[biased_idx] = np.clip(
        np.round(alt[biased_idx] * bias), 0, depth[biased_idx],
    )
    return alt


def _add_neutral_tail(
    alt: np.ndarray, depth: np.ndarray, adj_factor: np.ndarray,
    true_assignments: np.ndarray,
    N: int, S: int, K: int, mean_depth: int,
    noise: NoiseParams, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Append neutral 1/f passenger mutations (MOBSTER-style tail).

    These are low-VAF mutations not belonging to any clone, following
    a power-law: f(vaf) ∝ 1/vaf^shape.  They're assigned to a
    virtual "tail clone" (index K, not in true_centers) but labelled
    as clone 0 (clonal) for the assignment vector since they represent
    neutral evolution on the clonal background.
    """
    n_tail = max(1, int(noise.neutral_tail_frac * N))

    # Power-law VAFs in (0.01, 0.25) — inverse CDF sampling
    u = rng.uniform(0, 1, size=n_tail)
    shape = noise.neutral_tail_shape
    vaf_lo, vaf_hi = 0.01, 0.25
    # Inverse CDF of Pareto-like: F^{-1}(u) = vaf_lo * ((vaf_hi/vaf_lo)^u)^(1/shape)
    tail_vaf = vaf_lo * (vaf_hi / vaf_lo) ** (u ** (1.0 / shape))
    tail_vaf = np.clip(tail_vaf, 0.001, 0.499)

    tail_depth = rng.poisson(mean_depth, size=(n_tail, S)).astype(np.float64)
    tail_depth = np.maximum(tail_depth, 1)
    tail_vaf_2d = np.broadcast_to(tail_vaf[:, np.newaxis], (n_tail, S))
    tail_alt = rng.binomial(tail_depth.astype(int), tail_vaf_2d).astype(np.float64)
    tail_adj = np.ones((n_tail, S), dtype=np.float64)

    # Tail mutations are neutral passengers — assign to a dedicated label (K)
    # so they don't inflate clone 0's size in ground-truth assignments.
    # Scoring functions filter out label == -1; we use K here and let
    # the caller's scoring logic handle it via the "valid clone" check.
    tail_assign = np.full(n_tail, K, dtype=true_assignments.dtype)

    return (
        np.vstack([alt, tail_alt]),
        np.vstack([depth, tail_depth]),
        np.vstack([adj_factor, tail_adj]),
        np.concatenate([true_assignments, tail_assign]),
    )
