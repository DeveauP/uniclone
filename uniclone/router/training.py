"""
uniclone.router.training
=========================

Training pipeline for the NeuralTS MetaRouter.

**Batch pipeline** (original):

1. Generate tumours (fast) → save to disk
2. Score configs per tumour (slow) → adaptive elimination to skip hopeless configs
3. Pre-train encoder via cross-entropy on best-config classification
4. Initialize Bayesian heads from all (z, reward) pairs

**Online pipeline** (``train_online``):

1. Score all configs on a small pilot set → pre-train encoder
2. For each remaining tumour, Thompson-sample one config, score it,
   update the Bayesian head, persist the score — one fit per tumour.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from uniclone.router.constants import (
    CONFIG_NAMES,
    DEFAULT_EXCLUDE_CONFIGS,
    N_CONFIGS,
    SUBCHALLENGES,
    SubChallenge,
)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class CorpusEntry:
    """A single training example: features + config score for a subchallenge."""

    features: np.ndarray  # (N_FEATURES,) meta-feature vector
    subchallenge: SubChallenge
    config_name: str
    score: float


def score_result(
    result: Any,
    ground_truth: Any,
    subchallenge: SubChallenge,
) -> float:
    """
    Score a CloneResult against ground truth for a specific subchallenge.

    Parameters
    ----------
    result : CloneResult
    ground_truth : QuantumCatResult
    subchallenge : SubChallenge

    Returns
    -------
    Score in [0, 1], higher is better.
    """
    if subchallenge == SubChallenge.SC1A:
        # Purity estimation: 1 - |estimated - true|
        est_purity = float(np.max(result.centers))
        true_purity = ground_truth.params.purity
        return max(0.0, 1.0 - abs(est_purity - true_purity))

    elif subchallenge == SubChallenge.SC1B:
        # CCF estimation: RMSE-based score (robust to K=1 / constant predictions)
        # Average across all samples for multi-sample tumours.
        true_centers = ground_truth.true_centers
        true_assign = ground_truth.true_assignments
        n_samples = true_centers.shape[1]

        est_assign = result.assignments
        n_true = len(true_assign)
        n_est = len(est_assign)
        n = min(n_true, n_est)
        if n == 0:
            return 0.0

        true_assign_trunc = true_assign[:n]
        est_assign_trunc = est_assign[:n]

        # Exclude tail mutations (label >= n_clones) and noise-filtered (-1)
        n_true_clones = true_centers.shape[0]
        valid_true = true_assign_trunc < n_true_clones
        valid_est = est_assign_trunc >= 0
        valid = valid_true & valid_est
        if not valid.any():
            return 0.0

        # Score per sample, then average
        n_est_samples = min(n_samples, result.centers.shape[1])
        sample_scores = []
        for s in range(n_est_samples):
            true_ccf = true_centers[true_assign_trunc[valid], s]
            est_ccf = result.centers[est_assign_trunc[valid], s]
            rmse = float(np.sqrt(np.mean((true_ccf - est_ccf) ** 2)))
            sample_scores.append(max(0.0, 1.0 - rmse))

        return float(np.mean(sample_scores))

    elif subchallenge == SubChallenge.SC2A:
        # Number of clones: symmetric formula that gives partial credit
        # for small over/under-estimates instead of binary 0/1 at K=1.
        true_k = ground_truth.params.n_clones
        est_k = result.K
        if est_k <= 0:
            return 0.0
        return max(0.0, 1.0 - abs(true_k - est_k) / max(true_k, est_k))

    elif subchallenge == SubChallenge.SC2B:
        # Cluster assignment: V-measure approximation.
        # Noise modules may expand assignments with -1 labels; truncate
        # or filter to the common valid subset.
        true_assign = ground_truth.true_assignments
        est_assign = result.assignments
        n = min(len(true_assign), len(est_assign))
        if n == 0:
            return 0.0
        true_assign = true_assign[:n]
        est_assign = est_assign[:n]

        # Exclude noise-filtered mutations (est == -1) and neutral tail
        # mutations (true label >= n_clones, from _add_neutral_tail)
        n_true_clones = ground_truth.true_centers.shape[0]
        valid = (est_assign >= 0) & (true_assign < n_true_clones)
        if valid.sum() == 0:
            return 0.0
        return _v_measure(true_assign[valid], est_assign[valid])

    elif subchallenge == SubChallenge.SC3:
        # Tree accuracy: fraction of correct parent-child relationships.
        # If the config doesn't produce a tree, score based on K accuracy
        # instead of returning a flat 0 — this avoids penalising clustering-
        # only configs on a subchallenge they weren't designed for.
        if result.tree is None:
            if ground_truth.true_tree is None:
                return 1.0
            # No tree produced but ground truth has one: partial credit
            # based on K accuracy (a necessary condition for a correct tree)
            true_k = ground_truth.params.n_clones
            est_k = result.K
            k_score = 1.0 - abs(true_k - est_k) / max(true_k, est_k, 1)
            return max(0.0, 0.5 * k_score)
        if ground_truth.true_tree is None:
            return 0.0
        true_tree = ground_truth.true_tree
        est_tree = result.tree.adjacency
        if true_tree.shape != est_tree.shape:
            # Shape mismatch (different K): partial credit for K proximity
            true_k = ground_truth.params.n_clones
            est_k = result.K
            k_score = 1.0 - abs(true_k - est_k) / max(true_k, est_k, 1)
            return max(0.0, 0.5 * k_score)
        n_edges = max(int(true_tree.sum()), 1)
        correct = int((true_tree * est_tree).sum())
        return correct / n_edges

    return 0.0


def _v_measure(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """Simplified V-measure (harmonic mean of homogeneity and completeness)."""
    n = len(true_labels)
    if n == 0:
        return 0.0

    # Contingency matrix
    true_classes = np.unique(true_labels)
    pred_classes = np.unique(pred_labels)
    if len(pred_classes) == 1 and len(true_classes) == 1:
        return 1.0

    contingency = np.zeros((len(true_classes), len(pred_classes)))
    true_map = {c: i for i, c in enumerate(true_classes)}
    pred_map = {c: i for i, c in enumerate(pred_classes)}
    for t, p in zip(true_labels, pred_labels):
        contingency[true_map[t], pred_map[p]] += 1

    # Entropy-based homogeneity and completeness
    def entropy(labels: np.ndarray) -> float:
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / counts.sum()
        return float(-np.sum(probs * np.log(probs + 1e-15)))

    h_true = entropy(true_labels)
    h_pred = entropy(pred_labels)

    if h_true == 0:
        return 1.0

    # Conditional entropy H(true|pred)
    h_true_given_pred = 0.0
    for j in range(len(pred_classes)):
        col = contingency[:, j]
        col_sum = col.sum()
        if col_sum > 0:
            probs = col / col_sum
            h_true_given_pred -= col_sum / n * float(np.sum(probs * np.log(probs + 1e-15)))

    homogeneity = 1.0 - h_true_given_pred / h_true if h_true > 0 else 1.0

    # Conditional entropy H(pred|true)
    h_pred_given_true = 0.0
    for i in range(len(true_classes)):
        row = contingency[i, :]
        row_sum = row.sum()
        if row_sum > 0:
            probs = row / row_sum
            h_pred_given_true -= row_sum / n * float(np.sum(probs * np.log(probs + 1e-15)))

    completeness = 1.0 - h_pred_given_true / h_pred if h_pred > 0 else 1.0

    if homogeneity + completeness == 0:
        return 0.0
    return 2 * homogeneity * completeness / (homogeneity + completeness)


def _evaluate_config(args: tuple) -> list[CorpusEntry]:
    """Worker function to evaluate one tumour with one config."""
    from uniclone.core.config import CONFIGS
    from uniclone.core.model import GenerativeModel
    from uniclone.router.meta_features import extract_meta_features, features_to_tensor

    tumour_result = args[0]
    config_name: str = args[1]

    features = extract_meta_features(
        alt=tumour_result.alt,
        depth=tumour_result.depth,
        adj_factor=tumour_result.adj_factor,
    )
    feat_vec = features_to_tensor(features)

    entries = []
    try:
        model = GenerativeModel(CONFIGS[config_name])
        result = model.fit(
            tumour_result.alt,
            tumour_result.depth,
            tumour_result.adj_factor,
        )
        for sc in SUBCHALLENGES:
            s = score_result(result, tumour_result, sc)
            entries.append(CorpusEntry(
                features=feat_vec,
                subchallenge=sc,
                config_name=config_name,
                score=s,
            ))
    except Exception as exc:
        import logging
        import traceback

        logging.getLogger(__name__).warning(
            "Config %s failed (N=%d): %s\n%s",
            config_name, tumour_result.alt.shape[0], exc, traceback.format_exc(),
        )
        for sc in SUBCHALLENGES:
            entries.append(CorpusEntry(
                features=feat_vec,
                subchallenge=sc,
                config_name=config_name,
                score=0.0,
            ))

    return entries


# ---------------------------------------------------------------------------
# Step 1: Tumour generation (fast — seconds)
# ---------------------------------------------------------------------------

def _save_tumour(tumour: Any, path: Path) -> None:
    """Serialize a QuantumCatResult / BAMSurgeonResult to an .npz file."""
    data = {
        "alt": tumour.alt,
        "depth": tumour.depth,
        "adj_factor": tumour.adj_factor,
        "true_assignments": tumour.true_assignments,
        "true_centers": tumour.true_centers,
    }
    if tumour.true_tree is not None:
        data["true_tree"] = tumour.true_tree
    # Store params as separate arrays/scalars
    data["param_n_clones"] = np.array(tumour.params.n_clones)
    data["param_n_mutations"] = np.array(tumour.params.n_mutations)
    data["param_n_samples"] = np.array(tumour.params.n_samples)
    data["param_purity"] = np.array(tumour.params.purity)
    data["param_depth"] = np.array(tumour.params.depth)
    np.savez_compressed(path, **data)


def _load_tumour(path: Path) -> Any:
    """Deserialize a tumour .npz back to a QuantumCatResult."""
    from uniclone.simulate.quantum_cat import QuantumCatParams, QuantumCatResult

    data = np.load(path)
    params = QuantumCatParams(
        n_clones=int(data["param_n_clones"]),
        n_mutations=int(data["param_n_mutations"]),
        n_samples=int(data["param_n_samples"]),
        purity=float(data["param_purity"]),
        depth=int(data["param_depth"]),
    )
    return QuantumCatResult(
        alt=data["alt"],
        depth=data["depth"],
        adj_factor=data["adj_factor"],
        true_assignments=data["true_assignments"],
        true_centers=data["true_centers"],
        true_tree=data["true_tree"] if "true_tree" in data else None,
        params=params,
    )


def generate_tumours(
    out_dir: str | Path,
    n_tumours: int = 10_000,
    n_augmentations: int = 3,
    seed: int = 42,
    simulator: str = "bamsurgeon",
) -> int:
    """
    Generate synthetic tumours and save to disk.

    Each tumour is saved as ``out_dir/tumour_NNNNN.npz``. Fast — typically
    seconds for 10K tumours.

    Parameters
    ----------
    out_dir : path
        Directory to write tumour files into.
    n_tumours : int
        Number of base tumours.
    n_augmentations : int
        Augmented variants per base tumour.
    seed : int
    simulator : str
        ``"bamsurgeon"`` or ``"quantumcat"``.

    Returns
    -------
    Total number of tumour files written.
    """
    import warnings

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    actual_simulator = simulator
    if simulator == "bamsurgeon":
        from uniclone.simulate.bamsurgeon_wrap import is_available as _bs_ok

        if _bs_ok():
            from uniclone.simulate.bamsurgeon_wrap import (
                sample_tumour_params,
            )
            from uniclone.simulate.bamsurgeon_wrap import (
                simulate_bamsurgeon as _simulate,
            )
        else:
            actual_simulator = "quantumcat"
            warnings.warn(
                "BAMSurgeon is not available (missing pysam, addsnv.py, or "
                "env vars UNICLONE_BAMSURGEON_NORMAL_BAM / "
                "UNICLONE_BAMSURGEON_REFERENCE). Falling back to QuantumCat.",
                stacklevel=2,
            )
            from uniclone.simulate.quantum_cat import (
                sample_tumour_params,
            )
            from uniclone.simulate.quantum_cat import (
                simulate_quantumcat as _simulate,
            )
    elif simulator == "quantumcat":
        from uniclone.simulate.quantum_cat import (
            sample_tumour_params,
        )
        from uniclone.simulate.quantum_cat import (
            simulate_quantumcat as _simulate,
        )
    else:
        raise ValueError(
            f"Unknown simulator: {simulator!r}. Use 'bamsurgeon' or 'quantumcat'."
        )

    from uniclone.simulate.quantum_cat import augment_result

    rng = np.random.default_rng(seed)
    idx = 0
    for _ in tqdm(range(n_tumours), desc="Simulating tumours", unit="tumour"):
        params = sample_tumour_params(rng)
        tumour = _simulate(params)
        _save_tumour(tumour, out_dir / f"tumour_{idx:06d}.npz")
        idx += 1
        for _ in range(n_augmentations):
            aug = augment_result(tumour, rng)
            _save_tumour(aug, out_dir / f"tumour_{idx:06d}.npz")
            idx += 1

    # Write manifest
    manifest = {
        "n_base": n_tumours,
        "n_augmentations": n_augmentations,
        "n_total": idx,
        "seed": seed,
        "simulator": actual_simulator,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    return idx


# ---------------------------------------------------------------------------
# Step 2: Config scoring (slow — adaptive elimination)
# ---------------------------------------------------------------------------

# Configs to score first as a pilot round.  Chosen for diversity:
# - quantumclone_v1: default / baseline
# - pyclone_vi: variational, fast
# - wgs_cohort: different emission / inference axis
_PILOT_CONFIGS = ["quantumclone_v1", "pyclone_vi", "wgs_cohort"]


def _score_one_config(tumour: Any, config_name: str) -> dict[str, float]:
    """Score a single config on a single tumour.  Returns {sc_name: score}."""
    import logging
    import traceback

    from uniclone.core.config import CONFIGS
    from uniclone.core.model import GenerativeModel

    logger = logging.getLogger(__name__)
    scores: dict[str, float] = {}
    try:
        model = GenerativeModel(CONFIGS[config_name])
        result = model.fit(tumour.alt, tumour.depth, tumour.adj_factor)
        for sc in SUBCHALLENGES:
            scores[sc.name] = score_result(result, tumour, sc)
    except Exception as exc:
        logger.warning(
            "Config %s failed on tumour (N=%d): %s\n%s",
            config_name, tumour.alt.shape[0], exc, traceback.format_exc(),
        )
        for sc in SUBCHALLENGES:
            scores[sc.name] = 0.0
    return scores


def _score_tumour_worker(args: tuple) -> tuple[int, str, dict[str, float]]:
    """Worker for parallel scoring.  Returns (tumour_idx, config_name, scores)."""
    tumour_path, config_name, tumour_idx = args
    tumour = _load_tumour(Path(tumour_path))
    scores = _score_one_config(tumour, config_name)
    return tumour_idx, config_name, scores


def score_tumours(
    tumour_dir: str | Path,
    out_dir: str | Path,
    *,
    n_workers: int = 4,
    elimination_margin: float = 0.25,
    pilot_configs: list[str] | None = None,
    exclude_configs: frozenset[str] | None = None,
) -> list[CorpusEntry]:
    """
    Score configs on pre-generated tumours with adaptive elimination.

    Strategy:
    1. **Pilot round**: score a small set of representative configs on all
       tumours to establish baselines.
    2. **Elimination**: for each tumour, if the best pilot score exceeds
       ``0.5 + elimination_margin`` on **every** subchallenge individually,
       skip remaining configs (they're unlikely to beat the pilot winner).
    3. **Full round**: score only the surviving configs.
    4. **Resume**: already-scored (tumour, config) pairs in ``out_dir`` are
       skipped, so interrupted runs can be resumed.

    Parameters
    ----------
    tumour_dir : path
        Directory of ``tumour_NNNNNN.npz`` files from :func:`generate_tumours`.
    out_dir : path
        Directory to write per-tumour score JSONs (for resumability).
    n_workers : int
        Parallel workers.
    elimination_margin : float
        A non-pilot config is skipped for a tumour only if the best pilot
        score exceeds ``0.5 + margin`` on *every* subchallenge individually.
        Set to 0 to disable elimination (score everything).
    pilot_configs : list of str or None
        Override default pilot config set.
    exclude_configs : frozenset of str or None
        Config names to skip entirely during scoring.  Defaults to
        ``DEFAULT_EXCLUDE_CONFIGS`` (currently ``{"phyloclone_style"}``).
        Pass ``frozenset()`` to include all configs.

    Returns
    -------
    list of CorpusEntry
    """
    from uniclone.router.meta_features import extract_meta_features, features_to_tensor

    if exclude_configs is None:
        exclude_configs = DEFAULT_EXCLUDE_CONFIGS

    tumour_dir = Path(tumour_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    active_configs = [c for c in CONFIG_NAMES if c not in exclude_configs]
    pilots = [c for c in (pilot_configs or _PILOT_CONFIGS) if c not in exclude_configs]
    remaining_configs = [c for c in active_configs if c not in pilots]

    # Discover tumour files
    tumour_files = sorted(tumour_dir.glob("tumour_*.npz"))
    n_tumours = len(tumour_files)
    if n_tumours == 0:
        raise FileNotFoundError(f"No tumour files found in {tumour_dir}")

    # --- Load existing scores for resumability ---
    scores_db: dict[int, dict[str, dict[str, float]]] = {}
    for score_file in out_dir.glob("scores_*.json"):
        tid = int(score_file.stem.split("_")[1])
        scores_db[tid] = json.loads(score_file.read_text())

    def _save_scores(tid: int) -> None:
        path = out_dir / f"scores_{tid:06d}.json"
        path.write_text(json.dumps(scores_db[tid], indent=2) + "\n")

    # --- Pilot round ---
    pilot_work = []
    for i, tf in enumerate(tumour_files):
        if i not in scores_db:
            scores_db[i] = {}
        for cfg in pilots:
            if cfg not in scores_db[i]:
                pilot_work.append((str(tf), cfg, i))

    if pilot_work:
        desc = f"Pilot round ({len(pilots)} configs)"
        if n_workers <= 1:
            for args in tqdm(pilot_work, desc=desc, unit="fit"):
                tid, cfg, sc = _score_tumour_worker(args)
                scores_db[tid][cfg] = sc
                _save_scores(tid)
        else:
            with Pool(n_workers) as pool:
                for tid, cfg, sc in tqdm(
                    pool.imap_unordered(_score_tumour_worker, pilot_work),
                    total=len(pilot_work),
                    desc=desc,
                    unit="fit",
                ):
                    scores_db[tid][cfg] = sc
                    _save_scores(tid)

    # --- Elimination: decide which remaining configs to score per tumour ---
    full_work = []
    n_eliminated = 0
    n_total_remaining = 0

    for i, tf in enumerate(tumour_files):
        tumour_scores = scores_db.get(i, {})
        # Best pilot score averaged across subchallenges
        best_pilot_avg = 0.0
        if tumour_scores:
            per_sc_best: dict[str, float] = {}
            for cfg, sc_scores in tumour_scores.items():
                for sc_name, val in sc_scores.items():
                    per_sc_best[sc_name] = max(per_sc_best.get(sc_name, 0.0), val)
            if per_sc_best:
                best_pilot_avg = np.mean(list(per_sc_best.values()))

        for cfg in remaining_configs:
            if cfg in tumour_scores:
                continue  # already scored (resume)
            # Heuristic: if *every* pilot subchallenge individually exceeds a
            # high bar, skip remaining configs for this tumour.  The previous
            # logic used an *average* threshold of 0.65 — that's too loose
            # because one weak subchallenge can hide behind strong ones.
            # Require ALL subchallenges above (0.5 + margin) individually and
            # raise the bar to make elimination truly rare.
            if elimination_margin > 0 and tumour_scores:
                per_sc_pilot: dict[str, float] = {}
                for pcfg, sc_scores in tumour_scores.items():
                    if pcfg not in pilots:
                        continue
                    for sc_name, val in sc_scores.items():
                        per_sc_pilot[sc_name] = max(
                            per_sc_pilot.get(sc_name, 0.0), val
                        )
                threshold = 0.5 + elimination_margin
                all_above = (
                    len(per_sc_pilot) == len(SUBCHALLENGES)
                    and all(v > threshold for v in per_sc_pilot.values())
                )
                if all_above:
                    n_eliminated += 1
                    # Mark as skipped (not scored) so the assembler can
                    # distinguish elimination from genuine zero scores.
                    if i not in scores_db:
                        scores_db[i] = {}
                    scores_db[i][cfg] = {
                        sc.name: 0.0 for sc in SUBCHALLENGES
                    }
                    scores_db[i][cfg]["_eliminated"] = 1.0
                    continue
            full_work.append((str(tf), cfg, i))
            n_total_remaining += 1

    total_possible = n_tumours * len(remaining_configs)
    already_scored = total_possible - n_eliminated - n_total_remaining
    tqdm.write(
        f"Elimination: {n_eliminated}/{total_possible} fits skipped, "
        f"{already_scored} already scored, "
        f"{n_total_remaining} remaining"
    )

    # --- Full round ---
    if full_work:
        if n_workers <= 1:
            for args in tqdm(full_work, desc="Scoring remaining configs", unit="fit"):
                tid, cfg, sc = _score_tumour_worker(args)
                scores_db[tid][cfg] = sc
                _save_scores(tid)
        else:
            with Pool(n_workers) as pool:
                for tid, cfg, sc in tqdm(
                    pool.imap_unordered(_score_tumour_worker, full_work),
                    total=len(full_work),
                    desc="Scoring remaining configs",
                    unit="fit",
                ):
                    scores_db[tid][cfg] = sc
                    _save_scores(tid)

    # --- Assemble CorpusEntry list ---
    all_entries: list[CorpusEntry] = []
    for i, tf in tqdm(
        list(enumerate(tumour_files)),
        desc="Assembling corpus",
        unit="tumour",
    ):
        tumour = _load_tumour(tf)
        features = extract_meta_features(
            alt=tumour.alt, depth=tumour.depth, adj_factor=tumour.adj_factor,
        )
        feat_vec = features_to_tensor(features)

        tumour_scores = scores_db.get(i, {})
        for cfg, sc_scores in tumour_scores.items():
            if cfg not in CONFIG_NAMES:
                continue
            # Skip eliminated entries — they were never actually scored,
            # so injecting 0.0 would poison the Bayesian heads.
            if sc_scores.get("_eliminated"):
                continue
            for sc in SUBCHALLENGES:
                all_entries.append(CorpusEntry(
                    features=feat_vec,
                    subchallenge=sc,
                    config_name=cfg,
                    score=sc_scores.get(sc.name, 0.0),
                ))

    return all_entries


# ---------------------------------------------------------------------------
# Convenience wrapper (preserves old API)
# ---------------------------------------------------------------------------

def build_training_corpus(
    n_tumours: int = 10_000,
    n_workers: int = 4,
    seed: int = 42,
    simulator: str = "bamsurgeon",
    n_augmentations: int = 3,
    elimination_margin: float = 0.25,
    work_dir: str | Path | None = None,
) -> list[CorpusEntry]:
    """
    Generate tumours + score configs → labelled corpus.

    Convenience wrapper around :func:`generate_tumours` and
    :func:`score_tumours`.  For large runs, call them separately so you can
    inspect tumours before committing to the expensive scoring step.

    Parameters
    ----------
    n_tumours : int
        Number of base synthetic tumours.
    n_workers : int
        Parallel workers for scoring.
    seed : int
    simulator : str
    n_augmentations : int
    elimination_margin : float
        Margin for adaptive config elimination (0 = score everything).
    work_dir : path or None
        Working directory for intermediate files.  If None, uses a temp dir
        (no resumability).

    Returns
    -------
    list of CorpusEntry
    """
    import tempfile

    if work_dir is not None:
        wd = Path(work_dir)
    else:
        wd = Path(tempfile.mkdtemp(prefix="uniclone_corpus_"))

    tumour_dir = wd / "tumours"
    scores_dir = wd / "scores"

    # Step 1: generate (skip if tumours already exist)
    manifest_path = tumour_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        tqdm.write(
            f"Reusing {manifest['n_total']} existing tumours from {tumour_dir}"
        )
    else:
        generate_tumours(
            out_dir=tumour_dir,
            n_tumours=n_tumours,
            n_augmentations=n_augmentations,
            seed=seed,
            simulator=simulator,
        )

    # Step 2: score
    return score_tumours(
        tumour_dir=tumour_dir,
        out_dir=scores_dir,
        n_workers=n_workers,
        elimination_margin=elimination_margin,
    )


@dataclass
class TrainResult:
    """Result of detailed router training with loss tracking."""

    model: Any  # NeuralTSModel
    train_losses: list[float]  # per-epoch cross-entropy
    val_losses: list[float]  # per-epoch (empty if no val set)
    n_train: int
    n_val: int


def train_router_detailed(
    train_corpus: list[CorpusEntry],
    val_corpus: list[CorpusEntry] | None = None,
    n_epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> TrainResult:
    """
    Train NeuralTS model with full control over hyperparameters and loss tracking.

    Phase A: Pre-train encoder via cross-entropy on best-config classification.
    Phase B: Initialize Bayesian heads from training corpus.

    Parameters
    ----------
    train_corpus : list of CorpusEntry
    val_corpus : optional held-out corpus for validation loss
    n_epochs : number of training epochs
    batch_size : mini-batch size
    lr : learning rate

    Returns
    -------
    TrainResult with model, per-epoch losses, and dataset sizes.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("Training requires PyTorch: pip install uniclone[router]")

    import torch
    import torch.nn as nn
    import torch.optim as optim

    from uniclone.router.neural_ts import NeuralTSModel, SharedEncoder

    def _build_classification_data(
        corpus: list[CorpusEntry],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build (features, best_config_label) arrays from corpus."""
        sc_data: dict[SubChallenge, dict[str, list]] = {
            sc: {"features": [], "config_idx": [], "scores": []}
            for sc in SUBCHALLENGES
        }
        for entry in corpus:
            ci = CONFIG_NAMES.index(entry.config_name)
            sc_data[entry.subchallenge]["features"].append(entry.features)
            sc_data[entry.subchallenge]["config_idx"].append(ci)
            sc_data[entry.subchallenge]["scores"].append(entry.score)

        all_features = []
        all_labels = []
        for sc in SUBCHALLENGES:
            data = sc_data[sc]
            if not data["features"]:
                continue
            feats = np.array(data["features"])
            configs = np.array(data["config_idx"])
            scores = np.array(data["scores"])
            unique_feats: dict[bytes, dict] = {}
            for i in range(len(feats)):
                key = feats[i].tobytes()
                if key not in unique_feats:
                    unique_feats[key] = {"feat": feats[i], "scores": np.zeros(N_CONFIGS)}
                unique_feats[key]["scores"][configs[i]] = scores[i]
            for item in unique_feats.values():
                all_features.append(item["feat"])
                all_labels.append(int(np.argmax(item["scores"])))
        return np.array(all_features), np.array(all_labels)

    X_train_np, y_train_np = _build_classification_data(train_corpus)
    n_train = len(X_train_np)

    if n_train == 0:
        encoder = SharedEncoder()
        model = NeuralTSModel(encoder=encoder)
        return TrainResult(
            model=model, train_losses=[], val_losses=[],
            n_train=0, n_val=len(val_corpus) if val_corpus else 0,
        )

    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.long)

    X_val: torch.Tensor | None = None
    y_val: torch.Tensor | None = None
    n_val = 0
    if val_corpus:
        X_val_np, y_val_np = _build_classification_data(val_corpus)
        n_val = len(X_val_np)
        if n_val > 0:
            X_val = torch.tensor(X_val_np, dtype=torch.float32)
            y_val = torch.tensor(y_val_np, dtype=torch.long)

    # Phase A: Encoder pre-training
    encoder = SharedEncoder()
    classifier_head = nn.Linear(encoder.output_dim, N_CONFIGS)
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(classifier_head.parameters()), lr=lr,
    )
    loss_fn = nn.CrossEntropyLoss()

    train_losses: list[float] = []
    val_losses: list[float] = []
    effective_bs = min(batch_size, len(X_train))

    epoch_bar = tqdm(range(n_epochs), desc="Training encoder", unit="epoch")
    for epoch in epoch_bar:
        encoder.train()
        classifier_head.train()
        perm = torch.randperm(len(X_train))
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, len(X_train), effective_bs):
            idx = perm[i : i + effective_bs]
            z = encoder(X_train[idx])
            logits = classifier_head(z)
            loss = loss_fn(logits, y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        tl = epoch_loss / max(n_batches, 1)
        train_losses.append(tl)

        # Validation loss
        vl_str = ""
        if X_val is not None and y_val is not None:
            encoder.eval()
            classifier_head.eval()
            with torch.no_grad():
                z_val = encoder(X_val)
                logits_val = classifier_head(z_val)
                vloss = loss_fn(logits_val, y_val)
            val_losses.append(vloss.item())
            vl_str = f" val={vloss.item():.4f}"
        epoch_bar.set_postfix_str(f"loss={tl:.4f}{vl_str}")

    # Phase B: Bayesian head initialization (training corpus only)
    # Batch-encode all features in one forward pass, then update heads.
    model = NeuralTSModel(encoder=encoder)
    encoder.eval()
    if train_corpus:
        all_feats = torch.tensor(
            np.array([e.features for e in train_corpus]), dtype=torch.float32,
        )
        with torch.no_grad():
            all_z = encoder(all_feats)  # (N, dim)
        for i, entry in enumerate(
            tqdm(train_corpus, desc="Initializing Bayesian heads", unit="entry")
        ):
            ci = CONFIG_NAMES.index(entry.config_name)
            si = SUBCHALLENGES.index(entry.subchallenge)
            model.heads[(ci, si)].update(all_z[i], entry.score)

    return TrainResult(
        model=model,
        train_losses=train_losses,
        val_losses=val_losses,
        n_train=n_train,
        n_val=n_val,
    )


def train_router(corpus: list[CorpusEntry]) -> Any:
    """
    Train NeuralTS model from a labelled corpus.

    Phase A: Pre-train encoder via cross-entropy on best-config classification.
    Phase B: Initialize Bayesian heads from all (z, reward) pairs.

    Parameters
    ----------
    corpus : list of CorpusEntry

    Returns
    -------
    NeuralTSModel with trained encoder and initialized heads.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("Training requires PyTorch: pip install uniclone[router]")

    from uniclone.router.neural_ts import NeuralTSModel, SharedEncoder

    # Organize corpus by subchallenge
    sc_data: dict[SubChallenge, dict[str, list]] = {
        sc: {"features": [], "config_idx": [], "scores": []}
        for sc in SUBCHALLENGES
    }

    for entry in corpus:
        ci = CONFIG_NAMES.index(entry.config_name)
        sc_data[entry.subchallenge]["features"].append(entry.features)
        sc_data[entry.subchallenge]["config_idx"].append(ci)
        sc_data[entry.subchallenge]["scores"].append(entry.score)

    # ------------------------------------------------------------------ #
    # Phase A: Encoder pre-training via cross-entropy                     #
    # ------------------------------------------------------------------ #
    encoder = SharedEncoder()
    classifier_head = nn.Linear(encoder.output_dim, N_CONFIGS)
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(classifier_head.parameters()),
        lr=1e-3,
    )
    loss_fn = nn.CrossEntropyLoss()

    # Build classification dataset: for each unique (features, sc), label = best config
    train_features = []
    train_labels = []

    for sc in SUBCHALLENGES:
        data = sc_data[sc]
        if not data["features"]:
            continue
        feats = np.array(data["features"])
        configs = np.array(data["config_idx"])
        scores = np.array(data["scores"])

        # Group by unique feature vectors (same tumour)
        unique_feats = {}
        for i in range(len(feats)):
            key = feats[i].tobytes()
            if key not in unique_feats:
                unique_feats[key] = {"feat": feats[i], "scores": np.zeros(N_CONFIGS)}
            unique_feats[key]["scores"][configs[i]] = scores[i]

        for item in unique_feats.values():
            train_features.append(item["feat"])
            train_labels.append(int(np.argmax(item["scores"])))

    if not train_features:
        return NeuralTSModel(encoder=encoder)

    X_train = torch.tensor(np.array(train_features), dtype=torch.float32)
    y_train = torch.tensor(train_labels, dtype=torch.long)

    encoder.train()
    n_epochs = 50
    batch_size = min(256, len(X_train))
    losses = []

    epoch_bar = tqdm(range(n_epochs), desc="Training encoder", unit="epoch")
    for epoch in epoch_bar:
        perm = torch.randperm(len(X_train))
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, len(X_train), batch_size):
            idx = perm[i : i + batch_size]
            xb = X_train[idx]
            yb = y_train[idx]

            z = encoder(xb)
            logits = classifier_head(z)
            loss = loss_fn(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        tl = epoch_loss / max(n_batches, 1)
        losses.append(tl)
        epoch_bar.set_postfix_str(f"loss={tl:.4f}")

    # ------------------------------------------------------------------ #
    # Phase B: Bayesian head initialization                               #
    # ------------------------------------------------------------------ #
    model = NeuralTSModel(encoder=encoder)
    encoder.eval()

    if corpus:
        all_feats = torch.tensor(
            np.array([e.features for e in corpus]), dtype=torch.float32,
        )
        with torch.no_grad():
            all_z = encoder(all_feats)  # (N, dim)
        for i, entry in enumerate(
            tqdm(corpus, desc="Initializing Bayesian heads", unit="entry")
        ):
            ci = CONFIG_NAMES.index(entry.config_name)
            si = SUBCHALLENGES.index(entry.subchallenge)
            model.heads[(ci, si)].update(all_z[i], entry.score)

    return model


# ---------------------------------------------------------------------------
# Online bandit training
# ---------------------------------------------------------------------------


@dataclass
class OnlineTrainResult:
    """Result of online bandit training."""

    model: Any  # NeuralTSModel
    train_losses: list[float]  # encoder pre-training losses
    n_pilot: int  # tumours scored exhaustively
    n_online: int  # tumours scored with bandit selection
    n_total_fits: int  # total GenerativeModel.fit() calls
    selections: list[str]  # config selected per online tumour


def train_online(
    tumour_dir: str | Path,
    scores_dir: str | Path,
    *,
    n_pilot: int = 200,
    n_workers: int = 4,
    n_epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 42,
    exclude_configs: frozenset[str] | None = None,
    checkpoint_dir: str | Path | None = None,
    checkpoint_every: int = 1000,
    log_dir: str | Path | None = None,
) -> OnlineTrainResult:
    """
    Train the MetaRouter online: pilot phase + Thompson Sampling phase.

    **Pilot phase** (``n_pilot`` tumours): score all non-excluded configs
    exhaustively, then pre-train the encoder via cross-entropy and
    initialise heads.

    **Online phase** (remaining tumours): for each tumour, Thompson-sample
    one config, score it, update the corresponding Bayesian head, and
    persist the score to disk.

    All scores are persisted as ``scores_NNNNNN.json`` files in
    ``scores_dir``, in the same format as :func:`score_tumours`, so the
    resulting corpus can be assembled by :func:`assemble_corpus`.

    Parameters
    ----------
    tumour_dir : path
        Directory of ``tumour_NNNNNN.npz`` files from :func:`generate_tumours`.
    scores_dir : path
        Directory to write per-tumour score JSONs.
    n_pilot : int
        Number of tumours to score exhaustively for encoder pre-training.
    n_workers : int
        Parallel workers for the pilot phase.
    n_epochs : int
        Encoder pre-training epochs.
    batch_size : int
        Encoder pre-training batch size.
    lr : float
        Encoder pre-training learning rate.
    seed : int
        RNG seed for pilot/online split.
    exclude_configs : frozenset of str or None
        Config names to skip during training.  Defaults to
        ``DEFAULT_EXCLUDE_CONFIGS`` (currently ``{"phyloclone_style"}``).
    checkpoint_dir : path or None
        Directory to save periodic model checkpoints during the online phase.
        If None, defaults to ``scores_dir / "checkpoints"``.
    checkpoint_every : int
        Save a checkpoint every N online tumours (default 1000).
    log_dir : path or None
        Directory for TensorBoard logs.  If None, no TensorBoard logging.
        Pass ``frozenset()`` to include all configs.

    Returns
    -------
    OnlineTrainResult
    """
    if not TORCH_AVAILABLE:
        raise ImportError("Online training requires PyTorch: pip install uniclone[router]")


    from uniclone.router.meta_features import extract_meta_features, features_to_tensor
    from uniclone.router.neural_ts import NeuralTSModel

    if exclude_configs is None:
        exclude_configs = DEFAULT_EXCLUDE_CONFIGS
    active_configs = [c for c in CONFIG_NAMES if c not in exclude_configs]
    n_active = len(active_configs)

    tumour_dir = Path(tumour_dir)
    scores_dir = Path(scores_dir)
    scores_dir.mkdir(parents=True, exist_ok=True)

    # Checkpointing setup
    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else scores_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard setup (optional)
    tb_writer = None
    if log_dir is not None:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(log_dir=str(log_dir))
        except ImportError:
            import logging
            logging.getLogger(__name__).warning(
                "TensorBoard not available (pip install tensorboard). "
                "Continuing without logging."
            )

    tumour_files = sorted(tumour_dir.glob("tumour_*.npz"))
    n_tumours = len(tumour_files)
    if n_tumours == 0:
        raise FileNotFoundError(f"No tumour files found in {tumour_dir}")

    # Load existing scores for resumability
    scores_db: dict[int, dict[str, dict[str, float]]] = {}
    for score_file in scores_dir.glob("scores_*.json"):
        tid = int(score_file.stem.split("_")[1])
        scores_db[tid] = json.loads(score_file.read_text())

    def _save_scores(tid: int) -> None:
        path = scores_dir / f"scores_{tid:06d}.json"
        path.write_text(json.dumps(scores_db[tid], indent=2) + "\n")

    # Deterministic split: first n_pilot files are pilot, rest are online
    rng = np.random.default_rng(seed)
    indices = np.arange(n_tumours)
    rng.shuffle(indices)
    effective_pilot = min(n_pilot, n_tumours)
    pilot_indices = set(indices[:effective_pilot].tolist())
    online_indices = [int(i) for i in indices[effective_pilot:]]

    # ------------------------------------------------------------------ #
    # Phase 1: Pilot — score active configs exhaustively                  #
    # ------------------------------------------------------------------ #
    pilot_work = []
    for i in pilot_indices:
        if i not in scores_db:
            scores_db[i] = {}
        for cfg in active_configs:
            if cfg not in scores_db[i]:
                pilot_work.append((str(tumour_files[i]), cfg, i))

    n_pilot_fits = len(pilot_work)
    if pilot_work:
        desc = f"Pilot ({effective_pilot} tumours × {n_active} configs)"
        if n_workers <= 1:
            for args in tqdm(pilot_work, desc=desc, unit="fit"):
                tid, cfg, sc = _score_tumour_worker(args)
                scores_db[tid][cfg] = sc
                _save_scores(tid)
        else:
            with Pool(n_workers) as pool:
                for tid, cfg, sc in tqdm(
                    pool.imap_unordered(_score_tumour_worker, pilot_work),
                    total=len(pilot_work),
                    desc=desc,
                    unit="fit",
                ):
                    scores_db[tid][cfg] = sc
                    _save_scores(tid)

    # Assemble pilot corpus, split by tumour into train (80%) / val (20%)
    pilot_list = sorted(pilot_indices)
    val_size = max(1, len(pilot_list) // 5)  # 20% of tumours
    val_tumours = set(pilot_list[:val_size])

    train_entries: list[CorpusEntry] = []
    val_entries: list[CorpusEntry] = []
    all_pilot_entries: list[CorpusEntry] = []
    for i in pilot_indices:
        tumour = _load_tumour(tumour_files[i])
        features = extract_meta_features(
            alt=tumour.alt, depth=tumour.depth, adj_factor=tumour.adj_factor,
        )
        feat_vec = features_to_tensor(features)
        target = val_entries if i in val_tumours else train_entries
        for cfg, sc_scores in scores_db.get(i, {}).items():
            if cfg not in CONFIG_NAMES:
                continue
            for sc in SUBCHALLENGES:
                entry = CorpusEntry(
                    features=feat_vec,
                    subchallenge=sc,
                    config_name=cfg,
                    score=sc_scores.get(sc.name, 0.0),
                )
                target.append(entry)
                all_pilot_entries.append(entry)

    tqdm.write(
        f"Pilot: {effective_pilot} tumours ({len(train_entries)} train / "
        f"{len(val_entries)} val entries), {n_pilot_fits} new fits"
    )

    # ------------------------------------------------------------------ #
    # Phase 2: Pre-train encoder from pilot corpus (80/20 train/val)      #
    # ------------------------------------------------------------------ #
    pilot_result = train_router_detailed(
        train_corpus=train_entries,
        val_corpus=val_entries or None,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
    )
    model: NeuralTSModel = pilot_result.model

    # Bayesian heads were initialized from train entries only inside
    # train_router_detailed.  Update them with val entries too — the heads
    # are closed-form posteriors (no overfitting risk).
    if val_entries:
        import torch as _torch
        _val_feats = _torch.tensor(
            np.array([e.features for e in val_entries]), dtype=_torch.float32,
        )
        model.encoder.eval()
        with _torch.no_grad():
            _val_z = model.encoder(_val_feats)
        for j, entry in enumerate(val_entries):
            ci = CONFIG_NAMES.index(entry.config_name)
            si = SUBCHALLENGES.index(entry.subchallenge)
            model.heads[(ci, si)].update(_val_z[j], entry.score)

    tqdm.write(
        f"Encoder pre-trained: {n_epochs} epochs, "
        f"final loss={pilot_result.train_losses[-1]:.4f}"
        if pilot_result.train_losses else "Encoder pre-trained (no data)"
    )

    # Log encoder pre-training losses to TensorBoard
    if tb_writer is not None:
        for epoch_i, loss_val in enumerate(pilot_result.train_losses):
            tb_writer.add_scalar("encoder/train_loss", loss_val, epoch_i)
        for epoch_i, loss_val in enumerate(pilot_result.val_losses):
            tb_writer.add_scalar("encoder/val_loss", loss_val, epoch_i)

    # ------------------------------------------------------------------ #
    # Phase 3: Online — Thompson-sample one config per tumour             #
    # ------------------------------------------------------------------ #
    selections: list[str] = []
    n_online_fits = 0
    _reward_running = 0.0  # running mean reward for TB logging

    # Determine which online tumours already have a bandit-selected score
    # (for resumability — look for "_online" marker in their score dict)
    online_bar = tqdm(online_indices, desc="Online training", unit="tumour")
    for i in online_bar:
        tumour_scores = scores_db.get(i, {})

        # Check resume: if we already have an "_online" marker, skip
        if tumour_scores.get("_meta", {}).get("online"):
            selected = tumour_scores["_meta"]["selected"]
            selections.append(selected)
            # Still update the head from persisted scores
            tumour = _load_tumour(tumour_files[i])
            features = extract_meta_features(
                alt=tumour.alt, depth=tumour.depth, adj_factor=tumour.adj_factor,
            )
            feat_vec = features_to_tensor(features)
            sc_scores = tumour_scores.get(selected, {})
            for sc in SUBCHALLENGES:
                model.update(feat_vec, selected, sc, sc_scores.get(sc.name, 0.0))
            continue

        # Load tumour and extract features
        tumour = _load_tumour(tumour_files[i])
        features = extract_meta_features(
            alt=tumour.alt, depth=tumour.depth, adj_factor=tumour.adj_factor,
        )
        feat_vec = features_to_tensor(features)

        # Thompson-sample: pick the config with highest sampled reward
        # averaged across subchallenges.  Encode once, reuse for all arms.
        z = model._encode(feat_vec)
        best_cfg = None
        best_sampled = -float("inf")
        for cfg_name in active_configs:
            ci = CONFIG_NAMES.index(cfg_name)
            total = 0.0
            for sc in SUBCHALLENGES:
                si = SUBCHALLENGES.index(sc)
                total += model.heads[(ci, si)].thompson_sample(z)
            if total > best_sampled:
                best_sampled = total
                best_cfg = cfg_name

        # Score the selected config
        sc_scores_dict = _score_one_config(tumour, best_cfg)
        n_online_fits += 1

        # Update Bayesian heads
        for sc in SUBCHALLENGES:
            reward = sc_scores_dict.get(sc.name, 0.0)
            model.update(feat_vec, best_cfg, sc, reward)

        # Persist scores
        if i not in scores_db:
            scores_db[i] = {}
        scores_db[i][best_cfg] = sc_scores_dict
        scores_db[i]["_meta"] = {"online": True, "selected": best_cfg}
        _save_scores(i)

        selections.append(best_cfg)

        # TensorBoard logging
        mean_reward = np.mean([sc_scores_dict.get(sc.name, 0.0) for sc in SUBCHALLENGES])
        _reward_running = 0.99 * _reward_running + 0.01 * mean_reward
        if tb_writer is not None:
            tb_writer.add_scalar("online/reward", mean_reward, n_online_fits)
            tb_writer.add_scalar("online/reward_ema", _reward_running, n_online_fits)

        online_bar.set_postfix_str(f"sel={best_cfg} r={mean_reward:.2f}")

        # Periodic checkpoint
        if checkpoint_every > 0 and n_online_fits % checkpoint_every == 0:
            ckpt_path = ckpt_dir / f"checkpoint_{n_online_fits:06d}.pt"
            model.save(ckpt_path)
            tqdm.write(f"  Checkpoint saved: {ckpt_path}")

    # Final checkpoint
    if n_online_fits > 0:
        ckpt_path = ckpt_dir / "checkpoint_final.pt"
        model.save(ckpt_path)
        tqdm.write(f"  Final checkpoint saved: {ckpt_path}")

    if tb_writer is not None:
        # Log selection distribution
        from collections import Counter
        sel_counts = Counter(selections)
        for cfg_name, count in sel_counts.items():
            tb_writer.add_scalar(f"online/selection/{cfg_name}", count, n_online_fits)
        tb_writer.close()

    n_total_fits = n_pilot_fits + n_online_fits
    tqdm.write(
        f"Online: {len(online_indices)} tumours, {n_online_fits} new fits, "
        f"{n_total_fits} total fits "
        f"(vs {n_tumours * n_active} for exhaustive)"
    )

    return OnlineTrainResult(
        model=model,
        train_losses=pilot_result.train_losses,
        n_pilot=effective_pilot,
        n_online=len(online_indices),
        n_total_fits=n_total_fits,
        selections=selections,
    )


def assemble_corpus(
    tumour_dir: str | Path,
    scores_dir: str | Path,
) -> list[CorpusEntry]:
    """
    Assemble a CorpusEntry list from persisted tumour + score files.

    Works with scores produced by either :func:`score_tumours` (batch)
    or :func:`train_online` (online).

    Parameters
    ----------
    tumour_dir : path
        Directory of ``tumour_NNNNNN.npz`` files.
    scores_dir : path
        Directory of ``scores_NNNNNN.json`` files.

    Returns
    -------
    list of CorpusEntry
    """
    from uniclone.router.meta_features import extract_meta_features, features_to_tensor

    tumour_dir = Path(tumour_dir)
    scores_dir = Path(scores_dir)

    tumour_files = sorted(tumour_dir.glob("tumour_*.npz"))

    # Load all scores
    scores_db: dict[int, dict[str, Any]] = {}
    for score_file in scores_dir.glob("scores_*.json"):
        tid = int(score_file.stem.split("_")[1])
        scores_db[tid] = json.loads(score_file.read_text())

    all_entries: list[CorpusEntry] = []
    for i, tf in tqdm(
        list(enumerate(tumour_files)),
        desc="Assembling corpus",
        unit="tumour",
    ):
        tumour = _load_tumour(tf)
        features = extract_meta_features(
            alt=tumour.alt, depth=tumour.depth, adj_factor=tumour.adj_factor,
        )
        feat_vec = features_to_tensor(features)

        tumour_scores = scores_db.get(i, {})
        for cfg, sc_scores in tumour_scores.items():
            if cfg not in CONFIG_NAMES:
                continue  # skip _meta, _eliminated, etc.
            if not isinstance(sc_scores, dict):
                continue
            if sc_scores.get("_eliminated"):
                continue
            for sc in SUBCHALLENGES:
                all_entries.append(CorpusEntry(
                    features=feat_vec,
                    subchallenge=sc,
                    config_name=cfg,
                    score=sc_scores.get(sc.name, 0.0),
                ))

    return all_entries
