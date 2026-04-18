"""
scripts.evaluate_router
========================

Evaluate a trained NeuralTS MetaRouter and produce human + machine-readable reports.

Usage::

    python -m scripts.evaluate_router \
        --model data/models/router_v1.pt \
        --test-corpus data/models/router_v1.val_corpus.npz \
        --losses data/models/router_v1.losses.json \
        --baseline quantumclone_v1 \
        --n-ig-samples 200 \
        --out data/reports/eval_router_v1
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from scripts._corpus_io import load_corpus
from uniclone.router.constants import CONFIG_NAMES, FEATURE_NAMES, SUBCHALLENGES, SubChallenge
from uniclone.router.evaluate import cumulative_regret, oracle_regret, routing_gain
from uniclone.router.neural_ts import NeuralTSModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _algo_selection_breakdown(
    model: NeuralTSModel,
    corpus: list,
) -> dict[str, dict[str, int]]:
    """Count how often each config is selected per subchallenge."""

    # Group by unique (features, sc)
    seen: dict[tuple[bytes, str], None] = {}
    counts: dict[str, dict[str, int]] = {
        sc.name: {c: 0 for c in CONFIG_NAMES} for sc in SUBCHALLENGES
    }
    for entry in corpus:
        key = (entry.features.tobytes(), entry.subchallenge.name)
        if key in seen:
            continue
        seen[key] = None
        selected = model.select(entry.features, entry.subchallenge, explore=False)
        counts[entry.subchallenge.name][selected] += 1
    return counts


def _score_distributions(
    model: NeuralTSModel,
    corpus: list,
) -> dict[str, dict[str, dict[str, float]]]:
    """Compute score distributions for router and oracle per subchallenge."""

    groups: dict[tuple[bytes, SubChallenge], dict[str, float]] = {}
    for entry in corpus:
        key = (entry.features.tobytes(), entry.subchallenge)
        if key not in groups:
            groups[key] = {}
        groups[key][entry.config_name] = entry.score

    sc_router: dict[str, list[float]] = {sc.name: [] for sc in SUBCHALLENGES}
    sc_oracle: dict[str, list[float]] = {sc.name: [] for sc in SUBCHALLENGES}

    for (feat_bytes, sc), config_scores in groups.items():
        features = np.frombuffer(feat_bytes, dtype=np.float64).copy()
        selected = model.select(features, sc, explore=False)
        sc_router[sc.name].append(config_scores.get(selected, 0.0))
        sc_oracle[sc.name].append(max(config_scores.values()))

    def _stats(vals: list[float]) -> dict[str, float]:
        if not vals:
            return {"mean": 0, "std": 0, "median": 0, "min": 0, "max": 0}
        a = np.array(vals)
        return {
            "mean": float(np.mean(a)),
            "std": float(np.std(a)),
            "median": float(np.median(a)),
            "min": float(np.min(a)),
            "max": float(np.max(a)),
        }

    result: dict[str, dict[str, dict[str, float]]] = {}
    for sc in SUBCHALLENGES:
        result[sc.name] = {
            "router": _stats(sc_router[sc.name]),
            "oracle": _stats(sc_oracle[sc.name]),
        }
    return result


def _feature_importance_ig(
    model: NeuralTSModel,
    corpus: list,
    n_samples: int = 200,
    n_steps: int = 20,
) -> dict[str, list[tuple[str, float]]]:
    """
    Top-10 features per subchallenge by mean |integrated gradient|.

    Uses the encoder to compute gradients of the selected config's head
    mean prediction w.r.t. input features.
    """
    # Gather unique (features, sc) pairs
    seen: dict[tuple[bytes, str], tuple[np.ndarray, SubChallenge]] = {}
    for entry in corpus:
        key = (entry.features.tobytes(), entry.subchallenge.name)
        if key not in seen:
            seen[key] = (entry.features, entry.subchallenge)

    items = list(seen.values())
    rng = np.random.default_rng(0)
    if len(items) > n_samples:
        indices = rng.choice(len(items), size=n_samples, replace=False)
        items = [items[i] for i in indices]

    # Accumulate |IG| per (sc, feature)
    sc_ig: dict[str, np.ndarray] = {sc.name: np.zeros(len(FEATURE_NAMES)) for sc in SUBCHALLENGES}
    sc_counts: dict[str, int] = {sc.name: 0 for sc in SUBCHALLENGES}

    encoder = model.encoder
    encoder.eval()

    for features, sc in items:
        si = SUBCHALLENGES.index(sc)
        # Select best config (greedy)
        selected = model.select(features, sc, explore=False)
        ci = CONFIG_NAMES.index(selected)
        head = model.heads[(ci, si)]

        baseline = torch.zeros(len(FEATURE_NAMES), dtype=torch.float32)
        x_input = torch.tensor(features, dtype=torch.float32)

        ig = torch.zeros(len(FEATURE_NAMES))
        for step in range(1, n_steps + 1):
            alpha = step / n_steps
            interpolated = baseline + alpha * (x_input - baseline)
            interpolated.requires_grad_(True)
            z = encoder(interpolated.unsqueeze(0)).squeeze(0)
            pred = head.mu @ z
            pred.backward()
            if interpolated.grad is not None:
                ig += interpolated.grad.detach()

        ig = (x_input - baseline) * ig / n_steps
        sc_ig[sc.name] += np.abs(ig.numpy())
        sc_counts[sc.name] += 1

    result: dict[str, list[tuple[str, float]]] = {}
    for sc in SUBCHALLENGES:
        if sc_counts[sc.name] > 0:
            mean_ig = sc_ig[sc.name] / sc_counts[sc.name]
        else:
            mean_ig = np.zeros(len(FEATURE_NAMES))
        ranked = sorted(zip(FEATURE_NAMES, mean_ig.tolist()), key=lambda x: x[1], reverse=True)
        result[sc.name] = ranked[:10]
    return result


def _uncertainty_calibration(
    model: NeuralTSModel,
    corpus: list,
) -> dict[str, list[dict[str, float]]]:
    """Mean actual score per uncertainty quartile per subchallenge."""

    groups: dict[tuple[bytes, SubChallenge], dict[str, float]] = {}
    for entry in corpus:
        key = (entry.features.tobytes(), entry.subchallenge)
        if key not in groups:
            groups[key] = {}
        groups[key][entry.config_name] = entry.score

    sc_data: dict[str, list[tuple[float, float]]] = {sc.name: [] for sc in SUBCHALLENGES}

    for (feat_bytes, sc), config_scores in groups.items():
        features = np.frombuffer(feat_bytes, dtype=np.float64).copy()
        selected = model.select(features, sc, explore=False)
        router_score = config_scores.get(selected, 0.0)
        ci = CONFIG_NAMES.index(selected)
        si = SUBCHALLENGES.index(sc)
        z = model._encode(features)
        unc = model.heads[(ci, si)].uncertainty(z)
        sc_data[sc.name].append((unc, router_score))

    result: dict[str, list[dict[str, float]]] = {}
    for sc in SUBCHALLENGES:
        pairs = sc_data[sc.name]
        if len(pairs) < 4:
            result[sc.name] = []
            continue
        pairs.sort(key=lambda x: x[0])
        n = len(pairs)
        quartiles = []
        for q in range(4):
            start = q * n // 4
            end = (q + 1) * n // 4
            chunk = pairs[start:end]
            if chunk:
                quartiles.append(
                    {
                        "quartile": q + 1,
                        "mean_uncertainty": float(np.mean([p[0] for p in chunk])),
                        "mean_score": float(np.mean([p[1] for p in chunk])),
                        "n": len(chunk),
                    }
                )
        result[sc.name] = quartiles
    return result


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _format_report(data: dict) -> str:
    """Format evaluation data as human-readable text report."""
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("MetaRouter Evaluation Report")
    lines.append("=" * 70)
    lines.append("")

    # 1. Loss curves
    if "loss_curves" in data:
        lc = data["loss_curves"]
        lines.append("--- Loss Curves ---")
        lines.append(f"  Final train loss:  {lc['final_train_loss']:.4f}")
        if lc["final_val_loss"] is not None:
            lines.append(f"  Final val loss:    {lc['final_val_loss']:.4f}")
            lines.append(
                f"  Best val loss:     {lc['best_val_loss']:.4f} (epoch {lc['best_val_epoch']})"
            )
        lines.append("")

    # 2. Oracle regret
    if "oracle_regret" in data:
        lines.append("--- Oracle Regret (lower is better) ---")
        for sc, val in data["oracle_regret"].items():
            lines.append(f"  {sc:6s}  {val:.4f}")
        lines.append("")

    # 3. Routing gain
    if "routing_gain" in data:
        lines.append(f"--- Routing Gain vs {data.get('baseline', '?')} (higher is better) ---")
        for sc, val in data["routing_gain"].items():
            lines.append(f"  {sc:6s}  {val:+.4f}")
        lines.append("")

    # 4. Algorithm selection
    if "selection_breakdown" in data:
        lines.append("--- Algorithm Selection Frequency ---")
        for sc, counts in data["selection_breakdown"].items():
            nonzero = {k: v for k, v in counts.items() if v > 0}
            if nonzero:
                lines.append(f"  {sc}:")
                for cfg, cnt in sorted(nonzero.items(), key=lambda x: -x[1]):
                    lines.append(f"    {cfg:30s} {cnt:4d}")
        lines.append("")

    # 5. Score distributions
    if "score_distributions" in data:
        lines.append("--- Score Distributions ---")
        for sc, dists in data["score_distributions"].items():
            r = dists["router"]
            o = dists["oracle"]
            lines.append(f"  {sc}:")
            lines.append(
                f"    Router: mean={r['mean']:.3f} std={r['std']:.3f} "
                f"med={r['median']:.3f} min={r['min']:.3f} max={r['max']:.3f}"
            )
            lines.append(
                f"    Oracle: mean={o['mean']:.3f} std={o['std']:.3f} "
                f"med={o['median']:.3f} min={o['min']:.3f} max={o['max']:.3f}"
            )
        lines.append("")

    # 6. Cumulative regret
    if "cumulative_regret" in data:
        cr = data["cumulative_regret"]
        lines.append("--- Cumulative Regret (online) ---")
        if cr:
            lines.append(f"  Final cumulative regret: {cr[-1]:.4f} over {len(cr)} tumours")
        else:
            lines.append("  (not computed)")
        lines.append("")

    # 7. Feature importance
    if "feature_importance" in data:
        lines.append("--- Feature Importance (top-10 by mean |IG|) ---")
        for sc, feats in data["feature_importance"].items():
            lines.append(f"  {sc}:")
            for name, val in feats:
                lines.append(f"    {name:25s} {val:.4f}")
        lines.append("")

    # 8. Uncertainty calibration
    if "uncertainty_calibration" in data:
        lines.append("--- Uncertainty Calibration ---")
        for sc, quartiles in data["uncertainty_calibration"].items():
            if quartiles:
                lines.append(f"  {sc}:")
                for q in quartiles:
                    lines.append(
                        f"    Q{q['quartile']}: unc={q['mean_uncertainty']:.4f}  "
                        f"score={q['mean_score']:.3f}  n={q['n']}"
                    )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained NeuralTS MetaRouter.",
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--test-corpus", type=str, required=True)
    parser.add_argument("--losses", type=str, default=None)
    parser.add_argument("--baseline", type=str, default="quantumclone_v1")
    parser.add_argument("--n-ig-samples", type=int, default=200)
    parser.add_argument("--out", type=str, default="data/reports/eval_router_v1")
    args = parser.parse_args()

    out_stem = Path(args.out)
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from {args.model}")
    model = NeuralTSModel.from_pretrained(args.model)

    # Load test corpus
    print(f"Loading test corpus from {args.test_corpus}")
    test_corpus = load_corpus(args.test_corpus)
    print(f"  {len(test_corpus)} entries")

    report: dict = {"baseline": args.baseline}

    # 1. Loss curves
    if args.losses:
        print("Computing loss curve summary...")
        losses = json.loads(Path(args.losses).read_text())
        train_l = losses.get("train", [])
        val_l = losses.get("val", [])
        lc: dict = {
            "final_train_loss": train_l[-1] if train_l else None,
            "final_val_loss": val_l[-1] if val_l else None,
            "best_val_loss": min(val_l) if val_l else None,
            "best_val_epoch": (val_l.index(min(val_l)) + 1) if val_l else None,
            "train": train_l,
            "val": val_l,
        }
        report["loss_curves"] = lc

    # 2. Oracle regret
    print("Computing oracle regret...")
    oreg = oracle_regret(model, test_corpus)
    report["oracle_regret"] = {sc.name: v for sc, v in oreg.items()}

    # 3. Routing gain
    print(f"Computing routing gain vs {args.baseline}...")
    rgain = routing_gain(model, test_corpus, baseline_config=args.baseline)
    report["routing_gain"] = {sc.name: v for sc, v in rgain.items()}

    # 4. Algorithm selection breakdown
    print("Computing algorithm selection breakdown...")
    report["selection_breakdown"] = _algo_selection_breakdown(model, test_corpus)

    # 5. Score distributions
    print("Computing score distributions...")
    report["score_distributions"] = _score_distributions(model, test_corpus)

    # 6. Cumulative regret
    print("Computing cumulative regret (online)...")
    model_copy = copy.deepcopy(model)
    # Regularize precision matrices to avoid numerical issues with Thompson sampling.
    # After many rank-1 updates, precision inversion can yield non-PD covariance
    # due to float32 accumulation. We symmetrize and add a jitter to precision.
    for head in model_copy.heads.values():
        p = head.precision.double()
        p = (p + p.T) / 2
        p = p + 1e-4 * torch.eye(p.shape[0], dtype=torch.float64)
        head.precision = p.float()
        head._invalidate_cache()
    try:
        report["cumulative_regret"] = cumulative_regret(model_copy, test_corpus)
    except ValueError as exc:
        print(f"  Warning: cumulative regret failed ({exc}), skipping")
        report["cumulative_regret"] = []

    # 7. Feature importance
    print(f"Computing feature importance ({args.n_ig_samples} samples)...")
    report["feature_importance"] = _feature_importance_ig(
        model,
        test_corpus,
        n_samples=args.n_ig_samples,
    )

    # 8. Uncertainty calibration
    print("Computing uncertainty calibration...")
    report["uncertainty_calibration"] = _uncertainty_calibration(model, test_corpus)

    # Write outputs
    json_path = out_stem.with_suffix(".json")
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved JSON report to {json_path}")

    txt_path = out_stem.with_suffix(".txt")
    txt_path.write_text(_format_report(report))
    print(f"Saved text report to {txt_path}")


if __name__ == "__main__":
    main()
