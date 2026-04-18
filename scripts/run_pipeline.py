"""
scripts.run_pipeline
=====================

End-to-end pipeline: generate corpus → train router → evaluate.

Produces two dataset tiers:
- **debug**: small, fast (~30s). For development iteration.
- **full**: production-scale (~10K base tumours + augmentations). For final training.

Usage::

    # Quick debug run (default):
    python -m scripts.run_pipeline

    # Full training run:
    python -m scripts.run_pipeline --preset full

    # Custom:
    python -m scripts.run_pipeline --n-tumours 500 --n-augmentations 5 \
        --epochs 30 --tag experiment_1
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PipelineConfig:
    """All knobs for a pipeline run."""

    # Corpus
    n_tumours: int
    n_augmentations: int
    n_workers: int
    seed: int
    simulator: str

    # Training
    epochs: int
    batch_size: int
    lr: float
    val_frac: float

    # Evaluation
    baseline: str
    n_ig_samples: int

    # Output
    tag: str


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

PRESETS: dict[str, dict] = {
    "debug": dict(
        n_tumours=50,
        n_augmentations=2,
        n_workers=1,
        seed=42,
        simulator="quantumcat",
        elimination_margin=0.15,
        epochs=10,
        batch_size=64,
        lr=1e-3,
        val_frac=0.2,
        baseline="quantumclone_v1",
        n_ig_samples=50,
        tag="debug",
    ),
    "full": dict(
        n_tumours=10_000,
        n_augmentations=3,
        n_workers=8,
        seed=42,
        simulator="quantumcat",
        elimination_margin=0.15,
        epochs=50,
        batch_size=256,
        lr=1e-3,
        val_frac=0.15,
        baseline="quantumclone_v1",
        n_ig_samples=200,
        tag="full",
    ),
}


def _parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(
        description="End-to-end MetaRouter pipeline: generate → train → evaluate.",
    )
    parser.add_argument(
        "--preset", choices=list(PRESETS), default="debug",
        help="Preset configuration (default: debug).",
    )

    # Corpus overrides
    parser.add_argument("--n-tumours", type=int, default=None)
    parser.add_argument("--n-augmentations", type=int, default=None)
    parser.add_argument("--n-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--simulator", type=str, default=None,
                        choices=["bamsurgeon", "quantumcat"])
    parser.add_argument("--elimination-margin", type=float, default=None)

    # Training overrides
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--val-frac", type=float, default=None)

    # Evaluation overrides
    parser.add_argument("--baseline", type=str, default=None)
    parser.add_argument("--n-ig-samples", type=int, default=None)

    # Output
    parser.add_argument("--tag", type=str, default=None)

    # Skip steps
    parser.add_argument("--skip-generate", action="store_true",
                        help="Skip corpus generation (reuse existing).")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training (reuse existing model).")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip evaluation.")

    args = parser.parse_args()

    # Start from preset, override with explicit args
    cfg = dict(PRESETS[args.preset])
    for key in [
        "n_tumours", "n_augmentations", "n_workers", "seed", "simulator",
        "elimination_margin",
        "epochs", "batch_size", "lr", "val_frac", "baseline", "n_ig_samples",
        "tag",
    ]:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val

    # Store skip flags
    cfg["_skip_generate"] = args.skip_generate
    cfg["_skip_train"] = args.skip_train
    cfg["_skip_eval"] = args.skip_eval

    return cfg


def main() -> None:
    cfg = _parse_args()
    skip_generate = cfg.pop("_skip_generate")
    skip_train = cfg.pop("_skip_train")
    skip_eval = cfg.pop("_skip_eval")

    tag = cfg["tag"]
    total_tumours = cfg["n_tumours"] * (1 + cfg["n_augmentations"])

    # Paths
    corpus_path = Path(f"data/corpus/corpus_{tag}.npz")
    model_path = Path(f"data/models/router_{tag}.pt")
    model_stem = model_path.with_suffix("")
    val_corpus_path = Path(f"data/models/router_{tag}.val_corpus.npz")
    losses_path = Path(f"data/models/router_{tag}.losses.json")
    report_stem = Path(f"data/reports/eval_{tag}")

    print("=" * 60)
    print(f"MetaRouter Pipeline — preset={tag}")
    print("=" * 60)
    print(f"  Tumours:       {cfg['n_tumours']} base × {1 + cfg['n_augmentations']} "
          f"= {total_tumours}")
    print(f"  Simulator:     {cfg['simulator']}")
    print(f"  Elimination:   margin={cfg['elimination_margin']}")
    print(f"  Workers:       {cfg['n_workers']}")
    print(f"  Epochs:        {cfg['epochs']}")
    print(f"  Val fraction:  {cfg['val_frac']}")
    print(f"  Seed:          {cfg['seed']}")
    print()

    t_start = time.perf_counter()

    # ==================================================================
    # Step 1: Generate corpus (simulate → score, decoupled + resumable)
    # ==================================================================
    work_dir = Path(f"data/corpus/{tag}_work")

    if skip_generate:
        print(f"[1/3] SKIP corpus generation (reusing {corpus_path})")
        if not corpus_path.exists():
            raise FileNotFoundError(
                f"--skip-generate but corpus not found: {corpus_path}"
            )
    else:
        from scripts._corpus_io import save_corpus
        from uniclone.router.training import generate_tumours, score_tumours

        tumour_dir = work_dir / "tumours"
        scores_dir = work_dir / "scores"

        # Step 1a: simulate tumours (fast)
        manifest_path = tumour_dir / "manifest.json"
        if manifest_path.exists():
            print(f"[1/3a] Reusing existing tumours in {tumour_dir}")
        else:
            print(f"[1/3a] Simulating tumours → {tumour_dir}")
            t0 = time.perf_counter()
            n = generate_tumours(
                out_dir=tumour_dir,
                n_tumours=cfg["n_tumours"],
                n_augmentations=cfg["n_augmentations"],
                seed=cfg["seed"],
                simulator=cfg["simulator"],
            )
            elapsed = time.perf_counter() - t0
            print(f"        {n} tumours in {elapsed:.1f}s")

        # Step 1b: score configs (slow, resumable, adaptive elimination)
        print(f"[1/3b] Scoring configs → {scores_dir}")
        t0 = time.perf_counter()
        corpus = score_tumours(
            tumour_dir=tumour_dir,
            out_dir=scores_dir,
            n_workers=cfg["n_workers"],
            elimination_margin=cfg["elimination_margin"],
        )
        elapsed = time.perf_counter() - t0
        print(f"        {len(corpus)} entries in {elapsed:.1f}s")

        save_corpus(
            corpus, corpus_path,
            extras={
                "n_tumours": cfg["n_tumours"],
                "n_augmentations": cfg["n_augmentations"],
                "total_tumours": total_tumours,
                "seed": cfg["seed"],
                "simulator": cfg["simulator"],
                "elimination_margin": cfg["elimination_margin"],
                "pipeline_tag": tag,
            },
        )
        print(f"        Saved to {corpus_path}")
        del corpus
    print()

    # ==================================================================
    # Step 2: Train router
    # ==================================================================
    if skip_train:
        print(f"[2/3] SKIP training (reusing {model_path})")
        if not model_path.exists():
            raise FileNotFoundError(
                f"--skip-train but model not found: {model_path}"
            )
    else:
        print(f"[2/3] Training router → {model_path}")

        import numpy as np

        from scripts._corpus_io import load_corpus
        from scripts._corpus_io import save_corpus as _save
        from uniclone.router.training import train_router_detailed

        corpus = load_corpus(corpus_path)
        print(f"       Loaded {len(corpus)} entries")

        # Split by tumour
        tumour_map: dict[bytes, list] = {}
        for e in corpus:
            tumour_map.setdefault(e.features.tobytes(), []).append(e)

        keys = sorted(tumour_map.keys())
        rng = np.random.default_rng(cfg["seed"])
        rng.shuffle(keys)
        n_val = max(1, int(len(keys) * cfg["val_frac"]))
        val_keys = set(keys[:n_val])

        train_corpus = [e for k in keys if k not in val_keys for e in tumour_map[k]]
        val_corpus = [e for k in keys if k in val_keys for e in tumour_map[k]]
        print(f"       Split: {len(keys) - n_val} train / {n_val} val tumours")

        t0 = time.perf_counter()
        result = train_router_detailed(
            train_corpus=train_corpus,
            val_corpus=val_corpus if val_corpus else None,
            n_epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            lr=cfg["lr"],
        )
        elapsed = time.perf_counter() - t0

        if result.train_losses:
            print(f"       Final train loss: {result.train_losses[-1]:.4f} "
                  f"({elapsed:.1f}s)")
        if result.val_losses:
            best_val = min(result.val_losses)
            print(f"       Best val loss:    {best_val:.4f} "
                  f"(epoch {result.val_losses.index(best_val) + 1})")

        # Save model
        model_path.parent.mkdir(parents=True, exist_ok=True)
        result.model.save(model_path)
        print(f"       Model → {model_path}")

        # Save losses
        losses_path.write_text(json.dumps({
            "train": result.train_losses,
            "val": result.val_losses,
        }, indent=2) + "\n")

        # Save val corpus
        if val_corpus:
            _save(val_corpus, val_corpus_path)

        # Save training metadata
        meta = {
            "tag": tag,
            "corpus": str(corpus_path),
            "n_train": len(train_corpus),
            "n_val": len(val_corpus),
            "epochs": cfg["epochs"],
            "batch_size": cfg["batch_size"],
            "lr": cfg["lr"],
            "training_time_s": round(elapsed, 2),
            "final_train_loss": result.train_losses[-1] if result.train_losses else None,
            "final_val_loss": result.val_losses[-1] if result.val_losses else None,
        }
        meta_path = Path(f"data/models/router_{tag}.meta.json")
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")

        del corpus, train_corpus, val_corpus
    print()

    # ==================================================================
    # Step 3: Evaluate
    # ==================================================================
    if skip_eval:
        print("[3/3] SKIP evaluation")
    else:
        print(f"[3/3] Evaluating → {report_stem}.*")

        import copy

        import torch

        from scripts._corpus_io import load_corpus
        from uniclone.router.evaluate import cumulative_regret, oracle_regret, routing_gain
        from uniclone.router.neural_ts import NeuralTSModel

        model = NeuralTSModel.from_pretrained(model_path)
        test_corpus = load_corpus(val_corpus_path)
        print(f"       {len(test_corpus)} test entries")

        report: dict = {"baseline": cfg["baseline"], "pipeline_tag": tag}

        # Loss curves
        if losses_path.exists():
            losses = json.loads(losses_path.read_text())
            tl = losses.get("train", [])
            vl = losses.get("val", [])
            report["loss_curves"] = {
                "final_train_loss": tl[-1] if tl else None,
                "final_val_loss": vl[-1] if vl else None,
                "best_val_loss": min(vl) if vl else None,
                "best_val_epoch": (vl.index(min(vl)) + 1) if vl else None,
            }

        # Oracle regret
        oreg = oracle_regret(model, test_corpus)
        report["oracle_regret"] = {sc.name: v for sc, v in oreg.items()}
        mean_regret = np.mean(list(report["oracle_regret"].values()))
        print(f"       Mean oracle regret: {mean_regret:.4f}")

        # Routing gain
        rgain = routing_gain(
            model, test_corpus, baseline_config=cfg["baseline"],
        )
        report["routing_gain"] = {sc.name: v for sc, v in rgain.items()}

        # Cumulative regret
        model_copy = copy.deepcopy(model)
        for head in model_copy.heads.values():
            p = head.precision.double()
            p = (p + p.T) / 2
            p = p + 1e-4 * torch.eye(p.shape[0], dtype=torch.float64)
            head.precision = p.float()
            head._invalidate_cache()
        try:
            report["cumulative_regret"] = cumulative_regret(
                model_copy, test_corpus,
            )
        except ValueError as exc:
            print(f"       Warning: cumulative regret failed ({exc})")
            report["cumulative_regret"] = []

        # Write reports
        report_stem.parent.mkdir(parents=True, exist_ok=True)
        json_path = report_stem.with_suffix(".json")
        json_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"       Report → {json_path}")
    print()

    # ==================================================================
    # Summary
    # ==================================================================
    total_time = time.perf_counter() - t_start
    print("=" * 60)
    print(f"Pipeline complete in {total_time:.1f}s")
    print()
    print("Outputs:")
    print(f"  Corpus:  {corpus_path}")
    if not skip_train:
        print(f"  Model:   {model_path}")
    if not skip_eval:
        print(f"  Report:  {report_stem.with_suffix('.json')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
