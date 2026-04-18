"""
scripts.train_online
=====================

Train a NeuralTS MetaRouter using online bandit learning.

Instead of scoring all configs on every tumour (~110K fits for 10K tumours),
this pipeline:

1. Scores active configs on a small pilot set (200 tumours × 11 = 2,200 fits)
   MCMC-based configs (phyloclone_style) are excluded by default via
   ``--exclude-configs`` / ``DEFAULT_EXCLUDE_CONFIGS``.
2. Pre-trains the encoder from pilot data
3. For each remaining tumour, Thompson-samples one config, scores it,
   and updates the Bayesian head online (9,800 fits)

Total: ~12,000 fits instead of 110,000 — a ~9× speedup.

All scores are persisted to disk for resumability and corpus assembly.

Usage::

    # Generate tumours first (if not already done):
    python -m scripts.generate_corpus generate --n-tumours 10000

    # Online training (default: excludes phyloclone_style):
    python -m scripts.train_online \
        --tumour-dir data/corpus/tumours \
        --n-pilot 200 --n-workers 8 \
        --out data/models/router_online.pt

    # Include all 12 configs (slow — MCMC fits take minutes to hours):
    python -m scripts.train_online \
        --tumour-dir data/corpus/tumours \
        --exclude-configs \
        --out data/models/router_online.pt

    # Exclude specific configs:
    python -m scripts.train_online \
        --exclude-configs phyloclone_style calder_style \
        --out data/models/router_online.pt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train NeuralTS MetaRouter via online bandit learning.",
    )
    parser.add_argument(
        "--tumour-dir", type=str, default=None,
        help="Directory of tumour_NNNNNN.npz files.  If omitted, tumours are "
             "generated to --work-dir/tumours.",
    )
    parser.add_argument("--n-tumours", type=int, default=10_000,
                        help="Number of base tumours to generate (if --tumour-dir not set).")
    parser.add_argument("--n-augmentations", type=int, default=3)
    parser.add_argument("--simulator", type=str, default="bamsurgeon",
                        choices=["bamsurgeon", "quantumcat"])
    parser.add_argument("--n-pilot", type=int, default=200,
                        help="Tumours scored exhaustively for encoder pre-training.")
    parser.add_argument("--n-workers", type=int, default=4,
                        help="Parallel workers for pilot phase.")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Encoder pre-training epochs.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--work-dir", type=str, default="data/online_train")
    parser.add_argument("--out", type=str, default="data/models/router_online.pt",
                        help="Output path for trained model weights.")
    parser.add_argument("--exclude-configs", type=str, nargs="*", default=None,
                        help="Config names to skip during training.  Defaults to "
                             "['phyloclone_style'].  Pass --exclude-configs (no args) "
                             "to include all configs.")
    parser.add_argument("--checkpoint-every", type=int, default=1000,
                        help="Save model checkpoint every N online tumours (default 1000).")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="TensorBoard log directory.  If omitted, defaults to "
                             "<work-dir>/tb_logs.")
    parser.add_argument("--save-corpus", type=str, default=None,
                        help="Optional: assemble and save corpus to this .npz path.")
    args = parser.parse_args()

    # Resolve exclude_configs: None → use default, empty list → no exclusions
    if args.exclude_configs is None:
        exclude_configs = None  # train_online will use DEFAULT_EXCLUDE_CONFIGS
    else:
        exclude_configs = frozenset(args.exclude_configs)

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Resolve tumour directory ---
    if args.tumour_dir is not None:
        tumour_dir = Path(args.tumour_dir)
    else:
        tumour_dir = work_dir / "tumours"
        manifest_path = tumour_dir / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            print(f"Reusing {manifest['n_total']} existing tumours from {tumour_dir}")
        else:
            from uniclone.router.training import generate_tumours

            print(f"Generating {args.n_tumours} base tumours...")
            t0 = time.perf_counter()
            n = generate_tumours(
                out_dir=tumour_dir,
                n_tumours=args.n_tumours,
                n_augmentations=args.n_augmentations,
                seed=args.seed,
                simulator=args.simulator,
            )
            print(f"  Generated {n} tumours in {time.perf_counter() - t0:.1f}s")

    scores_dir = work_dir / "scores"

    # --- Online training ---
    from uniclone.router.constants import DEFAULT_EXCLUDE_CONFIGS, N_CONFIGS
    from uniclone.router.training import train_online

    n_active = N_CONFIGS - len(exclude_configs if exclude_configs is not None
                                else DEFAULT_EXCLUDE_CONFIGS)
    n_tumour_files = len(list(tumour_dir.glob("tumour_*.npz")))
    exhaustive = n_tumour_files * n_active
    estimated = args.n_pilot * n_active + max(0, n_tumour_files - args.n_pilot)
    print(
        f"Online training: {n_tumour_files} tumours, "
        f"{n_active} active configs (excluding "
        f"{list(exclude_configs if exclude_configs is not None else DEFAULT_EXCLUDE_CONFIGS)}), "
        f"{args.n_pilot} pilot (exhaustive), rest bandit\n"
        f"  Estimated fits: ~{estimated:,} (vs {exhaustive:,} exhaustive, "
        f"{exhaustive / max(estimated, 1):.1f}× speedup)"
    )

    log_dir = Path(args.log_dir) if args.log_dir else work_dir / "tb_logs"

    t0 = time.perf_counter()
    result = train_online(
        tumour_dir=tumour_dir,
        scores_dir=scores_dir,
        n_pilot=args.n_pilot,
        n_workers=args.n_workers,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        exclude_configs=exclude_configs,
        checkpoint_dir=work_dir / "checkpoints",
        checkpoint_every=args.checkpoint_every,
        log_dir=log_dir,
    )
    elapsed = time.perf_counter() - t0

    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"  Pilot: {result.n_pilot} tumours")
    print(f"  Online: {result.n_online} tumours")
    print(f"  Total fits: {result.n_total_fits:,}")

    # Selection distribution
    from collections import Counter
    sel_counts = Counter(result.selections)
    print(f"  Online config selections ({len(result.selections)} tumours):")
    for cfg, count in sel_counts.most_common():
        print(f"    {cfg}: {count} ({100 * count / len(result.selections):.1f}%)")

    # Save model
    result.model.save(out_path)
    print(f"\nSaved model to {out_path}")

    # Save metadata
    stem = out_path.with_suffix("")
    meta = {
        "n_pilot": result.n_pilot,
        "n_online": result.n_online,
        "n_total_fits": result.n_total_fits,
        "n_active_configs": n_active,
        "excluded_configs": sorted(exclude_configs if exclude_configs is not None
                                    else DEFAULT_EXCLUDE_CONFIGS),
        "n_epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "training_time_s": round(elapsed, 2),
        "final_encoder_loss": result.train_losses[-1] if result.train_losses else None,
        "selection_counts": dict(sel_counts),
    }
    meta_path = Path(str(stem) + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Saved metadata to {meta_path}")

    # Optionally assemble and save corpus
    if args.save_corpus:
        from scripts._corpus_io import save_corpus
        from uniclone.router.training import assemble_corpus

        corpus = assemble_corpus(tumour_dir, scores_dir)
        save_corpus(corpus, args.save_corpus, extras=meta)
        print(f"Saved corpus ({len(corpus)} entries) to {args.save_corpus}")


if __name__ == "__main__":
    main()
