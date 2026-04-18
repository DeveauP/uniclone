"""
scripts.generate_corpus
========================

Generate a labelled training corpus in two decoupled steps:

1. ``generate`` — simulate tumours to disk (fast, seconds)
2. ``score`` — score configs with adaptive elimination (slow, resumable)
3. ``all`` (default) — both steps in sequence

Usage::

    # Full pipeline:
    python -m scripts.generate_corpus all --n-tumours 10000 --n-workers 8

    # Step 1 only (inspect tumours before committing to scoring):
    python -m scripts.generate_corpus generate --n-tumours 10000

    # Step 2 only (resume or re-run scoring):
    python -m scripts.generate_corpus score --tumour-dir data/corpus/tumours \
        --n-workers 8 --elimination-margin 0.15

    # Legacy one-shot (backwards compatible):
    python -m scripts.generate_corpus --n-tumours 10000
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from scripts._corpus_io import save_corpus


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data/corpus/corpus.npz")


def _cmd_generate(args: argparse.Namespace) -> None:
    from uniclone.router.training import generate_tumours

    tumour_dir = Path(args.tumour_dir)
    t0 = time.perf_counter()
    n = generate_tumours(
        out_dir=tumour_dir,
        n_tumours=args.n_tumours,
        n_augmentations=args.n_augmentations,
        seed=args.seed,
        simulator=args.simulator,
    )
    elapsed = time.perf_counter() - t0
    print(f"Generated {n} tumours in {elapsed:.1f}s → {tumour_dir}")


def _cmd_score(args: argparse.Namespace) -> None:
    from uniclone.router.training import score_tumours

    tumour_dir = Path(args.tumour_dir)
    scores_dir = tumour_dir.parent / "scores"

    t0 = time.perf_counter()
    corpus = score_tumours(
        tumour_dir=tumour_dir,
        out_dir=scores_dir,
        n_workers=args.n_workers,
        elimination_margin=args.elimination_margin,
    )
    elapsed = time.perf_counter() - t0
    print(f"Scored {len(corpus)} entries in {elapsed:.1f}s")

    save_corpus(
        corpus, args.out,
        extras={
            "tumour_dir": str(tumour_dir),
            "elimination_margin": args.elimination_margin,
        },
    )
    print(f"Saved to {args.out}")


def _cmd_all(args: argparse.Namespace) -> None:
    from uniclone.router.training import build_training_corpus

    total = args.n_tumours * (1 + args.n_augmentations)
    print(
        f"Generating corpus: {args.n_tumours} base tumours "
        f"× {1 + args.n_augmentations} = {total} tumours, "
        f"{args.n_workers} workers, seed={args.seed}, "
        f"simulator={args.simulator}, "
        f"elimination_margin={args.elimination_margin}"
    )

    work_dir = Path(args.out).with_suffix("") / "work"

    t0 = time.perf_counter()
    corpus = build_training_corpus(
        n_tumours=args.n_tumours,
        n_workers=args.n_workers,
        seed=args.seed,
        simulator=args.simulator,
        n_augmentations=args.n_augmentations,
        elimination_margin=args.elimination_margin,
        work_dir=work_dir,
    )
    elapsed = time.perf_counter() - t0
    print(f"Generated {len(corpus)} entries in {elapsed:.1f}s")

    save_corpus(
        corpus, args.out,
        extras={
            "n_tumours": args.n_tumours,
            "n_augmentations": args.n_augmentations,
            "seed": args.seed,
            "elimination_margin": args.elimination_margin,
        },
    )
    print(f"Saved to {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate MetaRouter training corpus.",
    )
    sub = parser.add_subparsers(dest="command")

    # --- generate ---
    p_gen = sub.add_parser("generate", help="Simulate tumours to disk (fast).")
    p_gen.add_argument("--n-tumours", type=int, default=10_000)
    p_gen.add_argument("--n-augmentations", type=int, default=3)
    p_gen.add_argument("--simulator", type=str, default="bamsurgeon",
                       choices=["bamsurgeon", "quantumcat"])
    p_gen.add_argument("--tumour-dir", type=str, default="data/corpus/tumours")
    _add_common_args(p_gen)

    # --- score ---
    p_score = sub.add_parser("score", help="Score configs on existing tumours (resumable).")
    p_score.add_argument("--tumour-dir", type=str, default="data/corpus/tumours")
    p_score.add_argument("--n-workers", type=int, default=4)
    p_score.add_argument("--elimination-margin", type=float, default=0.25)
    _add_common_args(p_score)

    # --- all (default) ---
    p_all = sub.add_parser("all", help="Generate + score in one go (default).")
    p_all.add_argument("--n-tumours", type=int, default=10_000)
    p_all.add_argument("--n-augmentations", type=int, default=3)
    p_all.add_argument("--n-workers", type=int, default=4)
    p_all.add_argument("--simulator", type=str, default="bamsurgeon",
                       choices=["bamsurgeon", "quantumcat"])
    p_all.add_argument("--elimination-margin", type=float, default=0.25)
    _add_common_args(p_all)

    # Backwards-compat: if no subcommand, treat as "all"
    parser.add_argument("--n-tumours", type=int, default=None)
    parser.add_argument("--n-augmentations", type=int, default=None)
    parser.add_argument("--n-workers", type=int, default=None)
    parser.add_argument("--simulator", type=str, default=None)
    parser.add_argument("--elimination-margin", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=str, default=None)

    args = parser.parse_args()

    if args.command == "generate":
        _cmd_generate(args)
    elif args.command == "score":
        _cmd_score(args)
    elif args.command == "all":
        _cmd_all(args)
    else:
        # No subcommand — backwards compat, run "all" with top-level args
        args.command = "all"
        args.n_tumours = args.n_tumours or 10_000
        args.n_augmentations = args.n_augmentations or 3
        args.n_workers = args.n_workers or 4
        args.simulator = args.simulator or "bamsurgeon"
        args.elimination_margin = args.elimination_margin if args.elimination_margin is not None else 0.25
        args.seed = args.seed or 42
        args.out = args.out or "data/corpus/corpus.npz"
        _cmd_all(args)


if __name__ == "__main__":
    main()
