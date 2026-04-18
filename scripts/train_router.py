"""
scripts.train_router
=====================

Train a NeuralTS MetaRouter from a serialized corpus.

Usage::

    python -m scripts.train_router --corpus data/corpus/corpus.npz \
        --val-frac 0.2 --epochs 50 --batch-size 256 --lr 1e-3 \
        --seed 0 --out data/models/router_v1.pt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from scripts._corpus_io import load_corpus, save_corpus
from uniclone.router.training import CorpusEntry, train_router_detailed


def _split_by_tumour(
    corpus: list[CorpusEntry],
    val_frac: float,
    seed: int,
) -> tuple[list[CorpusEntry], list[CorpusEntry]]:
    """Split corpus into train/val by unique tumour (feature bytes), not by entry."""
    # Group entries by tumour identity
    tumour_map: dict[bytes, list[CorpusEntry]] = {}
    for e in corpus:
        key = e.features.tobytes()
        tumour_map.setdefault(key, []).append(e)

    keys = sorted(tumour_map.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)

    n_val = max(1, int(len(keys) * val_frac))
    val_keys = set(keys[:n_val])

    train_entries: list[CorpusEntry] = []
    val_entries: list[CorpusEntry] = []
    for key in keys:
        target = val_entries if key in val_keys else train_entries
        target.extend(tumour_map[key])

    return train_entries, val_entries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train NeuralTS MetaRouter from a corpus.",
    )
    parser.add_argument("--corpus", type=str, required=True)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="data/models/router_v1.pt")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stem = out_path.with_suffix("")  # e.g. data/models/router_v1

    # Load corpus
    print(f"Loading corpus from {args.corpus}")
    corpus = load_corpus(args.corpus)
    print(f"  {len(corpus)} entries loaded")

    # Split
    train_corpus, val_corpus = _split_by_tumour(corpus, args.val_frac, args.seed)
    n_train_tumours = len({e.features.tobytes() for e in train_corpus})
    n_val_tumours = len({e.features.tobytes() for e in val_corpus})
    print(
        f"  Split: {n_train_tumours} train tumours ({len(train_corpus)} entries), "
        f"{n_val_tumours} val tumours ({len(val_corpus)} entries)"
    )

    # Train
    print(
        f"Training: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}"
    )
    t0 = time.perf_counter()
    result = train_router_detailed(
        train_corpus=train_corpus,
        val_corpus=val_corpus if val_corpus else None,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    elapsed = time.perf_counter() - t0
    print(f"  Training complete in {elapsed:.1f}s")

    if result.train_losses:
        print(f"  Final train loss: {result.train_losses[-1]:.4f}")
    if result.val_losses:
        best_val = min(result.val_losses)
        best_epoch = result.val_losses.index(best_val)
        print(f"  Best val loss: {best_val:.4f} (epoch {best_epoch + 1})")

    # Save model
    result.model.save(out_path)
    print(f"Saved model to {out_path}")

    # Save losses
    losses_path = Path(str(stem) + ".losses.json")
    losses_data = {"train": result.train_losses, "val": result.val_losses}
    losses_path.write_text(json.dumps(losses_data, indent=2) + "\n")
    print(f"Saved losses to {losses_path}")

    # Save metadata
    meta_path = Path(str(stem) + ".meta.json")
    meta = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "val_frac": args.val_frac,
        "corpus": args.corpus,
        "n_train_entries": len(train_corpus),
        "n_val_entries": len(val_corpus),
        "n_train_tumours": n_train_tumours,
        "n_val_tumours": n_val_tumours,
        "training_time_s": round(elapsed, 2),
        "final_train_loss": result.train_losses[-1] if result.train_losses else None,
        "final_val_loss": result.val_losses[-1] if result.val_losses else None,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Saved metadata to {meta_path}")

    # Save validation corpus
    if val_corpus:
        val_corpus_path = Path(str(stem) + ".val_corpus.npz")
        save_corpus(val_corpus, val_corpus_path)
        print(f"Saved val corpus to {val_corpus_path}")


if __name__ == "__main__":
    main()
