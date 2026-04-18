"""
scripts._corpus_io
==================

Serialize / deserialize a list[CorpusEntry] as NPZ + JSON sidecar.

NPZ arrays:
    features       (N, 21)  float64
    subchallenges  (N,)     int32
    config_indices (N,)     int32
    scores         (N,)     float64

JSON sidecar (.meta.json):
    CONFIG_NAMES, SUBCHALLENGES, FEATURE_NAMES, n_entries, timestamp, extras
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from uniclone.router.constants import CONFIG_NAMES, FEATURE_NAMES, SUBCHALLENGES
from uniclone.router.training import CorpusEntry


def save_corpus(
    entries: list[CorpusEntry],
    path: str | Path,
    *,
    extras: dict[str, Any] | None = None,
) -> None:
    """
    Save corpus entries to ``path`` (.npz) with a ``.meta.json`` sidecar.

    Parameters
    ----------
    entries : list of CorpusEntry
    path : output .npz path
    extras : optional dict merged into the JSON sidecar
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n = len(entries)
    n_feat = len(FEATURE_NAMES)

    features = np.empty((n, n_feat), dtype=np.float64)
    subchallenges = np.empty(n, dtype=np.int32)
    config_indices = np.empty(n, dtype=np.int32)
    scores = np.empty(n, dtype=np.float64)

    sc_list = list(SUBCHALLENGES)
    cn_list = list(CONFIG_NAMES)

    for i, e in enumerate(entries):
        features[i] = e.features
        subchallenges[i] = sc_list.index(e.subchallenge)
        config_indices[i] = cn_list.index(e.config_name)
        scores[i] = e.score

    np.savez_compressed(
        path,
        features=features,
        subchallenges=subchallenges,
        config_indices=config_indices,
        scores=scores,
    )

    meta: dict[str, Any] = {
        "n_entries": n,
        "timestamp": datetime.now(UTC).isoformat(),
        "CONFIG_NAMES": cn_list,
        "SUBCHALLENGES": [sc.name for sc in sc_list],
        "FEATURE_NAMES": list(FEATURE_NAMES),
    }
    if extras:
        meta.update(extras)

    meta_path = path.with_suffix("").with_suffix(path.suffix + ".meta.json")
    # e.g. corpus.npz -> corpus.npz.meta.json
    # Simpler: just replace .npz with .meta.json
    meta_path = path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")


def load_corpus(path: str | Path) -> list[CorpusEntry]:
    """
    Load corpus from NPZ + sidecar.

    Parameters
    ----------
    path : .npz file path

    Returns
    -------
    list of CorpusEntry
    """
    path = Path(path)
    data = np.load(path)

    features = data["features"]
    subchallenges_idx = data["subchallenges"]
    config_indices = data["config_indices"]
    scores = data["scores"]

    sc_list = list(SUBCHALLENGES)
    cn_list = list(CONFIG_NAMES)

    entries: list[CorpusEntry] = []
    for i in range(len(scores)):
        entries.append(
            CorpusEntry(
                features=features[i],
                subchallenge=sc_list[int(subchallenges_idx[i])],
                config_name=cn_list[int(config_indices[i])],
                score=float(scores[i]),
            )
        )
    return entries
