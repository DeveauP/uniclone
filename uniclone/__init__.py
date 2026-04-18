"""
uniclone
========

Unified probabilistic framework for tumour clonal reconstruction.

A single framework that subsumes PyClone, PyClone-VI, MOBSTER, DeCiFer,
CONIPHER, CALDER, and Pairtree as named configuration presets, with a
learned MetaRouter that selects the right configuration per tumour.

Usage levels
------------

**Level 1 — fully automatic** (Phase 6, MetaRouter not yet available)::

    from uniclone import UniClone  # coming in Phase 6

**Level 2 — named preset**::

    from uniclone import GenerativeModel, CONFIGS

    model = GenerativeModel(CONFIGS["quantumclone_v1"])
    result = model.fit(alt, depth, adj_factor)

**Level 3 — custom configuration**::

    from uniclone import GenerativeModel, CloneConfig
    from uniclone import EmissionModel, InferenceEngine, KPrior, PhyloMode, NoiseModel

    config = CloneConfig(
        emission=EmissionModel.BINOMIAL,
        inference=InferenceEngine.EM,
        k_prior=KPrior.BIC,
        phylo=PhyloMode.POST_HOC,
        noise=NoiseModel.NONE,
    )
    model = GenerativeModel(config)
    result = model.fit(alt, depth)
"""

from uniclone.core.backend import get_backend, set_backend
from uniclone.core.config import (
    CONFIGS,
    CloneConfig,
    EmissionModel,
    InferenceEngine,
    KPrior,
    NoiseModel,
    PhyloMode,
)
from uniclone.core.model import GenerativeModel
from uniclone.core.types import CloneResult
from uniclone.uniclone import UniClone

__all__ = [
    "CloneConfig",
    "EmissionModel",
    "InferenceEngine",
    "KPrior",
    "PhyloMode",
    "NoiseModel",
    "CONFIGS",
    "GenerativeModel",
    "CloneResult",
    "UniClone",
    "set_backend",
    "get_backend",
]
