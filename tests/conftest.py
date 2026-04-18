"""
Shared pytest fixtures for UniClone Phase 0 tests.
"""
from __future__ import annotations

import numpy as np
import pytest

from uniclone.core.config import (
    CONFIGS,
    CloneConfig,
    EmissionModel,
    InferenceEngine,
    KPrior,
    NoiseModel,
    PhyloMode,
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def simple_1clone(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Single clone at cellularity 0.5, 200 mutations, depth 100, 1 sample.

    Returns (alt, depth, adj_factor), all shape (200, 1).
    """
    n_mut = 200
    depth = np.full((n_mut, 1), 100, dtype=float)
    alt = rng.binomial(100, 0.5, size=(n_mut, 1)).astype(float)
    adj = np.ones((n_mut, 1), dtype=float)
    return alt, depth, adj


@pytest.fixture
def simple_2clone(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Two clones at cellularities 0.2 and 0.5, 200 mutations each, depth 100.

    Returns (alt, depth, adj_factor), all shape (400, 1).
    """
    depth = np.full((400, 1), 100, dtype=float)
    alt = np.concatenate([
        rng.binomial(100, 0.2, size=(200, 1)),
        rng.binomial(100, 0.5, size=(200, 1)),
    ], axis=0).astype(float)
    adj = np.ones((400, 1), dtype=float)
    return alt, depth, adj


@pytest.fixture
def multisample_2clone(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Two clones, 2 samples, 200 mutations each.
    Sample 1: cellularities 0.3 and 0.6.
    Sample 2: cellularities 0.1 and 0.4.
    """
    n = 400
    depth = np.full((n, 2), 100, dtype=float)
    alt = np.zeros((n, 2), dtype=float)
    # Clone 1 mutations
    alt[:200, 0] = rng.binomial(100, 0.3, size=200)
    alt[:200, 1] = rng.binomial(100, 0.1, size=200)
    # Clone 2 mutations
    alt[200:, 0] = rng.binomial(100, 0.6, size=200)
    alt[200:, 1] = rng.binomial(100, 0.4, size=200)
    adj = np.ones((n, 2), dtype=float)
    return alt, depth, adj


@pytest.fixture
def quantumclone_v1_config() -> CloneConfig:
    return CONFIGS["quantumclone_v1"]


@pytest.fixture
def binomial_emission(quantumclone_v1_config: CloneConfig) -> object:
    from uniclone.emission.binomial import BinomialEmission
    return BinomialEmission(quantumclone_v1_config)


@pytest.fixture
def longitudinal_3clone(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Three clones, 3 timepoints (longitudinal), 200 mutations each.

    Timepoint 1: cellularities 0.7, 0.4, 0.2
    Timepoint 2: cellularities 0.6, 0.3, 0.1
    Timepoint 3: cellularities 0.5, 0.2, 0.05
    """
    n = 600
    depth = np.full((n, 3), 100, dtype=float)
    alt = np.zeros((n, 3), dtype=float)
    cells = np.array([
        [0.7, 0.6, 0.5],
        [0.4, 0.3, 0.2],
        [0.2, 0.1, 0.05],
    ])
    for k in range(3):
        for t in range(3):
            alt[k * 200:(k + 1) * 200, t] = rng.binomial(100, cells[k, t], size=200)
    adj = np.ones((n, 3), dtype=float)
    return alt, depth, adj


@pytest.fixture
def multisample_3clone(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Three clones, 2 samples, 200 mutations each.

    Sample 1: cellularities 0.6, 0.35, 0.15
    Sample 2: cellularities 0.5, 0.3, 0.1
    """
    n = 600
    depth = np.full((n, 2), 100, dtype=float)
    alt = np.zeros((n, 2), dtype=float)
    cells = np.array([
        [0.6, 0.5],
        [0.35, 0.3],
        [0.15, 0.1],
    ])
    for k in range(3):
        for s in range(2):
            alt[k * 200:(k + 1) * 200, s] = rng.binomial(100, cells[k, s], size=200)
    adj = np.ones((n, 2), dtype=float)
    return alt, depth, adj


@pytest.fixture
def dirichlet_mfvi_config() -> CloneConfig:
    """Config matching pyclone_vi: Beta-Binomial + MFVI + Dirichlet K-prior."""
    return CloneConfig(
        emission=EmissionModel.BETA_BINOMIAL,
        inference=InferenceEngine.MFVI,
        k_prior=KPrior.DIRICHLET,
        phylo=PhyloMode.NONE,
        noise=NoiseModel.NONE,
        phi=50.0,
    )


# ---------------------------------------------------------------------------
# Fixtures (MetaRouter / NeuralTS)
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_tumour():
    """A synthetic tumour from QuantumCat for tests."""
    from uniclone.simulate.quantum_cat import QuantumCatParams, simulate_quantumcat

    params = QuantumCatParams(n_clones=3, n_mutations=200, n_samples=1, purity=0.8, seed=42)
    return simulate_quantumcat(params)


@pytest.fixture
def mini_corpus():
    """Minimal training corpus with 10 entries for fast tests."""
    from uniclone.router.constants import N_FEATURES, SUBCHALLENGES
    from uniclone.router.training import CorpusEntry

    rng_fixture = np.random.default_rng(42)
    entries = []
    for _ in range(2):
        feat = rng_fixture.standard_normal(N_FEATURES)
        for config_name in ["quantumclone_v1", "pyclone_vi", "mobster", "wes_clinical", "wgs_cohort"]:
            for sc in SUBCHALLENGES:
                entries.append(CorpusEntry(
                    features=feat.copy(),
                    subchallenge=sc,
                    config_name=config_name,
                    score=rng_fixture.uniform(0, 1),
                ))
    return entries


@pytest.fixture
def pretrained_router(mini_corpus):
    """A router trained on the mini corpus."""
    try:
        from uniclone.router.router import MetaRouter
        from uniclone.router.training import train_router

        model = train_router(mini_corpus)
        router = MetaRouter()
        router._model = model
        return router
    except ImportError:
        pytest.skip("torch not installed")
