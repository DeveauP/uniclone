"""
Tests for MCMCInference.

All tests are skipped if pymc is not installed.
Uses small data (50 mutations, 500 draws) for speed.
"""
from __future__ import annotations

import numpy as np
import pytest

try:
    import pymc  # noqa: F401

    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False

from uniclone.core.config import CONFIGS, CloneConfig
from uniclone.emission.binomial import BinomialEmission
from uniclone.k_prior.bic import BICPrior
from uniclone.phylo.none import NoPhylo


@pytest.fixture
def quantumclone_v1_config() -> CloneConfig:
    return CONFIGS["quantumclone_v1"]


@pytest.fixture
def emission(quantumclone_v1_config: CloneConfig) -> BinomialEmission:
    return BinomialEmission(quantumclone_v1_config)


@pytest.fixture
def phylo(quantumclone_v1_config: CloneConfig) -> NoPhylo:
    return NoPhylo(quantumclone_v1_config)


@pytest.fixture
def small_1clone() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Small 1-clone dataset: 50 mutations, depth 100, cellularity 0.5."""
    rng = np.random.default_rng(42)
    n_mut = 50
    depth = np.full((n_mut, 1), 100, dtype=float)
    alt = rng.binomial(100, 0.5, size=(n_mut, 1)).astype(float)
    adj = np.ones((n_mut, 1), dtype=float)
    return alt, depth, adj


class TestMCMCImport:
    def test_import_error_when_pymc_absent(self) -> None:
        """MCMCInference raises ImportError if pymc is not installed."""
        if HAS_PYMC:
            pytest.skip("pymc is installed")
        from uniclone.inference.mcmc import MCMCInference

        with pytest.raises(ImportError, match="PyMC"):
            MCMCInference(CONFIGS["quantumclone_v1"])


@pytest.mark.skipif(not HAS_PYMC, reason="pymc not installed")
class TestMCMC1Clone:
    def test_recovers_single_center(
        self,
        small_1clone: tuple[np.ndarray, np.ndarray, np.ndarray],
        emission: BinomialEmission,
        phylo: NoPhylo,
        quantumclone_v1_config: CloneConfig,
    ) -> None:
        from uniclone.inference.mcmc import MCMCInference

        alt, depth, adj = small_1clone
        mcmc = MCMCInference(
            quantumclone_v1_config, n_draws=500, n_tune=200, target_accept=0.8
        )
        bic_prior = BICPrior(quantumclone_v1_config, nclone_range=[1])
        centers_init = bic_prior.initialise(1, alt, depth, adj)
        result = mcmc.run(alt, depth, adj, emission, phylo, centers_init, K=1)
        np.testing.assert_allclose(result.centers[0, 0], 0.5, atol=0.1)

    def test_result_fields(
        self,
        small_1clone: tuple[np.ndarray, np.ndarray, np.ndarray],
        emission: BinomialEmission,
        phylo: NoPhylo,
        quantumclone_v1_config: CloneConfig,
    ) -> None:
        from uniclone.inference.mcmc import MCMCInference

        alt, depth, adj = small_1clone
        mcmc = MCMCInference(
            quantumclone_v1_config, n_draws=500, n_tune=200, target_accept=0.8
        )
        bic_prior = BICPrior(quantumclone_v1_config, nclone_range=[1])
        centers_init = bic_prior.initialise(1, alt, depth, adj)
        result = mcmc.run(alt, depth, adj, emission, phylo, centers_init, K=1)
        assert result.K == 1
        assert np.isfinite(result.log_likelihood)
        assert np.isfinite(result.bic)
        assert result.converged is True
        np.testing.assert_allclose(
            result.responsibilities.sum(axis=1), 1.0, atol=1e-10
        )
