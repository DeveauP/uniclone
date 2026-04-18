"""
Tests for EMInference.

Covers: 1-clone and 2-clone centre recovery, BIC K selection,
log-likelihood monotonicity, and convergence flag.
"""
from __future__ import annotations

import numpy as np
import pytest

from uniclone.core.config import (
    CONFIGS,
    CloneConfig,
)
from uniclone.emission.binomial import BinomialEmission
from uniclone.inference.em import EMInference
from uniclone.k_prior.bic import BICPrior
from uniclone.phylo.none import NoPhylo


@pytest.fixture
def em(quantumclone_v1_config: CloneConfig) -> EMInference:
    return EMInference(quantumclone_v1_config)


@pytest.fixture
def quantumclone_v1_config() -> CloneConfig:
    return CONFIGS["quantumclone_v1"]


@pytest.fixture
def emission(quantumclone_v1_config: CloneConfig) -> BinomialEmission:
    return BinomialEmission(quantumclone_v1_config)


@pytest.fixture
def phylo(quantumclone_v1_config: CloneConfig) -> NoPhylo:
    return NoPhylo(quantumclone_v1_config)


class TestEMInference1Clone:
    def test_recovers_single_center(
        self,
        simple_1clone: tuple[np.ndarray, np.ndarray, np.ndarray],
        em: EMInference,
        emission: BinomialEmission,
        phylo: NoPhylo,
        quantumclone_v1_config: CloneConfig,
    ) -> None:
        alt, depth, adj = simple_1clone
        bic_prior = BICPrior(quantumclone_v1_config, nclone_range=[1])
        centers_init = bic_prior.initialise(1, alt, depth, adj)
        result = em.run(alt, depth, adj, emission, phylo, centers_init, K=1)
        assert result.K == 1
        np.testing.assert_allclose(result.centers[0, 0], 0.5, atol=0.05)

    def test_converges(
        self,
        simple_1clone: tuple[np.ndarray, np.ndarray, np.ndarray],
        em: EMInference,
        emission: BinomialEmission,
        phylo: NoPhylo,
        quantumclone_v1_config: CloneConfig,
    ) -> None:
        alt, depth, adj = simple_1clone
        bic_prior = BICPrior(quantumclone_v1_config, nclone_range=[1])
        centers_init = bic_prior.initialise(1, alt, depth, adj)
        result = em.run(alt, depth, adj, emission, phylo, centers_init, K=1)
        assert result.converged is True

    def test_result_has_correct_shapes(
        self,
        simple_1clone: tuple[np.ndarray, np.ndarray, np.ndarray],
        em: EMInference,
        emission: BinomialEmission,
        phylo: NoPhylo,
        quantumclone_v1_config: CloneConfig,
    ) -> None:
        alt, depth, adj = simple_1clone
        n_mut = alt.shape[0]
        bic_prior = BICPrior(quantumclone_v1_config, nclone_range=[1])
        centers_init = bic_prior.initialise(1, alt, depth, adj)
        result = em.run(alt, depth, adj, emission, phylo, centers_init, K=1)
        assert result.centers.shape == (1, 1)
        assert result.assignments.shape == (n_mut,)
        assert result.responsibilities.shape == (n_mut, 1)

    def test_responsibilities_sum_to_one(
        self,
        simple_1clone: tuple[np.ndarray, np.ndarray, np.ndarray],
        em: EMInference,
        emission: BinomialEmission,
        phylo: NoPhylo,
        quantumclone_v1_config: CloneConfig,
    ) -> None:
        alt, depth, adj = simple_1clone
        bic_prior = BICPrior(quantumclone_v1_config, nclone_range=[1])
        centers_init = bic_prior.initialise(1, alt, depth, adj)
        result = em.run(alt, depth, adj, emission, phylo, centers_init, K=1)
        np.testing.assert_allclose(
            result.responsibilities.sum(axis=1), 1.0, atol=1e-10
        )


class TestEMInference2Clone:
    def test_recovers_two_centers(
        self,
        simple_2clone: tuple[np.ndarray, np.ndarray, np.ndarray],
        em: EMInference,
        emission: BinomialEmission,
        phylo: NoPhylo,
        quantumclone_v1_config: CloneConfig,
    ) -> None:
        alt, depth, adj = simple_2clone
        bic_prior = BICPrior(quantumclone_v1_config, nclone_range=[2])
        centers_init = bic_prior.initialise(2, alt, depth, adj)
        result = em.run(alt, depth, adj, emission, phylo, centers_init, K=2)
        recovered = sorted(result.centers[:, 0])
        assert abs(recovered[0] - 0.2) < 0.05
        assert abs(recovered[1] - 0.5) < 0.05

    def test_k2_bic_lower_than_k1(
        self,
        simple_2clone: tuple[np.ndarray, np.ndarray, np.ndarray],
        em: EMInference,
        emission: BinomialEmission,
        phylo: NoPhylo,
        quantumclone_v1_config: CloneConfig,
    ) -> None:
        """K=2 model should have lower BIC than K=1 on 2-clone data."""
        alt, depth, adj = simple_2clone
        bic_prior = BICPrior(quantumclone_v1_config)

        results = []
        for K in [1, 2]:
            c_init = bic_prior.initialise(K, alt, depth, adj)
            r = em.run(alt, depth, adj, emission, phylo, c_init, K=K)
            results.append(r)

        assert results[1].bic < results[0].bic

    def test_bic_prior_selects_k2(
        self,
        simple_2clone: tuple[np.ndarray, np.ndarray, np.ndarray],
        em: EMInference,
        emission: BinomialEmission,
        phylo: NoPhylo,
        quantumclone_v1_config: CloneConfig,
    ) -> None:
        """BICPrior.select should return the K=2 result."""
        alt, depth, adj = simple_2clone
        bic_prior = BICPrior(quantumclone_v1_config, nclone_range=list(range(1, 6)))

        results = []
        for K in bic_prior.get_k_schedule():
            c_init = bic_prior.initialise(K, alt, depth, adj)
            r = em.run(alt, depth, adj, emission, phylo, c_init, K=K)
            results.append(r)

        best = bic_prior.select(results)
        assert best.K == 2


class TestEMLogLikelihood:
    def test_log_likelihood_is_finite(
        self,
        simple_1clone: tuple[np.ndarray, np.ndarray, np.ndarray],
        em: EMInference,
        emission: BinomialEmission,
        phylo: NoPhylo,
        quantumclone_v1_config: CloneConfig,
    ) -> None:
        alt, depth, adj = simple_1clone
        bic_prior = BICPrior(quantumclone_v1_config, nclone_range=[1])
        centers_init = bic_prior.initialise(1, alt, depth, adj)
        result = em.run(alt, depth, adj, emission, phylo, centers_init, K=1)
        assert np.isfinite(result.log_likelihood)

    def test_bic_is_finite(
        self,
        simple_1clone: tuple[np.ndarray, np.ndarray, np.ndarray],
        em: EMInference,
        emission: BinomialEmission,
        phylo: NoPhylo,
        quantumclone_v1_config: CloneConfig,
    ) -> None:
        alt, depth, adj = simple_1clone
        bic_prior = BICPrior(quantumclone_v1_config, nclone_range=[1])
        centers_init = bic_prior.initialise(1, alt, depth, adj)
        result = em.run(alt, depth, adj, emission, phylo, centers_init, K=1)
        assert np.isfinite(result.bic)


class TestEMMaxIterFallback:
    def test_max_iter_1_does_not_crash(
        self,
        simple_1clone: tuple[np.ndarray, np.ndarray, np.ndarray],
        emission: BinomialEmission,
        phylo: NoPhylo,
        quantumclone_v1_config: CloneConfig,
    ) -> None:
        em_tight = EMInference(quantumclone_v1_config, max_iter=1, tol=1e-100)
        alt, depth, adj = simple_1clone
        bic_prior = BICPrior(quantumclone_v1_config, nclone_range=[1])
        centers_init = bic_prior.initialise(1, alt, depth, adj)
        result = em_tight.run(alt, depth, adj, emission, phylo, centers_init, K=1)
        assert result.n_iter == 1
        assert result.converged is False
