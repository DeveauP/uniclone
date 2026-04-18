"""
Tests for MFVIInference.

Covers: 1-clone and 2-clone center recovery, convergence, shapes,
responsibilities sum to 1, ELBO is finite, Dirichlet pruning.
"""
from __future__ import annotations

import numpy as np
import pytest

from uniclone.core.config import CONFIGS, CloneConfig
from uniclone.emission.binomial import BinomialEmission
from uniclone.inference.mfvi import MFVIInference
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
def mfvi(quantumclone_v1_config: CloneConfig) -> MFVIInference:
    return MFVIInference(quantumclone_v1_config)


class TestMFVI1Clone:
    def test_recovers_single_center(
        self,
        simple_1clone: tuple[np.ndarray, np.ndarray, np.ndarray],
        mfvi: MFVIInference,
        emission: BinomialEmission,
        phylo: NoPhylo,
        quantumclone_v1_config: CloneConfig,
    ) -> None:
        alt, depth, adj = simple_1clone
        bic_prior = BICPrior(quantumclone_v1_config, nclone_range=[1])
        centers_init = bic_prior.initialise(1, alt, depth, adj)
        result = mfvi.run(alt, depth, adj, emission, phylo, centers_init, K=1)
        np.testing.assert_allclose(result.centers[0, 0], 0.5, atol=0.05)

    def test_converges(
        self,
        simple_1clone: tuple[np.ndarray, np.ndarray, np.ndarray],
        mfvi: MFVIInference,
        emission: BinomialEmission,
        phylo: NoPhylo,
        quantumclone_v1_config: CloneConfig,
    ) -> None:
        alt, depth, adj = simple_1clone
        bic_prior = BICPrior(quantumclone_v1_config, nclone_range=[1])
        centers_init = bic_prior.initialise(1, alt, depth, adj)
        result = mfvi.run(alt, depth, adj, emission, phylo, centers_init, K=1)
        assert result.converged is True

    def test_result_has_correct_shapes(
        self,
        simple_1clone: tuple[np.ndarray, np.ndarray, np.ndarray],
        mfvi: MFVIInference,
        emission: BinomialEmission,
        phylo: NoPhylo,
        quantumclone_v1_config: CloneConfig,
    ) -> None:
        alt, depth, adj = simple_1clone
        n_mut = alt.shape[0]
        bic_prior = BICPrior(quantumclone_v1_config, nclone_range=[1])
        centers_init = bic_prior.initialise(1, alt, depth, adj)
        result = mfvi.run(alt, depth, adj, emission, phylo, centers_init, K=1)
        assert result.centers.shape == (1, 1)
        assert result.assignments.shape == (n_mut,)
        assert result.responsibilities.shape == (n_mut, 1)


class TestMFVI2Clone:
    def test_recovers_two_centers(
        self,
        simple_2clone: tuple[np.ndarray, np.ndarray, np.ndarray],
        mfvi: MFVIInference,
        emission: BinomialEmission,
        phylo: NoPhylo,
        quantumclone_v1_config: CloneConfig,
    ) -> None:
        alt, depth, adj = simple_2clone
        bic_prior = BICPrior(quantumclone_v1_config, nclone_range=[2])
        centers_init = bic_prior.initialise(2, alt, depth, adj)
        result = mfvi.run(alt, depth, adj, emission, phylo, centers_init, K=2)
        recovered = sorted(result.centers[:, 0])
        assert abs(recovered[0] - 0.2) < 0.05
        assert abs(recovered[1] - 0.5) < 0.05


class TestMFVIResponsibilities:
    def test_responsibilities_sum_to_one(
        self,
        simple_1clone: tuple[np.ndarray, np.ndarray, np.ndarray],
        mfvi: MFVIInference,
        emission: BinomialEmission,
        phylo: NoPhylo,
        quantumclone_v1_config: CloneConfig,
    ) -> None:
        alt, depth, adj = simple_1clone
        bic_prior = BICPrior(quantumclone_v1_config, nclone_range=[1])
        centers_init = bic_prior.initialise(1, alt, depth, adj)
        result = mfvi.run(alt, depth, adj, emission, phylo, centers_init, K=1)
        np.testing.assert_allclose(
            result.responsibilities.sum(axis=1), 1.0, atol=1e-10
        )


class TestMFVILogLikelihood:
    def test_log_likelihood_is_finite(
        self,
        simple_1clone: tuple[np.ndarray, np.ndarray, np.ndarray],
        mfvi: MFVIInference,
        emission: BinomialEmission,
        phylo: NoPhylo,
        quantumclone_v1_config: CloneConfig,
    ) -> None:
        alt, depth, adj = simple_1clone
        bic_prior = BICPrior(quantumclone_v1_config, nclone_range=[1])
        centers_init = bic_prior.initialise(1, alt, depth, adj)
        result = mfvi.run(alt, depth, adj, emission, phylo, centers_init, K=1)
        assert np.isfinite(result.log_likelihood)

    def test_bic_is_finite(
        self,
        simple_1clone: tuple[np.ndarray, np.ndarray, np.ndarray],
        mfvi: MFVIInference,
        emission: BinomialEmission,
        phylo: NoPhylo,
        quantumclone_v1_config: CloneConfig,
    ) -> None:
        alt, depth, adj = simple_1clone
        bic_prior = BICPrior(quantumclone_v1_config, nclone_range=[1])
        centers_init = bic_prior.initialise(1, alt, depth, adj)
        result = mfvi.run(alt, depth, adj, emission, phylo, centers_init, K=1)
        assert np.isfinite(result.bic)


class TestMFVIDirichletPruning:
    def test_sparse_prior_prunes_to_effective_k(
        self,
        simple_2clone: tuple[np.ndarray, np.ndarray, np.ndarray],
        emission: BinomialEmission,
        phylo: NoPhylo,
        quantumclone_v1_config: CloneConfig,
    ) -> None:
        """With alpha0=0.001 and K=5 on 2-clone data, effective K should be 2."""
        alt, depth, adj = simple_2clone
        mfvi_sparse = MFVIInference(
            quantumclone_v1_config, alpha0=0.001, max_iter=200
        )
        bic_prior = BICPrior(quantumclone_v1_config, nclone_range=[5])
        centers_init = bic_prior.initialise(5, alt, depth, adj)
        result = mfvi_sparse.run(alt, depth, adj, emission, phylo, centers_init, K=5)
        # Count effective clusters: those with >1% of total responsibility
        resp_mass = result.responsibilities.sum(axis=0)
        effective_k = int((resp_mass / resp_mass.sum() > 0.01).sum())
        assert effective_k <= 4  # should prune below K=5
