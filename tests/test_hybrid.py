"""
Tests for HybridInference (EM warm-up + MFVI refinement).

Covers: 1-clone and 2-clone center recovery, results similar to standalone MFVI.
"""

from __future__ import annotations

import numpy as np
import pytest

from uniclone.core.config import CONFIGS, CloneConfig
from uniclone.emission.binomial import BinomialEmission
from uniclone.inference.hybrid import HybridInference
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
def hybrid(quantumclone_v1_config: CloneConfig) -> HybridInference:
    return HybridInference(quantumclone_v1_config)


class TestHybrid1Clone:
    def test_recovers_single_center(
        self,
        simple_1clone: tuple[np.ndarray, np.ndarray, np.ndarray],
        hybrid: HybridInference,
        emission: BinomialEmission,
        phylo: NoPhylo,
        quantumclone_v1_config: CloneConfig,
    ) -> None:
        alt, depth, adj = simple_1clone
        bic_prior = BICPrior(quantumclone_v1_config, nclone_range=[1])
        centers_init = bic_prior.initialise(1, alt, depth, adj)
        result = hybrid.run(alt, depth, adj, emission, phylo, centers_init, K=1)
        np.testing.assert_allclose(result.centers[0, 0], 0.5, atol=0.05)


class TestHybrid2Clone:
    def test_recovers_two_centers(
        self,
        simple_2clone: tuple[np.ndarray, np.ndarray, np.ndarray],
        hybrid: HybridInference,
        emission: BinomialEmission,
        phylo: NoPhylo,
        quantumclone_v1_config: CloneConfig,
    ) -> None:
        alt, depth, adj = simple_2clone
        bic_prior = BICPrior(quantumclone_v1_config, nclone_range=[2])
        centers_init = bic_prior.initialise(2, alt, depth, adj)
        result = hybrid.run(alt, depth, adj, emission, phylo, centers_init, K=2)
        recovered = sorted(result.centers[:, 0])
        assert abs(recovered[0] - 0.2) < 0.05
        assert abs(recovered[1] - 0.5) < 0.05


class TestHybridVsMFVI:
    def test_results_similar_to_mfvi(
        self,
        simple_2clone: tuple[np.ndarray, np.ndarray, np.ndarray],
        emission: BinomialEmission,
        phylo: NoPhylo,
        quantumclone_v1_config: CloneConfig,
    ) -> None:
        """Hybrid results should be close to standalone MFVI."""
        alt, depth, adj = simple_2clone
        bic_prior = BICPrior(quantumclone_v1_config, nclone_range=[2])
        centers_init = bic_prior.initialise(2, alt, depth, adj)

        mfvi = MFVIInference(quantumclone_v1_config)
        mfvi_result = mfvi.run(alt, depth, adj, emission, phylo, centers_init, K=2)

        hybrid = HybridInference(quantumclone_v1_config)
        hybrid_result = hybrid.run(alt, depth, adj, emission, phylo, centers_init, K=2)

        mfvi_centers = sorted(mfvi_result.centers[:, 0])
        hybrid_centers = sorted(hybrid_result.centers[:, 0])
        np.testing.assert_allclose(mfvi_centers, hybrid_centers, atol=0.1)

    def test_n_iter_includes_em_warmup(
        self,
        simple_1clone: tuple[np.ndarray, np.ndarray, np.ndarray],
        emission: BinomialEmission,
        phylo: NoPhylo,
        quantumclone_v1_config: CloneConfig,
    ) -> None:
        """n_iter should be at least n_em."""
        alt, depth, adj = simple_1clone
        n_em = 20
        hybrid = HybridInference(quantumclone_v1_config, n_em=n_em)
        bic_prior = BICPrior(quantumclone_v1_config, nclone_range=[1])
        centers_init = bic_prior.initialise(1, alt, depth, adj)
        result = hybrid.run(alt, depth, adj, emission, phylo, centers_init, K=1)
        assert result.n_iter >= n_em
