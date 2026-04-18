"""
Tests for EM-MFVI parity.

With a very large alpha0 (flat Dirichlet), MFVI should reduce to EM and
produce near-identical centers.
"""

from __future__ import annotations

import numpy as np
import pytest

from uniclone.core.config import CONFIGS, CloneConfig
from uniclone.emission.binomial import BinomialEmission
from uniclone.inference.em import EMInference
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


class TestEMMFVIParity:
    def test_1clone_parity(
        self,
        simple_1clone: tuple[np.ndarray, np.ndarray, np.ndarray],
        emission: BinomialEmission,
        phylo: NoPhylo,
        quantumclone_v1_config: CloneConfig,
    ) -> None:
        alt, depth, adj = simple_1clone
        bic_prior = BICPrior(quantumclone_v1_config, nclone_range=[1])
        centers_init = bic_prior.initialise(1, alt, depth, adj)

        em = EMInference(quantumclone_v1_config)
        em_result = em.run(alt, depth, adj, emission, phylo, centers_init, K=1)

        mfvi = MFVIInference(quantumclone_v1_config, alpha0=1e6)
        mfvi_result = mfvi.run(alt, depth, adj, emission, phylo, centers_init, K=1)

        np.testing.assert_allclose(em_result.centers, mfvi_result.centers, atol=0.02)

    def test_2clone_parity(
        self,
        simple_2clone: tuple[np.ndarray, np.ndarray, np.ndarray],
        emission: BinomialEmission,
        phylo: NoPhylo,
        quantumclone_v1_config: CloneConfig,
    ) -> None:
        alt, depth, adj = simple_2clone
        bic_prior = BICPrior(quantumclone_v1_config, nclone_range=[2])
        centers_init = bic_prior.initialise(2, alt, depth, adj)

        em = EMInference(quantumclone_v1_config)
        em_result = em.run(alt, depth, adj, emission, phylo, centers_init, K=2)

        mfvi = MFVIInference(quantumclone_v1_config, alpha0=1e6)
        mfvi_result = mfvi.run(alt, depth, adj, emission, phylo, centers_init, K=2)

        em_sorted = sorted(em_result.centers[:, 0])
        mfvi_sorted = sorted(mfvi_result.centers[:, 0])
        np.testing.assert_allclose(em_sorted, mfvi_sorted, atol=0.02)
