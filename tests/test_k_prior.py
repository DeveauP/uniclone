"""
Tests for DirichletPrior and TSSBPrior (K-prior modules).

Covers: unit tests for schedule/init/select, end-to-end with MFVI and Hybrid.
"""
from __future__ import annotations

import warnings

import numpy as np

from uniclone import CONFIGS, GenerativeModel
from uniclone.core.config import (
    CloneConfig,
    EmissionModel,
    InferenceEngine,
    KPrior,
    NoiseModel,
    PhyloMode,
)
from uniclone.core.types import CloneResult
from uniclone.k_prior.dirichlet import DirichletPrior
from uniclone.k_prior.tssb import TSSBPrior

# ──────────────────────────────────────────────────────────────────────
# DirichletPrior unit tests
# ──────────────────────────────────────────────────────────────────────


class TestDirichletPriorUnit:
    def _make_config(self) -> CloneConfig:
        return CloneConfig(
            emission=EmissionModel.BETA_BINOMIAL,
            inference=InferenceEngine.MFVI,
            k_prior=KPrior.DIRICHLET,
            phylo=PhyloMode.NONE,
            noise=NoiseModel.NONE,
            phi=50.0,
        )

    def test_get_k_schedule_returns_k_max(self) -> None:
        prior = DirichletPrior(self._make_config(), k_max=8)
        assert prior.get_k_schedule() == [8]

    def test_get_k_schedule_uses_n_clones(self) -> None:
        cfg = CloneConfig(
            emission=EmissionModel.BETA_BINOMIAL,
            inference=InferenceEngine.MFVI,
            k_prior=KPrior.DIRICHLET,
            phylo=PhyloMode.NONE,
            noise=NoiseModel.NONE,
            n_clones=5,
        )
        prior = DirichletPrior(cfg, k_max=10)
        assert prior.get_k_schedule() == [5]

    def test_initialise_shape(self) -> None:
        prior = DirichletPrior(self._make_config(), k_max=6)
        rng = np.random.default_rng(0)
        alt = rng.binomial(100, 0.3, size=(200, 2)).astype(float)
        depth = np.full((200, 2), 100.0)
        adj = np.ones((200, 2))
        centers = prior.initialise(6, alt, depth, adj)
        assert centers.shape == (6, 2)
        assert (centers > 0).all()
        assert (centers < 1).all()

    def test_select_prunes_empty_clusters(self) -> None:
        prior = DirichletPrior(self._make_config(), k_max=10, min_cluster_weight=0.01)
        n_mut = 400
        # Mock: K=10, only clusters 2 and 7 have weight
        resp = np.full((n_mut, 10), 1e-6)
        resp[:200, 2] = 0.99
        resp[200:, 7] = 0.99
        # Normalise rows
        resp = resp / resp.sum(axis=1, keepdims=True)

        mock_result = CloneResult(
            centers=np.random.default_rng(0).uniform(0.1, 0.9, (10, 1)),
            assignments=np.zeros(n_mut, dtype=int),
            responsibilities=resp,
            log_likelihood=-500.0,
            bic=1200.0,
            K=10,
            n_iter=50,
            converged=True,
        )

        pruned = prior.select([mock_result])
        assert pruned.K == 2
        assert pruned.centers.shape == (2, 1)
        assert pruned.responsibilities.shape == (n_mut, 2)
        # Rows should sum to ~1
        row_sums = pruned.responsibilities.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_select_preserves_all_when_no_pruning(self) -> None:
        prior = DirichletPrior(self._make_config(), k_max=3, min_cluster_weight=0.01)
        n_mut = 300
        resp = np.zeros((n_mut, 3))
        resp[:100, 0] = 0.9
        resp[:100, 1] = 0.05
        resp[:100, 2] = 0.05
        resp[100:200, 1] = 0.9
        resp[100:200, 0] = 0.05
        resp[100:200, 2] = 0.05
        resp[200:, 2] = 0.9
        resp[200:, 0] = 0.05
        resp[200:, 1] = 0.05

        mock_result = CloneResult(
            centers=np.array([[0.2], [0.4], [0.6]]),
            assignments=np.repeat([0, 1, 2], 100),
            responsibilities=resp,
            log_likelihood=-400.0,
            bic=900.0,
            K=3,
            n_iter=30,
            converged=True,
        )

        result = prior.select([mock_result])
        assert result.K == 3


# ──────────────────────────────────────────────────────────────────────
# TSSBPrior unit tests
# ──────────────────────────────────────────────────────────────────────


class TestTSSBPriorUnit:
    def _make_config(self) -> CloneConfig:
        return CloneConfig(
            emission=EmissionModel.BETA_BINOMIAL,
            inference=InferenceEngine.MFVI,
            k_prior=KPrior.TSSB,
            phylo=PhyloMode.NONE,
            noise=NoiseModel.NONE,
            phi=50.0,
        )

    def test_experimental_warning(self) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            TSSBPrior(self._make_config())
            assert len(w) == 1
            assert "experimental" in str(w[0].message).lower()

    def test_get_k_schedule_returns_k_max(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prior = TSSBPrior(self._make_config(), k_max=12)
        assert prior.get_k_schedule() == [12]

    def test_stick_breaking_weights_sum_to_one(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prior = TSSBPrior(self._make_config(), k_max=15, alpha=1.0)
        rng = np.random.default_rng(0)
        alt = rng.binomial(100, 0.3, size=(200, 1)).astype(float)
        depth = np.full((200, 1), 100.0)
        adj = np.ones((200, 1))
        prior.initialise(15, alt, depth, adj)
        np.testing.assert_allclose(prior._init_weights.sum(), 1.0, atol=1e-10)

    def test_initialise_shape(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prior = TSSBPrior(self._make_config(), k_max=8)
        rng = np.random.default_rng(0)
        alt = rng.binomial(100, 0.3, size=(200, 2)).astype(float)
        depth = np.full((200, 2), 100.0)
        adj = np.ones((200, 2))
        centers = prior.initialise(8, alt, depth, adj)
        assert centers.shape == (8, 2)
        assert (centers > 0).all()
        assert (centers < 1).all()


# ──────────────────────────────────────────────────────────────────────
# DirichletPrior end-to-end tests
# ──────────────────────────────────────────────────────────────────────


class TestDirichletPriorE2E:
    def test_mfvi_2clone_prunes_to_reasonable_k(
        self,
        simple_2clone: tuple[np.ndarray, np.ndarray, np.ndarray],
        dirichlet_mfvi_config: CloneConfig,
    ) -> None:
        """MFVI with K_max=10 on 2-clone data should prune to a small effective K."""
        alt, depth, adj = simple_2clone
        model = GenerativeModel(dirichlet_mfvi_config)
        result = model.fit(alt, depth, adj)
        assert isinstance(result, CloneResult)
        assert 1 <= result.K <= 5  # should find ~2-3 clusters

    def test_hybrid_2clone(
        self,
        simple_2clone: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Hybrid inference + Dirichlet K-prior on 2-clone data."""
        cfg = CloneConfig(
            emission=EmissionModel.BETA_BINOMIAL,
            inference=InferenceEngine.HYBRID,
            k_prior=KPrior.DIRICHLET,
            phylo=PhyloMode.NONE,
            noise=NoiseModel.NONE,
            phi=50.0,
        )
        alt, depth, adj = simple_2clone
        model = GenerativeModel(cfg)
        result = model.fit(alt, depth, adj)
        assert isinstance(result, CloneResult)
        assert 1 <= result.K <= 5

    def test_pyclone_vi_preset(
        self,
        simple_2clone: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """The pyclone_vi preset should now work end-to-end."""
        alt, depth, adj = simple_2clone
        model = GenerativeModel(CONFIGS["pyclone_vi"])
        result = model.fit(alt, depth, adj)
        assert isinstance(result, CloneResult)
        assert result.K >= 1
        assert np.isfinite(result.bic)


# ──────────────────────────────────────────────────────────────────────
# TSSBPrior end-to-end tests
# ──────────────────────────────────────────────────────────────────────


class TestTSSBPriorE2E:
    def test_mfvi_2clone_reasonable_k(
        self,
        simple_2clone: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """TSSB + MFVI on 2-clone data should yield a reasonable effective K."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg = CloneConfig(
                emission=EmissionModel.BETA_BINOMIAL,
                inference=InferenceEngine.MFVI,
                k_prior=KPrior.TSSB,
                phylo=PhyloMode.NONE,
                noise=NoiseModel.NONE,
                phi=50.0,
            )
            alt, depth, adj = simple_2clone
            model = GenerativeModel(cfg)
        result = model.fit(alt, depth, adj)
        assert isinstance(result, CloneResult)
        assert 1 <= result.K <= 10
