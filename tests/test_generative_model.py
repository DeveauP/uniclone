"""
Tests for GenerativeModel end-to-end pipeline.

Covers: quantumclone_v1 runs successfully, stub configs raise NotImplementedError,
CloneResult has all required fields, multi-sample inputs work.
"""
from __future__ import annotations

import numpy as np
import pytest

from uniclone import CONFIGS, CloneResult, GenerativeModel


class TestGenerativeModelQuantumcloneV1:
    def test_fit_returns_clone_result(
        self,
        simple_2clone: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        alt, depth, adj = simple_2clone
        model = GenerativeModel(CONFIGS["quantumclone_v1"])
        result = model.fit(alt, depth, adj)
        assert isinstance(result, CloneResult)

    def test_fit_result_k_is_positive(
        self,
        simple_2clone: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        alt, depth, adj = simple_2clone
        model = GenerativeModel(CONFIGS["quantumclone_v1"])
        result = model.fit(alt, depth, adj)
        assert result.K >= 1

    def test_fit_result_centers_shape(
        self,
        simple_1clone: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        alt, depth, adj = simple_1clone
        model = GenerativeModel(CONFIGS["quantumclone_v1"])
        result = model.fit(alt, depth, adj)
        assert result.centers.ndim == 2
        assert result.centers.shape[1] == 1  # 1 sample

    def test_fit_result_assignments_shape(
        self,
        simple_2clone: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        alt, depth, adj = simple_2clone
        n_mut = alt.shape[0]
        model = GenerativeModel(CONFIGS["quantumclone_v1"])
        result = model.fit(alt, depth, adj)
        assert result.assignments.shape == (n_mut,)

    def test_fit_result_log_likelihood_finite(
        self,
        simple_1clone: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        alt, depth, adj = simple_1clone
        model = GenerativeModel(CONFIGS["quantumclone_v1"])
        result = model.fit(alt, depth, adj)
        assert np.isfinite(result.log_likelihood)

    def test_fit_result_bic_finite(
        self,
        simple_1clone: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        alt, depth, adj = simple_1clone
        model = GenerativeModel(CONFIGS["quantumclone_v1"])
        result = model.fit(alt, depth, adj)
        assert np.isfinite(result.bic)

    def test_fit_result_config_attached(
        self,
        simple_1clone: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        alt, depth, adj = simple_1clone
        model = GenerativeModel(CONFIGS["quantumclone_v1"])
        result = model.fit(alt, depth, adj)
        assert result.config is CONFIGS["quantumclone_v1"]

    def test_fit_result_tree_attached_post_hoc(
        self,
        simple_2clone: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """POST_HOC phylo should attach TreeResult to result.tree."""
        alt, depth, adj = simple_2clone
        model = GenerativeModel(CONFIGS["quantumclone_v1"])
        result = model.fit(alt, depth, adj)
        assert result.tree is not None
        K = result.K
        from uniclone.phylo.tree_utils import TreeResult
        assert isinstance(result.tree, TreeResult)
        assert result.tree.is_included.shape == (K, K)
        assert result.tree.adjacency.shape == (K, K)
        assert result.tree.parent.shape == (K,)

    def test_fit_without_adj_factor(
        self,
        simple_1clone: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """adj_factor defaults to ones when not provided."""
        alt, depth, _ = simple_1clone
        model = GenerativeModel(CONFIGS["quantumclone_v1"])
        result = model.fit(alt, depth)
        assert isinstance(result, CloneResult)

    def test_fit_1d_input_accepted(self) -> None:
        """1-D inputs (no sample axis) should be auto-expanded."""
        rng = np.random.default_rng(7)
        alt = rng.binomial(100, 0.4, size=100).astype(float)
        depth = np.full(100, 100.0)
        model = GenerativeModel(CONFIGS["quantumclone_v1"])
        result = model.fit(alt, depth)
        assert isinstance(result, CloneResult)

    def test_fit_multisample(
        self,
        multisample_2clone: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        alt, depth, adj = multisample_2clone
        model = GenerativeModel(CONFIGS["quantumclone_v1"])
        result = model.fit(alt, depth, adj)
        assert result.centers.shape[1] == 2  # 2 samples


class TestGenerativeModelPycloneVI:
    def test_pyclone_vi_fit_returns_clone_result(
        self,
        simple_2clone: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        alt, depth, adj = simple_2clone
        model = GenerativeModel(CONFIGS["pyclone_vi"])
        result = model.fit(alt, depth, adj)
        assert isinstance(result, CloneResult)
        assert result.K >= 1
        assert np.isfinite(result.bic)
        assert result.config is CONFIGS["pyclone_vi"]


class TestGenerativeModelStubsRaise:
    def test_phyloclone_style_raises_without_pymc(self) -> None:
        """phyloclone_style requires PyMC (MCMC) which may not be installed."""
        try:
            import pymc  # noqa: F401
            pytest.skip("pymc is installed — ImportError cannot be triggered")
        except ImportError:
            pass
        with pytest.raises(ImportError):
            GenerativeModel(CONFIGS["phyloclone_style"])

    def test_repr_is_informative(self) -> None:
        model = GenerativeModel(CONFIGS["quantumclone_v1"])
        r = repr(model)
        assert "BINOMIAL" in r
        assert "EM" in r


class TestGenerativeModelResultFields:
    def test_all_required_fields_present(
        self,
        simple_1clone: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        alt, depth, adj = simple_1clone
        result = GenerativeModel(CONFIGS["quantumclone_v1"]).fit(alt, depth, adj)
        required = ["centers", "assignments", "responsibilities",
                    "log_likelihood", "bic", "K", "n_iter", "converged"]
        for field in required:
            assert hasattr(result, field), f"CloneResult missing field: {field}"

    def test_centers_in_valid_range(
        self,
        simple_2clone: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        alt, depth, adj = simple_2clone
        result = GenerativeModel(CONFIGS["quantumclone_v1"]).fit(alt, depth, adj)
        assert (result.centers > 0).all()
        assert (result.centers < 1).all()

    def test_assignments_in_valid_range(
        self,
        simple_2clone: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        alt, depth, adj = simple_2clone
        result = GenerativeModel(CONFIGS["quantumclone_v1"]).fit(alt, depth, adj)
        assert (result.assignments >= 0).all()
        assert (result.assignments < result.K).all()


class TestGenerativeModelWgsCohortPhylo:
    def test_wgs_cohort_phylo_fit_returns_clone_result(
        self,
        simple_2clone: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        alt, depth, adj = simple_2clone
        model = GenerativeModel(CONFIGS["wgs_cohort_phylo"])
        result = model.fit(alt, depth, adj)
        assert isinstance(result, CloneResult)
        assert result.K >= 1

    def test_wgs_cohort_phylo_has_tree(
        self,
        simple_2clone: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        alt, depth, adj = simple_2clone
        model = GenerativeModel(CONFIGS["wgs_cohort_phylo"])
        result = model.fit(alt, depth, adj)
        assert result.tree is not None
