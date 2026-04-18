"""
Tests for phylogenetic modules: tree_utils, PostHocPhylo, ConstrainedPhylo,
LongitudinalPhylo, PairwisePhylo, JointMCMCPhylo.
"""
from __future__ import annotations

import warnings

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
from uniclone.core.types import CloneResult
from uniclone.phylo.tree_utils import (
    TreeResult,
    adjacency_to_parent_vector,
    build_nesting_order,
    enumerate_trees,
    is_included,
    is_valid_dag,
    topological_sort,
)

# =========================================================================
# TestTreeUtils
# =========================================================================


class TestTreeUtils:
    def test_topological_sort_chain(self) -> None:
        """0→1→2 should give [0, 1, 2]."""
        adj = np.array([
            [False, True, False],
            [False, False, True],
            [False, False, False],
        ])
        order = topological_sort(adj)
        assert order == [0, 1, 2]

    def test_topological_sort_fan(self) -> None:
        """0→1, 0→2 should start with 0."""
        adj = np.array([
            [False, True, True],
            [False, False, False],
            [False, False, False],
        ])
        order = topological_sort(adj)
        assert order[0] == 0
        assert set(order) == {0, 1, 2}

    def test_topological_sort_cycle_raises(self) -> None:
        adj = np.array([
            [False, True, False],
            [False, False, True],
            [True, False, False],
        ])
        with pytest.raises(ValueError, match="cycle"):
            topological_sort(adj)

    def test_is_valid_dag_true(self) -> None:
        adj = np.array([
            [False, True, False],
            [False, False, True],
            [False, False, False],
        ])
        assert is_valid_dag(adj) is True

    def test_is_valid_dag_false(self) -> None:
        adj = np.array([
            [False, True],
            [True, False],
        ])
        assert is_valid_dag(adj) is False

    def test_build_nesting_order(self) -> None:
        # Clone 0: mean=0.3, Clone 1: mean=0.7, Clone 2: mean=0.5
        centers = np.array([[0.3, 0.3], [0.7, 0.7], [0.5, 0.5]])
        order = build_nesting_order(centers)
        assert list(order) == [1, 2, 0]

    def test_is_included_basic(self) -> None:
        centers = np.array([[0.2, 0.1], [0.5, 0.4]])
        mat = is_included(centers)
        # Clone 0 is nested in clone 1 (0.2<=0.5, 0.1<=0.4)
        assert mat[0, 1] is np.True_
        # Clone 1 is NOT nested in clone 0
        assert mat[1, 0] is np.False_
        # Diagonal is False
        assert mat[0, 0] is np.False_

    def test_is_included_tolerance(self) -> None:
        centers = np.array([[0.5001], [0.5]])
        mat = is_included(centers, tol=1e-6)
        # 0.5001 <= 0.5 + 1e-6 is False
        assert mat[0, 1] is np.False_

    def test_enumerate_trees_k2(self) -> None:
        trees = enumerate_trees(2)
        assert len(trees) == 1
        assert trees[0].shape == (2, 2)
        assert trees[0].sum() == 1  # one edge

    def test_enumerate_trees_k3(self) -> None:
        trees = enumerate_trees(3)
        # 3^1 = 3 labelled trees on 3 nodes
        assert len(trees) == 3
        for t in trees:
            assert t.shape == (3, 3)
            assert is_valid_dag(t)

    def test_enumerate_trees_k4(self) -> None:
        trees = enumerate_trees(4)
        # 4^2 = 16 labelled trees on 4 nodes
        assert len(trees) == 16

    def test_enumerate_trees_large_k_warns(self) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            trees = enumerate_trees(8, rng=np.random.default_rng(0))
            assert any("sampling" in str(x.message).lower() for x in w)
            # Should have some trees (may be less than 10000 due to dedup)
            assert len(trees) > 0

    def test_adjacency_to_parent_vector_chain(self) -> None:
        adj = np.array([
            [False, True, False],
            [False, False, True],
            [False, False, False],
        ])
        parent = adjacency_to_parent_vector(adj)
        assert parent[0] == -1
        assert parent[1] == 0
        assert parent[2] == 1

    def test_adjacency_to_parent_vector_fan(self) -> None:
        adj = np.array([
            [False, True, True],
            [False, False, False],
            [False, False, False],
        ])
        parent = adjacency_to_parent_vector(adj)
        assert parent[0] == -1
        assert parent[1] == 0
        assert parent[2] == 0

    def test_tree_result_dataclass(self) -> None:
        adj = np.zeros((2, 2), dtype=bool)
        parent = np.array([-1, 0])
        nest = np.zeros((2, 2), dtype=bool)
        tr = TreeResult(adjacency=adj, parent=parent, is_included=nest)
        assert tr.adjacency.shape == (2, 2)
        assert tr.parent[0] == -1


# =========================================================================
# TestPostHocPhylo
# =========================================================================


class TestPostHocPhylo:
    def test_constrain_is_noop(self) -> None:
        from uniclone.phylo.post_hoc import PostHocPhylo

        config = CONFIGS["quantumclone_v1"]
        phylo = PostHocPhylo(config)
        centers = np.array([[0.5, 0.3], [0.2, 0.1]])
        raw = centers.copy()
        result = phylo.constrain(centers, raw)
        np.testing.assert_array_equal(result, centers)

    def test_postprocess_attaches_tree_result(self) -> None:
        from uniclone.phylo.post_hoc import PostHocPhylo

        config = CONFIGS["quantumclone_v1"]
        phylo = PostHocPhylo(config)
        centers = np.array([[0.5, 0.4], [0.2, 0.1]])
        result = CloneResult(
            centers=centers,
            assignments=np.array([0, 1]),
            responsibilities=np.array([[0.9, 0.1], [0.1, 0.9]]),
            log_likelihood=-100.0,
            bic=200.0,
            K=2,
            n_iter=10,
            converged=True,
        )
        result = phylo.postprocess(result)
        assert isinstance(result.tree, TreeResult)
        assert result.tree.adjacency.shape == (2, 2)
        assert result.tree.parent.shape == (2,)
        assert result.tree.is_included.shape == (2, 2)

    def test_postprocess_nesting_correct(self) -> None:
        from uniclone.phylo.post_hoc import PostHocPhylo

        config = CONFIGS["quantumclone_v1"]
        phylo = PostHocPhylo(config)
        # Clone 0 has higher cellularity → root; clone 1 nested in 0
        centers = np.array([[0.6, 0.5], [0.3, 0.2]])
        result = CloneResult(
            centers=centers,
            assignments=np.array([0, 1]),
            responsibilities=np.array([[0.9, 0.1], [0.1, 0.9]]),
            log_likelihood=-100.0,
            bic=200.0,
            K=2,
            n_iter=10,
            converged=True,
        )
        result = phylo.postprocess(result)
        # Clone 1 nested in clone 0
        assert result.tree.is_included[1, 0]
        # Parent of clone 1 is clone 0
        assert result.tree.parent[1] == 0
        assert result.tree.parent[0] == -1

    def test_e2e_quantumclone_v1(
        self,
        simple_2clone: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        from uniclone import GenerativeModel

        alt, depth, adj = simple_2clone
        model = GenerativeModel(CONFIGS["quantumclone_v1"])
        result = model.fit(alt, depth, adj)
        assert isinstance(result.tree, TreeResult)
        K = result.K
        assert result.tree.adjacency.shape == (K, K)
        assert result.tree.parent.shape == (K,)
        # Root has parent -1
        assert -1 in result.tree.parent


# =========================================================================
# TestConstrainedPhylo
# =========================================================================


class TestConstrainedPhylo:
    def test_constrain_enforces_nesting(self) -> None:
        from uniclone.phylo.constrained import ConstrainedPhylo

        config = CloneConfig(
            emission=EmissionModel.BETA_BINOMIAL,
            inference=InferenceEngine.HYBRID,
            k_prior=KPrior.DIRICHLET,
            phylo=PhyloMode.CONSTRAINED,
            noise=NoiseModel.NONE,
        )
        phylo = ConstrainedPhylo(config)
        # Clone 0: high, Clone 1: should be clamped below clone 0
        centers = np.array([[0.6, 0.5], [0.7, 0.8]])
        result = phylo.constrain(centers, centers)
        # After projection, clone 1 (originally higher) becomes root,
        # clone 0 gets clamped below it
        # The key invariant: sorted by mean, each child <= parent
        order = build_nesting_order(result)
        for idx in range(1, len(order)):
            child = order[idx]
            parent = order[idx - 1]
            assert np.all(result[child] <= result[parent])

    def test_constrain_preserves_valid_nesting(self) -> None:
        from uniclone.phylo.constrained import ConstrainedPhylo

        config = CloneConfig(
            emission=EmissionModel.BETA_BINOMIAL,
            inference=InferenceEngine.HYBRID,
            k_prior=KPrior.DIRICHLET,
            phylo=PhyloMode.CONSTRAINED,
            noise=NoiseModel.NONE,
        )
        phylo = ConstrainedPhylo(config)
        # Already valid nesting
        centers = np.array([[0.6, 0.5], [0.3, 0.2]])
        result = phylo.constrain(centers, centers)
        # Should be essentially unchanged (within tolerance)
        np.testing.assert_allclose(result, centers, atol=1e-5)

    def test_postprocess_builds_tree(self) -> None:
        from uniclone.phylo.constrained import ConstrainedPhylo

        config = CloneConfig(
            emission=EmissionModel.BETA_BINOMIAL,
            inference=InferenceEngine.HYBRID,
            k_prior=KPrior.DIRICHLET,
            phylo=PhyloMode.CONSTRAINED,
            noise=NoiseModel.NONE,
        )
        phylo = ConstrainedPhylo(config)
        centers = np.array([[0.6, 0.5], [0.3, 0.2]])
        cr = CloneResult(
            centers=centers,
            assignments=np.array([0, 1]),
            responsibilities=np.array([[0.9, 0.1], [0.1, 0.9]]),
            log_likelihood=-100.0,
            bic=200.0,
            K=2,
            n_iter=10,
            converged=True,
        )
        cr = phylo.postprocess(cr)
        assert isinstance(cr.tree, TreeResult)
        assert cr.tree.parent[0] == -1 or cr.tree.parent[1] == -1

    def test_constrain_single_clone(self) -> None:
        from uniclone.phylo.constrained import ConstrainedPhylo

        config = CloneConfig(
            emission=EmissionModel.BETA_BINOMIAL,
            inference=InferenceEngine.HYBRID,
            k_prior=KPrior.DIRICHLET,
            phylo=PhyloMode.CONSTRAINED,
            noise=NoiseModel.NONE,
        )
        phylo = ConstrainedPhylo(config)
        centers = np.array([[0.5, 0.4]])
        result = phylo.constrain(centers, centers)
        assert result.shape == (1, 2)


# =========================================================================
# TestLongitudinalPhylo
# =========================================================================


class TestLongitudinalPhylo:
    def test_constrain_is_noop(self) -> None:
        from uniclone.phylo.longitudinal import LongitudinalPhylo

        config = CloneConfig(
            emission=EmissionModel.BETA_BINOMIAL,
            inference=InferenceEngine.EM,
            k_prior=KPrior.FIXED,
            phylo=PhyloMode.LONGITUDINAL,
            noise=NoiseModel.NONE,
            longitudinal=True,
            n_clones=3,
        )
        phylo = LongitudinalPhylo(config)
        centers = np.array([[0.5, 0.4], [0.2, 0.1]])
        raw = centers.copy()
        result = phylo.constrain(centers, raw)
        np.testing.assert_array_equal(result, centers)

    def test_ilp_finds_valid_tree(self) -> None:
        from uniclone.phylo.longitudinal import LongitudinalPhylo

        config = CloneConfig(
            emission=EmissionModel.BETA_BINOMIAL,
            inference=InferenceEngine.EM,
            k_prior=KPrior.FIXED,
            phylo=PhyloMode.LONGITUDINAL,
            noise=NoiseModel.NONE,
            longitudinal=True,
            n_clones=3,
        )
        phylo = LongitudinalPhylo(config)
        # 3 clones, 3 timepoints with clear nesting
        centers = np.array([
            [0.7, 0.6, 0.5],
            [0.4, 0.3, 0.2],
            [0.2, 0.1, 0.05],
        ])
        cr = CloneResult(
            centers=centers,
            assignments=np.array([0, 1, 2]),
            responsibilities=np.eye(3),
            log_likelihood=-100.0,
            bic=200.0,
            K=3,
            n_iter=10,
            converged=True,
        )
        cr = phylo.postprocess(cr)
        assert isinstance(cr.tree, TreeResult)
        assert cr.tree.adjacency.shape == (3, 3)
        # Valid DAG
        assert is_valid_dag(cr.tree.adjacency)
        # Root has parent -1
        assert -1 in cr.tree.parent

    def test_single_clone(self) -> None:
        from uniclone.phylo.longitudinal import LongitudinalPhylo

        config = CloneConfig(
            emission=EmissionModel.BETA_BINOMIAL,
            inference=InferenceEngine.EM,
            k_prior=KPrior.FIXED,
            phylo=PhyloMode.LONGITUDINAL,
            noise=NoiseModel.NONE,
            longitudinal=True,
            n_clones=1,
        )
        phylo = LongitudinalPhylo(config)
        centers = np.array([[0.5, 0.4, 0.3]])
        cr = CloneResult(
            centers=centers,
            assignments=np.array([0]),
            responsibilities=np.array([[1.0]]),
            log_likelihood=-50.0,
            bic=100.0,
            K=1,
            n_iter=5,
            converged=True,
        )
        cr = phylo.postprocess(cr)
        assert isinstance(cr.tree, TreeResult)
        assert cr.tree.parent[0] == -1

    def test_fallback_on_infeasible(self) -> None:
        from uniclone.phylo.longitudinal import LongitudinalPhylo

        config = CloneConfig(
            emission=EmissionModel.BETA_BINOMIAL,
            inference=InferenceEngine.EM,
            k_prior=KPrior.FIXED,
            phylo=PhyloMode.LONGITUDINAL,
            noise=NoiseModel.NONE,
            longitudinal=True,
            n_clones=2,
        )
        phylo = LongitudinalPhylo(config)
        # Clones with crossing cellularities (hard to form valid tree with sum rule)
        centers = np.array([
            [0.8, 0.2],
            [0.2, 0.8],
        ])
        cr = CloneResult(
            centers=centers,
            assignments=np.array([0, 1]),
            responsibilities=np.eye(2),
            log_likelihood=-100.0,
            bic=200.0,
            K=2,
            n_iter=10,
            converged=True,
        )
        # Should not raise — falls back to greedy if ILP fails
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            cr = phylo.postprocess(cr)
        assert isinstance(cr.tree, TreeResult)


# =========================================================================
# TestPairwisePhylo
# =========================================================================


class TestPairwisePhylo:
    def test_constrain_is_noop(self) -> None:
        from uniclone.phylo.pairwise import PairwisePhylo

        config = CloneConfig(
            emission=EmissionModel.DCF_BB,
            inference=InferenceEngine.HYBRID,
            k_prior=KPrior.DIRICHLET,
            phylo=PhyloMode.PAIRWISE,
            noise=NoiseModel.NONE,
        )
        phylo = PairwisePhylo(config)
        centers = np.array([[0.5, 0.4], [0.2, 0.1]])
        result = phylo.constrain(centers, centers)
        np.testing.assert_array_equal(result, centers)

    def test_pairs_tensor_shape(self) -> None:
        from uniclone.phylo.pairwise import PairwisePhylo

        centers = np.array([[0.6, 0.5], [0.3, 0.2], [0.1, 0.05]])
        pairs = PairwisePhylo._build_pairs_tensor(centers, K=3)
        assert pairs.shape == (3, 3, 4)

    def test_pairs_tensor_rows_sum_to_one(self) -> None:
        from uniclone.phylo.pairwise import PairwisePhylo

        centers = np.array([[0.6, 0.5], [0.3, 0.2]])
        pairs = PairwisePhylo._build_pairs_tensor(centers, K=2)
        for i in range(2):
            for j in range(2):
                np.testing.assert_allclose(pairs[i, j].sum(), 1.0, atol=1e-6)

    def test_pairs_tensor_diagonal_is_same_clone(self) -> None:
        from uniclone.phylo.pairwise import PairwisePhylo

        centers = np.array([[0.6, 0.5], [0.3, 0.2]])
        pairs = PairwisePhylo._build_pairs_tensor(centers, K=2)
        # Diagonal should have same_clone = 1.0
        for i in range(2):
            assert pairs[i, i, 3] == 1.0  # _SAME_CLONE = 3

    def test_ancestor_scoring_correct(self) -> None:
        from uniclone.phylo.pairwise import PairwisePhylo

        # Clone 0 clearly ancestor of clone 1 (much higher cellularity)
        centers = np.array([[0.8, 0.7], [0.1, 0.05]])
        pairs = PairwisePhylo._build_pairs_tensor(centers, K=2)
        # pairs[0,1,ANCESTOR] should be high
        assert pairs[0, 1, 0] > 0.5  # _ANCESTOR = 0

    def test_postprocess_builds_tree(self) -> None:
        from uniclone.phylo.pairwise import PairwisePhylo

        config = CloneConfig(
            emission=EmissionModel.DCF_BB,
            inference=InferenceEngine.HYBRID,
            k_prior=KPrior.DIRICHLET,
            phylo=PhyloMode.PAIRWISE,
            noise=NoiseModel.NONE,
        )
        phylo = PairwisePhylo(config)
        centers = np.array([[0.6, 0.5], [0.3, 0.2], [0.1, 0.05]])
        cr = CloneResult(
            centers=centers,
            assignments=np.array([0, 1, 2]),
            responsibilities=np.eye(3),
            log_likelihood=-100.0,
            bic=200.0,
            K=3,
            n_iter=10,
            converged=True,
        )
        cr = phylo.postprocess(cr)
        assert isinstance(cr.tree, TreeResult)
        assert cr.tree.adjacency.shape == (3, 3)
        assert hasattr(cr.tree, "pairs")
        assert cr.tree.pairs.shape == (3, 3, 4)

    def test_single_clone(self) -> None:
        from uniclone.phylo.pairwise import PairwisePhylo

        config = CloneConfig(
            emission=EmissionModel.DCF_BB,
            inference=InferenceEngine.HYBRID,
            k_prior=KPrior.DIRICHLET,
            phylo=PhyloMode.PAIRWISE,
            noise=NoiseModel.NONE,
        )
        phylo = PairwisePhylo(config)
        centers = np.array([[0.5, 0.4]])
        cr = CloneResult(
            centers=centers,
            assignments=np.array([0]),
            responsibilities=np.array([[1.0]]),
            log_likelihood=-50.0,
            bic=100.0,
            K=1,
            n_iter=5,
            converged=True,
        )
        cr = phylo.postprocess(cr)
        assert isinstance(cr.tree, TreeResult)


# =========================================================================
# TestJointMCMCPhylo
# =========================================================================


class TestJointMCMCPhylo:
    def test_init_warns_experimental(self) -> None:
        from uniclone.phylo.joint_mcmc import JointMCMCPhylo

        config = CloneConfig(
            emission=EmissionModel.BETA_BINOMIAL,
            inference=InferenceEngine.HYBRID,
            k_prior=KPrior.DIRICHLET,
            phylo=PhyloMode.JOINT_MCMC,
            noise=NoiseModel.NONE,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            JointMCMCPhylo(config)
            assert any("experimental" in str(x.message).lower() for x in w)

    def test_constrain_applies_projection(self) -> None:
        from uniclone.phylo.joint_mcmc import JointMCMCPhylo

        config = CloneConfig(
            emission=EmissionModel.BETA_BINOMIAL,
            inference=InferenceEngine.HYBRID,
            k_prior=KPrior.DIRICHLET,
            phylo=PhyloMode.JOINT_MCMC,
            noise=NoiseModel.NONE,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            phylo = JointMCMCPhylo(config)

        # Violating nesting: clone 1 higher than clone 0
        centers = np.array([[0.3, 0.2], [0.7, 0.6]])
        result = phylo.constrain(centers, centers)
        order = build_nesting_order(result)
        for idx in range(1, len(order)):
            child = order[idx]
            parent = order[idx - 1]
            assert np.all(result[child] <= result[parent])

    def test_postprocess_builds_tree(self) -> None:
        from uniclone.phylo.joint_mcmc import JointMCMCPhylo

        config = CloneConfig(
            emission=EmissionModel.BETA_BINOMIAL,
            inference=InferenceEngine.HYBRID,
            k_prior=KPrior.DIRICHLET,
            phylo=PhyloMode.JOINT_MCMC,
            noise=NoiseModel.NONE,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            phylo = JointMCMCPhylo(config)

        centers = np.array([[0.6, 0.5], [0.3, 0.2]])
        cr = CloneResult(
            centers=centers,
            assignments=np.array([0, 1]),
            responsibilities=np.array([[0.9, 0.1], [0.1, 0.9]]),
            log_likelihood=-100.0,
            bic=200.0,
            K=2,
            n_iter=10,
            converged=True,
        )
        cr = phylo.postprocess(cr)
        assert isinstance(cr.tree, TreeResult)
        assert cr.tree.parent.shape == (2,)
