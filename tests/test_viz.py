"""Tests for uniclone.viz (Phase 7)."""

from __future__ import annotations

import numpy as np
import pytest

go = pytest.importorskip("plotly.graph_objects")

from uniclone.core.types import CloneResult
from uniclone.phylo.tree_utils import TreeResult


@pytest.fixture
def mock_result() -> CloneResult:
    """K=3, 100 mutations, 2 samples, with TreeResult."""
    rng = np.random.default_rng(42)
    K, n_mut, n_samples = 3, 100, 2

    centers = np.array([[0.6, 0.5], [0.35, 0.3], [0.15, 0.1]])
    assignments = np.repeat(np.arange(K), [40, 35, 25])
    resp = np.zeros((n_mut, K))
    for i, k in enumerate(assignments):
        resp[i, k] = 0.9
        for j in range(K):
            if j != k:
                resp[i, j] = 0.05

    # Tree: 0 -> 1, 0 -> 2
    adj = np.zeros((K, K), dtype=bool)
    adj[0, 1] = True
    adj[0, 2] = True
    parent = np.array([-1, 0, 0])
    incl = np.zeros((K, K), dtype=bool)
    incl[1, 0] = True
    incl[2, 0] = True
    tree = TreeResult(adjacency=adj, parent=parent, is_included=incl)

    return CloneResult(
        centers=centers,
        assignments=assignments,
        responsibilities=resp,
        log_likelihood=-500.0,
        bic=1050.0,
        K=K,
        n_iter=50,
        converged=True,
        tree=tree,
    )


@pytest.fixture
def mock_alt_depth(mock_result: CloneResult) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic alt/depth arrays matching mock_result."""
    rng = np.random.default_rng(42)
    n_mut, n_samples = 100, 2
    depth = np.full((n_mut, n_samples), 100, dtype=float)
    centers = np.asarray(mock_result.centers)
    alt = np.zeros((n_mut, n_samples), dtype=float)
    for i in range(n_mut):
        k = int(mock_result.assignments[i])
        for s in range(n_samples):
            alt[i, s] = rng.binomial(100, centers[k, s])
    return alt, depth


@pytest.fixture
def mock_result_no_tree(mock_result: CloneResult) -> CloneResult:
    """Same as mock_result but without a tree."""
    return CloneResult(
        centers=mock_result.centers,
        assignments=mock_result.assignments,
        responsibilities=mock_result.responsibilities,
        log_likelihood=mock_result.log_likelihood,
        bic=mock_result.bic,
        K=mock_result.K,
        n_iter=mock_result.n_iter,
        converged=mock_result.converged,
        tree=None,
    )


@pytest.fixture
def mock_result_k1() -> CloneResult:
    """K=1, 50 mutations, 1 sample, no tree."""
    n_mut = 50
    return CloneResult(
        centers=np.array([[0.5]]),
        assignments=np.zeros(n_mut, dtype=int),
        responsibilities=np.ones((n_mut, 1)),
        log_likelihood=-200.0,
        bic=420.0,
        K=1,
        n_iter=10,
        converged=True,
        tree=None,
    )


# ---------------------------------------------------------------------------
# Cellularity tests
# ---------------------------------------------------------------------------


class TestVafHistogram:
    def test_returns_figure(
        self, mock_alt_depth: tuple[np.ndarray, np.ndarray], mock_result: CloneResult
    ) -> None:
        from uniclone.viz.cellularity import vaf_histogram

        alt, depth = mock_alt_depth
        fig = vaf_histogram(alt, depth, mock_result)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == mock_result.K  # one histogram per clone

    def test_k1(self, mock_result_k1: CloneResult) -> None:
        from uniclone.viz.cellularity import vaf_histogram

        alt = np.random.default_rng(0).binomial(100, 0.5, (50, 1)).astype(float)
        depth = np.full((50, 1), 100, dtype=float)
        fig = vaf_histogram(alt, depth, mock_result_k1)
        assert len(fig.data) == 1


class TestCellularityScatter:
    def test_returns_figure(
        self, mock_alt_depth: tuple[np.ndarray, np.ndarray], mock_result: CloneResult
    ) -> None:
        from uniclone.viz.cellularity import cellularity_scatter

        alt, depth = mock_alt_depth
        fig = cellularity_scatter(alt, depth, mock_result)
        assert isinstance(fig, go.Figure)
        # One scatter per clone
        assert len(fig.data) == mock_result.K


class TestMultiSampleComparison:
    def test_returns_figure(self, mock_result: CloneResult) -> None:
        from uniclone.viz.cellularity import multi_sample_comparison

        fig = multi_sample_comparison(mock_result)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == mock_result.K

    def test_custom_labels(self, mock_result: CloneResult) -> None:
        from uniclone.viz.cellularity import multi_sample_comparison

        fig = multi_sample_comparison(mock_result, sample_labels=["Primary", "Met"])
        assert fig.data[0].x is not None


class TestResponsibilityHeatmap:
    def test_returns_figure(self, mock_result: CloneResult) -> None:
        from uniclone.viz.cellularity import responsibility_heatmap

        fig = responsibility_heatmap(mock_result)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1  # single heatmap trace
        assert isinstance(fig.data[0], go.Heatmap)


# ---------------------------------------------------------------------------
# Evolution tests
# ---------------------------------------------------------------------------


class TestFishPlot:
    def test_returns_figure_with_tree(self, mock_result: CloneResult) -> None:
        from uniclone.viz.evolution import fish_plot

        fig = fish_plot(mock_result)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == mock_result.K

    def test_returns_figure_no_tree(self, mock_result_no_tree: CloneResult) -> None:
        from uniclone.viz.evolution import fish_plot

        fig = fish_plot(mock_result_no_tree)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == mock_result_no_tree.K


class TestCloneProportionBar:
    def test_returns_figure(self, mock_result: CloneResult) -> None:
        from uniclone.viz.evolution import clone_proportion_bar

        fig = clone_proportion_bar(mock_result)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == mock_result.K


# ---------------------------------------------------------------------------
# Phylo tree tests
# ---------------------------------------------------------------------------


class TestCloneTree:
    def test_returns_figure(self, mock_result: CloneResult) -> None:
        from uniclone.viz.phylo_tree import clone_tree

        fig = clone_tree(mock_result)
        assert isinstance(fig, go.Figure)
        # Edge trace + node trace
        assert len(fig.data) == 2

    def test_no_tree_raises(self, mock_result_no_tree: CloneResult) -> None:
        from uniclone.viz.phylo_tree import clone_tree

        with pytest.raises(ValueError, match="No tree"):
            clone_tree(mock_result_no_tree)

    def test_edge_labels(self, mock_result: CloneResult) -> None:
        from uniclone.viz.phylo_tree import clone_tree

        fig = clone_tree(mock_result, show_edge_labels=True)
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# Diagnostics tests
# ---------------------------------------------------------------------------


class TestBicCurve:
    def test_returns_figure(self, mock_result: CloneResult) -> None:
        from uniclone.viz.diagnostics import bic_curve

        results = []
        for k in range(1, 6):
            results.append(
                CloneResult(
                    centers=np.random.default_rng(k).random((k, 2)),
                    assignments=np.zeros(100, dtype=int),
                    responsibilities=np.ones((100, k)) / k,
                    log_likelihood=-500.0 + k * 10,
                    bic=1000.0 - k * 50 + k**2 * 20,
                    K=k,
                    n_iter=50,
                    converged=True,
                )
            )
        fig = bic_curve(results)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # line + star marker


class TestConvergenceTrace:
    def test_returns_figure(self) -> None:
        from uniclone.viz.diagnostics import convergence_trace

        lls = [-1000.0, -800.0, -600.0, -550.0, -520.0, -510.0]
        fig = convergence_trace(lls)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1


class TestFeatureAttributionBar:
    def test_returns_figure(self) -> None:
        from uniclone.viz.diagnostics import feature_attribution_bar

        attrs = {"feat_a": 0.5, "feat_b": -0.3, "feat_c": 0.1}
        fig = feature_attribution_bar(attrs, top_n=3)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Bar)


class TestResponsibilityDistribution:
    def test_returns_figure(self, mock_result: CloneResult) -> None:
        from uniclone.viz.diagnostics import responsibility_distribution

        fig = responsibility_distribution(mock_result)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == mock_result.K

    def test_k1(self, mock_result_k1: CloneResult) -> None:
        from uniclone.viz.diagnostics import responsibility_distribution

        fig = responsibility_distribution(mock_result_k1)
        assert len(fig.data) == 1


# ---------------------------------------------------------------------------
# __init__ re-exports
# ---------------------------------------------------------------------------


class TestImports:
    def test_all_public_functions_importable(self) -> None:
        from uniclone.viz import (
            bic_curve,
            cellularity_scatter,
            clone_proportion_bar,
            clone_tree,
            convergence_trace,
            feature_attribution_bar,
            fish_plot,
            multi_sample_comparison,
            responsibility_distribution,
            responsibility_heatmap,
            vaf_histogram,
        )

        assert callable(vaf_histogram)
        assert callable(cellularity_scatter)
        assert callable(multi_sample_comparison)
        assert callable(responsibility_heatmap)
        assert callable(fish_plot)
        assert callable(clone_proportion_bar)
        assert callable(clone_tree)
        assert callable(bic_curve)
        assert callable(convergence_trace)
        assert callable(feature_attribution_bar)
        assert callable(responsibility_distribution)
