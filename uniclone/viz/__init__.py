"""uniclone.viz — Visualisation suite (Phase 7)."""

from __future__ import annotations

from uniclone.viz.cellularity import (
    cellularity_scatter,
    multi_sample_comparison,
    responsibility_heatmap,
    vaf_histogram,
)
from uniclone.viz.diagnostics import (
    bic_curve,
    convergence_trace,
    feature_attribution_bar,
    responsibility_distribution,
)
from uniclone.viz.evolution import clone_proportion_bar, fish_plot
from uniclone.viz.phylo_tree import clone_tree

__all__ = [
    # cellularity
    "vaf_histogram",
    "cellularity_scatter",
    "multi_sample_comparison",
    "responsibility_heatmap",
    # evolution
    "fish_plot",
    "clone_proportion_bar",
    # phylo_tree
    "clone_tree",
    # diagnostics
    "bic_curve",
    "convergence_trace",
    "feature_attribution_bar",
    "responsibility_distribution",
]
