"""uniclone.viz.evolution — Clonal evolution visualisation (Phase 7)."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from uniclone.viz._style import _require_plotly, clone_colors, default_layout

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from uniclone.core.types import CloneResult


def fish_plot(
    result: CloneResult,
    *,
    sample_labels: list[str] | None = None,
    title: str | None = None,
) -> go.Figure:
    """Stacked area (Muller) plot of clone proportions across samples."""
    _require_plotly()
    import plotly.graph_objects as go

    centers = np.asarray(result.centers)  # (K, n_samples)
    n_clones, n_samples = centers.shape
    labels = sample_labels or [f"Sample {i}" for i in range(n_samples)]
    colors = clone_colors(n_clones)

    # Determine ordering: topological if tree available, else by mean cellularity
    if result.tree is not None:
        from uniclone.phylo.tree_utils import topological_sort

        order = topological_sort(result.tree.adjacency)
    else:
        order = list(np.argsort(-centers.mean(axis=1)))

    fig = go.Figure()
    for k in order:
        fig.add_trace(go.Scatter(
            x=labels,
            y=centers[k],
            mode="lines",
            name=f"Clone {k}",
            line=dict(width=0.5, color=colors[k]),
            stackgroup="one",
            fillcolor=colors[k],
        ))
    fig.update_layout(xaxis_title="Sample", yaxis_title="Cellularity")
    return default_layout(fig, title or "Fish Plot")


def clone_proportion_bar(
    result: CloneResult,
    *,
    sample_labels: list[str] | None = None,
    title: str | None = None,
) -> go.Figure:
    """Stacked bar of clone proportions per sample."""
    _require_plotly()
    import plotly.graph_objects as go

    centers = np.asarray(result.centers)  # (K, n_samples)
    n_clones, n_samples = centers.shape
    labels = sample_labels or [f"Sample {i}" for i in range(n_samples)]
    colors = clone_colors(n_clones)

    fig = go.Figure()
    for k in range(n_clones):
        fig.add_trace(go.Bar(
            x=labels,
            y=centers[k],
            name=f"Clone {k}",
            marker_color=colors[k],
        ))
    fig.update_layout(barmode="stack", xaxis_title="Sample", yaxis_title="Cellularity")
    return default_layout(fig, title or "Clone Proportions")
