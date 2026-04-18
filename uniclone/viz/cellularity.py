"""uniclone.viz.cellularity — Clone cellularity visualisation (Phase 7)."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from uniclone.viz._style import _require_plotly, clone_colors, default_layout

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from uniclone.core.types import CloneResult, Tensor


def vaf_histogram(
    alt: Tensor,
    depth: Tensor,
    result: CloneResult,
    *,
    sample_idx: int = 0,
    nbins: int = 50,
    title: str | None = None,
) -> go.Figure:
    """VAF histogram coloured by clone assignment."""
    _require_plotly()
    import plotly.graph_objects as go

    alt = np.asarray(alt, dtype=float)
    depth = np.asarray(depth, dtype=float)
    vaf = alt[:, sample_idx] / np.maximum(depth[:, sample_idx], 1.0)
    assignments = np.asarray(result.assignments)
    colors = clone_colors(result.K)

    fig = go.Figure()
    for k in range(result.K):
        mask = assignments == k
        fig.add_trace(go.Histogram(
            x=vaf[mask],
            nbinsx=nbins,
            name=f"Clone {k}",
            marker_color=colors[k],
            opacity=0.7,
        ))
    fig.update_layout(barmode="overlay", xaxis_title="VAF", yaxis_title="Count")
    return default_layout(fig, title or "VAF Histogram")


def cellularity_scatter(
    alt: Tensor,
    depth: Tensor,
    result: CloneResult,
    *,
    sample_idx: int = 0,
    title: str | None = None,
) -> go.Figure:
    """VAF scatter plot with clone centre lines."""
    _require_plotly()
    import plotly.graph_objects as go

    alt = np.asarray(alt, dtype=float)
    depth = np.asarray(depth, dtype=float)
    vaf = alt[:, sample_idx] / np.maximum(depth[:, sample_idx], 1.0)
    assignments = np.asarray(result.assignments)
    centers = np.asarray(result.centers)
    colors = clone_colors(result.K)

    fig = go.Figure()
    for k in range(result.K):
        mask = assignments == k
        fig.add_trace(go.Scatter(
            x=np.where(mask)[0],
            y=vaf[mask],
            mode="markers",
            name=f"Clone {k}",
            marker=dict(color=colors[k], size=4, opacity=0.6),
        ))
        # Centre line
        fig.add_hline(
            y=float(centers[k, sample_idx]),
            line_dash="dash",
            line_color=colors[k],
            annotation_text=f"φ={centers[k, sample_idx]:.2f}",
        )
    fig.update_layout(xaxis_title="Mutation index", yaxis_title="VAF")
    return default_layout(fig, title or "Cellularity Scatter")


def multi_sample_comparison(
    result: CloneResult,
    *,
    sample_labels: list[str] | None = None,
    title: str | None = None,
) -> go.Figure:
    """Grouped bar of clone cellularities across samples."""
    _require_plotly()
    import plotly.graph_objects as go

    centers = np.asarray(result.centers)  # (K, n_samples)
    n_samples = centers.shape[1]
    labels = sample_labels or [f"Sample {i}" for i in range(n_samples)]
    colors = clone_colors(result.K)

    fig = go.Figure()
    for k in range(result.K):
        fig.add_trace(go.Bar(
            x=labels,
            y=centers[k],
            name=f"Clone {k}",
            marker_color=colors[k],
        ))
    fig.update_layout(barmode="group", xaxis_title="Sample", yaxis_title="Cellularity")
    return default_layout(fig, title or "Multi-sample Comparison")


def responsibility_heatmap(
    result: CloneResult,
    *,
    sort_by_cluster: bool = True,
    title: str | None = None,
) -> go.Figure:
    """Soft assignment heatmap (mutations × clones)."""
    _require_plotly()
    import plotly.graph_objects as go

    resp = np.asarray(result.responsibilities)  # (n_mut, K)
    if sort_by_cluster:
        order = np.argsort(result.assignments)
        resp = resp[order]

    fig = go.Figure(go.Heatmap(
        z=resp.T,
        colorscale="Viridis",
        colorbar=dict(title="P(clone)"),
    ))
    fig.update_layout(xaxis_title="Mutation", yaxis_title="Clone")
    return default_layout(fig, title or "Responsibility Heatmap")
