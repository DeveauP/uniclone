"""uniclone.viz.diagnostics — Diagnostic plots for EM/MFVI convergence (Phase 7)."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from uniclone.viz._style import _require_plotly, clone_colors, default_layout

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from uniclone.core.types import CloneResult


def bic_curve(
    results: list[CloneResult],
    *,
    title: str | None = None,
) -> go.Figure:
    """BIC vs K line plot for model selection."""
    _require_plotly()
    import plotly.graph_objects as go

    ks = [r.K for r in results]
    bics = [r.bic for r in results]
    order = np.argsort(ks)
    ks_sorted = [ks[i] for i in order]
    bics_sorted = [bics[i] for i in order]

    fig = go.Figure(go.Scatter(
        x=ks_sorted,
        y=bics_sorted,
        mode="lines+markers",
        marker=dict(size=8),
        line=dict(color=clone_colors(1)[0]),
    ))
    # Highlight minimum
    best = int(np.argmin(bics_sorted))
    fig.add_trace(go.Scatter(
        x=[ks_sorted[best]],
        y=[bics_sorted[best]],
        mode="markers",
        marker=dict(size=14, color="red", symbol="star"),
        name="Best K",
    ))
    fig.update_layout(xaxis_title="K (number of clones)", yaxis_title="BIC")
    return default_layout(fig, title or "BIC Curve")


def convergence_trace(
    log_likelihoods: list[float],
    *,
    title: str | None = None,
) -> go.Figure:
    """Log-likelihood per iteration convergence plot."""
    _require_plotly()
    import plotly.graph_objects as go

    fig = go.Figure(go.Scatter(
        x=list(range(len(log_likelihoods))),
        y=log_likelihoods,
        mode="lines+markers",
        marker=dict(size=4),
        line=dict(color=clone_colors(1)[0]),
    ))
    fig.update_layout(xaxis_title="Iteration", yaxis_title="Log-likelihood")
    return default_layout(fig, title or "Convergence Trace")


def feature_attribution_bar(
    attributions: dict[str, float],
    *,
    top_n: int = 10,
    title: str | None = None,
) -> go.Figure:
    """Horizontal bar chart of feature attributions."""
    _require_plotly()
    import plotly.graph_objects as go

    # Sort by absolute value, take top N
    sorted_items = sorted(attributions.items(), key=lambda x: abs(x[1]), reverse=True)
    if top_n > 0:
        sorted_items = sorted_items[:top_n]
    sorted_items.reverse()  # lowest at top for horizontal bar

    names = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    bar_colors = ["#D55E00" if v < 0 else "#0072B2" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation="h",
        marker_color=bar_colors,
    ))
    fig.update_layout(xaxis_title="Attribution", yaxis_title="Feature")
    return default_layout(fig, title or "Feature Attribution")


def responsibility_distribution(
    result: CloneResult,
    *,
    title: str | None = None,
) -> go.Figure:
    """Violin plot of max responsibility per mutation, grouped by cluster."""
    _require_plotly()
    import plotly.graph_objects as go

    resp = np.asarray(result.responsibilities)  # (n_mut, K)
    assignments = np.asarray(result.assignments)
    max_resp = resp.max(axis=1)
    colors = clone_colors(result.K)

    fig = go.Figure()
    for k in range(result.K):
        mask = assignments == k
        fig.add_trace(go.Violin(
            y=max_resp[mask],
            name=f"Clone {k}",
            box_visible=True,
            meanline_visible=True,
            fillcolor=colors[k],
            line_color=colors[k],
            opacity=0.7,
        ))
    fig.update_layout(xaxis_title="Clone", yaxis_title="Max responsibility")
    return default_layout(fig, title or "Responsibility Distribution")
