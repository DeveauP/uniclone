"""uniclone.viz.dashboard — Interactive Dash dashboard (Phase 7, optional)."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from uniclone.core.types import CloneResult, Tensor


def _require_dash() -> None:
    """Raise a helpful error if dash is not installed."""
    try:
        import dash  # noqa: F401
    except ImportError:
        raise ImportError(
            "Dashboard requires dash: pip install uniclone[dashboard]"
        ) from None


def create_dashboard(
    result: CloneResult,
    *,
    alt: Tensor | None = None,
    depth: Tensor | None = None,
    sample_labels: list[str] | None = None,
    host: str = "127.0.0.1",
    port: int = 8050,
) -> Any:
    """Create a tabbed Dash app combining all visualisation plots.

    Returns a ``dash.Dash`` instance. Does **not** call ``run_server()``.
    """
    _require_dash()

    import dash
    from dash import dcc, html

    from uniclone.viz.cellularity import (
        multi_sample_comparison,
        responsibility_heatmap,
    )
    from uniclone.viz.diagnostics import responsibility_distribution
    from uniclone.viz.evolution import clone_proportion_bar, fish_plot

    app = dash.Dash(__name__)

    tabs: list[Any] = []

    # Cellularity tab
    cellularity_children: list[Any] = []
    if alt is not None and depth is not None:
        from uniclone.viz.cellularity import cellularity_scatter, vaf_histogram

        cellularity_children.append(dcc.Graph(
            figure=vaf_histogram(alt, depth, result),
        ))
        cellularity_children.append(dcc.Graph(
            figure=cellularity_scatter(alt, depth, result),
        ))
    cellularity_children.append(dcc.Graph(
        figure=multi_sample_comparison(result, sample_labels=sample_labels),
    ))
    cellularity_children.append(dcc.Graph(
        figure=responsibility_heatmap(result),
    ))
    tabs.append(dcc.Tab(label="Cellularity", children=cellularity_children))

    # Evolution tab
    tabs.append(dcc.Tab(label="Evolution", children=[
        dcc.Graph(figure=fish_plot(result, sample_labels=sample_labels)),
        dcc.Graph(figure=clone_proportion_bar(result, sample_labels=sample_labels)),
    ]))

    # Tree tab
    if result.tree is not None:
        from uniclone.viz.phylo_tree import clone_tree

        tabs.append(dcc.Tab(label="Tree", children=[
            dcc.Graph(figure=clone_tree(result)),
        ]))

    # Diagnostics tab
    tabs.append(dcc.Tab(label="Diagnostics", children=[
        dcc.Graph(figure=responsibility_distribution(result)),
    ]))

    app.layout = html.Div([
        html.H1("UniClone Dashboard"),
        dcc.Tabs(children=tabs),
    ])

    return app


def run_dashboard(
    result: CloneResult,
    *,
    alt: Tensor | None = None,
    depth: Tensor | None = None,
    sample_labels: list[str] | None = None,
    host: str = "127.0.0.1",
    port: int = 8050,
    debug: bool = False,
) -> None:
    """Create and run the dashboard server."""
    app = create_dashboard(
        result,
        alt=alt,
        depth=depth,
        sample_labels=sample_labels,
        host=host,
        port=port,
    )
    app.run_server(host=host, port=port, debug=debug)
