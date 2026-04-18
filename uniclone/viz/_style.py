"""uniclone.viz._style — Shared visual utilities for the viz suite."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import plotly.graph_objects as go


def _require_plotly() -> None:
    """Raise a helpful error if plotly is not installed."""
    try:
        import plotly  # noqa: F401
    except ImportError:
        raise ImportError("Visualisation requires plotly: pip install uniclone[viz]") from None


# 12-colour colourblind-safe palette (Wong + extended)
CLONE_PALETTE: list[str] = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
    "#332288",  # indigo
    "#88CCEE",  # cyan
    "#44AA99",  # teal
    "#882255",  # wine
]


def clone_colors(k: int) -> list[str]:
    """Return *k* colours, cycling the palette if necessary."""
    return [CLONE_PALETTE[i % len(CLONE_PALETTE)] for i in range(k)]


def default_layout(fig: go.Figure, title: str | None) -> go.Figure:
    """Apply consistent Plotly white template and styling."""
    fig.update_layout(
        template="plotly_white",
        title=title,
        font=dict(family="Arial, sans-serif", size=12),
        margin=dict(l=60, r=30, t=50 if title else 30, b=50),
    )
    return fig
