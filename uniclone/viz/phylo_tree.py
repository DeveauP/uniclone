"""uniclone.viz.phylo_tree — Phylogenetic tree visualisation (Phase 7)."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import numpy as np

from uniclone.viz._style import _require_plotly, clone_colors, default_layout

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from uniclone.core.types import CloneResult


def _tree_layout(parent: np.ndarray) -> dict[int, tuple[float, float]]:
    """Layered BFS layout — no networkx dependency.

    Returns dict mapping node index to (x, y) coordinates.
    y=0 is root, increasing y goes deeper.
    """
    n_nodes = len(parent)
    root = int(np.where(parent == -1)[0][0])

    # Build children map
    children: dict[int, list[int]] = {i: [] for i in range(n_nodes)}
    for i in range(n_nodes):
        p = int(parent[i])
        if p >= 0:
            children[p].append(i)

    # BFS to assign layers
    pos: dict[int, tuple[float, float]] = {}
    queue: deque[int] = deque([root])
    depth_nodes: dict[int, list[int]] = {}
    node_depth: dict[int, int] = {root: 0}

    while queue:
        node = queue.popleft()
        d = node_depth[node]
        depth_nodes.setdefault(d, []).append(node)
        for child in children[node]:
            node_depth[child] = d + 1
            queue.append(child)

    # Assign x positions: spread nodes evenly at each depth
    for d, nodes in depth_nodes.items():
        for i, node in enumerate(nodes):
            x = (i + 0.5) / len(nodes)
            pos[node] = (x, -d)

    return pos


def clone_tree(
    result: CloneResult,
    *,
    sample_idx: int = 0,
    clone_labels: list[str] | None = None,
    show_edge_labels: bool = False,
    title: str | None = None,
) -> go.Figure:
    """Visualise the clone tree from ``result.tree``."""
    _require_plotly()
    import plotly.graph_objects as go

    if result.tree is None:
        raise ValueError("No tree available in result (result.tree is None).")

    parent = np.asarray(result.tree.parent)
    n_clones = len(parent)
    centers = np.asarray(result.centers)  # (K, n_samples)
    colors = clone_colors(n_clones)
    labels = clone_labels or [f"Clone {i}" for i in range(n_clones)]
    pos = _tree_layout(parent)

    # Draw edges
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for child in range(n_clones):
        p = int(parent[child])
        if p >= 0:
            px, py = pos[p]
            cx, cy = pos[child]
            edge_x.extend([px, cx, None])
            edge_y.extend([py, cy, None])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(color="#888", width=2),
            hoverinfo="none",
            showlegend=False,
        )
    )

    # Draw nodes
    node_x = [pos[i][0] for i in range(n_clones)]
    node_y = [pos[i][1] for i in range(n_clones)]
    node_text = [f"{labels[i]}<br>φ={centers[i, sample_idx]:.3f}" for i in range(n_clones)]
    node_sizes = [max(15, float(centers[i, sample_idx]) * 50) for i in range(n_clones)]

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=[labels[i] for i in range(n_clones)],
            textposition="top center",
            hovertext=node_text,
            hoverinfo="text",
            marker=dict(
                size=node_sizes,
                color=[colors[i] for i in range(n_clones)],
                line=dict(width=1, color="white"),
            ),
            showlegend=False,
        )
    )

    # Edge labels (cellularity difference)
    if show_edge_labels:
        for child in range(n_clones):
            p = int(parent[child])
            if p >= 0:
                mx = (pos[p][0] + pos[child][0]) / 2
                my = (pos[p][1] + pos[child][1]) / 2
                diff = abs(centers[p, sample_idx] - centers[child, sample_idx])
                fig.add_annotation(
                    x=mx,
                    y=my,
                    text=f"Δ={diff:.2f}",
                    showarrow=False,
                    font=dict(size=9),
                )

    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    return default_layout(fig, title or "Clone Tree")
