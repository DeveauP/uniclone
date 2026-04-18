"""
uniclone.phylo.constrained
===========================

Constrained phylogenetic M-step via projection.

Applies a nesting projection to clone centers every M-step so that
child clones always have cellularity <= their parent in all samples.
After convergence, ``postprocess`` builds the full tree.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from uniclone.core.types import CloneResult, Tensor
from uniclone.phylo.tree_utils import (
    TreeResult,
    adjacency_to_parent_vector,
    build_nesting_order,
    is_included,
)

if TYPE_CHECKING:
    from uniclone.core.config import CloneConfig

_TOL = 1e-6


class ConstrainedPhylo:
    """
    Projection-based phylogenetic constraint.

    During each M-step, ``constrain`` clamps child cellularities to be
    at most their parent's cellularity (determined by descending mean
    cellularity order).  After convergence, ``postprocess`` attaches
    a ``TreeResult``.
    """

    def __init__(self, config: CloneConfig) -> None:
        self.config = config

    def constrain(self, centers: Tensor, raw_params: Tensor) -> Tensor:
        """
        Project centers so children have cellularity <= parent - tol.

        Clones are ordered by descending mean cellularity; each non-root
        clone is clamped sample-wise.
        """
        centers = np.array(centers, dtype=np.float64)
        K = centers.shape[0]
        if K <= 1:
            return np.clip(centers, _TOL, 1.0 - _TOL)

        order = build_nesting_order(centers)

        # Clamp in nesting order: each child <= its immediate predecessor
        # in the ordering (greedy chain).
        for idx in range(1, K):
            child = int(order[idx])
            parent = int(order[idx - 1])
            centers[child] = np.minimum(centers[child], centers[parent] - _TOL)

        return np.clip(centers, _TOL, 1.0 - _TOL)

    def postprocess(self, result: CloneResult) -> CloneResult:
        """Build nesting matrix, derive parent vector, attach TreeResult."""
        centers = result.centers
        K = result.K

        nesting = is_included(centers)
        order = build_nesting_order(centers)
        root = int(order[0])
        adj = np.zeros((K, K), dtype=bool)
        mean_cell = centers.mean(axis=1)

        for idx in range(1, K):
            child = int(order[idx])
            best_parent = root
            best_cell = np.inf
            for p in range(K):
                if p != child and nesting[child, p]:
                    if mean_cell[p] < best_cell and mean_cell[p] > mean_cell[child]:
                        best_cell = mean_cell[p]
                        best_parent = p
            adj[best_parent, child] = True

        parent = adjacency_to_parent_vector(adj)
        result.tree = TreeResult(adjacency=adj, parent=parent, is_included=nesting)
        return result
