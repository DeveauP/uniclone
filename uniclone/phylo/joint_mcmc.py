"""
uniclone.phylo.joint_mcmc
==========================

Joint MCMC phylogenetic inference using TSSB prior (PhyClone-style).

Applies nesting projection during M-step (same as ConstrainedPhylo) for
the HYBRID engine path, and builds a greedy tree in postprocess.
True joint topology MCMC (Metropolis-Hastings tree moves) is not yet
implemented — this module provides a nesting-constrained approximation.

References
----------
- PhyClone: Remeseiro et al. (2023)
"""
from __future__ import annotations

import warnings
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


class JointMCMCPhylo:
    """
    PhyClone-style joint MCMC phylo (experimental).

    Uses the same nesting projection as ``ConstrainedPhylo`` during
    the M-step.  ``postprocess`` builds a greedy tree from nesting
    order.

    .. warning::
       This module is experimental.  It uses nesting-constrained
       EM/HYBRID as a pragmatic approximation rather than full
       Metropolis-Hastings tree topology moves.
    """

    def __init__(self, config: CloneConfig) -> None:
        warnings.warn(
            "JointMCMCPhylo is experimental — true joint topology MCMC "
            "is not yet implemented. Using nesting projection.",
            stacklevel=1,
        )
        self.config = config

    def constrain(self, centers: Tensor, raw_params: Tensor) -> Tensor:
        """Nesting projection (same as ConstrainedPhylo)."""
        centers = np.array(centers, dtype=np.float64)
        K = centers.shape[0]
        if K <= 1:
            return np.clip(centers, _TOL, 1.0 - _TOL)

        order = build_nesting_order(centers)
        for idx in range(1, K):
            child = int(order[idx])
            parent = int(order[idx - 1])
            centers[child] = np.minimum(centers[child], centers[parent] - _TOL)

        return np.clip(centers, _TOL, 1.0 - _TOL)

    def postprocess(self, result: CloneResult) -> CloneResult:
        """Greedy tree from nesting order."""
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
