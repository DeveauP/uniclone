"""
uniclone.phylo.post_hoc
========================

Post-hoc phylogenetic tree construction (QuantumClone-style).

After EM convergence, builds an ``is_included`` nesting matrix:
clone j is nested within clone k if its cellularity is <= clone k's
cellularity in all samples (consistent with j being a subclone of k).

This corresponds to the ``Phylogeny_tree.R`` step in QuantumClone.

During inference, ``constrain`` is a no-op — the constraint is applied
only in ``postprocess``, which is called once after the best K is selected.
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


class PostHocPhylo:
    """
    Post-hoc phylogenetic nesting matrix.

    During inference, ``constrain`` is a no-op.
    After convergence, ``postprocess`` attaches a ``TreeResult`` to
    ``result.tree``.
    """

    def __init__(self, config: CloneConfig) -> None:
        self.config = config

    def constrain(self, centers: Tensor, raw_params: Tensor) -> Tensor:
        """No-op during EM: return ``centers`` unchanged."""
        return centers

    def postprocess(self, result: CloneResult) -> CloneResult:
        """
        Build is_included nesting matrix and derive a greedy parent tree.

        Attaches a ``TreeResult`` to ``result.tree``.
        """
        centers = result.centers  # (K, n_samples)
        K = result.K

        nesting = is_included(centers)

        # Greedy tree: each clone's parent is the smallest-cellularity clone
        # that nests it.
        order = build_nesting_order(centers)
        root = int(order[0])
        adj = np.zeros((K, K), dtype=bool)
        mean_cell = centers.mean(axis=1)

        for idx in range(1, K):
            child = int(order[idx])
            # Find candidate parents: clones that nest this child
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
