"""
uniclone.phylo.pairwise
========================

Pairwise tensor phylogenetic constraint (Pairtree-style).

Constructs a pairwise relationship tensor encoding ancestor/descendant/
branching/same_clone relationships between mutation clusters, then scores
candidate trees.

References
----------
- Pairtree: Wintersinger et al. (2022) *Nature Genetics*
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from uniclone.core.types import CloneResult, Tensor
from uniclone.phylo.tree_utils import (
    TreeResult,
    adjacency_to_parent_vector,
    build_nesting_order,
    enumerate_trees,
    is_included,
)

if TYPE_CHECKING:
    from uniclone.core.config import CloneConfig

# Relation indices in the pairs tensor
_ANCESTOR = 0
_DESCENDANT = 1
_BRANCHING = 2
_SAME_CLONE = 3

_TEMP = 0.1  # sigmoid temperature


def _sigmoid(x: Tensor) -> Tensor:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class PairwisePhylo:
    """
    Pairtree-style pairwise relationship scoring.

    ``constrain`` is a no-op.
    ``postprocess`` builds a pairs tensor, scores candidate trees, and
    attaches the best tree plus the pairs tensor to the result.
    """

    def __init__(self, config: CloneConfig) -> None:
        self.config = config

    def constrain(self, centers: Tensor, raw_params: Tensor) -> Tensor:
        """No-op: pairwise scoring is applied only in postprocess."""
        return centers

    def postprocess(self, result: CloneResult) -> CloneResult:
        """
        Build pairs tensor, score trees, attach ``TreeResult`` and
        pairs tensor to ``result.tree``.
        """
        centers = result.centers  # (K, n_samples)
        K = result.K

        if K <= 1:
            pairs = np.zeros((K, K, 4), dtype=np.float64)
            adj = np.zeros((K, K), dtype=bool)
            parent = np.full(K, -1, dtype=int)
            nesting = np.zeros((K, K), dtype=bool)
            result.tree = TreeResult(adjacency=adj, parent=parent, is_included=nesting)
            return result

        pairs = self._build_pairs_tensor(centers, K)

        if K <= 7:
            adj = self._enumerate_and_score(centers, K, pairs)
        else:
            adj = self._greedy_tree(centers, K, pairs)

        parent = adjacency_to_parent_vector(adj)
        nesting = is_included(centers)
        tree = TreeResult(adjacency=adj, parent=parent, is_included=nesting)
        # Attach pairs tensor as extra attribute
        tree.pairs = pairs  # type: ignore[attr-defined]
        result.tree = tree
        return result

    @staticmethod
    def _build_pairs_tensor(centers: Tensor, K: int) -> Tensor:
        """
        Build pairs tensor P of shape (K, K, 4).

        Channels: [ancestor, descendant, branching, same_clone].
        Soft scoring via sigmoid; each (i, j) row normalised to sum to 1.
        """
        centers = np.asarray(centers, dtype=np.float64)
        pairs = np.zeros((K, K, 4), dtype=np.float64)

        for i in range(K):
            for j in range(K):
                if i == j:
                    pairs[i, j, _SAME_CLONE] = 1.0
                    continue

                diff = centers[i] - centers[j]  # (n_samples,)
                # Ancestor: i has higher cellularity than j
                score_anc = float(np.mean(_sigmoid(diff / _TEMP)))
                # Descendant: j has higher cellularity than i
                score_desc = float(np.mean(_sigmoid(-diff / _TEMP)))
                # Same clone: cellularities very similar
                score_same = float(np.mean(_sigmoid(-np.abs(diff) / _TEMP + 5.0)))
                # Branching: residual
                raw = np.array([score_anc, score_desc, 1e-3, score_same])
                raw = np.clip(raw, 1e-10, None)
                raw /= raw.sum()
                pairs[i, j] = raw

        return pairs

    @staticmethod
    def _relation_for_edge(adj: Tensor, i: int, j: int) -> int:
        """Determine the relation between i and j given a tree adjacency."""
        K = adj.shape[0]
        # Check if i is ancestor of j (path from i to j)
        visited = set()
        stack = [i]
        while stack:
            node = stack.pop()
            if node == j:
                return _ANCESTOR
            if node not in visited:
                visited.add(node)
                for c in range(K):
                    if adj[node, c]:
                        stack.append(c)

        # Check if j is ancestor of i
        visited = set()
        stack = [j]
        while stack:
            node = stack.pop()
            if node == i:
                return _DESCENDANT
            if node not in visited:
                visited.add(node)
                for c in range(K):
                    if adj[node, c]:
                        stack.append(c)

        return _BRANCHING

    def _score_tree(self, adj: Tensor, K: int, pairs: Tensor) -> float:
        """Score a candidate tree by summing log P[i,j,relation]."""
        score = 0.0
        for i in range(K):
            for j in range(K):
                if i == j:
                    continue
                rel = self._relation_for_edge(adj, i, j)
                score += np.log(max(pairs[i, j, rel], 1e-30))
        return score

    def _enumerate_and_score(self, centers: Tensor, K: int, pairs: Tensor) -> Tensor:
        """Enumerate all trees and pick the highest scoring one."""
        trees = enumerate_trees(K)
        best_adj = trees[0]
        best_score = -np.inf

        for adj in trees:
            score = self._score_tree(adj, K, pairs)
            if score > best_score:
                best_score = score
                best_adj = adj

        return best_adj

    @staticmethod
    def _greedy_tree(centers: Tensor, K: int, pairs: Tensor) -> Tensor:
        """Greedy tree construction for large K."""
        order = build_nesting_order(centers)
        root = int(order[0])
        adj = np.zeros((K, K), dtype=bool)
        attached = {root}

        for idx in range(1, K):
            child = int(order[idx])
            # Pick parent from attached nodes with highest ancestor score
            best_parent = root
            best_score = -np.inf
            for p in attached:
                score = pairs[p, child, _ANCESTOR]
                if score > best_score:
                    best_score = score
                    best_parent = p
            adj[best_parent, child] = True
            attached.add(child)

        return adj
