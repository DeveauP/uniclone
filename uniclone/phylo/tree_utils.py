"""
uniclone.phylo.tree_utils
==========================

Shared phylogenetic tree utilities: topological sort, DAG validation,
nesting checks, tree enumeration, and adjacency/parent conversions.
"""
from __future__ import annotations

import warnings
from collections import deque
from dataclasses import dataclass
from itertools import product

import numpy as np

from uniclone.core.types import Tensor


@dataclass
class TreeResult:
    """Typed container for phylogenetic tree output."""

    adjacency: Tensor   # (K, K) bool — adjacency[i,j]=True means i→j edge
    parent: Tensor      # (K,) int — parent[root]=-1
    is_included: Tensor  # (K, K) bool — nesting matrix


def topological_sort(adj: Tensor) -> list[int]:
    """
    Kahn's algorithm on a (K, K) boolean adjacency matrix.

    Parameters
    ----------
    adj : (K, K) bool
        ``adj[i, j] = True`` means there is a directed edge i → j.

    Returns
    -------
    list[int]
        Nodes in topological order.

    Raises
    ------
    ValueError
        If the graph contains a cycle.
    """
    adj = np.asarray(adj, dtype=bool)
    K = adj.shape[0]
    in_degree = adj.sum(axis=0).astype(int)  # column sums
    queue: deque[int] = deque()
    for i in range(K):
        if in_degree[i] == 0:
            queue.append(i)

    order: list[int] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for j in range(K):
            if adj[node, j]:
                in_degree[j] -= 1
                if in_degree[j] == 0:
                    queue.append(j)

    if len(order) != K:
        raise ValueError("Adjacency matrix contains a cycle — not a valid DAG.")
    return order


def is_valid_dag(adj: Tensor) -> bool:
    """Check whether the adjacency matrix encodes a valid DAG."""
    try:
        topological_sort(adj)
        return True
    except ValueError:
        return False


def build_nesting_order(centers: Tensor) -> Tensor:
    """
    Sort clones by descending mean cellularity.

    Parameters
    ----------
    centers : (K, n_samples)

    Returns
    -------
    (K,) int — index array, highest mean cellularity first.
    """
    mean_cell = np.asarray(centers).mean(axis=1)
    return np.argsort(-mean_cell)


def is_included(centers: Tensor, tol: float = 1e-6) -> Tensor:
    """
    Build nesting matrix: ``out[j, k] = True`` iff clone j is nested in
    clone k (cellularity_j <= cellularity_k + tol for all samples).

    Parameters
    ----------
    centers : (K, n_samples)
    tol : float

    Returns
    -------
    (K, K) bool
    """
    centers = np.asarray(centers)
    K = centers.shape[0]
    mat = np.zeros((K, K), dtype=bool)
    for j in range(K):
        for k in range(K):
            if j != k:
                mat[j, k] = bool(np.all(centers[j] <= centers[k] + tol))
    return mat


def enumerate_trees(K: int, rng: np.random.Generator | None = None) -> list[Tensor]:
    """
    Enumerate rooted labelled trees on K nodes via Prufer sequences.

    For K <= 7 returns all trees; for K > 7 samples 10000 at random.

    Each tree is returned as a (K, K) boolean adjacency matrix rooted at
    the node with highest degree (ties broken by lowest index).

    Parameters
    ----------
    K : int
    rng : Generator, optional

    Returns
    -------
    list of (K, K) bool arrays
    """
    if K == 1:
        return [np.zeros((1, 1), dtype=bool)]

    if K == 2:
        # Only one tree: 0→1
        adj = np.zeros((2, 2), dtype=bool)
        adj[0, 1] = True
        return [adj]

    n_possible = K ** (K - 2)  # Cayley's formula
    sample = K > 7

    if sample:
        warnings.warn(
            f"K={K} > 7: sampling 10000 of {n_possible} possible trees.",
            stacklevel=2,
        )
        if rng is None:
            rng = np.random.default_rng()
        sequences = [tuple(rng.integers(0, K, size=K - 2)) for _ in range(10000)]
        # deduplicate
        sequences = list(set(sequences))
    else:
        sequences = list(product(range(K), repeat=K - 2))

    trees: list[Tensor] = []
    for seq in sequences:
        adj = _prufer_to_rooted_adjacency(list(seq), K)
        trees.append(adj)
    return trees


def _prufer_to_rooted_adjacency(seq: list[int], K: int) -> Tensor:
    """Convert a Prufer sequence to a rooted adjacency matrix."""
    # Decode Prufer sequence to edge list (unrooted tree)
    degree = np.ones(K, dtype=int)
    for i in seq:
        degree[i] += 1

    edges: list[tuple[int, int]] = []
    seq_iter = iter(seq)
    for s in seq_iter:
        for leaf in range(K):
            if degree[leaf] == 1:
                edges.append((leaf, s))
                degree[leaf] -= 1
                degree[s] -= 1
                break

    # Last edge: two remaining nodes with degree 1
    remaining = [i for i in range(K) if degree[i] == 1]
    if len(remaining) == 2:
        edges.append((remaining[0], remaining[1]))

    # Build undirected adjacency
    undirected = np.zeros((K, K), dtype=bool)
    for u, v in edges:
        undirected[u, v] = True
        undirected[v, u] = True

    # Root at node 0 (convention: node 0 is root)
    # BFS to orient edges away from root
    adj = np.zeros((K, K), dtype=bool)
    visited = np.zeros(K, dtype=bool)
    queue: deque[int] = deque([0])
    visited[0] = True
    while queue:
        node = queue.popleft()
        for j in range(K):
            if undirected[node, j] and not visited[j]:
                adj[node, j] = True
                visited[j] = True
                queue.append(j)

    return adj


def adjacency_to_parent_vector(adj: Tensor) -> Tensor:
    """
    BFS from the root (in-degree 0 node) to produce a parent vector.

    Parameters
    ----------
    adj : (K, K) bool

    Returns
    -------
    (K,) int — ``parent[root] = -1``.
    """
    adj = np.asarray(adj, dtype=bool)
    K = adj.shape[0]
    in_deg = adj.sum(axis=0)
    roots = np.where(in_deg == 0)[0]

    if len(roots) == 0:
        raise ValueError("No root found (no node with in-degree 0).")

    root = int(roots[0])
    parent = np.full(K, -1, dtype=int)

    queue: deque[int] = deque([root])
    visited = np.zeros(K, dtype=bool)
    visited[root] = True

    while queue:
        node = queue.popleft()
        for j in range(K):
            if adj[node, j] and not visited[j]:
                parent[j] = node
                visited[j] = True
                queue.append(j)

    return parent
