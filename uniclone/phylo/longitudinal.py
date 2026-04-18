"""
uniclone.phylo.longitudinal
============================

Longitudinal ILP time-ordering (CALDER-style).

Enforces that clone cellularities are consistent with a time-ordered
series of samples (diagnosis -> treatment -> relapse) using integer linear
programming via ``scipy.optimize.milp``.

References
----------
- CALDER: Myers et al. (2022)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import LinearConstraint, milp

from uniclone.core.types import CloneResult, Tensor
from uniclone.phylo.tree_utils import (
    TreeResult,
    adjacency_to_parent_vector,
    build_nesting_order,
    is_included,
)

if TYPE_CHECKING:
    from uniclone.core.config import CloneConfig


class LongitudinalPhylo:
    """
    CALDER-style ILP phylogenetic reconstruction for longitudinal data.

    ``constrain`` is a no-op (ILP is too expensive per M-step).
    ``postprocess`` solves an ILP to find the optimal tree.
    """

    def __init__(self, config: CloneConfig) -> None:
        self.config = config

    def constrain(self, centers: Tensor, raw_params: Tensor) -> Tensor:
        """No-op: ILP is applied only in postprocess."""
        return centers

    def postprocess(self, result: CloneResult) -> CloneResult:
        """
        Solve ILP to find optimal tree respecting the sum rule across
        timepoints, then attach ``TreeResult``.
        """
        centers = result.centers  # (K, n_samples)
        K = result.K

        if K <= 1:
            adj = np.zeros((K, K), dtype=bool)
            parent = np.full(K, -1, dtype=int)
            nesting = np.zeros((K, K), dtype=bool)
            result.tree = TreeResult(adjacency=adj, parent=parent, is_included=nesting)
            return result

        try:
            adj = self._solve_ilp(centers, K)
        except Exception:
            adj = self._greedy_fallback(centers, K)

        parent = adjacency_to_parent_vector(adj)
        nesting = is_included(centers)
        result.tree = TreeResult(adjacency=adj, parent=parent, is_included=nesting)
        return result

    def _solve_ilp(self, centers: Tensor, K: int) -> Tensor:
        """
        ILP formulation:

        Decision variables:
          e[i,j] binary (K*K) — edge i->j
          u[i] continuous (K) — MTZ ordering variable

        Objective: maximize total edge weight (correlation of CCF profiles).

        Constraints:
          - Each non-root has exactly 1 parent
          - Root (highest mean cellularity) has 0 parents
          - Sum rule: centers[i,t] >= sum_j(e[i,j] * centers[j,t]) per timepoint
          - MTZ subtour elimination
        """
        centers = np.asarray(centers, dtype=np.float64)
        n_samples = centers.shape[1]
        order = build_nesting_order(centers)
        root = int(order[0])

        n_edge = K * K  # e[i,j] = var index i*K + j
        n_u = K  # u[i] = var index n_edge + i
        n_vars = n_edge + n_u

        # Edge weights: correlation of CCF profiles (higher = better)
        weights = np.zeros(n_edge)
        for i in range(K):
            for j in range(K):
                if i != j:
                    ci = centers[i]
                    cj = centers[j]
                    if np.std(ci) > 1e-10 and np.std(cj) > 1e-10:
                        weights[i * K + j] = np.corrcoef(ci, cj)[0, 1]
                    else:
                        weights[i * K + j] = 0.5

        # Objective: maximize edge weight = minimize negative
        c = np.zeros(n_vars)
        c[:n_edge] = -weights  # minimize negative = maximize

        # Variable bounds
        # e[i,j] in {0, 1}, u[i] in [0, K-1]
        bounds_lower = np.zeros(n_vars)
        bounds_upper = np.ones(n_vars)
        bounds_upper[n_edge:] = K - 1  # u bounds

        # No self-loops: e[i,i] = 0
        for i in range(K):
            bounds_upper[i * K + i] = 0.0

        # No edges into root
        for i in range(K):
            bounds_upper[i * K + root] = 0.0

        # Integrality: e vars are integer (1), u vars are continuous (0)
        integrality = np.zeros(n_vars, dtype=int)
        integrality[:n_edge] = 1

        # Constraint matrices
        A_rows: list[np.ndarray] = []
        b_lower: list[float] = []
        b_upper: list[float] = []

        # C1: Each non-root node has exactly 1 parent
        for j in range(K):
            if j == root:
                continue
            row = np.zeros(n_vars)
            for i in range(K):
                if i != j:
                    row[i * K + j] = 1.0
            A_rows.append(row)
            b_lower.append(1.0)
            b_upper.append(1.0)

        # C2: Root has 0 parents (already enforced by bounds, but explicit)
        row = np.zeros(n_vars)
        for i in range(K):
            if i != root:
                row[i * K + root] = 1.0
        A_rows.append(row)
        b_lower.append(0.0)
        b_upper.append(0.0)

        # C3: Sum rule: for each parent i and timepoint t,
        # centers[i,t] >= sum_j e[i,j]*centers[j,t]
        # => sum_j e[i,j]*centers[j,t] <= centers[i,t]
        for i in range(K):
            for t in range(n_samples):
                row = np.zeros(n_vars)
                for j in range(K):
                    if j != i:
                        row[i * K + j] = centers[j, t]
                A_rows.append(row)
                b_lower.append(-np.inf)
                b_upper.append(float(centers[i, t]))

        # C4: MTZ subtour elimination: u[j] >= u[i] + 1 - K*(1-e[i,j])
        # => u[j] - u[i] + K*e[i,j] >= 1 - K + K = 1? No:
        # u[j] - u[i] - K*e[i,j] >= 1 - K
        # Rearranged: u[i] - u[j] + K*e[i,j] <= K - 1
        for i in range(K):
            for j in range(K):
                if i != j and j != root:
                    row = np.zeros(n_vars)
                    row[n_edge + i] = 1.0
                    row[n_edge + j] = -1.0
                    row[i * K + j] = float(K)
                    A_rows.append(row)
                    b_lower.append(-np.inf)
                    b_upper.append(float(K - 1))

        # u[root] = 0
        row = np.zeros(n_vars)
        row[n_edge + root] = 1.0
        A_rows.append(row)
        b_lower.append(0.0)
        b_upper.append(0.0)

        A = np.array(A_rows)
        constraints = LinearConstraint(A, b_lower, b_upper)

        from scipy.optimize import Bounds

        result = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=Bounds(lb=bounds_lower, ub=bounds_upper),
        )

        if not result.success:
            raise RuntimeError(f"ILP failed: {result.message}")

        # Extract adjacency
        e = result.x[:n_edge]
        adj = np.zeros((K, K), dtype=bool)
        for i in range(K):
            for j in range(K):
                if e[i * K + j] > 0.5:
                    adj[i, j] = True

        return adj

    @staticmethod
    def _greedy_fallback(centers: Tensor, K: int) -> Tensor:
        """Greedy nesting tree (same as PostHocPhylo)."""
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

        return adj
