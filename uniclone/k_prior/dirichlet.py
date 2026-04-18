"""
uniclone.k_prior.dirichlet
===========================

Dirichlet concentration prior for automatic K selection.

A single run with a Dirichlet prior on cluster weights naturally prunes
empty or near-empty clusters, yielding an effective K without a BIC sweep.
The concentration parameter ``alpha`` controls how aggressively clusters
are pruned (smaller alpha → fewer active clusters).

References
----------
- PyClone-VI: Gillis et al. (2020) *Genome Biology*

Status: IMPLEMENTED — Phase 3.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from uniclone.core.types import CloneResult, Tensor
from uniclone.inference._utils import compute_bic

if TYPE_CHECKING:
    from uniclone.core.config import CloneConfig


class DirichletPrior:
    """
    Dirichlet concentration prior — single run with K_max, prune empty clusters.

    Parameters
    ----------
    config : CloneConfig
    k_max : int
        Maximum number of clusters to initialise (default 10).
    min_cluster_weight : float
        Minimum fraction of mutations assigned to keep a cluster (default 0.01).
    """

    def __init__(
        self,
        config: CloneConfig,
        k_max: int = 10,
        min_cluster_weight: float = 0.01,
    ) -> None:
        self.config = config
        self.k_max: int = config.n_clones if config.n_clones is not None else k_max
        self.min_cluster_weight = min_cluster_weight

    def get_k_schedule(self) -> list[int]:
        """Single run at K_max; pruning happens in ``select()``."""
        return [self.k_max]

    def initialise(self, K: int, alt: Tensor, depth: Tensor, adj_factor: Tensor) -> Tensor:
        """Quantile-based initialisation (same pattern as BICPrior)."""
        vaf = alt.astype(float) / np.maximum(depth.astype(float), 1)
        quantiles = np.linspace(0.0, 1.0, K + 2)[1:-1]
        centers = np.quantile(vaf, quantiles, axis=0)
        return np.clip(centers, 1e-6, 1.0 - 1e-6)

    def select(self, results: list[CloneResult]) -> CloneResult:
        """
        Prune and merge clusters from the single result.

        Two-stage process:
        1. Merge clusters whose centers are within ``merge_tol`` (L-inf).
        2. Prune clusters where ``resp[:, k].sum() / n_mut < min_cluster_weight``.
        """
        result = results[0]
        return _prune_and_merge(result, self.min_cluster_weight)


def _prune_and_merge(
    result: CloneResult,
    min_cluster_weight: float,
    merge_tol: float = 0.01,
) -> CloneResult:
    """
    Merge near-duplicate clusters, then prune low-weight ones.

    Parameters
    ----------
    result : CloneResult
    min_cluster_weight : float
        Minimum resp fraction to keep a cluster.
    merge_tol : float
        L-inf distance below which two cluster centers are merged.
    """
    centers = result.centers
    resp = result.responsibilities
    K = centers.shape[0]
    n_mut = resp.shape[0]

    # Stage 1: merge clusters with near-identical centers
    # Build a mapping from each cluster to its canonical representative
    canonical = list(range(K))
    for i in range(K):
        if canonical[i] != i:
            continue
        for j in range(i + 1, K):
            if canonical[j] != j:
                continue
            if np.max(np.abs(centers[i] - centers[j])) < merge_tol:
                canonical[j] = i

    # Aggregate responsibilities for merged clusters
    unique_ids = sorted(set(canonical))
    new_K = len(unique_ids)
    id_map = {old: new for new, old in enumerate(unique_ids)}

    new_resp = np.zeros((n_mut, new_K), dtype=resp.dtype)
    for k in range(K):
        new_resp[:, id_map[canonical[k]]] += resp[:, k]

    # New centers: weighted average by total responsibility
    new_centers = np.zeros((new_K, centers.shape[1]), dtype=centers.dtype)
    for k in range(K):
        nk = id_map[canonical[k]]
        weight = resp[:, k].sum()
        new_centers[nk] += centers[k] * weight
    for nk in range(new_K):
        total = new_resp[:, nk].sum()
        if total > 0:
            new_centers[nk] /= total
    new_centers = np.clip(new_centers, 1e-6, 1.0 - 1e-6)

    # Stage 2: prune low-weight clusters
    cluster_weights = new_resp.sum(axis=0) / n_mut
    active = np.where(cluster_weights >= min_cluster_weight)[0]

    if len(active) == 0:
        active = np.array([np.argmax(cluster_weights)])

    new_centers = new_centers[active]
    new_resp = new_resp[:, active]

    # Renormalise rows
    row_sums = new_resp.sum(axis=1, keepdims=True)
    new_resp = new_resp / np.maximum(row_sums, 1e-10)
    new_assignments = new_resp.argmax(axis=1)
    final_K = len(active)

    n_samples = new_centers.shape[1]
    bic = compute_bic(result.log_likelihood, final_K, n_samples, n_mut)

    return CloneResult(
        centers=new_centers,
        assignments=new_assignments,
        responsibilities=new_resp,
        log_likelihood=result.log_likelihood,
        bic=bic,
        K=final_K,
        n_iter=result.n_iter,
        converged=result.converged,
    )
