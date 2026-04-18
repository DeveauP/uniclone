"""
uniclone.inference.em
=====================

Expectation-Maximisation inference engine.

This is a direct port of the QuantumClone R EM algorithm to Python/NumPy.

Algorithm
---------
For a fixed K (number of clones):

1. **E-step**: For each mutation i and clone k, compute the log responsibility::

       log r_{ik} = sum_s log P(alt_{is} | depth_{is}, centers[k,s] * adj[i,s])

   then normalise by softmax across K to get r_{ik} ∈ [0,1].

2. **M-step**: Update clone centers by weighted mean::

       centers[k, s] = sum_i(r_{ik} * alt[i,s]) / sum_i(r_{ik} * depth[i,s])

3. **Convergence**: Stop when |ΔLL| < tol or max_iter reached.

BIC convention: ``BIC = -2 * log_likelihood + n_params * log(n_mut)``
Lower BIC = better model.  ``BICPrior.select`` minimises BIC.

Status: IMPLEMENTED — Phase 0.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from uniclone.core.backend import B
from uniclone.core.types import CloneResult, EmissionModule, PhyloModule, Tensor
from uniclone.inference._utils import (
    compute_bic,
    compute_log_resp,
    compute_marginal_ll,
    m_step_centers,
    softmax_rows,
)

if TYPE_CHECKING:
    from uniclone.core.config import CloneConfig


class EMInference:
    """
    EM inference engine for fixed K.

    Parameters
    ----------
    config : CloneConfig
    max_iter : int
        Maximum number of EM iterations (default 200).
    tol : float
        Convergence tolerance on log-likelihood change (default 1e-6).
    """

    def __init__(self, config: CloneConfig, max_iter: int = 200, tol: float = 1e-6) -> None:
        self.config = config
        self.max_iter = max_iter
        self.tol = tol

    def run(
        self,
        alt: Tensor,
        depth: Tensor,
        adj_factor: Tensor,
        emission: EmissionModule,
        phylo: PhyloModule,
        centers_init: Tensor,
        K: int,
    ) -> CloneResult:
        """
        Run EM for the given K and return a ``CloneResult``.

        Parameters
        ----------
        alt, depth, adj_factor : (n_mut, n_samples)
        emission   : active EmissionModule
        phylo      : active PhyloModule (constrain() called after M-step)
        centers_init : (K, n_samples) initial clone centers
        K          : number of clones
        """
        centers = centers_init.copy()
        log_pi = B.full((K,), -np.log(K))
        prev_ll = -np.inf
        n_iter = 0
        converged = False

        for n_iter in range(1, self.max_iter + 1):
            # --- E-step ---
            log_resp = compute_log_resp(alt, depth, adj_factor, emission, centers, log_pi)
            resp = softmax_rows(log_resp)

            # --- M-step (centers and mixing weights) ---
            centers = m_step_centers(alt, depth, resp)
            pi = resp.mean(axis=0)
            log_pi = B.log(B.clip(pi, 1e-10, None))

            # --- Phylo constraint (no-op for Phase 0) ---
            centers = phylo.constrain(centers, centers)

            # --- Log-likelihood ---
            ll = compute_marginal_ll(log_resp)

            if abs(ll - prev_ll) < self.tol:
                converged = True
                break
            prev_ll = ll

        # Hard assignments
        assignments = resp.argmax(axis=1)

        # BIC
        n_mut = alt.shape[0]
        n_samples = alt.shape[1]
        bic = compute_bic(ll, K, n_samples, n_mut)

        return CloneResult(
            centers=centers,
            assignments=assignments,
            responsibilities=resp,
            log_likelihood=ll,
            bic=bic,
            K=K,
            n_iter=n_iter,
            converged=converged,
        )
