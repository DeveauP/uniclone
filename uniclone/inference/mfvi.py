"""
uniclone.inference.mfvi
========================

Mean-field variational inference engine (PyClone-VI style).

Algorithm
---------
Coordinate Ascent VI with Dirichlet-Categorical conjugacy:

1. **E-step**: ``log r_{ik} = digamma(alpha_k) - digamma(sum alpha)
   + log P(x_i | centers_k)``, then softmax.
2. **M-step**: Weighted MLE center update (same as EM).
3. **Dirichlet update**: ``alpha_k = alpha0 + sum_i r_{ik}``.
4. **ELBO** monitoring for convergence.
5. **Return**: CloneResult with marginal LL and BIC (same formulas as EM).

References
----------
- PyClone-VI: Gillis et al. (2020) *Genome Biology*

Status: IMPLEMENTED — Phase 2.
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


class MFVIInference:
    """
    Mean-field variational inference engine for fixed K.

    Parameters
    ----------
    config : CloneConfig
    max_iter : int
        Maximum CAVI iterations (default 200).
    tol : float
        Convergence tolerance on ELBO change (default 1e-6).
    alpha0 : float
        Dirichlet concentration prior per component (default 1.0).
    """

    def __init__(
        self,
        config: CloneConfig,
        max_iter: int = 200,
        tol: float = 1e-6,
        alpha0: float = 1.0,
    ) -> None:
        self.config = config
        self.max_iter = max_iter
        self.tol = tol
        self.alpha0 = alpha0

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
        """Run MFVI for the given K and return a ``CloneResult``."""
        centers = centers_init.copy()
        alpha = B.full((K,), self.alpha0)
        prev_elbo = -np.inf
        n_iter = 0
        converged = False

        for n_iter in range(1, self.max_iter + 1):
            # --- E-step (variational) ---
            # E[log pi_k] = digamma(alpha_k) - digamma(sum alpha)
            e_log_pi = B.digamma(alpha) - B.digamma(alpha.sum())

            # log_resp using E[log pi] instead of log pi
            log_resp = self._compute_vi_log_resp(
                alt, depth, adj_factor, emission, centers, e_log_pi
            )
            resp = softmax_rows(log_resp)

            # --- M-step (centers) ---
            centers = m_step_centers(alt, depth, resp)

            # --- Phylo constraint ---
            centers = phylo.constrain(centers, centers)

            # --- Dirichlet update ---
            alpha = self.alpha0 + resp.sum(axis=0)

            # --- ELBO ---
            elbo = self._compute_elbo(log_resp, resp, alpha, e_log_pi)

            if abs(elbo - prev_elbo) < self.tol:
                converged = True
                break
            prev_elbo = elbo

        # Hard assignments
        assignments = resp.argmax(axis=1)

        # Compute actual marginal LL at converged parameters for BIC
        log_pi = B.log(B.clip(alpha / alpha.sum(), 1e-10, None))
        log_resp_final = compute_log_resp(alt, depth, adj_factor, emission, centers, log_pi)
        ll = compute_marginal_ll(log_resp_final)

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

    @staticmethod
    def _compute_vi_log_resp(
        alt: Tensor,
        depth: Tensor,
        adj_factor: Tensor,
        emission: EmissionModule,
        centers: Tensor,
        e_log_pi: Tensor,
    ) -> Tensor:
        """Log responsibilities using E[log pi] from variational Dirichlet."""
        K = centers.shape[0]
        n_mut = alt.shape[0]
        log_resp = B.empty((n_mut, K))

        for k in range(K):
            effective_mu = adj_factor * centers[k]
            log_resp[:, k] = e_log_pi[k] + emission.log_prob(alt, depth, effective_mu)

        return log_resp

    def _compute_elbo(
        self,
        log_resp: Tensor,
        resp: Tensor,
        alpha: Tensor,
        e_log_pi: Tensor,
    ) -> float:
        """
        ELBO = data_term + E[log pi] + entropy - KL(q(pi)||p(pi))

        data_term: sum_i sum_k r_{ik} * (log P(x_i|theta_k) + E[log pi_k])
        entropy:   -sum_i sum_k r_{ik} * log r_{ik}
        KL:        KL(Dir(alpha) || Dir(alpha0))
        """
        K = alpha.shape[0]

        # Data term: sum of resp * log_resp (unnormalised)
        data_term = float((resp * log_resp).sum())

        # Entropy of q(z)
        log_resp_safe = B.log(B.clip(resp, 1e-300, None))
        entropy = -float((resp * log_resp_safe).sum())

        # KL(Dir(alpha) || Dir(alpha0))
        alpha0_vec = B.full((K,), self.alpha0)
        kl = float(
            B.gammaln(alpha.sum())
            - B.gammaln(alpha0_vec.sum())
            - B.gammaln(alpha).sum()
            + B.gammaln(alpha0_vec).sum()
            + ((alpha - alpha0_vec) * (B.digamma(alpha) - B.digamma(alpha.sum()))).sum()
        )

        return data_term + entropy - kl
