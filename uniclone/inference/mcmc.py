"""
uniclone.inference.mcmc
========================

MCMC inference engine using PyMC (optional dependency).

Implements full-posterior sampling via NUTS (No-U-Turn Sampler).
Requires ``pip install uniclone[mcmc]`` to enable the ``pymc`` dependency.

References
----------
- PyClone: Roth et al. (2014) *Nature Methods*
- PhyClone: McPherson et al. (2021) *Nature Methods*

Status: IMPLEMENTED — Phase 2.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from uniclone.core.types import CloneResult, EmissionModule, PhyloModule, Tensor
from uniclone.inference._utils import (
    compute_bic,
    compute_log_resp,
    compute_marginal_ll,
    softmax_rows,
)

try:
    import pymc as pm
    import pytensor.tensor as pt

    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False

if TYPE_CHECKING:
    from uniclone.core.config import CloneConfig


class MCMCInference:
    """
    MCMC inference engine for fixed K using PyMC/NUTS.

    Parameters
    ----------
    config : CloneConfig
    n_draws : int
        Number of posterior draws (default 2000).
    n_tune : int
        Number of tuning steps (default 1000).
    target_accept : float
        Target acceptance probability for NUTS (default 0.9).
    """

    def __init__(
        self,
        config: CloneConfig,
        n_draws: int = 2000,
        n_tune: int = 1000,
        target_accept: float = 0.9,
    ) -> None:
        if not HAS_PYMC:
            raise ImportError(
                "MCMCInference requires PyMC. "
                "Install the optional MCMC dependency: pip install uniclone[mcmc]"
            )
        self.config = config
        self.n_draws = n_draws
        self.n_tune = n_tune
        self.target_accept = target_accept

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
        """Run MCMC sampling for the given K and return a ``CloneResult``."""
        n_mut, n_samples = alt.shape

        with pm.Model() as model:
            # Priors
            pi = pm.Dirichlet("pi", a=np.ones(K), shape=(K,))
            centers = pm.Uniform(
                "centers", lower=1e-6, upper=1.0 - 1e-6, shape=(K, n_samples)
            )

            # Beta-binomial log-likelihood via pytensor
            # effective_mu[i, k, s] = centers[k, s] * adj_factor[i, s]
            adj_pt = pt.as_tensor_variable(adj_factor)  # (n_mut, n_samples)
            alt_pt = pt.as_tensor_variable(alt)
            depth_pt = pt.as_tensor_variable(depth)

            # (n_mut, K, n_samples)
            effective_mu = centers[None, :, :] * adj_pt[:, None, :]
            effective_mu = pt.clip(effective_mu, 1e-10, 1.0 - 1e-10)

            # Binomial log-prob per (mutation, clone, sample)
            from pytensor.tensor import gammaln as pt_gammaln

            log_binom = (
                pt_gammaln(depth_pt[:, None, :] + 1)
                - pt_gammaln(alt_pt[:, None, :] + 1)
                - pt_gammaln(depth_pt[:, None, :] - alt_pt[:, None, :] + 1)
                + alt_pt[:, None, :] * pt.log(effective_mu)
                + (depth_pt[:, None, :] - alt_pt[:, None, :]) * pt.log(1.0 - effective_mu)
            )
            # Sum over samples -> (n_mut, K)
            log_lik_per_clone = log_binom.sum(axis=-1)

            # Mixture log-likelihood: log sum_k pi_k * P(x_i | theta_k)
            log_mix = pt.log(pi)[None, :] + log_lik_per_clone
            ll_per_mut = pt.logsumexp(log_mix, axis=1)
            total_ll = ll_per_mut.sum()

            pm.Potential("loglik", total_ll)

            # Sample
            trace = pm.sample(
                draws=self.n_draws,
                tune=self.n_tune,
                target_accept=self.target_accept,
                cores=1,
                chains=1,
                progressbar=False,
                return_inferencedata=True,
            )

        # Posterior mean summarisation
        post_centers = trace.posterior["centers"].values.mean(axis=(0, 1))  # (K, n_samples)
        post_pi = trace.posterior["pi"].values.mean(axis=(0, 1))  # (K,)

        post_centers = np.clip(post_centers, 1e-6, 1.0 - 1e-6)

        # Single E-step at posterior means to get responsibilities
        log_pi = np.log(np.clip(post_pi, 1e-10, None))
        log_resp = compute_log_resp(
            alt, depth, adj_factor, emission, post_centers, log_pi
        )
        resp = softmax_rows(log_resp)
        assignments = resp.argmax(axis=1)

        # Marginal LL and BIC at posterior means
        ll = compute_marginal_ll(log_resp)
        bic = compute_bic(ll, K, n_samples, n_mut)

        return CloneResult(
            centers=post_centers,
            assignments=assignments,
            responsibilities=resp,
            log_likelihood=ll,
            bic=bic,
            K=K,
            n_iter=self.n_draws + self.n_tune,
            converged=True,
        )
