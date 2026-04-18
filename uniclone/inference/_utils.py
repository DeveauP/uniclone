"""
uniclone.inference._utils
=========================

Shared helper functions used by EM, MFVI, Hybrid, and MCMC inference engines.
"""

from __future__ import annotations

from uniclone.core.backend import B
from uniclone.core.types import EmissionModule, Tensor


def compute_log_resp(
    alt: Tensor,
    depth: Tensor,
    adj_factor: Tensor,
    emission: EmissionModule,
    centers: Tensor,
    log_pi: Tensor,
) -> Tensor:
    """
    Compute unnormalised log responsibilities including mixing weights.

    log_resp[i, k] = log pi_k + log P(x_i | theta_k)

    Returns
    -------
    (n_mut, K) float
    """
    K = centers.shape[0]
    n_mut = alt.shape[0]
    log_resp = B.empty((n_mut, K))

    for k in range(K):
        effective_mu = adj_factor * centers[k]
        log_resp[:, k] = log_pi[k] + emission.log_prob(alt, depth, effective_mu)

    return log_resp


def softmax_rows(log_resp: Tensor) -> Tensor:
    """Numerically stable row-wise softmax -> responsibilities."""
    shifted = log_resp - log_resp.max(axis=1, keepdims=True)
    exp = B.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def m_step_centers(alt: Tensor, depth: Tensor, resp: Tensor) -> Tensor:
    """
    Weighted MLE update for clone centers.

    centers[k, s] = sum_i(r_{ik} * alt[i,s]) / sum_i(r_{ik} * depth[i,s])

    Returns
    -------
    (K, n_samples) float
    """
    weighted_alt = resp.T @ alt
    weighted_depth = resp.T @ depth
    new_centers = weighted_alt / B.maximum(weighted_depth, 1e-10)
    return B.clip(new_centers, 1e-6, 1.0 - 1e-6)


def compute_marginal_ll(log_resp: Tensor) -> float:
    """sum_i log(sum_k exp(log_resp[i,k]))"""
    return float(B.logsumexp(log_resp, axis=1).sum())


def compute_bic(ll: float, K: int, n_samples: int, n_mut: int) -> float:
    """BIC = -2*ll + n_params * log(n_mut)"""
    n_params = K * n_samples
    return -2.0 * ll + n_params * float(B.log(max(n_mut, 1)))
