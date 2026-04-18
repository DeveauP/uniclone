"""
uniclone.noise.tail_filter
===========================

Pareto tail detection and mutation removal (MOBSTER-style).

Detects mutations assigned to a neutral Pareto tail component and removes
them before clonal inference.  The tail is modelled as a Pareto distribution
over VAF values; mutations with high posterior assignment to the tail are
masked out.

References
----------
- MOBSTER: Caravagna et al. (2020) *Nature Methods*
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from uniclone.core.types import CloneResult, Tensor
from uniclone.noise._utils import expand_result

if TYPE_CHECKING:
    from uniclone.core.config import CloneConfig

# Pareto tail shape parameter (alpha = 1 gives the 1/f neutral tail)
_PARETO_ALPHA = 1.0
# Minimum VAF to avoid log(0) in Pareto density
_VAF_FLOOR = 1e-8
# EM convergence for internal tau/phi estimation
_EM_TOL = 1e-4
_EM_MAX_ITER = 50


class TailFilterNoise:
    """
    MOBSTER-style Pareto tail filter.

    Pre-processing:
        Fits a Beta-Binomial + Pareto mixture to VAF values and masks
        mutations where P(tail) > ``config.tail_threshold`` in ANY sample.

    Post-processing:
        Expands the filtered result back to the original mutation count,
        assigning filtered mutations label -1.
    """

    def __init__(self, config: CloneConfig) -> None:
        self.config = config
        self.threshold = config.tail_threshold
        self.phi = config.phi
        self.tau = config.tail_weight

    def preprocess(
        self,
        alt: Tensor,
        depth: Tensor,
        adj_factor: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        n_mut, n_samples = alt.shape
        vaf = alt / np.maximum(depth, 1.0)

        # Per-sample posterior P(tail)
        p_tail = np.zeros((n_mut, n_samples))

        for s in range(n_samples):
            p_tail[:, s] = self._compute_tail_prob(vaf[:, s], alt[:, s], depth[:, s])

        # Mask: mutation is tail if P(tail) > threshold in ANY sample
        is_tail = np.any(p_tail > self.threshold, axis=1)
        mask = ~is_tail

        # Guard: keep all if too few remain
        if mask.sum() < 2:
            mask = np.ones(n_mut, dtype=bool)

        return alt[mask], depth[mask], adj_factor[mask], mask

    def postprocess(self, result: CloneResult, mask: Tensor) -> CloneResult:
        if mask.all():
            result.noise_mask = mask
            return result
        return expand_result(result, mask)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_tail_prob(self, vaf: Tensor, alt_s: Tensor, depth_s: Tensor) -> Tensor:
        """Compute per-mutation posterior P(tail) for one sample."""
        vaf_safe = np.clip(vaf, _VAF_FLOOR, 1.0 - _VAF_FLOOR)

        # Log-likelihood under 1/f neutral tail (power law on (0, 1))
        # p(f) ∝ f^{-alpha}, so log p(f) = -alpha * log(f) + const
        # Normalise over (_VAF_FLOOR, 1): integral = (1 - eps^{1-a})/(1-a)
        # For alpha=1: integral = -log(eps), so log_norm = log(-log(eps))
        log_norm = np.log(-np.log(_VAF_FLOOR))
        log_pareto = -_PARETO_ALPHA * np.log(vaf_safe) - log_norm

        # Log-likelihood under Beta clonal component (continuous density on VAF)
        # Using Beta density so it's comparable in scale with the Pareto density
        phi = self.phi if self.phi is not None else self._estimate_phi_from_vaf(vaf_safe)
        mu = np.clip(np.median(vaf_safe), 0.05, 0.95)
        log_bb = self._beta_logpdf(vaf_safe, mu, phi)

        # Mixing weight
        tau = self.tau if self.tau is not None else self._estimate_tau(log_pareto, log_bb)
        tau = np.clip(tau, 1e-6, 1.0 - 1e-6)

        # Posterior P(tail | data) via Bayes
        log_num = np.log(tau) + log_pareto
        log_den = np.logaddexp(log_num, np.log(1.0 - tau) + log_bb)
        p_tail = np.exp(log_num - log_den)

        return p_tail

    @staticmethod
    def _beta_logpdf(vaf: Tensor, mu: float, phi: float) -> Tensor:
        """Beta log-PDF for each mutation's VAF."""
        from scipy.special import betaln

        a = mu * phi
        b = (1.0 - mu) * phi
        log_p = (a - 1.0) * np.log(vaf) + (b - 1.0) * np.log(1.0 - vaf) - betaln(a, b)
        return log_p

    @staticmethod
    def _estimate_phi_from_vaf(vaf: Tensor) -> float:
        """Quick moment-based estimate of Beta precision from VAFs."""
        mu = np.mean(vaf)
        var = np.var(vaf)
        mu = np.clip(mu, 0.01, 0.99)
        if var <= 0 or var >= mu * (1 - mu):
            return 50.0
        phi = mu * (1 - mu) / var - 1.0
        return float(np.clip(phi, 1.0, 1000.0))

    @staticmethod
    def _estimate_tau(log_pareto: Tensor, log_bb: Tensor) -> float:
        """Simple EM to estimate mixing weight tau."""
        tau = 0.1
        for _ in range(_EM_MAX_ITER):
            log_num = np.log(tau) + log_pareto
            log_den = np.logaddexp(log_num, np.log(1.0 - tau) + log_bb)
            resp = np.exp(log_num - log_den)
            tau_new = float(np.mean(resp))
            tau_new = np.clip(tau_new, 1e-6, 1.0 - 1e-6)
            if abs(tau_new - tau) < _EM_TOL:
                break
            tau = tau_new
        return tau
