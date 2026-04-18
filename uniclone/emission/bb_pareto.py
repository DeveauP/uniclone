"""
uniclone.emission.bb_pareto
============================

Beta-Binomial + Pareto tail mixture emission (MOBSTER-style).

Mathematical form::

    log P(k | n, mu, phi, tau) = log[
        (1 - tau) * BetaBinomial(k; n, mu, phi)
        + tau * Pareto(k/n; alpha)
    ]

The Pareto component models the neutral tail of passenger mutations.
``tau`` is the tail mixing weight.

References
----------
- MOBSTER: Caravagna et al. (2020) *Nature Methods*

Status: IMPLEMENTED — Phase 1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from uniclone.core.backend import B
from uniclone.core.types import Tensor

if TYPE_CHECKING:
    from uniclone.core.config import CloneConfig

# MOBSTER default Pareto shape parameter
_PARETO_ALPHA = 2.0


class BBParetoEmission:
    """
    Beta-Binomial + Pareto tail mixture emission.

    Parameters
    ----------
    config : CloneConfig
        Configuration object.  ``phi`` controls BB dispersion,
        ``tail_weight`` (tau) controls the mixing weight of the
        Pareto neutral tail component.
    """

    def __init__(self, config: CloneConfig) -> None:
        self.config = config
        self.phi: float = config.phi if config.phi is not None else 100.0
        self.tau: float = config.tail_weight if config.tail_weight is not None else 0.05

    def log_prob(self, alt: Tensor, depth: Tensor, mu: Tensor) -> Tensor:
        """
        Compute per-mutation log-likelihood summed over samples.

        Parameters
        ----------
        alt   : (n_mut, n_samples) int   — alternate read counts
        depth : (n_mut, n_samples) int   — total read depth
        mu    : (n_mut, n_samples) float — effective allele frequency

        Returns
        -------
        (n_mut,) float — sum_{s} log P(alt_{i,s} | depth_{i,s}, mu_{i,s}, phi, tau)
        """
        mu_clipped = B.clip(mu, 1e-10, 1.0 - 1e-10)
        phi = self.phi
        tau = self.tau

        alpha_bb = mu_clipped * phi
        beta_bb = (1.0 - mu_clipped) * phi

        # --- Beta-Binomial component (per sample) ---
        log_binom_coef = B.gammaln(depth + 1) - B.gammaln(alt + 1) - B.gammaln(depth - alt + 1)
        log_beta_num = (
            B.gammaln(alt + alpha_bb)
            + B.gammaln(depth - alt + beta_bb)
            - B.gammaln(depth + alpha_bb + beta_bb)
        )
        log_beta_den = B.gammaln(alpha_bb) + B.gammaln(beta_bb) - B.gammaln(alpha_bb + beta_bb)
        log_bb = log_binom_coef + log_beta_num - log_beta_den

        # --- Pareto component (per sample) ---
        # f(x) = (alpha - 1) * x^{-alpha} for x = alt/depth (VAF)
        vaf = B.clip(alt / B.maximum(depth, 1.0), 1e-10, 1.0)
        log_pareto = B.log(_PARETO_ALPHA - 1.0) - _PARETO_ALPHA * B.log(vaf)

        # --- Mixture in log-space (per sample) ---
        log_mix = B.logaddexp(
            B.log(1.0 - tau) + log_bb,
            B.log(tau) + log_pareto,
        )

        return log_mix.sum(axis=-1)
