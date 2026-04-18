"""
uniclone.emission.dcf
======================

DCF-corrected Beta-Binomial emission (DeCiFer variable multiplicity model).

The DCF (Descendant Cell Fraction) correction accounts for variable mutation
multiplicities given the copy-number state.  The correction is applied
*upstream* by the ``MULTIPLICITY`` noise module, which adjusts the
``adj_factor`` tensor so that::

    effective_mu[i, s] = centers[k, s] * adj_factor[i, s]

already incorporates the multiplicity-dependent frequency shift.  The emission
itself is therefore a standard Beta-Binomial — the DeCiFer-specific logic
lives in the noise module, not here.

References
----------
- DeCiFer: Marass et al. (2021) *Science Advances*

Status: IMPLEMENTED — Phase 1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from uniclone.core.backend import B
from uniclone.core.types import Tensor

if TYPE_CHECKING:
    from uniclone.core.config import CloneConfig


class DCFBBEmission:
    """
    DCF-corrected Beta-Binomial emission.

    Mathematically identical to ``BetaBinomialEmission`` — the DeCiFer
    multiplicity correction enters through ``adj_factor``, not through the
    emission distribution itself.

    Parameters
    ----------
    config : CloneConfig
        Configuration object.  ``phi`` controls dispersion.
    """

    def __init__(self, config: CloneConfig) -> None:
        self.config = config
        self.phi: float = config.phi if config.phi is not None else 100.0

    def log_prob(self, alt: Tensor, depth: Tensor, mu: Tensor) -> Tensor:
        """
        Compute per-mutation log-likelihood summed over samples.

        Parameters
        ----------
        alt   : (n_mut, n_samples) int   — alternate read counts
        depth : (n_mut, n_samples) int   — total read depth
        mu    : (n_mut, n_samples) float — effective allele frequency
                (already DCF-corrected via adj_factor by caller)

        Returns
        -------
        (n_mut,) float — sum_{s} log P(alt_{i,s} | depth_{i,s}, mu_{i,s}, phi)
        """
        mu_clipped = B.clip(mu, 1e-10, 1.0 - 1e-10)
        phi = self.phi

        alpha = mu_clipped * phi
        beta = (1.0 - mu_clipped) * phi

        log_binom_coef = B.gammaln(depth + 1) - B.gammaln(alt + 1) - B.gammaln(depth - alt + 1)
        log_beta_num = (
            B.gammaln(alt + alpha) + B.gammaln(depth - alt + beta) - B.gammaln(depth + alpha + beta)
        )
        log_beta_den = B.gammaln(alpha) + B.gammaln(beta) - B.gammaln(alpha + beta)

        log_lik = log_binom_coef + log_beta_num - log_beta_den
        return log_lik.sum(axis=-1)
