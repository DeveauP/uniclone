"""
uniclone.emission.beta_binomial
================================

Beta-Binomial emission distribution.

Mathematical form::

    log P(k | n, mu, phi) = log C(n, k)
                            + log B(k + mu*phi, n-k + (1-mu)*phi)
                            - log B(mu*phi, (1-mu)*phi)

where ``B`` is the Beta function, ``mu`` is the mean allele frequency, and
``phi`` is the precision parameter (phi -> inf recovers Binomial).

References
----------
- PyClone: Roth et al. (2014) *Nature Methods*
- PyClone-VI: Gillis et al. (2020) *Genome Biology*

Status: IMPLEMENTED — Phase 1.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from uniclone.core.backend import B
from uniclone.core.types import Tensor

if TYPE_CHECKING:
    from uniclone.core.config import CloneConfig


class BetaBinomialEmission:
    """
    Beta-Binomial emission: log P(alt | depth, mu, phi).

    Parameters
    ----------
    config : CloneConfig
        Configuration object.  ``phi`` controls dispersion:
        ``phi = None`` uses a default of 100.0 (estimation is an
        inference-module concern); larger phi -> less overdispersion.
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

        Returns
        -------
        (n_mut,) float — sum_{s} log P(alt_{i,s} | depth_{i,s}, mu_{i,s}, phi)
        """
        mu_clipped = B.clip(mu, 1e-10, 1.0 - 1e-10)
        phi = self.phi

        alpha = mu_clipped * phi
        beta = (1.0 - mu_clipped) * phi

        # log C(n, k) = log(n!) - log(k!) - log((n-k)!)
        log_binom_coef = (
            B.gammaln(depth + 1)
            - B.gammaln(alt + 1)
            - B.gammaln(depth - alt + 1)
        )

        # log B(k + alpha, n-k + beta) - log B(alpha, beta)
        # where log B(a, b) = gammaln(a) + gammaln(b) - gammaln(a + b)
        log_beta_num = (
            B.gammaln(alt + alpha)
            + B.gammaln(depth - alt + beta)
            - B.gammaln(depth + alpha + beta)
        )
        log_beta_den = (
            B.gammaln(alpha)
            + B.gammaln(beta)
            - B.gammaln(alpha + beta)
        )

        log_lik = log_binom_coef + log_beta_num - log_beta_den
        return log_lik.sum(axis=-1)  # type: ignore[no-any-return]
