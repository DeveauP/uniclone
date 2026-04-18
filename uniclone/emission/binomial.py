"""
uniclone.emission.binomial
==========================

Binomial emission distribution.

Mathematical form::

    log P(k | n, p) = log C(n, k) + k log(p) + (n-k) log(1-p)

where ``k`` is alt reads, ``n`` is depth, and ``p`` is the effective allele
frequency (cellularity × adj_factor, computed by the caller).

This corresponds to the QuantumClone original emission (φ = 0 limit).

Status: IMPLEMENTED — Phase 0.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from uniclone.core.backend import B
from uniclone.core.types import Tensor

if TYPE_CHECKING:
    from uniclone.core.config import CloneConfig


class BinomialEmission:
    """
    Binomial emission: log P(alt | depth, mu).

    Parameters
    ----------
    config : CloneConfig
        Configuration object (``phi`` is ignored; Binomial has no dispersion).
    """

    def __init__(self, config: CloneConfig) -> None:
        self.config = config

    def log_prob(self, alt: Tensor, depth: Tensor, mu: Tensor) -> Tensor:
        """
        Compute per-mutation log-likelihood summed over samples.

        Parameters
        ----------
        alt   : (n_mut, n_samples) int   — alternate read counts
        depth : (n_mut, n_samples) int   — total read depth
        mu    : (n_mut, n_samples) float — effective allele frequency
                (cellularity × adj_factor, pre-computed by caller)

        Returns
        -------
        (n_mut,) float — sum_{s} log P(alt_{i,s} | depth_{i,s}, mu_{i,s})

        Notes
        -----
        Uses ``scipy.special.xlogy`` which evaluates ``x * log(y)`` as 0
        when ``x == 0``, avoiding ``0 * log(0) = NaN``.
        """
        mu_clipped = B.clip(mu, 1e-10, 1.0 - 1e-10)

        log_binom_coef = (
            B.gammaln(depth + 1)
            - B.gammaln(alt + 1)
            - B.gammaln(depth - alt + 1)
        )
        log_lik = (
            log_binom_coef
            + B.xlogy(alt, mu_clipped)
            + B.xlogy(depth - alt, 1.0 - mu_clipped)
        )
        # Sum over samples axis
        return log_lik.sum(axis=-1)  # type: ignore[no-any-return]
