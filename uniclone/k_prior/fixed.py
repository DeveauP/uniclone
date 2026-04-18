"""
uniclone.k_prior.fixed
=======================

Fixed-K prior: the user specifies exactly how many clones to fit.

No model selection is performed — the single result for the user-specified
K is returned directly.  Requires ``CloneConfig.n_clones`` to be set.

Status: IMPLEMENTED — Phase 0 (minimal).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from uniclone.core.types import CloneResult, Tensor

if TYPE_CHECKING:
    from uniclone.core.config import CloneConfig


class FixedKPrior:
    """
    Fixed K prior: no sweep, no model selection.

    Parameters
    ----------
    config : CloneConfig
        Must have ``n_clones`` set (enforced by ``CloneConfig.validate``).
    """

    def __init__(self, config: CloneConfig) -> None:
        if config.n_clones is None:
            raise ValueError("n_clones must be set when k_prior=KPrior.FIXED")
        self.config = config
        self.K: int = config.n_clones

    def get_k_schedule(self) -> list[int]:
        return [self.K]

    def initialise(self, K: int, alt: Tensor, depth: Tensor, adj_factor: Tensor) -> Tensor:
        """Quantile-based initialisation (same as BICPrior)."""
        vaf = alt.astype(float) / np.maximum(depth.astype(float), 1)
        quantiles = np.linspace(0.0, 1.0, K + 2)[1:-1]
        centers = np.quantile(vaf, quantiles, axis=0)
        return np.clip(centers, 1e-6, 1.0 - 1e-6)

    def select(self, results: list[CloneResult]) -> CloneResult:
        """Return the single result (K is fixed, so len(results) == 1)."""
        return results[0]
