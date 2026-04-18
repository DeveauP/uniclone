"""
uniclone.k_prior.bic
=====================

BIC-based K selection: sweep over a range of K values, run inference for
each, and select the model with the lowest BIC.

BIC convention (lower = better)::

    BIC = -2 * log_likelihood + n_params * log(n_mut)

where ``n_params = K * n_samples`` (one center coordinate per clone per sample).

This corresponds to the QuantumClone / FastClone K-selection approach.

Status: IMPLEMENTED — Phase 0.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from uniclone.core.types import CloneResult, Tensor

if TYPE_CHECKING:
    from uniclone.core.config import CloneConfig


class BICPrior:
    """
    BIC-based K prior: sweep over ``nclone_range`` and pick the best K.

    Parameters
    ----------
    config : CloneConfig
    nclone_range : list[int]
        K values to try.  Defaults to ``range(1, 11)``.
    """

    def __init__(self, config: CloneConfig, nclone_range: list[int] | None = None) -> None:
        self.config = config
        self.nclone_range: list[int] = nclone_range if nclone_range is not None else list(range(1, 11))

    def get_k_schedule(self) -> list[int]:
        """Return the K values to sweep over."""
        return self.nclone_range

    def initialise(self, K: int, alt: Tensor, depth: Tensor, adj_factor: Tensor) -> Tensor:
        """
        Initialise K clone centers using quantile spacing on the raw VAF.

        Parameters
        ----------
        K : int
        alt, depth, adj_factor : (n_mut, n_samples)

        Returns
        -------
        (K, n_samples) float — initial clone centers
        """
        vaf = alt.astype(float) / np.maximum(depth.astype(float), 1)
        quantiles = np.linspace(0.0, 1.0, K + 2)[1:-1]  # K interior quantiles
        centers = np.quantile(vaf, quantiles, axis=0)    # (K, n_samples)
        return np.clip(centers, 1e-6, 1.0 - 1e-6)

    def select(self, results: list[CloneResult]) -> CloneResult:
        """Return the result with the lowest BIC (lower = better)."""
        return min(results, key=lambda r: r.bic)
