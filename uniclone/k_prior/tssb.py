"""
uniclone.k_prior.tssb
======================

Tree-structured stick-breaking (TSSB) prior (PhyClone-style).

Implements the flat (non-tree) truncated stick-breaking version.
Full tree-topology integration lives in JointMCMCPhylo.

References
----------
- PhyloWGS: Deshwar et al. (2015) *Genome Biology*
- PhyClone: McPherson et al. (2021) *Nature Methods*

Status: IMPLEMENTED (experimental).
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from uniclone.core.types import CloneResult, Tensor
from uniclone.k_prior.dirichlet import _prune_and_merge

if TYPE_CHECKING:
    from uniclone.core.config import CloneConfig


class TSSBPrior:
    """
    Truncated stick-breaking prior — flat (non-tree) version.

    Uses truncated stick-breaking weights for initialisation and
    quantile-based centers.  Tree-topology integration is handled
    by ``JointMCMCPhylo`` when using ``PhyloMode.JOINT_MCMC``.

    Parameters
    ----------
    config : CloneConfig
    k_max : int
        Truncation level (default 15).
    alpha : float
        Concentration parameter for stick-breaking (default 1.0).
    min_cluster_weight : float
        Minimum fraction of mutations to keep a cluster (default 0.005).
    """

    def __init__(
        self,
        config: CloneConfig,
        k_max: int = 15,
        alpha: float = 1.0,
        min_cluster_weight: float = 0.005,
    ) -> None:
        warnings.warn(
            "TSSBPrior is experimental. Uses flat truncated stick-breaking; "
            "pair with JointMCMCPhylo for tree-topology integration.",
            UserWarning,
            stacklevel=1,
        )
        self.config = config
        self.k_max: int = config.n_clones if config.n_clones is not None else k_max
        self.alpha = alpha
        self.min_cluster_weight = min_cluster_weight

    def get_k_schedule(self) -> list[int]:
        """Single run at K_max; pruning happens in ``select()``."""
        return [self.k_max]

    def initialise(self, K: int, alt: Tensor, depth: Tensor, adj_factor: Tensor) -> Tensor:
        """
        Stick-breaking weight initialisation + quantile-based centers.

        The stick-breaking weights are used only to inform the initialisation;
        the inference engine handles the actual mixing weights.
        """
        # Stick-breaking weights (stored for reference, not used in centers)
        nu = np.full(K, 1.0 / (1.0 + self.alpha))
        weights = np.empty(K)
        cumulative_stick = 1.0
        for k in range(K):
            weights[k] = nu[k] * cumulative_stick
            cumulative_stick *= (1.0 - nu[k])
        # Normalise for finite truncation
        weights /= weights.sum()
        self._init_weights = weights

        # Quantile-based centers (same pattern as BIC/Dirichlet)
        vaf = alt.astype(float) / np.maximum(depth.astype(float), 1)
        quantiles = np.linspace(0.0, 1.0, K + 2)[1:-1]
        centers = np.quantile(vaf, quantiles, axis=0)
        return np.clip(centers, 1e-6, 1.0 - 1e-6)

    def select(self, results: list[CloneResult]) -> CloneResult:
        """Prune and merge clusters (same logic as Dirichlet, lower threshold)."""
        result = results[0]
        return _prune_and_merge(result, self.min_cluster_weight)
