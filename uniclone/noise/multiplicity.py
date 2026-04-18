"""
uniclone.noise.multiplicity
============================

DeCiFer-style variable mutation multiplicity enumeration.

Enumerates all possible mutation multiplicities (copy numbers) consistent
with the observed copy number state and chooses the most likely assignment,
adjusting the expected allele frequency accordingly.

References
----------
- DeCiFer: Marass et al. (2021) *Science Advances*
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from uniclone.core.types import CloneResult, Tensor

if TYPE_CHECKING:
    from uniclone.core.config import CloneConfig


class MultiplicityNoise:
    """
    DeCiFer-style multiplicity enumeration.

    If copy-number state is provided via ``set_cn_state``, pre-processing
    enumerates valid multiplicities for each mutation and picks the one
    whose implied CCF is closest to (but not exceeding) 1.0, then updates
    ``adj_factor`` accordingly.

    If no CN state is set, acts as identity (pass-through).
    """

    def __init__(self, config: CloneConfig) -> None:
        self.config = config
        self.purity = config.purity
        self._total_cn: Tensor | None = None
        self._minor_cn: Tensor | None = None

    def set_cn_state(
        self, total_cn: Tensor, minor_cn: Tensor | None = None
    ) -> None:
        """
        Store per-mutation copy number data.

        Parameters
        ----------
        total_cn : (n_mut,) or (n_mut, n_samples) int
            Total copy number per mutation (per sample).
        minor_cn : (n_mut,) or (n_mut, n_samples) int or None
            Minor allele copy number. Not used currently but stored for
            future extensions.
        """
        self._total_cn = np.asarray(total_cn, dtype=np.float64)
        if minor_cn is not None:
            self._minor_cn = np.asarray(minor_cn, dtype=np.float64)

    def preprocess(
        self,
        alt: Tensor,
        depth: Tensor,
        adj_factor: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        n_mut = alt.shape[0]
        mask = np.ones(n_mut, dtype=bool)

        if self._total_cn is None:
            return alt, depth, adj_factor, mask

        vaf = alt / np.maximum(depth, 1.0)
        total_cn = self._total_cn
        purity = self.purity

        # Ensure 2-D
        if total_cn.ndim == 1:
            total_cn = total_cn[:, np.newaxis]
            total_cn = np.broadcast_to(total_cn, alt.shape).copy()

        new_adj = adj_factor.copy()

        for i in range(n_mut):
            for s in range(alt.shape[1]):
                cn = total_cn[i, s]
                if cn < 1:
                    continue
                best_m_adj = 1.0
                best_dist = np.inf
                for m in range(1, int(cn) + 1):
                    adj_m = m / (cn * purity + 2.0 * (1.0 - purity))
                    if adj_m <= 0:
                        continue
                    implied_ccf = vaf[i, s] / adj_m
                    if implied_ccf > 1.0:
                        dist = implied_ccf - 1.0
                    else:
                        dist = 1.0 - implied_ccf
                    if dist < best_dist:
                        best_dist = dist
                        best_m_adj = adj_m
                new_adj[i, s] = best_m_adj

        return alt, depth, new_adj, mask

    def postprocess(self, result: CloneResult, mask: Tensor) -> CloneResult:
        result.noise_mask = mask
        return result
