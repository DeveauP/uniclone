"""
uniclone.inference.hybrid
==========================

Hybrid EM -> MFVI inference engine (new in UniClone).

Warms up with EM to find a good initial point, then refines with MFVI to
obtain a proper posterior approximation and automatic cluster pruning.

Status: IMPLEMENTED — Phase 2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from uniclone.core.types import CloneResult, EmissionModule, PhyloModule, Tensor
from uniclone.inference.em import EMInference
from uniclone.inference.mfvi import MFVIInference

if TYPE_CHECKING:
    from uniclone.core.config import CloneConfig


class HybridInference:
    """
    Hybrid EM -> MFVI inference engine for fixed K.

    Parameters
    ----------
    config : CloneConfig
    n_em : int
        Number of EM warm-up iterations (default 20).
    max_iter : int
        Maximum MFVI iterations after warm-up (default 200).
    tol : float
        Convergence tolerance (default 1e-6).
    alpha0 : float
        Dirichlet concentration prior per component (default 1.0).
    """

    def __init__(
        self,
        config: CloneConfig,
        n_em: int = 20,
        max_iter: int = 200,
        tol: float = 1e-6,
        alpha0: float = 1.0,
    ) -> None:
        self.config = config
        self.n_em = n_em
        self.max_iter = max_iter
        self.tol = tol
        self.alpha0 = alpha0

    def run(
        self,
        alt: Tensor,
        depth: Tensor,
        adj_factor: Tensor,
        emission: EmissionModule,
        phylo: PhyloModule,
        centers_init: Tensor,
        K: int,
    ) -> CloneResult:
        """Run EM warm-up followed by MFVI refinement."""
        # Phase 1: EM warm-up
        em = EMInference(self.config, max_iter=self.n_em, tol=self.tol)
        em_result = em.run(alt, depth, adj_factor, emission, phylo, centers_init, K)

        # Phase 2: MFVI refinement with warm centers
        mfvi = MFVIInference(self.config, max_iter=self.max_iter, tol=self.tol, alpha0=self.alpha0)
        mfvi_result = mfvi.run(alt, depth, adj_factor, emission, phylo, em_result.centers, K)

        # Combined iteration count
        mfvi_result.n_iter = self.n_em + mfvi_result.n_iter

        return mfvi_result
