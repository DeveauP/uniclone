"""
uniclone.noise.none
====================

No-op noise module — all mutations are passed through unmodified.

Used when ``noise=NoiseModel.NONE`` (e.g. QuantumClone, PyClone, Pairtree).
Both ``preprocess`` and ``postprocess`` are identity operations.

Status: IMPLEMENTED — Phase 0 (trivial).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from uniclone.core.types import CloneResult, Tensor

if TYPE_CHECKING:
    from uniclone.core.config import CloneConfig


class NoNoise:
    """No-op noise model — identity on all inputs."""

    def __init__(self, config: CloneConfig) -> None:
        self.config = config

    def preprocess(
        self,
        alt: Tensor,
        depth: Tensor,
        adj_factor: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return inputs unchanged; mask is all-True."""
        mask = np.ones(alt.shape[0], dtype=bool)
        return alt, depth, adj_factor, mask

    def postprocess(self, result: CloneResult, mask: Tensor) -> CloneResult:
        """Return ``result`` unchanged."""
        return result
