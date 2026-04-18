"""
uniclone.phylo.none
====================

No-op phylogenetic module — pure pass-through.

Used when ``phylo=PhyloMode.NONE`` (e.g. PyClone-VI, MOBSTER).
Both ``constrain`` and ``postprocess`` return their inputs unchanged.

Status: IMPLEMENTED — Phase 0 (trivial).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from uniclone.core.types import CloneResult, Tensor

if TYPE_CHECKING:
    from uniclone.core.config import CloneConfig


class NoPhylo:
    """No-op phylogenetic constraint — identity on all inputs."""

    def __init__(self, config: CloneConfig) -> None:
        self.config = config

    def constrain(self, centers: Tensor, raw_params: Tensor) -> Tensor:
        """Return ``centers`` unchanged."""
        return centers

    def postprocess(self, result: CloneResult) -> CloneResult:
        """Return ``result`` unchanged."""
        return result
