"""
uniclone.phylo
==============

Phylogenetic module registry.  Maps ``PhyloMode`` enum values to their
implementing classes.
"""
from __future__ import annotations

from uniclone.core.config import PhyloMode
from uniclone.phylo.constrained import ConstrainedPhylo
from uniclone.phylo.joint_mcmc import JointMCMCPhylo
from uniclone.phylo.longitudinal import LongitudinalPhylo
from uniclone.phylo.none import NoPhylo
from uniclone.phylo.pairwise import PairwisePhylo
from uniclone.phylo.post_hoc import PostHocPhylo

PhyloRegistry: dict[PhyloMode, type] = {
    PhyloMode.NONE: NoPhylo,
    PhyloMode.POST_HOC: PostHocPhylo,
    PhyloMode.CONSTRAINED: ConstrainedPhylo,
    PhyloMode.JOINT_MCMC: JointMCMCPhylo,
    PhyloMode.LONGITUDINAL: LongitudinalPhylo,
    PhyloMode.PAIRWISE: PairwisePhylo,
}

__all__ = [
    "PhyloRegistry",
    "NoPhylo",
    "PostHocPhylo",
    "ConstrainedPhylo",
    "LongitudinalPhylo",
    "PairwisePhylo",
    "JointMCMCPhylo",
]
