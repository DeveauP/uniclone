"""
uniclone.emission
=================

Emission module registry.  Maps ``EmissionModel`` enum values to their
implementing classes.
"""

from __future__ import annotations

from uniclone.core.config import EmissionModel
from uniclone.emission.bb_pareto import BBParetoEmission
from uniclone.emission.beta_binomial import BetaBinomialEmission
from uniclone.emission.binomial import BinomialEmission
from uniclone.emission.dcf import DCFBBEmission

EmissionRegistry: dict[EmissionModel, type] = {
    EmissionModel.BINOMIAL: BinomialEmission,
    EmissionModel.BETA_BINOMIAL: BetaBinomialEmission,
    EmissionModel.BB_PARETO: BBParetoEmission,
    EmissionModel.DCF_BB: DCFBBEmission,
}

__all__ = [
    "EmissionRegistry",
    "BinomialEmission",
    "BetaBinomialEmission",
    "BBParetoEmission",
    "DCFBBEmission",
]
