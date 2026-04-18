"""
uniclone.k_prior
================

K-prior module registry.  Maps ``KPrior`` enum values to their
implementing classes.
"""
from __future__ import annotations

from uniclone.core.config import KPrior
from uniclone.k_prior.bic import BICPrior
from uniclone.k_prior.dirichlet import DirichletPrior
from uniclone.k_prior.fixed import FixedKPrior
from uniclone.k_prior.tssb import TSSBPrior

KPriorRegistry: dict[KPrior, type] = {
    KPrior.BIC: BICPrior,
    KPrior.FIXED: FixedKPrior,
    KPrior.DIRICHLET: DirichletPrior,
    KPrior.TSSB: TSSBPrior,
}

__all__ = ["KPriorRegistry", "BICPrior", "FixedKPrior", "DirichletPrior", "TSSBPrior"]
