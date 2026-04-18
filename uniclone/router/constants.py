"""
uniclone.router.constants
==========================

Shared constants for the MetaRouter: DREAM subchallenges, feature names,
and configuration names.
"""

from __future__ import annotations

from enum import Enum, auto


class SubChallenge(Enum):
    """DREAM SMC-Het subchallenges."""

    SC1A = auto()  # Purity estimation
    SC1B = auto()  # CCF estimation per mutation
    SC2A = auto()  # Number of clones (K)
    SC2B = auto()  # Cluster assignment
    SC3 = auto()  # Phylogenetic tree

    def __str__(self) -> str:
        return self.name


SUBCHALLENGES: list[SubChallenge] = list(SubChallenge)

FEATURE_NAMES: list[str] = [
    # Depth statistics (3)
    "depth_median",
    "depth_cv",
    "frac_low_depth",
    # Copy-number landscape (6)
    "frac_cna",
    "wgd",
    "ploidy",
    "n_cn_states",
    "frac_loh",
    "subclonal_cn",
    # VAF/CCF shape (6)
    "skewness",
    "kurtosis",
    "n_peaks",
    "clonal_peak_width",
    "has_pareto_tail",
    "purity_estimate",
    # Data modality (4)
    "n_samples",
    "is_longitudinal",
    "n_mutations",
    "sequencing_type",
    # Derived (2)
    "nrpcc",
    "mappability",
]

N_FEATURES: int = len(FEATURE_NAMES)

# The 12 named configs that the router can select from
CONFIG_NAMES: list[str] = [
    "quantumclone_v1",
    "pyclone_vi",
    "mobster",
    "decifer_style",
    "conipher_style",
    "phyloclone_style",
    "calder_style",
    "wes_clinical",
    "wgs_cohort",
    "wgs_cohort_phylo",
    "multiregion_phylo",
    "longitudinal_clinical",
]

N_CONFIGS: int = len(CONFIG_NAMES)

# Configs excluded from training by default (MCMC-based, prohibitive runtime).
DEFAULT_EXCLUDE_CONFIGS: frozenset[str] = frozenset({"phyloclone_style"})
