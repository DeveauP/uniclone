"""
uniclone.core.config
====================

CloneConfig dataclass — the single object that controls every axis of the
unified model — together with all constituent Enums and the ``CONFIGS`` dict
of named presets.

Every ``CloneConfig`` instance is validated on construction via
``__post_init__``, so invalid combinations are caught immediately.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

# ---------------------------------------------------------------------------
# Axis enumerations
# ---------------------------------------------------------------------------


class EmissionModel(Enum):
    BINOMIAL = auto()       # φ = 0 (QuantumClone original)
    BETA_BINOMIAL = auto()  # φ estimated or fixed (PyClone, PyClone-VI)
    BB_PARETO = auto()      # + neutral Pareto tail (MOBSTER)
    DCF_BB = auto()         # variable multiplicity correction (DeCiFer)


class InferenceEngine(Enum):
    EM = auto()      # MAP / expectation-maximisation, fast, local-optima risk
    MFVI = auto()    # mean-field variational inference, ELBO, auto-K
    MCMC = auto()    # full posterior via PyMC (optional heavy dependency)
    HYBRID = auto()  # EM warm-start → MFVI refinement (new in UniClone)


class KPrior(Enum):
    BIC = auto()        # explicit K-range sweep, select by BIC
    DIRICHLET = auto()  # Dirichlet concentration α, automatic K
    TSSB = auto()       # tree-structured stick-breaking (PhyClone-style)
    FIXED = auto()      # K specified by user via ``n_clones``


class PhyloMode(Enum):
    NONE = auto()        # clustering only, no tree
    POST_HOC = auto()    # is_included nesting matrix after convergence
    CONSTRAINED = auto() # sigmoid reparametrisation in M-step (plan v1)
    JOINT_MCMC = auto()  # TSSB prior over trees (PhyClone-style)
    LONGITUDINAL = auto()# time-ordering ILP (CALDER-style)
    PAIRWISE = auto()    # pairs tensor (Pairtree-style)


class NoiseModel(Enum):
    NONE = auto()          # no noise correction
    TAIL_FILTER = auto()   # Pareto tail detection + mutation removal (MOBSTER)
    ARTEFACT = auto()      # cluster consistency check (CONIPHER)
    MULTIPLICITY = auto()  # variable multiplicity enumeration (DeCiFer)


# ---------------------------------------------------------------------------
# CloneConfig
# ---------------------------------------------------------------------------


@dataclass
class CloneConfig:
    """
    Configuration object controlling all five axes of the unified model.

    Parameters
    ----------
    emission     : which emission distribution to use
    inference    : which inference algorithm to use
    k_prior      : how to determine / select the number of clones K
    phylo        : phylogenetic constraint mode
    noise        : noise / artefact correction mode
    phi          : Beta-Binomial precision parameter; ``None`` → estimate
    tail_weight  : Pareto tail mixing weight; ``None`` → estimate (BB_PARETO)
    n_clones     : required when ``k_prior=FIXED``
    n_samples    : number of tumour samples (timepoints / regions)
    longitudinal : whether samples represent timepoints (enables LONGITUDINAL)
    depth_median : median sequencing depth (used for depth-adaptive settings)
    """

    emission: EmissionModel = EmissionModel.BETA_BINOMIAL
    inference: InferenceEngine = InferenceEngine.HYBRID
    k_prior: KPrior = KPrior.DIRICHLET
    phylo: PhyloMode = PhyloMode.CONSTRAINED
    noise: NoiseModel = NoiseModel.ARTEFACT
    phi: float | None = None
    tail_weight: float | None = None
    n_clones: int | None = None
    n_samples: int = 1
    longitudinal: bool = False
    depth_median: float = 100.0
    tail_threshold: float = 0.5
    artefact_absence_threshold: float = 0.05
    purity: float = 1.0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Raise ValueError for any incompatible combination of settings."""
        if self.k_prior is KPrior.FIXED and self.n_clones is None:
            raise ValueError(
                "n_clones must be set when k_prior=FIXED"
            )
        if self.phylo is PhyloMode.LONGITUDINAL and not self.longitudinal:
            raise ValueError(
                "PhyloMode.LONGITUDINAL requires longitudinal=True"
            )
        if self.phylo is PhyloMode.JOINT_MCMC and self.inference not in (
            InferenceEngine.MCMC,
            InferenceEngine.HYBRID,
        ):
            raise ValueError(
                "PhyloMode.JOINT_MCMC requires InferenceEngine.MCMC or HYBRID"
            )
        if self.phi is not None and self.phi <= 0:
            raise ValueError(f"phi must be positive, got {self.phi}")
        if self.tail_weight is not None and not (0.0 < self.tail_weight < 1.0):
            raise ValueError(
                f"tail_weight must be in (0, 1), got {self.tail_weight}"
            )
        if self.n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {self.n_samples}")
        if self.depth_median <= 0:
            raise ValueError(
                f"depth_median must be positive, got {self.depth_median}"
            )
        if not (0.0 < self.tail_threshold <= 1.0):
            raise ValueError(
                f"tail_threshold must be in (0, 1], got {self.tail_threshold}"
            )
        if not (0.0 <= self.artefact_absence_threshold <= 1.0):
            raise ValueError(
                f"artefact_absence_threshold must be in [0, 1], "
                f"got {self.artefact_absence_threshold}"
            )
        if not (0.0 < self.purity <= 1.0):
            raise ValueError(
                f"purity must be in (0, 1], got {self.purity}"
            )

    def __repr__(self) -> str:
        return (
            f"CloneConfig("
            f"emission={self.emission.name}, "
            f"inference={self.inference.name}, "
            f"k_prior={self.k_prior.name}, "
            f"phylo={self.phylo.name}, "
            f"noise={self.noise.name})"
        )


# ---------------------------------------------------------------------------
# Named presets
# ---------------------------------------------------------------------------

CONFIGS: dict[str, CloneConfig] = {
    # ------------------------------------------------------------------ #
    # Reproductions of existing tools                                     #
    # ------------------------------------------------------------------ #
    "quantumclone_v1": CloneConfig(
        emission=EmissionModel.BINOMIAL,
        inference=InferenceEngine.EM,
        k_prior=KPrior.BIC,
        phylo=PhyloMode.POST_HOC,
        noise=NoiseModel.NONE,
    ),
    "pyclone_vi": CloneConfig(
        emission=EmissionModel.BETA_BINOMIAL,
        inference=InferenceEngine.MFVI,
        k_prior=KPrior.DIRICHLET,
        phylo=PhyloMode.NONE,
        noise=NoiseModel.NONE,
        phi=50.0,
    ),
    "mobster": CloneConfig(
        emission=EmissionModel.BB_PARETO,
        inference=InferenceEngine.EM,
        k_prior=KPrior.BIC,
        phylo=PhyloMode.NONE,
        noise=NoiseModel.TAIL_FILTER,
    ),
    "decifer_style": CloneConfig(
        emission=EmissionModel.DCF_BB,
        inference=InferenceEngine.EM,
        k_prior=KPrior.BIC,
        phylo=PhyloMode.NONE,
        noise=NoiseModel.MULTIPLICITY,
    ),
    "conipher_style": CloneConfig(
        emission=EmissionModel.BETA_BINOMIAL,
        inference=InferenceEngine.MFVI,
        k_prior=KPrior.DIRICHLET,
        phylo=PhyloMode.POST_HOC,
        noise=NoiseModel.ARTEFACT,
    ),
    "phyloclone_style": CloneConfig(
        emission=EmissionModel.BETA_BINOMIAL,
        inference=InferenceEngine.MCMC,
        k_prior=KPrior.TSSB,
        phylo=PhyloMode.JOINT_MCMC,
        noise=NoiseModel.NONE,
    ),
    "calder_style": CloneConfig(
        emission=EmissionModel.BETA_BINOMIAL,
        inference=InferenceEngine.EM,
        k_prior=KPrior.FIXED,
        phylo=PhyloMode.LONGITUDINAL,
        noise=NoiseModel.NONE,
        longitudinal=True,
        n_clones=3,
    ),
    # ------------------------------------------------------------------ #
    # New configurations not available in any existing tool               #
    # ------------------------------------------------------------------ #
    "wes_clinical": CloneConfig(
        emission=EmissionModel.BETA_BINOMIAL,
        inference=InferenceEngine.HYBRID,
        k_prior=KPrior.DIRICHLET,
        phylo=PhyloMode.CONSTRAINED,
        noise=NoiseModel.ARTEFACT,
        phi=None,
    ),
    "wgs_cohort": CloneConfig(
        emission=EmissionModel.BB_PARETO,
        inference=InferenceEngine.MFVI,
        k_prior=KPrior.DIRICHLET,
        phylo=PhyloMode.NONE,
        noise=NoiseModel.TAIL_FILTER,
        phi=100.0,
    ),
    "wgs_cohort_phylo": CloneConfig(
        emission=EmissionModel.BB_PARETO,
        inference=InferenceEngine.HYBRID,
        k_prior=KPrior.DIRICHLET,
        phylo=PhyloMode.CONSTRAINED,
        noise=NoiseModel.TAIL_FILTER,
        phi=100.0,
    ),
    "multiregion_phylo": CloneConfig(
        emission=EmissionModel.DCF_BB,
        inference=InferenceEngine.HYBRID,
        k_prior=KPrior.DIRICHLET,
        phylo=PhyloMode.PAIRWISE,
        noise=NoiseModel.MULTIPLICITY,
    ),
    "longitudinal_clinical": CloneConfig(
        emission=EmissionModel.BETA_BINOMIAL,
        inference=InferenceEngine.HYBRID,
        k_prior=KPrior.BIC,
        phylo=PhyloMode.LONGITUDINAL,
        noise=NoiseModel.ARTEFACT,
        longitudinal=True,
    ),
}
