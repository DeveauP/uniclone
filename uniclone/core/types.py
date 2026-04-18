"""
uniclone.core.types
===================

Shared type definitions, the CloneResult dataclass, and Protocol interfaces
for all five swappable module types.

This module has NO internal uniclone imports (except via TYPE_CHECKING) to
prevent circular dependencies.

adj_factor contract
-------------------
The ``adj_factor`` tensor encodes copy-number and purity corrections.
Before calling ``EmissionModule.log_prob``, callers (e.g. EMInference) must
compute the effective allele frequency::

    effective_mu[i, s] = centers[k, s] * adj_factor[i, s]

and pass that as the ``mu`` argument.  Emission modules are unaware of
``adj_factor``; they receive only the already-adjusted expected frequency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from uniclone.core.config import CloneConfig

# ---------------------------------------------------------------------------
# Tensor type alias
# ---------------------------------------------------------------------------

Tensor = np.ndarray
"""Alias for numpy.ndarray."""


# ---------------------------------------------------------------------------
# CloneResult
# ---------------------------------------------------------------------------


@dataclass
class CloneResult:
    """
    Output of a single inference run for a given K.

    Shapes
    ------
    centers          : (K, n_samples)  — clone cellularity per sample
    assignments      : (n_mut,)  int  — hard cluster assignment per mutation
    responsibilities : (n_mut, K) float — soft assignment probabilities
    """

    centers: Tensor
    assignments: Tensor
    responsibilities: Tensor
    log_likelihood: float
    bic: float
    K: int
    n_iter: int
    converged: bool
    config: CloneConfig | None = field(default=None, repr=False)
    tree: Any | None = field(default=None, repr=False)
    noise_mask: Tensor | None = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Module Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class EmissionModule(Protocol):
    """
    Computes per-mutation log-probabilities under a specific emission distribution.

    ``mu`` is the **effective allele frequency** (cellularity × adj_factor already
    applied by the caller).
    """

    def log_prob(self, alt: Tensor, depth: Tensor, mu: Tensor) -> Tensor:
        """
        Parameters
        ----------
        alt   : (n_mut, n_samples) int
        depth : (n_mut, n_samples) int
        mu    : (n_mut, n_samples) float  — effective allele frequency

        Returns
        -------
        (n_mut,) float  — log P(alt | depth, mu) summed over samples
        """
        ...


@runtime_checkable
class InferenceModule(Protocol):
    """Runs the inference algorithm for a fixed K and returns a CloneResult."""

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
        """
        Parameters
        ----------
        alt, depth, adj_factor : (n_mut, n_samples)
        emission   : active EmissionModule
        phylo      : active PhyloModule (may be no-op)
        centers_init : (K, n_samples) initial clone centers
        K          : number of clones for this run
        """
        ...


@runtime_checkable
class KPriorModule(Protocol):
    """Determines the K schedule, initialises centers, and selects the best result."""

    def get_k_schedule(self) -> list[int]:
        """Return the list of K values to try (e.g. [1,2,...,10] for BIC)."""
        ...

    def initialise(
        self,
        K: int,
        alt: Tensor,
        depth: Tensor,
        adj_factor: Tensor,
    ) -> Tensor:
        """
        Return initial clone centers of shape (K, n_samples).
        """
        ...

    def select(self, results: list[CloneResult]) -> CloneResult:
        """Select the best CloneResult from a list (one per K value)."""
        ...


@runtime_checkable
class PhyloModule(Protocol):
    """Imposes phylogenetic constraints during and after inference."""

    def constrain(self, centers: Tensor, raw_params: Tensor) -> Tensor:
        """
        Apply phylogenetic constraint to clone centers during M-step.
        No-op implementations return ``centers`` unchanged.
        """
        ...

    def postprocess(self, result: CloneResult) -> CloneResult:
        """Post-hoc phylogenetic processing after inference converges."""
        ...


@runtime_checkable
class NoiseModule(Protocol):
    """Pre- and post-processes mutations to handle noise/artefacts."""

    def preprocess(
        self,
        alt: Tensor,
        depth: Tensor,
        adj_factor: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Filter or transform input tensors before inference.

        Returns
        -------
        alt, depth, adj_factor : filtered/transformed copies
        mask : (n_mut,) bool — True for mutations kept in inference
        """
        ...

    def postprocess(self, result: CloneResult, mask: Tensor) -> CloneResult:
        """Re-attach noise-filtered mutations to the result."""
        ...
