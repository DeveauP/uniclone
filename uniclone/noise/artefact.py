"""
uniclone.noise.artefact
========================

CONIPHER-style artefact cluster removal.

Identifies and removes clusters whose presence/absence pattern across
samples is inconsistent with any valid clone tree (e.g. a cluster that
appears in a subset of samples not nestable with other clusters).

References
----------
- CONIPHER: Dentro et al. (2021) *Cell*
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from uniclone.core.types import CloneResult, Tensor

if TYPE_CHECKING:
    from uniclone.core.config import CloneConfig


class ArtefactNoise:
    """
    CONIPHER-style artefact cluster filter.

    Pre-processing: identity (pass-through, like NoNoise).
    Post-processing: checks each cluster's presence/absence pattern across
    samples; clusters whose pattern is not nestable with any other cluster
    are marked as artefacts and their mutations are reassigned.
    """

    def __init__(self, config: CloneConfig) -> None:
        self.config = config
        self.absence_threshold = config.artefact_absence_threshold

    def preprocess(
        self,
        alt: Tensor,
        depth: Tensor,
        adj_factor: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Identity pre-processing — all logic is post-inference."""
        mask = np.ones(alt.shape[0], dtype=bool)
        return alt, depth, adj_factor, mask

    def postprocess(self, result: CloneResult, mask: Tensor) -> CloneResult:
        result.noise_mask = mask
        K = result.K
        n_samples = result.centers.shape[1]

        # Single-sample: no cross-sample check possible
        if n_samples < 2:
            return result

        # Determine presence/absence pattern for each cluster
        # present = CCF > threshold
        presence = result.centers > self.absence_threshold  # (K, n_samples) bool

        # Check nestability: cluster i is nestable if there exists some other
        # cluster j such that presence[i] is a subset of presence[j] or
        # presence[j] is a subset of presence[i].
        artefact_clusters = set()
        for i in range(K):
            is_nestable = False
            for j in range(K):
                if i == j:
                    continue
                # i subset of j, or j subset of i
                if np.all(presence[i] <= presence[j]) or np.all(presence[j] <= presence[i]):
                    is_nestable = True
                    break
            if not is_nestable:
                artefact_clusters.add(i)

        if not artefact_clusters:
            return result

        # Reassign artefact cluster mutations to nearest valid cluster
        valid_clusters = [k for k in range(K) if k not in artefact_clusters]
        if not valid_clusters:
            # All clusters are artefacts — keep everything as-is
            return result

        assignments = result.assignments.copy()
        responsibilities = result.responsibilities.copy()
        centers = result.centers.copy()

        for ac in artefact_clusters:
            ac_muts = assignments == ac
            if not np.any(ac_muts):
                continue
            # Pick nearest valid cluster by highest responsibility
            valid_resp = responsibilities[ac_muts][:, valid_clusters]
            best_idx = np.argmax(valid_resp, axis=1)
            assignments[ac_muts] = np.array(valid_clusters)[best_idx]
            # Zero out artefact cluster centers
            centers[ac] = 0.0

        result.assignments = assignments
        result.responsibilities = responsibilities
        result.centers = centers

        return result
