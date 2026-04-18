"""
uniclone.core.model
====================

GenerativeModel — the central object that assembles five swappable modules
(emission, inference, k_prior, phylo, noise) from a ``CloneConfig`` and
exposes a unified ``fit()`` API.

The six-step pipeline
----------------------

1. **Noise pre-processing** — filter artefacts, Pareto tail, or enumerate
   multiplicities before inference.
2. **K schedule** — ask the k_prior module which K values to try.
3. **Inference loop** — for each K, initialise centers and run inference.
4. **Model selection** — ask the k_prior module to pick the best result.
5. **Phylo post-processing** — build the clone tree (post-hoc or joint).
6. **Noise post-processing** — re-attach filtered mutations to the result.
"""

from __future__ import annotations

import numpy as np

from uniclone.core.backend import B
from uniclone.core.config import CloneConfig
from uniclone.core.types import CloneResult, Tensor
from uniclone.emission import EmissionRegistry
from uniclone.inference import InferenceRegistry
from uniclone.k_prior import KPriorRegistry
from uniclone.noise import NoiseRegistry
from uniclone.phylo import PhyloRegistry


class GenerativeModel:
    """
    Unified probabilistic clonal reconstruction model.

    Assembles emission, inference, k_prior, phylo, and noise modules from a
    ``CloneConfig`` at construction time and exposes a single ``fit()`` method.

    Parameters
    ----------
    config : CloneConfig
        Configuration controlling all five axes of the model.

    Examples
    --------
    >>> from uniclone import GenerativeModel, CONFIGS
    >>> model = GenerativeModel(CONFIGS["quantumclone_v1"])
    >>> result = model.fit(alt, depth, adj_factor)
    """

    def __init__(self, config: CloneConfig) -> None:
        self.config = config
        self.emission = EmissionRegistry[config.emission](config)
        self.inference = InferenceRegistry[config.inference](config)
        self.k_prior = KPriorRegistry[config.k_prior](config)
        self.phylo = PhyloRegistry[config.phylo](config)
        self.noise = NoiseRegistry[config.noise](config)

    def fit(
        self,
        alt: Tensor,
        depth: Tensor,
        adj_factor: Tensor | None = None,
        cn_state: dict | None = None,
    ) -> CloneResult:
        """
        Run the full six-step clonal reconstruction pipeline.

        Parameters
        ----------
        alt : (n_mut, n_samples) int
            Alternate read counts.
        depth : (n_mut, n_samples) int
            Total read depth.
        adj_factor : (n_mut, n_samples) float or None
            Copy-number and purity adjustment factor.  If ``None``, defaults
            to an all-ones tensor (no correction).

        Returns
        -------
        CloneResult
            Best clonal reconstruction result.
        """
        alt = B.asarray(alt, dtype=np.float64)
        depth = B.asarray(depth, dtype=np.float64)

        if adj_factor is None:
            adj_factor = np.ones_like(B.to_numpy(alt), dtype=np.float64)
            adj_factor = B.asarray(adj_factor)
        else:
            adj_factor = B.asarray(adj_factor, dtype=np.float64)

        # Ensure 2-D inputs
        if alt.ndim == 1:
            alt = alt[:, np.newaxis]
            depth = depth[:, np.newaxis]
            adj_factor = adj_factor[:, np.newaxis]

        # Forward CN state to noise module if supported
        if cn_state is not None and hasattr(self.noise, "set_cn_state"):
            self.noise.set_cn_state(**cn_state)

        # ------------------------------------------------------------------ #
        # Step 1: Noise pre-processing                                        #
        # ------------------------------------------------------------------ #
        alt_f, depth_f, adj_f, noise_mask = self.noise.preprocess(alt, depth, adj_factor)

        # ------------------------------------------------------------------ #
        # Step 2: K schedule                                                  #
        # ------------------------------------------------------------------ #
        k_schedule = self.k_prior.get_k_schedule()

        # ------------------------------------------------------------------ #
        # Step 3: Inference loop                                              #
        # ------------------------------------------------------------------ #
        results: list[CloneResult] = []
        for K in k_schedule:
            centers_init = self.k_prior.initialise(K, alt_f, depth_f, adj_f)
            result = self.inference.run(
                alt_f,
                depth_f,
                adj_f,
                emission=self.emission,
                phylo=self.phylo,
                centers_init=centers_init,
                K=K,
            )
            results.append(result)

        # ------------------------------------------------------------------ #
        # Step 4: Model selection                                             #
        # ------------------------------------------------------------------ #
        best = self.k_prior.select(results)

        # ------------------------------------------------------------------ #
        # Step 5: Phylo post-processing                                       #
        # ------------------------------------------------------------------ #
        best = self.phylo.postprocess(best)

        # ------------------------------------------------------------------ #
        # Step 6: Noise post-processing (re-attach filtered mutations)        #
        # ------------------------------------------------------------------ #
        best = self.noise.postprocess(best, noise_mask)

        # Convert results back to numpy
        best.centers = B.to_numpy(best.centers)
        best.assignments = B.to_numpy(best.assignments)
        best.responsibilities = B.to_numpy(best.responsibilities)

        # Attach config for reference
        best.config = self.config

        return best

    def __repr__(self) -> str:
        return f"GenerativeModel({self.config!r})"
