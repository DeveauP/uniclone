"""
uniclone.uniclone
==================

UniClone Level 1 API — fully automatic clonal reconstruction.

Extracts meta-features, routes to the best configuration via the MetaRouter,
and runs the selected GenerativeModel pipeline.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from uniclone.core.config import CONFIGS
from uniclone.core.model import GenerativeModel
from uniclone.core.types import CloneResult
from uniclone.router.constants import SUBCHALLENGES, SubChallenge


class UniClone:
    """
    Fully automatic clonal reconstruction.

    Extracts meta-features from input data, selects the best configuration
    via the MetaRouter, and runs GenerativeModel.fit().

    Parameters
    ----------
    router : MetaRouter instance, or None for a fresh (untrained) router.
    subchallenge : target subchallenge for config selection (default SC2B).
    device : torch device for the router.
    verbose : whether to print routing decisions.

    Examples
    --------
    >>> from uniclone import UniClone
    >>> uc = UniClone()
    >>> result = uc.fit(alt, depth, adj_factor)
    """

    def __init__(
        self,
        router: Any | None = None,
        subchallenge: SubChallenge = SubChallenge.SC2B,
        device: str = "cpu",
        verbose: bool = True,
    ) -> None:
        if router is not None:
            self._router = router
        else:
            from uniclone.router.router import MetaRouter

            self._router = MetaRouter(device=device)

        self._subchallenge = subchallenge
        self._verbose = verbose

    def fit(
        self,
        alt: np.ndarray,
        depth: np.ndarray,
        adj_factor: np.ndarray | None = None,
        vcf_df: Any | None = None,
        cn_df: Any | None = None,
    ) -> CloneResult:
        """
        Run fully automatic clonal reconstruction.

        1. Extract meta-features from input data
        2. Route to best configuration via MetaRouter
        3. Run GenerativeModel.fit() with selected config

        Parameters
        ----------
        alt : (n_mut, n_samples) int — alternate read counts
        depth : (n_mut, n_samples) int — total read depths
        adj_factor : (n_mut, n_samples) float or None — CN/purity adjustment
        vcf_df : optional VCF dataframe for extra features
        cn_df : optional copy-number dataframe

        Returns
        -------
        CloneResult from the selected configuration.
        """
        from uniclone.router.meta_features import extract_meta_features

        features = extract_meta_features(
            alt=alt,
            depth=depth,
            adj_factor=adj_factor,
            vcf_df=vcf_df,
            cn_df=cn_df,
        )

        config_name = self._router.predict(features, self._subchallenge)

        if self._verbose:
            print(f"UniClone: selected config '{config_name}' for {self._subchallenge.name}")

        config = CONFIGS[config_name]
        model = GenerativeModel(config)
        return model.fit(alt, depth, adj_factor)

    def fit_all_subchallenges(
        self,
        alt: np.ndarray,
        depth: np.ndarray,
        adj_factor: np.ndarray | None = None,
        vcf_df: Any | None = None,
        cn_df: Any | None = None,
    ) -> dict[SubChallenge, CloneResult]:
        """
        Run reconstruction with the best config per subchallenge.

        Returns
        -------
        Dict mapping SubChallenge → CloneResult.
        """
        from uniclone.router.meta_features import extract_meta_features

        features = extract_meta_features(
            alt=alt,
            depth=depth,
            adj_factor=adj_factor,
            vcf_df=vcf_df,
            cn_df=cn_df,
        )

        config_selections = self._router.predict_all(features)

        results: dict[SubChallenge, CloneResult] = {}
        seen_configs: dict[str, CloneResult] = {}

        for sc in SUBCHALLENGES:
            config_name = config_selections[sc]

            if self._verbose:
                print(f"UniClone: selected config '{config_name}' for {sc.name}")

            # Avoid re-running the same config
            if config_name in seen_configs:
                results[sc] = seen_configs[config_name]
            else:
                config = CONFIGS[config_name]
                model = GenerativeModel(config)
                result = model.fit(alt, depth, adj_factor)
                seen_configs[config_name] = result
                results[sc] = result

        return results

    def __repr__(self) -> str:
        return f"UniClone(subchallenge={self._subchallenge.name})"
