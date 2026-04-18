"""
uniclone.router.router
=======================

MetaRouter: user-facing API for automatic configuration selection
using Neural Thompson Sampling.

Wraps a NeuralTSModel and provides predict/explain/update/scores/uncertainty.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from uniclone.router.constants import SUBCHALLENGES, SubChallenge


class MetaRouter:
    """
    Neural Thompson Sampling meta-routing model.

    Automatically selects the best named configuration for a given tumour
    based on 21 meta-features extracted from the input data.

    Parameters
    ----------
    model_path : path to pretrained model weights, or None for fresh model.
    device : torch device string.
    explore : if True (default False), use Thompson sampling for exploration.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str = "cpu",
        explore: bool = False,
    ) -> None:
        from uniclone.router.neural_ts import NeuralTSModel

        if model_path is not None:
            self._model = NeuralTSModel.from_pretrained(model_path, device=device)
        else:
            self._model = NeuralTSModel(device=device)

        self._explore = explore

    def predict(
        self,
        features: dict[str, float] | np.ndarray,
        subchallenge: SubChallenge | str,
    ) -> str:
        """
        Select best configuration for the given features and subchallenge.

        Parameters
        ----------
        features : meta-feature dict or array
        subchallenge : SubChallenge enum or string name (e.g. "SC2B")

        Returns
        -------
        Name of selected configuration (key into uniclone.CONFIGS).
        """
        sc = _parse_subchallenge(subchallenge)
        return self._model.select(features, sc, explore=self._explore)

    def predict_all(
        self,
        features: dict[str, float] | np.ndarray,
    ) -> dict[SubChallenge, str]:
        """Select best config per subchallenge."""
        return {sc: self._model.select(features, sc, explore=self._explore) for sc in SUBCHALLENGES}

    def explain(
        self,
        features: dict[str, float] | np.ndarray,
        subchallenge: SubChallenge | str,
    ) -> dict[str, float]:
        """
        Compute feature attributions for the routing decision.

        Uses integrated gradients on the differentiable score function.

        Returns
        -------
        Dict mapping feature name → attribution value.
        """
        from uniclone.router.explain import compute_feature_attribution

        sc = _parse_subchallenge(subchallenge)
        return compute_feature_attribution(self._model, features, sc)

    def update(
        self,
        features: dict[str, float] | np.ndarray,
        config_name: str,
        subchallenge: SubChallenge | str,
        reward: float,
    ) -> None:
        """Online posterior update after observing a reward."""
        sc = _parse_subchallenge(subchallenge)
        self._model.update(features, config_name, sc, reward)

    def scores(
        self,
        features: dict[str, float] | np.ndarray,
        subchallenge: SubChallenge | str,
    ) -> dict[str, float]:
        """Return posterior mean score for each configuration."""
        sc = _parse_subchallenge(subchallenge)
        return self._model.scores(features, sc)

    def uncertainty(
        self,
        features: dict[str, float] | np.ndarray,
        subchallenge: SubChallenge | str,
    ) -> dict[str, float]:
        """Return posterior uncertainty for each configuration."""
        sc = _parse_subchallenge(subchallenge)
        return self._model.uncertainty(features, sc)

    def save(self, path: str | Path) -> None:
        """Save model weights to disk."""
        self._model.save(path)

    def __repr__(self) -> str:
        return f"MetaRouter(explore={self._explore})"


def _parse_subchallenge(sc: SubChallenge | str) -> SubChallenge:
    """Convert string to SubChallenge enum if needed."""
    if isinstance(sc, SubChallenge):
        return sc
    try:
        return SubChallenge[sc]
    except KeyError:
        raise ValueError(
            f"Unknown subchallenge '{sc}'. Valid: {[s.name for s in SubChallenge]}"
        ) from None
