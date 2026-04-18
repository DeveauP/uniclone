"""
uniclone.router.evaluate
=========================

Evaluation metrics for the NeuralTS MetaRouter.

- oracle_regret: gap between router and oracle (best hindsight config)
- routing_gain: improvement over a fixed baseline config
- cumulative_regret: online regret over a stream of tumours
"""

from __future__ import annotations

import numpy as np

from uniclone.router.constants import SUBCHALLENGES, SubChallenge
from uniclone.router.training import CorpusEntry


def oracle_regret(
    model: object,
    test_corpus: list[CorpusEntry],
) -> dict[SubChallenge, float]:
    """
    Compute oracle regret per subchallenge.

    Oracle regret = mean(oracle_score - router_score) over test tumours.
    A perfect router has 0 regret.

    Parameters
    ----------
    model : NeuralTSModel
    test_corpus : list of CorpusEntry (all configs scored per tumour)

    Returns
    -------
    Dict mapping SubChallenge → mean regret.
    """
    from uniclone.router.neural_ts import NeuralTSModel

    assert isinstance(model, NeuralTSModel)

    # Group by (tumour_features_hash, subchallenge)
    groups: dict[tuple[bytes, SubChallenge], dict[str, float]] = {}
    for entry in test_corpus:
        key = (entry.features.tobytes(), entry.subchallenge)
        if key not in groups:
            groups[key] = {}
        groups[key][entry.config_name] = entry.score

    regrets: dict[SubChallenge, list[float]] = {sc: [] for sc in SUBCHALLENGES}

    for (feat_bytes, sc), config_scores in groups.items():
        features = np.frombuffer(feat_bytes, dtype=np.float64).copy()
        oracle_score = max(config_scores.values())
        selected = model.select(features, sc, explore=False)
        router_score = config_scores.get(selected, 0.0)
        regrets[sc].append(oracle_score - router_score)

    return {sc: float(np.mean(vals)) if vals else 0.0 for sc, vals in regrets.items()}


def routing_gain(
    model: object,
    test_corpus: list[CorpusEntry],
    baseline_config: str = "quantumclone_v1",
) -> dict[SubChallenge, float]:
    """
    Compute routing gain over a fixed baseline config per subchallenge.

    Gain = mean(router_score - baseline_score) over test tumours.
    Positive means the router improves over the baseline.

    Parameters
    ----------
    model : NeuralTSModel
    test_corpus : list of CorpusEntry
    baseline_config : name of baseline configuration

    Returns
    -------
    Dict mapping SubChallenge → mean gain.
    """
    from uniclone.router.neural_ts import NeuralTSModel

    assert isinstance(model, NeuralTSModel)

    groups: dict[tuple[bytes, SubChallenge], dict[str, float]] = {}
    for entry in test_corpus:
        key = (entry.features.tobytes(), entry.subchallenge)
        if key not in groups:
            groups[key] = {}
        groups[key][entry.config_name] = entry.score

    gains: dict[SubChallenge, list[float]] = {sc: [] for sc in SUBCHALLENGES}

    for (feat_bytes, sc), config_scores in groups.items():
        features = np.frombuffer(feat_bytes, dtype=np.float64).copy()
        baseline_score = config_scores.get(baseline_config, 0.0)
        selected = model.select(features, sc, explore=False)
        router_score = config_scores.get(selected, 0.0)
        gains[sc].append(router_score - baseline_score)

    return {sc: float(np.mean(vals)) if vals else 0.0 for sc, vals in gains.items()}


def cumulative_regret(
    model: object,
    tumour_stream: list[CorpusEntry],
) -> list[float]:
    """
    Compute cumulative regret over a stream of tumours (online setting).

    For each tumour in the stream:
    1. Router selects a config (with exploration)
    2. Observes reward
    3. Updates posterior
    4. Regret += oracle_reward - observed_reward

    Should grow sublinearly (O(√T)) for a good algorithm.

    Parameters
    ----------
    model : NeuralTSModel
    tumour_stream : list of CorpusEntry, grouped by tumour
        (all configs for tumour 1, then all for tumour 2, etc.)

    Returns
    -------
    List of cumulative regret values, one per tumour.
    """
    from uniclone.router.neural_ts import NeuralTSModel

    assert isinstance(model, NeuralTSModel)

    # Group by (features, subchallenge) preserving order
    groups: list[tuple[np.ndarray, SubChallenge, dict[str, float]]] = []
    seen: dict[tuple[bytes, SubChallenge], int] = {}

    for entry in tumour_stream:
        key = (entry.features.tobytes(), entry.subchallenge)
        if key not in seen:
            seen[key] = len(groups)
            groups.append((entry.features.copy(), entry.subchallenge, {}))
        groups[seen[key]][2][entry.config_name] = entry.score

    cum_regret: list[float] = []
    total = 0.0

    for features, sc, config_scores in groups:
        oracle_score = max(config_scores.values())
        selected = model.select(features, sc, explore=True)
        observed = config_scores.get(selected, 0.0)

        # Update posterior
        model.update(features, selected, sc, observed)

        total += oracle_score - observed
        cum_regret.append(total)

    return cum_regret
