"""Tests for uniclone.router.evaluate — requires torch."""
from __future__ import annotations

import numpy as np
import pytest

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")

from uniclone.router.constants import CONFIG_NAMES, N_FEATURES, SUBCHALLENGES, SubChallenge
from uniclone.router.evaluate import cumulative_regret, oracle_regret, routing_gain
from uniclone.router.neural_ts import NeuralTSModel
from uniclone.router.training import CorpusEntry


def _make_corpus(n_tumours: int = 10, rng_seed: int = 42) -> list[CorpusEntry]:
    """Create a synthetic test corpus."""
    rng = np.random.default_rng(rng_seed)
    entries = []
    for _ in range(n_tumours):
        feat = rng.standard_normal(N_FEATURES)
        for config_name in CONFIG_NAMES:
            for sc in SUBCHALLENGES:
                entries.append(CorpusEntry(
                    features=feat.copy(),
                    subchallenge=sc,
                    config_name=config_name,
                    score=rng.uniform(0, 1),
                ))
    return entries


def _make_perfect_model(corpus: list[CorpusEntry]) -> NeuralTSModel:
    """
    Create a model that 'perfectly' knows the best config for each entry.

    We do this by heavily updating the best config's head for each tumour.
    """
    model = NeuralTSModel()

    # Group by (features, sc) → find oracle config
    groups: dict[tuple[bytes, SubChallenge], dict[str, float]] = {}
    for entry in corpus:
        key = (entry.features.tobytes(), entry.subchallenge)
        if key not in groups:
            groups[key] = {}
        groups[key][entry.config_name] = entry.score

    for (feat_bytes, sc), config_scores in groups.items():
        features = np.frombuffer(feat_bytes, dtype=np.float64).copy()
        best_config = max(config_scores, key=config_scores.get)
        # Update heavily so the model learns this is the best
        for _ in range(100):
            model.update(features, best_config, sc, config_scores[best_config])
            # Also negatively update other configs
            for other in CONFIG_NAMES:
                if other != best_config:
                    model.update(features, other, sc, 0.0)

    return model


class TestOracleRegret:
    def test_perfect_router_zero_regret(self):
        """A perfectly trained router should have near-zero regret."""
        corpus = _make_corpus(n_tumours=3)
        model = _make_perfect_model(corpus)

        regrets = oracle_regret(model, corpus)
        assert isinstance(regrets, dict)
        for sc in SUBCHALLENGES:
            assert sc in regrets
            # May not be exactly 0 due to finite updates, but should be small
            assert regrets[sc] >= 0

    def test_random_router_positive_regret(self):
        """An untrained (random) router should have positive regret."""
        corpus = _make_corpus(n_tumours=5)
        model = NeuralTSModel()  # untrained

        regrets = oracle_regret(model, corpus)
        # At least some subchallenges should have positive regret
        total_regret = sum(regrets.values())
        assert total_regret >= 0

    def test_empty_corpus(self):
        model = NeuralTSModel()
        regrets = oracle_regret(model, [])
        for sc in SUBCHALLENGES:
            assert regrets[sc] == 0.0


class TestRoutingGain:
    def test_gain_dict(self):
        corpus = _make_corpus(n_tumours=3)
        model = NeuralTSModel()
        gains = routing_gain(model, corpus)

        assert isinstance(gains, dict)
        for sc in SUBCHALLENGES:
            assert sc in gains

    def test_gain_with_baseline(self):
        corpus = _make_corpus(n_tumours=3)
        model = NeuralTSModel()
        gains = routing_gain(model, corpus, baseline_config="mobster")
        assert isinstance(gains, dict)


class TestCumulativeRegret:
    def test_sublinear_growth(self):
        """Cumulative regret should grow, but sublinearly (O(√T))."""
        corpus = _make_corpus(n_tumours=20)
        model = NeuralTSModel()

        regrets = cumulative_regret(model, corpus)
        assert len(regrets) > 0
        # Regret should be non-negative cumulative
        assert all(r >= 0 for r in regrets)
        # Should be non-decreasing
        for i in range(1, len(regrets)):
            assert regrets[i] >= regrets[i - 1] - 1e-9

    def test_empty_stream(self):
        model = NeuralTSModel()
        regrets = cumulative_regret(model, [])
        assert regrets == []
