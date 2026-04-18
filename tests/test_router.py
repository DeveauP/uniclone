"""Tests for uniclone.router.router — MetaRouter user-facing API."""
from __future__ import annotations

import numpy as np
import pytest

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")

from uniclone.router.constants import CONFIG_NAMES, FEATURE_NAMES, SUBCHALLENGES, SubChallenge
from uniclone.router.router import MetaRouter


@pytest.fixture
def router():
    return MetaRouter()


@pytest.fixture
def sample_features():
    return {name: float(np.random.default_rng(42).standard_normal()) for name in FEATURE_NAMES}


class TestMetaRouter:
    def test_predict_returns_valid_config(self, router, sample_features):
        config = router.predict(sample_features, SubChallenge.SC2B)
        assert config in CONFIG_NAMES

    def test_predict_string_subchallenge(self, router, sample_features):
        config = router.predict(sample_features, "SC2B")
        assert config in CONFIG_NAMES

    def test_predict_invalid_subchallenge(self, router, sample_features):
        with pytest.raises(ValueError, match="Unknown subchallenge"):
            router.predict(sample_features, "INVALID")

    def test_predict_all(self, router, sample_features):
        results = router.predict_all(sample_features)
        assert isinstance(results, dict)
        assert len(results) == len(SUBCHALLENGES)
        for sc in SUBCHALLENGES:
            assert sc in results
            assert results[sc] in CONFIG_NAMES

    def test_scores_dict(self, router, sample_features):
        scores = router.scores(sample_features, SubChallenge.SC2B)
        assert isinstance(scores, dict)
        assert len(scores) == len(CONFIG_NAMES)

    def test_uncertainty_dict(self, router, sample_features):
        unc = router.uncertainty(sample_features, SubChallenge.SC2B)
        assert isinstance(unc, dict)
        for name in CONFIG_NAMES:
            assert unc[name] > 0

    def test_update(self, router, sample_features):
        """Update should not raise and should change scores."""
        scores_before = router.scores(sample_features, SubChallenge.SC2B)
        for _ in range(20):
            router.update(sample_features, "wes_clinical", SubChallenge.SC2B, 0.9)
        scores_after = router.scores(sample_features, SubChallenge.SC2B)
        assert scores_after["wes_clinical"] > scores_before["wes_clinical"]

    def test_explain_completeness(self, router, sample_features):
        """Attribution sum should approximately equal score difference."""
        attributions = router.explain(sample_features, SubChallenge.SC2B)
        assert isinstance(attributions, dict)
        assert len(attributions) == len(FEATURE_NAMES)
        for name in FEATURE_NAMES:
            assert name in attributions
            assert isinstance(attributions[name], float)

    def test_repr(self, router):
        assert "MetaRouter" in repr(router)

    def test_numpy_features(self, router):
        features = np.random.default_rng(42).standard_normal(len(FEATURE_NAMES))
        config = router.predict(features, SubChallenge.SC1A)
        assert config in CONFIG_NAMES
