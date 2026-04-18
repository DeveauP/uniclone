"""Tests for uniclone.router.meta_features — no torch required."""

from __future__ import annotations

import time

import numpy as np
import pytest

from uniclone.router.constants import FEATURE_NAMES, N_FEATURES
from uniclone.router.meta_features import extract_meta_features, features_to_tensor


@pytest.fixture
def synthetic_data():
    """Simple synthetic tumour data for feature extraction."""
    rng = np.random.default_rng(42)
    n_mut = 500
    depth = np.full((n_mut, 1), 100, dtype=float)
    alt = rng.binomial(100, 0.3, size=(n_mut, 1)).astype(float)
    adj = np.ones((n_mut, 1), dtype=float)
    # Add some CN-altered mutations
    adj[:50, 0] = 0.7
    adj[50:80, 0] = 1.3
    return alt, depth, adj


class TestExtractMetaFeatures:
    def test_all_features_returned(self, synthetic_data):
        alt, depth, adj = synthetic_data
        features = extract_meta_features(alt=alt, depth=depth, adj_factor=adj)

        assert isinstance(features, dict)
        for name in FEATURE_NAMES:
            assert name in features, f"Missing feature: {name}"
            assert isinstance(features[name], float), f"{name} is not float"

    def test_feature_count(self, synthetic_data):
        alt, depth, adj = synthetic_data
        features = extract_meta_features(alt=alt, depth=depth, adj_factor=adj)
        assert len(features) >= N_FEATURES

    def test_fast_runtime(self, synthetic_data):
        """Feature extraction should complete in < 1 second."""
        alt, depth, adj = synthetic_data
        start = time.time()
        extract_meta_features(alt=alt, depth=depth, adj_factor=adj)
        elapsed = time.time() - start
        assert elapsed < 1.0, f"Feature extraction took {elapsed:.2f}s"

    def test_depth_features(self, synthetic_data):
        alt, depth, adj = synthetic_data
        features = extract_meta_features(alt=alt, depth=depth, adj_factor=adj)

        assert features["depth_median"] == pytest.approx(100.0, abs=5)
        assert features["depth_cv"] >= 0
        assert 0 <= features["frac_low_depth"] <= 1

    def test_cn_features(self, synthetic_data):
        alt, depth, adj = synthetic_data
        features = extract_meta_features(alt=alt, depth=depth, adj_factor=adj)

        assert features["frac_cna"] > 0  # We added CN-altered mutations
        assert features["n_cn_states"] >= 1
        assert 0 <= features["frac_loh"] <= 1

    def test_shape_features(self, synthetic_data):
        alt, depth, adj = synthetic_data
        features = extract_meta_features(alt=alt, depth=depth, adj_factor=adj)

        assert isinstance(features["skewness"], float)
        assert isinstance(features["kurtosis"], float)
        assert features["n_peaks"] >= 1

    def test_tensor_only_mode(self):
        """Should work with just alt/depth arrays, no vcf_df or cn_df."""
        rng = np.random.default_rng(42)
        alt = rng.binomial(100, 0.4, size=(100, 1)).astype(float)
        depth = np.full((100, 1), 100, dtype=float)

        features = extract_meta_features(alt=alt, depth=depth)
        assert len(features) >= N_FEATURES

    def test_1d_input(self):
        """Should handle 1-D input arrays."""
        rng = np.random.default_rng(42)
        alt = rng.binomial(100, 0.4, size=100).astype(float)
        depth = np.full(100, 100, dtype=float)

        features = extract_meta_features(alt=alt, depth=depth)
        assert features["n_samples"] == 1.0

    def test_multisample(self):
        """Should handle multi-sample data."""
        rng = np.random.default_rng(42)
        alt = rng.binomial(100, 0.3, size=(200, 3)).astype(float)
        depth = np.full((200, 3), 100, dtype=float)

        features = extract_meta_features(alt=alt, depth=depth)
        assert features["n_samples"] == 3.0


class TestFeaturesToTensor:
    def test_correct_shape(self, synthetic_data):
        alt, depth, adj = synthetic_data
        features = extract_meta_features(alt=alt, depth=depth, adj_factor=adj)
        tensor = features_to_tensor(features)

        assert tensor.shape == (N_FEATURES,)
        assert tensor.dtype == np.float64

    def test_correct_order(self, synthetic_data):
        alt, depth, adj = synthetic_data
        features = extract_meta_features(alt=alt, depth=depth, adj_factor=adj)
        tensor = features_to_tensor(features)

        for i, name in enumerate(FEATURE_NAMES):
            assert tensor[i] == features[name]
