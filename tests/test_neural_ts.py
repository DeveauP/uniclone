"""Tests for uniclone.router.neural_ts — requires torch."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")

from uniclone.router.constants import CONFIG_NAMES, N_FEATURES, SubChallenge
from uniclone.router.neural_ts import BayesianLinearHead, NeuralTSModel, SharedEncoder


class TestSharedEncoder:
    def test_output_shape(self):
        encoder = SharedEncoder()
        x = torch.randn(5, N_FEATURES)
        z = encoder(x)
        assert z.shape == (5, 32)

    def test_single_input(self):
        encoder = SharedEncoder()
        x = torch.randn(1, N_FEATURES)
        z = encoder(x)
        assert z.shape == (1, 32)

    def test_custom_dims(self):
        encoder = SharedEncoder(input_dim=10, hidden_dim=32, output_dim=16)
        x = torch.randn(3, 10)
        z = encoder(x)
        assert z.shape == (3, 16)


class TestBayesianLinearHead:
    def test_initial_state(self):
        head = BayesianLinearHead(dim=8)
        assert head.n_updates == 0
        assert head.precision.shape == (8, 8)
        assert head.zy_sum.shape == (8,)

    def test_uncertainty_decreases(self):
        """Uncertainty should decrease with more observations."""
        head = BayesianLinearHead(dim=8)
        z = torch.randn(8)

        unc_before = head.uncertainty(z)
        for _ in range(10):
            head.update(z, 1.0)
        unc_after = head.uncertainty(z)

        assert unc_after < unc_before

    def test_posterior_convergence(self):
        """Posterior mean should converge to true weights with enough data."""
        dim = 4
        true_w = torch.tensor([1.0, -0.5, 0.3, 0.8])
        head = BayesianLinearHead(dim=dim, reg_lambda=0.01)

        rng = np.random.default_rng(42)
        for _ in range(500):
            z = torch.randn(dim)
            reward = float(true_w @ z) + rng.normal(0, 0.1)
            head.update(z, reward)

        mu = head.mu
        # Posterior mean should be close to true weights
        assert torch.allclose(mu, true_w, atol=0.2)

    def test_thompson_variance(self):
        """Thompson samples should vary (not deterministic)."""
        head = BayesianLinearHead(dim=8)
        z = torch.randn(8)

        samples = [head.thompson_sample(z) for _ in range(20)]
        # Should have non-zero variance
        assert np.std(samples) > 0

    def test_mean_predict(self):
        head = BayesianLinearHead(dim=4)
        z = torch.ones(4)
        # With no updates, mean should be ~0
        pred = head.mean_predict(z)
        assert abs(pred) < 1e-6


class TestNeuralTSModel:
    def test_select_returns_valid_config(self):
        model = NeuralTSModel()
        features = np.random.randn(N_FEATURES)
        config = model.select(features, SubChallenge.SC2B, explore=False)
        assert config in CONFIG_NAMES

    def test_select_with_dict_features(self):
        from uniclone.router.constants import FEATURE_NAMES

        model = NeuralTSModel()
        features = {name: float(np.random.randn()) for name in FEATURE_NAMES}
        config = model.select(features, SubChallenge.SC1A, explore=True)
        assert config in CONFIG_NAMES

    def test_update_changes_predictions(self):
        model = NeuralTSModel()
        features = np.random.randn(N_FEATURES)

        # Record scores before
        scores_before = model.scores(features, SubChallenge.SC2B)

        # Update heavily for one config
        for _ in range(50):
            model.update(features, "pyclone_vi", SubChallenge.SC2B, 1.0)

        scores_after = model.scores(features, SubChallenge.SC2B)
        # pyclone_vi score should increase
        assert scores_after["pyclone_vi"] > scores_before["pyclone_vi"]

    def test_scores_dict(self):
        model = NeuralTSModel()
        features = np.random.randn(N_FEATURES)
        scores = model.scores(features, SubChallenge.SC2B)

        assert isinstance(scores, dict)
        assert len(scores) == len(CONFIG_NAMES)
        for name in CONFIG_NAMES:
            assert name in scores

    def test_uncertainty_dict(self):
        model = NeuralTSModel()
        features = np.random.randn(N_FEATURES)
        unc = model.uncertainty(features, SubChallenge.SC2B)

        assert isinstance(unc, dict)
        assert len(unc) == len(CONFIG_NAMES)
        for name in CONFIG_NAMES:
            assert name in unc
            assert unc[name] > 0  # uncertainty should be positive

    def test_save_load_roundtrip(self):
        model = NeuralTSModel()
        features = np.random.randn(N_FEATURES)

        # Update some heads
        model.update(features, "mobster", SubChallenge.SC1A, 0.8)

        scores_before = model.scores(features, SubChallenge.SC1A)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            model.save(path)

            loaded = NeuralTSModel.from_pretrained(path)
            scores_after = loaded.scores(features, SubChallenge.SC1A)

        for name in CONFIG_NAMES:
            assert abs(scores_before[name] - scores_after[name]) < 1e-5

    def test_n_heads(self):
        model = NeuralTSModel()
        assert len(model.heads) == 60  # 12 configs × 5 subchallenges
