"""
Tests for online bandit training pipeline (uniclone.router.training.train_online).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")

from uniclone.router.constants import CONFIG_NAMES, DEFAULT_EXCLUDE_CONFIGS, SUBCHALLENGES
from uniclone.router.training import (
    OnlineTrainResult,
    assemble_corpus,
    generate_tumours,
    train_online,
)


@pytest.fixture
def tumour_dir(tmp_path):
    """Generate a small set of tumours for testing."""
    out = tmp_path / "tumours"
    generate_tumours(
        out_dir=out,
        n_tumours=6,
        n_augmentations=0,
        seed=99,
        simulator="quantumcat",
    )
    return out


class TestTrainOnline:
    def test_returns_online_result(self, tumour_dir, tmp_path):
        scores_dir = tmp_path / "scores"
        result = train_online(
            tumour_dir=tumour_dir,
            scores_dir=scores_dir,
            n_pilot=2,
            n_workers=1,
            n_epochs=2,
            batch_size=64,
            seed=42,
        )
        assert isinstance(result, OnlineTrainResult)
        assert result.n_pilot == 2
        assert result.n_online == 4
        assert result.n_total_fits > 0

    def test_selections_are_valid_configs(self, tumour_dir, tmp_path):
        scores_dir = tmp_path / "scores"
        result = train_online(
            tumour_dir=tumour_dir,
            scores_dir=scores_dir,
            n_pilot=2,
            n_workers=1,
            n_epochs=2,
            batch_size=64,
            seed=42,
        )
        assert len(result.selections) == 4  # n_online
        for cfg in result.selections:
            assert cfg in CONFIG_NAMES

    def test_scores_persisted(self, tumour_dir, tmp_path):
        scores_dir = tmp_path / "scores"
        train_online(
            tumour_dir=tumour_dir,
            scores_dir=scores_dir,
            n_pilot=2,
            n_workers=1,
            n_epochs=2,
            batch_size=64,
            seed=42,
        )
        score_files = list(scores_dir.glob("scores_*.json"))
        assert len(score_files) == 6  # all tumours have scores

        # Pilot tumours should have all configs scored
        # Online tumours should have 1 config + _meta
        active_configs = [c for c in CONFIG_NAMES if c not in DEFAULT_EXCLUDE_CONFIGS]
        n_pilot_full = 0
        n_online_single = 0
        for sf in score_files:
            data = json.loads(sf.read_text())
            config_keys = [k for k in data if k in CONFIG_NAMES]
            if len(config_keys) == len(active_configs):
                n_pilot_full += 1
            elif "_meta" in data and data["_meta"].get("online"):
                n_online_single += 1
                assert len(config_keys) == 1
                assert data["_meta"]["selected"] == config_keys[0]

        assert n_pilot_full == 2
        assert n_online_single == 4

    def test_total_fits_much_less_than_exhaustive(self, tumour_dir, tmp_path):
        scores_dir = tmp_path / "scores"
        result = train_online(
            tumour_dir=tumour_dir,
            scores_dir=scores_dir,
            n_pilot=2,
            n_workers=1,
            n_epochs=2,
            batch_size=64,
            seed=42,
        )
        exhaustive = 6 * len(CONFIG_NAMES)  # 72
        # pilot: 2 * 12 = 24, online: 4 * 1 = 4, total = 28
        assert result.n_total_fits <= 2 * len(CONFIG_NAMES) + 4
        assert result.n_total_fits < exhaustive

    def test_model_can_select(self, tumour_dir, tmp_path):
        scores_dir = tmp_path / "scores"
        result = train_online(
            tumour_dir=tumour_dir,
            scores_dir=scores_dir,
            n_pilot=2,
            n_workers=1,
            n_epochs=2,
            batch_size=64,
            seed=42,
        )
        features = np.random.default_rng(0).standard_normal(21)
        cfg = result.model.select(features, SUBCHALLENGES[0], explore=False)
        assert cfg in CONFIG_NAMES

    def test_resumability(self, tumour_dir, tmp_path):
        """Running train_online twice should not re-do already-scored work."""
        scores_dir = tmp_path / "scores"

        r1 = train_online(
            tumour_dir=tumour_dir,
            scores_dir=scores_dir,
            n_pilot=2,
            n_workers=1,
            n_epochs=2,
            batch_size=64,
            seed=42,
        )

        # Second run — all scores already on disk
        r2 = train_online(
            tumour_dir=tumour_dir,
            scores_dir=scores_dir,
            n_pilot=2,
            n_workers=1,
            n_epochs=2,
            batch_size=64,
            seed=42,
        )

        # No new fits on second run (all resumed)
        assert r2.n_total_fits == 0


class TestAssembleCorpus:
    def test_assembles_pilot_and_online(self, tumour_dir, tmp_path):
        scores_dir = tmp_path / "scores"
        train_online(
            tumour_dir=tumour_dir,
            scores_dir=scores_dir,
            n_pilot=2,
            n_workers=1,
            n_epochs=2,
            batch_size=64,
            seed=42,
        )
        corpus = assemble_corpus(tumour_dir, scores_dir)
        assert len(corpus) > 0
        n_active = len([c for c in CONFIG_NAMES if c not in DEFAULT_EXCLUDE_CONFIGS])
        # Pilot: 2 tumours × n_active configs × 5 subchallenges
        # Online: 4 tumours × 1 config × 5 subchallenges
        expected = 2 * n_active * len(SUBCHALLENGES) + 4 * 1 * len(SUBCHALLENGES)
        assert len(corpus) == expected
        for entry in corpus:
            assert entry.config_name in CONFIG_NAMES
            assert 0 <= entry.score <= 1
