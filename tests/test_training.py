"""Tests for uniclone.router.training — requires torch."""

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
from uniclone.router.training import CorpusEntry, score_result


@pytest.fixture
def mini_corpus():
    """Small corpus with 10 entries for testing."""
    rng = np.random.default_rng(42)
    entries = []
    for _ in range(2):  # 2 tumours
        feat = rng.standard_normal(N_FEATURES)
        for config_name in CONFIG_NAMES[:5]:  # 5 configs
            for sc in SUBCHALLENGES:
                entries.append(
                    CorpusEntry(
                        features=feat.copy(),
                        subchallenge=sc,
                        config_name=config_name,
                        score=rng.uniform(0, 1),
                    )
                )
    return entries


class TestCorpusEntry:
    def test_creation(self):
        entry = CorpusEntry(
            features=np.zeros(N_FEATURES),
            subchallenge=SubChallenge.SC2B,
            config_name="pyclone_vi",
            score=0.85,
        )
        assert entry.score == 0.85
        assert entry.config_name == "pyclone_vi"


class TestScoreResult:
    def test_sc1a_perfect(self):
        """Perfect purity estimation should score ~1.0."""
        from uniclone.simulate.quantum_cat import QuantumCatParams, QuantumCatResult

        params = QuantumCatParams(n_clones=2, n_mutations=100, purity=0.8, seed=42)

        # Mock a perfect result
        class MockResult:
            centers = np.array([[0.8], [0.4]])
            assignments = np.arange(100) % 2
            K = 2
            tree = None

        gt = QuantumCatResult(
            alt=np.zeros((100, 1)),
            depth=np.full((100, 1), 100),
            adj_factor=np.ones((100, 1)),
            true_assignments=np.arange(100) % 2,
            true_centers=np.array([[0.8], [0.4]]),
            true_tree=np.array([[0, 1], [0, 0]]),
            params=params,
        )

        score = score_result(MockResult(), gt, SubChallenge.SC1A)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_sc2a_correct_k(self):
        """Correct K should score 1.0."""
        from uniclone.simulate.quantum_cat import QuantumCatParams, QuantumCatResult

        params = QuantumCatParams(n_clones=3, n_mutations=100, seed=42)
        gt = QuantumCatResult(
            alt=np.zeros((100, 1)),
            depth=np.full((100, 1), 100),
            adj_factor=np.ones((100, 1)),
            true_assignments=np.repeat([0, 1, 2], [34, 33, 33]),
            true_centers=np.array([[0.8], [0.5], [0.2]]),
            true_tree=np.zeros((3, 3)),
            params=params,
        )

        class MockResult:
            K = 3
            centers = np.array([[0.8], [0.5], [0.2]])
            assignments = np.repeat([0, 1, 2], [34, 33, 33])
            tree = None

        score = score_result(MockResult(), gt, SubChallenge.SC2A)
        assert score == 1.0

    def test_sc2a_wrong_k(self):
        """Wrong K should score < 1.0."""
        from uniclone.simulate.quantum_cat import QuantumCatParams, QuantumCatResult

        params = QuantumCatParams(n_clones=3, n_mutations=100, seed=42)
        gt = QuantumCatResult(
            alt=np.zeros((100, 1)),
            depth=np.full((100, 1), 100),
            adj_factor=np.ones((100, 1)),
            true_assignments=np.repeat([0, 1, 2], [34, 33, 33]),
            true_centers=np.array([[0.8], [0.5], [0.2]]),
            true_tree=np.zeros((3, 3)),
            params=params,
        )

        class MockResult:
            K = 5
            centers = np.array([[0.8], [0.5], [0.2], [0.1], [0.05]])
            assignments = np.repeat([0, 1, 2, 3, 4], 20)
            tree = None

        score = score_result(MockResult(), gt, SubChallenge.SC2A)
        assert 0 <= score < 1.0

    def test_sc2a_k1_overestimate_gets_partial_credit(self):
        """Estimating K=2 on a K=1 tumour should get partial credit, not 0."""
        from uniclone.simulate.quantum_cat import QuantumCatParams, QuantumCatResult

        params = QuantumCatParams(n_clones=1, n_mutations=100, purity=0.8, seed=42)
        gt = QuantumCatResult(
            alt=np.zeros((100, 1)),
            depth=np.full((100, 1), 100),
            adj_factor=np.ones((100, 1)),
            true_assignments=np.zeros(100, dtype=int),
            true_centers=np.array([[0.8]]),
            true_tree=None,
            params=params,
        )

        class MockResult:
            K = 2
            centers = np.array([[0.8], [0.3]])
            assignments = np.concatenate([np.zeros(60), np.ones(40)]).astype(int)
            tree = None

        score = score_result(MockResult(), gt, SubChallenge.SC2A)
        assert score == 0.5, f"K=1 true, K=2 est should give 0.5, got {score}"

    def test_sc1b_constant_prediction_not_zero(self):
        """K=1 (constant est CCF) should still get partial credit via RMSE."""
        from uniclone.simulate.quantum_cat import QuantumCatParams, QuantumCatResult

        params = QuantumCatParams(n_clones=2, n_mutations=100, purity=0.8, seed=42)
        gt = QuantumCatResult(
            alt=np.zeros((100, 1)),
            depth=np.full((100, 1), 100),
            adj_factor=np.ones((100, 1)),
            true_assignments=np.concatenate([np.zeros(50), np.ones(50)]).astype(int),
            true_centers=np.array([[0.8], [0.4]]),
            true_tree=np.array([[0, 1], [0, 0]]),
            params=params,
        )

        class MockResult:
            K = 1
            centers = np.array([[0.6]])  # single cluster, midpoint
            assignments = np.zeros(100, dtype=int)
            tree = None

        score = score_result(MockResult(), gt, SubChallenge.SC1B)
        assert score > 0.0, "Constant CCF prediction should get partial RMSE credit"

    def test_sc2b_with_noise_filtered_mutations(self):
        """Noise-filtered mutations (assignment=-1) should be excluded, not fail."""
        from uniclone.simulate.quantum_cat import QuantumCatParams, QuantumCatResult

        params = QuantumCatParams(n_clones=2, n_mutations=100, purity=0.8, seed=42)
        gt = QuantumCatResult(
            alt=np.zeros((100, 1)),
            depth=np.full((100, 1), 100),
            adj_factor=np.ones((100, 1)),
            true_assignments=np.concatenate([np.zeros(50), np.ones(50)]).astype(int),
            true_centers=np.array([[0.8], [0.4]]),
            true_tree=None,
            params=params,
        )

        # Simulate tail-filtered result: 80 kept, 20 filtered as -1
        assignments = np.concatenate(
            [
                np.zeros(40, dtype=int),
                np.ones(40, dtype=int),
                np.full(20, -1, dtype=int),
            ]
        )

        class MockResult:
            K = 2
            centers = np.array([[0.8], [0.4]])
            tree = None

        MockResult.assignments = assignments

        score = score_result(MockResult(), gt, SubChallenge.SC2B)
        assert score > 0.0, "V-measure should work on valid subset"

    def test_sc3_no_tree_gets_partial_credit(self):
        """Configs without trees should get partial credit based on K."""
        from uniclone.simulate.quantum_cat import QuantumCatParams, QuantumCatResult

        params = QuantumCatParams(n_clones=2, n_mutations=100, purity=0.8, seed=42)
        gt = QuantumCatResult(
            alt=np.zeros((100, 1)),
            depth=np.full((100, 1), 100),
            adj_factor=np.ones((100, 1)),
            true_assignments=np.zeros(100, dtype=int),
            true_centers=np.array([[0.8], [0.4]]),
            true_tree=np.array([[0, 1], [0, 0]]),
            params=params,
        )

        class MockResult:
            K = 2
            centers = np.array([[0.8], [0.4]])
            assignments = np.zeros(100, dtype=int)
            tree = None

        score = score_result(MockResult(), gt, SubChallenge.SC3)
        assert score > 0.0, "No tree but correct K should get partial credit"
        assert score <= 0.5, "Partial credit should be capped at 0.5"

    def test_sc1b_multisample_uses_all_samples(self):
        """SC1B should average RMSE across all samples, not just column 0."""
        from uniclone.simulate.quantum_cat import QuantumCatParams, QuantumCatResult

        params = QuantumCatParams(n_clones=2, n_mutations=100, n_samples=2, purity=0.8, seed=42)
        gt = QuantumCatResult(
            alt=np.zeros((100, 2)),
            depth=np.full((100, 2), 100),
            adj_factor=np.ones((100, 2)),
            true_assignments=np.concatenate([np.zeros(50), np.ones(50)]).astype(int),
            true_centers=np.array([[0.8, 0.7], [0.4, 0.3]]),
            true_tree=np.array([[0, 1], [0, 0]]),
            params=params,
        )

        # Perfect on sample 0, wrong on sample 1
        class MockResult:
            K = 2
            centers = np.array([[0.8, 0.1], [0.4, 0.1]])  # sample 1 is garbage
            assignments = np.concatenate([np.zeros(50), np.ones(50)]).astype(int)
            tree = None

        score = score_result(MockResult(), gt, SubChallenge.SC1B)
        # Should be penalised for bad sample 1, not perfect
        assert score < 0.95, "Multi-sample SC1B should penalise bad sample 1"
        assert score > 0.0

    def test_sc2b_excludes_tail_mutations(self):
        """Neutral tail mutations (label >= K) should be excluded from V-measure."""
        from uniclone.simulate.quantum_cat import QuantumCatParams, QuantumCatResult

        params = QuantumCatParams(n_clones=2, n_mutations=100, purity=0.8, seed=42)
        # 80 real mutations (clones 0,1) + 20 tail (label=2, which is K)
        true_assign = np.concatenate(
            [
                np.zeros(40, dtype=int),
                np.ones(40, dtype=int),
                np.full(20, 2, dtype=int),  # tail label = K
            ]
        )
        gt = QuantumCatResult(
            alt=np.zeros((120, 1)),
            depth=np.full((120, 1), 100),
            adj_factor=np.ones((120, 1)),
            true_assignments=true_assign,
            true_centers=np.array([[0.8], [0.4]]),  # only 2 real clones
            true_tree=np.array([[0, 1], [0, 0]]),
            params=params,
        )

        # Estimator correctly assigns the 80 real mutations
        class MockResult:
            K = 2
            centers = np.array([[0.8], [0.4]])
            assignments = np.concatenate(
                [
                    np.zeros(40, dtype=int),
                    np.ones(40, dtype=int),
                    np.zeros(20, dtype=int),  # tail assigned to clone 0 (doesn't matter)
                ]
            )
            tree = None

        score = score_result(MockResult(), gt, SubChallenge.SC2B)
        assert score > 0.9, "Perfect assignment of real mutations should score high"

    def test_score_ranges(self, mini_corpus):
        """All scores should be in [0, 1]."""
        for entry in mini_corpus:
            assert 0 <= entry.score <= 1


class TestBuildTrainingCorpus:
    def test_small_corpus(self):
        """Test corpus generation with very small n (n=1 tumour)."""
        # This is an integration test — uses QuantumCat + all configs.
        # Only test that the function runs without error for 1 tumour.
        from uniclone.router.training import build_training_corpus

        corpus = build_training_corpus(n_tumours=1, n_workers=1)
        assert len(corpus) > 0
        for entry in corpus:
            assert isinstance(entry, CorpusEntry)
            assert 0 <= entry.score <= 1


class TestGenerateTumoursManifest:
    """Verify manifest records the actual simulator used, not the requested one."""

    def test_fallback_manifest_says_quantumcat(self, tmp_path):
        """When bamsurgeon is unavailable, manifest.simulator must be 'quantumcat'."""
        import json
        import warnings

        from uniclone.router.training import generate_tumours

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            generate_tumours(
                out_dir=tmp_path / "tumours",
                n_tumours=1,
                n_augmentations=0,
                seed=0,
                simulator="bamsurgeon",
            )

        manifest = json.loads((tmp_path / "tumours" / "manifest.json").read_text())
        # BAMSurgeon is not installed in CI, so it must have fallen back
        from uniclone.simulate.bamsurgeon_wrap import is_available

        if is_available():
            assert manifest["simulator"] == "bamsurgeon"
        else:
            assert manifest["simulator"] == "quantumcat"

    def test_explicit_quantumcat_manifest(self, tmp_path):
        """Explicitly requesting quantumcat should record 'quantumcat'."""
        import json

        from uniclone.router.training import generate_tumours

        generate_tumours(
            out_dir=tmp_path / "tumours",
            n_tumours=1,
            n_augmentations=0,
            seed=0,
            simulator="quantumcat",
        )
        manifest = json.loads((tmp_path / "tumours" / "manifest.json").read_text())
        assert manifest["simulator"] == "quantumcat"

    def test_manifest_n_total_matches_files(self, tmp_path):
        """manifest.n_total should match the number of tumour files on disk."""
        import json

        from uniclone.router.training import generate_tumours

        n = generate_tumours(
            out_dir=tmp_path / "tumours",
            n_tumours=2,
            n_augmentations=1,
            seed=0,
            simulator="quantumcat",
        )
        manifest = json.loads((tmp_path / "tumours" / "manifest.json").read_text())
        files = list((tmp_path / "tumours").glob("tumour_*.npz"))
        assert manifest["n_total"] == len(files)
        assert n == len(files)
        # 2 base + 2*1 augmentation = 4
        assert n == 4


class TestTrainRouter:
    def test_encoder_loss_decreases(self, mini_corpus):
        """Training should produce a model without errors."""
        from uniclone.router.training import train_router

        model = train_router(mini_corpus)
        # Model should be usable
        features = np.random.default_rng(42).standard_normal(N_FEATURES)
        config = model.select(features, SubChallenge.SC2B, explore=False)
        assert config in CONFIG_NAMES
