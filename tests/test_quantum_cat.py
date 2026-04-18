"""Tests for uniclone.simulate.quantum_cat — no torch required."""

from __future__ import annotations

import numpy as np
import pytest

from uniclone.simulate.quantum_cat import (
    NoiseParams,
    QuantumCatParams,
    augment_result,
    sample_tumour_params,
    simulate_quantumcat,
)


class TestQuantumCatParams:
    def test_defaults(self):
        p = QuantumCatParams()
        assert p.n_clones == 3
        assert p.n_mutations == 500
        assert p.n_samples == 1

    def test_invalid_n_clones(self):
        with pytest.raises(ValueError, match="n_clones"):
            QuantumCatParams(n_clones=0)

    def test_invalid_n_mutations(self):
        with pytest.raises(ValueError, match="n_mutations"):
            QuantumCatParams(n_mutations=-1)


class TestSimulateQuantumcat:
    def test_output_shapes(self):
        params = QuantumCatParams(n_clones=3, n_mutations=200, n_samples=1, seed=42)
        result = simulate_quantumcat(params)

        assert result.alt.shape == (200, 1)
        assert result.depth.shape == (200, 1)
        assert result.adj_factor.shape == (200, 1)
        assert result.true_assignments.shape == (200,)
        assert result.true_centers.shape == (3, 1)
        assert result.true_tree.shape == (3, 3)

    def test_multisample_shapes(self):
        params = QuantumCatParams(n_clones=2, n_mutations=100, n_samples=3, seed=42)
        result = simulate_quantumcat(params)

        assert result.alt.shape == (100, 3)
        assert result.depth.shape == (100, 3)
        assert result.true_centers.shape == (2, 3)

    def test_vaf_distribution(self):
        """VAFs should be in [0, 1] and cluster around true cellularities."""
        params = QuantumCatParams(n_clones=2, n_mutations=1000, n_samples=1, seed=42)
        result = simulate_quantumcat(params)

        vaf = result.alt[:, 0] / np.maximum(result.depth[:, 0], 1)
        assert np.all(vaf >= 0)
        assert np.all(vaf <= 1)

    def test_reproducibility(self):
        """Same seed → same result."""
        params = QuantumCatParams(n_clones=3, n_mutations=100, seed=123)
        r1 = simulate_quantumcat(params)
        r2 = simulate_quantumcat(params)

        np.testing.assert_array_equal(r1.alt, r2.alt)
        np.testing.assert_array_equal(r1.depth, r2.depth)
        np.testing.assert_array_equal(r1.true_assignments, r2.true_assignments)

    def test_different_seeds(self):
        """Different seeds → different results."""
        p1 = QuantumCatParams(n_clones=2, n_mutations=100, seed=1)
        p2 = QuantumCatParams(n_clones=2, n_mutations=100, seed=2)
        r1 = simulate_quantumcat(p1)
        r2 = simulate_quantumcat(p2)

        assert not np.array_equal(r1.alt, r2.alt)

    def test_cn_adjustment(self):
        """adj_factor should deviate from 1.0 for some mutations."""
        params = QuantumCatParams(n_clones=2, n_mutations=500, seed=42)
        result = simulate_quantumcat(params)

        # Some mutations should have adj != 1 (CN-altered)
        not_one = np.abs(result.adj_factor[:, 0] - 1.0) > 0.01
        assert np.sum(not_one) > 0

    def test_single_clone(self):
        """Single clone case should work and have no tree."""
        params = QuantumCatParams(n_clones=1, n_mutations=100, seed=42)
        result = simulate_quantumcat(params)

        assert result.true_centers.shape == (1, 1)
        assert result.true_tree is None
        assert np.all(result.true_assignments == 0)

    def test_clone_fractions(self):
        """Custom clone fractions should be respected."""
        params = QuantumCatParams(
            n_clones=3,
            n_mutations=300,
            clone_fractions=np.array([0.5, 0.3, 0.2]),
            seed=42,
        )
        result = simulate_quantumcat(params)

        _, counts = np.unique(result.true_assignments, return_counts=True)
        # Approximate: 150, 90, 60
        assert counts[0] == 150
        assert counts[1] == 90
        assert counts[2] == 60

    def test_depth_positive(self):
        """All depths should be positive."""
        params = QuantumCatParams(n_clones=2, n_mutations=100, depth=20, seed=42)
        result = simulate_quantumcat(params)
        assert np.all(result.depth > 0)

    def test_clonal_cluster_at_purity(self):
        """First clone center should equal purity."""
        params = QuantumCatParams(n_clones=3, n_mutations=100, purity=0.7, seed=42)
        result = simulate_quantumcat(params)
        np.testing.assert_allclose(result.true_centers[0, :], 0.7)


class TestNeutralTailLabels:
    def test_tail_mutations_get_dedicated_label(self):
        """Neutral tail mutations should NOT be labelled as clone 0."""
        noise = NoiseParams(neutral_tail_frac=0.3, neutral_tail_shape=1.0)
        params = QuantumCatParams(
            n_clones=2,
            n_mutations=200,
            purity=0.8,
            seed=42,
            noise=noise,
        )
        result = simulate_quantumcat(params)

        K = params.n_clones
        # Tail mutations should have label == K (not 0)
        tail_mask = result.true_assignments >= K
        assert tail_mask.sum() > 0, "Expected neutral tail mutations"
        # No tail mutation should be labelled as a real clone
        assert np.all(result.true_assignments[tail_mask] == K)

    def test_no_tail_means_no_extra_labels(self):
        """Without neutral tail, all labels should be in [0, K-1]."""
        params = QuantumCatParams(n_clones=3, n_mutations=200, purity=0.8, seed=42)
        result = simulate_quantumcat(params)
        assert result.true_assignments.max() < params.n_clones


class TestAugmentResult:
    def test_augment_preserves_clone_consistency(self):
        """After augmentation + subsampling, n_clones should match actual data."""
        rng = np.random.default_rng(42)
        # Use many clones + few mutations to make clone orphaning likely
        params = QuantumCatParams(
            n_clones=5,
            n_mutations=100,
            purity=0.8,
            seed=42,
            clone_fractions=np.array([0.4, 0.3, 0.15, 0.10, 0.05]),
        )
        result = simulate_quantumcat(params)

        # Try many augmentations — some should trigger subsampling
        for i in range(50):
            aug = augment_result(result, np.random.default_rng(i))
            unique_labels = np.unique(aug.true_assignments[aug.true_assignments >= 0])
            assert aug.params.n_clones == len(unique_labels), (
                f"n_clones={aug.params.n_clones} but {len(unique_labels)} unique labels"
            )
            assert aug.true_centers.shape[0] == aug.params.n_clones
            # All valid labels should be in [0, n_clones-1]
            assert unique_labels.max() < aug.params.n_clones


class TestSampleTumourParams:
    def test_returns_params(self):
        rng = np.random.default_rng(42)
        params = sample_tumour_params(rng)
        assert isinstance(params, QuantumCatParams)
        assert 1 <= params.n_clones <= 5
        assert params.n_mutations >= 100
        assert 0 < params.purity <= 1
        assert params.depth >= 20

    def test_reproducibility(self):
        p1 = sample_tumour_params(np.random.default_rng(42))
        p2 = sample_tumour_params(np.random.default_rng(42))
        assert p1.n_clones == p2.n_clones
        assert p1.n_mutations == p2.n_mutations
        assert p1.purity == p2.purity
