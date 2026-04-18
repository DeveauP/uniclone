"""
Tests for noise modules: TailFilterNoise, ArtefactNoise, MultiplicityNoise,
and the shared expand_result utility.
"""
from __future__ import annotations

import numpy as np
import pytest

from uniclone.core.config import (
    CloneConfig,
    EmissionModel,
    InferenceEngine,
    KPrior,
    NoiseModel,
    PhyloMode,
)
from uniclone.core.types import CloneResult
from uniclone.noise._utils import expand_result
from uniclone.noise.artefact import ArtefactNoise
from uniclone.noise.multiplicity import MultiplicityNoise
from uniclone.noise.tail_filter import TailFilterNoise

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> CloneConfig:
    """Create a CloneConfig with sensible defaults for noise testing."""
    defaults = dict(
        emission=EmissionModel.BETA_BINOMIAL,
        inference=InferenceEngine.EM,
        k_prior=KPrior.BIC,
        phylo=PhyloMode.NONE,
        noise=NoiseModel.NONE,
        phi=50.0,
    )
    defaults.update(overrides)
    return CloneConfig(**defaults)


def _make_result(n_mut: int, K: int, n_samples: int = 1) -> CloneResult:
    """Build a minimal CloneResult for testing post-processing."""
    rng = np.random.default_rng(0)
    centers = rng.uniform(0.1, 0.9, size=(K, n_samples))
    responsibilities = rng.dirichlet(np.ones(K), size=n_mut)
    assignments = responsibilities.argmax(axis=1)
    return CloneResult(
        centers=centers,
        assignments=assignments,
        responsibilities=responsibilities,
        log_likelihood=-100.0,
        bic=200.0,
        K=K,
        n_iter=10,
        converged=True,
    )


# ---------------------------------------------------------------------------
# expand_result tests
# ---------------------------------------------------------------------------

class TestExpandResult:
    def test_expand_identity(self) -> None:
        """All-True mask returns unchanged dimensions."""
        result = _make_result(50, 3)
        mask = np.ones(50, dtype=bool)
        out = expand_result(result, mask)
        assert out.assignments.shape == (50,)
        assert out.responsibilities.shape == (50, 3)
        assert out.noise_mask is not None
        assert out.noise_mask.all()

    def test_expand_filters_some(self) -> None:
        """Filtered mutations get assignment -1 and zero responsibilities."""
        mask = np.array([True, True, False, True, False])
        result = _make_result(3, 2)  # 3 kept mutations
        out = expand_result(result, mask)
        assert out.assignments.shape == (5,)
        assert out.responsibilities.shape == (5, 2)
        # Filtered positions
        assert out.assignments[2] == -1
        assert out.assignments[4] == -1
        np.testing.assert_array_equal(out.responsibilities[2], 0.0)
        np.testing.assert_array_equal(out.responsibilities[4], 0.0)
        # Kept positions preserved
        assert out.assignments[0] >= 0
        assert out.assignments[1] >= 0
        assert out.assignments[3] >= 0

    def test_expand_custom_label(self) -> None:
        mask = np.array([False, True])
        result = _make_result(1, 2)
        out = expand_result(result, mask, noise_label=-99)
        assert out.assignments[0] == -99


# ---------------------------------------------------------------------------
# TailFilterNoise tests
# ---------------------------------------------------------------------------

class TestTailFilterNoise:
    def test_instantiation(self) -> None:
        config = _make_config(noise=NoiseModel.TAIL_FILTER, tail_threshold=0.5)
        tf = TailFilterNoise(config)
        assert tf.threshold == 0.5

    def test_preprocess_returns_correct_shapes(self) -> None:
        config = _make_config(tail_threshold=0.5, phi=50.0, tail_weight=0.1)
        tf = TailFilterNoise(config)
        rng = np.random.default_rng(42)
        alt = rng.binomial(100, 0.3, size=(100, 1)).astype(float)
        depth = np.full((100, 1), 100.0)
        adj = np.ones((100, 1))
        alt_f, depth_f, adj_f, mask = tf.preprocess(alt, depth, adj)
        assert mask.shape == (100,)
        assert alt_f.shape[0] == mask.sum()
        assert depth_f.shape[0] == mask.sum()
        assert adj_f.shape[0] == mask.sum()

    def test_high_vaf_mutations_kept(self) -> None:
        """Clonal mutations (high VAF) should mostly be kept."""
        config = _make_config(tail_threshold=0.5, phi=50.0, tail_weight=0.1)
        tf = TailFilterNoise(config)
        rng = np.random.default_rng(42)
        # All mutations at VAF ~0.5 (clonal)
        alt = rng.binomial(100, 0.5, size=(50, 1)).astype(float)
        depth = np.full((50, 1), 100.0)
        adj = np.ones((50, 1))
        _, _, _, mask = tf.preprocess(alt, depth, adj)
        # Most should be kept
        assert mask.sum() >= 40

    def test_low_vaf_tail_mutations_filtered(self) -> None:
        """Very low VAF mutations (tail-like) should be filtered with generous params."""
        config = _make_config(tail_threshold=0.3, phi=50.0, tail_weight=0.5)
        tf = TailFilterNoise(config)
        rng = np.random.default_rng(42)
        # Mix: 80 clonal + 20 very low VAF (tail)
        alt_clonal = rng.binomial(100, 0.4, size=(80, 1)).astype(float)
        alt_tail = rng.binomial(100, 0.01, size=(20, 1)).astype(float)
        alt = np.vstack([alt_clonal, alt_tail])
        depth = np.full((100, 1), 100.0)
        adj = np.ones((100, 1))
        _, _, _, mask = tf.preprocess(alt, depth, adj)
        # Some tail mutations should be filtered
        assert mask.sum() < 100

    def test_guard_too_few_remaining(self) -> None:
        """If filtering would leave < 2 mutations, keep all."""
        config = _make_config(tail_threshold=0.001, phi=50.0, tail_weight=0.99)
        tf = TailFilterNoise(config)
        # 3 mutations all at low VAF — aggressive tail weight
        alt = np.array([[1], [2], [1]], dtype=float)
        depth = np.full((3, 1), 100.0)
        adj = np.ones((3, 1))
        _, _, _, mask = tf.preprocess(alt, depth, adj)
        # Guard: should keep all
        assert mask.sum() >= 2

    def test_postprocess_expands_result(self) -> None:
        config = _make_config(tail_threshold=0.5)
        tf = TailFilterNoise(config)
        mask = np.array([True, True, False, True, False])
        result = _make_result(3, 2)
        out = tf.postprocess(result, mask)
        assert out.assignments.shape == (5,)
        assert out.noise_mask is not None

    def test_postprocess_all_kept(self) -> None:
        config = _make_config(tail_threshold=0.5)
        tf = TailFilterNoise(config)
        mask = np.ones(5, dtype=bool)
        result = _make_result(5, 2)
        out = tf.postprocess(result, mask)
        assert out.noise_mask is not None
        assert out.noise_mask.all()

    def test_multisample(self) -> None:
        """Multi-sample: tail in ANY sample triggers masking."""
        config = _make_config(tail_threshold=0.3, phi=50.0, tail_weight=0.5)
        tf = TailFilterNoise(config)
        rng = np.random.default_rng(42)
        n = 60
        depth = np.full((n, 2), 100.0)
        alt = np.zeros((n, 2))
        # Sample 1: all clonal
        alt[:, 0] = rng.binomial(100, 0.4, size=n)
        # Sample 2: first 40 clonal, last 20 tail
        alt[:40, 1] = rng.binomial(100, 0.4, size=40)
        alt[40:, 1] = rng.binomial(100, 0.01, size=20)
        adj = np.ones((n, 2))
        _, _, _, mask = tf.preprocess(alt, depth, adj)
        assert mask.shape == (n,)

    def test_estimate_phi(self) -> None:
        """Internal phi estimator should return a reasonable value."""
        rng = np.random.default_rng(42)
        vaf = rng.binomial(100, 0.3, size=200).astype(float) / 100.0
        phi = TailFilterNoise._estimate_phi_from_vaf(vaf)
        assert 1.0 <= phi <= 1000.0


# ---------------------------------------------------------------------------
# ArtefactNoise tests
# ---------------------------------------------------------------------------

class TestArtefactNoise:
    def test_instantiation(self) -> None:
        config = _make_config(noise=NoiseModel.ARTEFACT, artefact_absence_threshold=0.05)
        an = ArtefactNoise(config)
        assert an.absence_threshold == 0.05

    def test_preprocess_identity(self) -> None:
        """Preprocess is identity — all mutations kept."""
        config = _make_config()
        an = ArtefactNoise(config)
        alt = np.ones((10, 2))
        depth = np.ones((10, 2))
        adj = np.ones((10, 2))
        alt_f, depth_f, adj_f, mask = an.preprocess(alt, depth, adj)
        np.testing.assert_array_equal(alt_f, alt)
        assert mask.all()

    def test_single_sample_noop(self) -> None:
        """Single-sample: postprocess returns unchanged."""
        config = _make_config()
        an = ArtefactNoise(config)
        result = _make_result(50, 3, n_samples=1)
        mask = np.ones(50, dtype=bool)
        out = an.postprocess(result, mask)
        assert out.noise_mask is not None

    def test_nestable_clusters_kept(self) -> None:
        """Clusters with nestable presence patterns should all be kept."""
        config = _make_config(artefact_absence_threshold=0.05)
        an = ArtefactNoise(config)
        result = _make_result(60, 3, n_samples=2)
        # Make all clusters present in both samples (perfectly nestable)
        result.centers = np.array([[0.6, 0.5], [0.3, 0.2], [0.1, 0.1]])
        mask = np.ones(60, dtype=bool)
        out = an.postprocess(result, mask)
        # All assignments should be unchanged (no artefact detected)
        assert not np.any(out.centers == 0.0)

    def test_all_clusters_nestable_with_clonal(self) -> None:
        """With a clonal cluster present in all samples, all sub-clusters
        are nestable (subset), so no artefacts are detected."""
        config = _make_config(artefact_absence_threshold=0.05)
        an = ArtefactNoise(config)
        result = _make_result(60, 3, n_samples=2)
        result.centers = np.array([
            [0.6, 0.5],   # present in both (clonal)
            [0.3, 0.01],  # present in sample 0 only → subset of cluster 0
            [0.01, 0.3],  # present in sample 1 only → subset of cluster 0
        ])
        mask = np.ones(60, dtype=bool)
        out = an.postprocess(result, mask)
        # All nestable with the clonal cluster → no artefacts
        assert not np.any(out.centers == 0.0)

    def test_artefact_cluster_detected(self) -> None:
        """A cluster whose presence pattern is not a subset/superset of
        ANY other cluster is detected as an artefact."""
        config = _make_config(artefact_absence_threshold=0.05)
        an = ArtefactNoise(config)
        # 3 clusters, 3 samples:
        # Cluster 0: [T, T, F] — present in samples 0, 1
        # Cluster 1: [T, F, F] — subset of cluster 0 → nestable
        # Cluster 2: [F, T, T] — not subset/superset of 0 or 1 → artefact
        result = _make_result(60, 3, n_samples=3)
        result.centers = np.array([
            [0.3, 0.3, 0.01],   # present in samples 0, 1
            [0.2, 0.01, 0.01],  # present in sample 0 only (subset of cluster 0)
            [0.01, 0.2, 0.2],   # present in samples 1, 2 (not nestable with 0 or 1)
        ])
        result.assignments[:20] = 0
        result.assignments[20:40] = 1
        result.assignments[40:] = 2
        mask = np.ones(60, dtype=bool)
        out = an.postprocess(result, mask)
        # Cluster 2 should be zeroed out
        np.testing.assert_array_equal(out.centers[2], 0.0)

    def test_artefact_mutations_reassigned(self) -> None:
        """Mutations from artefact clusters should be reassigned to valid ones."""
        config = _make_config(artefact_absence_threshold=0.05)
        an = ArtefactNoise(config)
        result = _make_result(30, 3, n_samples=3)
        result.centers = np.array([
            [0.3, 0.3, 0.01],   # valid (nestable with cluster 1)
            [0.2, 0.01, 0.01],  # valid (nestable with cluster 0)
            [0.01, 0.2, 0.2],   # artefact
        ])
        result.assignments[:10] = 0
        result.assignments[10:20] = 1
        result.assignments[20:] = 2
        mask = np.ones(30, dtype=bool)
        out = an.postprocess(result, mask)
        # Artefact cluster mutations should now point to a valid cluster
        for i in range(20, 30):
            assert out.assignments[i] in [0, 1]

    def test_all_artefacts_guard(self) -> None:
        """When all clusters are non-nestable, guard returns unchanged."""
        config = _make_config(artefact_absence_threshold=0.05)
        an = ArtefactNoise(config)
        result = _make_result(20, 2, n_samples=2)
        # Neither is subset of the other → both artefacts
        result.centers = np.array([
            [0.3, 0.01],
            [0.01, 0.3],
        ])
        orig_assignments = result.assignments.copy()
        mask = np.ones(20, dtype=bool)
        out = an.postprocess(result, mask)
        # Guard: all artefacts → return unchanged
        np.testing.assert_array_equal(out.assignments, orig_assignments)


# ---------------------------------------------------------------------------
# MultiplicityNoise tests
# ---------------------------------------------------------------------------

class TestMultiplicityNoise:
    def test_instantiation(self) -> None:
        config = _make_config(noise=NoiseModel.MULTIPLICITY, purity=0.8)
        mn = MultiplicityNoise(config)
        assert mn.purity == 0.8

    def test_preprocess_no_cn_identity(self) -> None:
        """Without CN state, preprocess is identity."""
        config = _make_config()
        mn = MultiplicityNoise(config)
        alt = np.ones((10, 1))
        depth = np.full((10, 1), 100.0)
        adj = np.ones((10, 1))
        alt_f, depth_f, adj_f, mask = mn.preprocess(alt, depth, adj)
        np.testing.assert_array_equal(adj_f, adj)
        assert mask.all()

    def test_set_cn_state(self) -> None:
        config = _make_config()
        mn = MultiplicityNoise(config)
        cn = np.array([2, 3, 4])
        mn.set_cn_state(cn)
        assert mn._total_cn is not None
        np.testing.assert_array_equal(mn._total_cn, cn)

    def test_set_cn_state_with_minor(self) -> None:
        config = _make_config()
        mn = MultiplicityNoise(config)
        mn.set_cn_state(np.array([2, 3]), minor_cn=np.array([1, 1]))
        assert mn._minor_cn is not None

    def test_preprocess_with_cn_updates_adj(self) -> None:
        """CN state should modify adj_factor."""
        config = _make_config(purity=1.0)
        mn = MultiplicityNoise(config)
        mn.set_cn_state(np.array([2, 4]))
        alt = np.array([[30], [25]], dtype=float)
        depth = np.full((2, 1), 100.0)
        adj = np.ones((2, 1))
        _, _, adj_f, mask = mn.preprocess(alt, depth, adj)
        # adj_factor should have been updated
        assert not np.allclose(adj_f, adj)
        assert mask.all()

    def test_multiplicity_diploid_purity1(self) -> None:
        """Diploid CN=2, purity=1: m=1 → adj=0.5."""
        config = _make_config(purity=1.0)
        mn = MultiplicityNoise(config)
        mn.set_cn_state(np.array([2]))
        # VAF ~0.3 → implied CCF = 0.3/0.5 = 0.6 for m=1
        alt = np.array([[30]], dtype=float)
        depth = np.full((1, 1), 100.0)
        adj = np.ones((1, 1))
        _, _, adj_f, _ = mn.preprocess(alt, depth, adj)
        # m=1: adj = 1/(2*1 + 2*0) = 0.5
        assert np.isclose(adj_f[0, 0], 0.5)

    def test_multiplicity_tetraploid(self) -> None:
        """CN=4, purity=1, VAF~0.5: should pick m=2 → adj=0.5."""
        config = _make_config(purity=1.0)
        mn = MultiplicityNoise(config)
        mn.set_cn_state(np.array([4]))
        alt = np.array([[50]], dtype=float)
        depth = np.full((1, 1), 100.0)
        adj = np.ones((1, 1))
        _, _, adj_f, _ = mn.preprocess(alt, depth, adj)
        # m=2: adj = 2/4 = 0.5, implied CCF = 0.5/0.5 = 1.0 (perfect)
        assert np.isclose(adj_f[0, 0], 0.5)

    def test_postprocess_sets_mask(self) -> None:
        config = _make_config()
        mn = MultiplicityNoise(config)
        result = _make_result(10, 2)
        mask = np.ones(10, dtype=bool)
        out = mn.postprocess(result, mask)
        assert out.noise_mask is not None
        assert out.noise_mask.all()

    def test_multisample_cn(self) -> None:
        """Per-sample CN state should work."""
        config = _make_config(purity=1.0)
        mn = MultiplicityNoise(config)
        mn.set_cn_state(np.array([[2, 4]]))
        alt = np.array([[30, 25]], dtype=float)
        depth = np.full((1, 2), 100.0)
        adj = np.ones((1, 2))
        _, _, adj_f, mask = mn.preprocess(alt, depth, adj)
        assert mask.all()
        assert adj_f.shape == (1, 2)


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------

class TestNoiseModuleProtocol:
    """All three modules satisfy the NoiseModule protocol."""

    @pytest.mark.parametrize("cls,kw", [
        (TailFilterNoise, {}),
        (ArtefactNoise, {}),
        (MultiplicityNoise, {}),
    ])
    def test_protocol_compliance(self, cls, kw) -> None:
        from uniclone.core.types import NoiseModule
        config = _make_config()
        obj = cls(config, **kw)
        assert isinstance(obj, NoiseModule)


# ---------------------------------------------------------------------------
# Integration: GenerativeModel with noise configs
# ---------------------------------------------------------------------------

class TestNoiseIntegration:
    """Verify that noise-enabled configs can construct and run GenerativeModel."""

    def test_mobster_config_runs(self) -> None:
        from uniclone import CONFIGS, GenerativeModel
        model = GenerativeModel(CONFIGS["mobster"])
        rng = np.random.default_rng(42)
        alt = rng.binomial(100, 0.3, size=(100, 1)).astype(float)
        depth = np.full((100, 1), 100.0)
        result = model.fit(alt, depth)
        assert result.assignments.shape == (100,)
        assert result.noise_mask is not None

    def test_conipher_config_runs(self) -> None:
        from uniclone import CONFIGS, GenerativeModel
        model = GenerativeModel(CONFIGS["conipher_style"])
        rng = np.random.default_rng(42)
        alt = rng.binomial(100, 0.3, size=(100, 1)).astype(float)
        depth = np.full((100, 1), 100.0)
        result = model.fit(alt, depth)
        assert result.assignments.shape == (100,)

    def test_decifer_config_runs(self) -> None:
        from uniclone import CONFIGS, GenerativeModel
        model = GenerativeModel(CONFIGS["decifer_style"])
        rng = np.random.default_rng(42)
        alt = rng.binomial(100, 0.3, size=(100, 1)).astype(float)
        depth = np.full((100, 1), 100.0)
        result = model.fit(alt, depth)
        assert result.assignments.shape == (100,)

    def test_wes_clinical_config_runs(self) -> None:
        from uniclone import CONFIGS, GenerativeModel
        model = GenerativeModel(CONFIGS["wes_clinical"])
        rng = np.random.default_rng(42)
        alt = rng.binomial(100, 0.3, size=(100, 1)).astype(float)
        depth = np.full((100, 1), 100.0)
        result = model.fit(alt, depth)
        assert result.assignments.shape == (100,)

    def test_model_fit_with_cn_state(self) -> None:
        from uniclone import CONFIGS, GenerativeModel
        model = GenerativeModel(CONFIGS["decifer_style"])
        rng = np.random.default_rng(42)
        alt = rng.binomial(100, 0.3, size=(50, 1)).astype(float)
        depth = np.full((50, 1), 100.0)
        cn_state = {"total_cn": np.full(50, 2)}
        result = model.fit(alt, depth, cn_state=cn_state)
        assert result.assignments.shape == (50,)
