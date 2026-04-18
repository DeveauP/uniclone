"""
Tests for CloneConfig, enums, validate(), and CONFIGS presets.
"""
from __future__ import annotations

import pytest

from uniclone.core.config import (
    CONFIGS,
    CloneConfig,
    EmissionModel,
    InferenceEngine,
    KPrior,
    NoiseModel,
    PhyloMode,
)

# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------


class TestCloneConfigDefaults:
    def test_default_emission(self) -> None:
        assert CloneConfig().emission is EmissionModel.BETA_BINOMIAL

    def test_default_inference(self) -> None:
        assert CloneConfig().inference is InferenceEngine.HYBRID

    def test_default_k_prior(self) -> None:
        assert CloneConfig().k_prior is KPrior.DIRICHLET

    def test_default_phylo(self) -> None:
        assert CloneConfig().phylo is PhyloMode.CONSTRAINED

    def test_default_noise(self) -> None:
        assert CloneConfig().noise is NoiseModel.ARTEFACT

    def test_default_phi_is_none(self) -> None:
        assert CloneConfig().phi is None

    def test_default_n_clones_is_none(self) -> None:
        assert CloneConfig().n_clones is None

    def test_default_n_samples(self) -> None:
        assert CloneConfig().n_samples == 1

    def test_default_longitudinal(self) -> None:
        assert CloneConfig().longitudinal is False

    def test_default_depth_median(self) -> None:
        assert CloneConfig().depth_median == 100.0


# ---------------------------------------------------------------------------
# Validation rules
# ---------------------------------------------------------------------------


class TestCloneConfigValidation:
    def test_fixed_k_requires_n_clones(self) -> None:
        with pytest.raises(ValueError, match="n_clones"):
            CloneConfig(k_prior=KPrior.FIXED)

    def test_fixed_k_with_n_clones_ok(self) -> None:
        cfg = CloneConfig(k_prior=KPrior.FIXED, n_clones=3)
        assert cfg.n_clones == 3

    def test_longitudinal_phylo_requires_flag(self) -> None:
        with pytest.raises(ValueError, match="longitudinal=True"):
            CloneConfig(phylo=PhyloMode.LONGITUDINAL)

    def test_longitudinal_phylo_with_flag_ok(self) -> None:
        cfg = CloneConfig(phylo=PhyloMode.LONGITUDINAL, longitudinal=True)
        assert cfg.longitudinal is True

    def test_joint_mcmc_requires_mcmc_or_hybrid(self) -> None:
        with pytest.raises(ValueError, match="MCMC or HYBRID"):
            CloneConfig(phylo=PhyloMode.JOINT_MCMC, inference=InferenceEngine.EM)

    def test_joint_mcmc_with_mcmc_ok(self) -> None:
        cfg = CloneConfig(
            phylo=PhyloMode.JOINT_MCMC, inference=InferenceEngine.MCMC,
            k_prior=KPrior.TSSB,
        )
        assert cfg.phylo is PhyloMode.JOINT_MCMC

    def test_joint_mcmc_with_hybrid_ok(self) -> None:
        cfg = CloneConfig(phylo=PhyloMode.JOINT_MCMC, inference=InferenceEngine.HYBRID)
        assert cfg.inference is InferenceEngine.HYBRID

    def test_negative_phi_raises(self) -> None:
        with pytest.raises(ValueError, match="phi must be positive"):
            CloneConfig(phi=-1.0)

    def test_zero_phi_raises(self) -> None:
        with pytest.raises(ValueError, match="phi must be positive"):
            CloneConfig(phi=0.0)

    def test_positive_phi_ok(self) -> None:
        cfg = CloneConfig(phi=50.0)
        assert cfg.phi == 50.0

    @pytest.mark.parametrize("tail_weight", [0.0, 1.0, -0.1, 1.5])
    def test_invalid_tail_weight(self, tail_weight: float) -> None:
        with pytest.raises(ValueError, match="tail_weight must be in"):
            CloneConfig(tail_weight=tail_weight)

    def test_valid_tail_weight(self) -> None:
        cfg = CloneConfig(tail_weight=0.3)
        assert cfg.tail_weight == 0.3

    def test_zero_n_samples_raises(self) -> None:
        with pytest.raises(ValueError, match="n_samples must be"):
            CloneConfig(n_samples=0)

    def test_negative_depth_median_raises(self) -> None:
        with pytest.raises(ValueError, match="depth_median must be positive"):
            CloneConfig(depth_median=-10.0)

    def test_zero_depth_median_raises(self) -> None:
        with pytest.raises(ValueError, match="depth_median must be positive"):
            CloneConfig(depth_median=0.0)


# ---------------------------------------------------------------------------
# CONFIGS presets
# ---------------------------------------------------------------------------


EXPECTED_CONFIG_NAMES = {
    "quantumclone_v1",
    "pyclone_vi",
    "mobster",
    "decifer_style",
    "conipher_style",
    "phyloclone_style",
    "calder_style",
    "wes_clinical",
    "wgs_cohort",
    "wgs_cohort_phylo",
    "multiregion_phylo",
    "longitudinal_clinical",
}


class TestCONFIGSDict:
    def test_all_12_configs_present(self) -> None:
        assert set(CONFIGS.keys()) == EXPECTED_CONFIG_NAMES

    def test_all_configs_are_cloneconfig_instances(self) -> None:
        for name, config in CONFIGS.items():
            assert isinstance(config, CloneConfig), f"{name} is not a CloneConfig"

    def test_quantumclone_v1_axes(self) -> None:
        c = CONFIGS["quantumclone_v1"]
        assert c.emission is EmissionModel.BINOMIAL
        assert c.inference is InferenceEngine.EM
        assert c.k_prior is KPrior.BIC
        assert c.phylo is PhyloMode.POST_HOC
        assert c.noise is NoiseModel.NONE

    def test_pyclone_vi_axes(self) -> None:
        c = CONFIGS["pyclone_vi"]
        assert c.emission is EmissionModel.BETA_BINOMIAL
        assert c.inference is InferenceEngine.MFVI
        assert c.phi == 50.0

    def test_mobster_axes(self) -> None:
        c = CONFIGS["mobster"]
        assert c.emission is EmissionModel.BB_PARETO
        assert c.noise is NoiseModel.TAIL_FILTER

    def test_calder_style_longitudinal(self) -> None:
        c = CONFIGS["calder_style"]
        assert c.longitudinal is True
        assert c.phylo is PhyloMode.LONGITUDINAL

    def test_wes_clinical_phi_estimated(self) -> None:
        c = CONFIGS["wes_clinical"]
        assert c.phi is None

    def test_wgs_cohort_phi_fixed(self) -> None:
        c = CONFIGS["wgs_cohort"]
        assert c.phi == 100.0

    def test_all_configs_validated_at_import(self) -> None:
        # All configs are constructed at import time; this test verifies
        # none of them raise ValidationError (they were created successfully).
        assert len(CONFIGS) == 12

    def test_wgs_cohort_phylo_axes(self) -> None:
        c = CONFIGS["wgs_cohort_phylo"]
        assert c.emission is EmissionModel.BB_PARETO
        assert c.inference is InferenceEngine.HYBRID
        assert c.k_prior is KPrior.DIRICHLET
        assert c.phylo is PhyloMode.CONSTRAINED
        assert c.noise is NoiseModel.TAIL_FILTER
        assert c.phi == 100.0

    def test_repr_is_informative(self) -> None:
        r = repr(CONFIGS["quantumclone_v1"])
        assert "BINOMIAL" in r
        assert "EM" in r
