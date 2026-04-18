"""
Tests for emission modules.

Validates correctness against scipy reference implementations, edge cases,
multi-sample behaviour, and cross-module consistency.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import betabinom, binom

from uniclone.core.config import (
    CONFIGS,
    CloneConfig,
    EmissionModel,
    InferenceEngine,
    KPrior,
    NoiseModel,
    PhyloMode,
)
from uniclone.core.types import EmissionModule
from uniclone.emission.bb_pareto import BBParetoEmission
from uniclone.emission.beta_binomial import BetaBinomialEmission
from uniclone.emission.binomial import BinomialEmission
from uniclone.emission.dcf import DCFBBEmission

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def emission() -> BinomialEmission:
    return BinomialEmission(CONFIGS["quantumclone_v1"])


@pytest.fixture
def bb_emission() -> BetaBinomialEmission:
    return BetaBinomialEmission(CONFIGS["pyclone_vi"])


@pytest.fixture
def bb_pareto_emission() -> BBParetoEmission:
    return BBParetoEmission(CONFIGS["mobster"])


@pytest.fixture
def dcf_emission() -> DCFBBEmission:
    return DCFBBEmission(CONFIGS["decifer_style"])


# ---------------------------------------------------------------------------
# Binomial tests
# ---------------------------------------------------------------------------

class TestBinomialLogProb:
    def test_single_value_matches_scipy(self, emission: BinomialEmission) -> None:
        alt = np.array([[10.0]])
        depth = np.array([[50.0]])
        mu = np.array([[0.3]])
        result = emission.log_prob(alt, depth, mu)
        expected = binom.logpmf(10, 50, 0.3)
        np.testing.assert_allclose(result[0], expected, rtol=1e-6)

    @pytest.mark.parametrize("k,n,p", [
        (0, 50, 0.3),
        (10, 50, 0.3),
        (25, 50, 0.5),
        (50, 50, 0.9),
        (1, 100, 0.01),
        (99, 100, 0.99),
        (50, 100, 0.5),
    ])
    def test_matches_scipy_parametrize(
        self, emission: BinomialEmission, k: int, n: int, p: float
    ) -> None:
        alt = np.array([[float(k)]])
        depth = np.array([[float(n)]])
        mu = np.array([[p]])
        result = emission.log_prob(alt, depth, mu)
        expected = binom.logpmf(k, n, p)
        np.testing.assert_allclose(result[0], expected, rtol=1e-6)

    def test_no_nan_when_alt_zero(self, emission: BinomialEmission) -> None:
        alt = np.array([[0.0]])
        depth = np.array([[100.0]])
        mu = np.array([[0.5]])
        result = emission.log_prob(alt, depth, mu)
        assert np.isfinite(result).all()

    def test_no_nan_when_alt_equals_depth(self, emission: BinomialEmission) -> None:
        alt = np.array([[100.0]])
        depth = np.array([[100.0]])
        mu = np.array([[0.5]])
        result = emission.log_prob(alt, depth, mu)
        assert np.isfinite(result).all()

    def test_no_nan_when_mu_near_zero(self, emission: BinomialEmission) -> None:
        alt = np.array([[0.0]])
        depth = np.array([[100.0]])
        mu = np.array([[1e-12]])
        result = emission.log_prob(alt, depth, mu)
        assert np.isfinite(result).all()

    def test_no_nan_when_mu_near_one(self, emission: BinomialEmission) -> None:
        alt = np.array([[100.0]])
        depth = np.array([[100.0]])
        mu = np.array([[1 - 1e-12]])
        result = emission.log_prob(alt, depth, mu)
        assert np.isfinite(result).all()

    def test_output_shape_single_mutation(self, emission: BinomialEmission) -> None:
        alt = np.array([[10.0]])
        depth = np.array([[50.0]])
        mu = np.array([[0.3]])
        result = emission.log_prob(alt, depth, mu)
        assert result.shape == (1,)

    def test_output_shape_multiple_mutations(self, emission: BinomialEmission) -> None:
        n_mut = 100
        alt = np.full((n_mut, 1), 10.0)
        depth = np.full((n_mut, 1), 50.0)
        mu = np.full((n_mut, 1), 0.3)
        result = emission.log_prob(alt, depth, mu)
        assert result.shape == (n_mut,)

    def test_multisample_sums_log_probs(self, emission: BinomialEmission) -> None:
        """Multi-sample result == sum of single-sample results."""
        alt_s1 = np.array([[10.0]])
        alt_s2 = np.array([[20.0]])
        depth_s1 = np.array([[50.0]])
        depth_s2 = np.array([[80.0]])
        mu_s1 = np.array([[0.3]])
        mu_s2 = np.array([[0.25]])

        lp_s1 = emission.log_prob(alt_s1, depth_s1, mu_s1)[0]
        lp_s2 = emission.log_prob(alt_s2, depth_s2, mu_s2)[0]

        alt_both = np.array([[10.0, 20.0]])
        depth_both = np.array([[50.0, 80.0]])
        mu_both = np.array([[0.3, 0.25]])

        lp_both = emission.log_prob(alt_both, depth_both, mu_both)[0]

        np.testing.assert_allclose(lp_both, lp_s1 + lp_s2, rtol=1e-10)

    def test_log_prob_is_negative(self, emission: BinomialEmission) -> None:
        """Log probability of any observation is <= 0."""
        rng = np.random.default_rng(0)
        alt = rng.integers(0, 50, size=(20, 1)).astype(float)
        depth = np.full((20, 1), 50.0)
        mu = rng.uniform(0.1, 0.9, size=(20, 1))
        result = emission.log_prob(alt, depth, mu)
        assert (result <= 0).all()

    def test_mode_has_highest_log_prob(self, emission: BinomialEmission) -> None:
        """The mode p=k/n gives the highest log prob among nearby values."""
        n = 100
        k = 30
        depth = np.array([[float(n)]])
        alt = np.array([[float(k)]])
        mu_mode = np.array([[k / n]])
        mu_off = np.array([[k / n + 0.1]])
        lp_mode = emission.log_prob(alt, depth, mu_mode)[0]
        lp_off = emission.log_prob(alt, depth, mu_off)[0]
        assert lp_mode > lp_off


# ---------------------------------------------------------------------------
# Beta-Binomial tests
# ---------------------------------------------------------------------------

class TestBetaBinomialLogProb:
    @pytest.mark.parametrize("k,n,mu_val,phi", [
        (10, 50, 0.3, 50.0),
        (0, 50, 0.3, 50.0),
        (50, 50, 0.9, 100.0),
        (25, 100, 0.5, 200.0),
        (1, 100, 0.01, 30.0),
        (5, 20, 0.4, 10.0),
    ])
    def test_matches_scipy_betabinom(
        self, k: int, n: int, mu_val: float, phi: float
    ) -> None:
        """Validate against scipy.stats.betabinom.logpmf."""
        config = CloneConfig(
            emission=EmissionModel.BETA_BINOMIAL,
            inference=InferenceEngine.EM,
            k_prior=KPrior.BIC,
            phylo=PhyloMode.NONE,
            noise=NoiseModel.NONE,
            phi=phi,
        )
        em = BetaBinomialEmission(config)
        alt = np.array([[float(k)]])
        depth = np.array([[float(n)]])
        mu = np.array([[mu_val]])
        result = em.log_prob(alt, depth, mu)

        a = mu_val * phi
        b = (1.0 - mu_val) * phi
        expected = betabinom.logpmf(k, n, a, b)
        np.testing.assert_allclose(result[0], expected, rtol=1e-6)

    def test_large_phi_recovers_binomial(self) -> None:
        """As phi -> infinity, Beta-Binomial converges to Binomial."""
        config_bb = CloneConfig(
            emission=EmissionModel.BETA_BINOMIAL,
            inference=InferenceEngine.EM,
            k_prior=KPrior.BIC,
            phylo=PhyloMode.NONE,
            noise=NoiseModel.NONE,
            phi=1e6,
        )
        bb = BetaBinomialEmission(config_bb)
        binom_em = BinomialEmission(CONFIGS["quantumclone_v1"])

        rng = np.random.default_rng(42)
        alt = rng.integers(0, 50, size=(20, 1)).astype(float)
        depth = np.full((20, 1), 50.0)
        mu = rng.uniform(0.1, 0.9, size=(20, 1))

        bb_result = bb.log_prob(alt, depth, mu)
        binom_result = binom_em.log_prob(alt, depth, mu)
        np.testing.assert_allclose(bb_result, binom_result, rtol=1e-3)

    def test_no_nan_edge_cases(self, bb_emission: BetaBinomialEmission) -> None:
        """No NaN for alt=0, alt=depth, mu near 0/1."""
        cases = [
            (0.0, 100.0, 0.5),
            (100.0, 100.0, 0.5),
            (0.0, 100.0, 1e-8),
            (100.0, 100.0, 1.0 - 1e-8),
        ]
        for k, n, p in cases:
            alt = np.array([[k]])
            depth = np.array([[n]])
            mu = np.array([[p]])
            result = bb_emission.log_prob(alt, depth, mu)
            assert np.isfinite(result).all(), f"NaN for k={k}, n={n}, mu={p}"

    def test_output_shape(self, bb_emission: BetaBinomialEmission) -> None:
        n_mut = 50
        alt = np.full((n_mut, 2), 10.0)
        depth = np.full((n_mut, 2), 50.0)
        mu = np.full((n_mut, 2), 0.3)
        result = bb_emission.log_prob(alt, depth, mu)
        assert result.shape == (n_mut,)

    def test_multisample_sums(self, bb_emission: BetaBinomialEmission) -> None:
        """Multi-sample result == sum of single-sample results."""
        alt_s1 = np.array([[10.0]])
        alt_s2 = np.array([[20.0]])
        depth_s1 = np.array([[50.0]])
        depth_s2 = np.array([[80.0]])
        mu_s1 = np.array([[0.3]])
        mu_s2 = np.array([[0.25]])

        lp_s1 = bb_emission.log_prob(alt_s1, depth_s1, mu_s1)[0]
        lp_s2 = bb_emission.log_prob(alt_s2, depth_s2, mu_s2)[0]

        alt_both = np.array([[10.0, 20.0]])
        depth_both = np.array([[50.0, 80.0]])
        mu_both = np.array([[0.3, 0.25]])
        lp_both = bb_emission.log_prob(alt_both, depth_both, mu_both)[0]

        np.testing.assert_allclose(lp_both, lp_s1 + lp_s2, rtol=1e-10)

    def test_log_prob_is_negative(self, bb_emission: BetaBinomialEmission) -> None:
        rng = np.random.default_rng(0)
        alt = rng.integers(0, 50, size=(20, 1)).astype(float)
        depth = np.full((20, 1), 50.0)
        mu = rng.uniform(0.1, 0.9, size=(20, 1))
        result = bb_emission.log_prob(alt, depth, mu)
        assert (result <= 0).all()

    def test_default_phi_when_none(self) -> None:
        """phi=None in config should use default 100.0."""
        config = CloneConfig(
            emission=EmissionModel.BETA_BINOMIAL,
            inference=InferenceEngine.EM,
            k_prior=KPrior.BIC,
            phylo=PhyloMode.NONE,
            noise=NoiseModel.NONE,
            phi=None,
        )
        em = BetaBinomialEmission(config)
        assert em.phi == 100.0


# ---------------------------------------------------------------------------
# BB + Pareto tests
# ---------------------------------------------------------------------------

class TestBBParetoLogProb:
    def test_small_tau_close_to_beta_binomial(self) -> None:
        """With tau very small, BB+Pareto should be close to pure BB.

        Note: The Pareto density f(x)=(alpha-1)/x^alpha is unbounded near 0,
        so for mutations with very low observed VAF relative to expected mu,
        even tiny tau can shift the log-prob noticeably.  We use mutations
        where observed VAF is close to mu to avoid this regime.
        """
        config_bbp = CloneConfig(
            emission=EmissionModel.BB_PARETO,
            inference=InferenceEngine.EM,
            k_prior=KPrior.BIC,
            phylo=PhyloMode.NONE,
            noise=NoiseModel.TAIL_FILTER,
            phi=50.0,
            tail_weight=1e-10,
        )
        config_bb = CloneConfig(
            emission=EmissionModel.BETA_BINOMIAL,
            inference=InferenceEngine.EM,
            k_prior=KPrior.BIC,
            phylo=PhyloMode.NONE,
            noise=NoiseModel.NONE,
            phi=50.0,
        )
        bbp = BBParetoEmission(config_bbp)
        bb = BetaBinomialEmission(config_bb)

        # Use mutations where observed VAF ~ mu (BB dominates, Pareto is small)
        mu_vals = np.array([0.3, 0.5, 0.4, 0.6, 0.2])
        alt = (mu_vals * 50).reshape(5, 1)
        depth = np.full((5, 1), 50.0)
        mu = mu_vals.reshape(5, 1)

        bbp_result = bbp.log_prob(alt, depth, mu)
        bb_result = bb.log_prob(alt, depth, mu)
        np.testing.assert_allclose(bbp_result, bb_result, rtol=1e-6)

    def test_pareto_tail_lifts_low_vaf(self) -> None:
        """For very low VAF mutations, BB+Pareto should assign higher prob than BB."""
        config_bbp = CloneConfig(
            emission=EmissionModel.BB_PARETO,
            inference=InferenceEngine.EM,
            k_prior=KPrior.BIC,
            phylo=PhyloMode.NONE,
            noise=NoiseModel.TAIL_FILTER,
            phi=50.0,
            tail_weight=0.2,
        )
        config_bb = CloneConfig(
            emission=EmissionModel.BETA_BINOMIAL,
            inference=InferenceEngine.EM,
            k_prior=KPrior.BIC,
            phylo=PhyloMode.NONE,
            noise=NoiseModel.NONE,
            phi=50.0,
        )
        bbp = BBParetoEmission(config_bbp)
        bb = BetaBinomialEmission(config_bb)

        # Low VAF: alt=1, depth=100, mu=0.5 -> observed VAF far from expected
        alt = np.array([[1.0]])
        depth = np.array([[100.0]])
        mu = np.array([[0.5]])

        bbp_result = bbp.log_prob(alt, depth, mu)[0]
        bb_result = bb.log_prob(alt, depth, mu)[0]
        assert bbp_result > bb_result

    def test_no_nan_edge_cases(self, bb_pareto_emission: BBParetoEmission) -> None:
        cases = [
            (0.0, 100.0, 0.5),
            (100.0, 100.0, 0.5),
            (1.0, 100.0, 0.01),
        ]
        for k, n, p in cases:
            alt = np.array([[k]])
            depth = np.array([[n]])
            mu = np.array([[p]])
            result = bb_pareto_emission.log_prob(alt, depth, mu)
            assert np.isfinite(result).all(), f"NaN for k={k}, n={n}, mu={p}"

    def test_output_shape(self, bb_pareto_emission: BBParetoEmission) -> None:
        n_mut = 30
        alt = np.full((n_mut, 1), 10.0)
        depth = np.full((n_mut, 1), 50.0)
        mu = np.full((n_mut, 1), 0.3)
        result = bb_pareto_emission.log_prob(alt, depth, mu)
        assert result.shape == (n_mut,)

    def test_default_tau_when_none(self) -> None:
        config = CloneConfig(
            emission=EmissionModel.BB_PARETO,
            inference=InferenceEngine.EM,
            k_prior=KPrior.BIC,
            phylo=PhyloMode.NONE,
            noise=NoiseModel.TAIL_FILTER,
        )
        em = BBParetoEmission(config)
        assert em.tau == 0.05


# ---------------------------------------------------------------------------
# DCF Beta-Binomial tests
# ---------------------------------------------------------------------------

class TestDCFBBLogProb:
    def test_matches_beta_binomial(self) -> None:
        """DCFBBEmission should produce identical results to BetaBinomialEmission."""
        config_dcf = CONFIGS["decifer_style"]
        config_bb = CloneConfig(
            emission=EmissionModel.BETA_BINOMIAL,
            inference=InferenceEngine.EM,
            k_prior=KPrior.BIC,
            phylo=PhyloMode.NONE,
            noise=NoiseModel.NONE,
            phi=config_dcf.phi,
        )
        dcf = DCFBBEmission(config_dcf)
        bb = BetaBinomialEmission(config_bb)

        rng = np.random.default_rng(42)
        alt = rng.integers(0, 50, size=(20, 1)).astype(float)
        depth = np.full((20, 1), 50.0)
        mu = rng.uniform(0.1, 0.9, size=(20, 1))

        dcf_result = dcf.log_prob(alt, depth, mu)
        bb_result = bb.log_prob(alt, depth, mu)
        np.testing.assert_allclose(dcf_result, bb_result, rtol=1e-10)

    def test_no_nan_edge_cases(self, dcf_emission: DCFBBEmission) -> None:
        cases = [
            (0.0, 100.0, 0.5),
            (100.0, 100.0, 0.5),
            (0.0, 100.0, 1e-8),
        ]
        for k, n, p in cases:
            alt = np.array([[k]])
            depth = np.array([[n]])
            mu = np.array([[p]])
            result = dcf_emission.log_prob(alt, depth, mu)
            assert np.isfinite(result).all(), f"NaN for k={k}, n={n}, mu={p}"

    def test_output_shape(self, dcf_emission: DCFBBEmission) -> None:
        n_mut = 30
        alt = np.full((n_mut, 2), 10.0)
        depth = np.full((n_mut, 2), 50.0)
        mu = np.full((n_mut, 2), 0.3)
        result = dcf_emission.log_prob(alt, depth, mu)
        assert result.shape == (n_mut,)


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------

class TestProtocolCompliance:
    def test_binomial_is_emission_module(self) -> None:
        assert isinstance(BinomialEmission(CONFIGS["quantumclone_v1"]), EmissionModule)

    def test_beta_binomial_is_emission_module(self) -> None:
        assert isinstance(BetaBinomialEmission(CONFIGS["pyclone_vi"]), EmissionModule)

    def test_bb_pareto_is_emission_module(self) -> None:
        assert isinstance(BBParetoEmission(CONFIGS["mobster"]), EmissionModule)

    def test_dcf_bb_is_emission_module(self) -> None:
        assert isinstance(DCFBBEmission(CONFIGS["decifer_style"]), EmissionModule)
