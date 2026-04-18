"""
Tests for the compute backend abstraction (uniclone.core.backend).
"""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest
from scipy.special import gammaln as scipy_gammaln

from uniclone.core import backend as backend_mod
from uniclone.core.backend import _NumpyBackend, get_backend, set_backend


class TestNumpyBackendIsDefault:
    def test_default_backend_is_numpy(self) -> None:
        assert backend_mod.B.name == "numpy"

    def test_get_backend_returns_singleton(self) -> None:
        assert get_backend() is backend_mod.B


class TestNumpyBackendGammaln:
    def test_gammaln_matches_scipy(self) -> None:
        x = np.array([0.5, 1.0, 2.5, 5.0, 10.0])
        result = _NumpyBackend.gammaln(x)
        expected = scipy_gammaln(x)
        np.testing.assert_allclose(result, expected)


class TestSetBackendNumpy:
    def test_set_backend_numpy_explicit(self) -> None:
        set_backend("numpy")
        assert backend_mod.B.name == "numpy"

    def test_round_trip(self) -> None:
        set_backend("numpy")
        b = get_backend()
        assert b.name == "numpy"
        x = np.array([1.0, 2.0, 3.0])
        result = b.gammaln(x)
        np.testing.assert_allclose(result, scipy_gammaln(x))


class TestNumpyBackendOps:
    def test_digamma(self) -> None:
        from scipy.special import digamma as scipy_digamma

        x = np.array([1.0, 2.0, 5.0])
        np.testing.assert_allclose(_NumpyBackend.digamma(x), scipy_digamma(x))

    def test_xlogy(self) -> None:
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([1.0, np.e, np.e])
        result = _NumpyBackend.xlogy(x, y)
        assert result[0] == 0.0  # 0 * log(1) = 0
        np.testing.assert_allclose(result[1], 1.0, atol=1e-10)

    def test_logsumexp(self) -> None:
        from scipy.special import logsumexp as scipy_logsumexp

        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_allclose(
            _NumpyBackend.logsumexp(x, axis=1),
            scipy_logsumexp(x, axis=1),
        )

    def test_clip(self) -> None:
        x = np.array([-1.0, 0.5, 2.0])
        np.testing.assert_array_equal(_NumpyBackend.clip(x, 0.0, 1.0), [0.0, 0.5, 1.0])

    def test_constructors(self) -> None:
        assert _NumpyBackend.empty((2, 3)).shape == (2, 3)
        np.testing.assert_array_equal(_NumpyBackend.full((3,), 7.0), [7.0, 7.0, 7.0])
        np.testing.assert_array_equal(_NumpyBackend.zeros((2,)), [0.0, 0.0])

    def test_to_numpy_identity(self) -> None:
        x = np.array([1.0, 2.0])
        assert _NumpyBackend.to_numpy(x) is x


class TestInvalidBackend:
    def test_unknown_name_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend"):
            set_backend("jax")

    def test_auto_fallback_without_torch(self) -> None:
        with mock.patch.dict("sys.modules", {"torch": None}):
            # Force re-creation
            set_backend("auto")
            # Should fall back to numpy
            assert backend_mod.B.name == "numpy"


# ---------------------------------------------------------------------------
# Torch backend tests (skipped if torch is not installed)
# ---------------------------------------------------------------------------

torch_available = pytest.importorskip("torch", reason="torch not installed")


class TestTorchBackend:
    def setup_method(self) -> None:
        set_backend("torch", device="cpu")

    def teardown_method(self) -> None:
        set_backend("numpy")

    def test_backend_is_torch(self) -> None:
        assert backend_mod.B.name == "torch"

    def test_gammaln_matches_scipy(self) -> None:
        x = np.array([0.5, 1.0, 2.5, 5.0, 10.0])
        result = backend_mod.B.to_numpy(backend_mod.B.gammaln(x))
        expected = scipy_gammaln(x)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_digamma_matches_scipy(self) -> None:
        from scipy.special import digamma as scipy_digamma

        x = np.array([1.0, 2.0, 5.0])
        result = backend_mod.B.to_numpy(backend_mod.B.digamma(x))
        np.testing.assert_allclose(result, scipy_digamma(x), atol=1e-10)

    def test_logsumexp(self) -> None:
        from scipy.special import logsumexp as scipy_logsumexp

        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = backend_mod.B.to_numpy(backend_mod.B.logsumexp(x, axis=1))
        np.testing.assert_allclose(result, scipy_logsumexp(x, axis=1), atol=1e-10)

    def test_device_auto(self) -> None:
        import torch

        set_backend("torch", device="auto")
        b = get_backend()
        if torch.cuda.is_available():
            assert "cuda" in str(b.device)
        else:
            assert "cpu" in str(b.device)

    def test_to_numpy_roundtrip(self) -> None:
        x = np.array([1.0, 2.0, 3.0])
        t = backend_mod.B.asarray(x)
        result = backend_mod.B.to_numpy(t)
        np.testing.assert_array_equal(result, x)


class TestEmissionParityBothBackends:
    """BetaBinomialEmission.log_prob should match across backends."""

    def test_beta_binomial_parity(self) -> None:
        from uniclone.core.config import CloneConfig
        from uniclone.emission.beta_binomial import BetaBinomialEmission

        rng = np.random.default_rng(42)
        alt = rng.binomial(100, 0.4, size=(50, 1)).astype(float)
        depth = np.full((50, 1), 100.0)
        mu = np.full((50, 1), 0.4)
        config = CloneConfig(phi=50.0)

        # NumPy result
        set_backend("numpy")
        emission_np = BetaBinomialEmission(config)
        result_np = emission_np.log_prob(alt, depth, mu)

        # Torch result
        set_backend("torch", device="cpu")
        emission_torch = BetaBinomialEmission(config)
        result_torch = backend_mod.B.to_numpy(emission_torch.log_prob(alt, depth, mu))

        set_backend("numpy")
        np.testing.assert_allclose(result_torch, result_np, atol=1e-8)


class TestEMFullParity:
    """Full EM run should produce matching results across backends."""

    def test_em_parity(self) -> None:
        from uniclone import CONFIGS, GenerativeModel

        rng = np.random.default_rng(42)
        alt = np.concatenate(
            [
                rng.binomial(100, 0.2, size=(100, 1)),
                rng.binomial(100, 0.5, size=(100, 1)),
            ]
        ).astype(float)
        depth = np.full((200, 1), 100.0)

        # NumPy
        set_backend("numpy")
        model_np = GenerativeModel(CONFIGS["quantumclone_v1"])
        result_np = model_np.fit(alt, depth)

        # Torch
        set_backend("torch", device="cpu")
        model_torch = GenerativeModel(CONFIGS["quantumclone_v1"])
        result_torch = model_torch.fit(alt, depth)

        set_backend("numpy")
        np.testing.assert_allclose(result_torch.centers, result_np.centers, atol=1e-6)
        np.testing.assert_allclose(result_torch.log_likelihood, result_np.log_likelihood, atol=1e-4)
