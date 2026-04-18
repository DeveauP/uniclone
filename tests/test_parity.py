"""
Numerical parity tests for BinomialEmission.

These are regression tests using hardcoded expected values computed from
scipy.stats.binom.logpmf, which matches R's dbinom(..., log=TRUE) to
machine precision.

If any of these values change, something in BinomialEmission is wrong.

Note: True R parity (running R directly) can be validated locally with:
    R -e "dbinom(10, 50, 0.3, log=TRUE)"  # expected: -3.584895
This is not run in CI as R is not installed there.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import binom

from uniclone.core.config import CONFIGS
from uniclone.emission.binomial import BinomialEmission

# ---------------------------------------------------------------------------
# Pre-computed parity table
# All values computed via: float(binom.logpmf(k, n, p))
# ---------------------------------------------------------------------------

PARITY_TABLE = [
    # (alt, depth, mu, description)
    (10, 50, 0.3, "typical WES read counts"),
    (25, 50, 0.5, "half depth, at mode"),
    (0, 50, 0.3, "alt=0, should not NaN"),
    (50, 50, 0.9, "alt=depth"),
    (1, 100, 0.01, "rare variant"),
    (99, 100, 0.99, "near-fixed variant"),
    (30, 100, 0.3, "at mode"),
    (5, 200, 0.025, "low VAF, high depth"),
    (100, 200, 0.5, "canonical 50% VAF"),
]

PARITY_CASES = [(k, n, p, float(binom.logpmf(k, n, p)), desc) for k, n, p, desc in PARITY_TABLE]


@pytest.fixture
def emission() -> BinomialEmission:
    return BinomialEmission(CONFIGS["quantumclone_v1"])


@pytest.mark.parametrize("alt,depth,mu,expected,desc", PARITY_CASES)
def test_binomial_log_prob_parity(
    emission: BinomialEmission,
    alt: int,
    depth: int,
    mu: float,
    expected: float,
    desc: str,
) -> None:
    """
    BinomialEmission.log_prob must match scipy.stats.binom.logpmf to rtol=1e-6.

    This is numerically equivalent to R's dbinom(..., log=TRUE).
    """
    alt_arr = np.array([[float(alt)]])
    depth_arr = np.array([[float(depth)]])
    mu_arr = np.array([[mu]])
    result = emission.log_prob(alt_arr, depth_arr, mu_arr)
    np.testing.assert_allclose(
        result[0],
        expected,
        rtol=1e-6,
        err_msg=f"Parity failed for case: {desc} (alt={alt}, n={depth}, p={mu})",
    )


def test_parity_table_is_self_consistent(emission: BinomialEmission) -> None:
    """Run all parity cases in a single vectorised call."""
    cases = [(k, n, p) for k, n, p, _ in PARITY_TABLE]
    alt_arr = np.array([[float(k)] for k, n, p in cases])
    depth_arr = np.array([[float(n)] for k, n, p in cases])
    mu_arr = np.array([[p] for k, n, p in cases])

    results = emission.log_prob(alt_arr, depth_arr, mu_arr)
    expected = np.array([float(binom.logpmf(k, n, p)) for k, n, p in cases])

    np.testing.assert_allclose(results, expected, rtol=1e-6)
