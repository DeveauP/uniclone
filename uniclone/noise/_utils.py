"""
uniclone.noise._utils
=====================

Shared utilities for noise modules.
"""

from __future__ import annotations

import numpy as np

from uniclone.core.types import CloneResult, Tensor


def expand_result(result: CloneResult, mask: Tensor, noise_label: int = -1) -> CloneResult:
    """
    Expand a filtered ``CloneResult`` back to the original n_mut size.

    Mutations that were filtered out (``mask[i] == False``) get
    ``assignments = noise_label`` and ``responsibilities = 0.0``.

    Parameters
    ----------
    result : CloneResult
        Result from inference on the filtered subset.
    mask : (n_mut_original,) bool
        True for mutations that were kept during inference.
    noise_label : int
        Assignment label for filtered-out mutations (default -1).

    Returns
    -------
    CloneResult
        Expanded result with ``noise_mask`` set.
    """
    n_total = mask.shape[0]
    n_kept = mask.sum()
    K = result.K

    # Expand assignments
    full_assignments = np.full(n_total, noise_label, dtype=result.assignments.dtype)
    full_assignments[mask] = result.assignments

    # Expand responsibilities
    full_resp = np.zeros((n_total, K), dtype=result.responsibilities.dtype)
    full_resp[mask] = result.responsibilities

    result.assignments = full_assignments
    result.responsibilities = full_resp
    result.noise_mask = mask

    return result
