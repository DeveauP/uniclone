"""
uniclone.core.backend
======================

Thin backend abstraction that dispatches numerical primitives to either
NumPy/SciPy (default) or PyTorch, enabling GPU-accelerated EM/MFVI when
a CUDA device is available.

Usage
-----
All emission and inference hot-path code calls ``B.gammaln(x)`` etc.
instead of ``gammaln(x)`` directly.  By default ``B`` is a NumPy backend,
so there is zero behaviour change unless the user explicitly calls::

    from uniclone import set_backend
    set_backend("torch")   # or "auto"
"""

from __future__ import annotations

from typing import Any

import numpy as np


class _NumpyBackend:
    """Default backend dispatching to NumPy/SciPy."""

    name = "numpy"

    # --- special functions ---

    @staticmethod
    def gammaln(x: Any) -> Any:
        from scipy.special import gammaln

        return gammaln(x)

    @staticmethod
    def digamma(x: Any) -> Any:
        from scipy.special import digamma

        return digamma(x)

    @staticmethod
    def xlogy(x: Any, y: Any) -> Any:
        from scipy.special import xlogy

        return xlogy(x, y)

    @staticmethod
    def logsumexp(a: Any, axis: int | None = None) -> Any:
        from scipy.special import logsumexp

        return logsumexp(a, axis=axis)

    # --- elementwise ---

    @staticmethod
    def logaddexp(x: Any, y: Any) -> Any:
        return np.logaddexp(x, y)

    @staticmethod
    def clip(x: Any, lo: float | None, hi: float | None) -> Any:
        return np.clip(x, lo, hi)

    @staticmethod
    def exp(x: Any) -> Any:
        return np.exp(x)

    @staticmethod
    def log(x: Any) -> Any:
        return np.log(x)

    @staticmethod
    def maximum(x: Any, y: Any) -> Any:
        return np.maximum(x, y)

    # --- constructors ---

    @staticmethod
    def empty(shape: Any, dtype: Any = np.float64) -> Any:
        return np.empty(shape, dtype=dtype)

    @staticmethod
    def full(shape: Any, fill_value: Any, dtype: Any = np.float64) -> Any:
        return np.full(shape, fill_value, dtype=dtype)

    @staticmethod
    def zeros(shape: Any, dtype: Any = np.float64) -> Any:
        return np.zeros(shape, dtype=dtype)

    @staticmethod
    def asarray(x: Any, dtype: Any = np.float64) -> Any:
        return np.asarray(x, dtype=dtype)

    @staticmethod
    def to_numpy(x: Any) -> Any:
        return np.asarray(x)


class _TorchBackend:
    """GPU-capable backend dispatching to PyTorch."""

    name = "torch"

    def __init__(self, device: str = "auto") -> None:
        import torch

        self._torch = torch
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.dtype = torch.float64

    def _ensure_tensor(self, x: Any) -> Any:
        torch = self._torch
        if isinstance(x, torch.Tensor):
            return x.to(device=self.device, dtype=self.dtype)
        return torch.as_tensor(np.asarray(x), dtype=self.dtype, device=self.device)

    # --- special functions ---

    def gammaln(self, x: Any) -> Any:
        return self._torch.lgamma(self._ensure_tensor(x))

    def digamma(self, x: Any) -> Any:
        return self._torch.digamma(self._ensure_tensor(x))

    def xlogy(self, x: Any, y: Any) -> Any:
        return self._torch.xlogy(self._ensure_tensor(x), self._ensure_tensor(y))

    def logsumexp(self, a: Any, axis: int | None = None) -> Any:
        t = self._ensure_tensor(a)
        if axis is None:
            return self._torch.logsumexp(t.reshape(-1), dim=0)
        return self._torch.logsumexp(t, dim=axis)

    # --- elementwise ---

    def logaddexp(self, x: Any, y: Any) -> Any:
        return self._torch.logaddexp(self._ensure_tensor(x), self._ensure_tensor(y))

    def clip(self, x: Any, lo: float | None, hi: float | None) -> Any:
        return self._torch.clamp(self._ensure_tensor(x), min=lo, max=hi)

    def exp(self, x: Any) -> Any:
        return self._torch.exp(self._ensure_tensor(x))

    def log(self, x: Any) -> Any:
        return self._torch.log(self._ensure_tensor(x))

    def maximum(self, x: Any, y: Any) -> Any:
        return self._torch.maximum(self._ensure_tensor(x), self._ensure_tensor(y))

    # --- constructors ---

    def empty(self, shape: Any, dtype: Any = None) -> Any:
        return self._torch.empty(shape, dtype=self.dtype, device=self.device)

    def full(self, shape: Any, fill_value: Any, dtype: Any = None) -> Any:
        return self._torch.full(shape, fill_value, dtype=self.dtype, device=self.device)

    def zeros(self, shape: Any, dtype: Any = None) -> Any:
        return self._torch.zeros(shape, dtype=self.dtype, device=self.device)

    def asarray(self, x: Any, dtype: Any = None) -> Any:
        return self._ensure_tensor(x)

    def to_numpy(self, x: Any) -> Any:
        torch = self._torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)


# ---------------------------------------------------------------------------
# Module-level singleton and public API
# ---------------------------------------------------------------------------

B: _NumpyBackend | _TorchBackend = _NumpyBackend()


def set_backend(
    name: str = "auto",
    device: str = "auto",
) -> None:
    """
    Switch the compute backend.

    Parameters
    ----------
    name : ``"numpy"`` | ``"torch"`` | ``"auto"``
        ``"auto"`` uses torch if available, else numpy.
    device : ``"auto"`` | ``"cpu"`` | ``"cuda"``
        Only relevant when name is ``"torch"`` or ``"auto"``.
    """
    global B  # noqa: PLW0603

    if name == "numpy":
        B = _NumpyBackend()
        return

    if name in ("torch", "auto"):
        try:
            B = _TorchBackend(device=device)
        except ImportError:
            if name == "torch":
                raise
            # "auto" falls back to numpy
            B = _NumpyBackend()
        return

    raise ValueError(f"Unknown backend: {name!r}. Use 'numpy', 'torch', or 'auto'.")


def get_backend() -> _NumpyBackend | _TorchBackend:
    """Return the current backend singleton."""
    return B
