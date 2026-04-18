"""
uniclone.router.neural_ts
==========================

Neural Thompson Sampling model for MetaRouter configuration selection.

Architecture:
- SharedEncoder: maps 21 meta-features → 32-dim representation
- BayesianLinearHead: one per (config, subchallenge) pair with O(d^2) rank-1
  posterior updates and Thompson sampling for exploration
- NeuralTSModel: wraps encoder + 55 heads (11 configs × 5 subchallenges)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from uniclone.router.constants import (
    CONFIG_NAMES,
    N_CONFIGS,
    N_FEATURES,
    SUBCHALLENGES,
    SubChallenge,
)


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise ImportError(
            "Neural Thompson Sampling requires PyTorch. "
            "Install with: pip install uniclone[router]"
        )


# ---------------------------------------------------------------------------
# SharedEncoder
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:

    class SharedEncoder(nn.Module):
        """
        Shared feature encoder: Input(21) → 64 → 64 → 32.

        Maps raw meta-features to a learned representation used by all
        Bayesian linear heads.
        """

        def __init__(self, input_dim: int = N_FEATURES, hidden_dim: int = 64, output_dim: int = 32):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, output_dim),
            )
            self.output_dim = output_dim

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


# ---------------------------------------------------------------------------
# BayesianLinearHead
# ---------------------------------------------------------------------------


class BayesianLinearHead:
    """
    Bayesian linear regression head with rank-1 O(d²) online updates.

    Maintains a precision matrix Λ = λI + Σ zᵢzᵢᵀ and sufficient
    statistic zy_sum = Σ zᵢyᵢ for closed-form posterior updates.

    Thompson sampling draws θ ~ N(μ, Λ⁻¹) and predicts θᵀz.
    """

    def __init__(self, dim: int = 32, reg_lambda: float = 1.0):
        _require_torch()
        self.dim = dim
        self.reg_lambda = reg_lambda
        # Precision matrix Λ = λI
        self.precision = torch.eye(dim, dtype=torch.float32) * reg_lambda
        # Sufficient statistic: Σ z_i * y_i
        self.zy_sum = torch.zeros(dim, dtype=torch.float32)
        # Cache
        self._cov_cache: torch.Tensor | None = None
        self._mu_cache: torch.Tensor | None = None
        self.n_updates = 0

    def _invalidate_cache(self) -> None:
        self._cov_cache = None
        self._mu_cache = None

    @property
    def covariance(self) -> torch.Tensor:
        if self._cov_cache is None:
            cov = torch.linalg.inv(self.precision)
            self._cov_cache = (cov + cov.T) / 2  # ensure symmetry
        return self._cov_cache

    @property
    def mu(self) -> torch.Tensor:
        """Posterior mean: Λ⁻¹ zy_sum."""
        if self._mu_cache is None:
            self._mu_cache = self.covariance @ self.zy_sum
        return self._mu_cache

    def thompson_sample(self, z: torch.Tensor) -> float:
        """
        Thompson sample: draw θ ~ N(μ, Λ⁻¹), return θᵀz.

        Parameters
        ----------
        z : (dim,) encoding of input features

        Returns
        -------
        Sampled predicted reward.
        """
        cov = self.covariance
        # Add jitter to guarantee positive-definiteness after many updates
        jitter = 1e-6 * torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
        cov_safe = cov + jitter
        try:
            theta = torch.distributions.MultivariateNormal(
                self.mu, covariance_matrix=cov_safe
            ).sample()
        except ValueError:
            # Fallback: use diagonal approximation
            var = torch.clamp(torch.diag(cov), min=1e-6)
            theta = self.mu + torch.randn_like(self.mu) * torch.sqrt(var)
        return float(theta @ z)

    def mean_predict(self, z: torch.Tensor) -> float:
        """Return μᵀz (posterior mean prediction)."""
        return float(self.mu @ z)

    def update(self, z: torch.Tensor, reward: float) -> None:
        """
        Rank-1 posterior update: Λ ← Λ + zzᵀ, zy_sum ← zy_sum + z·y.

        O(d²) per update.
        """
        zzT = torch.outer(z, z)
        self.precision = self.precision + (zzT + zzT.T) / 2
        self.zy_sum = self.zy_sum + z * reward
        self.n_updates += 1
        self._invalidate_cache()

    def uncertainty(self, z: torch.Tensor) -> float:
        """Return zᵀ Λ⁻¹ z (posterior variance of prediction)."""
        cov = self.covariance
        return float(z @ cov @ z)

    def state_dict(self) -> dict[str, Any]:
        return {
            "precision": self.precision,
            "zy_sum": self.zy_sum,
            "n_updates": self.n_updates,
            "dim": self.dim,
            "reg_lambda": self.reg_lambda,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.precision = state["precision"]
        self.zy_sum = state["zy_sum"]
        self.n_updates = state["n_updates"]
        self._invalidate_cache()


# ---------------------------------------------------------------------------
# NeuralTSModel
# ---------------------------------------------------------------------------


class NeuralTSModel:
    """
    Neural Thompson Sampling model: shared encoder + Bayesian linear heads.

    Maintains 55 heads: one per (config, subchallenge) pair.
    The encoder maps 21 meta-features → 32-dim representation z.
    Each head maintains a Bayesian linear posterior over reward = θᵀz.

    Parameters
    ----------
    encoder : SharedEncoder or None
        Pre-trained encoder. If None, creates a fresh one.
    device : str
        Torch device for encoder.
    reg_lambda : float
        Prior precision for Bayesian heads.
    """

    def __init__(
        self,
        encoder: Any | None = None,
        device: str = "cpu",
        reg_lambda: float = 1.0,
    ):
        _require_torch()
        self.device = torch.device(device)

        if encoder is not None:
            self.encoder = encoder.to(self.device)
        else:
            self.encoder = SharedEncoder().to(self.device)

        self.encoder_dim = self.encoder.output_dim

        # Create heads: keyed by (config_index, subchallenge_index)
        self.heads: dict[tuple[int, int], BayesianLinearHead] = {}
        for ci in range(N_CONFIGS):
            for si in range(len(SUBCHALLENGES)):
                self.heads[(ci, si)] = BayesianLinearHead(
                    dim=self.encoder_dim, reg_lambda=reg_lambda
                )

    def _encode(self, features: np.ndarray | dict[str, float]) -> torch.Tensor:
        """Encode features to z representation."""
        if isinstance(features, dict):
            from uniclone.router.meta_features import features_to_tensor

            features = features_to_tensor(features)

        x = torch.tensor(features, dtype=torch.float32, device=self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        self.encoder.eval()
        with torch.no_grad():
            z = self.encoder(x).squeeze(0)
        return z

    def _config_index(self, config_name: str) -> int:
        return CONFIG_NAMES.index(config_name)

    def _sc_index(self, sc: SubChallenge) -> int:
        return SUBCHALLENGES.index(sc)

    def select(
        self,
        features: np.ndarray | dict[str, float],
        sc: SubChallenge,
        explore: bool = True,
    ) -> str:
        """
        Select best configuration for given features and subchallenge.

        Parameters
        ----------
        features : feature array or dict
        sc : SubChallenge
        explore : if True, use Thompson sampling; if False, use posterior mean

        Returns
        -------
        Name of selected configuration.
        """
        z = self._encode(features)
        si = self._sc_index(sc)

        best_score = -float("inf")
        best_config = CONFIG_NAMES[0]

        for ci, name in enumerate(CONFIG_NAMES):
            head = self.heads[(ci, si)]
            score = head.thompson_sample(z) if explore else head.mean_predict(z)
            if score > best_score:
                best_score = score
                best_config = name

        return best_config

    def update(
        self,
        features: np.ndarray | dict[str, float],
        config_name: str,
        sc: SubChallenge,
        reward: float,
    ) -> None:
        """Update the posterior for a (config, subchallenge) head."""
        z = self._encode(features)
        ci = self._config_index(config_name)
        si = self._sc_index(sc)
        self.heads[(ci, si)].update(z, reward)

    def scores(
        self,
        features: np.ndarray | dict[str, float],
        sc: SubChallenge,
    ) -> dict[str, float]:
        """Return posterior mean score for each config."""
        z = self._encode(features)
        si = self._sc_index(sc)
        return {
            name: self.heads[(ci, si)].mean_predict(z)
            for ci, name in enumerate(CONFIG_NAMES)
        }

    def uncertainty(
        self,
        features: np.ndarray | dict[str, float],
        sc: SubChallenge,
    ) -> dict[str, float]:
        """Return posterior variance for each config."""
        z = self._encode(features)
        si = self._sc_index(sc)
        return {
            name: self.heads[(ci, si)].uncertainty(z)
            for ci, name in enumerate(CONFIG_NAMES)
        }

    def save(self, path: str | Path) -> None:
        """Save model to disk."""
        _require_torch()
        path = Path(path)
        state = {
            "encoder": self.encoder.state_dict(),
            "heads": {
                f"{ci}_{si}": head.state_dict()
                for (ci, si), head in self.heads.items()
            },
        }
        torch.save(state, path)

    def load(self, path: str | Path) -> None:
        """Load model from disk."""
        _require_torch()
        path = Path(path)
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.encoder.load_state_dict(state["encoder"])
        for key_str, head_state in state["heads"].items():
            ci, si = map(int, key_str.split("_"))
            self.heads[(ci, si)].load_state_dict(head_state)

    @classmethod
    def from_pretrained(cls, path: str | Path, device: str = "cpu") -> NeuralTSModel:
        """Load a pretrained model from disk."""
        model = cls(device=device)
        model.load(path)
        return model
