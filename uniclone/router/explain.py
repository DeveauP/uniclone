"""
uniclone.router.explain
========================

Gradient-based feature attribution for MetaRouter decisions.

Uses integrated gradients on the differentiable path
μᵀ encoder(x) to explain why a particular configuration was selected.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from uniclone.router.constants import CONFIG_NAMES, FEATURE_NAMES, SUBCHALLENGES, SubChallenge


def compute_feature_attribution(
    model: object,
    features: np.ndarray | dict[str, float],
    sc: SubChallenge,
    config: str | None = None,
    method: str = "integrated_gradients",
    n_steps: int = 50,
) -> dict[str, float]:
    """
    Compute per-feature attributions for a routing decision.

    Uses integrated gradients on the differentiable score function
    f(x) = μᵀ encoder(x), where μ is the posterior mean of the
    Bayesian linear head.

    Satisfies completeness: sum(attributions) ≈ f(x) - f(baseline).

    Parameters
    ----------
    model : NeuralTSModel
    features : feature array or dict
    sc : SubChallenge
    config : config name (if None, uses the model's top prediction)
    method : attribution method (currently only "integrated_gradients")
    n_steps : number of interpolation steps

    Returns
    -------
    Dict mapping feature name → attribution value.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("Explanations require PyTorch: pip install uniclone[router]")

    from uniclone.router.meta_features import features_to_tensor
    from uniclone.router.neural_ts import NeuralTSModel

    assert isinstance(model, NeuralTSModel)

    if isinstance(features, dict):
        features = features_to_tensor(features)

    x = torch.tensor(features, dtype=torch.float32, device=model.device)

    # Select config if not provided
    if config is None:
        config = model.select(features, sc, explore=False)

    ci = CONFIG_NAMES.index(config)
    si = SUBCHALLENGES.index(sc)
    head = model.heads[(ci, si)]
    mu = head.mu.to(model.device)

    if method == "integrated_gradients":
        attributions = _integrated_gradients(model.encoder, mu, x, n_steps)
    else:
        raise ValueError(f"Unknown attribution method: {method}")

    return dict(zip(FEATURE_NAMES, attributions.tolist()))


def _integrated_gradients(
    encoder: Any,
    mu: Any,
    x: Any,
    n_steps: int,
) -> np.ndarray:
    """
    Integrated gradients: IG_i = (x_i - x̄_i) × ∫₀¹ ∂f/∂x_i(x̄ + α(x-x̄)) dα

    where f(x) = μᵀ encoder(x) and x̄ = 0 (zero baseline).
    """
    import torch

    baseline = torch.zeros_like(x)
    alphas = torch.linspace(0, 1, n_steps + 1, device=x.device)

    # Accumulate gradients along interpolation path
    grads = torch.zeros_like(x)

    encoder.eval()

    for alpha in alphas:
        interp = baseline + alpha * (x - baseline)
        interp = interp.detach().requires_grad_(True)
        z = encoder(interp.unsqueeze(0)).squeeze(0)
        score = mu @ z
        score.backward()
        grads += interp.grad

    # Average and multiply by (x - baseline)
    avg_grads = grads / (n_steps + 1)
    attributions = (x - baseline) * avg_grads

    return attributions.detach().cpu().numpy()
