"""
uniclone.inference
==================

Inference module registry.  Maps ``InferenceEngine`` enum values to their
implementing classes.
"""
from __future__ import annotations

from uniclone.core.config import InferenceEngine
from uniclone.inference.em import EMInference
from uniclone.inference.hybrid import HybridInference
from uniclone.inference.mcmc import MCMCInference
from uniclone.inference.mfvi import MFVIInference

InferenceRegistry: dict[InferenceEngine, type] = {
    InferenceEngine.EM: EMInference,
    InferenceEngine.MFVI: MFVIInference,
    InferenceEngine.MCMC: MCMCInference,
    InferenceEngine.HYBRID: HybridInference,
}

__all__ = [
    "InferenceRegistry",
    "EMInference",
    "MFVIInference",
    "HybridInference",
    "MCMCInference",
]
