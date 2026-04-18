"""
uniclone.noise
==============

Noise module registry.  Maps ``NoiseModel`` enum values to their
implementing classes.
"""

from __future__ import annotations

from uniclone.core.config import NoiseModel
from uniclone.noise.artefact import ArtefactNoise
from uniclone.noise.multiplicity import MultiplicityNoise
from uniclone.noise.none import NoNoise
from uniclone.noise.tail_filter import TailFilterNoise

NoiseRegistry: dict[NoiseModel, type] = {
    NoiseModel.NONE: NoNoise,
    NoiseModel.TAIL_FILTER: TailFilterNoise,
    NoiseModel.ARTEFACT: ArtefactNoise,
    NoiseModel.MULTIPLICITY: MultiplicityNoise,
}

__all__ = ["NoiseRegistry", "NoNoise", "_utils"]
