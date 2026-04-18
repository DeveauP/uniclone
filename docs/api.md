# API Reference

## Core

### CloneConfig

::: uniclone.core.config.CloneConfig
    options:
      show_source: false
      members:
        - validate

### GenerativeModel

::: uniclone.core.model.GenerativeModel
    options:
      show_source: false
      members:
        - fit

### CloneResult

::: uniclone.core.types.CloneResult

## Configuration enums

### EmissionModel

::: uniclone.core.config.EmissionModel

### InferenceEngine

::: uniclone.core.config.InferenceEngine

### KPrior

::: uniclone.core.config.KPrior

### PhyloMode

::: uniclone.core.config.PhyloMode

### NoiseModel

::: uniclone.core.config.NoiseModel

## Named presets

::: uniclone.core.config.CONFIGS

## Backend

### set_backend

::: uniclone.core.backend.set_backend

### get_backend

::: uniclone.core.backend.get_backend

## Emission modules

### BinomialEmission

::: uniclone.emission.binomial.BinomialEmission
    options:
      members:
        - log_prob

### BetaBinomialEmission

::: uniclone.emission.beta_binomial.BetaBinomialEmission
    options:
      members:
        - log_prob

### BBParetoEmission

::: uniclone.emission.bb_pareto.BBParetoEmission
    options:
      members:
        - log_prob

### DCFBBEmission

::: uniclone.emission.dcf.DCFBBEmission
    options:
      members:
        - log_prob
