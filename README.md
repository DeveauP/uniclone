# UniClone

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Docs](https://img.shields.io/badge/docs-readthedocs-blue.svg)](https://uniclone.readthedocs.io)

**Unified probabilistic framework for tumour clonal reconstruction.**

UniClone subsumes PyClone, PyClone-VI, MOBSTER, DeCiFer, CONIPHER, CALDER, and Pairtree as named configuration presets within a single codebase, with a learned MetaRouter that selects the optimal configuration per tumour.

## Installation

```bash
# Core (NumPy/SciPy only)
pip install uniclone

# With GPU acceleration
pip install uniclone[gpu]

# With MCMC support (PyMC)
pip install uniclone[mcmc]

# With visualization
pip install uniclone[viz]

# Everything
pip install uniclone[all]
```

## Quick start

### Level 1 — Fully automatic (MetaRouter)

```python
from uniclone import UniClone

result = UniClone(alt, depth, adj_factor)
```

The MetaRouter uses Neural Thompson Sampling to select the best configuration for your tumour based on sequencing features.

### Level 2 — Named preset

```python
from uniclone import GenerativeModel, CONFIGS

model = GenerativeModel(CONFIGS["pyclone_vi"])
result = model.fit(alt, depth, adj_factor)
```

### Level 3 — Custom configuration

```python
from uniclone import GenerativeModel, CloneConfig
from uniclone import EmissionModel, InferenceEngine, KPrior, PhyloMode, NoiseModel

config = CloneConfig(
    emission=EmissionModel.BB_PARETO,
    inference=InferenceEngine.HYBRID,
    k_prior=KPrior.DIRICHLET,
    phylo=PhyloMode.CONSTRAINED,
    noise=NoiseModel.TAIL_FILTER,
    phi=100.0,
)
model = GenerativeModel(config)
result = model.fit(alt, depth, adj_factor)
```

### GPU acceleration

```python
from uniclone import set_backend

set_backend("torch")  # or "auto" to use GPU if available
# All subsequent fit() calls use GPU-accelerated EM/MFVI
```

## Named presets

| Preset | Emission | Inference | K Prior | Phylo | Noise | Based on |
|--------|----------|-----------|---------|-------|-------|----------|
| `quantumclone_v1` | Binomial | EM | BIC | Post-hoc | None | QuantumClone |
| `pyclone_vi` | Beta-Binomial | MFVI | Dirichlet | None | None | PyClone-VI |
| `mobster` | BB+Pareto | EM | BIC | None | Tail filter | MOBSTER |
| `decifer_style` | DCF-BB | EM | BIC | None | Multiplicity | DeCiFer |
| `conipher_style` | Beta-Binomial | MFVI | Dirichlet | Post-hoc | Artefact | CONIPHER |
| `phyloclone_style` | Beta-Binomial | MCMC | TSSB | Joint MCMC | None | PhyClone |
| `calder_style` | Beta-Binomial | EM | Fixed | Longitudinal | None | CALDER |
| `wes_clinical` | Beta-Binomial | Hybrid | Dirichlet | Constrained | Artefact | *UniClone* |
| `wgs_cohort` | BB+Pareto | MFVI | Dirichlet | None | Tail filter | *UniClone* |
| `wgs_cohort_phylo` | BB+Pareto | Hybrid | Dirichlet | Constrained | Tail filter | *UniClone* |
| `multiregion_phylo` | DCF-BB | Hybrid | Dirichlet | Pairwise | Multiplicity | *UniClone* |
| `longitudinal_clinical` | Beta-Binomial | Hybrid | BIC | Longitudinal | Artefact | *UniClone* |

## Documentation

Full documentation is available at [uniclone.readthedocs.io](https://uniclone.readthedocs.io).

- [Quickstart guide](https://uniclone.readthedocs.io/quickstart/)
- [Custom configurations](https://uniclone.readthedocs.io/custom_configs/)
- [MetaRouter guide](https://uniclone.readthedocs.io/router/)
- [API reference](https://uniclone.readthedocs.io/api/)

## License

MIT
