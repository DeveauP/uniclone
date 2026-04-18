# UniClone

**Unified probabilistic framework for tumour clonal reconstruction.**

UniClone subsumes PyClone, PyClone-VI, MOBSTER, DeCiFer, CONIPHER, CALDER, and Pairtree as named configuration presets within a single codebase, with a learned MetaRouter that selects the optimal configuration per tumour.

## Features

- **12 named presets** reproducing 7 published tools plus 5 novel configurations
- **5 configuration axes**: emission, inference, K prior, phylo, noise
- **MetaRouter**: Neural Thompson Sampling selects the best config per tumour
- **GPU acceleration**: optional PyTorch backend for EM/MFVI hot paths
- **Visualization suite**: VAF histograms, clone trees, fish plots, and more

## Getting started

- [Quickstart](quickstart.md) — install and run your first analysis
- [Custom configurations](custom_configs.md) — build configs from the 5 axes
- [MetaRouter](router.md) — automatic configuration selection
- [API reference](api.md) — full class and function documentation

## Installation

```bash
pip install uniclone          # core
pip install uniclone[gpu]     # + PyTorch GPU backend
pip install uniclone[viz]     # + Plotly visualization
pip install uniclone[all]     # everything
```
