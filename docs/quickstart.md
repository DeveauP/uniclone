# Quickstart

This guide walks through installing UniClone and running a clonal reconstruction.

## Installation

```bash
pip install uniclone
```

For GPU acceleration, install with the `gpu` extra:

```bash
pip install uniclone[gpu]
```

## Level 2: Named preset

The simplest way to use UniClone is with a named preset:

```python
import numpy as np
from uniclone import GenerativeModel, CONFIGS

# Simulated data: 400 mutations, 2 clones at VAF 0.2 and 0.5
rng = np.random.default_rng(42)
depth = np.full((400, 1), 100.0)
alt = np.concatenate([
    rng.binomial(100, 0.2, size=(200, 1)),
    rng.binomial(100, 0.5, size=(200, 1)),
]).astype(float)

# Fit using the PyClone-VI preset
model = GenerativeModel(CONFIGS["pyclone_vi"])
result = model.fit(alt, depth)
```

## Inspecting CloneResult

The `fit()` method returns a `CloneResult` with all reconstruction outputs:

```python
print(f"Number of clones: {result.K}")
print(f"Clone centers (CCF): {result.centers}")
print(f"Converged: {result.converged} in {result.n_iter} iterations")
print(f"Log-likelihood: {result.log_likelihood:.2f}")
print(f"BIC: {result.bic:.2f}")

# Per-mutation hard assignments (0..K-1)
print(f"Assignments shape: {result.assignments.shape}")

# Soft responsibilities (n_mut, K)
print(f"Responsibilities shape: {result.responsibilities.shape}")
```

## Visualization

With the `viz` extra installed (`pip install uniclone[viz]`):

```python
from uniclone.viz import vaf_histogram, clone_proportion_bar

# VAF histogram coloured by clone assignment
fig = vaf_histogram(result, alt, depth)
fig.show()

# Clone proportion bar chart
fig = clone_proportion_bar(result)
fig.show()
```

## GPU acceleration

Enable the PyTorch backend for faster EM/MFVI on large datasets:

```python
from uniclone import set_backend

set_backend("auto")  # uses CUDA if available, else CPU
# All subsequent fit() calls use the torch backend
```

## Next steps

- [Custom configurations](custom_configs.md) — build your own config from the 5 axes
- [MetaRouter](router.md) — let UniClone choose the best config automatically
- [API reference](api.md) — full documentation of all classes and functions
