# Custom configurations

UniClone's power comes from its 5 independent configuration axes. You can combine them freely to create configurations that no single existing tool provides.

## The 5 axes

### 1. Emission model (`EmissionModel`)

Controls the per-mutation likelihood function.

| Value | Description |
|-------|-------------|
| `BINOMIAL` | No overdispersion (QuantumClone-style) |
| `BETA_BINOMIAL` | Overdispersion via precision parameter phi |
| `BB_PARETO` | Beta-Binomial + neutral Pareto tail (MOBSTER-style) |
| `DCF_BB` | Beta-Binomial with variable multiplicity correction (DeCiFer-style) |

### 2. Inference engine (`InferenceEngine`)

Controls the optimization algorithm.

| Value | Description |
|-------|-------------|
| `EM` | Expectation-Maximisation (fast, local optima risk) |
| `MFVI` | Mean-field variational inference (ELBO, automatic K pruning) |
| `MCMC` | Full posterior via PyMC (requires `pymc` extra) |
| `HYBRID` | EM warm-start followed by MFVI refinement |

### 3. K prior (`KPrior`)

Controls how the number of clones K is determined.

| Value | Description |
|-------|-------------|
| `BIC` | Sweep K range, select by BIC |
| `DIRICHLET` | Dirichlet concentration, automatic K |
| `TSSB` | Tree-structured stick-breaking (PhyClone-style) |
| `FIXED` | User-specified K via `n_clones` |

### 4. Phylo mode (`PhyloMode`)

Controls phylogenetic tree reconstruction.

| Value | Description |
|-------|-------------|
| `NONE` | Clustering only, no tree |
| `POST_HOC` | Nesting matrix after convergence |
| `CONSTRAINED` | Sigmoid reparametrisation in M-step (UniClone-novel) |
| `JOINT_MCMC` | TSSB prior over trees (PhyClone-style) |
| `LONGITUDINAL` | Time-ordering ILP (CALDER-style) |
| `PAIRWISE` | Pairs tensor (Pairtree-style) |

### 5. Noise model (`NoiseModel`)

Controls artefact detection and filtering.

| Value | Description |
|-------|-------------|
| `NONE` | No noise correction |
| `TAIL_FILTER` | Pareto tail detection + mutation removal |
| `ARTEFACT` | Cluster consistency check (CONIPHER-style) |
| `MULTIPLICITY` | Variable multiplicity enumeration (DeCiFer-style) |

## Building a custom config

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

This is the `wgs_cohort_phylo` preset — a configuration unique to UniClone that combines MOBSTER's neutral tail detection with automatic K selection and sigmoid-projected phylogenetic constraints.

## Validation rules

`CloneConfig` validates axis combinations on construction:

- `k_prior=FIXED` requires `n_clones` to be set
- `phylo=LONGITUDINAL` requires `longitudinal=True`
- `phylo=JOINT_MCMC` requires `inference=MCMC` or `HYBRID`
- `phi` must be positive (if set)
- `tail_weight` must be in (0, 1) (if set)

Invalid combinations raise `ValueError` immediately.

## The `wgs_cohort_phylo` example

This preset demonstrates a novel combination unavailable in any existing tool:

```python
from uniclone import CONFIGS

config = CONFIGS["wgs_cohort_phylo"]
# EmissionModel.BB_PARETO   — neutral tail modelling
# InferenceEngine.HYBRID    — EM warm-start then MFVI
# KPrior.DIRICHLET          — automatic K (no BIC sweep)
# PhyloMode.CONSTRAINED     — sigmoid-projected tree in M-step
# NoiseModel.TAIL_FILTER    — remove Pareto-assigned mutations
# phi=100.0                 — fixed precision for WGS depth
```

The constrained phylo mode applies sigmoid reparametrisation during the M-step to ensure clone centers respect tree nesting constraints — a capability unique to UniClone.
