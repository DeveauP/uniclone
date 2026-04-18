# MetaRouter

The MetaRouter automatically selects the best UniClone configuration for a given tumour based on its sequencing features.

## Level 1: Fully automatic

```python
from uniclone import UniClone

result = UniClone(alt, depth, adj_factor)
```

`UniClone()` extracts features from the input data, queries the MetaRouter to select the best configuration, then runs the full pipeline.

## How it works

The MetaRouter uses **Neural Thompson Sampling** (NeuralTS) — a contextual bandit approach that balances exploration and exploitation:

1. A shared neural network encoder maps 21 tumour features to an embedding
2. Per-arm Bayesian linear heads (one per config x subchallenge) predict expected score
3. Thompson Sampling samples from the posterior to select the configuration

### Feature extraction

The router uses 21 features in 5 categories:

- **Depth statistics** (3): median depth, depth CV, fraction of low-depth mutations
- **Copy-number landscape** (6): CNA fraction, WGD, ploidy, CN states, LOH, subclonal CN
- **VAF/CCF shape** (6): skewness, kurtosis, number of peaks, clonal peak width, Pareto tail, purity
- **Data modality** (4): number of samples, longitudinal flag, mutation count, sequencing type
- **Derived** (2): normalised reads per CN copy, mappability

### Training pipeline

The MetaRouter is trained on a corpus of simulated tumours via online bandit learning:

```bash
# Online training (default: excludes phyloclone_style for speed)
python -m scripts.train_online \
    --tumour-dir data/corpus/tumours \
    --n-pilot 200 --n-workers 8 \
    --out data/models/router_online.pt
```

By default, `phyloclone_style` (MCMC-based, ~10-70 min per fit) is excluded from
training. The router still has a head for it, but it receives no training signal
and is never selected. You can control this with `--exclude-configs`:

```bash
# Include all 12 configs (slow — MCMC fits can take hours)
python -m scripts.train_online --exclude-configs ...

# Exclude specific configs
python -m scripts.train_online --exclude-configs phyloclone_style calder_style ...
```

Programmatic usage:

```python
from uniclone.router.training import train_online

# Default: excludes phyloclone_style
result = train_online(tumour_dir="data/corpus/tumours", scores_dir="data/scores")

# Include all configs
result = train_online(..., exclude_configs=frozenset())

# Custom exclusions
result = train_online(..., exclude_configs=frozenset({"phyloclone_style", "calder_style"}))
```

### Retraining

As new real-world results become available, the router can be retrained incrementally:

```python
from uniclone.router.training import CorpusEntry

# Add new observations
new_entry = CorpusEntry(
    features=extracted_features,
    subchallenge=SubChallenge.SC2B,
    config_name="wgs_cohort_phylo",
    score=0.85,
)
```

## Configuration space

The router architecture has heads for all 12 named presets, but only the 11 non-excluded
configs receive training signal by default. Configs in `DEFAULT_EXCLUDE_CONFIGS`
(currently `{"phyloclone_style"}`) are skipped during training due to prohibitive
MCMC runtime. See the [preset table](index.md) for the full list.
