"""
uniclone.router.meta_features
==============================

Extract ~21 scalar meta-features from tumour data for MetaRouter routing.

Features are computable in ~1 second before running any reconstruction
and are used by the MetaRouter to select the best configuration.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from uniclone.router.constants import FEATURE_NAMES


def extract_meta_features(
    *,
    alt: np.ndarray,
    depth: np.ndarray,
    adj_factor: np.ndarray | None = None,
    vcf_df: object | None = None,
    cn_df: object | None = None,
) -> dict[str, float]:
    """
    Extract meta-features for MetaRouter routing decisions.

    Parameters
    ----------
    alt : (n_mut, n_samples) array
        Alternate read counts.
    depth : (n_mut, n_samples) array
        Total read depths.
    adj_factor : (n_mut, n_samples) array or None
        Copy-number / purity adjustment factor.
    vcf_df : optional
        VCF dataframe for additional features (mappability, etc.).
    cn_df : optional
        Copy-number dataframe for CN landscape features.

    Returns
    -------
    dict mapping feature name to float value.
    """
    alt = np.asarray(alt, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    if alt.ndim == 1:
        alt = alt[:, np.newaxis]
        depth = depth[:, np.newaxis]

    n_mut, n_samples = alt.shape

    if adj_factor is not None:
        adj_factor = np.asarray(adj_factor, dtype=np.float64)
        if adj_factor.ndim == 1:
            adj_factor = adj_factor[:, np.newaxis]
    else:
        adj_factor = np.ones_like(alt)

    # VAF per mutation per sample
    safe_depth = np.maximum(depth, 1)
    vaf = alt / safe_depth  # (n_mut, n_samples)

    # Use first sample for shape features
    vaf_s0 = vaf[:, 0]
    depth_s0 = depth[:, 0]

    # --- Depth statistics (3) ---
    depth_median = float(np.median(depth_s0))
    depth_cv = float(np.std(depth_s0) / max(np.mean(depth_s0), 1e-9))
    frac_low_depth = float(np.mean(depth_s0 < 20))

    # --- Copy-number landscape (6) ---
    cn_features = _extract_cn_features(adj_factor, cn_df)

    # --- VAF/CCF shape (6) ---
    ccf = _compute_ccf(vaf_s0, adj_factor[:, 0])
    skewness = float(sp_stats.skew(ccf)) if len(ccf) > 2 else 0.0
    kurtosis = float(sp_stats.kurtosis(ccf)) if len(ccf) > 2 else 0.0
    n_peaks = _count_kde_peaks(ccf)
    clonal_peak_width = _clonal_peak_fwhm(ccf)
    has_pareto_tail = float(_mobster_tail_test(vaf_s0))
    purity_estimate = float(_estimate_purity(ccf))

    # --- Data modality (4) ---
    is_longitudinal = 0.0  # default, can be overridden by caller
    # Heuristic: WGS has more mutations, WES fewer
    sequencing_type = 1.0 if n_mut > 1000 else 0.0  # 1=WGS, 0=WES

    # --- Derived (2) ---
    # NRPCC: number of reads per clone per cellularity
    nrpcc = float(depth_median * n_mut / max(n_peaks, 1))
    mappability = _extract_mappability(vcf_df)

    features: dict[str, float] = {
        "depth_median": depth_median,
        "depth_cv": depth_cv,
        "frac_low_depth": frac_low_depth,
        "frac_cna": cn_features["frac_cna"],
        "wgd": cn_features["wgd"],
        "ploidy": cn_features["ploidy"],
        "n_cn_states": cn_features["n_cn_states"],
        "frac_loh": cn_features["frac_loh"],
        "subclonal_cn": cn_features["subclonal_cn"],
        "skewness": skewness,
        "kurtosis": kurtosis,
        "n_peaks": float(n_peaks),
        "clonal_peak_width": clonal_peak_width,
        "has_pareto_tail": has_pareto_tail,
        "purity_estimate": purity_estimate,
        "n_samples": float(n_samples),
        "is_longitudinal": is_longitudinal,
        "n_mutations": float(n_mut),
        "sequencing_type": sequencing_type,
        "nrpcc": nrpcc,
        "mappability": mappability,
    }

    # Ensure all features present
    for name in FEATURE_NAMES:
        if name not in features:
            features[name] = 0.0

    return features


def features_to_tensor(features: dict[str, float]) -> np.ndarray:
    """Convert feature dict to ordered numpy array of shape (N_FEATURES,)."""
    return np.array([features[name] for name in FEATURE_NAMES], dtype=np.float64)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _compute_ccf(vaf: np.ndarray, adj: np.ndarray) -> np.ndarray:
    """Estimate CCF from VAF and adjustment factor."""
    ccf = vaf / np.maximum(adj * 0.5, 1e-9)
    return np.clip(ccf, 0, 2.0)


def _count_kde_peaks(ccf: np.ndarray) -> int:
    """Count number of peaks in CCF distribution using KDE."""
    if len(ccf) < 10:
        return 1
    valid = ccf[(ccf > 0.01) & (ccf < 1.99)]
    if len(valid) < 10:
        return 1

    try:
        kde = sp_stats.gaussian_kde(valid, bw_method=0.1)
        x = np.linspace(0.01, 1.5, 200)
        density = kde(x)
        # Count peaks: local maxima
        peaks = 0
        for i in range(1, len(density) - 1):
            if density[i] > density[i - 1] and density[i] > density[i + 1]:
                # Require minimum prominence
                if density[i] > 0.1 * density.max():
                    peaks += 1
        return max(peaks, 1)
    except Exception:
        return 1


def _clonal_peak_fwhm(ccf: np.ndarray) -> float:
    """Estimate FWHM of the clonal peak (CCF ~ 1.0)."""
    if len(ccf) < 10:
        return 0.5
    clonal = ccf[(ccf > 0.5) & (ccf < 1.5)]
    if len(clonal) < 5:
        return 0.5
    return float(np.std(clonal) * 2.355)  # FWHM ≈ 2.355 × σ for Gaussian


def _mobster_tail_test(vaf: np.ndarray) -> bool:
    """
    Simple test for Pareto-like tail in VAF distribution (low-VAF excess).

    Returns True if there's evidence of a neutral tail.
    """
    if len(vaf) < 20:
        return False
    low_vaf = vaf[(vaf > 0.01) & (vaf < 0.15)]
    frac_low = len(low_vaf) / len(vaf)
    # If >30% of mutations have VAF < 0.15, suggestive of neutral tail
    return frac_low > 0.3


def _estimate_purity(ccf: np.ndarray) -> float:
    """Estimate purity from the position of the clonal peak."""
    if len(ccf) < 5:
        return 0.5
    # The clonal peak should be near 1.0 if purity adjustment is correct
    # Use the mode in the 0.5-1.5 range as an estimate
    clonal = ccf[(ccf > 0.3) & (ccf < 1.5)]
    if len(clonal) < 5:
        return 0.5
    try:
        kde = sp_stats.gaussian_kde(clonal, bw_method=0.15)
        x = np.linspace(0.3, 1.5, 100)
        return float(x[np.argmax(kde(x))])
    except Exception:
        return float(np.median(clonal))


def _extract_cn_features(adj_factor: np.ndarray, cn_df: object | None) -> dict[str, float]:
    """Extract copy-number landscape features."""
    # Infer CN state from adj_factor: adj ≈ 2 / (purity * CN + (1-purity) * 2)
    # For diploid CN=2, adj=1.0
    is_altered = np.abs(adj_factor[:, 0] - 1.0) > 0.05
    frac_cna = float(np.mean(is_altered))

    # Estimate ploidy from adj_factor
    # adj = 2 / (p * CN + (1-p)*2), so CN ≈ (2/adj - (1-p)*2) / p
    # Use simple heuristic: ploidy ~ 2 / mean(adj_factor)
    mean_adj = float(np.mean(adj_factor[:, 0]))
    ploidy = 2.0 / max(mean_adj, 0.1)

    # WGD: ploidy > 3
    wgd = 1.0 if ploidy > 3.0 else 0.0

    # Estimate number of distinct CN states
    adj_rounded = np.round(adj_factor[:, 0], 2)
    n_cn_states = float(len(np.unique(adj_rounded)))

    # LOH fraction: adj_factor significantly > 1 suggests CN loss
    frac_loh = float(np.mean(adj_factor[:, 0] > 1.2))

    # Subclonal CN: variation in adj_factor across samples
    if adj_factor.shape[1] > 1:
        subclonal_cn = float(np.mean(np.std(adj_factor, axis=1) > 0.05))
    else:
        subclonal_cn = 0.0

    return {
        "frac_cna": frac_cna,
        "wgd": wgd,
        "ploidy": ploidy,
        "n_cn_states": n_cn_states,
        "frac_loh": frac_loh,
        "subclonal_cn": subclonal_cn,
    }


def _extract_mappability(vcf_df: object | None) -> float:
    """Extract mean mappability from VCF dataframe, or return default."""
    if vcf_df is None:
        return 1.0
    # If vcf_df has a 'mappability' column, use it
    if hasattr(vcf_df, "get"):
        mapp = vcf_df.get("mappability")
        if mapp is not None:
            return float(np.mean(mapp))
    return 1.0
