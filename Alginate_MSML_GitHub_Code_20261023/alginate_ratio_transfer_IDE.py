#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experimental ratio-count study for the alginate HCD-MS/MS + ML manuscript.

Tests whether the final model really needs 8 fragment-ion ratios, or whether
2, 4, or 6 ratios provide equal/better unseen-composition performance.

Fixed pipeline:
- individual MS2 scans (~330 per acquisition)
- no automatic outlier removal
- precursor retained
- NMF components = 2
- Ridge alpha = 1.0
- StandardScaler fit on outer-training data only
- composition-balanced weights for scaler and Ridge
- outer LORO holds out one COMPLETE composition
- primary fold error is based on the mean prediction for the held-out
  composition
- exact 24 predefined peak centers
- current DP precursor excluded from local ratio candidates

Candidate-ratio hard filters:
1. denominator valid in >=95% of scans in EVERY training composition
2. |Spearman rho| >= 0.75 on composition-level mean ratio values
3. <=1 monotonicity violation

Eligible ratios are ranked by:
1. |Spearman rho|
2. technical stability SNR
3. |Pearson r|
4. minimum denominator-valid fraction

technical stability SNR =
    range(composition-level mean ratios)
    / median(within-composition robust SD)

robust SD = 1.4826 * MAD across scans.

Redundancy:
- similarity of composition-mean trajectories is a SOFT penalty only.
- NO hard ratio redundancy exclusion and no peak reuse cap.

Ratio counts tested: 2, 4, 6, 8.

Required packages: numpy, scipy, pandas, scikit-learn, pyteomics,
                   matplotlib, joblib.
Required files in DATA_DIR:
    DP_spectrum_process_Class.py (next to this script)
    Raw_F3_M0G100.mzML ... Raw_F7_M100G0.mzML
    AlgM_LowDPMix_1_051625.mzML etc. (for external mode only)
"""

from __future__ import annotations

import json
import time
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
import argparse
import re
import joblib
from pyteomics import mzml
from DP_spectrum_process_Class import FractionSpectra, filter_peaks_by_centers
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import NMF
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

from DP_spectrum_process_Class import DPSpectrumProcessor


# =============================================================================
# Configuration
# =============================================================================

MIX_RATIOS = [0, 20, 40, 50, 60, 80, 100]
DPS = [3, 4, 5, 6, 7]
ENERGIES = [18.0, 22.0, 26.0]
RATIO_COUNTS = [2, 4, 6, 8]

N_NMF_COMPONENTS = 2
RIDGE_ALPHA = 1.0

CLIP_LO = 100.0
CONV_SIGMA = 1.0
PEAK_WINDOW = 2.0

GROUP_BOUNDARIES = [
    (100, 200),
    (300, 370),
    (460, 530),
    (580, 710),
    (830, 880),
    (1030, 1060),
    (1180, 1240),
]

ALL_FILTER_CENTERS = [
    113.0244,   # BZ1-CO2
    131.0350,   # Z1-CO2
    157.0142,   # BZ1
    175.0248,   # B1/Z1
    193.0354,   # C1/Y1
    291.0358,   # BX2
    309.0463,   # A2/X2
    351.0569,   # B2/Z2
    369.0675,   # C2/Y2
    467.0679,   # BX3
    485.0784,   # A3/X3
    545.0996,   # C3/Y3 (DP3 precursor)
    661.1105,   # A4/X4
    703.1211,   # B4/Z4
    721.1316,   # C4/Y4 (DP4 precursor)
    837.1426,   # A5/X5
    879.1532,   # B5/Z5
    897.1637,   # C5/Y5 (DP5 precursor)
    1037.1747,  # BZ6
    1055.1853,  # B6/Z6
    1073.1958,  # C6/Y6 (DP6 precursor)
    1189.2068,  # A7/X7
    1231.2174,  # B7/Z7
    1249.2279,  # DP7 precursor
]

DP_TARGET_MZ = {
    3: 545.099609375,
    4: 721.131591796875,
    5: 897.1636962890625,
    6: 1073.19580078125,
    7: 1249.2279052734375,
}

NMF_PARAMS = {
    "n_components": N_NMF_COMPONENTS,
    "init": "nndsvda",
    "max_iter": 3000,
    "beta_loss": "frobenius",
    "solver": "cd",
    "tol": 1e-4,
    "random_state": 42,
}

RATIO_EPS = 1e-9
DENOM_MIN = 1e-6
MIN_DENOM_VALID_FRACTION_PER_COMPOSITION = 0.95
MIN_ABS_SPEARMAN = 0.75
MAX_MONOTONIC_VIOLATIONS = 1
REDUNDANCY_PENALTY = 0.075
STABILITY_EPS = 1e-12

# -----------------------------------------------------------------------------
# IDE SETTINGS -- edit here, then click Run in Spyder / PyCharm / VS Code.
# No command-line arguments are required.
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR                 # directory with Raw_F3_...Raw_F7_...mzML
OUT_ROOT = SCRIPT_DIR / "ratio_transfer_ide_results"

# "experiment": 105 outer LORO folds, save final frozen models and CSVs.
# "external":   apply already-saved frozen models to independent mzMLs.
# "both":       run experiment first, then immediately external prediction.
RUN_MODE = "experiment"

# Use the files from the original DP3/DP4 external-testing scripts.
EXTERNAL_FILES = {
    "MM_block_rich": "AlgM_LowDPMix_1_051625.mzML",
    "MG_block_rich": "AlgMG_LowDPMix_1_051625.mzML",
    "GG_block_rich": "AlgG_LowDPMix_1_051625.mzML",
}
EXTERNAL_DPS = [3, 4]            # change to [3,4,5,6,7] if those precursors exist
EXTERNAL_ENERGIES = [26.0]      # [18.0, 22.0, 26.0] for all energies
EXTERNAL_RATIO_COUNTS = [2, 4, 6, 8]

# Leave empty to generate a manifest automatically from EXTERNAL_FILES.
# Or provide a CSV with: sample_id,mzml_path,dp,nce,
#                         dp_specific_nmr_M_fraction,bulk_nmr_M_fraction,batch_id
EXTERNAL_MANIFEST_CSV = ""

# References MUST be independently measured DP-SPECIFIC M fractions.
# Bulk NMR alone must NOT be scored as external DP prediction accuracy.
# Example: {("MM_block_rich", 3): 0.81} only if separately validated for DP3.
DP_SPECIFIC_EXTERNAL_REFERENCES = {}

# Optional BULK-NMR reference, displayed as CONTEXT only, not as a DP target.
# Leave blank rather than assuming the batch values match historical material.
BULK_EXTERNAL_REFERENCES = {}

# Independent batch ID if known; do not infer that several scans are replicates.
EXTERNAL_BATCH_ID = ""

# In external mode missing files can be skipped, but no missing model is
# silently substituted with another fold or DP/NCE model.
SKIP_MISSING_EXTERNAL_FILES = True



# =============================================================================
# General helpers
# =============================================================================

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_file_list(fraction: str, ratios: List[int]) -> List[str]:
    return [str(DATA_DIR / f"Raw_{fraction}_M{r}G{100-r}.mzML") for r in ratios]


def get_preprocess_centers(dp: int) -> List[float]:
    upper = float(DP_TARGET_MZ[dp]) + 1e-3
    return [p for p in ALL_FILTER_CENTERS if p <= upper]


def get_ratio_candidate_centers(dp: int) -> List[float]:
    precursor = float(DP_TARGET_MZ[dp])
    return [
        p for p in get_preprocess_centers(dp)
        if abs(float(p) - precursor) > PEAK_WINDOW
    ]


def extract_individual_spectra_from_fs(fs_dict, energy: float, ratios: List[int]):
    scans, y, nominal = [], [], []

    for ratio_int in ratios:
        ratio_float = ratio_int / 100.0
        if ratio_float not in fs_dict:
            continue

        fs = fs_dict[ratio_float]

        for scan in fs.scans.get(energy, []):
            if scan.raw_intensity is None or len(scan.raw_intensity) == 0:
                continue
            if scan.NMR_MG_ratio is None:
                continue

            scans.append(scan)
            y.append(float(scan.NMR_MG_ratio))
            nominal.append(float(ratio_float))

    return (
        scans,
        np.asarray(y, dtype=np.float64),
        np.asarray(nominal, dtype=np.float64),
    )


def spectra_to_matrix(scans, grid: np.ndarray, lo: float, hi: float) -> np.ndarray:
    idx = np.where((grid >= lo) & (grid <= hi))[0]

    X = np.vstack(
        [scan.raw_intensity[idx] for scan in scans]
    ).astype(np.float64)

    return np.clip(X, 0.0, None)


def robust_sd(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return np.nan

    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return 1.4826 * mad if np.isfinite(mad) else np.nan


def composition_balanced_weights(nominal: np.ndarray) -> np.ndarray:
    weights = np.zeros(len(nominal), dtype=float)

    for g in sorted(np.unique(nominal)):
        mask = np.isclose(nominal, g, atol=1e-8)
        n = int(np.sum(mask))
        if n > 0:
            weights[mask] = 1.0 / n

    if np.sum(weights) > 0:
        weights *= len(weights) / np.sum(weights)

    return weights


# =============================================================================
# Load/cache one DP x NCE
# =============================================================================

def load_cached_condition(dp: int, energy: float):
    fraction = f"F{dp}"
    centers = get_preprocess_centers(dp)

    proc = DPSpectrumProcessor(
        conv_sigma=CONV_SIGMA,
        default_groups=GROUP_BOUNDARIES,
    )

    fs = proc.load_scans(
        filepaths=build_file_list(fraction, MIX_RATIOS),
        dp=dp,
        use_group_normalization=True,
        group_boundaries=GROUP_BOUNDARIES,
        filter_peaks=True,
        filter_centers=centers,
        peak_window=PEAK_WINDOW,
        zero_precursor=False,
        zero_window=2.0,
        zero_lower_precursors=False,
    )

    scans, y, nominal = extract_individual_spectra_from_fs(
        fs,
        energy,
        MIX_RATIOS,
    )

    clip_hi = float(DP_TARGET_MZ[dp]) + 5.0

    X = spectra_to_matrix(
        scans,
        proc.grid,
        CLIP_LO,
        clip_hi,
    )

    grid = proc.grid[
        (proc.grid >= CLIP_LO)
        & (proc.grid <= clip_hi)
    ].astype(np.float64)

    return {
        "dp": dp,
        "nce": float(energy),
        "scans": scans,
        "X": X,
        "grid": grid,
        "y": y,
        "nominal": nominal,
    }


# =============================================================================
# NMF
# =============================================================================

def fit_two_component_nmf(X_train: np.ndarray, y_train: np.ndarray):
    active = np.any(X_train > 0, axis=0)

    if int(np.sum(active)) < N_NMF_COMPONENTS:
        raise RuntimeError("Not enough nonzero training columns for NMF.")

    X_active = np.clip(X_train[:, active], 0.0, None)

    nmf = NMF(**NMF_PARAMS)
    C_train = nmf.fit_transform(X_active)

    corr = []
    for j in range(C_train.shape[1]):
        if np.std(C_train[:, j]) == 0 or np.std(y_train) == 0:
            c = 0.0
        else:
            c = float(np.corrcoef(C_train[:, j], y_train)[0, 1])
        corr.append((j, c))

    order = [
        j for j, _ in sorted(
            corr,
            key=lambda z: abs(z[1]),
            reverse=True,
        )
    ]

    nmf.components_ = nmf.components_[order]
    C_train = nmf.transform(X_active)

    return nmf, active, C_train


# =============================================================================
# Pairwise ratio features
# =============================================================================

def build_ratio_matrices(X: np.ndarray, grid: np.ndarray, centers: List[float]):
    peak_values = {}

    for p in centers:
        idx = int(np.argmin(np.abs(grid - p)))
        peak_values[float(p)] = X[:, idx].astype(float)

    ratios, denominators, names, metadata = [], [], [], []

    for i, p1 in enumerate(centers):
        numerator = peak_values[float(p1)]

        for p2 in centers[i + 1:]:
            denominator = peak_values[float(p2)]
            ratio = numerator / (denominator + RATIO_EPS)

            name = f"I_{p1:g}/I_{p2:g}"

            ratios.append(ratio)
            denominators.append(denominator)
            names.append(name)
            metadata.append({
                "name": name,
                "p1": float(p1),
                "p2": float(p2),
            })

    if not ratios:
        raise RuntimeError("No ratio candidates.")

    return (
        np.column_stack(ratios),
        np.column_stack(denominators),
        names,
        metadata,
    )


def composition_statistics(feature, y, nominal):
    rows = []

    for g in sorted(np.unique(nominal)):
        mask = np.isclose(nominal, g, atol=1e-8)
        if not np.any(mask):
            continue

        vals = np.asarray(feature[mask], dtype=float)

        rows.append({
            "nominal": float(g),
            "y_mean": float(np.mean(y[mask])),
            "feature_mean": float(np.mean(vals)),
            "feature_robust_sd": float(robust_sd(vals)),
        })

    rows.sort(key=lambda r: r["y_mean"])
    return rows


def min_denom_valid_fraction_by_composition(denominator, nominal):
    fractions = []

    for g in sorted(np.unique(nominal)):
        mask = np.isclose(nominal, g, atol=1e-8)
        if not np.any(mask):
            continue

        fractions.append(
            float(np.mean(denominator[mask] > DENOM_MIN))
        )

    return float(np.min(fractions)) if fractions else 0.0


def evaluate_ratio_candidate(feature, denominator, y, nominal):
    min_valid = min_denom_valid_fraction_by_composition(
        denominator,
        nominal,
    )

    if min_valid < MIN_DENOM_VALID_FRACTION_PER_COMPOSITION:
        return {
            "eligible": False,
            "failure_reason": "denominator_reliability",
            "min_denom_valid_fraction": min_valid,
        }

    stats = composition_statistics(feature, y, nominal)

    if len(stats) < 4:
        return {
            "eligible": False,
            "failure_reason": "too_few_compositions",
            "min_denom_valid_fraction": min_valid,
        }

    y_means = np.asarray([r["y_mean"] for r in stats], dtype=float)
    f_means = np.asarray([r["feature_mean"] for r in stats], dtype=float)
    within_sd = np.asarray(
        [r["feature_robust_sd"] for r in stats],
        dtype=float,
    )

    if not np.all(np.isfinite(f_means)) or np.std(f_means) == 0:
        return {
            "eligible": False,
            "failure_reason": "constant_or_nonfinite",
            "min_denom_valid_fraction": min_valid,
        }

    rho_s = float(spearmanr(y_means, f_means).statistic)
    rho_p = float(pearsonr(y_means, f_means).statistic)

    if not np.isfinite(rho_s) or not np.isfinite(rho_p):
        return {
            "eligible": False,
            "failure_reason": "correlation_nonfinite",
            "min_denom_valid_fraction": min_valid,
        }

    direction = 1.0 if rho_s >= 0 else -1.0
    diffs = np.diff(f_means) * direction

    tolerance = 0.02 * max(float(np.ptp(f_means)), 1e-12)
    violations = int(np.sum(diffs < -tolerance))

    between_range = float(np.ptp(f_means))
    valid_sd = within_sd[np.isfinite(within_sd)]

    median_within_sd = (
        float(np.median(valid_sd))
        if valid_sd.size
        else np.nan
    )

    technical_snr = (
        between_range / (median_within_sd + STABILITY_EPS)
        if np.isfinite(median_within_sd)
        else np.nan
    )

    eligible = (
        abs(rho_s) >= MIN_ABS_SPEARMAN
        and violations <= MAX_MONOTONIC_VIOLATIONS
    )

    failure_reason = ""
    if not eligible:
        if abs(rho_s) < MIN_ABS_SPEARMAN:
            failure_reason = "spearman"
        else:
            failure_reason = "monotonicity"

    return {
        "eligible": bool(eligible),
        "failure_reason": failure_reason,
        "min_denom_valid_fraction": min_valid,
        "spearman": rho_s,
        "abs_spearman": abs(rho_s),
        "pearson": rho_p,
        "abs_pearson": abs(rho_p),
        "monotonic_violations": violations,
        "between_composition_range": between_range,
        "median_within_robust_sd": median_within_sd,
        "technical_snr": float(technical_snr),
        "trajectory": f_means,
    }


def rank_ratio_candidates(
    R_train,
    D_train,
    y_train,
    nominal_train,
    names,
    metadata,
):
    diagnostics = []
    eligible = []

    for j, name in enumerate(names):
        stat = evaluate_ratio_candidate(
            R_train[:, j],
            D_train[:, j],
            y_train,
            nominal_train,
        )

        row = dict(metadata[j])
        row["idx"] = int(j)

        for key, value in stat.items():
            if key != "trajectory":
                row[key] = value

        diagnostics.append(row)

        if stat.get("eligible", False):
            item = dict(row)
            item["trajectory"] = stat["trajectory"]
            eligible.append(item)

    eligible.sort(
        key=lambda z: (
            z["abs_spearman"],
            z["technical_snr"] if np.isfinite(z["technical_snr"]) else -np.inf,
            z["abs_pearson"],
            z["min_denom_valid_fraction"],
        ),
        reverse=True,
    )

    return eligible, diagnostics


def trajectory_redundancy(a, b):
    if (
        len(a) != len(b)
        or len(a) < 3
        or np.std(a) == 0
        or np.std(b) == 0
    ):
        return 0.0

    r = float(np.corrcoef(a, b)[0, 1])
    return abs(r) if np.isfinite(r) else 0.0


def select_nonredundant_ratios(ranked, n_select):
    """Choose exactly K eligible ratios, with a SOFT diversity penalty.

    No eligible ratio is excluded because it closely follows the same
    monotonic M%-trajectory as another. On six training compositions these
    correlations are expected to be high. Quality stays the main criterion.
    Scored independently within each outer training fold (never test labels).
    """
    selected = []
    available = list(ranked)
    while available and len(selected) < n_select:
        scored = []
        for feat in available:
            # Cap stability to prevent an artificially tiny MAD dominating.
            bounded_snr = min(max(float(feat['technical_snr']),0.0), 50.0)
            score = (
                0.57 * float(feat['abs_spearman'])
                + 0.23 * float(feat['abs_pearson'])
                + 0.12 * np.log1p(bounded_snr)/np.log1p(50.0)
                + 0.08 * float(feat['min_denom_valid_fraction'])
                - 0.02 * float(feat['monotonic_violations'])
            )
            max_corr = max(
                (trajectory_redundancy(feat['trajectory'], other['trajectory'])
                 for other in selected),
                default=0.0,
            )
            score -= REDUNDANCY_PENALTY * max_corr**4
            scored.append((score, feat['name'], feat, max_corr))
        best_score, _, feat, max_corr = max(scored, key=lambda z:(z[0],z[1]))
        item = dict(feat)
        item['selection_score'] = float(best_score)
        item['max_trajectory_correlation_selected'] = float(max_corr)
        selected.append(item)
        available = [v for v in available if v['idx'] != feat['idx']]

    peak_use = Counter()
    for feat in selected:
        peak_use[feat['p1']] += 1
        peak_use[feat['p2']] += 1
    for feat in selected:
        feat['numerator_peak_use_in_selected_set'] = int(peak_use[feat['p1']])
        feat['denominator_peak_use_in_selected_set'] = int(peak_use[feat['p2']])
    return selected


# =============================================================================
# One outer LORO fold
# =============================================================================

def run_outer_fold(cache, holdout_ratio):
    fold_start = time.time()

    dp = cache["dp"]
    nce = cache["nce"]
    X_all = cache["X"]
    y_all = cache["y"]
    nominal_all = cache["nominal"]
    scans_all = cache["scans"]
    grid = cache["grid"]

    holdout_float = holdout_ratio / 100.0

    train_mask = ~np.isclose(nominal_all, holdout_float, atol=1e-8)
    test_mask = np.isclose(nominal_all, holdout_float, atol=1e-8)

    X_train = X_all[train_mask]
    y_train = y_all[train_mask]
    nominal_train = nominal_all[train_mask]

    X_test = X_all[test_mask]
    y_test = y_all[test_mask]

    test_scans = [
        scan
        for scan, keep in zip(scans_all, test_mask)
        if keep
    ]

    # NMF is fit ONCE per outer fold and reused for all ratio counts.
    nmf, active_cols, C_train = fit_two_component_nmf(
        X_train,
        y_train,
    )
    C_test = nmf.transform(X_test[:, active_cols])

    ratio_centers = get_ratio_candidate_centers(dp)

    R_train, D_train, ratio_names, ratio_meta = build_ratio_matrices(
        X_train,
        grid,
        ratio_centers,
    )
    R_test, _, _, _ = build_ratio_matrices(
        X_test,
        grid,
        ratio_centers,
    )

    ranked, diagnostics = rank_ratio_candidates(
        R_train,
        D_train,
        y_train,
        nominal_train,
        ratio_names,
        ratio_meta,
    )

    diagnostic_rows = []
    for row in diagnostics:
        rr = dict(row)
        rr.update({
            "dp": dp,
            "nce": nce,
            "outer_holdout_ratio": holdout_ratio,
        })
        diagnostic_rows.append(rr)

    sample_weight = composition_balanced_weights(nominal_train)

    fold_rows = []
    scan_rows = []
    selected_rows = []

    for n_ratios in RATIO_COUNTS:
        selected = select_nonredundant_ratios(
            ranked,
            n_ratios,
        )

        if len(selected) < n_ratios:
            fold_rows.append({
                "dp": dp,
                "nce": nce,
                "outer_holdout_ratio": holdout_ratio,
                "n_ratios_requested": n_ratios,
                "n_ratios_selected": len(selected),
                "status": "insufficient_eligible_nonredundant_ratios",
                "true_M_percent": 100.0 * float(np.mean(y_test)),
                "mean_pred_M_percent": np.nan,
                "fold_abs_error_pct": np.nan,
                "scan_level_mae_pct": np.nan,
                "scan_level_rmse_pct": np.nan,
                "prediction_sd_pct": np.nan,
                "n_train_scans": int(len(y_train)),
                "n_test_scans": int(len(y_test)),
                "n_eligible_candidates": int(len(ranked)),
                "selected_ratios": ";".join(f["name"] for f in selected),
                "fold_seconds": float(time.time() - fold_start),
            })
            continue

        selected_idx = [f["idx"] for f in selected]

        F_train = np.column_stack([
            C_train,
            R_train[:, selected_idx],
        ])
        F_test = np.column_stack([
            C_test,
            R_test[:, selected_idx],
        ])

        scaler = StandardScaler()
        scaler.fit(
            F_train,
            sample_weight=sample_weight,
        )

        F_train_scaled = scaler.transform(F_train)
        F_test_scaled = scaler.transform(F_test)

        model = Ridge(alpha=RIDGE_ALPHA)
        model.fit(
            F_train_scaled,
            y_train,
            sample_weight=sample_weight,
        )

        pred_test = np.clip(
            model.predict(F_test_scaled),
            0.0,
            1.0,
        )

        true_fold = float(np.mean(y_test))
        pred_mean = float(np.mean(pred_test))
        pred_sd = (
            float(np.std(pred_test, ddof=1))
            if len(pred_test) > 1
            else 0.0
        )

        fold_abs_error = abs(pred_mean - true_fold)

        scan_mae = float(mean_absolute_error(y_test, pred_test))
        scan_rmse = float(
            np.sqrt(mean_squared_error(y_test, pred_test))
        )

        fold_rows.append({
            "dp": dp,
            "nce": nce,
            "outer_holdout_ratio": holdout_ratio,
            "n_ratios_requested": n_ratios,
            "n_ratios_selected": len(selected),
            "status": "ok",
            "true_M_percent": 100.0 * true_fold,
            "mean_pred_M_percent": 100.0 * pred_mean,
            "fold_abs_error_pct": 100.0 * fold_abs_error,
            "scan_level_mae_pct": 100.0 * scan_mae,
            "scan_level_rmse_pct": 100.0 * scan_rmse,
            "prediction_sd_pct": 100.0 * pred_sd,
            "n_train_scans": int(len(y_train)),
            "n_test_scans": int(len(y_test)),
            "n_eligible_candidates": int(len(ranked)),
            "selected_ratios": ";".join(f["name"] for f in selected),
            "fold_seconds": float(time.time() - fold_start),
        })

        for rank, feat in enumerate(selected, start=1):
            selected_rows.append({
                "dp": dp,
                "nce": nce,
                "outer_holdout_ratio": holdout_ratio,
                "n_ratios": n_ratios,
                "selection_rank": rank,
                "feature": feat["name"],
                "numerator_mz": feat["p1"],
                "denominator_mz": feat["p2"],
                "spearman": feat["spearman"],
                "abs_spearman": feat["abs_spearman"],
                "pearson": feat["pearson"],
                "abs_pearson": feat["abs_pearson"],
                "monotonic_violations": feat["monotonic_violations"],
                "technical_snr": feat["technical_snr"],
                "selection_score": feat['selection_score'],
                "max_trajectory_correlation_selected": feat['max_trajectory_correlation_selected'],
                "between_composition_range": feat["between_composition_range"],
                "median_within_robust_sd": feat["median_within_robust_sd"],
                "min_denom_valid_fraction": feat["min_denom_valid_fraction"],
                "selection_score": feat['selection_score'],
                "max_trajectory_correlation_selected": feat['max_trajectory_correlation_selected'],
                "numerator_peak_use": feat[
                    "numerator_peak_use_in_selected_set"
                ],
                "denominator_peak_use": feat[
                    "denominator_peak_use_in_selected_set"
                ],
            })

        for scan, yt, yp in zip(test_scans, y_test, pred_test):
            scan_rows.append({
                "dp": dp,
                "nce": nce,
                "outer_holdout_ratio": holdout_ratio,
                "n_ratios": n_ratios,
                "scan_time": (
                    float(scan.scan_start_time)
                    if scan.scan_start_time is not None
                    else np.nan
                ),
                "true_M_percent": 100.0 * float(yt),
                "pred_M_percent": 100.0 * float(yp),
                "abs_error_pct": 100.0 * abs(float(yt) - float(yp)),
            })

    return fold_rows, scan_rows, selected_rows, diagnostic_rows


# =============================================================================
# Final full-calibration feature sets
# =============================================================================

def select_final_ratios_full_calibration(cache):
    dp = cache["dp"]
    nce = cache["nce"]

    centers = get_ratio_candidate_centers(dp)

    R, D, names, meta = build_ratio_matrices(
        cache["X"],
        cache["grid"],
        centers,
    )

    ranked, _ = rank_ratio_candidates(
        R,
        D,
        cache["y"],
        cache["nominal"],
        names,
        meta,
    )

    rows = []

    for n_ratios in RATIO_COUNTS:
        selected = select_nonredundant_ratios(
            ranked,
            n_ratios,
        )

        status = "ok" if len(selected) == n_ratios else "insufficient"

        for rank, feat in enumerate(selected, start=1):
            rows.append({
                "dp": dp,
                "nce": nce,
                "n_ratios": n_ratios,
                "status": status,
                "selection_rank": rank,
                "feature": feat["name"],
                "numerator_mz": feat["p1"],
                "denominator_mz": feat["p2"],
                "spearman_all_7": feat["spearman"],
                "pearson_all_7": feat["pearson"],
                "monotonic_violations_all_7": feat[
                    "monotonic_violations"
                ],
                "technical_snr_all_7": feat["technical_snr"],
                "min_denom_valid_fraction_all_7": feat[
                    "min_denom_valid_fraction"
                ],
                "numerator_peak_use": feat[
                    "numerator_peak_use_in_selected_set"
                ],
                "denominator_peak_use": feat[
                    "denominator_peak_use_in_selected_set"
                ],
            })

    return rows


# =============================================================================
# Summaries
# =============================================================================

def make_overall_summary(folds):
    ok = folds[folds["status"] == "ok"].copy()

    return (
        ok
        .groupby(
            ["dp", "nce", "n_ratios_requested"],
            as_index=False,
        )
        .agg(
            mean_loro_mae_pct=("fold_abs_error_pct", "mean"),
            sd_loro_abs_error_pct=("fold_abs_error_pct", "std"),
            mean_prediction_sd_pct=("prediction_sd_pct", "mean"),
            mean_scan_level_mae_pct=("scan_level_mae_pct", "mean"),
            n_valid_outer_folds=("outer_holdout_ratio", "nunique"),
            mean_eligible_candidates=("n_eligible_candidates", "mean"),
        )
    )


def make_global_ratio_count_summary(overall):
    complete = overall[
        overall["n_valid_outer_folds"] == len(MIX_RATIOS)
    ].copy()

    return (
        complete
        .groupby("n_ratios_requested", as_index=False)
        .agg(
            mean_mae_across_15_models_pct=("mean_loro_mae_pct", "mean"),
            median_mae_across_15_models_pct=("mean_loro_mae_pct", "median"),
            sd_mae_across_15_models_pct=("mean_loro_mae_pct", "std"),
            mean_prediction_sd_across_models_pct=(
                "mean_prediction_sd_pct",
                "mean",
            ),
            n_dp_nce_models=("mean_loro_mae_pct", "size"),
        )
    )


def make_best_ratio_count_by_condition(overall):
    rows = []

    for (dp, nce), g in overall.groupby(["dp", "nce"]):
        g = g[
            g["n_valid_outer_folds"] == len(MIX_RATIOS)
        ].copy()

        if g.empty:
            continue

        best_mae = float(g["mean_loro_mae_pct"].min())

        # Exploratory parsimonious tie rule:
        # choose smallest K within 0.10 percentage points of the minimum.
        acceptable = g[
            g["mean_loro_mae_pct"] <= best_mae + 0.10
        ].sort_values("n_ratios_requested")

        chosen = acceptable.iloc[0]

        rows.append({
            "dp": dp,
            "nce": nce,
            "minimum_observed_mae_pct": best_mae,
            "parsimonious_ratio_count_within_0.10pp": int(
                chosen["n_ratios_requested"]
            ),
            "mae_of_parsimonious_choice_pct": float(
                chosen["mean_loro_mae_pct"]
            ),
        })

    return pd.DataFrame(rows)


def make_feature_stability(selected):
    if selected.empty:
        return pd.DataFrame()

    return (
        selected
        .groupby(
            [
                "dp",
                "nce",
                "n_ratios",
                "feature",
                "numerator_mz",
                "denominator_mz",
            ],
            as_index=False,
        )
        .agg(
            selected_in_folds=("outer_holdout_ratio", "nunique"),
            mean_selection_rank=("selection_rank", "mean"),
            mean_abs_spearman=("abs_spearman", "mean"),
            mean_abs_pearson=("abs_pearson", "mean"),
            mean_technical_snr=("technical_snr", "mean"),
            mean_monotonic_violations=("monotonic_violations", "mean"),
            min_denom_valid_fraction_seen=(
                "min_denom_valid_fraction",
                "min",
            ),
        )
        .sort_values(
            [
                "dp",
                "nce",
                "n_ratios",
                "selected_in_folds",
                "mean_selection_rank",
            ],
            ascending=[True, True, True, False, True],
        )
    )


# =============================================================================
# Main
# =============================================================================

def main():
    ensure_dir(OUT_ROOT)

    if len(ALL_FILTER_CENTERS) != 24:
        raise RuntimeError(
            f"Expected 24 peak centers, found {len(ALL_FILTER_CENTERS)}."
        )

    config = {
        "MIX_RATIOS": MIX_RATIOS,
        "DPS": DPS,
        "ENERGIES": ENERGIES,
        "RATIO_COUNTS": RATIO_COUNTS,
        "N_NMF_COMPONENTS": N_NMF_COMPONENTS,
        "RIDGE_ALPHA": RIDGE_ALPHA,
        "PRECURSOR_RETAINED": True,
        "AUTOMATIC_OUTLIER_FILTER": False,
        "MIN_DENOM_VALID_FRACTION_PER_COMPOSITION":
            MIN_DENOM_VALID_FRACTION_PER_COMPOSITION,
        "MIN_ABS_SPEARMAN": MIN_ABS_SPEARMAN,
        "MAX_MONOTONIC_VIOLATIONS": MAX_MONOTONIC_VIOLATIONS,
        "REDUNDANCY_PENALTY": REDUNDANCY_PENALTY,
        "ranking": [
            "within_training_fold_composite_quality",
            "soft_trajectory_diversity_penalty",
        ],
        "primary_validation_metric":
            "mean of seven composition-level outer-LORO absolute errors",
        "note":
            "Exploratory global ratio-count model-development experiment.",
    }

    with open(OUT_ROOT / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)

    fold_rows = []
    scan_rows = []
    selected_rows = []
    diagnostic_rows = []
    final_rows = []

    total_outer_folds = (
        len(DPS)
        * len(ENERGIES)
        * len(MIX_RATIOS)
    )

    completed = 0
    start_global = time.time()

    for dp in DPS:
        for nce in ENERGIES:
            log(f"Loading/preprocessing once: DP{dp} NCE{nce:g}")

            cache = load_cached_condition(dp, nce)

            final_rows.extend(
                select_final_ratios_full_calibration(cache)
            )
            # These refits are separate from validation and are intended
            # only for future, genuinely independent external samples.
            fit_and_save_all_full_models(cache)

            pd.DataFrame(final_rows).to_csv(
                OUT_ROOT / "final_ratios_full_calibration.csv",
                index=False,
            )

            for holdout in MIX_RATIOS:
                completed += 1

                log(
                    f"[{completed}/{total_outer_folds}] "
                    f"DP{dp} NCE{nce:g} holdout={holdout}"
                )

                (
                    folds_this,
                    scans_this,
                    selected_this,
                    diagnostics_this,
                ) = run_outer_fold(
                    cache,
                    holdout,
                )

                fold_rows.extend(folds_this)
                scan_rows.extend(scans_this)
                selected_rows.extend(selected_this)
                diagnostic_rows.extend(diagnostics_this)

                # Incremental checkpoints.
                pd.DataFrame(fold_rows).to_csv(
                    OUT_ROOT / "fold_predictions.csv",
                    index=False,
                )
                pd.DataFrame(selected_rows).to_csv(
                    OUT_ROOT / "selected_ratios_by_fold.csv",
                    index=False,
                )
                pd.DataFrame(diagnostic_rows).to_csv(
                    OUT_ROOT / "candidate_diagnostics.csv",
                    index=False,
                )

                avg_sec = (time.time() - start_global) / completed
                remaining = total_outer_folds - completed
                eta_h = avg_sec * remaining / 3600.0

                log(f"ETA≈{eta_h:.2f} h")

    # Large export written once at the end.
    pd.DataFrame(scan_rows).to_csv(
        OUT_ROOT / "scan_predictions.csv",
        index=False,
    )

    folds = pd.DataFrame(fold_rows)
    selected = pd.DataFrame(selected_rows)

    overall = make_overall_summary(folds)
    global_summary = make_global_ratio_count_summary(overall)
    best_by_condition = make_best_ratio_count_by_condition(overall)
    stability = make_feature_stability(selected)

    overall.to_csv(
        OUT_ROOT / "overall_by_dp_nce_ratio_count.csv",
        index=False,
    )
    global_summary.to_csv(
        OUT_ROOT / "global_ratio_count_summary.csv",
        index=False,
    )
    best_by_condition.to_csv(
        OUT_ROOT / "best_ratio_count_by_condition.csv",
        index=False,
    )
    stability.to_csv(
        OUT_ROOT / "feature_stability_by_ratio_count.csv",
        index=False,
    )

    total_h = (time.time() - start_global) / 3600.0

    log("")
    log(f"Experiment complete in {total_h:.2f} h.")
    log(f"Results: {OUT_ROOT.resolve()}")
    log("")
    log("Start with global_ratio_count_summary.csv.")
    log(
        "Prefer one global ratio count for all DP/NCE models rather than "
        "choosing a different count per condition from the same LORO results."
    )



# =============================================================================
# Frozen full-calibration models and external evaluation
# =============================================================================

def fit_and_save_all_full_models(cache):
    """Refit each K on all seven *calibration* compositions and save model.

    DO NOT use these in the outer LORO fold predictions: they have seen all
    seven calibration labels. Only apply them to NEW independent samples.
    """
    dp, nce = int(cache['dp']), float(cache['nce'])
    X, y, nominal, grid = cache['X'], cache['y'], cache['nominal'], cache['grid']
    nmf, active, C = fit_two_component_nmf(X, y)
    centers = get_ratio_candidate_centers(dp)
    R, D, names, meta = build_ratio_matrices(X, grid, centers)
    ranked, diagnostics = rank_ratio_candidates(R, D, y, nominal, names, meta)
    weights = composition_balanced_weights(nominal)
    bdir = OUT_ROOT / 'frozen_models'
    ensure_dir(bdir)

    for k in RATIO_COUNTS:
        selected = select_nonredundant_ratios(ranked, k)
        if len(selected) != k:
            raise RuntimeError(f'DP{dp} NCE{nce:g}: only {len(selected)} eligible ratios for K={k}')
        indices = [int(f['idx']) for f in selected]
        features = np.column_stack((C, R[:, indices]))
        scaler = StandardScaler()
        scaler.fit(features, sample_weight=weights)
        ridge = Ridge(alpha=RIDGE_ALPHA)
        ridge.fit(scaler.transform(features), y, sample_weight=weights)
        means = np.asarray([
            np.mean(features[np.isclose(nominal,g,atol=1e-8)], axis=0)
            for g in sorted(np.unique(nominal))
        ])
        bundle = {
            'dp': dp, 'nce': nce, 'k': k,
            'mz_axis': np.array(grid,copy=True),
            'nmf_active_columns': np.array(active,copy=True),
            'nmf': nmf, 'scaler': scaler, 'ridge': ridge,
            'ratio_centers': list(centers), 'selected_indices': indices,
            'selected_ratios': [f['name'] for f in selected],
            'selected_numerator_mz': [float(f['p1']) for f in selected],
            'selected_denominator_mz': [float(f['p2']) for f in selected],
            'training_feature_composition_means': means,
            'training_nmr_fractions': sorted(set(float(v) for v in y)),
            'training_scan_count': int(len(y)),
            'preprocessing': 'same 24-center processor, area-normalization, full NMF m/z, precursor retained',
        }
        file = bdir / f'dp{dp}_nce{int(nce)}_K{k}.joblib'
        joblib.dump(bundle, file, compress=3)
    log(f'Saved {len(RATIO_COUNTS)} frozen full-calibration model bundles for DP{dp} NCE{nce:g}')


def load_external_spectra_generic(path, dp, nce):
    """Read arbitrary external mzML without filename parsing or NMR labels.

    Identical scan processing to DPSpectrumProcessor.load_scans:
    precursor matching, interpolation, smoothing, 24-center peak window,
    and group-area normalization. No reference M fraction is read here.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    proc = DPSpectrumProcessor(
        conv_sigma=CONV_SIGMA,
        default_groups=GROUP_BOUNDARIES,
    )
    kept = []
    n_matching = 0
    n_preproc_failed = 0
    with mzml.read(str(path)) as reader:
        for spec in reader:
            if spec.get('ms level') != 2 or 'precursorList' not in spec:
                continue
            try:
                precursor = spec['precursorList']['precursor'][0]
                isolation_target = precursor.get('isolationWindow',{}).get('isolation window target m/z')
                energy = precursor.get('activation',{}).get('collision energy')
                if isolation_target is None or energy is None:
                    continue
                if abs(float(isolation_target) - DP_TARGET_MZ[dp]) >= 1e-4:
                    continue
                if abs(float(energy)-float(nce)) > 1e-5:
                    continue
                n_matching += 1
                raw_mz = np.asarray(spec['m/z array'],dtype=np.float32)
                raw_int = np.asarray(spec['intensity array'],dtype=np.float32)
                mask = raw_mz <= DP_TARGET_MZ[dp] + 20.0
                raw_mz, raw_int = raw_mz[mask], raw_int[mask]
                if len(raw_mz) < 4 or not np.all(np.isfinite(raw_int)):
                    n_preproc_failed += 1
                    continue
                mzgrid, interpolated, _, _ = FractionSpectra.interpolate_to_grid_without_norm(
                    raw_mz,raw_int,proc.grid
                )
                smoothed = proc._smooth_impl(mzgrid,interpolated)
                smoothed = filter_peaks_by_centers(
                    mzgrid,smoothed,get_preprocess_centers(dp),window=PEAK_WINDOW
                )
                normalized = FractionSpectra.group_area_normalization(
                    mzgrid,smoothed,GROUP_BOUNDARIES
                )
                clip_mask = (mzgrid>=CLIP_LO) & (mzgrid<=DP_TARGET_MZ[dp]+5.0)
                x = np.clip(np.asarray(normalized[clip_mask],dtype=float),0,None)
                if np.any(~np.isfinite(x)) or not np.any(x>0):
                    n_preproc_failed += 1
                    continue
                kept.append(x)
            except (ValueError,IndexError,KeyError,TypeError):
                n_preproc_failed += 1
                continue
    if not kept:
        raise RuntimeError(
            f'No valid target DP{dp} NCE{nce:g} scans in {path}; '
            f'matching MS2 scans={n_matching}; processing failed={n_preproc_failed}. '
            'Check precursor isolation target, collision energy, and mzML file.'
        )
    return np.vstack(kept),n_matching,n_preproc_failed


def float_or_nan(value):
    if value is None or str(value).strip()=='' or str(value).strip().lower()=='nan':
        return np.nan
    return float(value)


def external_predict(manifest_path, bundle_root=None):
    """Predict independent samples using frozen models ONLY (no refitting).

    Reference DP-specific NMR values are used only *after* prediction for
    comparison. Bulk NMR is printed as a contextual difference, NOT accuracy.
    """
    if bundle_root is None:
        bundle_root = OUT_ROOT / 'frozen_models'
    else:
        bundle_root = Path(bundle_root)
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = pd.read_csv(manifest_path,keep_default_na=False)
    required={'sample_id','mzml_path','dp','nce'}
    if not required.issubset(manifest.columns):
        raise ValueError(f'External manifest requires columns: {sorted(required)}')
    ensure_dir(OUT_ROOT)
    pred_rows=[]
    scan_rows=[]
    ratio_diagnostic_rows=[]
    for _,record in manifest.iterrows():
        sample_id=str(record['sample_id']).strip()
        path = Path(str(record['mzml_path']).strip())
        if not path.is_absolute():
            path = manifest_path.parent / path
        dp,nce = int(record['dp']), float(record['nce'])
        if dp not in DPS or nce not in ENERGIES:
            raise ValueError(f'Unsupported external DP/NCE combination: {dp}/{nce}')
        train_paths = [Path(x).resolve() for x in build_file_list(f"F{dp}", MIX_RATIOS)]
        if path.resolve() in train_paths or re.fullmatch(
            r"Raw_F[3-7]_M(?:0|20|40|50|60|80|100)G(?:0|20|40|50|60|80|100)\.mzML",
            path.name, flags=re.IGNORECASE
        ):
            raise ValueError(f'External manifest points to a calibration training file: {path}')
        log(f'External independent sample {sample_id}: DP{dp}, NCE{nce:g}, file={path}')
        # External data are read independently of any calibration file naming.
        X,n_matching,n_failed=load_external_spectra_generic(path,dp,nce)
        dp_nmr=float_or_nan(record.get('dp_specific_nmr_M_fraction',''))
        bulk_nmr=float_or_nan(record.get('bulk_nmr_M_fraction',''))
        if (np.isfinite(dp_nmr) and not 0 <= dp_nmr <= 1) or (np.isfinite(bulk_nmr) and not 0 <= bulk_nmr <= 1):
            raise ValueError('NMR fractions in external manifest must be on the 0-to-1 scale, not percentages.')
        batch=str(record.get('batch_id','')).strip()
        for k in EXTERNAL_RATIO_COUNTS:
            file=bundle_root / f'dp{dp}_nce{int(nce)}_K{k}.joblib'
            if not file.exists():
                raise FileNotFoundError(f'Missing frozen model: {file}. Run --mode experiment first.')
            b=joblib.load(file)  # Only load trusted model bundles you generated locally.
            expected_axis = np.asarray(b['mz_axis'], dtype=float)
            generated_axis = DPSpectrumProcessor(
                conv_sigma=CONV_SIGMA, default_groups=GROUP_BOUNDARIES
            ).grid
            generated_axis = generated_axis[
                (generated_axis >= CLIP_LO) &
                (generated_axis <= DP_TARGET_MZ[dp] + 5.0)
            ]
            if (int(b['dp']) != dp or abs(float(b['nce'])-nce) > 1e-5
                or int(b['k']) != k):
                raise ValueError(f'Wrong DP/NCE/K frozen bundle loaded: {file}')
            if (X.shape[1] != len(expected_axis)
                or len(expected_axis) != len(generated_axis)
                or not np.allclose(expected_axis, generated_axis, rtol=0, atol=1e-6)):
                raise ValueError(f'External preprocessing m/z axis differs from saved calibration model: {file}')
            C=b['nmf'].transform(X[:,b['nmf_active_columns']])
            R,D,names,metadata=build_ratio_matrices(X,b['mz_axis'],b['ratio_centers'])
            chosen=b['selected_indices']
            F=np.column_stack([C,R[:,chosen]])
            yp=np.clip(b['ridge'].predict(b['scaler'].transform(F)),0.0,1.0)
            ymean=float(np.mean(yp))
            ext_feature_mean=np.mean(F,axis=0)
            tr_means=np.asarray(b['training_feature_composition_means'])
            lo,hi=np.min(tr_means,axis=0),np.max(tr_means,axis=0)
            out_of_range=(ext_feature_mean < lo) | (ext_feature_mean > hi)
            ratio_out_of_range_fraction=float(np.mean(out_of_range[2:]))
            denom_valid_fractions=[float(np.mean(D[:,idx]>DENOM_MIN)) for idx in chosen]
            low_denom=[b['selected_ratios'][i] for i,v in enumerate(denom_valid_fractions) if v<0.95]

            # Audit actual calibration-vs-external ratio shifts. These are
            # *not* independent reference-label comparisons.
            for i, col_idx in enumerate(chosen):
                ratio_train_values = tr_means[:, 2+i]
                ratio_ext_values = R[:, col_idx]
                ratio_diagnostic_rows.append({
                    'sample_id':sample_id, 'batch_id':batch,
                    'dp':dp,'nce':nce,'n_ratios':k,
                    'feature':b['selected_ratios'][i],
                    'numerator_mz':b['selected_numerator_mz'][i],
                    'denominator_mz':b['selected_denominator_mz'][i],
                    'external_mean_ratio':float(np.mean(ratio_ext_values)),
                    'external_median_ratio':float(np.median(ratio_ext_values)),
                    'external_sd_ratio':float(np.std(ratio_ext_values,ddof=1)) if len(ratio_ext_values)>1 else 0.,
                    'training_min_composition_mean_ratio':float(np.min(ratio_train_values)),
                    'training_max_composition_mean_ratio':float(np.max(ratio_train_values)),
                    'external_mean_outside_training_range':bool(out_of_range[2+i]),
                    'denominator_valid_scan_fraction':denom_valid_fractions[i],
                })

            # Diagnostic plot: separate technical-scan prediction distributions.
            plot_dir = OUT_ROOT / 'external_plots'
            ensure_dir(plot_dir)
            fig, ax = plt.subplots(figsize=(7.2,4.3))
            ax.hist(100*yp, bins=min(35,max(5,int(np.sqrt(len(yp))))), alpha=0.8)
            ax.axvline(100*ymean, color='black', linestyle='--', label=f'Mean: {100*ymean:.1f}%')
            ax.set_xlim(0,100)
            ax.set_xlabel('Predicted M fraction (%)')
            ax.set_ylabel('Technical MS/MS scans')
            ax.set_title(f'{sample_id} | DP{dp} NCE{nce:g} K={k}')
            ax.legend()
            fig.tight_layout()
            safe_id = re.sub(r'[^a-zA-Z0-9_-]+','_',sample_id)
            fig.savefig(plot_dir / f'{safe_id}_dp{dp}_nce{int(nce)}_K{k}.png',dpi=160)
            plt.close(fig)

            pred_rows.append({
                'sample_id':sample_id,'batch_id':batch,'mzml_path':str(path),'dp':dp,'nce':nce,'n_ratios':k,
                'n_matching_raw_scans':n_matching,'n_valid_scans':len(yp),'n_skipped_preproc':n_failed,
                'pred_M_percent':100.0*ymean,
                'scan_prediction_sd_pp':100.0*float(np.std(yp,ddof=1)) if len(yp)>1 else 0.0,
                'scan_prediction_p05_percent':100.0*float(np.quantile(yp,0.05)),
                'scan_prediction_p95_percent':100.0*float(np.quantile(yp,0.95)),
                'dp_specific_reference_M_percent':100.0*dp_nmr if np.isfinite(dp_nmr) else np.nan,
                'dp_specific_abs_error_pp':100.0*abs(ymean-dp_nmr) if np.isfinite(dp_nmr) else np.nan,
                'bulk_nmr_M_percent_context_only':100.0*bulk_nmr if np.isfinite(bulk_nmr) else np.nan,
                'difference_from_bulk_pp_NOT_DP_ACCURACY':100.0*(ymean-bulk_nmr) if np.isfinite(bulk_nmr) else np.nan,
                'fraction_ratios_outside_calibration_composition_mean_ranges':ratio_out_of_range_fraction,
                'n_NMF_coefficients_outside_calibration_means':int(np.sum(out_of_range[:2])),
                'warning_low_denominator':bool(low_denom),
                'warning_ratio_feature_shift':bool(ratio_out_of_range_fraction>0.5),
                'minimum_selected_denominator_valid_scan_fraction':min(denom_valid_fractions),
                'selected_ratios_below_95pct_denominator_valid':';'.join(low_denom),
                'selected_ratios':';'.join(b['selected_ratios']),
            })
            # Scan-level predictions are technical measurements, not independent external samples.
            scan_rows.extend({
                'sample_id':sample_id,'batch_id':batch,'dp':dp,'nce':nce,'n_ratios':k,
                'technical_scan_index':i,'pred_M_percent':100.0*float(v),
            } for i,v in enumerate(yp))
    preds=pd.DataFrame(pred_rows)
    preds.to_csv(OUT_ROOT/'external_predictions_by_sample.csv',index=False)
    pd.DataFrame(scan_rows).to_csv(OUT_ROOT/'external_predictions_scan_level.csv',index=False)
    pd.DataFrame(ratio_diagnostic_rows).to_csv(
        OUT_ROOT/'external_selected_ratio_diagnostics.csv',index=False
    )
    # Samples with a genuine DP-specific independent reference only.
    known=preds[preds['dp_specific_abs_error_pp'].notna()].copy()
    if not known.empty:
        agg=known.groupby('n_ratios',as_index=False).agg(
            mean_external_dp_reference_mae_pp=('dp_specific_abs_error_pp','mean'),
            median_external_dp_reference_mae_pp=('dp_specific_abs_error_pp','median'),
            n_independent_sample_condition_rows=('sample_id','size'),
            n_unique_sample_ids=('sample_id','nunique'),
            n_unique_batches=('batch_id','nunique'),
        )
        agg.to_csv(OUT_ROOT/'external_dp_reference_accuracy.csv',index=False)
        by_condition = known.groupby(['dp','nce','n_ratios'],as_index=False).agg(
            mean_external_dp_reference_mae_pp=('dp_specific_abs_error_pp','mean'),
            median_external_dp_reference_mae_pp=('dp_specific_abs_error_pp','median'),
            n_independent_sample_condition_rows=('sample_id','size'),
            n_unique_sample_ids=('sample_id','nunique'),
            n_unique_batches=('batch_id','nunique'),
        )
        by_condition.to_csv(OUT_ROOT/'external_dp_reference_accuracy_by_condition.csv',index=False)
    log(f'External results: {OUT_ROOT.resolve()} / external_predictions_by_sample.csv')
    if known.empty:
        log('No independent DP-specific reference values supplied. External agreement/error cannot be validated numerically.')
        log('Any differences from bulk NMR are descriptive, NOT DP-specific model accuracy.')


# =============================================================================
# IDE entry point
# =============================================================================

def create_ide_external_manifest() -> Path:
    """Build an explicit manifest from the three filenames in the old scripts.

    Only independent sample files are listed. Optional independent DP-specific
    reference labels are copied from config; bulk values are context only.
    """
    ensure_dir(OUT_ROOT)
    rows=[]
    for sample_id, filename in EXTERNAL_FILES.items():
        p=Path(filename)
        if not p.is_absolute():
            p=DATA_DIR / p
        if not p.exists():
            if SKIP_MISSING_EXTERNAL_FILES:
                log(f"[SKIP] External file not found: {p}")
                continue
            raise FileNotFoundError(f"Missing external mzML: {p}")
        for dp in EXTERNAL_DPS:
            for nce in EXTERNAL_ENERGIES:
                rows.append({
                    'sample_id':sample_id,
                    'mzml_path':str(p.resolve()),
                    'dp':dp,
                    'nce':nce,
                    'dp_specific_nmr_M_fraction':DP_SPECIFIC_EXTERNAL_REFERENCES.get(
                        (sample_id,dp),''),
                    'bulk_nmr_M_fraction':BULK_EXTERNAL_REFERENCES.get(sample_id,''),
                    'batch_id':EXTERNAL_BATCH_ID,
                })
    if not rows:
        raise FileNotFoundError(
            'No external mzML files found. Check DATA_DIR/EXTERNAL_FILES or set '
            'EXTERNAL_MANIFEST_CSV to an existing CSV.'
        )
    out=OUT_ROOT / 'ide_external_manifest.csv'
    pd.DataFrame(rows).to_csv(out,index=False)
    log(f"External manifest saved: {out}")
    return out


def run_from_ide():
    """Run from IDE with no command-line arguments. Edit RUN_MODE above."""
    mode=str(RUN_MODE).strip().lower()
    if mode not in {'experiment','external','both'}:
        raise ValueError('RUN_MODE must be experiment, external, or both')
    log(f"RUN_MODE={mode}; DATA_DIR={DATA_DIR}; OUT_ROOT={OUT_ROOT}")
    if mode in {'experiment','both'}:
        # 105 composition-held-out folds; full-calibration models saved separately.
        main()
    if mode in {'external','both'}:
        if EXTERNAL_MANIFEST_CSV:
            manifest=Path(EXTERNAL_MANIFEST_CSV)
            if not manifest.is_absolute():
                manifest=SCRIPT_DIR / manifest
        else:
            manifest=create_ide_external_manifest()
        # Frozen models only. No changes to NMF, ratio choice or Ridge.
        external_predict(manifest, OUT_ROOT / 'frozen_models')


if __name__ == "__main__":
    run_from_ide()
