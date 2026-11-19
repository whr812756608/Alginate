"""DP-specific HCD-MS/MS spectral preprocessing.

DP3/4: cubic interpolation, local peak support, Gaussian convolution,
       ion-center windows, and TIC/group-area normalization.
DP5–7: 0.5-Da full-range bins, Gaussian smoothing, and capped
       per-scan group normalization or global TIC normalization.
"""
from __future__ import annotations
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.signal import fftconvolve
from .config import selected_model, load_config

EPS = 1e-12
CENTERED_GRID = np.linspace(100., 1300., 60000).astype(np.float32)
CENTERS_ALL = np.asarray((113.0244, 131.0350, 157.0142, 175.0248,
    193.0354, 291.0358, 309.0463, 351.0569, 369.0675, 467.0679,
    485.0784, 545.0996, 661.1105, 703.1211, 721.1316,
    837.1426, 879.1532, 897.1637, 1037.1747, 1055.1853,
    1073.1958, 1189.2068, 1231.2174, 1249.2279), float)
ORIGINAL_GROUPS = [(100., 200.), (300., 370.), (460., 530.),
                   (580., 710.), (830., 880.), (1030., 1060.), (1180., 1240.)]


def tic_normalize(X):
    X = np.maximum(np.asarray(X, dtype=np.float64), 0.)
    if X.ndim != 2:
        raise ValueError("Expected [n_spectra, n_mz_bins]")
    return X / np.maximum(X.sum(axis=1, keepdims=True), EPS)


def centered_group_bounds(group):
    bounds = [list(x) for x in ORIGINAL_GROUPS]
    if group in ("G291", "GALL"):
        bounds[1][0] = 290.
    if group in ("G545", "GALL"):
        bounds[2][1] = 550.
    if group in ("G721", "GALL"):
        bounds[3][1] = 730.
    if group not in ("ORIGINAL7", "G291", "G545", "G721", "GALL"):
        raise ValueError(f"Unknown centered-spectrum group {group}")
    return bounds


def centered_normalize(masked, mz, group, lam):
    """TIC/window-area weighting; the implemented blend is not a simple convex mixture."""
    if not 0 <= lam <= 1:
        raise ValueError("lambda must be in [0, 1]")
    T = tic_normalize(masked)
    if lam == 0:
        return T
    G = T.copy()
    for lo, hi in centered_group_bounds(group):
        keep = (mz >= lo) & (mz <= hi)
        if keep.any():
            areas = G[:, keep].sum(axis=1, keepdims=True)
            G[:, keep] /= np.where(areas > 0, areas, 1.)
    if lam == 1:
        return G
    total = np.maximum(G.sum(axis=1, keepdims=True), EPS)
    return np.maximum(((1-lam)*T + lam*G/total)*((1-lam)+lam*total), 0.)


def centered_kernel(sigma):
    if sigma <= 0:
        raise ValueError("Centered-spectrum Gaussian sigma must be positive")
    step = float(np.mean(np.diff(CENTERED_GRID)))
    k = max(3, int(3 * sigma / step))
    k += int(k % 2 == 0)
    x = np.linspace(-k*step, k*step, k, dtype=np.float32)
    weights = np.exp(-0.5*(x/sigma)**2)
    return weights / weights.sum()


def centered_grid_mask(dp):
    model = selected_model(dp)
    centers = CENTERS_ALL[CENTERS_ALL <= model["precursor_mz"] + .001]
    mz = CENTERED_GRID
    return ((mz >= 100.) & (mz <= model["precursor_mz"] + 5.) &
            np.any(abs(mz[:,None] - centers[None,:]) <= 2. + 1e-8, axis=1))


def centered_process_scan(mz, intensity, dp, sigmas, grid_mask=None):
    """Interpolate and smooth DP3/4 scans, rejecting invalid or empty inputs."""
    model = selected_model(dp)
    mz = np.asarray(mz, np.float32)
    intensity = np.asarray(intensity, np.float32)
    valid = np.isfinite(mz) & np.isfinite(intensity) & (mz >= 0) & (mz <= model["precursor_mz"] + 20)
    mz, intensity = mz[valid], intensity[valid]
    if len(mz) < 4 or np.max(intensity, initial=0) <= 0:
        return None
    order = np.argsort(mz)
    mz, intensity = mz[order], intensity[order]
    grouped_mass = [float(mz[0])]
    grouped_intensity = [float(intensity[0])]
    for m, v in zip(mz[1:], intensity[1:]):
        if float(m) - grouped_mass[-1] <= 1.2:
            if v > grouped_intensity[-1]:
                grouped_mass[-1], grouped_intensity[-1] = float(m), float(v)
        else:
            grouped_mass.append(float(m))
            grouped_intensity.append(float(v))
    if len(grouped_mass) < 4:
        return None
    mz = np.asarray(grouped_mass, np.float32)
    intensity = np.asarray(grouped_intensity, np.float32)
    try:
        x = CubicSpline(mz, intensity, extrapolate=False)(CENTERED_GRID).astype(np.float32)
    except (ValueError, TypeError):
        return None
    x = np.nan_to_num(x, nan=0., posinf=0., neginf=0.)
    signal = intensity > .001*max(float(intensity.max()), 1.)
    support = np.zeros(len(CENTERED_GRID), bool)
    for m in mz[signal]:
        support[np.searchsorted(CENTERED_GRID, m - .5):np.searchsorted(CENTERED_GRID, m + .5)] = True
    x *= support
    height = float(x.max())
    peak = float(intensity.max())
    if height > 1e-6 and peak > 0:
        x *= peak / height
    x = np.maximum(x, 0.)
    mask = centered_grid_mask(dp) if grid_mask is None else grid_mask
    output = {}
    for s in sigmas:
        if s == 0:
            smoothed = x
        else:
            smoothed = fftconvolve(x, centered_kernel(float(s)), mode="same").astype(np.float32)
        output[s] = np.maximum(smoothed[mask], 0.).astype(np.float32)
    return output


def fullrange_edges(dp, width=.5):
    precursor = selected_model(dp)["precursor_mz"]
    return np.arange(100., precursor-4.+width, width, dtype=float)


def fullrange_process_scan(mz, intensity, dp, width=.5):
    """Bin intensities in 0.5-Da intervals and normalize by total intensity."""
    edges = fullrange_edges(dp, width)
    mz = np.asarray(mz, float)
    intensity = np.asarray(intensity, float)
    valid = (np.isfinite(mz) & np.isfinite(intensity) & (mz >= edges[0]) &
             (mz < edges[-1]) & (intensity > 0))
    if int(valid.sum()) < 3:
        return None
    hist = np.histogram(mz[valid], bins=edges, weights=intensity[valid])[0]
    if not np.isfinite(hist.sum()) or hist.sum() <= 0:
        return None
    return (hist / hist.sum()).astype(np.float32)


def fullrange_smooth(X, sigma_da, width=.5):
    X = np.maximum(np.asarray(X, float), 0.)
    sigma_bins = float(sigma_da) / float(width)
    if sigma_bins > .01:
        X = np.maximum(gaussian_filter1d(X, sigma_bins, axis=1, mode="constant"), 0.)
    return tic_normalize(X)


def fullrange_normalize(X, mids, dp, mode, lam, cap=6., min_area=1e-5):
    """Apply global TIC or capped five-band per-scan normalization."""
    if mode == "GLOBAL":
        return tic_normalize(X)
    if mode != "EQ5_SCAN":
        raise ValueError(f"Unsupported selected full-range mode: {mode}")
    if not 0 <= lam <= 1:
        raise ValueError("lambda must be in [0, 1]")
    X = tic_normalize(X)
    lo, hi = 100., selected_model(dp)["precursor_mz"] - 4.
    edges = np.linspace(lo, hi, 6)
    labels = np.minimum(4, np.searchsorted(edges[1:-1], mids, side="right"))
    areas = np.column_stack([X[:,labels==b].sum(axis=1) for b in range(5)])
    target = 1. / 5.
    boost = np.clip(target/np.maximum(areas, min_area), 0., cap)
    boost[areas < min_area] = 0.
    Q = X*np.take(boost, labels, axis=1)
    Q = tic_normalize(Q)
    return (1-lam)*X + lam*Q


def preprocess_records(records, dp, nce, sample_set="calibration", synthetic=False):
    """Process preselected scan records into matrix arrays with metadata.

    This stage prepares spectra without fitting a decomposition or regression.
    """
    c = load_config()
    model = selected_model(dp, c)
    if not records:
        raise ValueError("No scans matched precursor/NCE filters")
    raw, labels, filenames = [], [], []
    if model["spectrum_layout"] == "centered_window":
        sigma_nmf = float(model["nmf_sigma_da"])
        sigma_ratio = float(model["ratio_sigma_da"])
        sigmas = sorted(set([sigma_nmf, sigma_ratio]))
        mask = centered_grid_mask(dp)
        mz_grid = CENTERED_GRID[mask]
        nmf_smoothed, ratio_smoothed = [], []
        for record in records:
            d = centered_process_scan(record["mz"], record["intensity"], dp, sigmas, mask)
            if d is None or d[sigma_ratio].sum() <= EPS or d[sigma_nmf].sum() <= EPS:
                continue
            nmf_smoothed.append(d[sigma_nmf])
            ratio_smoothed.append(d[sigma_ratio])
            labels.append(record["label"])
            filenames.append(record["filename"])
        if not labels:
            raise ValueError("All scans were rejected by centered-window preprocessing")
        X_nmf = centered_normalize(np.stack(nmf_smoothed), mz_grid,
                                 model["nmf_normalization"], float(model["nmf_lambda"]))
        X_ratio = centered_normalize(np.stack(ratio_smoothed), mz_grid,
                                   model["ratio_normalization"], float(model["ratio_lambda"]))
    else:
        width = float(model["bin_width_da"])
        edges = fullrange_edges(dp, width)
        mz_grid = (edges[:-1] + edges[1:])/2.
        for record in records:
            vec = fullrange_process_scan(record["mz"], record["intensity"], dp, width)
            if vec is None:
                continue
            raw.append(vec)
            labels.append(record["label"])
            filenames.append(record["filename"])
        if not labels:
            raise ValueError("All scans were rejected by full-range preprocessing")
        X_ratio = np.stack(raw).astype(np.float64)  # upstream TIC bins; exact K8 extraction comes later
        X_nmf = fullrange_normalize(fullrange_smooth(X_ratio, float(model["nmf_sigma_da"]), width),
                                    mz_grid, dp, model["nmf_normalization"], float(model["nmf_lambda"]))
    labels = np.asarray(labels)
    nmr = np.full(len(labels), np.nan, float)
    nominal = np.full(len(labels), -1, int)
    if sample_set == "calibration":
        nominal_set = c["nominal_m_pct"]
        known = set(nominal_set)
        if not set(map(int, labels)).issubset(known):
            raise ValueError("Unexpected calibration composition label")
        nominal = labels.astype(int)
        nmr = np.asarray([model["nmr_m_fraction"][nominal_set.index(int(x))]*100
                          for x in nominal], float)
    return {"grid_mz": np.asarray(mz_grid, float),
            "nmf_spectra": np.asarray(X_nmf, np.float32),
            "ratio_input_spectra": np.asarray(X_ratio, np.float32),
            "nominal_m_pct": nominal,
            "nmr_m_pct": nmr,
            "sample_id": labels.astype(str),
            "source_file": np.asarray(filenames, str),
            "dp": np.asarray(int(dp)), "nce": np.asarray(int(nce)),
            "synthetic": np.asarray(bool(synthetic)),
            "sample_set": np.asarray(sample_set)}
