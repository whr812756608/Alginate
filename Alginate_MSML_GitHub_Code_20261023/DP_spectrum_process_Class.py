#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import os
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from pyteomics import mzml
from scipy.signal import fftconvolve
from scipy.interpolate import CubicSpline
from sklearn.base import BaseEstimator, TransformerMixin

# -------------------------
# Ground truth & constants
# -------------------------
NMR_GT = {
    'f3d3': [0.08, 0.234, 0.388, 0.465, 0.542, 0.696, 0.85],
    'f4d4': [0.07, 0.222, 0.374, 0.450, 0.526, 0.678, 0.83],
    'f5d5': [0.10, 0.256, 0.412, 0.4900, 0.5680, 0.7240, 0.88],
    'f6d6': [0.10, 0.254, 0.408, 0.485, 0.562, 0.716, 0.87],
    'f7d7': [0.04, 0.194, 0.348, 0.4250, 0.5020, 0.656, 0.81],
}
RATIO_IDX = {0.0:0, 0.2:1, 0.4:2, 0.5:3, 0.6:4, 0.8:5, 1.0:6}
VALID_ENERGIES = {18.0, 22.0, 26.0}

DP_TARGET_MZ = {
    3: 545.099609375,
    4: 721.131591796875,
    5: 897.1636962890625,
    6: 1073.19580078125,
    7: 1249.2279052734375
}

# -------------------------
# Small helpers
# -------------------------
def parse_filename(filepath: str) -> Tuple[int, float]:
    # "Raw_F5_M20G80.mzML" -> (5, 0.2)
    name = os.path.basename(filepath)
    base, _ = os.path.splitext(name)
    parts = base.split('_')              # ["Raw","F5","M20G80"]
    fraction = int(parts[1][1:])
    ratio = float(parts[2].split('G')[0][1:]) / 100.0
    return fraction, ratio

def generate_global_mz_grid(min_mz=100, max_mz=1300, num_steps=60000) -> np.ndarray:
    return np.linspace(min_mz, max_mz, num_steps).astype(np.float32)

def filter_peaks_by_centers(
    mz_values: np.ndarray,
    intensity_values: np.ndarray,
    filter_centers: Optional[List[float]],
    window: float = 2.0,
    min_rel_height: float = 0.0,
) -> np.ndarray:
    if not filter_centers:
        return intensity_values
    centers = np.asarray(sorted(set(filter_centers)), dtype=np.float32)
    centers = centers[(centers >= mz_values[0] - window) & (centers <= mz_values[-1] + window)]
    if centers.size == 0:
        return np.zeros_like(intensity_values)

    left = centers[:, None] - window
    right = centers[:, None] + window
    mask_windows = (mz_values[None, :] >= left) & (mz_values[None, :] <= right)
    union_mask = mask_windows.any(axis=0)

    y = np.array(intensity_values, copy=True)
    if y.ndim == 1:
        y[~union_mask] = 0.0
        if min_rel_height > 0:
            m = float(np.max(y)) if y.size else 0.0
            if m > 0:
                y[y < (min_rel_height * m)] = 0.0
        return y

    y[~union_mask, :] = 0.0
    if min_rel_height > 0:
        m = np.max(y, axis=0, keepdims=True)
        thr = np.where(m > 0, min_rel_height * m, 0.0)
        y[y < thr] = 0.0
    return y

# -------------------------
# Data containers
# -------------------------
class ScanData:
    def __init__(
        self,
        mz_array: np.ndarray,
        intensity_array: np.ndarray,
        fraction: int,
        dp: int,
        ratio: float,
        global_mz_grid: np.ndarray,
        scan_start_time: Optional[float] = None,
        NMR_MG_ratio: Optional[float] = None,
        energy: Optional[float] = None,
        group_boundaries: Optional[List[Tuple[float,float]]] = None,
        processed: bool = False,
    ):
        self.raw_mz = mz_array.astype(np.float32, copy=False)
        self.raw_intensity = intensity_array.astype(np.float32, copy=False)
        self.fraction = fraction
        self.dp = dp
        self.ratio = ratio
        self.global_mz_grid = global_mz_grid
        self.scan_start_time = scan_start_time
        self.NMR_MG_ratio = NMR_MG_ratio
        self.energy = energy
        self.group_boundaries = group_boundaries
        self.mz: Optional[np.ndarray] = None
        self.intensity: Optional[np.ndarray] = None
        if processed:
            self.mz = global_mz_grid
            self.intensity = np.clip(self.raw_intensity, 0, None).astype(np.float32, copy=False)

    def filter_mz_intensity(self, mz_upper_limit: float) -> None:
        if self.raw_mz is None or self.raw_intensity is None:
            return
        idx = self.raw_mz <= mz_upper_limit
        self.raw_mz = self.raw_mz[idx]
        self.raw_intensity = self.raw_intensity[idx]

class FractionSpectra:
    def __init__(self, fraction:int, ratio:float, dp:int, global_mz_grid:np.ndarray,
                 reference_peak_mz:float=369.0, window_size:float=0.2):
        self.fraction = fraction
        self.ratio = ratio
        self.dp = dp
        self.global_mz_grid = global_mz_grid
        self.scans: Dict[float, List[ScanData]] = {18.0: [], 22.0: [], 26.0: []}
        self.NMR_MG_ratio: Optional[float] = None
        key = f"f{fraction}d{dp}"
        if key in NMR_GT and ratio in RATIO_IDX:
            self.NMR_MG_ratio = NMR_GT[key][RATIO_IDX[ratio]]
        self.reference_peak_mz = reference_peak_mz
        self.window_size = window_size

    @staticmethod
    def merge_shoulder_peaks(peak_mzs: np.ndarray, peak_intensities: np.ndarray, delta: float):
        if not peak_mzs.size:
            return peak_mzs, peak_intensities
        idx = np.argsort(peak_mzs)
        mzs = peak_mzs[idx]; ints = peak_intensities[idx]
        merged_mz = []; merged_int = []
        cur_mz = mzs[0]; cur_int = ints[0]
        for mz, it in zip(mzs[1:], ints[1:]):
            if abs(mz - cur_mz) <= delta:
                if it > cur_int:
                    cur_mz, cur_int = mz, it
            else:
                merged_mz.append(cur_mz); merged_int.append(cur_int)
                cur_mz, cur_int = mz, it
        merged_mz.append(cur_mz); merged_int.append(cur_int)
        return np.array(merged_mz, dtype=np.float32), np.array(merged_int, dtype=np.float32)

    @staticmethod
    def interpolate_to_grid_without_norm(raw_mz, raw_intensity, global_mz_grid, delta=1.2, gap_threshold=0.001):
        idx = np.argsort(raw_mz)
        mz = raw_mz[idx].astype(np.float32, copy=False)
        it = raw_intensity[idx].astype(np.float32, copy=False)
        mz, it = FractionSpectra.merge_shoulder_peaks(mz, it, delta=delta)

        max_it = np.max(it) if it.size else 0.0
        thr = gap_threshold * (max_it if max_it > 0 else 1.0)
        mask = it > thr
        peak_mzs = mz[mask]
        peak_ints = it[mask]

        cs = CubicSpline(mz, it, extrapolate=False)
        interp = cs(global_mz_grid).astype(np.float32)
        interp = np.nan_to_num(interp, nan=0.0).astype(np.float32, copy=False)

        valid = np.zeros_like(global_mz_grid, dtype=bool)
        for p in peak_mzs:
            L = np.searchsorted(global_mz_grid, p - 0.5)
            R = np.searchsorted(global_mz_grid, p + 0.5)
            valid[L:R] = True
        interp *= valid.astype(np.float32)

        raw_max = float(np.max(it)) if it.size else 0.0
        proc_max = float(np.max(interp)) if interp.size else 0.0
        if proc_max > 1e-6 and raw_max > 0:
            interp *= (raw_max / proc_max)
        return global_mz_grid, np.clip(interp, 0, None).astype(np.float32, copy=False), peak_mzs, peak_ints

    @staticmethod
    def gaussian_kernel(mz_values, sigma):
        mz_diff = float(np.mean(np.diff(mz_values)))
        ksize = max(3, int(3 * sigma / mz_diff))
        if ksize % 2 == 0:
            ksize += 1
        x = np.linspace(-ksize * mz_diff, ksize * mz_diff, ksize, dtype=np.float32)
        kern = np.exp(-0.5 * (x / sigma) ** 2).astype(np.float32)
        kern /= np.sum(kern)
        return kern.astype(np.float32, copy=False)

    @staticmethod
    def smooth_spectrum(mz_values, intensity_values, sigma: float = 1.0):
        kernel = FractionSpectra.gaussian_kernel(mz_values, sigma)
        out = fftconvolve(intensity_values, kernel, mode='same')
        return out.astype(np.float32, copy=False)

    @staticmethod
    def group_area_normalization(mz_values, intensity_values, group_boundaries: Optional[List[Tuple[float,float]]]):
        if group_boundaries is None:
            group_boundaries = [(100,200), (200,400), (400,600)]
        y = intensity_values.astype(np.float32, copy=False)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        s = np.sum(y, axis=0, keepdims=True)
        s[s <= 0] = 1.0
        y = y / s
        for (lo, hi) in group_boundaries:
            mask = (mz_values >= lo) & (mz_values <= hi)
            if not np.any(mask):
                continue
            area = np.sum(y[mask, :], axis=0, keepdims=True)
            area[area <= 0] = 1.0
            y[mask, :] = y[mask, :] / area
        return y.squeeze().astype(np.float32, copy=False)

    @staticmethod
    def normalize_spectrum(mz_values, intensity_values, peak_mzs, peak_intensities,
                           reference_peak_mz=369.0, window_size=1.0):
        y = intensity_values.astype(np.float32, copy=False)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if peak_mzs.size == 0:
            return y.squeeze()
        mask = np.abs(peak_mzs - reference_peak_mz) <= window_size
        if np.any(mask):
            ref_int = np.max(peak_intensities[mask])
        else:
            ref_int = peak_intensities[np.argmin(np.abs(peak_mzs - reference_peak_mz))]
        if ref_int > 0:
            y = y / ref_int
        return y.squeeze().astype(np.float32, copy=False)

# -------------------------
# Main processor class
# -------------------------
class DPSpectrumProcessor:
    """
    Unified loader/processor/aggregator for DP3–DP7.
    """
    def __init__(
        self,
        min_mz: float = 100.0,
        max_mz: float = 1300.0,
        num_steps: int = 60000,
        conv_sigma: float = 1.0,
        default_groups: Optional[List[Tuple[float,float]]] = None
    ):
        self.grid = generate_global_mz_grid(min_mz, max_mz, num_steps)
        self.conv_sigma = float(conv_sigma)
        self.default_groups = default_groups or [(100,200),(200,400),(400,600)]

        def smooth_with_sigma(mz_values, intensity_values, sigma=self.conv_sigma):
            return FractionSpectra.smooth_spectrum(mz_values, intensity_values, sigma=sigma)
        self._smooth_impl = smooth_with_sigma

    # helpers: zero single / multiple precursor regions
    @staticmethod
    def _zero_precursor_region(mz: np.ndarray, y: np.ndarray, dp: int, win: float):
        prec = float(DP_TARGET_MZ[dp])
        mask = (mz >= (prec - win)) & (mz <= (prec + win))
        y = np.array(y, copy=True)
        y[mask] = 0.0
        return y

    @staticmethod
    def _zero_precursor_regions(mz: np.ndarray, y: np.ndarray, centers: List[float], win: float):
        y = np.array(y, copy=True)
        for c in centers:
            mask = (mz >= (c - win)) & (mz <= (c + win))
            y[mask] = 0.0
        return y

    # ---------- public one-stop builder ----------
    def build_dataset(self,
                      filepaths: List[str],
                      dp: int,
                      energy: float,
                      num_bins: int = 120,
                      clip_range: Optional[Tuple[float, float]] = None,
                      use_group_normalization: bool = True,
                      group_boundaries: Optional[List[Tuple[float,float]]] = None,
                      filter_peaks: bool = True,
                      filter_centers: Optional[List[float]] = None,
                      peak_window: float = 2.0,
                      min_rel_height: float = 0.0,
                      outlier_filter: Optional[str] = None,
                      outlier_params: Optional[Dict[str, Any]] = None,
                      save_outliers_dir: Optional[str] = None,
                      zero_precursor: bool = False,
                      zero_window: float = 2.0,
                      # NEW:
                      zero_lower_precursors: bool = False
                      ):
        fs_dict = self.load_scans(
            filepaths=filepaths, dp=dp,
            use_group_normalization=use_group_normalization,
            group_boundaries=group_boundaries,
            filter_peaks=filter_peaks, filter_centers=filter_centers,
            peak_window=peak_window, min_rel_height=min_rel_height,
            zero_precursor=zero_precursor, zero_window=zero_window,
            zero_lower_precursors=zero_lower_precursors
        )
        agg_list = self.aggregate(
            fs_dict, num_bins=num_bins, energy_filter=energy,
            outlier_filter=outlier_filter, outlier_params=outlier_params,
            save_outliers_dir=save_outliers_dir
        )
        X = self._to_matrix(agg_list, clip_range=clip_range)
        y, groups, ratios = self._labels_and_groups_from_agg(agg_list)
        return X, y, ratios, agg_list

    # ---------- core steps ----------
    def load_scans(
        self,
        filepaths: List[str],
        dp: int,
        use_group_normalization: bool = True,
        group_boundaries: Optional[List[Tuple[float,float]]] = None,
        filter_peaks: bool = True,
        filter_centers: Optional[List[float]] = None,
        peak_window: float = 2.0,
        min_rel_height: float = 0.0,
        reference_peak_mz: float = 369.0,
        window_size: float = 1.0,
        zero_precursor: bool = False,
        zero_window: float = 2.0,
        # NEW:
        zero_lower_precursors: bool = False
    ) -> Dict[float, FractionSpectra]:
        if dp not in DP_TARGET_MZ:
            raise ValueError(f"Unsupported DP={dp}")
        groups = group_boundaries or self.default_groups
        out: Dict[float, FractionSpectra] = {}
        desired_mz = DP_TARGET_MZ[dp]
        mz_upper_limit = desired_mz + 20.0

        for fp in filepaths:
            fraction, ratio = parse_filename(fp)
            if ratio not in out:
                out[ratio] = FractionSpectra(fraction, ratio, dp, self.grid, reference_peak_mz, window_size)
            fs = out[ratio]
            with mzml.read(fp) as reader:
                for spec in reader:
                    if spec.get('ms level') != 2 or 'precursorList' not in spec:
                        continue
                    precursor = spec['precursorList']['precursor'][0]
                    iso = precursor.get('isolationWindow', {})
                    target_mz = iso.get('isolation window target m/z')
                    ce = precursor.get('activation', {}).get('collision energy')
                    if (ce in VALID_ENERGIES) and (target_mz is not None) and (abs(target_mz - desired_mz) < 1e-4):
                        rt = spec.get('scanList', {}).get('scan', [{}])[0].get('scan start time', None)
                        s = ScanData(spec['m/z array'], spec['intensity array'],
                                     fraction=fraction, dp=dp, ratio=ratio,
                                     global_mz_grid=self.grid, scan_start_time=rt,
                                     NMR_MG_ratio=fs.NMR_MG_ratio, energy=ce,
                                     group_boundaries=groups)
                        s.filter_mz_intensity(mz_upper_limit)

                        mz, interp, p_mzs, p_ints = FractionSpectra.interpolate_to_grid_without_norm(
                            s.raw_mz, s.raw_intensity, self.grid
                        )
                        if interp is None or len(interp) == 0:
                            continue
                        smoothed = self._smooth_impl(mz, interp)

                        # Zero current + lower-DP precursor windows BEFORE any normalization
                        if zero_precursor:
                            centers_to_zero = [float(DP_TARGET_MZ[dp])]
                            if zero_lower_precursors:
                                centers_to_zero += [float(DP_TARGET_MZ[d]) for d in DP_TARGET_MZ if d < dp]
                            smoothed = self._zero_precursor_regions(mz, smoothed, centers=centers_to_zero, win=zero_window)

                        # peak-window filtering (optional)
                        if filter_peaks and filter_centers is not None:
                            smoothed = filter_peaks_by_centers(
                                mz, smoothed, filter_centers,
                                window=peak_window, min_rel_height=min_rel_height
                            )

                        # normalization
                        if use_group_normalization:
                            smoothed = FractionSpectra.group_area_normalization(mz, smoothed, groups)
                        else:
                            smoothed = FractionSpectra.normalize_spectrum(
                                mz, smoothed, p_mzs, p_ints, reference_peak_mz, window_size
                            )

                        s.mz = mz.astype(np.float32, copy=False)
                        s.intensity = smoothed.astype(np.float32, copy=False)
                        s.raw_intensity = s.intensity
                        fs.scans[ce].append(s)
        return out

    def aggregate(
            self,
            fs_dict: Dict[float, FractionSpectra],
            num_bins: int = 120,
            energy_filter: Optional[float] = None,
            outlier_filter: Optional[str] = None,
            outlier_params: Optional[Dict[str, Any]] = None,
            save_outliers_dir: Optional[str] = None,
            SAVE_OUTLIER_FILES:  Optional[bool] = False,
    ) -> List[ScanData]:
        out: List[ScanData] = []
        self.last_agg_stats: Dict[Tuple[float, float], Dict[str, Any]] = {}

        def bin_index(rt: Optional[float], bin_size: float, n_bins: int) -> int:
            if rt is None:
                return -1
            b = int(rt / bin_size) if bin_size > 0 else 0
            if b >= n_bins: b = n_bins - 1
            return b if b >= 0 else -1

        for ratio, fs in fs_dict.items():
            for ce, scans in fs.scans.items():
                if (energy_filter is not None) and (ce != energy_filter):
                    continue

                key = (float(ratio), float(ce))
                n_raw = len(scans)

                scans_in: List[ScanData] = [s for s in scans if s.intensity is not None]
                rts = [float(s.scan_start_time) for s in scans_in if s.scan_start_time is not None]
                if len(rts) == 0:
                    self.last_agg_stats[key] = {
                        "n_raw": int(n_raw),
                        "n_outliers": 0,
                        "n_kept": int(n_raw),
                        "n_bins_nonempty": 0,
                        "bin_size_min": None,
                        "max_rt_min": None,
                    }
                    continue

                max_rt = float(np.max(rts))
                if num_bins <= 0:
                    raise ValueError("num_bins must be positive")
                bin_size = max_rt / float(num_bins) if max_rt > 0 else 0.0

                buckets: Dict[int, List[np.ndarray]] = defaultdict(list)
                meta: Dict[int, Dict[str, Any]] = {}
                for s in scans_in:
                    if s.scan_start_time is None or s.intensity is None:
                        continue
                    b = bin_index(float(s.scan_start_time), bin_size, num_bins)
                    if b < 0:
                        continue
                    buckets[b].append(s.intensity)
                    if b not in meta:
                        meta[b] = dict(fraction=s.fraction, dp=s.dp, ratio=ratio,
                                       NMR_MG_ratio=s.NMR_MG_ratio, ce=ce)

                n_bins_nonempty = 0
                for b, mats in buckets.items():
                    if not mats:
                        continue
                    n_bins_nonempty += 1
                    arr = np.vstack(mats).astype(np.float32, copy=False)
                    avg = np.mean(arr, axis=0).astype(np.float32, copy=False)
                    avg = np.clip(avg, 0, None)
                    md = meta[b]
                    out.append(ScanData(
                        mz_array=self.grid, intensity_array=avg,
                        fraction=md['fraction'], dp=md['dp'], ratio=md['ratio'],
                        global_mz_grid=self.grid, scan_start_time=b * bin_size,
                        NMR_MG_ratio=md['NMR_MG_ratio'], energy=md['ce'],
                        group_boundaries=None, processed=True
                    ))

                self.last_agg_stats[key] = {
                    "n_raw": int(n_raw),
                    "n_outliers": 0,
                    "n_kept": int(n_raw),
                    "n_bins_nonempty": int(n_bins_nonempty),
                    "bin_size_min": float(bin_size) if bin_size > 0 else 0.0,
                    "max_rt_min": float(max_rt),
                }

        return out

    def _labels_and_groups_from_agg(self, agg_list: List[ScanData]):
        ys, gs, rs = [], [], []
        for s in agg_list:
            key = f"f{s.fraction}d{s.dp}"
            if key not in NMR_GT:
                raise KeyError(f"Unknown key for NMR_GT: {key}")
            if s.ratio not in RATIO_IDX:
                raise KeyError(f"Unknown ratio for RATIO_IDX: {s.ratio}")
            ys.append(NMR_GT[key][RATIO_IDX[s.ratio]])
            gs.append(RATIO_IDX[s.ratio])
            rs.append(float(s.ratio))
        return (np.asarray(ys, dtype=np.float32),
                np.asarray(gs, dtype=int),
                np.asarray(rs, dtype=float))

    def _to_matrix(self, agg_list: List[ScanData], clip_range: Optional[Tuple[float, float]] = None):
        if clip_range is not None:
            lo, hi = clip_range
            idx = (self.grid >= lo) & (self.grid <= hi)
        else:
            idx = slice(None)
        X = np.vstack([s.raw_intensity[idx] for s in agg_list]).astype(np.float32, copy=False)
        return X

# -------------------------
# Lightweight feature helper
# -------------------------
class ProportionFeatures(BaseEstimator, TransformerMixin):
    """Take 2D W -> [W0, W1, pM] with row-normalized proportion."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        s = np.sum(X, axis=1, keepdims=True) + 1e-9
        pM = (X[:, [0]] / s)
        return np.hstack([X, pM])
