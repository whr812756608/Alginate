"""Calibration/external mzML input and precursor / NCE selection.

Only MS2 scans matching the requested precursor isolation target and NCE
are retained. Composition labels come from source filenames, never from
individual scan intensities.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from .config import load_config, selected_model


def scan_matches(scan, dp, nce, config=None):
    model = selected_model(dp, config)
    try:
        if int(scan.get("ms level", 0)) != 2:
            return False
        p = scan["precursorList"]["precursor"][0]
        target = float(p["isolationWindow"]["isolation window target m/z"])
        energy = float(p["activation"]["collision energy"])
    except (KeyError, IndexError, TypeError, ValueError, OverflowError):
        return False
    tol_mz = 1e-4 if int(dp) <= 4 else 1e-3
    tol_nce = 1e-5 if int(dp) <= 4 else 1e-4
    return bool(np.isfinite(target) and np.isfinite(energy)
                and abs(target - model["precursor_mz"]) < tol_mz
                and abs(energy - float(nce)) < tol_nce)


def calibration_paths(data_dir, dp, config=None):
    c = load_config() if config is None else config
    directory = Path(data_dir)
    return [(int(m), directory / f"Raw_F{dp}_M{m}G{100-m}.mzML")
            for m in c["nominal_m_pct"]]


def external_paths(data_dir, config=None):
    c = load_config() if config is None else config
    return [(name, Path(data_dir) / fname) for name, fname in c["external_names"].items()]


def read_matching_scans(data_dir, dp, nce, sample_set="calibration"):
    """Return records holding raw peak arrays and source labels.

    The raw mzML files are read directly for each requested DP/NCE.
    """
    if sample_set not in ("calibration", "external"):
        raise ValueError("sample_set must be calibration or external")
    if sample_set == "external" and int(dp) not in (3, 4):
        raise ValueError("External inference is available for DP3 and DP4")
    try:
        from pyteomics import mzml
    except ImportError as err:
        raise RuntimeError("Real mzML reading requires pyteomics and lxml. Run: python -m pip install -e '.[mzml]'") from err
    c = load_config()
    paths = calibration_paths(data_dir, dp, c) if sample_set == "calibration" else external_paths(data_dir, c)
    missing = [str(p) for _, p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing mzML files:\n" + "\n".join(missing))
    records = []
    for label, path in paths:
        n_match = 0
        with mzml.read(str(path)) as reader:
            for scan in reader:
                if not scan_matches(scan, dp, nce, c):
                    continue
                try:
                    mass = np.asarray(scan["m/z array"], dtype=float)
                    intensity = np.asarray(scan["intensity array"], dtype=float)
                except (KeyError, TypeError, ValueError):
                    continue
                records.append({"mz": mass, "intensity": intensity,
                                "label": label, "filename": path.name})
                n_match += 1
        print(f"{path.name}: {n_match} precursor/NCE-matched MS2 scans")
    return records
