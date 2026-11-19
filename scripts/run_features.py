#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from alginate_ms2.publication_pipeline import load_calibration, select_k8, fit_pipeline
from alginate_ms2.publication_math import NCES, COMPOSITIONS


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=Path("outputs/preprocessed"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/features"))
    p.add_argument("--held-out", type=int, default=None,
                   help="Optional LOO composition; without this use all seven calibration groups")
    p.add_argument("--dp", nargs="+", default=["all"])
    p.add_argument("--nce", nargs="+", default=["all"])
    a = p.parse_args(argv)
    dps = range(3, 8) if a.dp == ["all"] else [int(x) for x in a.dp]
    nces = NCES if a.nce == ["all"] else tuple(int(x) for x in a.nce)
    if any(dp not in range(3, 8) for dp in dps) or any(n not in NCES for n in nces):
        p.error("Invalid DP or NCE")
    if a.held_out is not None and a.held_out not in COMPOSITIONS:
        p.error("Invalid held-out composition")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for dp in dps:
        calibration = load_calibration(a.input_dir, dp)
        selection = select_k8(calibration, dp, held=a.held_out)
        for nce in nces:
            model, features = fit_pipeline(calibration[nce], dp, nce, selection, held=a.held_out)
            if not model.converged:
                raise RuntimeError(f"DP{dp} NCE{nce}: NMF failed to converge")
            suffix = "full" if a.held_out is None else f"held_M{a.held_out:03d}"
            dest = a.out_dir / f"dp{dp}_nce{nce}_{suffix}_features.npz"
            np.savez_compressed(dest, **features)
            print(f"DP{dp} NCE{nce}: {features['features'].shape[0]} × 10 features -> {dest}", flush=True)
    print("FEATURE EXTRACTION COMPLETE")


if __name__ == "__main__":
    main()
