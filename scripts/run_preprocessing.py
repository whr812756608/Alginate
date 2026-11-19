#!/usr/bin/env python3
"""Stage 1: create DP/NCE-specific processed-spectra .npz files."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

# Permit direct execution from a source checkout,
# even when the editable install belongs to a different Python interpreter.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if SOURCE_ROOT.is_dir() and str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np
from alginate_ms2.io import read_matching_scans, synthetic_records
from alginate_ms2.preprocessing import preprocess_records


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, help="Directory with original Raw_F*.mzML files")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/preprocessed"))
    p.add_argument("--dp", nargs="+", default=["3"], help="One or more of 3,4,5,6,7 or all")
    p.add_argument("--nce", nargs="+", default=["26"], help="One or more of 18,22,26 or all")
    p.add_argument("--sample-set", choices=("calibration", "external"), default="calibration")
    p.add_argument("--mock", action="store_true", help="Synthetic SOFTWARE TEST; NOT experimental data")
    args = p.parse_args(argv)
    dps = range(3,8) if args.dp == ["all"] else tuple(int(x) for x in args.dp)
    nces = (18,22,26) if args.nce == ["all"] else tuple(int(x) for x in args.nce)
    if any(x not in (3,4,5,6,7) for x in dps) or any(x not in (18,22,26) for x in nces):
        p.error("Use --dp 3..7 and --nce 18/22/26, or all")
    if not args.mock and args.data_dir is None:
        p.error("--data-dir is required unless --mock is used")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for dp in dps:
        for nce in nces:
            if args.mock:
                records = synthetic_records(dp,nce,args.sample_set)
            else:
                records = read_matching_scans(args.data_dir,dp,nce,args.sample_set)
            d = preprocess_records(records,dp,nce,args.sample_set,synthetic=args.mock)
            if args.sample_set == "calibration" and len(set(d["nominal_m_pct"])) != 7:
                raise RuntimeError(f"DP{dp} NCE{nce}: not all seven calibration compositions present")
            filename = args.out_dir / f"dp{dp}_nce{nce}_{args.sample_set}{'_MOCK' if args.mock else ''}.npz"
            np.savez_compressed(filename, **d)
            print(f"{filename}: {len(d['sample_id'])} spectra x {len(d['grid_mz'])} m/z points")
    if args.mock:
        print("SYNTHETIC ONLY — this proves code execution, not scientific accuracy.")

if __name__ == "__main__":
    main()
