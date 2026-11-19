#!/usr/bin/env python3
"""Run calibration preprocessing, feature generation, LOO, final fitting, and external inference."""
from pathlib import Path
import argparse
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True, type=Path,
                   help="Directory containing calibration and external mzML files")
    p.add_argument("--preprocessed-dir", type=Path, default=None,
                   help="Read Stage 1 inputs from this directory when reusing preprocessing")
    p.add_argument("--reuse-preprocessed", action="store_true",
                   help="Use existing 15 Stage 1 calibration NPZ files")
    p.add_argument("--skip-external", action="store_true",
                   help="Skip external DP3/DP4 NCE26 predictions")
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = p.parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    raw = args.data_dir.expanduser().resolve()
    processed = (args.preprocessed_dir.expanduser().resolve() if args.preprocessed_dir else output / "preprocessed")
    if args.preprocessed_dir and not args.reuse_preprocessed:
        p.error("--preprocessed-dir requires --reuse-preprocessed")
    stages = []
    if not args.reuse_preprocessed:
        stages.append(("run_preprocessing.py", ["--data-dir", str(raw), "--dp", "all", "--nce", "all",
                                               "--out-dir", str(processed)]))
    stages.extend([
        ("run_features.py", ["--input-dir", str(processed),
                             "--out-dir", str(output / "features")]),
        ("run_loo.py", ["--input-dir", str(processed),
                        "--out-dir", str(output / "loo")]),
        ("train_final_models.py", ["--input-dir", str(processed),
                                   "--out-dir", str(output / "models")]),
    ])
    if not args.skip_external:
        stages.append(("predict_external.py", ["--data-dir", str(raw),
                                               "--model-dir", str(output / "models"),
                                               "--out-csv", str(output / "external_dp3_dp4_nce26.csv")]))
    for script, cli_args in stages:
        command = [sys.executable, str(ROOT / "scripts" / script), *cli_args]
        print(f"\n=== {script} ===", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    print("\nPIPELINE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
