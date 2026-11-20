#!/usr/bin/env python3
from pathlib import Path
import argparse
import csv
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from alginate_ms2.io import read_matching_scans
from alginate_ms2.preprocessing import preprocess_records
from alginate_ms2.publication_pipeline import load_pipeline


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True,
                   help="Directory containing the external GG/MG/MM mzML files")
    p.add_argument("--model-dir", type=Path, default=Path("outputs/models"))
    p.add_argument("--out-csv", type=Path,
                   default=Path("outputs/external_dp3_dp4_nce26.csv"))
    a = p.parse_args(argv)
    output = []
    for dp in (3, 4):
        nce = 26
        model_path = a.model_dir / f"dp{dp}_nce{nce}_model.npz"
        model = load_pipeline(model_path)
        records = read_matching_scans(a.data_dir, dp, nce, sample_set="external")
        data = preprocess_records(records, dp, nce, sample_set="external")
        pred, raw, _, _ = model.predict(data)
        labels = np.asarray(data["sample_id"], str)
        for label in ("GG", "MG", "MM"):
            ix = labels == label
            if not ix.any():
                raise ValueError(f"Missing external DP{dp} NCE26 sample: {label}")
            output.append(dict(
                dp=dp, nce=26, sample=label,
                external_predicted_M_pct=float(np.mean(pred[ix])),
                external_n_scans=int(ix.sum()),
                external_clipped_scans_pct=float(100*np.mean((raw[ix]<0)|(raw[ix]>100))),
                ratio_transform=model.transform,
                ratio_names=";".join(model.ratio_names),
            ))
            print(f"DP{dp} NCE26 {label}: {output[-1]['external_predicted_M_pct']:.4f}% M", flush=True)
    a.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with a.out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    assert len(output) == 6
    print(f"EXTERNAL INFERENCE COMPLETE: 6 samples at NCE26 -> {a.out_csv}")
    print("Bulk NMR values are DP-unresolved references, not DP-specific ground truth.")


if __name__ == "__main__":
    main()
