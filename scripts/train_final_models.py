#!/usr/bin/env python3
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))
"""Fit and serialize all 15 SI-fixed-ratio learned-NMF models on all 7 calibration compositions."""
import argparse, csv
from alginate_ms2.publication_pipeline import load_calibration,select_k8,fit_pipeline,save_pipeline
from alginate_ms2.publication_math import NCES

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input-dir',type=Path,default=Path('outputs/preprocessed'))
    ap.add_argument('--out-dir',type=Path,default=Path('outputs/models'))
    ap.add_argument('--allow-nonconverged',action='store_true')
    a=ap.parse_args(argv);a.out_dir.mkdir(parents=True,exist_ok=True)
    rows=[]
    for dp in range(3,8):
        arrays=load_calibration(a.input_dir,dp)
        sel=select_k8(arrays,dp,held=None)
        for nce in NCES:
            fit,metadata=fit_pipeline(arrays[nce],dp,nce,sel,held=None)
            if not fit.converged and not a.allow_nonconverged:raise RuntimeError(f'DP{dp} NCE{nce} NMF nonconverged')
            path=save_pipeline(fit,a.out_dir/f'dp{dp}_nce{nce}_model.npz')
            rows.append(dict(dp=dp,nce=nce,model_file=path.name,ratio_transform=fit.transform,
                             ratio_names=';'.join(fit.ratio_names),nmf_converged=fit.converged,
                             nmf_iterations=fit.nmf.n_iter,config_sha256=fit.config_hash))
            print('SAVED',path,flush=True)
    with (a.out_dir/'final_models_manifest.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(f'FINAL TRAINING PASS: {len(rows)}/15 models saved')
if __name__=='__main__':main()
