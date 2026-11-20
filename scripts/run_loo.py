#!/usr/bin/env python3
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))
import argparse, csv
import numpy as np
from alginate_ms2.publication_pipeline import load_calibration, evaluate_fold, configuration_sha256
from alginate_ms2.publication_math import NCES, COMPOSITIONS

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input-dir',type=Path,default=Path('outputs/preprocessed'))
    ap.add_argument('--out-dir',type=Path,default=Path('outputs/loo'))
    ap.add_argument('--dp',nargs='+',default=['all'])
    ap.add_argument('--heldout',nargs='+',default=['all'])
    ap.add_argument('--allow-nonconverged',action='store_true')
    a=ap.parse_args(argv)
    dps=range(3,8) if a.dp==['all'] else [int(v) for v in a.dp]
    helds=COMPOSITIONS if a.heldout==['all'] else [int(v) for v in a.heldout]
    if any(dp not in range(3,8) for dp in dps) or any(h not in COMPOSITIONS for h in helds):ap.error('Invalid DP or heldout')
    rows=[]
    for dp in dps:
        arrays=load_calibration(a.input_dir,dp)
        for held in helds:
            fold=evaluate_fold(arrays,dp,held)
            if len(fold)!=3:raise RuntimeError('Missing NCE condition')
            if not a.allow_nonconverged and any(not z['nmf_converged'] for z in fold):
                raise RuntimeError(f'DP{dp} heldout M{held}: NMF did not converge')
            rows.extend(fold)
            print(f'DP{dp} M{held:03d}: '+', '.join(f'NCE{z["nce"]}={z["absolute_error_pp"]:.3f}' for z in fold),flush=True)
    a.out_dir.mkdir(parents=True,exist_ok=True)
    fieldnames=list(dict.fromkeys(k for row in rows for k in row))
    with (a.out_dir/'loo_folds.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fieldnames);w.writeheader();w.writerows(rows)
    summary=[]
    for dp in dps:
        for nce in NCES:
            z=[r for r in rows if r['dp']==dp and r['nce']==nce]
            if len(z)!=len(helds):raise RuntimeError('Incomplete LOO')
            summary.append(dict(dp=dp,nce=nce,n_folds=len(z),loo_mae_pp=float(np.mean([r['absolute_error_pp'] for r in z])),
                n_nonconverged=sum(not r['nmf_converged'] for r in z)))
    with (a.out_dir/'loo_by_dp_nce.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    print(f'LOO COMPLETE: {len(rows)} fold rows. Configuration hash {configuration_sha256()}')
if __name__=='__main__':main()
