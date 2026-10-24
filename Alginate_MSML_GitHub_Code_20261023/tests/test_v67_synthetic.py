#!/usr/bin/env python3
"""Test V6.7 numerics and genuine grouped fold fitting on toy spectra (no mzML read).
This test uses a missing-pyteomics stub only for IMPORT. Mac runtime still needs
pyteomics to load the real dataset.
"""
import sys as _repo_sys
from pathlib import Path as _repo_Path
_repo_root = _repo_Path(__file__).resolve().parents[1]
_repo_sys.path.insert(0, str(_repo_root))
_repo_sys.path.insert(0, str(_repo_root / "experiments"))

import sys,types,tempfile
from pathlib import Path
try:
    import pyteomics.mzml
except ImportError:
    sys.modules['pyteomics']=types.ModuleType('pyteomics')
    sys.modules['pyteomics.mzml']=types.ModuleType('pyteomics.mzml')
import numpy as np
import pandas as pd
import alginate_V67_DP5_DP7_OVERNIGHT_IDE as v
from DP_spectrum_process_Class import FractionSpectra


def synthetic_pack(dp,nce):
    rng=np.random.default_rng(1000*dp+int(nce))
    grid=np.asarray(v.base.get_preprocess_centers(dp),float)
    sample=[];nom=[];y=[];data=[];support=[]
    for m in v.MIXES:
        truth=float(FractionSpectra(dp,m,dp,np.arange(2)).NMR_MG_ratio)
        for _ in range(8):
            q=np.arange(len(grid))
            signal=np.exp((.8*np.cos(q*.78)+.2*np.sin(q*.57))*(m-.5)*2.)
            signal*=(1.2+.05*q)*np.exp(rng.normal(0,.009,len(grid)))
            sample.append('M'+str(round(m*100)))
            nom.append(m);y.append(truth);data.append(signal);support.append(np.ones(len(grid),bool))
    X=np.asarray(data,np.float32)
    return dict(dp=dp,nce=float(nce),grid=grid,centers=grid,
        masked={s:X*(1+.01*s) for s in v.SIGMAS},
        sample=np.asarray(sample),nom=np.asarray(nom,float),y=np.asarray(y,float),
        support=np.asarray(support,bool))


def main():
    v.selftest()
    v.v66.SIGMAS=v.SIGMAS
    v.core.NMF_MAX_ITER=150
    data={(dp,e):synthetic_pack(dp,e) for dp in v.DPS for e in v.NCES}
    # All nine DP/NCE cells, three preprocessor conditions, matched K8 and joint.
    for (dp,e),pack in data.items():
        tr=~np.isclose(pack['nom'],.2);te=~tr
        chosen=v.ratios_for(pack,tr)
        assert len(chosen)==8
        for sigma,scheme in ((1.,'ORIGINAL7'),(.5,'EXTENDED7'),(.5,'GUARDED_EXTENDED7')):
            models=v.fit_and_score(pack,tr,te,sigma,scheme,chosen)
            assert set(models)==set(v.MODELS)
            assert all(np.isfinite(x['abs_error_pp']) for x in models.values())
    with tempfile.TemporaryDirectory(prefix='v67_synthetic_') as tmp:
        v.OUT_DIR=Path(tmp)
        pack=data[(7,26.)]
        # All 7 holdouts for the largest precursor, one setting (test-only).
        orig_sigma,orig_scheme=v.SIGMAS,v.SCHEMES
        v.SIGMAS=(1.,);v.SCHEMES=('ORIGINAL7',)
        try:
            v.run_grid({26.:pack},7,pilot=False,nces=(26.,))
        finally:
            v.SIGMAS=orig_sigma;v.SCHEMES=orig_scheme
        cp=list((v.OUT_DIR/'GRID_CHECKPOINTS').rglob('*.csv'))
        assert len(cp)==7
        # Two outer folds of full four-candidate *nested* selection.
        v.run_nested({22.:data[(5,22.)]},5,pilot=True,nces=(22.,))
        ne=list((v.OUT_DIR/'NESTED_CHECKPOINTS').rglob('*.csv'))
        assert len(ne)==2
        v.compile_report(packs_by_dp={7:{26.:pack}},scan_rows=[dict(
            dp=7,nce=26.,nominal_pct=0,acquired_target_scans=8,
            retained_scans=8,excluded_scans=0)],fixed=v.DEFAULT_FIXED)
        summary=pd.read_csv(v.OUT_DIR/'V67_GRID_BY_DP_NCE.csv')
        assert len(summary)==2 and (summary.n_holdouts==7).all()
        assert (v.OUT_DIR/'V67_SCAN_COUNTS.csv').exists()
        assert (v.OUT_DIR/'V67_NESTED_OUTER_ALL_FOLDS.csv').exists()
        assert (v.OUT_DIR/'V67_GROUP_MEMBERSHIP.csv').exists()
    print('PASS V6.7: nine DP/NCE cells, original/extended/guarded, K8 vs joint, seven-fold CV, nested selection, reports')

if __name__=='__main__':main()
