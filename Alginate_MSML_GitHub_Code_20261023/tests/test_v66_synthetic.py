#!/usr/bin/env python3
"""No real mzML required. Run python test_v66_synthetic.py."""
import sys as _repo_sys
from pathlib import Path as _repo_Path
_repo_root = _repo_Path(__file__).resolve().parents[1]
_repo_sys.path.insert(0, str(_repo_root))
_repo_sys.path.insert(0, str(_repo_root / "experiments"))

import sys
import types
import tempfile
from pathlib import Path

try:
    from pyteomics import mzml
except ImportError:
    pkg=types.ModuleType('pyteomics')
    pkg.mzml=types.ModuleType('pyteomics.mzml')
    sys.modules['pyteomics']=pkg
    sys.modules['pyteomics.mzml']=pkg.mzml

import numpy as np
import pandas as pd
import alginate_V66_ALL_NCE_OVERNIGHT_IDE as v
from DP_spectrum_process_Class import FractionSpectra


def fake_pack(dp, nce, external=False):
    rng=np.random.default_rng(dp*300+int(nce)+int(external))
    grid=np.array(v.base.get_preprocess_centers(dp),float)
    names=[];nom=[];y=[];values=[];support=[]
    mixes=v.MIXES if not external else (.10,.5,.9)
    for i,m in enumerate(mixes):
        label=(f'M{int(round(m*100))}' if not external else
               ('GG_block_rich','MG_block_rich','MM_block_rich')[i])
        true=float(FractionSpectra(dp,m,dp,np.arange(0,3,1.)).NMR_MG_ratio) if not external else np.nan
        for r in range(9):
            # Different spectral fragment responses to composition and energy.
            q=np.arange(len(grid))
            slope=0.2+0.03*q
            signal=(1.5+0.04*q)*np.exp(np.sin(q*.47)*(m-.5)*1.4)*(1+slope*m)
            signal*=np.exp(rng.normal(0,.012,len(grid)))
            names.append(label);nom.append(m if not external else np.nan)
            y.append(true);values.append(signal.astype('float32'))
            support.append(np.ones(len(grid),bool))
    B=np.asarray(values,'float32')
    return dict(dp=dp,nce=nce,grid=grid,centers=grid,
                masked={s:B*(1.0+.02*s) for s in v.SIGMAS},
                sample=np.asarray(names),nom=np.asarray(nom),y=np.asarray(y),
                support=np.asarray(support,bool))


def main():
    v.selftest()
    # Full toy end-to-end for DP3/4, three NCE, with 7 training composition folds.
    with tempfile.TemporaryDirectory(prefix='v66_synthetic_') as tmp:
        v.OUT_DIR=Path(tmp)
        v.OUT_DIR.mkdir(exist_ok=True)
        for dp in (3,4):
            for energy in v.NCES:
                p=fake_pack(dp,energy)
                e=fake_pack(dp,energy,external=True)
                train=~np.isclose(p['nom'],.2)
                test=~train
                ratios=v.choose_reference_ratios(p,train)
                assert len(ratios)==8
                for sigma,group,lam in ((1.,'ORIGINAL7',1.),(.25,'GALL',.5),(.0,'TIC_ONLY',0.)):
                    X=v.matrix(p,sigma,group,lam)
                    models=v.fit_models(p,train,X,ratios,sigma,group,lam)
                    assert set(models)==set(v.MODELS)
                    for name,fit in models.items():
                        row=v.summarize_predictions(p,fit,X[test],test,'held',p['y'][test].mean())
                        assert 0<=row['pred_M_pct']<=100 and np.isfinite(row['abs_error_pp'])
                        ex=v.matrix(e,sigma,group,lam)
                        r=v.summarize_predictions(e,fit,ex,np.ones(len(ex),bool),'synthetic')
                        assert 0<=r['pred_M_pct']<=100
                        assert name==fit.model
        # Exercise on-disk per-fold/per-external reporting and resume.
        p=fake_pack(3,22.)
        e=fake_pack(3,22.,True)
        v.run_cv(3,{22.:p},pilot=True,nces=(22.,))
        first=len(list((v.OUT_DIR/'FOLD_CHECKPOINTS').rglob('*.csv')))
        v.run_cv(3,{22.:p},pilot=True,nces=(22.,))
        assert len(list((v.OUT_DIR/'FOLD_CHECKPOINTS').rglob('*.csv')))==first
        v.run_external(3,{22.:p},{22.:e},pilot=True,nces=(22.,))
        v.compile_reports()
        cv=pd.read_csv(v.OUT_DIR/'V66_LORO_ALL_FOLDS.csv')
        ext=pd.read_csv(v.OUT_DIR/'V66_EXTERNAL_ALL_MODELS.csv')
        assert len(cv)==first*2 and len(ext)>0
        assert 'V66_EXTERNAL_ALL_MODELS.csv' in [p.name for p in v.OUT_DIR.iterdir()]
    # Cross-energy, seven-holdout reports with both DPs; trim the candidate grid
    # only in the TEST, so this dry run is fast without changing production.
    original_configurations = v.configurations
    with tempfile.TemporaryDirectory(prefix='v66_allnce_') as tmp2:
        v.OUT_DIR = Path(tmp2)
        subset=[(1.,'ORIGINAL7',1.),(.5,'GALL',.5),(.0,'TIC_ONLY',0.)]
        v.configurations = lambda dp: iter(subset)
        try:
            for dp in (3,4):
                cal={n:fake_pack(dp,n) for n in v.NCES}
                ext={n:fake_pack(dp,n,True) for n in v.NCES}
                v.run_cv(dp,cal,pilot=False,nces=v.NCES)
                v.run_external(dp,cal,ext,pilot=False,nces=v.NCES)
            v.compile_reports()
            winners=pd.read_csv(v.OUT_DIR/'V66_CALIBRATION_SELECTED_PER_DP_NCE.csv')
            shared=pd.read_csv(v.OUT_DIR/'V66_COMMON_NCE_CALIBRATION_SELECTION.csv')
            report=pd.read_csv(v.OUT_DIR/'V66_PRIMARY_COMPACT_ALL_NCE_TABLE.csv')
            assert len(winners)==6 and len(shared)==3 and len(report)==6
            assert all(winners.n_folds==7)
            assert all(shared.both_DPs==2)
            assert set(shared.nce)=={18.,22.,26.}
        finally:
            v.configurations=original_configurations
    print('PASS: 2 DPs × 3 NCEs × 7 holdouts, frozen K8, K2+K8, resume, external, common NCE and reporting')

if __name__=='__main__':main()
