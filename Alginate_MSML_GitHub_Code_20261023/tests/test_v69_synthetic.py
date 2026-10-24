#!/usr/bin/env python3
"""Small no-mzML end-to-end test of the V6.9 checkpoints/CSVs/figures.

For test environments without pyteomics, this synthetic-only test temporarily
creates an in-memory reader stub. Real analysis requires real pyteomics.
"""
from __future__ import annotations
import sys as _repo_sys
from pathlib import Path as _repo_Path
_repo_root = _repo_Path(__file__).resolve().parents[1]
_repo_sys.path.insert(0, str(_repo_root))
_repo_sys.path.insert(0, str(_repo_root / "experiments"))

import argparse
import importlib
import os
import sys
import tempfile
import types
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from pyteomics import mzml  # noqa: F401
except ImportError:
    pkg=types.ModuleType('pyteomics');pkg.__path__=[]
    mod=types.ModuleType('pyteomics.mzml')
    def no_mzml(*a,**k):raise RuntimeError('pyteomics required for REAL spectra')
    mod.read=no_mzml
    sys.modules['pyteomics']=pkg;sys.modules['pyteomics.mzml']=mod
    pkg.mzml=mod

import alginate_V69_DP3_ALL_NCE_DIAGNOSTIC_IDE as v69


def fake_packs():
    rng=np.random.default_rng(42)
    grid=np.arange(100.,551.,.2)
    precent=np.asarray(v69.base.get_preprocess_centers(3),float)
    ratios=list(v69.base.get_ratio_candidate_centers(3))
    def traces(mixes, n, lab):
        XX=[];yy=[];nn=[];samples=[]
        for mi,nom in enumerate(mixes):
            for scan in range(n):
                x=np.zeros(len(grid),float)
                for k,center in enumerate(ratios):
                    amp=1.1+.5*np.cos(k*.3)+nom*(.6+.5*np.sin(k*.65))+rng.normal(0,.01)
                    x+=max(amp,.005)*np.exp(-.5*((grid-center)/.38)**2)
                x+=.03*np.exp(-.5*((grid-v69.base.DP_TARGET_MZ[3])/.5)**2)
                XX.append(x.astype(np.float32));yy.append(.08+.75*nom)
                nn.append(nom);samples.append(lab[mi])
        X=np.stack(XX)
        return dict(dp=3,nce=18.,grid=grid,centers=precent,
            sample=np.asarray(samples),nom=np.asarray(nn,float),y=np.asarray(yy,float),
            support=np.ones((len(X),len(precent)),bool),bad_scans=0,
            masked={s:X for s in v69.v66.SIGMAS})
    cal=traces(v69.MIXES,10,[f'cal_{i}' for i in range(7)])
    ext=traces((.12,.55,.9),10,v69.EXTERNAL_NAMES)
    ext['y']=np.full(len(ext['y']),np.nan)
    ext['nom']=np.full(len(ext['nom']),np.nan)
    return cal,ext


def run(all_nce=False):
    v69.synthetic_test()
    cal,ext=fake_packs()
    with tempfile.TemporaryDirectory(prefix='v69_synthetic_accept_') as temp:
        v69.OUT_DIR=Path(temp)/'results'
        # Standard quick test: full seven-fold NCE18 ORIGINAL_REFERENCE.
        # --all-nce: complete 48-checkpoint production-shaped grid.
        energies=v69.NCES if all_nce else (18.,)
        preprocessings=v69.PREPROCESS_IDS if all_nce else ('ORIGINAL_REFERENCE',)
        for en in energies:
            for pp in preprocessings:
                v69.run_one(cal,ext,en,pp)
        v69.export_retained_scan_counts({en:cal for en in energies},
                                        {en:ext for en in energies})
        v69.compile_results()
        cv=pd.read_csv(v69.OUT_DIR/'V69_LOCO_ABLATION_SUMMARY.csv')
        external=pd.read_csv(v69.OUT_DIR/'V69_EXTERNAL_PREDICTIONS_AND_SCAN_BOOTSTRAP.csv')
        shifts=pd.read_csv(v69.OUT_DIR/'V69_EXTERNAL_RATIO_DOMAIN_SHIFT.csv')
        sample_counts=pd.read_csv(v69.OUT_DIR/'V69_RETAINED_SCAN_COUNTS_BY_SAMPLE_NCE.csv')
        influence=pd.read_csv(v69.OUT_DIR/'V69_RATIO_ABLATION_INFLUENCE_SUMMARY.csv')
        scale=6 if all_nce else 1
        assert len(sample_counts)==10*len(energies)
        assert len(influence)==8*scale
        assert len(cv)==10*scale and cv.n_holdouts.eq(7).all(),cv.to_string()
        assert len(external)==3*10*scale,(len(external),external.head())
        assert len(shifts)==3*8*scale
        figs=list((v69.OUT_DIR/'FIGURES').glob('*.png'))
        assert len(figs)>=2*scale+1,figs
        # Strict checkpoint resume: second execution should not create new files.
        checkpoints=list((v69.OUT_DIR/'CHECKPOINTS').rglob('*.joblib'))
        assert len(checkpoints)==8*scale
        v69.run_one(cal,ext,18.,'ORIGINAL_REFERENCE')
        assert len(list((v69.OUT_DIR/'CHECKPOINTS').rglob('*.joblib')))==8*scale
        print(f'PASS: {len(energies)} NCE(s), {scale*8} checkpoints, 10 arms per condition, '
              'three external samples, ratio shift records, figures, resume')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--all-nce',action='store_true')
    args=parser.parse_args()
    run(all_nce=args.all_nce)
