#!/usr/bin/env python3
"""Synthetic end-to-end smoke test; never a substitute for real mzML results."""
from __future__ import annotations
import sys as _repo_sys
from pathlib import Path as _repo_Path
_repo_root = _repo_Path(__file__).resolve().parents[1]
_repo_sys.path.insert(0, str(_repo_root))
_repo_sys.path.insert(0, str(_repo_root / "experiments"))

import tempfile
from pathlib import Path
import numpy as np
# For the synthetic test only: allow import when the optional mzML parser is absent.
# Real mzML processing requires installing the pinned requirements.
import sys as _sys
import types as _types
try:
    from pyteomics import mzml as _mzml  # noqa: F401
except ImportError:
    _pkg = _types.ModuleType('pyteomics'); _pkg.__path__=[]
    _mod = _types.ModuleType('pyteomics.mzml')
    def _no_mzml(*args, **kwargs):
        raise RuntimeError('Install pyteomics for real mzML processing')
    _mod.read=_no_mzml; _pkg.mzml=_mod
    _sys.modules['pyteomics']=_pkg
    _sys.modules['pyteomics.mzml']=_mod
import alginate_V68_DP3_NCE26_TRANSFER_IDE as v

def make_pack(external=False):
    rng=np.random.default_rng(31 if not external else 33)
    grid=np.arange(100.,551.,.2)
    candidates=v.base.get_ratio_candidate_centers(v.DP)
    samples=[];vals=[];ys=[];nom=[]
    groups=v.MIXES if not external else (.15,.55,.95)
    for m in groups:
        for scan in range(8):
            z=np.zeros(len(grid))
            for j,c in enumerate(candidates):
                amp=.5+(j%5)/5. + m*(.15+np.cos(j*.3)) + rng.normal(0,.012)
                z+=max(.05,amp)*np.exp(-.5*((grid-c)/.35)**2)
            vals.append(z.astype(np.float32));nom.append(m if not external else np.nan)
            ys.append(.08+.75*m if not external else np.nan)
            samples.append(f'cal_{int(m*100)}' if not external else
                           ('GG_block_rich','MG_block_rich','MM_block_rich')[groups.index(m)])
    X=np.stack(vals)
    pack=dict(dp=v.DP,nce=v.NCE,grid=grid,
              centers=np.asarray(v.base.get_preprocess_centers(v.DP)),
              nom=np.asarray(nom),y=np.asarray(ys),sample=np.asarray(samples),
              support=np.ones((len(X),len(v.base.get_preprocess_centers(v.DP))),bool),
              masked={.25:X,1.:X})
    return pack

def run():
    v.synthetic_test()
    cal=make_pack();ext=make_pack(external=True)
    train=~np.isclose(cal['nom'],v.MIXES[0]);X=v.v66.matrix(cal,.25,'ORIGINAL7',1.)
    ids,_=v.choose_ratios(cal,train,X,'REFERENCE','H000')
    # Exact numerical equivalence to the previous V6.6 baseline.
    old=v.v66.fit_models(cal,train,X,ids,.25,'ORIGINAL7',1.)['K2_PLUS_K8']
    new=v.fit_one(cal,train,X,'V66_SELECTED',.25,'ORIGINAL7',1.,
                  'REFERENCE','RAW','FULL','K2_PLUS_K8',ids)
    pred_old=old.predictions(cal,X)[0]
    pred_new=new.predict(cal,X)[0]
    np.testing.assert_allclose(pred_old,pred_new,rtol=1.e-6,atol=1.e-8)
    with tempfile.TemporaryDirectory(prefix='v68_synthetic_') as d:
        prior=v.OUT_DIR
        try:
            v.OUT_DIR=Path(d)
            subset=[a for a in v.arms() if a[0]=='V66_SELECTED' and a[4] in ('REFERENCE','RAW_SUPPORT')
                    and a[5]=='RAW' and a[7]=='K2_PLUS_K8' and a[6]=='FULL']
            assert len(subset)==2
            for hold in v.MIXES[:2]:v.train_condition(cal,None,hold,subset)
            v.train_condition(cal,ext,None,subset)
            v.compile_reports()
            cv=v.pd.read_csv(v.OUT_DIR/'V68_LORO_ALL.csv')
            ext_df=v.pd.read_csv(v.OUT_DIR/'V68_EXTERNAL_ALL.csv')
            assert len(cv)==2*2*4
            assert len(ext_df)==2*3*4
            assert set(ext_df['sample'])==set(v.NMR_BULK_CONTEXT)
            paths=list((v.OUT_DIR/'CHECKPOINTS').glob('H*/*.csv'))
            for hold in v.MIXES[:2]:v.train_condition(cal,None,hold,subset)
            assert len(paths)==len(list((v.OUT_DIR/'CHECKPOINTS').glob('H*/*.csv')))
            print('PASS: V6.6 matching model, checkpointing, six external sample-condition means, output reports')
        finally:v.OUT_DIR=prior

if __name__=='__main__':run()
