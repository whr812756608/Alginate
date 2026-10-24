#!/usr/bin/env python3
"""Synthetic acceptance for V7.0: NO real-world composition results produced."""
import sys as _repo_sys
from pathlib import Path as _repo_Path
_repo_root = _repo_Path(__file__).resolve().parents[1]
_repo_sys.path.insert(0, str(_repo_root))
_repo_sys.path.insert(0, str(_repo_root / "experiments"))

import sys
# The selftest flag allows the legacy mzML modules to import when pyteomics
# is unavailable. Real runs require pyteomics and real mzML files.
sys.argv=[sys.argv[0],'--run-mode','selftest']
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import alginate_V70_EVIDENCE_EXISTING_DATA_IDE as v70


def fake_pack(dp, nce, seed=321):
    rng=np.random.default_rng(seed+10*dp+int(nce))
    levels=np.asarray(v70.MIXES)
    n=18
    nom=np.repeat(levels,n)
    truth=np.interp(nom,levels,[.08,.20,.35,.46,.57,.70,.87])
    mz=np.asarray(v70.base.get_preprocess_centers(dp),float)
    m=len(mz)
    slope=np.linspace(-1.0,1.0,m)
    # Synthetic spectra are composition-linked but not based on real data.
    spec=np.exp((truth[:,None]-.46)*slope[None,:]*3.0+
                rng.normal(0,.022,size=(len(nom),m)))
    spec*=rng.uniform(.7,1.3,size=(len(nom),1))
    spec=np.maximum(spec,0).astype('float32')
    support=np.ones((len(nom),m),bool)
    return dict(dp=dp,nce=nce,grid=mz,centers=mz,
       masked={1.:spec},nom=nom,y=truth,sample=np.array([f'M{int(x*100)}' for x in nom]),
       support=support,delta=np.zeros_like(spec))


def run():
    v70.quick_selftest()
    with tempfile.TemporaryDirectory(prefix='alginate_v70_synthetic_') as t:
        v70.OUT_DIR=Path(t)
        v70.v66.SIGMAS=(1.,)
        perms=v70.permutations(2,seed=22)
        pk=fake_pack(3,26.)
        path=v70.OUT_DIR/'PERMUTATIONS'/'DP3_NCE26.csv'
        rows=v70.single_condition(3,26.,pk,perms,path)
        df=pd.DataFrame(rows)
        # The no-label-order NMF is mathematically a column permutation of the
        # original V6.6 raw coefficients: Ridge predictions must be invariant.
        X=v70.fixed_matrix(pk)
        train=~np.isclose(pk['nom'],.5)
        try:
            original_ratios=v70.v66.choose_reference_ratios(pk,train)
            original_fit=v70.v66.fit_models(pk,train,X,original_ratios,*v70.PREPROCESS)['K2_PLUS_K8']
            legacy_pred=float(original_fit.predictions(pk,X[~train])[0].mean())
            R,D,names,info=v70.v66.ratio_data(pk,X)
            features=v70.fold_feature_cache(pk,X,R,D,.5)
            neutral_pred,_,_,_=v70.fit_predict_group(pk['y'],pk['nom'],features,names,info,'K2_PLUS_K8')
            assert abs(legacy_pred-neutral_pred)<1e-5, (legacy_pred,neutral_pred)
            print('PASS V6.6 baseline prediction equivalence under label-neutral NMF ordering')
        except RuntimeError as exc:
            raise AssertionError(f'Synthetic baseline ratio selection failed: {exc}') from exc
        assert len(df)==3*2
        assert set(df.model)=={'K8_ONLY','K2_PLUS_K8'}
        assert df.groupby('model')['perm_id'].nunique().min()==3
        assert len(list((v70.OUT_DIR/'FOLDS').glob('*.csv')))==2
        # Resume must never add or alter a completed row.
        old=path.read_bytes()
        v70.single_condition(3,26.,pk,perms,path)
        assert path.read_bytes()==old
        # Verify publication summary is bounded to full grid.
        summ,_=v70.groupwise_summary(df,('K8_ONLY','K2_PLUS_K8'),2)
        assert len(summ)==2
        # Assembly/figures acceptance: replicate synthetic checkpoint SCHEMA, not
        # scientific measurements, for the complete 15-cell reporting path.
        for dp in v70.DPS:
            for nce in v70.NCES:
                dest=v70.OUT_DIR/'PERMUTATIONS'/f'DP{dp}_NCE{int(nce)}.csv'
                synthetic=df.copy();synthetic['dp']=dp;synthetic['nce']=nce
                v70.write_csv(synthetic,dest)
        nmr={'GG_block_rich':15.0,'MG_block_rich':50.0,'MM_block_rich':85.0}
        ext=[]
        for dp in (3,4):
            for energy in v70.NCES:
                for model in ('K2_PLUS_K8','K8_ONLY'):
                    for sample,val in nmr.items():
                        ext.append(dict(dp=dp,nce=energy,model=model,sample=sample,
                             pred_M_pct=val+(dp-3)*2+energy/10, n_scans=18,
                             nmf_ood_pct=12.,min_raw_denominator_support_pct=100.))
        v70.write_csv(ext,v70.OUT_DIR/'V70_EXTERNAL_FROZEN_PREDICTIONS.csv')
        v70.compile_reports(2)
        assert (v70.OUT_DIR/'V70_GLOBAL_PERMUTATION_TEST.csv').exists()
        assert len(pd.read_csv(v70.OUT_DIR/'V70_FIG3F_LOCKED_15_CONDITIONS.csv'))==15
        assert (v70.OUT_DIR/'V70_EXTERNAL_BULK_NMR_CONTEXT_ONLY.csv').exists()
        assert (v70.OUT_DIR/'V70_EXTERNAL_CROSS_NCE_QC.csv').exists()
        assert (v70.OUT_DIR/'FIGURES'/'V70_FIG3F_FIXED_MAE.png').exists()
        assert (v70.OUT_DIR/'FIGURES'/'V70_DP3_NCE26_BULK_CONTEXT_ONLY.png').exists()
        print('PASS synthetic all-15-cell report compilation, external context, four figure types')
        print('PASS synthetic full DP3/NCE26 permutation, both model arms, checkpoint resume, summary')
        print('Observed fixed-protocol synthetic MAE:',df[df.perm_id==0][['model','MAE_pp']].to_dict('records'))
        print('NOTE: These numbers are SYNTHETIC and have no evidential value for the manuscript.')

if __name__=='__main__':run()
