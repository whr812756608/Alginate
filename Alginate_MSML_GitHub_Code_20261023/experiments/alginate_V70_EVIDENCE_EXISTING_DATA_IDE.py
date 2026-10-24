#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V7.0 | Stronger alginate MS-ML manuscript evidence using EXISTING data only.

Primary new experiment: composition-GROUP label permutation of a frozen, fixed-
preprocessing DP3-DP7 / NCE18,22,26 K2+K8 Ridge prediction pipeline.

* Seven nominal composition groups per condition (never individual scans).
* The permutation is identical across NCEs and DPs within each repetition,
  preserving dependencies due to shared prepared mixtures/fraction stocks.
* Within EACH held-out composition, K8 association-based selection and Ridge
  fitting are redone from its six training compositions under permuted labels.
* NMF is unsupervised, fit ONCE per training fold and reused across label
  permutations. Its orientation is forced to be label-independent.
* A predeclared training-label-mean fallback handles permutations for which
  fewer than eight reliable K8 features exist (same rule for actual labels).
* External samples (GG/MG/MM) never enter fitting, filtering, tuning or p-values.
  Their bulk NMR values are DP-UNRESOLVED and not DP-level accuracy labels.
* Fixed preprocessing sigma=1.0 Da, ORIGINAL7, lambda=1.0 for ALL conditions;
  the script does not mix DP-dependent model-development minima into Fig. 3f.
* All scans within each mixture are technical observations, not independent n.

Requires original mzML and exact legacy processing modules bundled alongside.
Run `--run-mode selftest` without original mzML; run `--run-mode pilot` for small
proof of wiring; run `--run-mode overnight` to generate manuscript evidence.
"""
from __future__ import annotations
# Repository layout adapter: allow direct execution from experiments/.
import sys as _repo_sys
from pathlib import Path as _repo_Path
_repo_sys.path.insert(0, str(_repo_Path(__file__).resolve().parents[1]))

import argparse
import hashlib
import itertools
import json
import os
import time
import warnings
from pathlib import Path

# Unit tests and report rebuilding need no mzML parser. Legacy modules import
# pyteomics eagerly, so allow ONLY these modes to import without the package.
import sys
if any(flag in sys.argv for flag in ('selftest', 'report')):
    try:
        import pyteomics  # noqa
    except ModuleNotFoundError:
        import types
        _stub=types.ModuleType('pyteomics')
        _stub.mzml=types.SimpleNamespace(read=lambda *_a,**_k: (_ for _ in ()).throw(
            ModuleNotFoundError('Install pyteomics to read real mzML files')))
        sys.modules['pyteomics']=_stub

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

import alginate_ratio_transfer_IDE as base
import alginate_V66_ALL_NCE_OVERNIGHT_IDE as v66
import v62_core as core

HERE = Path(__file__).resolve().parent
DATA_DIR = Path.home() / 'Desktop' / 'alginate_experiment'
OUT_DIR = HERE / 'V70_EVIDENCE_RESULTS'
DPS = (3,4,5,6,7)
NCES = (18.,22.,26.)
MIXES = (0.,.2,.4,.5,.6,.8,1.)
SEED = 20261020
VERSION = 'V70-existing-data-fixed-protocol-whole-group-permutations-r1'
PREPROCESS = (1.,'ORIGINAL7',1.)
RIDGE_ALPHA = 1.
NMF_MAX_ITER = 650
EPS = 1e-12
N_PERM_DEFAULT = 199
EXTERNAL_NAMES = ('GG_block_rich','MG_block_rich','MM_block_rich')


def log(s):
    print(f'[{time.strftime("%H:%M:%S")}] [V7.0] {s}',flush=True)


def safe_float(x):
    return float(x) if np.isfinite(x) else None


def digest(o):
    return hashlib.sha256(json.dumps(o,sort_keys=True,default=str).encode()).hexdigest()[:20]


def write_csv(rows,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    df=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    tmp=path.with_name(path.name+'.tmp')
    df.to_csv(tmp,index=False)
    os.replace(tmp,path)


def read_csv(path):
    return pd.read_csv(path) if Path(path).exists() else pd.DataFrame()


def source_fingerprints(dps,nces,include_external):
    fingerprint={}
    for dp in dps:
        for label,p in v66.mzml_sources(dp,False).items():
            if not p.exists():raise FileNotFoundError(f'Missing DP{dp} calibration mzML: {p}')
            s=p.stat();fingerprint[f'DP{dp}:{label}']=[str(p.resolve()),s.st_size,s.st_mtime_ns]
        if include_external and dp in (3,4):
            for label,p in v66.mzml_sources(dp,True).items():
                fingerprint[f'EXT:{label}']='PRESENT' if p.exists() else 'MISSING'
                if p.exists():
                    s=p.stat();fingerprint[f'EXT:{label}']=[str(p.resolve()),s.st_size,s.st_mtime_ns]
    return fingerprint


def permutations(n, seed=SEED):
    """Unique nonidentity 7-group label maps shared across every DP and NCE.

    A map assigns new NMR-calibrated mixture label to the EXISTING physical
    mixture identity, never changes a scan's physical group membership.
    """
    if n<1 or n>5039:raise ValueError('n_permutations must be 1..5039 (7!-1)')
    rng=np.random.default_rng(seed)
    if n>700:
        all_perms=np.array(list(itertools.permutations(range(7))),dtype=np.uint8)
        idx=np.flatnonzero(np.any(all_perms!=np.arange(7),axis=1))
        take=rng.choice(idx,size=n,replace=False)
        return all_perms[take]
    seen={tuple(range(7))}; out=[]
    while len(out)<n:
        p=tuple(int(v) for v in rng.permutation(7))
        if p not in seen:seen.add(p);out.append(p)
    return np.asarray(out,np.uint8)


def map_group_y(nom,nominal_levels,assigned_y, permutation=None):
    """Assigned labels may depend on DP via NMR stock compositions."""
    vals=assigned_y if permutation is None else assigned_y[np.asarray(permutation,int)]
    inds=np.searchsorted(nominal_levels,nom)
    if np.any(inds>=len(nominal_levels)) or not np.allclose(nom,nominal_levels[inds]):
        raise ValueError('Unrecognized nominal mixture in pack')
    return np.asarray(vals[inds],float)


def fixed_matrix(pack):
    return v66.matrix(pack,*PREPROCESS)


def fold_feature_cache(pack, X, ratios, denominators, held):
    tr=~np.isclose(pack['nom'],held)
    te=~tr
    # IMPORTANT: zero vector to suppress y-based NMF component ordering.
    # The *spectral* NMF fit has no knowledge of actual/permuted M% labels.
    nmf=core.fit_rank2(X,np.zeros_like(pack['y']),pack['nom'],tr,
                       weighted=False,raw_coeff=True)
    H,_=nmf.features_residual(X)
    return dict(train=tr,test=te,H=H,ratio=ratios,denom=denominators)


def chosen_ratios(R,D,y,nom,train,names,metadata):
    """Match V6.6 original association selection; no held-out labels.

    A full eight-ratio set is required. Under a random-label permutation some
    folds have fewer eligible ratios. The predefined fallback in fit_fold then
    predicts the training-label mean, which makes the entire procedure defined
    for EVERY possible permutation (rather than discarding inconvenient nulls).
    """
    ranked,_=base.rank_ratio_candidates(R[train],D[train],y[train],
                                        nom[train],names,metadata)
    selected=base.select_nonredundant_ratios(ranked,8)
    return tuple(int(a['idx']) for a in selected)


def fit_predict_group(y,nom,cache,names,metadata,model,selected_override=None):
    """Actual or permuted labels; all predictors restricted to train scans."""
    train=cache['train']; test=cache['test'];R=cache['ratio'];D=cache['denom'];H=cache['H']
    ratios=(chosen_ratios(R,D,y,nom,train,names,metadata)
            if selected_override is None else tuple(selected_override))
    if len(ratios)<8:
        # Deterministic, conservative-to-model-capacity fallback defined for
        # observed AND permuted labels. Never quietly skip hard null cases.
        group_values=np.unique(nom[train]);value=np.mean([y[train&np.isclose(nom,g)][0] for g in group_values])
        return float(value),True,len(ratios),None
    design=R[:,list(ratios)]
    if model=='K2_PLUS_K8':design=np.column_stack((H,design))
    elif model!='K8_ONLY':raise ValueError(model)
    weights=core.group_weights(nom[train])
    scaler=StandardScaler()
    try:scaler.fit(design[train],sample_weight=weights)
    except TypeError:scaler.fit(design[train])
    reg=Ridge(alpha=RIDGE_ALPHA).fit(scaler.transform(design[train]),y[train],sample_weight=weights)
    pred=np.clip(reg.predict(scaler.transform(design[test])),0.,1.)
    return float(np.mean(pred)),False,len(ratios),tuple(ratios)


def single_condition(dp,e,pack,perm_matrix,output_file,include_k8=True):
    """Run all permutations for one DP×NCE, periodically checkpoint as CSV."""
    sample_names=list(sorted(np.unique(pack['nom'])))
    if len(sample_names)!=7 or not np.allclose(sample_names,MIXES):
        raise ValueError(f'DP{dp}/NCE{e} does not contain exact 7 compositions')
    levels=np.array(sample_names)
    assigned_y=np.array([float(np.median(pack['y'][np.isclose(pack['nom'],g)])) for g in levels])
    if np.ptp(assigned_y)<=1e-8:raise ValueError('Invalid reference labels')
    X=fixed_matrix(pack)
    R,D,names,info=v66.ratio_data(pack,X)
    models=('K2_PLUS_K8','K8_ONLY') if include_k8 else ('K2_PLUS_K8',)
    existing=read_csv(output_file)
    completed=set(zip(existing['perm_id'].astype(int),existing['model'].astype(str))) if not existing.empty else set()
    rows=existing.to_dict('records') if not existing.empty else []
    total=len(perm_matrix)+1
    folds=[fold_feature_cache(pack,X,R,D,held) for held in levels]
    for perm_id in range(total):
        perm=None if perm_id==0 else perm_matrix[perm_id-1]
        if all((perm_id,m) in completed for m in models):continue
        y=map_group_y(pack['nom'],levels,assigned_y,perm)
        # Ratio identities depend on labels, but not on the regression arm.
        # Select ONCE per fold/permutation for both K8 and K2+K8.
        fold_ratios=[chosen_ratios(R,D,y,pack['nom'],cache['train'],names,info)
                     for cache in folds]
        for model in models:
            if (perm_id,model) in completed:continue
            errs=[];fbacks=[];ratios=[];preds=[]
            for held,cache,selected_identity in zip(levels,folds,fold_ratios):
                mean_pred,fallback,n_selected,selected=fit_predict_group(
                    y,pack['nom'],cache,names,info,model,
                    selected_override=selected_identity)
                true=float(y[cache['test']][0]);error=abs(mean_pred-true)*100
                errs.append(error);fbacks.append(int(fallback));ratios.append(n_selected)
                if perm_id==0:
                    preds.append((held,true*100,mean_pred*100,error,n_selected,fallback,selected,
                                  int(cache['test'].sum())))
            rows.append(dict(dp=dp,nce=e,perm_id=perm_id,model=model,
                 MAE_pp=float(np.mean(errs)),max_error_pp=float(np.max(errs)),
                 fallback_folds=int(np.sum(fbacks)),min_ratio_count=int(min(ratios)),
                 n_holdout_compositions=7,
                 seed=SEED,permutation='identity' if perm_id==0 else '-'.join(map(str,perm)),
                 label_permutation_group_level=True))
            if perm_id==0:
                write_csv([dict(dp=dp,nce=e,model=model,
                     heldout_nominal_pct=int(round(h*100)),true_M_pct=t,pred_M_pct=p,
                     abs_error_pp=a,n_scans=nscans,
                     selected_ratio_indices=';'.join(map(str,sel or [])),
                     training_fallback_used=bool(f),n_ratios=nrat)
                     for h,t,p,a,nrat,f,sel,nscans in preds],
                     OUT_DIR/'FOLDS'/f'V70_DP{dp}_NCE{int(e)}_{model}_LOCO.csv')
        if perm_id%10==0 or perm_id==total-1:
            write_csv(rows,output_file)
            log(f'DP{dp} / NCE{int(e)} complete permutation {perm_id}/{total-1}')
    return rows


def load_calibration(dp):
    packs=v66.cache_dp(dp,False)
    for e in NCES:
        if e not in packs:raise ValueError(f'DP{dp}/NCE{e} not present in mzML')
    return packs


def external_inference(dp,packs):
    """Original-label full-fit models only. No external truth assumed."""
    ext=v66.cache_dp(dp,True)
    if not ext:return []
    output=[]
    for e in NCES:
        if e not in ext:continue
        cal=packs[e]; external=ext[e]
        xc=fixed_matrix(cal); xe=fixed_matrix(external)
        if not np.array_equal(cal['grid'],external['grid']):raise ValueError('Different mz grids')
        train=np.ones(len(cal['nom']),bool)
        # No label permutation; same ratio selection as original V6.6 baseline.
        ratios=v66.choose_reference_ratios(cal,train)
        fits=v66.fit_models(cal,train,xc,ratios,*PREPROCESS)
        for model,fit in fits.items():
            for name in EXTERNAL_NAMES:
                mask=external['sample']==name
                if not mask.any():continue
                out=v66.summarize_predictions(external,fit,xe[mask],mask,name,truth=None)
                out['model']=model;out['analysis']='Frozen original model; NO DP-level external truth'
                out['selected_ratio_indices']=';'.join(map(str,ratios))
                output.append(out)
        log(f'DP{dp}/NCE{int(e)} external predictions complete')
    return output


def groupwise_summary(permutations_df,models,expected_perms):
    obs=permutations_df[permutations_df.perm_id.eq(0)]
    null=permutations_df[permutations_df.perm_id.gt(0)]
    if obs.empty:return pd.DataFrame(),pd.DataFrame()
    group=[]
    for (dp,e,m),part in obs.groupby(['dp','nce','model']):
        arr=null[(null.dp==dp)&(np.isclose(null.nce,e))&(null.model==m)]
        if len(arr)!=expected_perms:
            raise RuntimeError(f'Incomplete null DP{dp}/NCE{e}/{m}: {len(arr)}/{expected_perms}')
        v=float(part.iloc[0].MAE_pp)
        nd=arr['MAE_pp'].to_numpy(float)
        count=int(np.sum(nd<=v+1e-12))
        group.append(dict(dp=int(dp),nce=float(e),model=m,
             observed_MAE_pp=v,permutation_p_one_sided=(count+1)/(len(nd)+1),
             null_median_MAE_pp=float(np.median(nd)),null_p05_MAE_pp=float(np.percentile(nd,5)),
             null_p95_MAE_pp=float(np.percentile(nd,95)),
             null_fallback_rate=float(arr['fallback_folds'].sum()/(len(arr)*7)),
             observed_fallback_folds=int(part.iloc[0].fallback_folds),
             null_permutations=len(nd),validation_unit='composition; n=7'))
    return pd.DataFrame(group).sort_values(['dp','nce','model']),obs


def omnibus_test(all_df, n, models):
    """Shared permutation ID produces a paired 15-condition macro-average null.

    This is an across-condition summary, not 15 independent biological samples.
    """
    om=[]
    for model in models:
        a=all_df[all_df.model==model]
        values=a.groupby('perm_id').agg(MAE_pp=('MAE_pp','mean'),n_conditions=('MAE_pp','count'))
        expected=len(DPS)*len(NCES)
        if not np.all(values['n_conditions']==expected):
            raise RuntimeError('Omnibus requires a FULL 5-DP x 3-NCE set and all permutations')
        obs=float(values.loc[0,'MAE_pp'])
        null=values.loc[values.index!=0,'MAE_pp'].to_numpy()
        p=(1+np.sum(null<=obs+1e-12))/(len(null)+1)
        om.append(dict(model=model,observed_macro_MAE_pp=obs,
             null_median_macro_MAE_pp=float(np.median(null)),
             null_5th_pct=float(np.percentile(null,5)),
             permutation_p_one_sided=float(p),permutations=int(len(null)),
             n_conditions=expected,n_composition_predictions=expected*7,
             independent_material_caveat='Seven compositions share source fractions; conditions are correlated'))
    return pd.DataFrame(om)


def external_nce_consistency(df):
    out=[]
    if df.empty:return pd.DataFrame()
    for (dp,model,sample),sub in df.groupby(['dp','model','sample']):
        sub=sub.sort_values('nce')
        if len(sub)<2:continue
        vals=sub['pred_M_pct'].to_numpy(float)
        out.append(dict(dp=dp,model=model,sample=sample,
          energies=';'.join(str(int(v)) for v in sub.nce),
          predictions_M_pct=';'.join(f'{v:.2f}' for v in vals),
          cross_NCE_range_pp=float(np.ptp(vals)),
          cross_NCE_SD_pp=float(np.std(vals,ddof=1)),
          max_NMF_OOD_pct=float(np.nanmax(sub.nmf_ood_pct.to_numpy(float))) if model=='K2_PLUS_K8' else np.nan,
          min_raw_denominator_support_pct=float(sub.min_raw_denominator_support_pct.min()),
          note='Cross-NCE stability, NOT DP-specific external accuracy'))
    return pd.DataFrame(out)


def calibration_fold_audit():
    """Endpoint versus interior composition prediction without additional fitting."""
    parts=[read_csv(p) for p in sorted((OUT_DIR/'FOLDS').glob('V70_DP*_LOCO.csv'))]
    df=pd.concat([p for p in parts if not p.empty],ignore_index=True) if parts else pd.DataFrame()
    if df.empty:return pd.DataFrame(),pd.DataFrame()
    df['holdout_type']=np.where(df.heldout_nominal_pct.isin([0,100]),
                                'Endpoint extrapolation','Interior composition')
    summary=df.groupby(['dp','nce','model','holdout_type'],as_index=False).agg(
        MAE_pp=('abs_error_pp','mean'),
        worst_error_pp=('abs_error_pp','max'),
        n_compositions=('abs_error_pp','count'))
    write_csv(df,OUT_DIR/'V70_COMPOSITION_LEVEL_OOF_PREDICTIONS.csv')
    write_csv(summary,OUT_DIR/'V70_ENDPOINT_VS_INTERIOR_ERRORS.csv')
    return df,summary


def external_bulk_context(external):
    """Bulk NMR numbers are descriptive ONLY and not fitted/selected."""
    if external.empty:return pd.DataFrame(),pd.DataFrame()
    bulk={'GG_block_rich':12.,'MG_block_rich':51.,'MM_block_rich':82.}
    df=external.copy()
    df['bulk_1H_NMR_M_pct_DP_UNRESOLVED']=df['sample'].map(bulk)
    df['DP_prediction_minus_bulk_pp_DESCRIPTIVE']=df['pred_M_pct']-df['bulk_1H_NMR_M_pct_DP_UNRESOLVED']
    # Do not label this as external MAE; DP-specific reference truth is unknown.
    g=df.groupby(['dp','nce','model'],as_index=False).agg(
       mean_absolute_bulk_discrepancy_DESCRIPTIVE_pp=('DP_prediction_minus_bulk_pp_DESCRIPTIVE',lambda x:float(np.mean(np.abs(x)))),
       n_external_materials=('sample','nunique'))
    df['warning']='Whole-material 1H NMR is NOT matched DP-specific truth'
    g['warning']='Descriptive comparison; NOT external validation MAE'
    write_csv(df,OUT_DIR/'V70_EXTERNAL_BULK_NMR_CONTEXT_ONLY.csv')
    write_csv(g,OUT_DIR/'V70_EXTERNAL_BULK_DISCREPANCY_CONTEXT_ONLY.csv')
    return df,g


def external_ordering(external):
    rows=[]
    if external.empty:return pd.DataFrame()
    for (dp,nce,model),sub in external.groupby(['dp','nce','model']):
        lookup=dict(zip(sub['sample'],sub['pred_M_pct']))
        if all(name in lookup for name in EXTERNAL_NAMES):
            gg,mg,mm=(float(lookup[name]) for name in EXTERNAL_NAMES)
            rows.append(dict(dp=dp,nce=nce,model=model,
                gg_prediction_M_pct=gg,mg_prediction_M_pct=mg,mm_prediction_M_pct=mm,
                gg_less_than_mg_less_than_mm=bool(gg<mg<mm),
                warning='Expected ordering is a plausibility diagnostic, not independent accuracy'))
    out=pd.DataFrame(rows)
    if not out.empty:write_csv(out,OUT_DIR/'V70_EXTERNAL_ORDER_DIAGNOSTIC_ONLY.csv')
    return out


def create_figures(summary,external):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=OUT_DIR/'FIGURES';figures.mkdir(exist_ok=True,parents=True)
    x=summary[summary.model=='K2_PLUS_K8']
    if len(x):
        mat=x.pivot(index='dp',columns='nce',values='observed_MAE_pp').reindex(index=DPS,columns=NCES)
        fig,ax=plt.subplots(figsize=(5.6,4.3))
        # Palette remains a visual encoding of measured numerical error.
        im=ax.imshow(mat.to_numpy(float),cmap='Greens',vmin=0)
        for i,d in enumerate(DPS):
            for j,e in enumerate(NCES):
                v=mat.loc[d,e]
                if np.isfinite(v):ax.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=9)
        ax.set(xticks=range(3),xticklabels=[f'{int(e)}%' for e in NCES],
               yticks=range(5),yticklabels=[f'DP{d}' for d in DPS],
               xlabel='Collision energy (NCE)',ylabel='Oligosaccharide degree',
               title='Fixed-protocol LOCO MAE (M percentage points)')
        fig.colorbar(im,ax=ax,label='MAE (percentage points)')
        fig.tight_layout();fig.savefig(figures/'V70_FIG3F_FIXED_MAE.png',dpi=210);plt.close(fig)
        fig,ax=plt.subplots(figsize=(5.6,4.2))
        x2=x.sort_values(['dp','nce'])
        labels=[f'{int(d)}/{int(e)}' for d,e in zip(x2.dp,x2.nce)]
        xx=np.arange(len(labels));ax.bar(xx,x2['observed_MAE_pp'],label='Observed')
        ax.plot(xx,x2['null_median_MAE_pp'],marker='o',linestyle='none',label='Permuted-label median')
        ax.set_xticks(xx,labels,rotation=60,ha='right')
        ax.set(xlabel='DP / NCE',ylabel='MAE (M percentage points)',
               title='Observed versus whole-composition label permutations')
        ax.legend();fig.tight_layout();fig.savefig(figures/'V70_LABEL_PERMUTATION_COMPARISON.png',dpi=210);plt.close(fig)
    if not external.empty:
        data=external[external.model=='K2_PLUS_K8']
        for dp in (3,4):
            sub=data[data.dp==dp]
            if sub.empty:continue
            fig,ax=plt.subplots(figsize=(6.5,4.5))
            for name in EXTERNAL_NAMES:
                part=sub[sub['sample']==name].sort_values('nce')
                if part.empty:continue
                ax.plot(part['nce'],part['pred_M_pct'],marker='o',label=name.replace('_block_rich','-rich'))
            ax.set(xlabel='NCE (%)',ylabel='Predicted DP-specific M (%)',ylim=(0,100),
                   title=f'DP{dp}: frozen-model external predictions across NCE')
            ax.legend();fig.tight_layout();fig.savefig(figures/f'V70_DP{dp}_EXTERNAL_CROSS_NCE.png',dpi=210);plt.close(fig)
            # Context only: display NCE26 predictions beside DP-unresolved bulk NMR.
            n26=sub[np.isclose(sub['nce'],26.)]
            if len(n26)==3:
                truth={'GG_block_rich':12.,'MG_block_rich':51.,'MM_block_rich':82.}
                names=list(EXTERNAL_NAMES)
                pred=[float(n26[n26['sample']==name].iloc[0]['pred_M_pct']) for name in names]
                target=[truth[name] for name in names]
                xx=np.arange(3);fig2,ax2=plt.subplots(figsize=(5.6,3.8))
                ax2.bar(xx-.18,target,width=.35,label='Bulk NMR (all DPs)')
                ax2.bar(xx+.18,pred,width=.35,label=f'DP{dp} MS–ML estimate')
                ax2.set(xticks=xx,xticklabels=['GG-rich','MG-rich','MM-rich'],
                        ylim=(0,105),ylabel='M fraction (%)',
                        title='Descriptive bulk NMR context, NOT DP-specific accuracy')
                ax2.legend();fig2.tight_layout()
                fig2.savefig(figures/f'V70_DP{dp}_NCE26_BULK_CONTEXT_ONLY.png',dpi=210)
                plt.close(fig2)


def compile_reports(n_perms):
    pieces=[read_csv(p) for p in sorted((OUT_DIR/'PERMUTATIONS').glob('DP*_NCE*.csv'))]
    all_df=pd.concat([p for p in pieces if not p.empty],ignore_index=True) if pieces else pd.DataFrame()
    if all_df.empty:return
    write_csv(all_df,OUT_DIR/'V70_ALL_PERMUTATION_RESULTS.csv')
    summary,observed=groupwise_summary(all_df,('K2_PLUS_K8','K8_ONLY'),n_perms)
    write_csv(summary,OUT_DIR/'V70_PERMUTATION_PVALUES_BY_DP_NCE.csv')
    write_csv(observed,OUT_DIR/'V70_FIXED_PROTOCOL_CONDITION_SCORES.csv')
    joint=summary[summary.model=='K2_PLUS_K8']
    if len(joint)==15:
        write_csv(joint[['dp','nce','observed_MAE_pp','permutation_p_one_sided','null_median_MAE_pp']],
                  OUT_DIR/'V70_FIG3F_LOCKED_15_CONDITIONS.csv')
        omni=omnibus_test(all_df,n_perms,('K2_PLUS_K8','K8_ONLY'))
        write_csv(omni,OUT_DIR/'V70_GLOBAL_PERMUTATION_TEST.csv')
    folds,endpoint=calibration_fold_audit()
    external=read_csv(OUT_DIR/'V70_EXTERNAL_FROZEN_PREDICTIONS.csv')
    if not external.empty:
        write_csv(external_nce_consistency(external),OUT_DIR/'V70_EXTERNAL_CROSS_NCE_QC.csv')
        external_bulk_context(external)
        external_ordering(external)
    create_figures(summary,external)
    known=(joint[['observed_MAE_pp']].mean().iloc[0] if len(joint)==15 else float('nan'))
    audit=[
       '# V7.0 claim-versus-evidence audit',
       '',
       'This experiment works with the EXISTING seven calibration-mixture compositions per DP and three unlabeled external AOS populations only.',
       '',
       '## Quantitative evidence',
       f'- Observed fixed-protocol K2+K8 15-condition macro-MAE: {known:.4f} M percentage points.' if len(joint)==15 else '- Incomplete 15-condition experiment; do NOT report pooled MAE.',
       '- Complete LOCO excludes every technical spectrum from the withheld composition; it is NOT a leave-one-scan-out design.',
       '- In each label permutation, K8 selection and supervised regression are repeated from training compositions; NMF remains unsupervised and is cached foldwise.',
       '- The seven physical mixture groups are shuffled as UNITS with identical permutations across DP and collision energy.',
       '- Null result estimates whether the labels are associated with fragment patterns within these source fractions; this does NOT establish independent external accuracy or a chemical mechanism.',
       '- Null folds lacking eight reliable ratios are assigned the TRAINING-LABEL MEAN by a rule applied identically to the observed labels.',
       '',
       '## External evidence',
       '- Frozen models use original training labels and original sigma1/ORIGINAL7 preprocessing; no external M% is used in selection.',
       '- External bulk 1H NMR values GG/MG/MM=12/51/82% refer to entire DP populations, NOT a DP3/DP4 ground-truth reference.',
       '- Cross-NCE agreement and sample ordering can support analytical consistency only, NOT DP-level predictive accuracy.',
       '- The three external samples were already inspected in previous development cycles, so this is a retrospective analytical challenge, not a blind validation study.',
       '- NMF OOD/ratio support must be considered before interpreting external numbers.',
       '',
       '## Important limitations',
       '- The seven compositions for a given DP are made from shared parent fractions; not seven independent source batches.',
       '- Label permutations preserve scan grouping but assume exchangeability of mixture labels under the null; they are a test of association, not causation.',
       '- The 15 DP/NCE cells are correlated. Prefer the shared-permutation global macro-MAE test over multiplying cell-level p-values.',
       '- 199 permutations gives minimum one-sided Monte Carlo p=0.005; increase only if stronger resolution is scientifically necessary.',
       '- NMR calibration-label measurement uncertainty is not explicitly propagated here; labels are read unchanged from the original code.',
       '',
       'Suggested manuscript language (USE ONLY after real-data run completes):',
       '"We tested whether the composition predictive relationship depended on the authentic reference labels by permuting the seven nominal mixture labels as intact groups and repeating training-only ratio selection, Ridge fitting and composition-held-out evaluation. The same permutation was used across DP and NCE conditions to preserve cross-condition dependence."',
       '"Application to three independently prepared unfractionated AOS populations yielded DP3/DP4 predictions and cross-energy spectral-QC diagnostics; agreement with bulk 1H NMR was considered context rather than independent DP-specific error."',
    ]
    (OUT_DIR/'V70_MANUSCRIPT_EVIDENCE_AUDIT.md').write_text('\n'.join(audit)+'\n')
    log(f'Reports updated in {OUT_DIR}')


def configure_inputs(args,require_files):
    global DATA_DIR,OUT_DIR
    DATA_DIR=args.data_dir.expanduser().resolve()
    OUT_DIR=args.out_dir.expanduser().resolve()
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    v66.DATA_DIR=DATA_DIR
    v66.OUT_DIR=OUT_DIR/'LEGACY_PREPROCESS_CACHE'
    v66.NMF_ITER=NMF_MAX_ITER
    # Cache only sigma 1.0 to avoid unused smoothing widths; keeps exact formula.
    v66.SIGMAS=(1.,)
    core.NMF_MAX_ITER=NMF_MAX_ITER
    specs=dict(version=VERSION,n_permutations=args.n_permutations,dps=args.dp,nces=args.nce,
               legacy_source_signature=v66.signature(),preprocess=PREPROCESS,
               seed=SEED,alpha=RIDGE_ALPHA,nmf_max_iter=NMF_MAX_ITER,
               inputs=source_fingerprints(args.dp,args.nce,args.external) if require_files else {})
    settings=OUT_DIR/'V70_MANIFEST.json'
    if settings.exists():
        original=json.loads(settings.read_text())
        if original!=json.loads(json.dumps(specs,default=str)):
            raise RuntimeError('Input files/settings differ from prior checkpoint. Specify NEW --out-dir.')
    else:settings.write_text(json.dumps(specs,indent=2,default=str))


def quick_selftest():
    p=permutations(50,seed=999)
    assert p.shape==(50,7)
    assert all(sorted(row)==list(range(7)) for row in p)
    assert len({tuple(row) for row in p})==len(p)
    groups=np.repeat(np.asarray(MIXES),3)
    labels=np.array([.1,.22,.33,.44,.55,.66,.8])
    a=map_group_y(groups,np.asarray(MIXES),labels)
    b=map_group_y(groups,np.asarray(MIXES),labels,np.arange(6,-1,-1))
    assert np.allclose(a[:3],.1) and np.allclose(b[:3],.8)
    # All scans of one group change together; original grouping remains intact.
    for ix in range(7):assert len(np.unique(b[ix*3:ix*3+3]))==1
    # Matched spectral features, train-only association, K8/Ridge, joint NMF path.
    rng=np.random.default_rng(123)
    nper=18
    group=np.repeat(np.asarray(MIXES),nper)
    y=np.interp(group,MIXES,[.10,.23,.35,.46,.54,.68,.84])
    cent=np.array(base.get_ratio_candidate_centers(3),float)
    if len(cent)<8:raise AssertionError('Insufficient original centers for test')
    X=np.zeros((len(group),len(cent)),float)
    for j in range(len(cent)):
        factor=(.35+1.5*(j%4)/4.)
        X[:,j]=np.exp((-1 if j%2 else 1)*factor*y)+.06*rng.lognormal(size=len(y))
    R,D,names,info=base.build_ratio_matrices(X,cent,cent.tolist())
    tr=group!=.5;te=~tr
    # This exercises exact ratio selector but may invoke predetermined null fallback.
    H=np.column_stack((X[:,0]+X[:,1],X[:,2]+X[:,3]))
    cache=dict(train=tr,test=te,ratio=R,denom=D,H=H)
    for model in ('K2_PLUS_K8','K8_ONLY'):
        pred,fallback,nr,chosen=fit_predict_group(y,group,cache,names,info,model)
        assert np.isfinite(pred) and 0<=pred<=1
        assert nr<=8
    # P-value tail direction and whole-condition synchronized null.
    df=pd.DataFrame([
        dict(dp=3,nce=18.,model='K2_PLUS_K8',perm_id=0,MAE_pp=1.,fallback_folds=0),
        dict(dp=3,nce=18.,model='K2_PLUS_K8',perm_id=1,MAE_pp=22.,fallback_folds=2),
        dict(dp=3,nce=18.,model='K2_PLUS_K8',perm_id=2,MAE_pp=33.,fallback_folds=2)])
    sm,_=groupwise_summary(df,('K2_PLUS_K8',),2)
    assert abs(float(sm.iloc[0].permutation_p_one_sided)-1/3)<1e-8
    log('PASS: label grouping, unique synchronized permutations, scoring, fallbacks, and p-values')


def main():
    global OUT_DIR
    ap=argparse.ArgumentParser(description='V7.0 existing-data-only manuscript evidence')
    ap.add_argument('--run-mode',choices=['selftest','pilot','overnight','report','external'],default='overnight')
    ap.add_argument('--data-dir',type=Path,default=DATA_DIR)
    ap.add_argument('--out-dir',type=Path,default=OUT_DIR)
    ap.add_argument('--n-permutations',type=int,default=N_PERM_DEFAULT)
    ap.add_argument('--dp',type=int,nargs='+',choices=DPS,default=list(DPS))
    ap.add_argument('--nce',type=float,nargs='+',choices=NCES,default=list(NCES))
    ap.add_argument('--external',action=argparse.BooleanOptionalAction,default=True,
                    help='Only the three existing external samples, DP3 and DP4')
    ap.add_argument('--threads',type=int,default=2)
    args=ap.parse_args()
    if len(set(args.dp))!=len(args.dp) or len(set(args.nce))!=len(args.nce):
        raise ValueError('Repeated --dp or --nce values are not allowed')
    if args.run_mode=='pilot':
        # No accidental contamination of an overnight checkpoint directory.
        if args.out_dir==OUT_DIR:
            args.out_dir=HERE/'V70_EVIDENCE_PILOT'
        args.n_permutations=min(args.n_permutations,9)
    if args.run_mode=='external' and args.dp==list(DPS):
        args.dp=[3,4]
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=max(1,args.threads))
    except ImportError:pass
    quick_selftest()
    if args.run_mode=='selftest':return
    if args.run_mode=='report':
        OUT_DIR=args.out_dir.expanduser().resolve()
        compile_reports(args.n_permutations);return
    configure_inputs(args,require_files=True)
    permmaps=permutations(args.n_permutations)
    ext_existing=read_csv(OUT_DIR/'V70_EXTERNAL_FROZEN_PREDICTIONS.csv')
    for dp in args.dp:
        packs=load_calibration(dp)
        if args.run_mode!='external':
            for e in args.nce:
                output_file=OUT_DIR/'PERMUTATIONS'/f'DP{dp}_NCE{int(e)}.csv'
                single_condition(dp,e,packs[e],permmaps,output_file)
        if args.external and dp in (3,4):
            if not (args.run_mode=='pilot'):
                new=external_inference(dp,packs)
                if new:
                    ext_existing=pd.concat([ext_existing,pd.DataFrame(new)],ignore_index=True)
                    ext_existing=ext_existing.drop_duplicates(['dp','nce','model','sample'],keep='last')
                    write_csv(ext_existing,OUT_DIR/'V70_EXTERNAL_FROZEN_PREDICTIONS.csv')
    # Only full-grid jobs can produce valid publication-level global table.
    files=list((OUT_DIR/'PERMUTATIONS').glob('DP*_NCE*.csv'))
    full_grid=len(files)==15 and args.dp==list(DPS) and args.nce==list(NCES)
    if full_grid:compile_reports(args.n_permutations)
    else:
        log('Partial run. Full 15-condition omnibus and heatmap not generated.')
        if not ext_existing.empty:
            write_csv(external_nce_consistency(ext_existing),OUT_DIR/'V70_EXTERNAL_CROSS_NCE_QC.csv')
    log('Finished. Note: real mzML data required for scientific results.')


if __name__=='__main__':main()
