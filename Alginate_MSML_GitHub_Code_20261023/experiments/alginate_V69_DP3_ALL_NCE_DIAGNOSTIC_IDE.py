#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V6.9 — DP3 spectral transfer diagnostics at NCE 18%, 22%, 26%.

Uses UNMODIFIED original V6.6 preprocessing and model helpers included in this ZIP.
Answers which ratio ions and global fragment regions shift between known-mixture
calibration and the three UNLABELED external materials, how deleting any one of
8 ratios changes predicted M%, and whether technical scan sampling is stable.

IMPORTANT: bulk 12/51/82% NMR values are NOT DP3 ground truth and are never
used for fitting, ranking, correction, ablation selection or QC thresholds.
Technical-scan block-bootstrap intervals are NOT calibration or accuracy CIs.

Default in PyCharm: overnight. Edit DATA_DIR below or --data-dir PATH.
"""
from __future__ import annotations
# Repository layout adapter: allow direct execution from experiments/.
import sys as _repo_sys
from pathlib import Path as _repo_Path
_repo_sys.path.insert(0, str(_repo_Path(__file__).resolve().parents[1]))

import argparse
import hashlib
import json
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import alginate_ratio_transfer_IDE as base
import v62_core as core
import alginate_V66_ALL_NCE_OVERNIGHT_IDE as v66

HERE = Path(__file__).resolve().parent
DATA_DIR = Path.home() / 'Desktop' / 'alginate_experiment'
OUT_DIR = HERE / 'V69_DP3_ALL_NCE_DIAGNOSTICS'
DP = 3
NCES = (18., 22., 26.)
MIXES = tuple(v66.MIXES)
# Fixed from uploaded V6.6 calibration-only selection, no external outcomes used:
SELECTED_V66 = {
    18.: (0., 'TIC_ONLY', 0.),
    22.: (.25, 'ORIGINAL7', .5),
    26.: (.25, 'ORIGINAL7', 1.),
}
# Reference preserves EXACT V6.6 original sigma1 / ORIGINAL7 / lambda1 model.
PREPROCESS_IDS = ('ORIGINAL_REFERENCE', 'V66_CAL_SELECTED')
N_RATIOS = 8
RIDGE_ALPHA = 1.
NMF_MAX_ITER = 650
BOOTSTRAPS = 300
BOOTSTRAP_BLOCK_SIZE = 10
RANDOM_SEED = 20261019
VERSION = 'V69-DP3-allNCE-diagnostics-20261019-r3'
EPS = 1e-12
EXTERNAL_NAMES = ('GG_block_rich', 'MG_block_rich', 'MM_block_rich')


def log(message):
    print(f'[{time.strftime("%H:%M:%S")}] [V6.9] {message}', flush=True)


def to_json(x):
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return float(x)
    if isinstance(x, (np.ndarray,)): return x.tolist()
    if isinstance(x, Path): return str(x)
    raise TypeError(type(x).__name__)


def settings():
    return dict(version=VERSION, dp=DP, nces=NCES, mixes=MIXES,
                selected_v66=SELECTED_V66, preprocess_ids=PREPROCESS_IDS,
                ratio_count=N_RATIOS, alpha=RIDGE_ALPHA, nmf_iter=NMF_MAX_ITER,
                bootstraps=BOOTSTRAPS, bootstrap_block_size=BOOTSTRAP_BLOCK_SIZE,
                seed=RANDOM_SEED, calibration_only_ratio_selection=True,
                original_preprocessor_signature=v66.signature(),
                external_nmr_used_for_model_selection=False)


def signature():
    return hashlib.sha256(json.dumps(settings(),sort_keys=True,default=to_json).encode()).hexdigest()[:20]


def export_csv(rows, path, allow_empty=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if isinstance(rows,pd.DataFrame):df=rows
    else:df=pd.DataFrame(rows)
    if df.empty and not allow_empty:return
    tmp=path.with_name(path.name+'.tmp')
    df.to_csv(tmp,index=False)
    os.replace(tmp,path)


def save_obj(obj,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    joblib.dump(obj,temp,compress=2)
    os.replace(temp,path)


def file_fingerprints():
    s={}
    for is_ext in (False,True):
        for label,path in sorted(v66.mzml_sources(DP,is_ext).items()):
            if not path.exists():
                if not is_ext:raise FileNotFoundError(f'Missing calibration mzML: {path}')
                s[f'external_{label}']='MISSING';continue
            st=path.stat();s[f'{"external" if is_ext else "cal"}_{label}']=(
                str(path.resolve()),st.st_size,st.st_mtime_ns)
    return s


def configure_paths(require_files=True):
    v66.DATA_DIR=DATA_DIR
    v66.OUT_DIR=OUT_DIR/'PREPROCESS_CACHE'
    core.NMF_MAX_ITER=NMF_MAX_ITER
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    info=dict(settings=settings(),signature=signature(),data_dir=str(DATA_DIR.resolve()),
              inputs=(file_fingerprints() if require_files else {}))
    p=OUT_DIR/'V69_SETTINGS.json'
    if p.exists():
        prev=json.loads(p.read_text())
        if prev!=json.loads(json.dumps(info,default=to_json)):
            raise RuntimeError('Experiment inputs or parameters changed. Choose a NEW --out-dir '
                               'rather than combining incompatible checkpoints.')
    else:p.write_text(json.dumps(info,indent=2,default=to_json))


def preprocessing(nce,preprocess_id):
    return ((1.,'ORIGINAL7',1.) if preprocess_id=='ORIGINAL_REFERENCE'
            else SELECTED_V66[nce])


def checkpoint(kind,nce,preprocess_id,hold=None):
    suffix='FULL' if hold is None else f'H{int(round(hold*100)):03d}'
    return OUT_DIR/'CHECKPOINTS'/kind/f'E{int(nce)}'/(preprocess_id+'__'+suffix+'.joblib')


def init_packs():
    packs=v66.cache_dp(DP,False)
    # Allows calibration-only analysis if external files are unavailable, but report incomplete.
    ext=v66.cache_dp(DP,True)
    return packs,ext


def ratio_details(pack):
    cent=list(base.get_ratio_candidate_centers(DP))
    _,_,names,metadata=base.build_ratio_matrices(
        np.ones((1,len(pack['grid'])),float),pack['grid'],cent)
    preprocess=np.asarray(pack['centers'],float)
    def support_i(mz):
        j=np.flatnonzero(np.isclose(preprocess,float(mz),atol=1e-5))
        if not len(j):raise ValueError(f'No raw fragment-support center for m/z {mz}')
        return int(j[0])
    def grid_i(mz):return int(np.argmin(np.abs(pack['grid']-float(mz))))
    return names,metadata,[(support_i(m['p1']),support_i(m['p2'])) for m in metadata],\
           [(grid_i(m['p1']),grid_i(m['p2'])) for m in metadata]


def dist_stats(vals):
    z=np.asarray(vals,float);z=z[np.isfinite(z)]
    if len(z)==0:return dict(n=0,mean=np.nan,sd=np.nan,median=np.nan,
                           q05=np.nan,q25=np.nan,q75=np.nan,q95=np.nan,min=np.nan,max=np.nan)
    q=np.quantile(z,[.05,.25,.75,.95])
    return dict(n=int(len(z)),mean=float(np.mean(z)),sd=float(np.std(z)),
                median=float(np.median(z)),q05=float(q[0]),q25=float(q[1]),
                q75=float(q[2]),q95=float(q[3]),min=float(np.min(z)),max=float(np.max(z)))


def predict_joint(fit,pack,X):
    clipped,raw=fit.predictions(pack,X)
    _,res=fit.nmf.features_residual(X)
    return clipped*100.,raw*100.,res


def block_bootstrap(values,n_boot=BOOTSTRAPS,block_size=BOOTSTRAP_BLOCK_SIZE,seed=1):
    """Circular moving-block resampling of *ordered* technical scans.

    This measures scan-sampling variability conditional on one already fitted
    model and one injection/file. Not a confidence interval for actual M%.
    """
    x=np.asarray(values,float);n=len(x)
    if not n:return np.nan,np.nan,np.nan
    if n<3:return float(np.mean(x)),float(np.mean(x)),float(np.mean(x))
    size=min(block_size,n); blocks=int(np.ceil(n/size))
    rng=np.random.default_rng(seed)
    estimates=[]
    for _ in range(n_boot):
        starts=rng.integers(0,n,size=blocks)
        inds=(starts[:,None]+np.arange(size)[None,:])%n
        estimates.append(float(np.mean(x[inds.ravel()[:n]])))
    q=np.quantile(estimates,[.025,.975])
    return float(np.mean(x)),float(q[0]),float(q[1])


def summarize_preds(sample,pred,raw,res,fit,pack,mask,nce,ppid,arm,removed_index=None):
    p=np.asarray(pred,float); raw=np.asarray(raw,float)
    n=len(p)
    lo,low,high=block_bootstrap(p,seed=(RANDOM_SEED+int(nce)*100+
                        sum(map(ord,sample))+ (0 if removed_index is None else removed_index)))
    return dict(dp=DP,nce=nce,preprocessing=ppid,arm=arm,
        removed_ratio_position=(removed_index if removed_index is not None else 0),
        sample=sample,n_retained_scans=n,pred_M_pct=lo,
        scan_sd_pp=float(np.std(p)),scan_median_M_pct=float(np.median(p)),
        bootstrap_technical_low_pp=low,bootstrap_technical_high_pp=high,
        block_bootstrap_size=min(BOOTSTRAP_BLOCK_SIZE,n),
        clipped_scan_pct=float(np.mean((raw<0)|(raw>100))*100),
        scan_half_mean_difference_pp=(float(abs(p[:n//2].mean()-p[n//2:].mean()))
            if n>1 else np.nan),
        nmf_ood_scans_pct=float(np.mean(res>fit.nmf.resid_q99)*100) if res is not None else np.nan,
        nmf_residual_mean=float(np.mean(res)) if res is not None else np.nan)


@dataclass
class Ablated:
    id:str
    dropped:int
    ids:tuple
    ridge:object
    scaler:object
    nmf:object

    def predict(self,pack,X):
        h,res=self.nmf.features_residual(X)
        r=v66.ratio_data(pack,X)[0][:,list(self.ids)]
        a=np.column_stack([h,r])
        raw=self.ridge.predict(self.scaler.transform(a))
        return np.clip(raw,0.,1.)*100.,raw*100.,res


def fit_ablation_set(pack,train,X,ref_ids,full):
    """Identical training folds, NMF bases and feature scaling policy as V6.6.

    No external labels; only change is deletion of one of 8 already-selected ratios.
    """
    all_ratios=v66.ratio_data(pack,X)[0]
    h,_=full.nmf.features_residual(X)
    y=pack['y'][train]
    w=core.group_weights(pack['nom'][train])
    models={}
    for dropped in range(1,9):
        kept=tuple(int(q) for i,q in enumerate(ref_ids,1) if i!=dropped)
        a=np.column_stack([h,all_ratios[:,list(kept)]])
        scale=StandardScaler()
        try:scale.fit(a[train],sample_weight=w)
        except TypeError:scale.fit(a[train])
        ridge=Ridge(alpha=RIDGE_ALPHA).fit(scale.transform(a[train]),y,sample_weight=w)
        models[dropped]=Ablated('DROP_RATIO_'+str(dropped),dropped,kept,ridge,scale,full.nmf)
    return models


def score_diagnostics(cal,ext,cal_x,ext_x,ratios,nce,ppid):
    """Calibration label based x-values; external M% is NEVER assigned on x-axis."""
    _,metas,rawids,gridids=ratio_details(cal)
    Rcal=v66.ratio_data(cal,cal_x)[0]
    Rext=v66.ratio_data(ext,ext_x)[0] if ext is not None else None
    records=[]; shifts=[]
    for position,ri in enumerate(ratios,1):
        meta=metas[int(ri)]
        ni,di=gridids[ri];nraw,draw=rawids[ri]
        for specimen,pack,X,R in [('CALIBRATION',cal,cal_x,Rcal),('EXTERNAL',ext,ext_x,Rext)]:
            if pack is None:continue
            if specimen=='CALIBRATION':
                series=[(f'cal_M{int(round(g*100))}',np.isclose(cal['nom'],g),g)
                        for g in MIXES]
            else:series=[(s,ext['sample']==s,np.nan) for s in sorted(set(ext['sample']))]
            for label,idx,nominal in series:
                if not np.any(idx):continue
                data=dict(nce=nce,preprocessing=ppid,ratio_position=position,
                          ratio_index=int(ri),ratio_label=meta['name'],
                          numerator_mz=float(meta['p1']),denominator_mz=float(meta['p2']),
                          dataset=specimen,sample=label,nominal_mix_M_pct=
                          (100*nominal if np.isfinite(nominal) else np.nan),
                          known_label_M_pct=(float(cal['y'][idx].mean()*100)
                              if specimen=='CALIBRATION' else np.nan),
                          numerator_centroid_support_pct=float(pack['support'][idx,nraw].mean()*100),
                          denominator_centroid_support_pct=float(pack['support'][idx,draw].mean()*100),
                          both_centroid_support_pct=float(np.mean(pack['support'][idx,nraw] &
                                                               pack['support'][idx,draw])*100),
                          numerator_median_normalized=float(np.median(X[idx,ni])),
                          denominator_median_normalized=float(np.median(X[idx,di])),
                          ratio_denominator_near_zero_pct=float(np.mean(X[idx,di]<=1e-9)*100))
                data.update({'ratio_'+k:v for k,v in dist_stats(R[idx,ri]).items()})
                records.append(data)
        if ext is not None:
            # Reference is full calibration distribution, NOT an assumed external M label.
            q1,q99=np.quantile(Rcal[:,ri],[.01,.99]);
            med=np.median(Rcal[:,ri]);iqr=max(np.quantile(Rcal[:,ri],.75)-np.quantile(Rcal[:,ri],.25),1e-8)
            for label in sorted(set(ext['sample'])):
                p=ext['sample']==label;z=Rext[p,ri]
                shifts.append(dict(nce=nce,preprocessing=ppid,ratio_position=position,
                    ratio_index=int(ri),ratio_label=meta['name'],sample=label,
                    calibration_p01=float(q1),calibration_p99=float(q99),
                    external_outside_calibration_p01_p99_pct=float(np.mean((z<q1)|(z>q99))*100),
                    external_median_minus_cal_median_in_IQR=float((np.median(z)-med)/iqr),
                    external_below_cal_p01_pct=float(np.mean(z<q1)*100),
                    external_above_cal_p99_pct=float(np.mean(z>q99)*100),
                    external_nmr_truth_known=False))
    return records,shifts


def reconstruction_hotspots(cal,ext,cal_x,ext_x,fit,nce,ppid):
    """Normalized absolute spectral errors at each active m/z, not NMR errors."""
    nmf=fit.nmf;active=np.flatnonzero(nmf.active)
    records=[]
    for category,pack,X in [('CALIBRATION',cal,cal_x),('EXTERNAL',ext,ext_x)]:
        if pack is None:continue
        if category=='CALIBRATION':groups=[(f'cal_M{int(g*100)}',np.isclose(pack['nom'],g)) for g in MIXES]
        else:groups=[(s,pack['sample']==s) for s in sorted(set(pack['sample']))]
        for label,ix in groups:
            if not ix.any():continue
            z=np.maximum(X[ix][:,nmf.active],0)*nmf.feature_weights[None,:]
            c=nmf.nmf.transform(z)
            est=c@nmf.nmf.components_
            err=np.abs(z-est)/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),EPS)
            means=err.mean(axis=0)
            ids=np.argsort(means)[-25:][::-1]
            for rank,j in enumerate(ids,1):
                records.append(dict(nce=nce,preprocessing=ppid,dataset=category,
                                    sample=label,rank=rank,mz=float(pack['grid'][active[j]]),
                                    normalized_abs_recon_error=float(means[j]),
                                    n_retained_scans=int(ix.sum())))
    return records


def export_retained_scan_counts(calibration_packs, external_packs):
    """Actual *analyzed* scan counts by DP, NCE and specimen.

    The mzML reader skips irrelevant MS levels/precursors/NCEs. These counts
    are NOT the original instrument acquisition counts; the latter require
    the instrument scan manifest / raw metadata and cannot be inferred here.
    """
    rows=[]
    for source,packs in (("CALIBRATION",calibration_packs),
                         ("EXTERNAL",external_packs)):
        for nce,pack in sorted(packs.items()):
            for specimen in sorted(set(pack['sample'])):
                idx=(pack['sample']==specimen)
                rows.append(dict(dp=DP,nce=float(nce),source=source,
                    sample=str(specimen),n_retained_MS2_spectra=int(idx.sum()),
                    acquired_MS2_spectra="NOT_DETERMINED_FROM_RETAINED_CACHE",
                    n_independent_composition_labels=(1 if source=="CALIBRATION" else 0),
                    nominal_calibration_M_pct=(float(pack['nom'][idx][0]*100)
                        if source=="CALIBRATION" else np.nan),
                    NMR_derived_calibration_M_pct=(float(pack['y'][idx][0]*100)
                        if source=="CALIBRATION" else np.nan),
                    note="technical scans, not biological replicates"))
    export_csv(rows,OUT_DIR/'V69_RETAINED_SCAN_COUNTS_BY_SAMPLE_NCE.csv')


def run_one(cal,ext,nce,ppid,run_mode='overnight'):
    sigma,group,lam=preprocessing(nce,ppid)
    Xcal=v66.matrix(cal,sigma,group,lam)
    Xext=v66.matrix(ext,sigma,group,lam) if ext is not None else None
    types=['FULL'] if run_mode=='external' else [f'H{int(round(g*100)):03d}' for g in MIXES]+['FULL']
    for tag in types:
        hold=None if tag=='FULL' else next(g for g in MIXES if tag==f'H{int(round(g*100)):03d}')
        is_full=hold is None
        out=checkpoint('RESULTS',nce,ppid,hold)
        if out.exists():continue
        try:
            train=np.ones(len(cal['y']),bool) if is_full else ~np.isclose(cal['nom'],hold)
            test=~train
            if not is_full and np.unique(cal['nom'][train]).size!=6:
                raise ValueError('Expected six independent composition holdouts')
            ratios=v66.choose_reference_ratios(cal,train)
            orig=v66.fit_models(cal,train,Xcal,ratios,sigma,group,lam)
            full=orig['K2_PLUS_K8'];k8=orig['K8_ONLY']
            dropped=fit_ablation_set(cal,train,Xcal,ratios,full)
            selected=[dict(nce=nce,preprocessing=ppid,fold=tag,ratio_position=i,
                ratio_index=int(j),ratio_name=ratio_details(cal)[1][j]['name'],
                numerator_mz=ratio_details(cal)[1][j]['p1'],
                denominator_mz=ratio_details(cal)[1][j]['p2'])
                for i,j in enumerate(ratios,1)]
            pred_rows=[]
            if is_full:
                if ext is not None:
                    for label in sorted(set(ext['sample'])):
                        mask=ext['sample']==label;xe=Xext[mask]
                        pred,raw,res=predict_joint(full,ext,xe)
                        parent=summarize_preds(label,pred,raw,res,full,ext,mask,nce,ppid,'JOINT_K2_K8')
                        pred_rows.append(parent)
                        pb,rb=k8.predictions(ext,xe)
                        pred_rows.append(summarize_preds(label,100*pb,100*rb,res,full,ext,mask,nce,ppid,'K8_ONLY'))
                        for position,model in dropped.items():
                            a,b,c=model.predict(ext,xe)
                            pred_rows.append(summarize_preds(label,a,b,c,full,ext,mask,nce,ppid,
                                                            'DROP_ONE_RATIO',position))
            else:
                label=f'heldout_M{int(round(hold*100))}'
                pred,raw,res=predict_joint(full,cal,Xcal[test])
                rows=[summarize_preds(label,pred,raw,res,full,cal,test,nce,ppid,'JOINT_K2_K8')]
                pb,rb=k8.predictions(cal,Xcal[test])
                rows.append(summarize_preds(label,100*pb,100*rb,res,full,cal,test,nce,ppid,'K8_ONLY'))
                for position,model in dropped.items():
                    a,b,c=model.predict(cal,Xcal[test])
                    rows.append(summarize_preds(label,a,b,c,full,cal,test,nce,ppid,'DROP_ONE_RATIO',position))
                truth=float(cal['y'][test].mean()*100)
                for row in rows:
                    row.update(heldout_nominal_M_pct=float(hold*100),true_M_pct=truth,
                               abs_error_pp=abs(row['pred_M_pct']-truth))
                pred_rows.extend(rows)
            diag=[];shift=[];hot=[]
            if is_full:
                diag,shift=score_diagnostics(cal,ext,Xcal,Xext,ratios,nce,ppid)
                hot=reconstruction_hotspots(cal,ext,Xcal,Xext,full,nce,ppid)
            store=dict(signature=signature(),fold=tag,selection=selected,
                       predictions=pred_rows,ratio_diagnostics=diag,
                       feature_shift=shift,residual_hotspots=hot,
                       calibration_retained_scans=int(len(cal['y'])),
                       external_retained_scans=(int(len(ext['sample'])) if ext is not None else 0),
                       calibration_total_unclassified_rejected=int(cal.get('bad_scans',0)),
                       external_total_unclassified_rejected=(int(ext.get('bad_scans',0)) if ext else 0))
            save_obj(store,out)
            log(f'NCE{nce:g} {ppid} {tag}: saved')
        except Exception:
            path=out.with_suffix('.FAILED.txt');path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(traceback.format_exc())
            log(f'ERROR {nce:g} {ppid} {tag}: see {path}')
            raise


def compile_results():
    files=sorted((OUT_DIR/'CHECKPOINTS'/'RESULTS').glob('E*/*.joblib'))
    selected=[];pred=[];diags=[];shift=[];hotspots=[];run_stats=[]
    for path in files:
        obj=joblib.load(path)
        if obj.get('signature')!=signature():raise RuntimeError(f'Old-incompatible checkpoint: {path}')
        selected.extend(obj['selection']);pred.extend(obj['predictions'])
        diags.extend(obj['ratio_diagnostics']);shift.extend(obj['feature_shift'])
        hotspots.extend(obj['residual_hotspots'])
        run_stats.append(dict(nce=int(path.parent.name[1:]),fold=obj['fold'],
            calibration_retained_scans=obj['calibration_retained_scans'],
            external_retained_scans=obj['external_retained_scans'],
            calibration_unclassified_rejected=obj['calibration_total_unclassified_rejected'],
            external_unclassified_rejected=obj['external_total_unclassified_rejected']))
    export_csv(selected,OUT_DIR/'V69_SELECTED_RATIO_IDENTITY_ALL_NCE.csv')
    export_csv(pred,OUT_DIR/'V69_ALL_PREDICTIONS.csv')
    export_csv(diags,OUT_DIR/'V69_RATIO_ION_DISTRIBUTIONS.csv')
    export_csv(shift,OUT_DIR/'V69_EXTERNAL_RATIO_DOMAIN_SHIFT.csv')
    export_csv(hotspots,OUT_DIR/'V69_NMF_RESIDUAL_HOTSPOTS.csv')
    export_csv(run_stats,OUT_DIR/'V69_SCAN_COUNTS_AUDIT.csv')
    if pred:
        d=pd.DataFrame(pred)
        cv=d[d['sample'].astype(str).str.startswith('heldout_')].copy()
        ext=d[~d['sample'].astype(str).str.startswith('heldout_')].copy()
        if len(cv):
            by=cv.groupby(['nce','preprocessing','arm','removed_ratio_position'],dropna=False).agg(
                LORO_MAE_pp=('abs_error_pp','mean'),worst_holdout_abs_error_pp=('abs_error_pp','max'),
                n_holdouts=('abs_error_pp','count')).reset_index()
            by['complete_seven_holdouts']=by.n_holdouts.eq(7)
            baseline=by[by.arm=='JOINT_K2_K8'][['nce','preprocessing','LORO_MAE_pp']].rename(
                columns={'LORO_MAE_pp':'reference_joint_MAE_pp'})
            by=by.merge(baseline,on=['nce','preprocessing'],how='left')
            by['delta_MAE_vs_joint_pp']=by.LORO_MAE_pp-by.reference_joint_MAE_pp
            export_csv(by,OUT_DIR/'V69_LOCO_ABLATION_SUMMARY.csv')
        if len(ext):
            export_csv(ext,OUT_DIR/'V69_EXTERNAL_PREDICTIONS_AND_SCAN_BOOTSTRAP.csv')
            baseline=ext[ext.arm=='JOINT_K2_K8'][['nce','preprocessing','sample','pred_M_pct']].rename(
                columns={'pred_M_pct':'joint_reference_pred_M_pct'})
            joined=ext.merge(baseline,on=['nce','preprocessing','sample'],how='left')
            joined['delta_external_prediction_vs_joint_pp']=joined.pred_M_pct-joined.joint_reference_pred_M_pct
            if len(cv):
                x=joined.merge(by,on=['nce','preprocessing','arm','removed_ratio_position'],how='left')
            else:x=joined
            export_csv(x,OUT_DIR/'V69_ABLATION_EXTERNAL_INFLUENCE_VS_LOCO.csv')
            # Descriptive absolute sensitivity only: never interpret the sign
            # of this influence as evidence of improved DP-specific accuracy.
            influence=x[x.arm=='DROP_ONE_RATIO'].groupby(
                ['nce','preprocessing','removed_ratio_position'],as_index=False).agg(
                max_abs_external_change_pp=('delta_external_prediction_vs_joint_pp',
                    lambda z:float(np.abs(z).max())),
                mean_abs_external_change_pp=('delta_external_prediction_vs_joint_pp',
                    lambda z:float(np.abs(z).mean())),
                n_external_materials=('sample','nunique'))
            if len(cv):
                penalty=by[by.arm=='DROP_ONE_RATIO'][
                    ['nce','preprocessing','removed_ratio_position',
                     'LORO_MAE_pp','delta_MAE_vs_joint_pp','worst_holdout_abs_error_pp']]
                influence=influence.merge(penalty,on=
                    ['nce','preprocessing','removed_ratio_position'],how='left')
            export_csv(influence,OUT_DIR/'V69_RATIO_ABLATION_INFLUENCE_SUMMARY.csv')
            # Across NCE for same specimen, same preprocessing definition family.
            common=ext[ext.arm=='JOINT_K2_K8'].groupby(['preprocessing','sample']).agg(
                num_nce=('nce','nunique'),nce_range_M_pp=('pred_M_pct',lambda z:float(z.max()-z.min())),
                nce_std_M_pp=('pred_M_pct','std'),max_nmf_ood_pct=('nmf_ood_scans_pct','max'),
                worst_scan_half_pp=('scan_half_mean_difference_pp','max')).reset_index()
            export_csv(common,OUT_DIR/'V69_NCE_CONSISTENCY_AND_OOD.csv')
    make_figures()
    (OUT_DIR/'V69_REPORT_AND_LIMITATIONS.txt').write_text(
        'V6.9 DP3 all-NCE spectral transfer diagnostics\n'
        f'Signature: {signature()}\nCompleted fold+full checkpoints: {len(files)} / 48 expected\n'
        'Expected: 3 NCE x 2 preprocessings x (7 calibration holdouts + 1 full fit).\n'
        'Eight selected intensity ratios per fold trained on SIX calibration compositions only.\n'
        'Eight drop-one-ratio models refitted with original fitted NMF, ridge and scaler refitted.\n'
        'External DP3 M% unknown. No external NMR used for calibration/model ranking/correction.\n'
        'An external ratio outside the pooled calibration distribution is a DOMAIN SHIFT FLAG, not evidence of wrong M%.\n'
        'Bootstrap low/high reflect TECHNICAL ORDERED SCAN SAMPLING only, not true M% accuracy.\n'
        'NMF q99 residual is model-specific; do not compare raw residuals across preprocessing models.\n'
        'LOCO MAE of ablations is a diagnostic development measure; DO NOT pick best arm from same folds and claim unbiased validation.\n'
        'Scan counts are RETAINED analyzed spectra, not actual instrument acquisitions or independent biological replicates.\n'
        'The V6.6-preselected preprocessings were chosen on the same seven calibration folds; estimates are post-selection.\n'
        'V6.6 raw preprocessing filters only exact targeted MS2 and matching three NCEs.\n')
    log(f'Compiled {len(files)} checkpoints')


def make_figures():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        log('Matplotlib unavailable; CSV reports still written');return
    folder=OUT_DIR/'FIGURES';folder.mkdir(parents=True,exist_ok=True)
    src=OUT_DIR/'V69_RATIO_ION_DISTRIBUTIONS.csv'
    if src.exists():
        table=pd.read_csv(src)
        for energy in NCES:
            for pp in PREPROCESS_IDS:
                data=table[(np.isclose(table.nce,energy))&(table.preprocessing==pp)]
                if data.empty:continue
                fig,axs=plt.subplots(4,4,figsize=(19,15),layout='constrained')
                for position in range(1,N_RATIOS+1):
                    ax_cal=axs[(position-1)//2,(position-1)%2*2]
                    ax_ext=axs[(position-1)//2,(position-1)%2*2+1]
                    z=data[data.ratio_position==position]
                    known=z[z.dataset=='CALIBRATION'].sort_values('known_label_M_pct')
                    ext=z[z.dataset=='EXTERNAL'].copy()
                    if known.empty:continue
                    x=known.known_label_M_pct.to_numpy(float)
                    y=known.ratio_median.to_numpy(float)
                    ymin=known.ratio_q25.to_numpy(float)
                    ymax=known.ratio_q75.to_numpy(float)
                    ax_cal.errorbar(x,y,yerr=np.vstack([np.maximum(y-ymin,0),np.maximum(ymax-y,0)]),
                                     fmt='o-',capsize=2,markersize=4)
                    ax_cal.set_title(f'Ratio {position}: {str(known.ratio_label.iloc[0])}',fontsize=9)
                    ax_cal.set_xlabel('Known calibration M%')
                    ax_cal.set_ylabel('Normalized ion ratio (median, IQR)')
                    if len(ext):
                        order=list(EXTERNAL_NAMES)
                        categories=[]
                        for k,nm in enumerate(order):
                            h=ext[ext['sample']==nm]
                            if h.empty:continue
                            v=h.iloc[0];med=float(v.ratio_median);q1=float(v.ratio_q25);q3=float(v.ratio_q75)
                            ax_ext.errorbar(k,med,yerr=[[max(med-q1,0)],[max(q3-med,0)]],
                                            fmt='o',capsize=3)
                            categories.append((k,nm.split('_')[0]))
                        if categories:
                            ax_ext.set_xticks([k for k,_ in categories],[nm for _,nm in categories],rotation=18)
                        ax_ext.axhline(known.ratio_q05.min(),ls='--',alpha=.55,label='Cal 5–95% pooled range')
                        ax_ext.axhline(known.ratio_q95.max(),ls='--',alpha=.55)
                    ax_ext.set_title('External (unknown DP3 M%)',fontsize=9)
                    ax_ext.set_ylabel('Same ratio (median, IQR)')
                    # Same linear y-axis facilitates qualitative comparison; cap only for readability
                    a=np.concatenate([known.ratio_q25.to_numpy(float),known.ratio_q75.to_numpy(float),
                                      ext.ratio_q25.to_numpy(float),ext.ratio_q75.to_numpy(float)])
                    a=a[np.isfinite(a)]
                    if len(a) and np.percentile(a,98)<1e7:
                        hi=max(float(np.percentile(a,98)*1.3),.01)
                        ax_cal.set_ylim(0,hi);ax_ext.set_ylim(0,hi)
                fig.suptitle(f'DP3 NCE {energy:g}% | {pp}: eight selected local ratios\n'
                             'Calibration x = NMR-derived label; external x = material identity, NOT M%')
                fig.savefig(folder/f'RATIO_OVERVIEW_NCE{int(energy)}_{pp}.png',dpi=150)
                plt.close(fig)
    op=OUT_DIR/'V69_EXTERNAL_PREDICTIONS_AND_SCAN_BOOTSTRAP.csv'
    if op.exists():
        d=pd.read_csv(op)
        fig,axs=plt.subplots(1,2,figsize=(14,5),layout='constrained')
        for ax,pp in zip(axs,PREPROCESS_IDS):
            z=d[(d.preprocessing==pp)&(d.arm=='JOINT_K2_K8')]
            for sample in EXTERNAL_NAMES:
                h=z[z['sample']==sample].sort_values('nce')
                if len(h):
                    ax.errorbar(h.nce,h.pred_M_pct,
                                yerr=[h.pred_M_pct-h.bootstrap_technical_low_pp,
                                      h.bootstrap_technical_high_pp-h.pred_M_pct],
                                fmt='o-',capsize=3,label=sample.replace('_block_rich','-rich'))
            ax.set_xticks(NCES);ax.set_ylim(0,102)
            ax.set_xlabel('NCE (%)');ax.set_ylabel('Predicted DP3 M%')
            ax.set_title(pp+' | technical block-bootstrap only')
            ax.legend(fontsize=8)
        fig.savefig(folder/'EXTERNAL_NCE_TRENDS_TECHNICAL_BOOTSTRAP.png',dpi=160)
        plt.close(fig)
    op=OUT_DIR/'V69_ABLATION_EXTERNAL_INFLUENCE_VS_LOCO.csv'
    if op.exists():
        d=pd.read_csv(op)
        for energy in NCES:
            for pp in PREPROCESS_IDS:
                z=d[(np.isclose(d.nce,energy))&(d.preprocessing==pp)&
                    (d.arm=='DROP_ONE_RATIO')]
                if z.empty:continue
                fig,ax=plt.subplots(figsize=(10,5))
                for sample in EXTERNAL_NAMES:
                    h=z[z['sample']==sample].sort_values('removed_ratio_position')
                    ax.plot(h.removed_ratio_position,h.delta_external_prediction_vs_joint_pp,
                            marker='o',label=sample.replace('_block_rich','-rich'))
                ax.axhline(0,color='black',ls='--',lw=.8)
                ax.set_xticks(range(1,9));ax.set_xlabel('Excluded ratio position (frozen full-calibration selection)')
                ax.set_ylabel('Change in predicted DP3 M% (percentage points)')
                ax.set_title(f'DP3 NCE{int(energy)} / {pp}: feature-ablation sensitivity')
                ax.legend(fontsize=8);fig.tight_layout()
                fig.savefig(folder/f'ABLATION_EXTERNAL_NCE{int(energy)}_{pp}.png',dpi=150)
                plt.close(fig)


def synthetic_test():
    """In-memory, no mzML needed; tests all three energies and both preprocessors."""
    rng=np.random.default_rng(1234)
    # The actual preprocessor candidate lists are used here, not fabricated ratio metadata.
    grid=np.arange(100.,551.,.2)
    ctr=np.asarray(base.get_preprocess_centers(DP),float)
    ri=list(base.get_ratio_candidate_centers(DP))
    def batch():
        XX=[];yy=[];mm=[];labels=[]
        for nominal in MIXES:
            for rep in range(9):
                a=np.zeros(len(grid),float)
                for k,c in enumerate(ri):
                    amp=(1.7+.45*np.cos(k*.87)+nominal*(.9+.6*np.sin(k*.74))+
                         rng.normal(0.,.013))
                    a+=max(.02,amp)*np.exp(-.5*((grid-c)/.38)**2)
                a+=.05*np.exp(-.5*((grid-base.DP_TARGET_MZ[DP])/.55)**2)
                XX.append(a);yy.append(.08+.75*nominal);mm.append(nominal)
                labels.append('cal')
        X=np.stack(XX).astype(np.float32)
        return dict(dp=DP,nce=18.,grid=grid,centers=ctr,
            sample=np.array(labels),nom=np.array(mm),y=np.array(yy),
            support=np.ones((len(XX),len(ctr)),bool),
            masked={s:X for s in (0.,.1,.25,.5,.75,1.)},
            bad_scans=0)
    cal=batch()
    ext=dict(cal,sample=np.array([EXTERNAL_NAMES[i%3] for i in range(len(cal['sample']))]),
             nom=np.full(len(cal['nom']),np.nan),y=np.full(len(cal['y']),np.nan))
    ext['masked']=dict(cal['masked'])
    for energy in NCES:
        cal['nce']=energy;ext['nce']=energy
        for ppid in PREPROCESS_IDS:
            sig,group,lam=preprocessing(energy,ppid)
            X=v66.matrix(cal,sig,group,lam)
            hold=MIXES[0];train=~np.isclose(cal['nom'],hold)
            ids=v66.choose_reference_ratios(cal,train)
            assert len(ids)==8
            mm=v66.fit_models(cal,train,X,ids,sig,group,lam)
            full=mm['K2_PLUS_K8']
            ref,raw,res=predict_joint(full,cal,X[~train])
            assert np.isfinite(ref).all() and np.isfinite(res).all()
            ablations=fit_ablation_set(cal,train,X,ids,full)
            assert len(ablations)==8
            for m in ablations.values():
                a,b,c=m.predict(cal,X[~train]);assert np.isfinite(a).all()
            xr=v66.matrix(ext,sig,group,lam)
            info,shift=score_diagnostics(cal,ext,X,xr,ids,energy,ppid)
            assert len(info)==8*(7+3) and len(shift)==24
            hotspots=reconstruction_hotspots(cal,ext,X,xr,full,energy,ppid)
            assert len(hotspots)>0
    m,lo,hi=block_bootstrap(np.arange(40),n_boot=50,seed=2)
    assert lo<=m<=hi
    log('SELFTEST PASSED: 3 NCEs × 2 preprocessors, fold-frozen ratios, 8 refit ablations, raw-ion QC, NMF hotspots, block bootstrap')


def main():
    global DATA_DIR,OUT_DIR,BOOTSTRAPS
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-mode',choices=('selftest','pilot','overnight','external','report'),default='overnight')
    p.add_argument('--data-dir',type=Path,default=DATA_DIR)
    p.add_argument('--out-dir',type=Path,default=OUT_DIR)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--bootstrap-replicates',type=int,default=BOOTSTRAPS)
    args=p.parse_args();DATA_DIR=args.data_dir.expanduser();OUT_DIR=args.out_dir.expanduser()
    BOOTSTRAPS=args.bootstrap_replicates
    os.environ['OMP_NUM_THREADS']=str(args.threads)
    os.environ['OPENBLAS_NUM_THREADS']=str(args.threads)
    os.environ['MKL_NUM_THREADS']=str(args.threads)
    if args.run_mode=='selftest':synthetic_test();return
    if args.run_mode=='report':
        # No mzML required to compile previously generated results.
        if not OUT_DIR.exists():raise FileNotFoundError(OUT_DIR)
        compile_results();return
    configure_paths()
    cal,external=init_packs()
    for nce in NCES:
        if nce not in external:
            log(f'WARNING: no retained DP3 external scans at NCE {nce:g}%; '
                'external diagnostics for this NCE will be absent.')
        elif set(EXTERNAL_NAMES) - set(external[nce]['sample']):
            missing=sorted(set(EXTERNAL_NAMES)-set(external[nce]['sample']))
            log(f'WARNING: DP3 NCE {nce:g}% missing external materials: {missing}')
    export_retained_scan_counts(cal,external)
    energies=NCES[:1] if args.run_mode=='pilot' else NCES
    for en in energies:
        for pp in PREPROCESS_IDS:
            run_one(cal[en],external.get(en),en,pp,
                    run_mode=('external' if args.run_mode=='external' else 'overnight'))
    compile_results()
    log('Complete. See results, QC and figures in '+str(OUT_DIR))


if __name__=='__main__':main()
