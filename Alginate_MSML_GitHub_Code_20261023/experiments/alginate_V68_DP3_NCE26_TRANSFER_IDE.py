#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V6.8: DP3 NCE26 reliable-ratio and global NMF external-transfer ablations.

One NCE, one DP. Compares V6.6-compatible original K8 selection, stronger
training-only denominator screening, inner-fold ratio consensus, and simulated
ion-perturbation resistance. Raw versus log-ratio features, standard versus
precursor-excluded rank-two NMF, and four reporting-only scan aggregations.

No external NMR value participates in fitting, ratio selection, model selection,
or scan weighting. The optional bulk-NMR comparison is explicitly post-hoc.

PyCharm default: --run-mode overnight; edit DATA_DIR below or provide --data-dir.
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
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

import alginate_ratio_transfer_IDE as base
import v62_core as core
import alginate_V66_ALL_NCE_OVERNIGHT_IDE as v66

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path.home() / 'Desktop' / 'alginate_experiment'
OUT_DIR = ROOT / 'V68_DP3_NCE26_RESULTS'
DP, NCE = 3, 26.
MIXES = tuple(v66.MIXES)
PREPROCESS = (('V66_SELECTED', 0.25, 'ORIGINAL7', 1.),
              ('ORIGINAL_REFERENCE', 1., 'ORIGINAL7', 1.))
SELECTORS = ('REFERENCE', 'RAW_SUPPORT', 'CONSENSUS', 'PERTURB_ROBUST')
TRANSFORMS = ('RAW', 'LOG_RATIO')
NMF_OPTIONS = ('FULL', 'NO_PRECURSOR')
FEATURES = ('K8_ONLY', 'K2_PLUS_K8')
AGGREGATIONS = ('MEAN', 'MEDIAN', 'TRIM10', 'QC_WEIGHTED')
SEED = 20261019
RIDGE_ALPHA = 1.0
MAX_NMF_ITER = 650
K_RATIOS = 8
RAW_SUPPORT_FRACTION = .98
LOG_FLOOR_FRACTION = .01
QC_RESIDUAL_MULTIPLIER = 1.5
NMR_BULK_CONTEXT = {'GG_block_rich':12., 'MG_block_rich':51., 'MM_block_rich':82.}
VERSION = 'DP3-E26-v68a-20261019'
EPS = 1.e-12


def log(s):
    print(f'[{time.strftime("%H:%M:%S")}] [V6.8] {s}',flush=True)


def jdefault(x):
    if isinstance(x,Path):return str(x)
    if isinstance(x,np.integer):return int(x)
    if isinstance(x,np.floating):return float(x)
    if isinstance(x,np.ndarray):return x.tolist()
    raise TypeError(str(type(x)))


def settings():
    return dict(version=VERSION,dp=DP,nce=NCE,mixes=MIXES,
        preprocessing=PREPROCESS,selectors=SELECTORS,transforms=TRANSFORMS,
        nmf_options=NMF_OPTIONS,features=FEATURES,aggregations=AGGREGATIONS,
        seed=SEED,alpha=RIDGE_ALPHA,nmf_iter=MAX_NMF_ITER,
        ratio_count=K_RATIOS,raw_support=RAW_SUPPORT_FRACTION,
        log_floor_fraction=LOG_FLOOR_FRACTION,ratio_baseline='V66 reference σ=1 full group normalization',
        bulk_nmr_used_for_fitting=False)


def signature():
    return hashlib.sha256(json.dumps(settings(),sort_keys=True,default=jdefault).encode()).hexdigest()[:16]


def write_csv(records,path):
    if len(records)==0:return
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    pd.DataFrame(records).to_csv(temp,index=False)
    os.replace(temp,path)


def save_joblib(obj,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    joblib.dump(obj,temp,compress=2);os.replace(temp,path)


def check_setup():
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    paths={**v66.mzml_sources(DP,False),**v66.mzml_sources(DP,True)}
    paths={k:str(p) for k,p in paths.items()}
    fingerprints={k:(str(p),Path(p).stat().st_size,Path(p).stat().st_mtime_ns)
                  for k,p in paths.items() if Path(p).exists()}
    info=dict(signature=signature(),settings=settings(),data_dir=str(DATA_DIR.resolve()),
              inputs=fingerprints)
    path=OUT_DIR/'V68_SETTINGS.json'
    if path.exists():
        if json.loads(path.read_text())!=json.loads(json.dumps(info,default=jdefault)):
            raise RuntimeError('Input files or experiment settings changed. Provide a NEW --out-dir '
                               'to avoid mixing incompatible checkpoints.')
    else:path.write_text(json.dumps(info,indent=2,default=jdefault))


def ratio_index_data(pack):
    # This list has an index independent of the smoothing/group variant.
    centers=list(base.get_ratio_candidate_centers(DP))
    _,_,names,metadata=base.build_ratio_matrices(np.ones((1,len(pack['grid']))),pack['grid'],centers)
    fragment_centers=np.asarray(pack['centers'],float)
    def support_index(mz):
        j=np.where(np.isclose(fragment_centers,mz,atol=1.e-5))[0]
        if not len(j):raise ValueError(f'No raw-centroid support column for m/z {mz}')
        return int(j[0])
    p1=np.asarray([s['p1'] for s in metadata],float)
    p2=np.asarray([s['p2'] for s in metadata],float)
    num_idx=np.asarray([support_index(p) for p in p1],int)
    den_idx=np.asarray([support_index(p) for p in p2],int)
    return names,metadata,num_idx,den_idx


def normalized_features(pack,X):
    return base.build_ratio_matrices(X,pack['grid'],list(base.get_ratio_candidate_centers(DP)))


def group_support_rate(mask,nom,support):
    return min(float(np.mean(support[mask & np.isclose(nom,g)]))
               for g in sorted(set(nom[mask])))


def select_greedy(ranked, scores, n=K_RATIOS):
    chosen=[]; avail=list(ranked)
    while avail and len(chosen)<n:
        proposals=[]
        for row in avail:
            j=int(row['idx']);score=float(scores[j]);corr=max(
                (base.trajectory_redundancy(row['trajectory'],b['trajectory']) for b in chosen),
                default=0.)
            score-=float(base.REDUNDANCY_PENALTY)*corr**4
            proposals.append((score,-j,row))
        winner=max(proposals,key=lambda p:(p[0],p[1]))[2]
        chosen.append(winner)
        avail=[r for r in avail if int(r['idx'])!=int(winner['idx'])]
    return [int(r['idx']) for r in chosen]


def candidate_base_scores(ranked):
    scores={}
    for row in ranked:
        bounded=np.clip(float(row['technical_snr']) if np.isfinite(row['technical_snr']) else 0.,0.,50.)
        scores[int(row['idx'])]=(0.57*row['abs_spearman']+0.23*row['abs_pearson']
             +0.12*np.log1p(bounded)/np.log1p(50.)
             +0.08*row['min_denom_valid_fraction']-.02*row['monotonic_violations'])
    return scores


def choose_ratios(pack,train,X,policy,fold_key):
    """Select exactly eight IDs using calibration TRAIN compositions only."""
    names,metadata,idx_num,idx_den=ratio_index_data(pack)
    if policy=='REFERENCE':
        X_rank=v66.matrix(pack,1.,'ORIGINAL7',1.)
    else:X_rank=X
    R,D,_,_=normalized_features(pack,X_rank)
    ranked,diagnostics=base.rank_ratio_candidates(R[train],D[train],
                                                  pack['y'][train],pack['nom'][train],names,metadata)
    diagnostic_rows=[]
    for row in diagnostics:
        diagnostic_rows.append(dict(fold=fold_key,selector=policy,**row))
    if policy=='REFERENCE':
        # Exact V6.6 train-fold ratio policy.
        selected=base.select_nonredundant_ratios(ranked,K_RATIOS)
        return tuple(int(row['idx']) for row in selected),diagnostic_rows
    train_nom=pack['nom'];support=pack['support']
    selected_ranked=list(ranked)
    scores=candidate_base_scores(selected_ranked)
    if policy in ('RAW_SUPPORT','PERTURB_ROBUST'):
        selected_ranked=[r for r in selected_ranked
             if group_support_rate(train,train_nom,support[:,idx_den[int(r['idx'])]])
                    >=RAW_SUPPORT_FRACTION]
        scores={i:scores[i] for i in [int(r['idx']) for r in selected_ranked]}
    if policy=='CONSENSUS':
        # Six inner holdouts (five calibrator populations for every sub-selection),
        # never using any outer-held-out composition labels.
        counts=Counter()
        for inner_g in sorted(np.unique(train_nom[train])):
            inner_train=train & ~np.isclose(train_nom,inner_g)
            sub,_=base.rank_ratio_candidates(R[inner_train],D[inner_train],
                                              pack['y'][inner_train],train_nom[inner_train],names,metadata)
            ids=[int(z['idx']) for z in base.select_nonredundant_ratios(sub,K_RATIOS)]
            counts.update(ids)
        scores={i:(.65*s+.35*counts[i]/max(len(np.unique(train_nom[train])),1))
                for i,s in scores.items()}
    if policy=='PERTURB_ROBUST':
        # Perturb *training* intensities only; fragment dropout is represented
        # as one ion at zero intensity in 3% of the technical observations.
        # Fragility is normalized by the composition signal's IQR, not by raw
        # intensity units. The same selected IDs feed both feature blocks.
        rng=np.random.default_rng(SEED+(int(fold_key[1:]) if fold_key.startswith('H') else 999))
        keep=np.flatnonzero(train)
        centers=list(base.get_ratio_candidate_centers(DP))
        idx_centers=[int(np.argmin(abs(pack['grid']-p))) for p in centers]
        peak=np.maximum(X_rank[keep][:,idx_centers],0.)
        c_idx={float(p):j for j,p in enumerate(centers)}
        fragility={}
        for row in selected_ranked:
            j=int(row['idx']);ni=c_idx[float(row['p1'])];di=c_idx[float(row['p2'])]
            ref=np.log((peak[:,ni]+1.e-7)/(peak[:,di]+1.e-7))
            shifts=[]
            for rep in range(4):
                n=peak[:,ni]*np.exp(rng.normal(0.,.15,len(peak)))
                d=peak[:,di]*np.exp(rng.normal(0.,.15,len(peak)))
                n[rng.random(len(n))<.03]=0.
                d[rng.random(len(d))<.03]=0.
                pert=np.log((n+1.e-7)/(d+1.e-7))
                shifts.append(np.abs(pert-ref))
            # high-percentile perturbation shift penalizes dropout fragility.
            signal=float(np.percentile(ref,90)-np.percentile(ref,10))
            fragility[j]=float(np.percentile(np.concatenate(shifts),90)/max(signal,.1))
        scores={j:s-0.10*np.log1p(fragility.get(j,0.)) for j,s in scores.items()}
    ids=select_greedy(selected_ranked,scores)
    return tuple(ids),diagnostic_rows


def ratio_transform_info(pack,X,train,ids,transform):
    """Train-derived pseudo-counts are fixed for held-out/external spectra."""
    _,_,names,metadata=normalized_features(pack,X)
    if transform=='RAW':return dict(transform=transform,eps=None)
    peak_centers=list(base.get_ratio_candidate_centers(DP))
    peaks=np.maximum(X[:,[int(np.argmin(abs(pack['grid']-p))) for p in peak_centers]],0.)
    floors=[]
    for col in range(peaks.shape[1]):
        pos=peaks[train,col];pos=pos[pos>0]
        floor=max(1e-8,LOG_FLOOR_FRACTION*float(np.median(pos)) if len(pos) else 1e-8)
        floors.append(floor)
    return dict(transform=transform,eps=np.asarray(floors,float))


def transform_selected(pack,X,ids,descriptor):
    if descriptor['transform']=='RAW':
        R,_,_,_=normalized_features(pack,X)
        return np.asarray(R[:,list(ids)],float)
    centers=list(base.get_ratio_candidate_centers(DP))
    idx={float(p):j for j,p in enumerate(centers)}
    peak=np.maximum(X[:,[int(np.argmin(abs(pack['grid']-p))) for p in centers]],0.)
    metadata=normalized_features(pack,np.ones((1,len(pack['grid'])),float))[3]
    flo=descriptor['eps'];cols=[]
    for j in ids:
        it=metadata[int(j)];a=idx[float(it['p1'])];b=idx[float(it['p2'])]
        cols.append(np.log(peak[:,a]+flo[a])-np.log(peak[:,b]+flo[b]))
    return np.clip(np.column_stack(cols),-30.,30.)


def feature_nmfs(X,train,pack,mode):
    N=np.maximum(X,0.)
    if mode=='NO_PRECURSOR':
        # Explicitly remove intact DP3 precursor-region from the NMF input ONLY.
        N=N.copy()
        N[:,np.abs(pack['grid']-base.DP_TARGET_MZ[DP])<=base.PEAK_WINDOW]=0.
    nmf=core.fit_rank2(N,pack['y'],pack['nom'],train,weighted=False,raw_coeff=True)
    return nmf


def nmf_matrix(X,pack,mode):
    if mode=='FULL':return X
    Z=X.copy();Z[:,np.abs(pack['grid']-base.DP_TARGET_MZ[DP])<=base.PEAK_WINDOW]=0.
    return Z


@dataclass
class Fit:
    preprocess:str
    sigma:float
    group:str
    lam:float
    selector:str
    transform:str
    nmf_option:str
    feature_mode:str
    ids:tuple
    ratio_descriptor:dict
    nmf:object
    scaler:object
    ridge:object

    def predict(self,pack,X):
        ratio=transform_selected(pack,X,self.ids,self.ratio_descriptor)
        resid=None
        if self.feature_mode=='K2_PLUS_K8':
            H,resid=self.nmf.features_residual(nmf_matrix(X,pack,self.nmf_option))
            feats=np.column_stack([H,ratio])
        else:feats=ratio
        raw=self.ridge.predict(self.scaler.transform(feats))
        pred=np.clip(raw,0.,1.)
        return pred,raw,resid


def fit_one(pack,train,X,preprocess,sigma,group,lam,selector,transform,nmf_mode,feature_mode,ids):
    rd=ratio_transform_info(pack,X,train,ids,transform)
    R=transform_selected(pack,X,ids,rd)
    nmf=None
    if feature_mode=='K2_PLUS_K8':
        nmf=feature_nmfs(X,train,pack,nmf_mode)
        H,_=nmf.features_residual(nmf_matrix(X,pack,nmf_mode))
        A=np.column_stack([H,R])
    else:A=R
    sc=StandardScaler()
    w=core.group_weights(pack['nom'][train])
    try:sc.fit(A[train],sample_weight=w)
    except TypeError:sc.fit(A[train])
    ridge=Ridge(alpha=RIDGE_ALPHA).fit(sc.transform(A[train]),pack['y'][train],sample_weight=w)
    return Fit(preprocess,sigma,group,lam,selector,transform,nmf_mode,
               feature_mode,ids,rd,nmf,sc,ridge)


def qc_weights(pack,mask,fit,resid):
    """Prediction-blind weighting for scans, no bulk targets.

    Denominator detection and an NMF residual, thresholded on calibration,
    determine weights. Constant or zero-quality groups fall back to uniform.
    """
    mask=np.asarray(mask,bool)
    n=int(mask.sum())
    if n==0:return np.zeros(0)
    _,metadata,_,den_idx=ratio_index_data(pack)
    support=pack['support'][mask]
    valid=support[:,den_idx[list(fit.ids)]]
    numerator_fraction=valid.mean(axis=1)
    # Never use the scan's estimated M value to set its quality weight.
    w=np.maximum(numerator_fraction,.02)**2
    if fit.nmf is not None and resid is not None:
        scale=max(float(fit.nmf.resid_q99),1.e-7)
        w*=np.exp(-QC_RESIDUAL_MULTIPLIER*np.maximum(resid,0)/scale)
    if w.sum()<=EPS:return np.ones(n)/n
    return w/w.sum()


def summarize_samples(pack,fit,X,mask,name,truth=None):
    if int(mask.sum())==0:raise ValueError('empty evaluated sample')
    pred,raw,resid=fit.predict(pack,X)
    pred=pred*100.; raw=raw*100.
    w=qc_weights(pack,mask,fit,resid)
    trims=np.sort(pred);cut=int(np.floor(.10*len(trims)))
    tvals=trims[cut:len(trims)-cut] if len(trims)-2*cut>0 else trims
    vals={'MEAN':float(np.mean(pred)),'MEDIAN':float(np.median(pred)),
          'TRIM10':float(np.mean(tvals)),'QC_WEIGHTED':float(np.dot(pred,w))}
    hit=ratio_index_data(pack)[3][list(fit.ids)]
    selected_denom_support=pack['support'][mask][:,hit]
    minimum_denom_support=float(np.min(np.mean(selected_denom_support,axis=0))*100)
    n=len(pred)
    half=float(abs(np.mean(pred[:n//2])-np.mean(pred[n//2:]))) if n>=2 else np.nan
    outcome=[]
    for how,v in vals.items():
        row=dict(dp=DP,nce=NCE,preprocessing=fit.preprocess,
                 gaussian_sigma_Da=fit.sigma,group_scheme=fit.group,
                 normalization_strength=fit.lam,ratio_policy=fit.selector,
                 ratio_transform=fit.transform,nmf_mode=fit.nmf_option,
                 feature_mode=fit.feature_mode,aggregation=how,sample=name,
                 pred_M_pct=v,n_scans=n,scan_sd_pp=float(np.std(pred)),
                 scan_half_difference_pp=half,clipped_scan_pct=float(np.mean((raw<0)|(raw>100))*100),
                 min_raw_ratio_denominator_support_pct=minimum_denom_support,
                 nmf_ood_pct=(float(np.mean(resid>fit.nmf.resid_q99)*100) if resid is not None else np.nan),
                 nmf_relative_residual_mean=(float(np.mean(resid)) if resid is not None else np.nan),
                 qc_effective_n=(float(1./max(np.sum(w*w),1.e-12)) if n else np.nan),
                 ratio_indices=';'.join(str(j) for j in fit.ids))
        if truth is not None:
            row['true_M_pct']=float(truth)*100
            row['abs_error_pp']=abs(v-row['true_M_pct'])
        outcome.append(row)
    return outcome


def arms():
    for name,sigma,group,lam in PREPROCESS:
        for selector in SELECTORS:
            for transform in TRANSFORMS:
                yield name,sigma,group,lam,selector,transform,'NONE','K8_ONLY'
                for nmf_mode in NMF_OPTIONS:
                    yield name,sigma,group,lam,selector,transform,nmf_mode,'K2_PLUS_K8'


def arm_key(arm):
    parts=[str(i).replace('.','p') for i in arm]
    return '__'.join(parts)


def prepare_cache():
    v66.DATA_DIR=DATA_DIR
    v66.OUT_DIR=OUT_DIR/'V66_PREPROCESS_CACHE'
    core.NMF_MAX_ITER=MAX_NMF_ITER
    return v66.cache_dp(DP,False)[NCE],v66.cache_dp(DP,True).get(NCE)


def train_condition(pack,ex,hold,selected_arms):
    train=(np.ones(len(pack['y']),bool) if hold is None
           else ~np.isclose(pack['nom'],hold))
    val=(~train if hold is not None else None)
    if hold is not None and np.unique(pack['nom'][train]).size!=6:
        raise RuntimeError('Expected six training compositions per heldout fold')
    for arm in selected_arms:
        name,sigma,group,lam,selector,transform,nmf_mode,feature_mode=arm
        where='EXTERNAL' if hold is None else f'H{int(hold*100):03d}'
        target=OUT_DIR/'CHECKPOINTS'/where/(arm_key(arm)+'.csv')
        if target.exists():continue
        try:
            X=v66.matrix(pack,sigma,group,lam)
            ratios,diagnostics=choose_ratios(pack,train,X,selector,where)
            if len(ratios)!=K_RATIOS:
                raise ValueError(f'Only {len(ratios)} reliable selected ratios; require {K_RATIOS}')
            fitted=fit_one(pack,train,X,name,sigma,group,lam,selector,transform,
                           nmf_mode,feature_mode,ratios)
            records=[]
            if hold is not None:
                records=summarize_samples(pack,fitted,X[val],val,f'heldout_{hold:g}',
                                          truth=float(np.mean(pack['y'][val])))
                for r in records:r['heldout_nominal_pct']=int(round(hold*100))
            elif ex is not None:
                Xe=v66.matrix(ex,sigma,group,lam)
                for label in sorted(set(ex['sample'])):
                    selected=ex['sample']==label
                    records+=summarize_samples(ex,fitted,Xe[selected],selected,label)
            else:continue
            write_csv(records,target)
            if hold is None:
                save_joblib(fitted,OUT_DIR/'FROZEN_MODELS'/(arm_key(arm)+'.joblib'))
            # Save selection names once per ratio policy+preprocessing per holdout;
            # reused across models/transforms so no duplicate output.
            selection_path=OUT_DIR/'RATIO_SELECTION'/where/(f'{name}__{selector}.csv')
            if not selection_path.exists():
                meta=ratio_index_data(pack)[1]
                write_csv([dict(preprocessing=name,selector=selector,fold=where,
                                rank=i+1,ratio_index=j,ratio_name=meta[j]['name'],
                                numerator_mz=meta[j]['p1'],denominator_mz=meta[j]['p2'])
                           for i,j in enumerate(ratios)],selection_path)
        except Exception:
            error=target.with_suffix('.error.txt')
            error.parent.mkdir(parents=True,exist_ok=True)
            error.write_text(traceback.format_exc())
            log(f'FAILED {where}: {arm_key(arm)} (details in {error})')


def compile_reports():
    files=list((OUT_DIR/'CHECKPOINTS').glob('H*/*.csv'))
    external=list((OUT_DIR/'CHECKPOINTS'/'EXTERNAL').glob('*.csv'))
    if files:
        df=pd.concat([pd.read_csv(f) for f in files],ignore_index=True)
        df.to_csv(OUT_DIR/'V68_LORO_ALL.csv',index=False)
        keys=['preprocessing','gaussian_sigma_Da','ratio_policy','ratio_transform',
              'nmf_mode','feature_mode','aggregation']
        grp=df.groupby(keys,dropna=False).agg(
            MAE_pp=('abs_error_pp','mean'),max_error_pp=('abs_error_pp','max'),
            n_folds=('heldout_nominal_pct','nunique'),
            min_raw_denom_support_pct=('min_raw_ratio_denominator_support_pct','min'),
            mean_nmf_ood_pct=('nmf_ood_pct','mean')).reset_index()
        grp=grp.sort_values(['aggregation','MAE_pp'])
        grp['seven_folds_complete']=grp.n_folds.eq(7)
        grp.to_csv(OUT_DIR/'V68_LORO_SUMMARY.csv',index=False)
        ref=grp[(grp.preprocessing=='V66_SELECTED')&(grp.ratio_policy=='REFERENCE')
                &(grp.ratio_transform=='RAW')&(grp.nmf_mode=='FULL')
                &(grp.feature_mode=='K2_PLUS_K8')&(grp.aggregation=='MEAN')]
        if len(ref):
            val=ref.iloc[0].MAE_pp
            grp['change_vs_V66_joint_baseline_pp']=grp.MAE_pp-val
            grp.to_csv(OUT_DIR/'V68_LORO_RANKED_VS_BASELINE.csv',index=False)
    if external:
        df=pd.concat([pd.read_csv(f) for f in external],ignore_index=True)
        df.to_csv(OUT_DIR/'V68_EXTERNAL_ALL.csv',index=False)
        context=df.copy()
        context['bulk_NMR_DP_UNRESOLVED_M_pct']=context['sample'].map(NMR_BULK_CONTEXT)
        context['abs_difference_from_bulk_pp']=abs(context.pred_M_pct-context.bulk_NMR_DP_UNRESOLVED_M_pct)
        context['interpretation']='Retrospective comparison to whole-material NMR; NOT DP3 accuracy'
        context.to_csv(OUT_DIR/'V68_EXTERNAL_BULK_CONTEXT_ONLY.csv',index=False)
        keys=['preprocessing','gaussian_sigma_Da','ratio_policy','ratio_transform',
              'nmf_mode','feature_mode','aggregation']
        ranked=context.groupby(keys,dropna=False).agg(
           mean_abs_bulk_difference_pp=('abs_difference_from_bulk_pp','mean'),
           populations=('sample','nunique'),
           min_raw_denominator_support_pct=('min_raw_ratio_denominator_support_pct','min'),
           max_scan_half_difference_pp=('scan_half_difference_pp','max'),
           mean_nmf_ood_pct=('nmf_ood_pct','mean')).reset_index()
        ranked=ranked[ranked.populations.eq(3)].sort_values('mean_abs_bulk_difference_pp')
        ranked.to_csv(OUT_DIR/'V68_POSTHOC_BULK_AGREEMENT_EXPLORATORY.csv',index=False)
        if files:
            joint=ranked.merge(grp,on=keys,suffixes=('','_cal'))
            joint.to_csv(OUT_DIR/'V68_CALIBRATION_EXTERNAL_JOINT_REVIEW.csv',index=False)
            # Selection based on calibration only and fixed acceptance criteria.
            valid=joint[(joint.aggregation=='MEAN')&(joint.seven_folds_complete)
                &(joint.min_raw_denom_support_pct>=80)]
            valid=valid.sort_values(['MAE_pp','max_error_pp'])
            valid['external_raw_denominator_QC_pass_80pct']=(
                valid['min_raw_denominator_support_pct']>=80)
            valid.head(12).to_csv(OUT_DIR/'V68_CALIBRATION_SHORTLIST_NOT_EXTERNAL_OPTIMAL.csv',index=False)
        top=ranked.head(10)
        if len(top):
            k=keys
            c=context.merge(top[k],on=k,how='inner')
            c.to_csv(OUT_DIR/'V68_TOP10_DESCRIPTIVE_EXTERNAL_PREDICTIONS.csv',index=False)
        ckeys=keys + ['aggregation'] if 'aggregation' not in keys else keys
        compact=df.pivot_table(index=keys, columns='sample',values='pred_M_pct',aggfunc='mean').reset_index()
        compact.to_csv(OUT_DIR/'V68_EXTERNAL_COMPACT_M_PERCENT.csv',index=False)
    report=[f'V6.8 DP3/NCE26 experiment signature {signature()}',
           f'Composition-holdout checkpoints: {len(files)}; external checkpoints: {len(external)}',
           f'Expected full evaluation: {len(list(arms()))} arms × 7 composition folds',
           'K8 ratio IDs are chosen from calibration TRAIN folds only.',
           'External whole-material bulk NMR references are NOT DP3 ground truth.',
           'QC aggregation never uses known bulk NMR or predicted M% to determine weights.',
           'These model-development MAEs are NOT nested post-selection evaluation.',
           'Predictions are clipped at 0..100% only for reporting; clipping frequency is recorded.']
    (OUT_DIR/'V68_EVIDENCE_AUDIT.txt').write_text('\n'.join(report)+'\n')
    make_figures()
    log(f'Compiled {len(files)} fold / {len(external)} external files')



def make_figures():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:return
    target=OUT_DIR/'FIGURES';target.mkdir(exist_ok=True)
    cpath=OUT_DIR/'V68_LORO_SUMMARY.csv'
    epath=OUT_DIR/'V68_EXTERNAL_ALL.csv'
    if cpath.exists():
        tab=pd.read_csv(cpath)
        sub=tab[(tab.feature_mode=='K2_PLUS_K8')&(tab.aggregation=='MEAN')
                &(tab.n_folds==7)].sort_values('MAE_pp').head(12)
        if len(sub):
            labels=[f'{r.preprocessing} / {r.ratio_policy} / {r.ratio_transform} / {r.nmf_mode}'
                    for r in sub.itertuples()]
            fig,ax=plt.subplots(figsize=(11,6))
            ax.barh(np.arange(len(sub)),sub['MAE_pp'].to_numpy()[::-1])
            ax.set_yticks(np.arange(len(sub)),labels[::-1],fontsize=8)
            ax.set_xlabel('Grouped LOCO MAE (M percentage points)')
            ax.set_title('V6.8: 12 lowest development-set errors, DP3 NCE26')
            fig.tight_layout();fig.savefig(target/'CALIBRATION_JOINT_TOP12.png',dpi=170);plt.close(fig)
    if cpath.exists() and epath.exists():
        tab=pd.read_csv(cpath); ext=pd.read_csv(epath)
        idx=['preprocessing','ratio_policy','ratio_transform','nmf_mode','feature_mode','aggregation']
        good=tab[(tab.feature_mode=='K2_PLUS_K8')&(tab.aggregation=='MEAN')&(tab.n_folds==7)]
        winners=good.sort_values('MAE_pp').head(5)
        both=ext.merge(winners[idx],on=idx,how='inner')
        if len(both):
            fig,ax=plt.subplots(figsize=(10,5))
            names=('GG_block_rich','MG_block_rich','MM_block_rich')
            for r in winners.itertuples():
                d=both[(both.preprocessing==r.preprocessing)&
                  (both.ratio_policy==r.ratio_policy)&
                  (both.ratio_transform==r.ratio_transform)&
                  (both.nmf_mode==r.nmf_mode)&(both.feature_mode==r.feature_mode)&
                  (both.aggregation==r.aggregation)]
                vals=[float(d[d['sample']==n]['pred_M_pct'].iloc[0]) if (d['sample']==n).any() else np.nan
                      for n in names]
                ax.plot(range(3),vals,'o-',alpha=.8,
                      label=f'{r.preprocessing} {r.ratio_policy} {r.ratio_transform} {r.nmf_mode}')
            ax.plot(range(3),[NMR_BULK_CONTEXT[n] for n in names],'k--',label='Bulk NMR (DP unresolved)')
            ax.set_xticks(range(3),('GG-rich','MG-rich','MM-rich'))
            ax.set_ylim(0,100);ax.set_ylabel('Predicted M%')
            ax.set_title('Calibration-shortlisted models: external estimates, not DP3 accuracy')
            ax.legend(fontsize=7);fig.tight_layout()
            fig.savefig(target/'CALIBRATION_SHORTLIST_EXTERNAL_CONTEXT.png',dpi=170);plt.close(fig)


def synthetic_test():
    """Feature fitting and selection test with seven synthetic composition groups."""
    rng=np.random.default_rng(2026)
    grid=np.arange(100.,551.,.2)
    centers=np.asarray(base.get_preprocess_centers(DP),float)
    z=[];n=[];y=[]
    sample=[]
    for m in MIXES:
        for r in range(8):
            s=np.zeros(len(grid),float)
            for k,c in enumerate(base.get_ratio_candidate_centers(DP)):
                amplitude=(0.35+1.3*(.5+.5*np.sin(k*.8)) +
                           m*(1.1*np.cos(k*.4)+.4) + rng.normal(0,.018))
                s+=max(.02,amplitude)*np.exp(-.5*((grid-c)/.35)**2)
            s+=.05*np.exp(-.5*((grid-base.DP_TARGET_MZ[DP])/.5)**2)
            z.append(s);n.append(m);y.append(.08+.75*m);sample.append(f'cal_{int(m*100)}')
    X=np.maximum(np.stack(z),0.)
    support=np.ones((len(X),len(centers)),bool)
    pack=dict(dp=DP,nce=NCE,grid=grid,centers=centers,support=support,
              nom=np.asarray(n,float),y=np.asarray(y,float),
              sample=np.asarray(sample),masked={.25:X.astype(np.float32),1.:X.astype(np.float32)})
    train=~np.isclose(pack['nom'],MIXES[0]);test=~train
    for policy in SELECTORS:
        ids,_=choose_ratios(pack,train,X,policy,'H000')
        if len(ids)<K_RATIOS:
            raise AssertionError(f'Synthetic policy {policy} selected only {len(ids)} ratios')
        for transform in TRANSFORMS:
            for nmf_mode,feature in (('NONE','K8_ONLY'),('FULL','K2_PLUS_K8'),
                                     ('NO_PRECURSOR','K2_PLUS_K8')):
                model=fit_one(pack,train,X,'SYNTH',1.,'ORIGINAL7',1.,policy,transform,
                              nmf_mode,feature,ids)
                report=summarize_samples(pack,model,X[test],test,'synthetic',float(pack['y'][test].mean()))
                assert len(report)==4 and all(np.isfinite(r['pred_M_pct']) for r in report)
    xx=nmf_matrix(X,pack,'NO_PRECURSOR')
    assert np.all(xx[:,abs(grid-base.DP_TARGET_MZ[DP])<=base.PEAK_WINDOW]==0)
    # Held-out labels must not influence ratio selection.
    a,_=choose_ratios(pack,train,X,'CONSENSUS','H000')
    pack2=dict(pack,y=pack['y'].copy());pack2['y'][test]=.999
    b,_=choose_ratios(pack2,train,X,'CONSENSUS','H000')
    assert a==b
    log('Synthetic end-to-end passed: four selectors, log ratios, two NMF representations, four aggregations')


def main():
    global DATA_DIR, OUT_DIR, MAX_NMF_ITER
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-mode',choices=('selftest','pilot','overnight','external','report'),default='overnight')
    p.add_argument('--data-dir',type=Path,default=None)
    p.add_argument('--out-dir',type=Path,default=None)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--nmf-max-iter',type=int,default=MAX_NMF_ITER)
    args=p.parse_args()
    if args.data_dir is not None:DATA_DIR=args.data_dir.expanduser().resolve()
    if args.out_dir is not None:OUT_DIR=args.out_dir.expanduser().resolve()
    MAX_NMF_ITER=int(args.nmf_max_iter)
    core.NMF_MAX_ITER=MAX_NMF_ITER
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=max(1,args.threads))
    except ImportError:pass
    if args.run_mode=='selftest':synthetic_test();return
    if args.run_mode=='report':compile_reports();return
    check_setup()
    cal,ext=prepare_cache()
    selected_arms=list(arms())
    if args.run_mode=='pilot':
        selected_arms=[a for a in selected_arms if a[0]=='V66_SELECTED' and
                  a[4] in ('REFERENCE','RAW_SUPPORT') and a[5]=='RAW']
        holds=MIXES[:2]
    elif args.run_mode=='external':holds=[]
    else:holds=MIXES
    log(f'Executing DP3 NCE26: {len(selected_arms)} model configurations')
    for h in holds:
        log(f'Holdout nominal {h*100:g}%')
        train_condition(cal,None,h,selected_arms)
    if ext is not None:
        log('Fitting complete calibration and predicting three external materials')
        # For external mode, evaluate all frozen models, including those from a prior pilot.
        train_condition(cal,ext,None,selected_arms)
    else:log('No external mzML detected; calibration-only outputs available')
    compile_reports()
    log('COMPLETE: '+str(OUT_DIR))


if __name__=='__main__':main()
