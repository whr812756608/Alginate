#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Alginate V6.2 group-scheme × Gaussian × normalization ablation.

Preserves two-component NMF + eight TRAIN-SELECTED ratios; tests 2x2 factors:
  S0/S1: interpolation only / Gaussian sigma=original 1 Da on interpolated m/z grid;
  G0/G1: full-spectrum TIC / original per-fragment-family group-area norm;
  Four group definitions x two smoothing levels x two normalization modes
  for all fourteen architectures: 14 x 4 x 2 x 2 = 224 model arms.
  For targeted CENTER/GUARD/WEIGHT models, both S0/S1 use the SAME
  peak-centered grid extraction; only the smoothing itself changes.
  Raw centroid support is retained as independent QC; not used as a target.

Architecture families:
  BASE:   original smoothing, +/-2 Da masking, group-area normalization;
  CENTER: centroid-informed narrow-mass fragment extraction;
  GUARD:  CENTER with TRAIN-ONLY guarded fragment-family normalization;
  WEIGHT: GUARD with TRAIN-ONLY fragment-reliability-weighted rank-2 NMF;
  COUPLED: energy-consistency penalized K8-only, K2-only, or joint Ridge.

All 7 outer heldout-composition folds withhold that composition at ALL NCE.
NO external NMR is used in training, model, DP/NCE or hyperparameter selection.
External sample metrics are exploratory only; not DP-specific accuracy.

Extract package and run from IDE (edits below) or:
  python alginate_V6_2_GROUP_SCHEMES_OVERNIGHT_IDE.py --run-mode selftest
  python alginate_V6_2_GROUP_SCHEMES_OVERNIGHT_IDE.py --run-mode pilot
  python alginate_V6_2_GROUP_SCHEMES_OVERNIGHT_IDE.py --run-mode pilot-overnight
  python alginate_V6_2_GROUP_SCHEMES_OVERNIGHT_IDE.py --run-mode external
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.decomposition import NMF
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path.home() / "Desktop" / "alginate_experiment"      # EDIT IN IDE
OUT_DIR = ROOT / "V6G_GROUP_SCHEME_RESULTS"
DP_LIST = (3, 4)
ENERGIES = (18., 22., 26.)
MIXES = (0., .2, .4, .5, .6, .8, 1.)
SEED = 20261019
RIDGE_ALPHA = 1.0
COUPLE_LAMBDA = 0.25         # pre-specified; NO external tuning
NMF_MAX_ITER = 800
NMF_TOL = 1e-4
CENTROID_PPM = 30.          # matched raw m/z tolerance; NOT a 2-Da window
CENTROID_MIN_DA = 0.02
GUARD_FLOOR_FRAC = 0.25     # train-derived family relative-abundance floor
GUARD_FLOOR_MIN = 0.003
ODD_MIN_FRACTION = 0.50
USE_CACHE = True
EPS = 1e-12
SCRIPT_VERSION = '6.2.0-group-scheme' 
CORE_ARMS = ('BASE_K8','BASE_K2_ONLY','BASE_K2_K8',
             'CENTER_K8','CENTER_K2_ONLY','CENTER_K2_K8',
             'GUARD_K8','GUARD_K2_ONLY','GUARD_K2_K8',
             'WEIGHT_K2_ONLY','WEIGHT_K2_K8',
             'COUPLED_GUARD_K8','COUPLED_GUARD_K2_ONLY','COUPLED_GUARD_K2_K8')
# GROUP SCHEMES: m/z intervals (rather than data-fitted clustering). Fixed a priori;
# no external sample labels, predictions or bulk NMR are used to construct groups.
# ORIGINAL7 is taken byte-for-byte from the original manuscript code at runtime.
# Other schemes tile 100–1300 Da without overlapping boundary values.
GROUP_SCHEME_NAMES = ('ORIGINAL7','TILED7','COARSE3','LADDER5')
ORIGINAL_GROUP_BOUNDARIES = [(100,200),(300,370),(460,530),(580,710),
                             (830,880),(1030,1060),(1180,1240)]
SELECTED_SCHEMES = GROUP_SCHEME_NAMES      # --group-schemes overrides for a short pilot

def tiled(cuts):
    return [(float(lo)+(0.0001 if i else 0.),float(hi))
            for i,(lo,hi) in enumerate(zip(cuts[:-1],cuts[1:]))]

def group_definitions(base=None):
    original = ORIGINAL_GROUP_BOUNDARIES if base is None else base.GROUP_BOUNDARIES
    return {'ORIGINAL7': [tuple(map(float,b)) for b in original],
            # Seven contiguous fragment ranges, includes ions in ORIGINAL7 gaps.
            'TILED7': tiled((100,250,415,555,770,955,1120,1300)),
            # Three broad bins for low-mass, medium-mass, high-mass ions.
            'COARSE3': tiled((100,370,710,1300)),
            # Five blocks roughly aligned with fragment-ion mass ladders.
            'LADDER5': tiled((100,215,410,575,790,1300))}

def make_factors():
    return tuple(f'S{sm}_G{gn}_{scheme}' for sm in (0,1) for gn in (0,1)
                 for scheme in SELECTED_SCHEMES)

FACTORS = make_factors()
ARMS = tuple(f'{arm}__{factor}' for arm in CORE_ARMS for factor in FACTORS)

def factor_parts(factor):
    s,g,scheme=factor.split('_',2)
    if scheme not in SELECTED_SCHEMES:raise ValueError(f'Unknown group scheme: {scheme}')
    return s=='S1',g=='G1',scheme

def original_baseline():
    return 'BASE_K2_K8__S1_G1_ORIGINAL7'

def model_arm(core,factor):
    return f'{core}__{factor}'


def log(message):
    print(f'[{time.strftime("%H:%M:%S")}] [V6.2] {message}', flush=True)


def get_base():
    try:
        import alginate_ratio_transfer_IDE as base
    except Exception as e:
        raise ImportError('The original alginate preprocessing dependency is unavailable. '
                          'Install requirements.txt, then rerun. Original: '+str(e)) from e
    return base


def json_default(v):
    if isinstance(v, Path): return str(v)
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,(np.integer,)):return int(v)
    if isinstance(v,(np.floating,)):return float(v)
    if isinstance(v,(np.bool_,)):return bool(v)
    raise TypeError(type(v).__name__)


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if rows: pd.DataFrame(rows).to_csv(path,index=False)


def atomic_dump(obj,path):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    joblib.dump(obj,tmp,compress=2)
    os.replace(tmp,path)


def run_signature():
    cfg={'version':SCRIPT_VERSION,'seed':SEED,'ridge_alpha':RIDGE_ALPHA,
         'couple_lambda':COUPLE_LAMBDA,'nmf_iter':NMF_MAX_ITER,
         'centroid_ppm':CENTROID_PPM,'centroid_min_da':CENTROID_MIN_DA,
         'guard_frac':GUARD_FLOOR_FRAC,'guard_floor_min':GUARD_FLOOR_MIN,
         'energies':ENERGIES,'mixes':MIXES,'factorial':FACTORS,'arms':CORE_ARMS,
         'group_bounds':group_definitions(),'schemes':SELECTED_SCHEMES}
    return hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()[:14]


def all_input_files(dp,base,external=False):
    if external:
        return {s:DATA_DIR/f for s,f in base.EXTERNAL_FILES.items()}
    return {f'cal_{r}':DATA_DIR/f'Raw_F{dp}_M{int(r*100)}G{100-int(r*100)}.mzML' for r in MIXES}


def input_fingerprint(paths):
    data=[]
    for label,path in sorted(paths.items()):
        if not path.exists():raise FileNotFoundError(f'Missing {label}: {path}')
        st=path.stat();data.append((label,str(path.resolve()),st.st_size,st.st_mtime_ns))
    return hashlib.sha256(repr(data).encode()).hexdigest()[:16]


def check_mixes(nom):
    g=np.sort(np.unique(np.round(nom,7)))
    if len(g)!=7 or not np.allclose(g,MIXES):
        raise ValueError(f'Expected 7 nominal mixtures: {MIXES}; got {g}')


def raw_centroid_peaks(mz,intens,centers):
    """High-resolution raw-ion maxima, no smoothing or group normalization.

    A matched centroid must fall within max(0.02 Da, 30 ppm). This cannot
    magically recover missing ions. Every selected ratio gets raw-support QC.
    """
    mz=np.asarray(mz,float);intens=np.asarray(intens,float)
    if mz.size==0:return np.zeros(len(centers)),np.zeros(len(centers),bool),np.full(len(centers),np.nan)
    order=np.argsort(mz);mz=mz[order];intens=np.maximum(intens[order],0.)
    vals=np.zeros(len(centers));ok=np.zeros(len(centers),bool)
    delta=np.full(len(centers),np.nan)
    for j,c in enumerate(centers):
        tol=max(CENTROID_MIN_DA,CENTROID_PPM*1.e-6*float(c))
        lo=np.searchsorted(mz,c-tol,side='left');hi=np.searchsorted(mz,c+tol,side='right')
        if hi>lo:
            ii=lo+int(np.argmax(intens[lo:hi]))
            vals[j]=intens[ii];ok[j]=vals[j]>0
            delta[j]=mz[ii]-c
    return vals,ok,delta


def fragment_groups(centers,boundaries):
    groups=[]
    for c in centers:
        val=next((i for i,(a,b) in enumerate(boundaries) if a <= c <= b),-1)
        groups.append(val)
    return np.asarray(groups,int)


def fragment_family_fractions(P,groups):
    P=np.maximum(np.asarray(P,float),0.)
    total=np.maximum(P.sum(axis=1),EPS)
    return np.column_stack([P[:,groups==i].sum(axis=1)/total
                            for i in sorted(set(groups)) if i>=0])


def fit_guard_floor(P,groups,train):
    """Guard thresholds fit to training scans only, never external or test folds."""
    tr=np.asarray(train,bool)
    if not tr.any():raise ValueError('Empty guard-training subset')
    fractions=fragment_family_fractions(P[tr],groups)
    present_ids=sorted(set(groups)-{-1})
    floors={}
    for j,k in enumerate(present_ids):
        v=fractions[:,j]
        detected=v[v>0]
        if detected.size:
            floors[k]=max(GUARD_FLOOR_MIN,GUARD_FLOOR_FRAC*float(np.quantile(detected,0.10)))
        else:
            floors[k]=GUARD_FLOOR_MIN
    return floors


def normalize_targeted(P,groups,guard_floors=None,group_norm=True):
    """Targets: G1 within-fragment-family normalization; G0 global TIC.

    GUARD retains train-only low-family suppression in BOTH G levels.
    Guard factors are computed from pre-normalized family fractions.
    Thus the paired G0/G1 contrast alters only the normalization step.
    """
    P=np.maximum(np.asarray(P,float),0.)
    tot=np.maximum(P.sum(axis=1,keepdims=True),EPS)
    fractions=P/tot
    out=fractions.copy()
    for g in sorted(set(groups)-{-1}):
        inds=groups==g
        family=fractions[:,inds].sum(axis=1,keepdims=True)
        if group_norm:
            out[:,inds]=fractions[:,inds]/np.where(family>EPS,family,1.)
        if guard_floors is not None:
            floor=float(guard_floors[g])
            out[:,inds]*=family/(family+floor)
    return np.maximum(out,0.)


def reliability_weights(X,nom,train):
    """Unsupervised, TRAIN-composition-derived stability/discriminability.

    Weight construction does not read external sample labels. Weights are
    clipped to avoid making a weak single fragment dominate the NMF fit.
    """
    x=np.maximum(np.asarray(X[train],float),0.)
    groups=np.asarray(nom[train]); levels=np.unique(groups)
    means=np.vstack([x[np.isclose(groups,g)].mean(axis=0) for g in levels])
    sds=np.vstack([x[np.isclose(groups,g)].std(axis=0) for g in levels])
    between=means.std(axis=0)
    within=np.median(sds,axis=0)
    floor=max(float(np.median(within))*0.25,1e-5)
    effect=between/(within+floor)
    detection=np.vstack([(x[np.isclose(groups,g)]>1e-6).mean(axis=0) for g in levels]).min(axis=0)
    raw=np.sqrt(np.maximum(effect,0.)+0.05)*np.sqrt(np.maximum(detection,0.05))
    w=np.clip(raw,0.4,2.5)
    return w/np.maximum(w.mean(),EPS)


def scaler_weighted_fit(scaler,X,weights):
    try: scaler.fit(X,sample_weight=weights)
    except TypeError:scaler.fit(X)
    return scaler


@dataclass
class Rank2:
    nmf:object
    active:np.ndarray
    feature_weights:np.ndarray
    order:np.ndarray
    mass:np.ndarray
    resid_q99:float
    raw_coeff:bool=False

    def features_residual(self,X):
        x=np.asarray(X,float)
        weighted=np.maximum(x[:,self.active],0.)*self.feature_weights[None,:]
        c=np.maximum(self.nmf.transform(weighted),0.)
        b=np.maximum(self.nmf.components_,0.)
        recon=c@b
        residual=np.linalg.norm(weighted-recon,axis=1)/np.maximum(np.linalg.norm(weighted,axis=1),EPS)
        if self.raw_coeff:
            feats=c[:,self.order]
        else:
            contribution=c*self.mass[None,:]
            total=contribution.sum(axis=1)
            share=contribution[:,self.order[0]]/np.maximum(total,EPS)
            feats=np.column_stack([share,np.log1p(total)])
        return feats,residual


def fit_rank2(X,y,nom,train,weighted=False,raw_coeff=False):
    active=np.any(X[train]>1e-15,axis=0)
    if active.sum()<2:raise ValueError('Fewer than two active spectral features')
    fweights=(reliability_weights(X[:,active],nom,train) if weighted else np.ones(active.sum()))
    z=np.maximum(X[train][:,active],0.)*fweights[None,:]
    nmf=NMF(n_components=2,init='nndsvda',solver='cd',max_iter=NMF_MAX_ITER,
            tol=NMF_TOL,random_state=SEED)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        coef=nmf.fit_transform(z)
    mass=np.maximum(nmf.components_.sum(axis=1),EPS)
    contrib=coef*mass[None,:]
    share=contrib[:,0]/np.maximum(contrib.sum(axis=1),EPS)
    c=float(np.corrcoef(share,y[train])[0,1]) if np.std(share)>EPS and np.std(y[train])>EPS else 0.
    order=np.array([0,1] if c>=0 else [1,0],int)
    mdl=Rank2(nmf,active,fweights,order,mass,0.,raw_coeff)
    _,res=mdl.features_residual(X[train])
    mdl.resid_q99=float(np.quantile(res,.99))
    return mdl


@dataclass
class Backbone:
    variant:str
    factor:str
    dp:int
    nce:float
    grid:np.ndarray
    centers:np.ndarray
    ratio_columns:np.ndarray
    ratio_names:tuple
    ratio_info:tuple
    floors:dict | None
    nmf:Rank2
    scaler_k8:object
    scaler_k2:object
    train_f_k8:np.ndarray
    train_f_k2:np.ndarray
    scaler_nmf:object
    train_f_nmf:np.ndarray
    k8_valid:bool

    def matrix(self,pack):
        smoothing,group_norm,scheme=factor_parts(self.factor)
        if self.variant=='BASE':return np.asarray(pack['full_by_factor'][self.factor],float)
        P=np.asarray(pack['center_by_smoothing']['S1' if smoothing else 'S0'],float)
        floors=self.floors if self.variant in ('GUARD','WEIGHT') else None
        return normalize_targeted(P,pack['groups_by_scheme'][scheme],floors,group_norm=group_norm)

    def design(self,pack,feature_mode):
        mat=self.matrix(pack)
        if feature_mode=='K2_ONLY':
            A,_=self.nmf.features_residual(mat)
            return self.scaler_nmf.transform(A)
        if not self.k8_valid:
            raise ValueError('Eight reliable calibration-selected ratios unavailable')
        from alginate_ratio_transfer_IDE import build_ratio_matrices
        ratios,_,_,_=build_ratio_matrices(mat,self.grid,list(self.centers))
        R=ratios[:,self.ratio_columns]
        if feature_mode=='K8':return self.scaler_k8.transform(R)
        A,_=self.nmf.features_residual(mat)
        if feature_mode=='K2_K8':
            return self.scaler_k2.transform(np.column_stack([A,R]))
        raise ValueError(f'Unknown feature mode {feature_mode}')

    def qc(self,pack):
        mat=self.matrix(pack)
        _,res=self.nmf.features_residual(mat)
        return {'nmf_ood_pct':float((res>self.nmf.resid_q99).mean()*100),
                'nmf_resid_mean':float(np.mean(res)),
                'nmf_calibration_q99':float(self.nmf.resid_q99)}


def group_weights(nom):
    nom=np.asarray(nom)
    w=np.zeros(len(nom),float)
    for g in np.unique(nom):
        m=np.isclose(nom,g);w[m]=1.0/max(int(m.sum()),1)
    w*=len(w)/np.maximum(w.sum(),EPS)
    return w


def fit_backbone(pack,train,base,variant,factor):
    """Every learned object is fit on TRAIN-only rows."""
    y=pack['y'];nom=pack['nom'];dp=pack['dp'];nce=pack['nce']
    smoothing,group_norm,scheme=factor_parts(factor)
    if variant=='BASE':
        mat=np.asarray(pack['full_by_factor'][factor],float);grid=pack['grid']
        floors=None
    else:
        grid=pack['centers']
        source=pack['center_by_smoothing']['S1' if smoothing else 'S0']
        floors=fit_guard_floor(source,pack['groups_by_scheme'][scheme],train) if variant in ('GUARD','WEIGHT') else None
        mat=normalize_targeted(source,pack['groups_by_scheme'][scheme],floors,group_norm=group_norm)
    ratio_centers=np.asarray(base.get_ratio_candidate_centers(dp),float)
    R,D,names,info=base.build_ratio_matrices(mat,grid,list(ratio_centers))
    ranked,_=base.rank_ratio_candidates(R[train],D[train],y[train],nom[train],names,info)
    selected=base.select_nonredundant_ratios(ranked,8)
    valid_k8=len(selected)==8
    if not valid_k8:
        # NMF-only is still scientifically valid when ratio QC rejects K8.
        # Do not relax K8 selection thresholds to rescue external agreement.
        selected=[]
    columns=np.asarray([int(row['idx']) for row in selected],int)
    nmf=fit_rank2(mat,y,nom,train,weighted=variant=='WEIGHT',raw_coeff=variant=='BASE')
    nmf_features,_=nmf.features_residual(mat[train])
    Rtr=R[train][:,columns]
    weights=group_weights(nom[train])
    scaler_nmf=scaler_weighted_fit(StandardScaler(),nmf_features,weights)
    # K8 and K2+K8 are unfit when 8 selected ratios fail calibration-only QC.
    if valid_k8:
        feat_k2=np.column_stack([nmf_features,Rtr])
        scaler8=scaler_weighted_fit(StandardScaler(),Rtr,weights)
        scaler2=scaler_weighted_fit(StandardScaler(),feat_k2,weights)
        x8=scaler8.transform(Rtr);x2=scaler2.transform(feat_k2)
    else:
        scaler8=None;scaler2=None;x8=None;x2=None
    return Backbone(variant,factor,dp,nce,np.asarray(grid,float),ratio_centers,columns,
                    tuple(row['name'] for row in selected),tuple((row['p1'],row['p2']) for row in selected),
                    floors,nmf,scaler8,scaler2,x8,x2,
                    scaler_nmf,scaler_nmf.transform(nmf_features),valid_k8)


@dataclass
class Predictor:
    arm:str
    backbone:Backbone
    ridge:object | None
    coef:np.ndarray | None
    feature_mode:str

    def predict_raw(self,pack):
        f=self.backbone.design(pack,self.feature_mode)
        if self.coef is None:return self.ridge.predict(f)
        return np.column_stack([np.ones(len(f)),f])@self.coef


def pack_subset(pack,mask):
    """Subset scans, including nested factorial arrays (no metadata slicing)."""
    n=len(pack['y'])
    def sub(v):
        if isinstance(v,dict):return {k:sub(z) for k,z in v.items()}
        return v[mask] if isinstance(v,np.ndarray) and v.ndim>0 and len(v)==n else v
    return {k:sub(v) for k,v in pack.items()}


def feature_mode_for_core(core):
    if core.endswith('_K2_ONLY'):return 'K2_ONLY'
    if core.endswith('_K2_K8'):return 'K2_K8'
    if core.endswith('_K8'):return 'K8'
    raise ValueError(f'Unrecognized architecture: {core}')


def fit_single(backbone,pack,train,arm,feature_mode):
    if feature_mode!='K2_ONLY' and not backbone.k8_valid:
        raise ValueError('Eight reliable train-selected ratios unavailable')
    x={'K8':backbone.train_f_k8,
       'K2_ONLY':backbone.train_f_nmf,
       'K2_K8':backbone.train_f_k2}[feature_mode]
    weights=group_weights(pack['nom'][train])
    ridge=Ridge(alpha=RIDGE_ALPHA)
    ridge.fit(x,pack['y'][train],sample_weight=weights)
    return Predictor(arm,backbone,ridge,None,feature_mode)


def coupled_solver(backbones,packs,train_mask,feature_mode,lam):
    """Convex multi-energy Ridge plus composition-level energy disagreement.

    Fitted separately for each DP, using only training compositions at all
    energies. Penalty uses calibration composition-level averages, not external.
    The common target y for each composition is supplied by its own label.
    """
    energies=sorted(backbones)
    mats={e:{'K8':backbones[e].train_f_k8,
             'K2_ONLY':backbones[e].train_f_nmf,
             'K2_K8':backbones[e].train_f_k2}[feature_mode] for e in energies}
    if any(x is None for x in mats.values()):
        raise ValueError('Eight training-selected peak ratios unavailable at one or more energies')
    p=mats[energies[0]].shape[1]+1
    if any(mats[e].shape[1]+1!=p for e in energies):raise ValueError('Inconsistent coupled feature dimensions')
    total=p*len(energies)
    block=[];yvec=[];wvec=[]
    levels=np.sort(np.unique(packs[energies[0]]['nom'][train_mask[energies[0]]]))
    for i,e in enumerate(energies):
        pack=packs[e];tr=train_mask[e];Z=np.column_stack([np.ones(len(mats[e])),mats[e]])
        rows=np.zeros((len(Z),total));rows[:,i*p:(i+1)*p]=Z
        block.append(rows);yvec.append(pack['y'][tr]);wvec.append(group_weights(pack['nom'][tr]))
    XX=np.vstack(block); yy=np.concatenate(yvec); w=np.concatenate(wvec)
    pieces=[XX*np.sqrt(w[:,None])]; targets=[yy*np.sqrt(w)]
    reg=np.zeros((len(energies)*(p-1),total));j=0
    for i in range(len(energies)):
        for k in range(1,p): reg[j,i*p+k]=np.sqrt(RIDGE_ALPHA);j+=1
    pieces.append(reg);targets.append(np.zeros(len(reg)))
    if lam>0:
        gamma=lam*len(yy)/max(len(levels),1)/max(len(list(combinations(energies,2))),1)
        penalties=[]
        for g in levels:
            means={}
            for e in energies:
                mask_g=np.isclose(packs[e]['nom'][train_mask[e]],g)
                means[e]=np.r_[1.,mats[e][mask_g].mean(axis=0)]
            for e1,e2 in combinations(energies,2):
                row=np.zeros(total)
                row[energies.index(e1)*p:(energies.index(e1)+1)*p]=means[e1]
                row[energies.index(e2)*p:(energies.index(e2)+1)*p]=-means[e2]
                penalties.append(row*np.sqrt(gamma))
        if penalties:
            pieces.append(np.asarray(penalties));targets.append(np.zeros(len(penalties)))
    coef=np.linalg.lstsq(np.vstack(pieces),np.concatenate(targets),rcond=None)[0]
    return {e:coef[i*p:(i+1)*p] for i,e in enumerate(energies)}


def train_all(packs,train_mask,base):
    """Fit all 14 architectures under four group definitions, both S/G levels.

    Training uses calibration-composition folds only. Each QC rejection
    records its actual reason; no external bulk label is used to rescue it.
    The original S1_G1 baseline must succeed so controls remain traceable.
    """
    models={}; failures={}
    # BASE and CENTER have mathematically group-independent G0 matrices.
    # Reuse their fitted backbones for all G0 scheme labels instead of
    # fitting numerically identical 2-component NMF models four times.
    backbone_cache={}
    for factor in FACTORS:
        back={}
        for variant in ('BASE','CENTER','GUARD','WEIGHT'):
            back[variant]={}
            for e,pack in sorted(packs.items()):
                cores={'BASE':('BASE_K8','BASE_K2_ONLY','BASE_K2_K8'),
                       'CENTER':('CENTER_K8','CENTER_K2_ONLY','CENTER_K2_K8'),
                       'GUARD':('GUARD_K8','GUARD_K2_ONLY','GUARD_K2_K8'),
                       'WEIGHT':('WEIGHT_K2_ONLY','WEIGHT_K2_K8',)}[variant]
                try:
                    sm,gn,scheme=factor_parts(factor)
                    shared_scheme='ORIGINAL7' if (not gn and variant in ('BASE','CENTER')) else scheme
                    cache_key=(variant,sm,gn,shared_scheme,e)
                    if cache_key not in backbone_cache:
                        reuse_factor=f'S{int(sm)}_G{int(gn)}_{shared_scheme}'
                        backbone_cache[cache_key]=fit_backbone(pack,train_mask[e],base,variant,reuse_factor)
                    bk=backbone_cache[cache_key]
                    back[variant][e]=bk
                    for core in cores:
                        arm=model_arm(core,factor)
                        try:
                            models[(arm,e)]=fit_single(bk,pack,train_mask[e],arm,
                                                       feature_mode_for_core(core))
                        except (ValueError,RuntimeError,ArithmeticError) as exc:
                            failures[(arm,e)]=str(exc)
                except (ValueError,RuntimeError,ArithmeticError) as exc:
                    if variant=='BASE' and factor=='S1_G1_ORIGINAL7':
                        raise RuntimeError('Original manuscript baseline S1_G1 fit failed. '+str(exc)) from exc
                    for core in cores: failures[(model_arm(core,factor),e)]=str(exc)
                    log(f'  QC arm rejection: {variant}/{factor}/NCE{e:g}: {exc}')
        for feature_mode,core in (('K8','COUPLED_GUARD_K8'),
                                  ('K2_ONLY','COUPLED_GUARD_K2_ONLY'),
                                  ('K2_K8','COUPLED_GUARD_K2_K8')):
            arm=model_arm(core,factor)
            if len(back['GUARD'])!=len(ENERGIES):
                for e in ENERGIES:failures[(arm,e)]='Guarded inputs failed at one or more collision energies'
                continue
            try:
                coef=coupled_solver(back['GUARD'],packs,train_mask,feature_mode,COUPLE_LAMBDA)
                for e in packs:
                    models[(arm,e)]=Predictor(arm,back['GUARD'][e],None,coef[e],feature_mode)
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                for e in ENERGIES:failures[(arm,e)]=str(exc)
                log(f'  Coupled arm rejected: {arm}: {exc}')
    return models,failures


def grid_scan_factorial(spec,dp,proc,base,relevant,centers,raw_support):
    """Extract S0 and S1 from an IDENTICAL interpolated m/z grid.

    BASE S1_G1_ORIGINAL7 is the original smooth->mask->group-area normalization.
    BASE G0 means global TIC on the very same masked spectrum.
    CENTER/GUARD/WEIGHT use peak maxima in the same tight m/z extraction
    tolerance on the grid with/without smoothing, with absent raw ions zeroed.
    No Gaussian is applied to a sparse centroid array directly.
    """
    from DP_spectrum_process_Class import FractionSpectra,filter_peaks_by_centers
    raw_mz=np.asarray(spec['m/z array'],np.float32)
    raw_int=np.asarray(spec['intensity array'],np.float32)
    mask=raw_mz<=base.DP_TARGET_MZ[dp]+20.
    mz=raw_mz[mask];inten=raw_int[mask]
    if len(mz)<4 or not np.isfinite(inten).all():return None
    mzgrid,interp,_,_=FractionSpectra.interpolate_to_grid_without_norm(mz,inten,proc.grid)
    prof={'S0':interp,'S1':proc._smooth_impl(mzgrid,interp)}
    full={};targets={}
    for smooth_name,profile in prof.items():
        masked=filter_peaks_by_centers(mzgrid,profile,base.get_preprocess_centers(dp),window=base.PEAK_WINDOW)
        masked=np.maximum(np.asarray(masked,np.float32),0.)
        g0=masked / max(float(masked.sum()),EPS)
        for scheme,bounds in group_definitions(base).items():
            if scheme not in SELECTED_SCHEMES:continue
            full[f'{smooth_name}_G0_{scheme}']=np.asarray(np.maximum(g0[relevant],0),np.float32)
            g1=FractionSpectra.group_area_normalization(mzgrid,masked,bounds)
            full[f'{smooth_name}_G1_{scheme}']=np.asarray(np.maximum(g1[relevant],0),np.float32)
        arr=np.zeros(len(centers),np.float32)
        for j,c in enumerate(centers):
            if not raw_support[j]:continue
            tol=max(CENTROID_MIN_DA,CENTROID_PPM*1e-6*float(c))
            lo=np.searchsorted(mzgrid,c-tol,'left')
            hi=np.searchsorted(mzgrid,c+tol,'right')
            if hi>lo:arr[j]=max(float(np.max(profile[lo:hi])),0.)
        targets[smooth_name]=arr
    return full,targets


def build_condition_cache(dp,external=False):
    """Read all three collision energies once per raw mzML file for one DP.

    S1_G1_ORIGINAL7 BASE reproduces original scan-level interpolation, Gaussian smoothing,
    +/-2-Da masking, and group-area normalization. Factors S0 and G0 disable
    these operations separately. The other model families use the SAME tight
    grid-extraction window before/after smoothing, with raw centroid support QC.
    This group-scheme cache is intentionally separate from prior V6 checkpoints.
    """
    from DP_spectrum_process_Class import DPSpectrumProcessor, FractionSpectra
    from pyteomics import mzml
    base=get_base()
    paths=all_input_files(dp,base,external)
    paths={k:v for k,v in paths.items() if v.exists()} if external else paths
    if not paths:
        if external:return {}
        raise FileNotFoundError(f'No DP{dp} mzMLs in {DATA_DIR}')
    assert group_definitions(base)['ORIGINAL7']==group_definitions()['ORIGINAL7'], (
        'Original group boundaries changed in alginate_ratio_transfer_IDE.py: inspect and update V6.2')
    fingerprint=input_fingerprint(paths)
    cache_dir=OUT_DIR/'CACHE';cache_dir.mkdir(parents=True,exist_ok=True)
    cachefile=cache_dir/f'{"external" if external else "calibration"}_DP{dp}_group_schemes.joblib'
    if USE_CACHE and cachefile.exists():
        old=joblib.load(cachefile)
        if old.get('fingerprint')==fingerprint and old.get('signature')==run_signature():
            log(f'Loaded cached {"external" if external else "calibration"} DP{dp}')
            return old['packs']
        log(f'Invalidating stale DP{dp} cache')
    proc=DPSpectrumProcessor(conv_sigma=base.CONV_SIGMA,default_groups=base.GROUP_BOUNDARIES)
    centers=np.asarray(base.get_preprocess_centers(dp),float)
    cands=np.asarray(base.get_ratio_candidate_centers(dp),float)
    # Preserve all originally nonzero bins after the DP-dependent clip.
    grid=proc.grid
    relevant=(grid>=base.CLIP_LO)&(grid<=base.DP_TARGET_MZ[dp]+5.)
    relevant&=np.any(np.abs(grid[:,None]-centers[None,:])<=base.PEAK_WINDOW+1e-8,axis=1)
    gsub=np.asarray(grid[relevant],float)
    groups_by_scheme={sch:fragment_groups(centers,b) for sch,b in group_definitions(base).items() if sch in SELECTED_SCHEMES}
    rows={e:defaultdict(list) for e in ENERGIES}
    for label,path in sorted(paths.items()):
        log(f'  Reading {path.name}, DP{dp} ({"external" if external else "calibration"})')
        r=float(label[4:]) if not external else None
        y=(FractionSpectra(dp,r,dp,proc.grid).NMR_MG_ratio if not external else np.nan)
        if not external and y is None:raise ValueError(f'No NMR calibration label for DP{dp} mixture {r}')
        with mzml.read(str(path)) as reader:
            for spec in reader:
                if spec.get('ms level')!=2 or 'precursorList' not in spec:continue
                try:
                    pre=spec['precursorList']['precursor'][0]
                    mz_target=pre.get('isolationWindow',{}).get('isolation window target m/z')
                    nce=pre.get('activation',{}).get('collision energy')
                    if mz_target is None or nce is None:continue
                    if abs(float(mz_target)-base.DP_TARGET_MZ[dp])>=1e-4:continue
                    e=next((ee for ee in ENERGIES if abs(float(nce)-ee)<1e-5),None)
                    if e is None:continue
                    mz=np.asarray(spec['m/z array'],float); inten=np.asarray(spec['intensity array'],float)
                    keep=mz<=base.DP_TARGET_MZ[dp]+20
                    raw,support,delta=raw_centroid_peaks(mz[keep],inten[keep],centers)
                    processed=grid_scan_factorial(spec,dp,proc,base,relevant,centers,support)
                    if processed is None:continue
                    full,center=processed
                    if np.sum(full['S1_G1_ORIGINAL7'])<=EPS:continue
                    rec=rows[e]
                    for factor in FACTORS:
                        # All G0 preprocessing is TIC and identical across schemes.
                        if '_G0_' in factor and not factor.endswith('_ORIGINAL7'):continue
                        rec[f'full_{factor}'].append(full[factor])
                    for sm in ('S0','S1'):rec[f'center_{sm}'].append(center[sm])
                    rec['raw'].append(raw.astype(np.float32))
                    rec['support'].append(support);rec['delta'].append(delta.astype(np.float32))
                    rec['nom'].append(float(r) if not external else np.nan)
                    rec['y'].append(float(y) if not external else np.nan)
                    rec['sample'].append(label)
                except (ValueError,KeyError,IndexError,TypeError,RuntimeError) as exc:
                    # Malformed scans are logged in summary; not substituted.
                    rows[e]['preprocess_failed'].append(str(exc)[:100])
    packs={}
    for e,rr in rows.items():
        if not rr['full_S1_G1_ORIGINAL7']:
            if not external:raise RuntimeError(f'No calibration scans DP{dp}/NCE{e}')
            continue
        unique_full={factor:np.asarray(rr[f'full_{factor}'],np.float32)
                     for factor in FACTORS
                     if ('_G0_' not in factor or factor.endswith('_ORIGINAL7'))}
        for factor in FACTORS:
            if '_G0_' in factor and factor not in unique_full:
                sm,gn,scheme=factor_parts(factor)
                unique_full[f'S{int(sm)}_G0_{scheme}']=unique_full[f'S{int(sm)}_G0_ORIGINAL7']
        pack={'dp':int(dp),'nce':float(e),
              'full_by_factor':unique_full,
              'center_by_smoothing':{sm:np.asarray(rr[f'center_{sm}'],np.float32) for sm in ('S0','S1')},
              'grid':gsub,'raw':np.asarray(rr['raw'],np.float32),
              'support':np.asarray(rr['support'],bool),'delta':np.asarray(rr['delta'],np.float32),
              'centers':centers,'groups_by_scheme':groups_by_scheme,'y':np.asarray(rr['y'],float),
              'nom':np.asarray(rr['nom'],float),'sample':np.asarray(rr['sample'],str),
              'preprocess_failed':len(rr['preprocess_failed'])}
        if not external:check_mixes(pack['nom'])
        packs[e]=pack
        log(f'  DP{dp}/NCE{int(e)}: {len(pack["y"])} scans; {len(gsub)} full-spectrum bins; raw ions {len(centers)}')
    atomic_dump({'fingerprint':fingerprint,'signature':run_signature(),'packs':packs},cachefile)
    return packs


def score_one(models,failures,packs,masks,dp,heldout,out):
    results=[];frag=[]
    for arm in ARMS:
        for e in ENERGIES:
            model=models.get((arm,e))
            if model is None:
                results.append(dict(dp=dp,nce=e,heldout_nominal_pct=int(round(heldout*100)),
                                    model=arm,status='QC_REJECTED',reason=failures.get((arm,e),'Not trained'),
                                    n_scans=0,true_M_pct=np.nan,pred_M_pct=np.nan,abs_error_pp=np.nan))
                continue
            pack=packs[e];te=~masks[e]
            probe=pack_subset(pack,te)
            pr=model.predict_raw(probe)
            pred=np.clip(pr,0,1)
            gt=float(np.mean(pack['y'][te]));mean=float(pred.mean())
            metric=dict(dp=dp,nce=e,heldout_nominal_pct=int(round(heldout*100)),model=arm,
                        status='OK',reason='',n_scans=len(pred),true_M_pct=gt*100,pred_M_pct=mean*100,
                        abs_error_pp=abs(mean-gt)*100,scan_sd_pp=float(pred.std()*100),
                        clipped_scan_pct=float(np.mean((pr<0)|(pr>1))*100))
            metric.update(model.backbone.qc(probe))
            results.append(metric)
            for name,(nu,de) in zip(model.backbone.ratio_names if model.feature_mode!='K2_ONLY' else (),
                                     model.backbone.ratio_info if model.feature_mode!='K2_ONLY' else ()):
                frag.append(dict(dp=dp,nce=e,heldout_nominal_pct=int(round(heldout*100)),
                                 model=arm,ratio=name,num_mz=nu,den_mz=de))
    for r in results:
        r['group_scheme']=r['model'].split('__')[1].split('_',2)[2]
        r['architecture']=r['model'].split('__')[0]
        r['gaussian_smoothing']=int('S1' in r['model'].split('__')[-1])
        r['group_normalization']=int('G1' in r['model'].split('__')[-1])
    write_csv(results,out/f'fold_{int(round(heldout*100)):03d}.csv')
    write_csv(frag,out/f'ratios_{int(round(heldout*100)):03d}.csv')
    return results


def run_cv(dp,packs,base):
    folder=OUT_DIR/f'DP{dp}';folder.mkdir(parents=True,exist_ok=True)
    sig_file=folder/'RUN_SIGNATURE.txt'
    if sig_file.exists() and sig_file.read_text().strip()!=run_signature():
        raise RuntimeError(f'Conflicting configuration in {folder}; use another output directory')
    sig_file.write_text(run_signature())
    for i,heldout in enumerate(MIXES):
        cp=folder/f'fold_{int(round(heldout*100)):03d}.csv'
        if cp.exists() and USE_CACHE:
            prior=folder/'RUN_SIGNATURE.txt'
            if not prior.exists() or prior.read_text().strip()!=run_signature():
                raise RuntimeError('Existing CV checkpoints have different settings. '
                                   'Use a new --out-dir or remove old checkpoints; never mix runs.')
            old=pd.read_csv(cp)
            if set(old['model'])==set(ARMS) and len(old)==len(ARMS)*len(ENERGIES):
                log(f'  Resume skip DP{dp} heldout {int(heldout*100)}%')
                continue
        masks={e:~np.isclose(packs[e]['nom'],heldout) for e in ENERGIES}
        if any(np.unique(packs[e]['nom'][masks[e]]).size!=6 for e in ENERGIES):
            raise AssertionError('LORO leakage/invalid composition counts')
        start=time.monotonic()
        models,failures=train_all(packs,masks,base)
        rows=score_one(models,failures,packs,masks,dp,heldout,folder)
        tiny=next(r for r in rows if r['model']=='COUPLED_GUARD_K2_K8__S1_G1_ORIGINAL7' and r['nce']==26.)
        err=tiny['abs_error_pp']
        log(f'  DP{dp} heldout {int(heldout*100):3d}% done {time.monotonic()-start:.0f}s; '
            f'coupled K2+K8 NCE26 err={err:.2f} pp')
    fullmask={e:np.ones(len(pack['y']),bool) for e,pack in packs.items()}
    modelpath=folder/'FROZEN_full_calibration_models.joblib'
    if not (modelpath.exists() and USE_CACHE):
        log(f'  Training all final DP{dp} models (all 7 calibration compositions)')
        models,failures=train_all(packs,fullmask,base)
        atomic_dump({'signature':run_signature(),'dp':dp,'models':models,'failures':failures},modelpath)
    else:log(f'  Reuse DP{dp} frozen model checkpoint')


def collect_fold_results(dps):
    frames=[];ratios=[]
    for dp in dps:
        folder=OUT_DIR/f'DP{dp}'
        for p in sorted(folder.glob('fold_[0-9][0-9][0-9].csv')):frames.append(pd.read_csv(p))
        for p in sorted(folder.glob('ratios_[0-9][0-9][0-9].csv')):ratios.append(pd.read_csv(p))
    if not frames:return None
    df=pd.concat(frames,ignore_index=True)
    df.to_csv(OUT_DIR/'V6G_LORO_all_folds.csv',index=False)
    if ratios:pd.concat(ratios,ignore_index=True).to_csv(OUT_DIR/'V6G_LORO_ratio_selection.csv',index=False)
    summ=df.groupby(['model','dp','nce'],as_index=False).agg(MAE_pp=('abs_error_pp','mean'),
                 median_pp=('abs_error_pp','median'),worst_pp=('abs_error_pp','max'),n_folds=('abs_error_pp','count'))
    summ.to_csv(OUT_DIR/'V6G_LORO_by_condition.csv',index=False)
    summary=df.groupby('model',as_index=False).agg(MAE_pp=('abs_error_pp','mean'),
               worst_pp=('abs_error_pp','max'),n_composition_conditions=('abs_error_pp','count'))
    summary.sort_values('MAE_pp').to_csv(OUT_DIR/'V6G_LORO_overall.csv',index=False)
    baselines=df[df['model']==original_baseline()][['dp','nce','heldout_nominal_pct','abs_error_pp']].rename(columns={'abs_error_pp':'baseline_MAE_pp'})
    pair=df.merge(baselines,on=['dp','nce','heldout_nominal_pct'],how='left')
    pair['delta_vs_manuscript_pp']=pair['abs_error_pp']-pair['baseline_MAE_pp']
    pair.to_csv(OUT_DIR/'V6G_paired_fold_deltas.csv',index=False)
    # Within a fixed scheme, separate smoothing and group normalization effects.
    df['architecture']=df['model'].str.split('__').str[0]
    df['factor']=df['model'].str.split('__').str[1]
    df['group_scheme']=df['factor'].str.split('_').str[2]
    df['SG']=df['factor'].str.slice(0,5)
    pvt=df.pivot_table(index=['dp','nce','heldout_nominal_pct','architecture','group_scheme'],
                       columns='SG',values='abs_error_pp',aggfunc='first')
    effects=[]
    for (dp,nce,hold,core,scheme),r in pvt.iterrows():
        for contrast,a,b in [('smoothing_at_G0','S0_G0','S1_G0'),
                             ('smoothing_at_G1','S0_G1','S1_G1'),
                             ('group_norm_at_S0','S0_G0','S0_G1'),
                             ('group_norm_at_S1','S1_G0','S1_G1')]:
            if a in r and b in r and pd.notna(r[a]) and pd.notna(r[b]):
                effects.append(dict(dp=dp,nce=nce,heldout_nominal_pct=hold,
                                    architecture=core,group_scheme=scheme,contrast=contrast,
                                    off_MAE_pp=float(r[a]),on_MAE_pp=float(r[b]),
                                    delta_on_minus_off_pp=float(r[b]-r[a])))
    write_csv(effects,OUT_DIR/'V6G_FACTORIAL_PAIRED_effects.csv')
    if effects:
        ef=pd.DataFrame(effects)
        ef.groupby(['architecture','group_scheme','contrast'],as_index=False).agg(
             mean_delta_on_minus_off_pp=('delta_on_minus_off_pp','mean'),
             median_delta_pp=('delta_on_minus_off_pp','median'),
             matched_composition_conditions=('delta_on_minus_off_pp','count')
        ).to_csv(OUT_DIR/'V6G_FACTORIAL_EFFECT_summary.csv',index=False)
    sg=df.groupby(['architecture','group_scheme','gaussian_smoothing','group_normalization'],as_index=False).agg(
         MAE_pp=('abs_error_pp','mean'),n_composition_conditions=('abs_error_pp','count'))
    sg.to_csv(OUT_DIR/'V6G_FACTORIAL_2x2_accuracy.csv',index=False)
    # Paired ORIGINAL7-versus-other group schemes at matched DP/NCE/holdout,
    # architecture, S level and G level. No bulk NMR appears in these comparisons.
    x=df.pivot_table(index=['dp','nce','heldout_nominal_pct','architecture',
                             'gaussian_smoothing','group_normalization'],
                     columns='group_scheme',values='abs_error_pp',aggfunc='first')
    pairs=[]
    if 'ORIGINAL7' in x:
        for idx,z in x.iterrows():
            orig=z.get('ORIGINAL7',np.nan)
            for scheme in SELECTED_SCHEMES:
                if scheme=='ORIGINAL7':continue
                val=z.get(scheme,np.nan)
                if np.isfinite(orig) and np.isfinite(val):
                    dp,nce,hold,arch,sm,gn=idx
                    pairs.append(dict(dp=dp,nce=nce,heldout_nominal_pct=hold,
                     architecture=arch,gaussian_smoothing=sm,group_normalization=gn,
                     candidate_scheme=scheme,original_MAE_pp=float(orig),
                     candidate_MAE_pp=float(val),delta_vs_original_pp=float(val-orig)))
    write_csv(pairs,OUT_DIR/'V6G_GROUP_SCHEME_LORO_PAIRED.csv')
    if pairs:
        pd.DataFrame(pairs).groupby(['architecture','gaussian_smoothing','group_normalization',
           'candidate_scheme'],as_index=False).agg(mean_delta_pp=('delta_vs_original_pp','mean'),
           median_delta_pp=('delta_vs_original_pp','median'),
           n_matched=('delta_vs_original_pp','count')
        ).to_csv(OUT_DIR/'V6G_GROUP_SCHEME_LORO_summary.csv',index=False)
    # Same fitted preprocessing/NMF, compare prediction modes at exactly the
    # same heldout composition: K2-only against K2+K8.
    nmfj=df[df['architecture'].str.endswith(('_K2_ONLY','_K2_K8'))].copy()
    nmfj['family']=nmfj['architecture'].str.replace('_K2_ONLY','',regex=False).str.replace('_K2_K8','',regex=False)
    nmfj['feature_set']=np.where(nmfj['architecture'].str.endswith('_K2_ONLY'),'NMF_ONLY','NMF_PLUS_K8')
    joint=nmfj.pivot_table(index=['dp','nce','heldout_nominal_pct','family','factor'],
                    columns='feature_set',values='abs_error_pp',aggfunc='first')
    if {'NMF_ONLY','NMF_PLUS_K8'}<=set(joint.columns):
        joint=joint.dropna(subset=['NMF_ONLY','NMF_PLUS_K8']).reset_index()
        joint['joint_minus_nmf_MAE_pp']=joint['NMF_PLUS_K8']-joint['NMF_ONLY']
        joint.to_csv(OUT_DIR/'V6G_NMF_vs_JOINT_LORO_PAIRED.csv',index=False)
        joint.groupby(['family','factor'],as_index=False).agg(
              mean_joint_minus_nmf_MAE_pp=('joint_minus_nmf_MAE_pp','mean'),
              n_matched=('joint_minus_nmf_MAE_pp','count')
        ).to_csv(OUT_DIR/'V6G_NMF_vs_JOINT_LORO_summary.csv',index=False)
    if len(set(dps))==2 and (summ.n_folds==7).any():
        # Calibration-only leaderboard; DO NOT use external NMR proximity.
        chosen=summ[summ.n_folds==7].sort_values(['dp','MAE_pp','model']).groupby('dp',as_index=False).head(1)
        chosen.to_csv(OUT_DIR/'V6G_CALIBRATION_SELECTION_EXPLORATORY.csv',index=False)
    return df


def calibrate_ood_models(dp,external):
    folder=OUT_DIR/f'DP{dp}'
    file=folder/'FROZEN_full_calibration_models.joblib'
    if not file.exists():raise FileNotFoundError(f'No frozen models for DP{dp}: {file}')
    ck=joblib.load(file)
    if ck['signature']!=run_signature():raise ValueError('Settings changed since frozen models were fit; retrain')
    return ck['models']


def external_predict(dp,external):
    models=calibrate_ood_models(dp,external)
    rows=[];peakrows=[];ratio_rows=[]
    for (arm,e),model in sorted(models.items()):
        if e not in external:continue
        epack=external[e]
        for sample in sorted(set(epack['sample'])):
            sel=(epack['sample']==sample)
            part=pack_subset(epack,sel)
            if len(part['y'])==0:continue
            p=model.predict_raw(part);c=np.clip(p,0,1)
            half=max(1,len(c)//2)
            quality=model.backbone.qc(part)
            selected_raw=[]
            for ratio,(num,den) in zip(model.backbone.ratio_names if model.feature_mode!='K2_ONLY' else (),
                                       model.backbone.ratio_info if model.feature_mode!='K2_ONLY' else ()):
                ci=np.where(np.isclose(part['centers'],num))[0]
                cj=np.where(np.isclose(part['centers'],den))[0]
                if not (len(ci) and len(cj)):continue
                num_support=float(np.mean(part['support'][:,ci[0]])*100)
                den_support=float(np.mean(part['support'][:,cj[0]])*100)
                selected_raw.append(den_support)
                ratio_rows.append(dict(dp=dp,nce=e,model=arm,sample=sample,ratio=ratio,
                                       numerator=num,denominator=den,raw_num_supported_pct=num_support,
                                       raw_den_supported_pct=den_support))
            rows.append(dict(dp=dp,nce=e,model=arm,sample=sample,n_scans=len(c),
                             pred_M_pct=float(c.mean()*100),scan_sd_pp=float(c.std()*100),
                             scan_half_difference_pp=abs(float(c[:half].mean())-float(c[half:].mean()))*100,
                             clipped_scan_pct=float(((p<0)|(p>1)).mean()*100),
                             min_raw_den_support_pct=float(min(selected_raw)) if selected_raw else np.nan,
                             **quality))
        # One line per selected model raw ion (overall external sample scan set)
    for r in rows:
        r['group_scheme']=r['model'].split('__')[1].split('_',2)[2]
        r['architecture']=r['model'].split('__')[0]
        r['gaussian_smoothing']=int('S1' in r['model'].split('__')[-1])
        r['group_normalization']=int('G1' in r['model'].split('__')[-1])
    return rows,ratio_rows


def fragment_transfer_diagnostics(dp,calibration,external):
    """Calibration-vs-external ion support and raw mass-offset diagnostics.

    Mass offsets are descriptive. A hotspot around m/z 484/485 can stem from
    chemistry, broad masking, isotope/adduct structure or other interference;
    this analysis DOES NOT automatically apply a mass correction.
    """
    rows=[];hotspots=[]
    models=calibrate_ood_models(dp,external)
    for e,ext in external.items():
        if e not in calibration:continue
        cal=calibration[e]
        ccent=np.asarray(cal['centers']);ctr=np.asarray(cal['raw'])
        cref=ctr/np.maximum(ctr.sum(axis=1,keepdims=True),EPS)
        eref=ext['raw']/np.maximum(ext['raw'].sum(axis=1,keepdims=True),EPS)
        original=models.get((original_baseline(),e))
        if original is not None:
            nmf=original.backbone.nmf
            # Compare reconstruction errors in the same original feature space.
            x=ext['full_by_factor']['S1_G1_ORIGINAL7']
            z=np.maximum(x[:,nmf.active],0.)*nmf.feature_weights[None,:]
            coef=np.maximum(nmf.nmf.transform(z),0.)
            recon=coef@nmf.nmf.components_
            err=np.abs(z-recon)
            activegrid=np.asarray(ext['grid'])[nmf.active]
            norm=np.maximum(np.linalg.norm(z,axis=1),EPS)
        for sample in sorted(set(ext['sample'])):
            mask=ext['sample']==sample
            for j,mz in enumerate(ccent):
                cd=cal['delta'][:,j];ed=ext['delta'][mask,j]
                c_support=float(np.mean(cal['support'][:,j])*100)
                x_support=float(np.mean(ext['support'][mask,j])*100)
                cmed=float(np.median(cref[:,j]));emed=float(np.median(eref[mask,j]))
                rows.append(dict(dp=dp,nce=e,sample=sample,mz=float(mz),
                    calibration_raw_support_pct=c_support,external_raw_support_pct=x_support,
                    support_difference_pp=x_support-c_support,
                    calibration_relative_raw_median=cmed,external_relative_raw_median=emed,
                    external_to_calibration_relative_ratio=emed/(cmed+EPS),
                    calibration_abs_mz_offset_median_Da=float(np.nanmedian(np.abs(cd))) if np.isfinite(cd).any() else np.nan,
                    external_abs_mz_offset_median_Da=float(np.nanmedian(np.abs(ed))) if np.isfinite(ed).any() else np.nan))
                if original is not None:
                    loc=np.abs(activegrid-mz)<=0.75
                    if loc.any():
                        frac=float(np.mean(err[mask][:,loc].sum(axis=1)/norm[mask]))
                        hotspots.append(dict(dp=dp,nce=e,sample=sample,peak_center_mz=float(mz),
                                             base_K2_mean_abs_residual_over_spectrum_norm=frac,
                                             residual_region_Da=0.75,diagnostic_type='peak_window'))
            if original is not None:
                # High-resolution residual-bin diagnostics reveal off-center
                # m/z~484 vs theoretical m/z~485 without applying a correction.
                by_bin=np.mean(err[mask]/norm[mask,None],axis=0)
                for ind in np.argsort(by_bin)[-15:][::-1]:
                    hotspots.append(dict(dp=dp,nce=e,sample=sample,
                         peak_center_mz=float(activegrid[ind]),
                         base_K2_mean_abs_residual_over_spectrum_norm=float(by_bin[ind]),
                         residual_region_Da=0.0,diagnostic_type='individual_mz_bin'))
    return rows,hotspots


def external_aggregate(external_dps):
    entries=[];ratios=[];fragment_rows=[];hotspot_rows=[]
    for dp in external_dps:
        ext=build_condition_cache(dp,external=True)
        if not ext:continue
        row,r=external_predict(dp,ext);entries+=row;ratios+=r
        cal=build_condition_cache(dp,external=False)
        fragment,hot=fragment_transfer_diagnostics(dp,cal,ext)
        fragment_rows+=fragment;hotspot_rows+=hot
    write_csv(fragment_rows,OUT_DIR/'V6G_FRAGMENT_DOMAIN_SHIFT.csv')
    write_csv(hotspot_rows,OUT_DIR/'V6G_NMF_EXTERNAL_RESIDUAL_HOTSPOTS.csv')
    write_csv(entries,OUT_DIR/'V6G_EXTERNAL_ALL_MODELS.csv')
    write_csv(ratios,OUT_DIR/'V6G_EXTERNAL_RAW_RATIO_QC.csv')
    if entries:
        df=pd.DataFrame(entries)
        cv= df.groupby(['dp','model','sample'],as_index=False).agg(
            NCE_range_pp=('pred_M_pct',lambda a:float(a.max()-a.min())),
            mean_ood_pct=('nmf_ood_pct','mean'),
            worst_scan_half_diff_pp=('scan_half_difference_pp','max'),
            min_raw_denom_support_pct=('min_raw_den_support_pct','min'))
        cv.to_csv(OUT_DIR/'V6G_EXTERNAL_NCE_CONSISTENCY.csv',index=False)
        # Matched S/G effects for each external sample at the SAME DP/NCE/model.
        ep=df.pivot_table(index=['dp','nce','sample','architecture','group_scheme'],
                         columns=['gaussian_smoothing','group_normalization'],
                         values=['pred_M_pct','scan_sd_pp','scan_half_difference_pp',
                                 'nmf_ood_pct','clipped_scan_pct'],aggfunc='first')
        differences=[]
        for (dp,nce,sample,arch,scheme), values in ep.iterrows():
            for metric in ('pred_M_pct','scan_sd_pp','scan_half_difference_pp',
                           'nmf_ood_pct','clipped_scan_pct'):
                for factor,a,b in [('smooth_at_G0',(0,0),(1,0)),
                                    ('smooth_at_G1',(0,1),(1,1)),
                                    ('group_norm_at_S0',(0,0),(0,1)),
                                    ('group_norm_at_S1',(1,0),(1,1))]:
                    k0=(metric,)+a;k1=(metric,)+b
                    if k0 in values.index and k1 in values.index and np.isfinite(values[k0]) and np.isfinite(values[k1]):
                        differences.append(dict(dp=dp,nce=nce,sample=sample,
                                  architecture=arch,group_scheme=scheme,metric=metric,contrast=factor,
                                  value_off=float(values[k0]),value_on=float(values[k1]),
                                  delta_on_minus_off=float(values[k1]-values[k0])))
        write_csv(differences,OUT_DIR/'V6G_EXTERNAL_FACTORIAL_PAIRED_effects.csv')
        if differences:
            ed=pd.DataFrame(differences)
            ed.groupby(['architecture','group_scheme','metric','contrast'],as_index=False).agg(
               mean_delta_on_minus_off=('delta_on_minus_off','mean'),
               median_delta=('delta_on_minus_off','median'),
               n_matched_external_conditions=('delta_on_minus_off','count')
            ).to_csv(OUT_DIR/'V6G_EXTERNAL_FACTORIAL_EFFECT_summary.csv',index=False)
        # At fixed DP/NCE/sample/architecture/S/G, isolate group-boundary change.
        gp=df.pivot_table(index=['dp','nce','sample','architecture',
                                 'gaussian_smoothing','group_normalization'],
                          columns='group_scheme',
                          values=['pred_M_pct','scan_sd_pp','scan_half_difference_pp',
                                  'nmf_ood_pct','clipped_scan_pct','min_raw_den_support_pct'],
                          aggfunc='first')
        group_pairs=[]
        for idx,z in gp.iterrows():
            for metric in ('pred_M_pct','scan_sd_pp','scan_half_difference_pp',
                           'nmf_ood_pct','clipped_scan_pct','min_raw_den_support_pct'):
                old=z.get((metric,'ORIGINAL7'),np.nan)
                for scheme in SELECTED_SCHEMES:
                    if scheme=='ORIGINAL7':continue
                    new=z.get((metric,scheme),np.nan)
                    if np.isfinite(old) and np.isfinite(new):
                        dp,nce,sample,arch,sm,gn=idx
                        group_pairs.append(dict(dp=dp,nce=nce,sample=sample,architecture=arch,
                           smoothing=sm,group_normalization=gn,metric=metric,
                           candidate_scheme=scheme,original_value=float(old),candidate_value=float(new),
                           delta_vs_original=float(new-old)))
        write_csv(group_pairs,OUT_DIR/'V6G_GROUP_SCHEME_EXTERNAL_PAIRED.csv')
        if group_pairs:
            pd.DataFrame(group_pairs).groupby(['architecture','smoothing','group_normalization',
                    'metric','candidate_scheme'],as_index=False).agg(
                mean_delta=('delta_vs_original','mean'),
                median_delta=('delta_vs_original','median'),
                n_matched=('delta_vs_original','count')
            ).to_csv(OUT_DIR/'V6G_GROUP_SCHEME_EXTERNAL_summary.csv',index=False)
        # Context is not a DP label, not an accuracy test, not a selection signal.
        context={'GG_block_rich':12.,'MG_block_rich':51.,'MM_block_rich':82.}
        df['bulk_NMR_context_M_pct']=df['sample'].map(context)
        df['absolute_bulk_context_difference_pp']=abs(df['pred_M_pct']-df['bulk_NMR_context_M_pct'])
        df.to_csv(OUT_DIR/'V6G_EXTERNAL_BULK_CONTEXT_NOT_ACCURACY.csv',index=False)
        # Descriptive matched comparison only: bulk NMR is not a DP-specific target.
        sample_count=df.groupby(['model','dp','nce'])['sample'].nunique().rename('n_samples')
        bulk=df.groupby(['model','dp','nce']).agg(
           mean_abs_bulk_context_disagreement_pp=('absolute_bulk_context_difference_pp','mean'),
           worst_bulk_context_disagreement_pp=('absolute_bulk_context_difference_pp','max'),
           mean_external_nmf_ood_pct=('nmf_ood_pct','mean'),
           worst_scan_half_difference_pp=('scan_half_difference_pp','max'),
           min_raw_den_support_pct=('min_raw_den_support_pct','min')
        ).join(sample_count).reset_index()
        bulk=bulk[bulk.n_samples==3].sort_values(['mean_abs_bulk_context_disagreement_pp','model'])
        bulk.insert(0,'NOTE','POSTHOC bulk NMR disagreement; NOT external DP-specific accuracy')
        bulk.to_csv(OUT_DIR/'V6G_POSTHOC_BULK_AGREEMENT_EXPLORATORY.csv',index=False)
        # Paired original NMF-only vs NMF+K8 per sample/energy/scheme.
        nj=df[df['architecture'].str.endswith(('_K2_ONLY','_K2_K8'))].copy()
        nj['family']=nj['architecture'].str.replace('_K2_ONLY','',regex=False).str.replace('_K2_K8','',regex=False)
        nj['feature_set']=np.where(nj['architecture'].str.endswith('_K2_ONLY'),'NMF_ONLY','NMF_PLUS_K8')
        pp=nj.pivot_table(index=['dp','nce','sample','family','group_scheme',
                                  'gaussian_smoothing','group_normalization'],
                          columns='feature_set',values=['pred_M_pct',
                           'absolute_bulk_context_difference_pp','scan_sd_pp'],aggfunc='first')
        if ('pred_M_pct','NMF_ONLY') in pp and ('pred_M_pct','NMF_PLUS_K8') in pp:
            pp=pp.dropna(subset=[('pred_M_pct','NMF_ONLY'),('pred_M_pct','NMF_PLUS_K8')])
            rows=[]
            for idx,z in pp.iterrows():
                dp,nce,sample,family,scheme,sm,gn=idx
                rows.append(dict(dp=dp,nce=nce,sample=sample,family=family,
                  group_scheme=scheme,gaussian_smoothing=sm,group_normalization=gn,
                  NMF_only_pred_M_pct=float(z[('pred_M_pct','NMF_ONLY')]),
                  NMF_plus_K8_pred_M_pct=float(z[('pred_M_pct','NMF_PLUS_K8')]),
                  pred_change_joint_minus_nmf_pp=float(z[('pred_M_pct','NMF_PLUS_K8')]-z[('pred_M_pct','NMF_ONLY')]),
                  NMF_only_bulk_context_diff_pp=float(z[('absolute_bulk_context_difference_pp','NMF_ONLY')]),
                  NMF_plus_K8_bulk_context_diff_pp=float(z[('absolute_bulk_context_difference_pp','NMF_PLUS_K8')]),
                  NMF_only_scan_sd_pp=float(z[('scan_sd_pp','NMF_ONLY')]),
                  NMF_plus_K8_scan_sd_pp=float(z[('scan_sd_pp','NMF_PLUS_K8')])))
            write_csv(rows,OUT_DIR/'V6G_NMF_vs_JOINT_EXTERNAL_PAIRED.csv')
    log('Saved frozen-model external predictions / ion-support shift / residual hotspots / NCE consistency')


def make_figures():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        a=OUT_DIR/'V6G_LORO_overall.csv'
        if a.exists():
            df=pd.read_csv(a).sort_values('MAE_pp')
            # 224 labels would be unreadable; compare the 28 best fully completed
            # calibration candidates, showing full counts in the original CSV.
            if not df.empty:
                df=df[df.n_composition_conditions==df.n_composition_conditions.max()].head(28)
            fig,ax=plt.subplots(figsize=(14,6))
            ax.bar(df['model'],df['MAE_pp'])
            ax.set(ylabel='Held-out composition MAE (percentage points)',title='V6 pre-specified ablation')
            ax.tick_params(axis='x',labelrotation=83,labelsize=6);fig.tight_layout()
            fig.savefig(OUT_DIR/'V6G_LORO_models.png',dpi=170);plt.close(fig)
        e=OUT_DIR/'V6G_EXTERNAL_ALL_MODELS.csv'
        if e.exists():
            df=pd.read_csv(e)
            for dp in sorted(df.dp.unique()):
                for mod in (original_baseline(),'GUARD_K2_K8__S1_G1_ORIGINAL7',
                            'WEIGHT_K2_K8__S1_G1_ORIGINAL7',
                            'COUPLED_GUARD_K2_K8__S1_G1_ORIGINAL7'):
                    sub=df[(df.dp==dp)&(df.model==mod)]
                    if sub.empty:continue
                    fig,ax=plt.subplots(figsize=(6.5,4.5))
                    for s,grp in sub.groupby('sample'):
                        grp=grp.sort_values('nce');ax.plot(grp.nce,grp.pred_M_pct,'o-',label=s)
                    ax.set(xticks=list(ENERGIES),xlabel='NCE (%)',ylabel='Predicted DP-specific M%',ylim=(0,100),
                           title=f'DP{dp} {mod} (exploratory external)')
                    ax.legend(fontsize=8);fig.tight_layout()
                    fig.savefig(OUT_DIR/f'V6G_EXTERNAL_DP{dp}_{mod}.png',dpi=170);plt.close(fig)
    except ImportError:
        log('matplotlib unavailable: CSVs saved, figures skipped')


def numerical_selftest():
    """Runs without mzML or pyteomics. Tests guard, K2 and coupled equations."""
    rng=np.random.default_rng(7)
    groups=np.array([0,0,1,1,2,2,3,3])
    c=np.array([100,110,175,193,369,485,661,703.],float)
    truth=np.repeat(np.array(MIXES),12)
    X=rng.uniform(.01,1.,size=(len(truth),len(c)))
    X[:,0]+=(truth*3)
    mask=truth<0.99
    floors=fit_guard_floor(X,groups,mask)
    n=normalize_targeted(X,groups,floors)
    assert n.shape==X.shape and np.isfinite(n).all() and (n>=0).all()
    assert np.allclose(normalize_targeted(X,groups),normalize_targeted(X*4,groups))
    assert np.allclose(normalize_targeted(X,groups,group_norm=False),
                       normalize_targeted(X*4,groups,group_norm=False))
    assert all(factor_parts(f)[:2]==(f.startswith('S1'),f.split('_')[1]=='G1') for f in FACTORS)
    assert len(ARMS)==len(CORE_ARMS)*4*len(SELECTED_SCHEMES) and original_baseline() in ARMS
    assert len(set(ARMS))==len(ARMS)
    assert 'BASE_K2_ONLY__S1_G1_ORIGINAL7' in ARMS
    assert 'COUPLED_GUARD_K2_ONLY__S0_G0_ORIGINAL7' in ARMS
    assert group_definitions()['ORIGINAL7']==[(float(a),float(b)) for a,b in ORIGINAL_GROUP_BOUNDARIES]
    # Baseline groups retain original gaps; tiled schemes capture the gap ions.
    test_centers=np.array([175.0248,291.0358,309.0463,485.0784,545.0996,703.1211])
    g0=fragment_groups(test_centers,group_definitions()['ORIGINAL7'])
    gt=fragment_groups(test_centers,group_definitions()['TILED7'])
    assert g0[1]==-1 and g0[2]==1 and g0[4]==-1
    assert np.all(gt>=0)
    assert len(set(tuple(x) for x in group_definitions()['ORIGINAL7']))==7
    B=fit_rank2(n,truth,truth,mask)
    features,res=B.features_residual(n)
    assert features.shape==(len(truth),2) and np.isfinite(res).all()
    # Energy-coupled algebra should be coherent and recoverable with lam=0.
    packs={};bbs={};masks={}
    for e in ENERGIES:
        zz=np.column_stack([truth+rng.normal(0,.025,len(truth)),truth*.7+rng.normal(0,.05,len(truth))])
        p={'nom':truth,'y':truth}
        b=type('Tiny',(),{})()
        b.train_f_k2=zz[mask];b.train_f_k8=zz[mask];b.train_f_nmf=zz[mask]
        packs[e]=p; bbs[e]=b;masks[e]=mask
    fit0=coupled_solver(bbs,packs,masks,'K2_K8',0.)
    fit1=coupled_solver(bbs,packs,masks,'K2_K8',0.25)
    assert len(fit0)==3 and all(len(x)==3 for x in fit1.values())
    # No label-free external sample is ever used by training routines.
    assert run_signature()==run_signature()
    log('Numerical selftest PASSED: raw-ion norm, guards, K2, coupled ridge, group schemes')


def run_main():
    global DATA_DIR,OUT_DIR,USE_CACHE,NMF_MAX_ITER
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-mode',choices=('selftest','pilot','overnight','pilot-overnight','external','report'),
                    default='pilot-overnight')
    ap.add_argument('--data-dir',default=None)
    ap.add_argument('--out-dir',default=None)
    ap.add_argument('--no-cache',action='store_true')
    ap.add_argument('--nmf-max-iter',type=int,default=None)
    ap.add_argument('--threads',type=int,default=2,help='BLAS/OpenMP numerical threads (default 2)')
    ap.add_argument('--dp',type=int,nargs='+',default=None,help='Subset of DP3/DP4; coupled still uses all three energies')
    ap.add_argument('--group-schemes',nargs='+',choices=GROUP_SCHEME_NAMES,default=None,
       help='Optional quick run; ORIGINAL7 is required, default all four schemes')
    args=ap.parse_args()
    global SELECTED_SCHEMES, FACTORS, ARMS
    if args.group_schemes is not None:
        selected=tuple(dict.fromkeys(args.group_schemes))
        if 'ORIGINAL7' not in selected:raise ValueError('ORIGINAL7 baseline is required in group-schemes')
        SELECTED_SCHEMES=selected;FACTORS=make_factors()
        ARMS=tuple(f'{arm}__{factor}' for arm in CORE_ARMS for factor in FACTORS)
    if args.data_dir:DATA_DIR=Path(args.data_dir).expanduser().resolve()
    if args.out_dir:OUT_DIR=Path(args.out_dir).expanduser().resolve()
    if args.no_cache:USE_CACHE=False
    if args.nmf_max_iter is not None:NMF_MAX_ITER=args.nmf_max_iter
    try:
        from threadpoolctl import threadpool_limits
        _thread_limit=threadpool_limits(limits=max(1,args.threads))
    except ImportError:
        log('threadpoolctl unavailable; using library defaults')
    if args.run_mode=='selftest':numerical_selftest();return
    if args.run_mode!='report':numerical_selftest()
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    config={'signature':run_signature(),'version':SCRIPT_VERSION,'data_dir':str(DATA_DIR),'output_dir':str(OUT_DIR),
            'energies':ENERGIES,'dps':DP_LIST,'coupling_lambda':COUPLE_LAMBDA,
            'centroid_ppm':CENTROID_PPM,'nmf_max_iter':NMF_MAX_ITER,'smoothing_sigma_Da':get_base().CONV_SIGMA,
            'factorial':FACTORS,'core_architectures':CORE_ARMS,
            'group_boundaries':group_definitions(),
            'selected_group_schemes':SELECTED_SCHEMES,
            'S1_G1_ORIGINAL7_is_original_manuscript_baseline':True,
            'NO_EXTERNAL_LABEL_MODEL_SELECTION':True,
            'BULK_NMR_NOT_DP_REFERENCE':True}
    (OUT_DIR/'V6G_SETTINGS.json').write_text(json.dumps(config,indent=2,default=json_default))
    # Transparent membership audit: original group gaps are explicitly -1.
    group_membership=[]
    base=get_base()
    for dp in DP_LIST:
        centers=base.get_preprocess_centers(dp)
        for scheme, bounds in group_definitions(base).items():
            if scheme not in SELECTED_SCHEMES:continue
            memberships=fragment_groups(centers,bounds)
            for mz,g in zip(centers,memberships):
                b=bounds[int(g)] if g>=0 else (np.nan,np.nan)
                group_membership.append(dict(dp=dp,group_scheme=scheme,
                   ion_mz=float(mz),group_id=int(g),group_low_mz=b[0],group_high_mz=b[1],
                   in_normalized_group=bool(g>=0)))
    write_csv(group_membership,OUT_DIR/'V6G_GROUP_MEMBERSHIP.csv')
    selected_dps=tuple(args.dp or DP_LIST)
    if any(dp not in DP_LIST for dp in selected_dps):raise ValueError('Currently external transfer supported for DP3 and DP4')
    if args.run_mode in ('pilot','overnight','pilot-overnight'):
        base=get_base();base.DATA_DIR=DATA_DIR
        todo=(selected_dps[:1] if args.run_mode=='pilot' else selected_dps)
        for dp in todo:
            log(f'BEGIN DP{dp} {args.run_mode}; input {DATA_DIR}')
            packs=build_condition_cache(dp)
            if any(e not in packs for e in ENERGIES):raise RuntimeError('Coupled-energy needs NCE 18/22/26')
            run_cv(dp,packs,base)
            collect_fold_results(todo)
        if args.run_mode=='pilot':
            log('PILOT complete; run --run-mode overnight for all DP (resumes DP3)')
            return
    if args.run_mode in ('external','overnight','pilot-overnight'):
        external_aggregate(selected_dps)
    collect_fold_results(selected_dps)
    make_figures()
    log(f'COMPLETE: {OUT_DIR}')


if __name__=='__main__':run_main()
