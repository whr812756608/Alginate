#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Alginate V6.6: all-NCE Gaussian × fragment-group × partial-normalization study.

Designed for the existing alginate manuscript MS/MS data. Runs DP3/DP4 at HCD
NCE 18, 22, 26; six Gaussian smoothing scales; eight chemically motivated
fragment-group definitions; and three normalization strengths, comparing
K8-only and global K2 NMF + K8. Every held-out fold excludes all scans of ONE
nominal composition. Eight K8 identities are selected ONCE in that fold from
the ORIGINAL7 sigma=1 lambda=1 reference, and reused across candidate settings.

External bulk NMR reference values are NOT DP-resolved training targets, do not
participate in fitting or model selection, and are tabulated separately as
DESCRIPTIVE, post-hoc comparisons only.

IDE default: --run-mode overnight
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import time
import traceback
import warnings
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# These files are included unchanged from the V6.3 package.
import alginate_ratio_transfer_IDE as base
import v62_core as core
from DP_spectrum_process_Class import FractionSpectra, DPSpectrumProcessor, filter_peaks_by_centers

HERE = Path(__file__).resolve().parent
DATA_DIR = Path.home() / 'Desktop' / 'alginate_experiment'  # IDE: edit if needed
OUT_DIR = HERE / 'V66_ALL_NCE_RESULTS'
DPS = (3, 4)
NCES = (18., 22., 26.)
MIXES = (0., .2, .4, .5, .6, .8, 1.)
SIGMAS = (0., .10, .25, .50, .75, 1.)
NORMALIZATION_STRENGTHS = (0., .5, 1.)
GROUP_CODES = ('ORIGINAL7', 'G291', 'G545', 'G721',
               'G291_G545', 'G291_G721', 'G545_G721', 'GALL')
MODELS = ('K8_ONLY', 'K2_PLUS_K8')
RIDGE_ALPHA = 1.0
SEED = 20261018
NMF_ITER = 650
EPS = 1e-12
VERSION = 'V66-all-three-NCE-six-sigma-eight-groups-v2'
# Predeclared QC rule; based on observed presence, NOT the expected M%.
EXTERNAL_DENOMINATOR_FLAG_THRESHOLD_PCT = 80.
NMR_CONTEXT = {'GG_block_rich':12., 'MG_block_rich':51., 'MM_block_rich':82.}


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] [V6.6] {msg}', flush=True)


def boundaries():
    orig = [tuple(map(float, pair)) for pair in base.GROUP_BOUNDARIES]
    if len(orig) != 7:
        raise RuntimeError('Expected 7 baseline fragment groups')
    out = {}
    for name in GROUP_CODES:
        group = list(orig)
        if name in ('G291', 'G291_G545', 'G291_G721', 'GALL'):
            a, b = group[1]
            group[1] = (290., b)
        if name in ('G545', 'G291_G545', 'G545_G721', 'GALL'):
            a, b = group[2]
            group[2] = (a, 550.)
        if name in ('G721', 'G291_G721', 'G545_G721', 'GALL'):
            a, b = group[3]
            group[3] = (a, 730.)
        out[name] = tuple(group)
    return out


def configurations(dp):
    """Deduplicate physically equivalent group descriptions and lambda=0.

    DP3 has no monitored ions near 721; 721-only membership changes disappear.
    Lambda zero is the same TIC-normalized input regardless of group labels.
    """
    groups = ('ORIGINAL7', 'G291', 'G545', 'G291_G545') if dp == 3 else GROUP_CODES
    for sigma in SIGMAS:
        yield sigma, 'TIC_ONLY', 0.
        for grp in groups:
            for strength in (.5, 1.):
                yield sigma, grp, strength


def combo_key(sigma, group, strength):
    return f'S{int(round(sigma*100)):03d}_L{int(round(strength*100)):03d}_{group}'


def json_default(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, Path): return str(obj)
    raise TypeError(type(obj).__name__)


def config_dict():
    return dict(version=VERSION, dps=DPS, nces=NCES, mixes=MIXES, sigmas=SIGMAS,
        strengths=NORMALIZATION_STRENGTHS, groups=boundaries(), seed=SEED,
        nmf_iter=NMF_ITER, ridge_alpha=RIDGE_ALPHA,
        peak_window=base.PEAK_WINDOW, original_sigma=base.CONV_SIGMA,
        blend='convex interpolation of TIC and equal-TIC group shapes, with '
              'linearly interpolated total scale; endpoints equal TIC and '
              'original group-area preprocessing',
        ratio_policy='reference ORIGINAL7 sigma1 lambda1 fitted in each train fold',
        external_bulk_not_dp_truth=True)


def signature():
    return hashlib.sha256(json.dumps(config_dict(), sort_keys=True,
                                     default=json_default).encode()).hexdigest()[:16]


def put_csv(rows, path):
    if rows is None or len(rows) == 0: return
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix('.tmp')
    pd.DataFrame(rows).to_csv(t, index=False)
    os.replace(t, p)


def put_obj(obj, path):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_name(p.name + '.tmp')
    joblib.dump(obj, t, compress=2)
    os.replace(t, p)


def mzml_sources(dp, external):
    if external:
        return {name: DATA_DIR / name_ for name, name_ in base.EXTERNAL_FILES.items()}
    return {f'M{int(round(m*100))}': DATA_DIR / f'Raw_F{dp}_M{int(round(m*100))}G{100-int(round(m*100))}.mzML'
            for m in MIXES}


def fingerprint_sources(files, strict):
    entries = []
    for label, path in sorted(files.items()):
        if not path.exists():
            if strict: raise FileNotFoundError(f'Missing {label}: {path}')
            continue
        stat = path.stat()
        entries.append((label, str(path.resolve()), stat.st_size, stat.st_mtime_ns))
    return hashlib.sha256(repr(entries).encode()).hexdigest()[:16]


def ensure_settings():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    f = OUT_DIR / 'V66_SETTINGS.json'
    settings = dict(config_signature=signature(), configuration=config_dict(),
        data_dir=str(DATA_DIR.resolve()),
        input_fingerprints={f'DP{dp}_{suffix}':fingerprint_sources(mzml_sources(dp, ext), not ext)
                            for dp in DPS for ext, suffix in ((False, 'cal'), (True, 'ext'))})
    if f.exists():
        old = json.loads(f.read_text())
        if old != json.loads(json.dumps(settings, default=json_default)):
            raise RuntimeError('Run configuration or input spectra changed. Use a NEW --out-dir '
                               'to avoid mixing incompatible checkpoint results.')
    else:
        f.write_text(json.dumps(settings, indent=2, default=json_default))


def exact_group_norm_matrix(masked, mzgrid, group):
    """Vectorized equivalent to FractionSpectra.group_area_normalization.

    Only nonzero fragment-masked bins are cached. Group windows are evaluated
    on those same grid coordinates, so omitted zero bins cannot change sums.
    """
    x = np.maximum(np.asarray(masked, dtype=np.float32), 0)
    T = x / np.where(x.sum(axis=1,keepdims=True)>0,
                        x.sum(axis=1,keepdims=True), 1.)
    if group == 'TIC_ONLY': return T.copy(), T.copy()
    G = T.copy()
    for lo, hi in boundaries()[group]:
        region = (mzgrid >= lo) & (mzgrid <= hi)
        if not region.any(): continue
        z = G[:, region]
        area = z.sum(axis=1, keepdims=True)
        G[:, region] = z / np.where(area>0, area, 1.)
    return T, G


def blended_normalization(masked, mzgrid, group, strength):
    """lambda=0 -> TIC exactly; lambda=1 -> baseline group normalization.

    Shape blending avoids a discontinuity in *relative* fragment abundances:
      T = TIC spectrum, G = group normalized spectrum, g = sum(G)
      shape = (1-lambda)*T + lambda*(G/g)
      total_scale = (1-lambda) + lambda*g
      return shape*total_scale
    This is not simply mixing raw and normalized intensities.
    """
    T, G = exact_group_norm_matrix(masked, mzgrid, group)
    if strength == 0: return T
    if strength == 1: return G
    mass = np.maximum(G.sum(axis=1,keepdims=True), EPS)
    output = ((1-strength)*T + strength*G/mass) * ((1-strength)+strength*mass)
    return np.maximum(np.asarray(output, np.float32), 0)


def interpolate_scan(scan, dp, processor, fullgrid, relevant):
    mz = np.asarray(scan['m/z array'], np.float32)
    y = np.asarray(scan['intensity array'], np.float32)
    keep = (mz >= 0) & (mz <= base.DP_TARGET_MZ[dp]+20)
    mz, y = mz[keep], y[keep]
    if len(mz) < 4 or not np.isfinite(y).all(): return None
    g, raw, _, _ = FractionSpectra.interpolate_to_grid_without_norm(mz, y, fullgrid)
    results = {}
    for s in SIGMAS:
        if s == 0: smooth = raw
        elif s == 1.: smooth = processor._smooth_impl(g, raw)
        else: smooth = FractionSpectra.smooth_spectrum(g, raw, sigma=s)
        filtered = filter_peaks_by_centers(g, smooth, base.get_preprocess_centers(dp),
                                           window=base.PEAK_WINDOW)
        results[s] = np.maximum(np.asarray(filtered, np.float32)[relevant], 0)
    cents = np.asarray(base.get_preprocess_centers(dp), float)
    # Exact centroid support from the V6.2 source code, not synthetic grid support.
    _, support, delta = core.raw_centroid_peaks(mz, y, cents)
    return results, np.asarray(support,bool), np.asarray(delta, np.float32)


def cache_dp(dp, external=False):
    from pyteomics import mzml
    files=mzml_sources(dp, external)
    if external:
        files={name:path for name,path in files.items() if path.exists()}
    if not files:
        if external: log(f'No DP{dp} external mzML files; skipping external inference');return {}
        raise FileNotFoundError('No calibration mzML files')
    fp=fingerprint_sources(files, not external)
    dest=OUT_DIR/'CACHE'/f'DP{dp}_{"ext" if external else "cal"}.joblib'
    if dest.exists():
        store=joblib.load(dest)
        if store.get('signature')==signature() and store.get('input_fp')==fp:
            log(f'Loaded DP{dp} {"external" if external else "calibration"} spectral cache')
            return store['packs']
    proc=DPSpectrumProcessor(conv_sigma=base.CONV_SIGMA, default_groups=base.GROUP_BOUNDARIES)
    grid=proc.grid
    centers=np.asarray(base.get_preprocess_centers(dp),float)
    rel=(grid>=base.CLIP_LO)&(grid<=base.DP_TARGET_MZ[dp]+5)
    rel &= np.any(np.abs(grid[:,None]-centers[None,:])<=base.PEAK_WINDOW+1e-8,axis=1)
    small_grid=np.asarray(grid[rel], float)
    acc={energy:defaultdict(list) for energy in NCES}
    for label,path in sorted(files.items()):
        log(f'Parsing DP{dp} {path.name}')
        nom=np.nan if external else float(label[1:])/100
        # Label follows the manuscript's original FractionSpectra assignment.
        yy=np.nan if external else float(FractionSpectra(dp,nom,dp,proc.grid).NMR_MG_ratio)
        with mzml.read(str(path)) as reader:
            for scan in reader:
                if scan.get('ms level') != 2 or 'precursorList' not in scan:continue
                try:
                    precursor=scan['precursorList']['precursor'][0]
                    target=precursor.get('isolationWindow',{}).get('isolation window target m/z')
                    energy=precursor.get('activation',{}).get('collision energy')
                    if target is None or energy is None:continue
                    if abs(float(target)-base.DP_TARGET_MZ[dp])>1e-4:continue
                    e=next((e for e in NCES if abs(float(energy)-e)<1e-5),None)
                    if e is None:continue
                    result=interpolate_scan(scan,dp,proc,grid,rel)
                    if result is None:continue
                    sm,sp,delta=result
                    if sm[1.].sum()<=EPS:continue
                    b=acc[e]
                    for s in SIGMAS:b[f'S_{s}'].append(sm[s])
                    b['sample'].append(label);b['nom'].append(nom);b['y'].append(yy)
                    b['support'].append(sp);b['delta'].append(delta)
                except (KeyError,ValueError,TypeError,IndexError,RuntimeError) as exc:
                    acc[e]['errors'].append(str(exc)[:120])
    packs={}
    for energy,b in acc.items():
        if not b['sample']:
            if not external:raise RuntimeError(f'No MS2 spectra found for DP{dp}/NCE{energy}')
            continue
        pack=dict(dp=dp,nce=energy,grid=small_grid,centers=centers,
                  sample=np.asarray(b['sample'],str),nom=np.asarray(b['nom'],float),
                  y=np.asarray(b['y'],float),support=np.asarray(b['support'],bool),
                  delta=np.asarray(b['delta'],np.float32),
                  masked={s:np.asarray(b[f'S_{s}'],np.float32) for s in SIGMAS},
                  bad_scans=len(b['errors']))
        if not external and not np.allclose(np.unique(pack['nom']), MIXES):
            raise RuntimeError(f'DP{dp}/NCE{energy}: incomplete calibration compositions')
        packs[energy]=pack
        log(f'  DP{dp} NCE{energy:g}: {len(pack["sample"])} scans, {len(small_grid)} bins')
    put_obj(dict(signature=signature(),input_fp=fp,packs=packs),dest)
    return packs


def matrix(pack,sigma,group,strength):
    return blended_normalization(pack['masked'][sigma],pack['grid'],group,strength)


def ratio_data(pack,X):
    return base.build_ratio_matrices(X,pack['grid'],
                                     list(base.get_ratio_candidate_centers(pack['dp'])))


def choose_reference_ratios(pack,train):
    X=matrix(pack,1.,'ORIGINAL7',1.)
    R,D,names,info=ratio_data(pack,X)
    ranked,_=base.rank_ratio_candidates(R[train],D[train],pack['y'][train],
                                         pack['nom'][train],names,info)
    chosen=base.select_nonredundant_ratios(ranked,8)
    if len(chosen)!=8:raise RuntimeError(f'Only {len(chosen)} reference ratios available')
    return tuple(int(row['idx']) for row in chosen)


def ref_ratios(dp,nce,pack,train,hold):
    key='FULL' if hold is None else f'H{int(round(hold*100)):03d}'
    path=OUT_DIR/'REFERENCE_RATIOS'/f'DP{dp}'/f'E{int(nce)}_{key}.joblib'
    if path.exists():
        saved=joblib.load(path)
        if saved.get('signature')==signature():return tuple(saved['ratios'])
    ratios=choose_reference_ratios(pack,train)
    put_obj(dict(signature=signature(),ratios=ratios),path)
    return ratios


def fold_weights(nom):
    return core.group_weights(nom)


@dataclass
class Fitted:
    model:str
    sigma:float
    group:str
    strength:float
    ratios:tuple
    nmf:object
    scaler:object
    reg:object

    def get_features(self,pack,X):
        R=ratio_data(pack,X)[0][:,list(self.ratios)]
        if self.model=='K8_ONLY':return R
        return np.column_stack((self.nmf.features_residual(X)[0],R))

    def predictions(self,pack,X):
        A=self.scaler.transform(self.get_features(pack,X))
        raw=self.reg.predict(A)
        return np.clip(raw,0.,1.),raw


def fit_models(pack,train,X,ratios,sigma,group,strength):
    """Both models share exactly the same frozen eight ratio IDs."""
    w=fold_weights(pack['nom'][train]);y=pack['y'][train]
    R=ratio_data(pack,X)[0][:,list(ratios)]
    nmf=core.fit_rank2(X,pack['y'],pack['nom'],train,weighted=False,raw_coeff=True)
    H=nmf.features_residual(X)[0]
    fits={}
    for name,A in (('K8_ONLY',R),('K2_PLUS_K8',np.column_stack((H,R)))):
        scaler=StandardScaler()
        try:scaler.fit(A[train],sample_weight=w)
        except TypeError:scaler.fit(A[train])
        model=Ridge(alpha=RIDGE_ALPHA).fit(scaler.transform(A[train]),y,sample_weight=w)
        fits[name]=Fitted(name,sigma,group,strength,tuple(ratios),
                          nmf if name=='K2_PLUS_K8' else None,scaler,model)
    return fits


def ratio_support(pack,selection,mask):
    _,_,_,metadata=ratio_data(pack,np.ones((1,len(pack['grid'])),np.float32))
    centers=np.asarray(pack['centers'],float)
    scores=[]
    for idx in selection:
        den=metadata[int(idx)]['p2']
        loc=np.where(np.isclose(centers,den,atol=1e-5))[0]
        if len(loc):scores.append(float(pack['support'][mask,loc[0]].mean()*100))
    return min(scores) if scores else np.nan


def summarize_predictions(pack,fit,X,select,sample,truth=None):
    pred,raw=fit.predictions(pack,X)
    ood=np.nan
    resid=np.nan
    if fit.nmf is not None:
        _,r=fit.nmf.features_residual(X)
        ood=float(np.mean(r>fit.nmf.resid_q99)*100)
        resid=float(np.mean(r))
    denom=ratio_support(pack,fit.ratios,select)
    halves=float(abs(np.mean(pred[:len(pred)//2])-np.mean(pred[len(pred)//2:]))*100) if len(pred)>1 else np.nan
    row=dict(dp=pack['dp'],nce=pack['nce'],gaussian_sigma_Da=fit.sigma,
             group_scheme=fit.group,normalization_strength=fit.strength,
             model=fit.model,sample=sample,n_scans=int(len(pred)),
             pred_M_pct=float(np.mean(pred)*100),scan_sd_pp=float(np.std(pred)*100),
             scan_half_difference_pp=halves,clipped_scan_pct=float(np.mean((raw<0)|(raw>1))*100),
             min_raw_denominator_support_pct=denom,
             weak_ratio_denominator_flag=bool(denom<EXTERNAL_DENOMINATOR_FLAG_THRESHOLD_PCT),
             nmf_ood_pct=ood,nmf_relative_residual_mean=resid)
    if truth is not None:
        row['true_M_pct']=float(truth*100)
        row['abs_error_pp']=abs(row['pred_M_pct']-row['true_M_pct'])
    return row


def checkpoint(dp,nce,hold,sigma,group,lam,external=False):
    if external:
        return OUT_DIR/'EXTERNAL_CHECKPOINTS'/f'DP{dp}'/f'E{int(nce)}'/(combo_key(sigma,group,lam)+'.csv')
    return OUT_DIR/'FOLD_CHECKPOINTS'/f'DP{dp}'/f'E{int(nce)}'/f'H{int(round(hold*100)):03d}'/(combo_key(sigma,group,lam)+'.csv')


def run_cv(dp,packs,pilot=False,nces=NCES):
    holds=MIXES[:2] if pilot else MIXES
    cfg=[(1.,'ORIGINAL7',1.),(.5,'ORIGINAL7',1.),(1.,'GALL',1.),(1.,'GALL',.5)] if pilot else list(configurations(dp))
    for e in nces:
        pack=packs[e]
        log(f'DP{dp} NCE{e:g}: training-fold predictions ({len(cfg)} preprocessing settings)')
        for hold in holds:
            train=~np.isclose(pack['nom'],hold);test=~train
            if np.unique(pack['nom'][train]).size != 6:raise RuntimeError('Expected six train compositions')
            ratios=ref_ratios(dp,e,pack,train,hold)
            for sigma,group,lam in cfg:
                path=checkpoint(dp,e,hold,sigma,group,lam)
                if path.exists(): continue
                try:
                    X=matrix(pack,sigma,group,lam)
                    models=fit_models(pack,train,X,ratios,sigma,group,lam)
                    rows=[]
                    for name,fit in models.items():
                        row=summarize_predictions(pack,fit,X[test],test,f'H{hold:g}',pack['y'][test].mean())
                        row['holdout_nominal_pct']=int(round(hold*100))
                        rows.append(row)
                    put_csv(rows,path)
                except Exception:
                    err=path.with_suffix('.failed.txt')
                    err.parent.mkdir(parents=True,exist_ok=True)
                    err.write_text(traceback.format_exc())
                    log(f'FAILED: DP{dp}/NCE{e:g}/hold{hold:g} {combo_key(sigma,group,lam)}')
            log(f'DP{dp}/NCE{e:g}: completed held-out nominal {hold*100:.0f}%')


def run_external(dp,cal,ext,pilot=False,nces=NCES):
    cfg=[(1.,'ORIGINAL7',1.),(.5,'ORIGINAL7',1.),(1.,'GALL',1.),(1.,'GALL',.5)] if pilot else list(configurations(dp))
    for e in nces:
        if e not in ext:
            log(f'No external DP{dp}/NCE{e:g}, skipping');continue
        train_pack=cal[e];ext_pack=ext[e]
        train=np.ones(len(train_pack['y']),bool)
        ratios=ref_ratios(dp,e,train_pack,train,None)
        log(f'DP{dp} NCE{e:g}: full-calibration models -> external ({len(cfg)} settings)')
        for sigma,group,lam in cfg:
            path=checkpoint(dp,e,None,sigma,group,lam,external=True)
            if path.exists():continue
            try:
                X=matrix(train_pack,sigma,group,lam)
                Xe=matrix(ext_pack,sigma,group,lam)
                fitted=fit_models(train_pack,train,X,ratios,sigma,group,lam)
                rows=[]
                for name,model in fitted.items():
                    for sample in sorted(set(ext_pack['sample'])):
                        hit=ext_pack['sample']==sample
                        rows.append(summarize_predictions(ext_pack,model,Xe[hit],hit,sample))
                put_csv(rows,path)
            except Exception:
                err=path.with_suffix('.failed.txt')
                err.parent.mkdir(parents=True,exist_ok=True)
                err.write_text(traceback.format_exc())
                log(f'FAILED external DP{dp}/NCE{e:g} {combo_key(sigma,group,lam)}')


def compile_reports():
    folds=sorted((OUT_DIR/'FOLD_CHECKPOINTS').glob('DP*/E*/H*/*.csv'))
    externals=sorted((OUT_DIR/'EXTERNAL_CHECKPOINTS').glob('DP*/E*/*.csv'))
    cv=None; ext=None
    if folds:
        cv=pd.concat([pd.read_csv(p) for p in folds],ignore_index=True)
        cv.to_csv(OUT_DIR/'V66_LORO_ALL_FOLDS.csv',index=False)
        dims=['dp','nce','gaussian_sigma_Da','group_scheme','normalization_strength','model']
        summary=cv.groupby(dims,dropna=False,as_index=False).agg(
            MAE_pp=('abs_error_pp','mean'),max_fold_error_pp=('abs_error_pp','max'),
            n_folds=('abs_error_pp','count'),
            mean_validation_min_den_support_pct=('min_raw_denominator_support_pct','mean'),
            worst_validation_min_den_support_pct=('min_raw_denominator_support_pct','min'))
        summary['completed_seven_folds']=summary.n_folds.eq(7)
        summary.to_csv(OUT_DIR/'V66_LORO_BY_CONDITION.csv',index=False)
        eligible=summary[(summary.model=='K2_PLUS_K8')&(summary.completed_seven_folds)].copy()
        eligible=eligible.sort_values(['dp','nce','MAE_pp','max_fold_error_pp'])
        winners=eligible.groupby(['dp','nce'],as_index=False).head(1).copy()
        winners.to_csv(OUT_DIR/'V66_CALIBRATION_SELECTED_PER_DP_NCE.csv',index=False)
        # Shared-NCE selection uses matched DP3+DP4, selected on calibration only.
        scores=winners.groupby('nce',as_index=False).agg(
            joint_DP3_DP4_MAE_pp=('MAE_pp','mean'),
            worst_DP_error_pp=('MAE_pp','max'),both_DPs=('dp','nunique'))
        scores=scores[scores.both_DPs==2].sort_values('joint_DP3_DP4_MAE_pp')
        scores.to_csv(OUT_DIR/'V66_COMMON_NCE_CALIBRATION_SELECTION.csv',index=False)
        # Also disclose calibration-only joint comparator with identical group/sigma/lambda across DPs.
        same=eligible.groupby(['nce','gaussian_sigma_Da','group_scheme','normalization_strength'],as_index=False).agg(
            combined_MAE_pp=('MAE_pp','mean'),n_dps=('dp','nunique'))
        same=same[same.n_dps==2].sort_values(['nce','combined_MAE_pp'])
        same.to_csv(OUT_DIR/'V66_COMMON_PREPROCESSING_BOTH_DP_RANKING.csv',index=False)
        base=summary[(summary.group_scheme=='ORIGINAL7')&
                     np.isclose(summary.gaussian_sigma_Da,1.)&
                     np.isclose(summary.normalization_strength,1.)]
        base.to_csv(OUT_DIR/'V66_ORIGINAL_BASELINE.csv',index=False)
        paired=eligible.merge(base[['dp','nce','model','MAE_pp']],on=['dp','nce','model'],suffixes=('','_base'))
        paired['delta_MAE_vs_original_pp']=paired.MAE_pp-paired.MAE_pp_base
        paired.to_csv(OUT_DIR/'V66_CALIBRATION_VS_ORIGINAL.csv',index=False)
    if externals:
        ext=pd.concat([pd.read_csv(p) for p in externals],ignore_index=True)
        ext.to_csv(OUT_DIR/'V66_EXTERNAL_ALL_MODELS.csv',index=False)
        ext['bulk_NMR_M_pct']=ext['sample'].map(NMR_CONTEXT)
        ext['difference_to_bulk_pp']=abs(ext['pred_M_pct']-ext['bulk_NMR_M_pct'])
        ext['NOTICE']='Bulk NMR is DP-unresolved; retrospective context, NOT external DP-specific accuracy'
        ext.to_csv(OUT_DIR/'V66_BULK_NMR_CONTEXT_NOT_ACCURACY.csv',index=False)
        dims=['dp','nce','gaussian_sigma_Da','group_scheme','normalization_strength','model']
        agg=ext.groupby(dims,as_index=False).agg(
            bulk_disagreement_pp=('difference_to_bulk_pp','mean'),
            n_external_populations=('sample','nunique'),
            min_external_den_support_pct=('min_raw_denominator_support_pct','min'),
            max_scan_half_difference_pp=('scan_half_difference_pp','max'),
            mean_ood_pct=('nmf_ood_pct','mean'))
        agg.sort_values('bulk_disagreement_pp').to_csv(OUT_DIR/'V66_POSTHOC_BULK_RANKING_EXPLORATORY.csv',index=False)
        consistency=ext.groupby(['dp','gaussian_sigma_Da','group_scheme','normalization_strength','model','sample'],as_index=False).agg(
            n_energies=('nce','nunique'),NCE_range_pp=('pred_M_pct',lambda x:float(x.max()-x.min())),
            min_den_support_pct=('min_raw_denominator_support_pct','min'))
        consistency.to_csv(OUT_DIR/'V66_EXTERNAL_NCE_CONSISTENCY.csv',index=False)
        if cv is not None:
            selected=pd.read_csv(OUT_DIR/'V66_CALIBRATION_SELECTED_PER_DP_NCE.csv')
            comparison=ext.merge(selected[dims+['MAE_pp','max_fold_error_pp']],on=dims,how='inner')
            comparison.to_csv(OUT_DIR/'V66_PRIMARY_DP_NCE_M_PERCENT_TABLE.csv',index=False)
            by_sample=comparison.pivot_table(index=['nce','dp','gaussian_sigma_Da','group_scheme','normalization_strength','model','MAE_pp'],
                columns='sample',values='pred_M_pct',aggfunc='mean').reset_index()
            by_sample.to_csv(OUT_DIR/'V66_PRIMARY_COMPACT_ALL_NCE_TABLE.csv',index=False)
    report=[f'V6.6 all-NCE factorial summary',f'Configuration signature: {signature()}',
            f'Calibration checkpoints: {len(folds)}',f'External checkpoints: {len(externals)}',
            f'Complete grid expected: DP3 {len(list(configurations(3)))} × 3 NCE × 7 folds + DP4 {len(list(configurations(4)))} × 3 NCE × 7 folds',
            'Model selection is by grouped calibration holdouts, NEVER by external bulk NMR.',
            'The three bulk NMR values are DP-unresolved and cannot measure external DP3/DP4 accuracy.',
            'Missing/weak ratio denominators are flagged; absence of a valid signal cannot be interpreted as accurate composition.']
    (OUT_DIR/'V66_EVIDENCE_AUDIT.txt').write_text('\n'.join(report)+'\n')
    make_figures(cv,ext)
    log('Compiled CSV reports and evidence audit')


def make_figures(cv,ext):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:return
    figdir=OUT_DIR/'FIGURES';figdir.mkdir(exist_ok=True)
    selected=OUT_DIR/'V66_CALIBRATION_SELECTED_PER_DP_NCE.csv'
    if selected.exists():
        df=pd.read_csv(selected)
        if len(df):
            fig,ax=plt.subplots(figsize=(8,4))
            pos=np.arange(len(df))
            ax.bar(pos,df['MAE_pp'])
            ax.set_xticks(pos,[f'DP{int(x.dp)} NCE{int(x.nce)}' for x in df.itertuples()],rotation=35,ha='right')
            ax.set_ylabel('Seven-composition LORO MAE (M percentage points)')
            ax.set_title('Calibration-selected K2 NMF + K8 models')
            fig.tight_layout();fig.savefig(figdir/'CALIBRATION_SELECTED_ALL_NCE.png',dpi=170);plt.close(fig)
    if ext is not None and selected.exists() and len(ext):
        sel=pd.read_csv(selected)
        keys=['dp','nce','gaussian_sigma_Da','group_scheme','normalization_strength','model']
        rows=ext.merge(sel[keys],on=keys)
        for dp in sorted(rows.dp.unique()):
            fig,ax=plt.subplots(figsize=(8,4))
            colors={'GG_block_rich':'C0','MG_block_rich':'C1','MM_block_rich':'C2'}
            for sample,z in rows[rows.dp==dp].groupby('sample'):
                z=z.sort_values('nce')
                ax.plot(z['nce'],z['pred_M_pct'],'o-',label=sample,color=colors.get(sample))
            ax.set(xticks=list(NCES),ylim=(0,100),xlabel='NCE (%)',ylabel='Predicted M%')
            ax.set_title(f'DP{dp} calibration-selected K2+K8 across all NCEs')
            ax.legend();fig.tight_layout()
            fig.savefig(figdir/f'DP{dp}_EXTERNAL_SHARED_NCE_COMPARISON.png',dpi=170);plt.close(fig)


def selftest():
    # Exact baseline equivalence on the ORIGINAL7 gaussian-smoothed spectrum.
    rng=np.random.default_rng(57)
    grid=np.arange(100.,801.,.2)
    x=np.zeros((4,len(grid)),np.float32)
    for j in range(4):
        for mz in (175.,291.,309.,351.,467.,485.,545.,661.,703.,721.):
            x[j] += float(rng.uniform(.1,5))*np.exp(-.5*((grid-mz)/.55)**2).astype(np.float32)
    t,g=exact_group_norm_matrix(x,grid,'ORIGINAL7')
    original=np.stack([FractionSpectra.group_area_normalization(grid,np.copy(row),base.GROUP_BOUNDARIES)
                       for row in x])
    assert np.allclose(g,original,atol=1e-6,rtol=1e-6),np.max(abs(g-original))
    assert np.allclose(blended_normalization(x,grid,'ORIGINAL7',1.),original,atol=1e-6)
    assert np.allclose(blended_normalization(x,grid,'ORIGINAL7',0.),t,atol=1e-6)
    mid=blended_normalization(x,grid,'ORIGINAL7',.5)
    assert np.all(np.isfinite(mid)) and np.min(mid)>=0
    assert len(list(configurations(3)))==54
    assert len(list(configurations(4)))==102
    assert len(set(combo_key(*c) for c in configurations(3)))==54
    assert len(set(combo_key(*c) for c in configurations(4)))==102
    # Real ratio selections/nmf tests are in test_v66_synthetic.py.
    log('Selftest passed: baseline group-area normalization and grid deduplication')


def main():
    global DATA_DIR, OUT_DIR, NMF_ITER
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-mode',choices=('selftest','pilot','overnight','external','report'),default='overnight')
    p.add_argument('--data-dir',type=Path,default=None)
    p.add_argument('--out-dir',type=Path,default=None)
    p.add_argument('--dp',nargs='+',type=int,choices=DPS,default=list(DPS))
    p.add_argument('--nces',nargs='+',type=float,choices=NCES,default=list(NCES))
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--nmf-max-iter',type=int,default=NMF_ITER)
    args=p.parse_args()
    if args.data_dir is not None:DATA_DIR=args.data_dir.expanduser().resolve()
    if args.out_dir is not None:OUT_DIR=args.out_dir.expanduser().resolve()
    NMF_ITER=int(args.nmf_max_iter)
    # Forward this setting to the NMF inherited from the V6.2 module.
    core.NMF_MAX_ITER=NMF_ITER
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=max(1,args.threads))
    except ImportError:pass
    selftest()
    if args.run_mode=='selftest':return
    ensure_settings()
    if args.run_mode=='report':compile_reports();return
    for dp in args.dp:
        log(f'BEGIN DP{dp}, NCE {args.nces}, mode={args.run_mode}')
        cal=cache_dp(dp)
        if args.run_mode in ('overnight','pilot'):
            run_cv(dp,cal,pilot=args.run_mode=='pilot',nces=tuple(args.nces))
        ext=cache_dp(dp,external=True)
        if ext:run_external(dp,cal,ext,pilot=args.run_mode=='pilot',nces=tuple(args.nces))
    compile_reports()
    log('COMPLETE -> '+str(OUT_DIR))


if __name__=='__main__':main()
