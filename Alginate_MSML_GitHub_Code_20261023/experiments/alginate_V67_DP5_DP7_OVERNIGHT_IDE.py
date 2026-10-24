#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V6.7: DP5-DP7 calibration-only transfer of the K2 NMF + K8 methodology.

Defaults to --run-mode overnight: full 7-composition LORO preprocessing ablation,
plus separately predeclared nested CV. No DP5-DP7 external ground truth or mzML
is presumed. A fixed, *predeclared* preprocessing protocol is reported separately
from development-selected minima and nested-selection performance.
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
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import alginate_ratio_transfer_IDE as base
import alginate_V66_ALL_NCE_OVERNIGHT_IDE as v66
from DP_spectrum_process_Class import FractionSpectra, DPSpectrumProcessor
import v62_core as core

HERE=Path(__file__).resolve().parent
DATA_DIR=Path.home()/'Desktop'/'alginate_experiment'
OUT_DIR=HERE/'V67_DP5_DP7_RESULTS'
DPS=(5,6,7)
NCES=(18.,22.,26.)
MIXES=(0.,.2,.4,.5,.6,.8,1.)
SIGMAS=(0.,.10,.25,.50,.75,1.)
SCHEMES=('ORIGINAL7','EXTENDED7','GUARDED_EXTENDED7')
MODELS=('K8_ONLY','K2_PLUS_K8')
VERSION='v67-DP5-7-group-aware-nested-v1'
SEED=20261018
NMFI=650
EPS=1e-12
# Candidate recipes for inner-CV selection MUST be fixed before looking at data.
NESTED_CHOICES=((1.,'ORIGINAL7'),(.5,'EXTENDED7'),(.5,'GUARDED_EXTENDED7'),
                (1.,'GUARDED_EXTENDED7'))
# Fixed-protocol transfer reference; V6.6 results are not yet available.
# Supply --transfer-protocol locked_protocol.json to use a subsequently fixed rule.
DEFAULT_FIXED={str(int(e)):{'sigma':1.,'scheme':'ORIGINAL7'} for e in NCES}
GUARD_MAX_BOOST=25.
GUARD_MIN_GROUP_SUPPORT=.90


def log(s): print(f'[{time.strftime("%H:%M:%S")}] [V6.7] {s}',flush=True)
def sha(obj): return hashlib.sha256(json.dumps(obj,sort_keys=True,default=str).encode()).hexdigest()[:16]

def atomic_csv(rows,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if len(rows)==0:return
    t=path.with_suffix('.tmp')
    pd.DataFrame(rows).to_csv(t,index=False)
    os.replace(t,path)

def atomic_joblib(obj,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    t=path.with_name(path.name+'.tmp')
    joblib.dump(obj,t,compress=2)
    os.replace(t,path)


def grid_factors():
    return [(s,g) for s in SIGMAS for g in SCHEMES]


def source_files(dp):
    return {f'M{int(round(m*100))}':DATA_DIR/f'Raw_F{dp}_M{int(round(m*100))}G{100-int(round(m*100))}.mzML'
            for m in MIXES}


def source_fingerprint(dp):
    items=[]
    for name,p in sorted(source_files(dp).items()):
        if not p.exists():raise FileNotFoundError(f'Missing DP{dp} calibration mzML: {p}')
        st=p.stat();items.append((name,str(p.resolve()),st.st_size,st.st_mtime_ns))
    return sha(items)


def fixed_protocol_from_json(path):
    if path is None:return DEFAULT_FIXED
    data=json.loads(Path(path).read_text())
    # Expected: {"18":{"sigma":0.5,"scheme":"GUARDED_EXTENDED7"}, ...}
    for e in NCES:
        item=data.get(str(int(e)))
        if not isinstance(item,dict):raise ValueError(f'Missing energy {int(e)} in {path}')
        if float(item['sigma']) not in SIGMAS or item['scheme'] not in SCHEMES:
            raise ValueError(f'Unsupported sigma/scheme for NCE {e}: {item}')
    return data


def setting_signature(fixed):
    return sha(dict(version=VERSION,dp=DPS,nce=NCES,mixes=MIXES,sigma=SIGMAS,
        schemes=SCHEMES,nested=NESTED_CHOICES,guard_maxboost=GUARD_MAX_BOOST,
        guard_min_support=GUARD_MIN_GROUP_SUPPORT,ratio_selection='TRAIN_ONLY_ORIGINAL7',
        nmf_iter=NMFI,alpha=1.,fixed=fixed,
        peak_window=base.PEAK_WINDOW,original_groups=base.GROUP_BOUNDARIES,
        all_centers=base.ALL_FILTER_CENTERS))


def ensure_manifest(fixed):
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    spec=dict(signature=setting_signature(fixed),inputs={str(dp):source_fingerprint(dp) for dp in DPS},
              fixed_protocol=fixed,folder=str(DATA_DIR.resolve()),version=VERSION)
    path=OUT_DIR/'V67_SETTINGS.json'
    if path.exists():
        prior=json.loads(path.read_text())
        if prior!=spec:raise RuntimeError('Input data or settings changed. Choose a new --out-dir '
                                           'or restore the original settings; checkpoints are incompatible.')
    else:path.write_text(json.dumps(spec,indent=2))


def extended_membership(dp,grid):
    """DP-adaptive fragment families, never assigning the DP's intact precursor.

    Mass windows are chemically informed candidate families. Only annotated centers
    are assigned; the rest of the masked grid keeps its TIC-normalized signal.
    """
    groups=((100.,205.),(275.,385.),(450.,565.),(585.,755.),
            (815.,930.),(1015.,1100.),(1170.,1270.))
    precursor=float(base.DP_TARGET_MZ[dp])
    centers=np.array(base.get_preprocess_centers(dp),float)
    all_masks=[]; rows=[]
    for j,(lo,hi) in enumerate(groups):
        members=[float(c) for c in centers if lo<=c<=hi and abs(c-precursor)>base.PEAK_WINDOW]
        if len(members)<2:continue # no isolated one-ion groups
        mask=np.zeros(len(grid),bool)
        for c in members:mask |= abs(grid-c)<=base.PEAK_WINDOW+1e-8
        # Guard against inclusion of the DP's intact precursor tail.
        mask &= abs(grid-precursor)>base.PEAK_WINDOW+1e-8
        if not mask.any():continue
        all_masks.append((f'FAMILY{j+1}',mask,np.array(members,float)))
        rows.extend(dict(dp=dp,family=f'FAMILY{j+1}',fragment_mz=c,nominal_start=lo,
                         nominal_end=hi,precursor_excluded=True) for c in members)
    return all_masks,rows


def preprocessing(pack,sigma,scheme,train=None):
    """TIC first; original or extended group normalizations.

    Guard floors are learned strictly from train scans and do not use the validation
    data's label/distribution. Weak groups cannot be amplified indefinitely.
    """
    raw=np.maximum(np.asarray(pack['masked'][sigma],float),0.)
    totals=raw.sum(axis=1,keepdims=True)
    TIC=raw/np.maximum(totals,EPS)
    if scheme=='ORIGINAL7':
        # Validate this against original group-area implementation in selftest.
        return v66.exact_group_norm_matrix(raw,pack['grid'],'ORIGINAL7')[1]
    dp=int(pack['dp'])
    masks,_=extended_membership(dp,pack['grid'])
    G=TIC.copy()
    if scheme=='GUARDED_EXTENDED7' and train is None:
        raise ValueError('Guard floors require an explicit training scan mask')
    if train is not None and train.dtype!=bool:train=np.asarray(train,bool)
    for name,region,members in masks:
        area=TIC[:,region].sum(axis=1)
        if scheme=='GUARDED_EXTENDED7':
            tr=area[train]
            if len(tr)==0:raise ValueError('Empty guard-training selection')
            # Training-only group validity and robust guard to prevent noise equalization.
            supported=float(np.mean(tr>1e-6))
            if supported<GUARD_MIN_GROUP_SUPPORT:continue
            positive=tr[tr>EPS]
            if len(positive)==0:continue
            floor=max(0.5*float(np.percentile(positive,20)),
                      1./GUARD_MAX_BOOST,EPS)
            denom=np.maximum(area,floor)
        else:
            denom=np.maximum(area,EPS)
        G[:,region]=TIC[:,region]/denom[:,None]
    return np.asarray(G,np.float32)


def cache_dp(dp):
    """Raw mzML parsing plus scan counts; the recorded spectra are technical observations."""
    from pyteomics import mzml
    cache=OUT_DIR/'CACHE'/f'DP{dp}_PREPROCESSED.joblib'
    fingerprint=source_fingerprint(dp)
    sig=sha(dict(version=VERSION,dp=dp,sigmas=SIGMAS,preprocessor='v66.interpolate_scan',
                 source=fingerprint))
    if cache.exists():
        obj=joblib.load(cache)
        if obj.get('signature')==sig:
            log(f'Loaded DP{dp} calibrated spectral cache')
            return obj['packs'],obj['counts']
    processor=DPSpectrumProcessor(conv_sigma=base.CONV_SIGMA,default_groups=base.GROUP_BOUNDARIES)
    grid=processor.grid
    centers=np.asarray(base.get_preprocess_centers(dp),float)
    relevant=(grid>=base.CLIP_LO)&(grid<=base.DP_TARGET_MZ[dp]+5)
    relevant &= np.any(abs(grid[:,None]-centers[None,:])<=base.PEAK_WINDOW+1e-8,axis=1)
    small=np.asarray(grid[relevant],float)
    batches={e:defaultdict(list) for e in NCES}
    counts=defaultdict(lambda:defaultdict(int))
    for label,path in sorted(source_files(dp).items()):
        log(f'Parsing DP{dp}: {path.name}')
        nom=float(label[1:])/100
        assigned=float(FractionSpectra(dp,nom,dp,grid).NMR_MG_ratio)
        with mzml.read(str(path)) as reader:
            for scan in reader:
                if scan.get('ms level')!=2:continue
                try:
                    precursor=scan['precursorList']['precursor'][0]
                    target=precursor.get('isolationWindow',{}).get('isolation window target m/z')
                    energy=precursor.get('activation',{}).get('collision energy')
                    if target is None or energy is None:continue
                    if abs(float(target)-base.DP_TARGET_MZ[dp])>1e-4:continue
                    e=next((e for e in NCES if abs(float(energy)-e)<1e-4),None)
                    if e is None:continue
                    cc=counts[(dp,e,label)];cc['acquired_target_scans']+=1
                    try:
                        result=v66.interpolate_scan(scan,dp,processor,grid,relevant)
                        if result is None:
                            cc['excluded_invalid_scans']+=1;continue
                        spectra,support,delta=result
                        if spectra[1.].sum()<=EPS:
                            cc['excluded_zero_signal_scans']+=1;continue
                        for s in SIGMAS:batches[e][f'S_{s}'].append(spectra[s])
                        batches[e]['sample'].append(label);batches[e]['nom'].append(nom)
                        batches[e]['y'].append(assigned)
                        batches[e]['support'].append(support)
                        cc['retained_scans']+=1
                    except Exception as exc:
                        cc['excluded_processing_error_scans']+=1
                        if cc['excluded_processing_error_scans']<=2:
                            log(f'  Rejected DP{dp}/NCE{e:g}/{label} scan: {exc}')
                except (KeyError,TypeError,ValueError,IndexError):
                    # Without known DP/NCE the scan is not counted as a target scan.
                    continue
    packs={}; rows=[]
    for (d,e,label),c in sorted(counts.items()):
        row=dict(dp=d,nce=e,nominal_pct=int(label[1:]),sample=label,
                 **{k:int(c.get(k,0)) for k in ('acquired_target_scans','retained_scans',
                     'excluded_invalid_scans','excluded_zero_signal_scans',
                     'excluded_processing_error_scans')})
        row['excluded_scans']=row['acquired_target_scans']-row['retained_scans']
        rows.append(row)
    for e,b in batches.items():
        if not b['sample']:
            raise RuntimeError(f'No DP{dp}/NCE{e} retained spectra; check mzML precursor metadata')
        pack=dict(dp=dp,nce=e,grid=small,centers=centers,
            masked={s:np.asarray(b[f'S_{s}'],np.float32) for s in SIGMAS},
            sample=np.asarray(b['sample'],str),nom=np.asarray(b['nom'],float),
            y=np.asarray(b['y'],float),support=np.asarray(b['support'],bool))
        if len(np.unique(pack['nom']))!=len(MIXES):
            raise RuntimeError(f'DP{dp}/NCE{e:g} has fewer than 7 retained compositions')
        packs[e]=pack
        log(f'DP{dp}/NCE{e:g}: retained {len(pack["sample"])} scans')
    atomic_joblib(dict(signature=sig,source=fingerprint,packs=packs,counts=rows),cache)
    return packs,rows


def ratios_for(pack,train):
    """Select exactly eight identities from original baseline inside the training subset."""
    return tuple(v66.choose_reference_ratios(pack,np.asarray(train,bool)))


def fit_and_score(pack,train,test,sigma,scheme,ratios,include_models=MODELS):
    Xtrain=preprocessing(pack,sigma,scheme,train=train)
    fit=v66.fit_models(pack,train,Xtrain,ratios,sigma,scheme,1.)
    rows={}
    for name in include_models:
        if name not in fit:continue
        result=v66.summarize_predictions(pack,fit[name],Xtrain[test],test,'HELD',pack['y'][test].mean())
        rows[name]=result
    return rows


def grid_checkpoint(dp,e,hold,sigma,scheme):
    return OUT_DIR/'GRID_CHECKPOINTS'/f'DP{dp}'/f'NCE{int(e)}'/f'H{int(hold*100):03d}'/f'S{int(sigma*100):03d}_{scheme}.csv'


def run_grid(packs,dp,pilot=False,nces=NCES):
    sigmas=(.5,1.) if pilot else SIGMAS
    schemes=('ORIGINAL7','EXTENDED7') if pilot else SCHEMES
    holds=MIXES[:2] if pilot else MIXES
    for e in nces:
        pack=packs[e]
        for hold in holds:
            tr=~np.isclose(pack['nom'],hold);te=~tr
            try:
                ratios=ratios_for(pack,tr)
            except Exception:
                fail=OUT_DIR/'GRID_CHECKPOINTS'/f'DP{dp}'/f'NCE{int(e)}'/f'H{int(hold*100):03d}'/'RATIO_SELECTION.failed.txt'
                fail.parent.mkdir(parents=True,exist_ok=True)
                fail.write_text(traceback.format_exc())
                log(f'No eight reliable ratio identities for DP{dp}/NCE{e:g}/hold{hold:g}; skipping this fold')
                continue
            for sigma in sigmas:
                for scheme in schemes:
                    cp=grid_checkpoint(dp,e,hold,sigma,scheme)
                    if cp.exists():continue
                    try:
                        rows=fit_and_score(pack,tr,te,sigma,scheme,ratios)
                        outputs=[]
                        for name,row in rows.items():
                            row.update(dict(holdout_nominal_pct=int(round(hold*100)),
                                stage='development_grid',selection_stage='preprocessing_development'))
                            outputs.append(row)
                        if len(outputs)!=len(MODELS):raise RuntimeError('Both models missing')
                        atomic_csv(outputs,cp)
                    except Exception:
                        fp=cp.with_suffix('.failed.txt');fp.parent.mkdir(parents=True,exist_ok=True)
                        fp.write_text(traceback.format_exc())
                        log(f'Grid failed DP{dp}/NCE{e:g}/hold{hold:g}/{sigma}/{scheme}')
            log(f'DP{dp}/NCE{e:g}: grid holdout {hold*100:.0f}% complete')


def nested_checkpoint(dp,e,hold):
    return OUT_DIR/'NESTED_CHECKPOINTS'/f'DP{dp}'/f'NCE{int(e)}'/f'H{int(hold*100):03d}.csv'


def run_nested(packs,dp,pilot=False,nces=NCES):
    """Outer 7-composition evaluation with inner 6-composition model selection.

    Candidates are declared in NESTED_CHOICES prior to running. Ratio IDs,
    preprocessing floors, NMF and Ridge are refitted in EACH inner split.
    """
    holds=MIXES[:2] if pilot else MIXES
    # Pilot evaluates fewer OUTER folds, but the SAME inner choices so its
    # checkpoints are safe to resume in a subsequent full overnight run.
    choices=NESTED_CHOICES
    for e in nces:
        pack=packs[e]
        for outer in holds:
            cp=nested_checkpoint(dp,e,outer)
            if cp.exists():continue
            try:
                tr_outer=~np.isclose(pack['nom'],outer);te_outer=~tr_outer
                available=np.unique(pack['nom'][tr_outer])
                if len(available)!=6:raise RuntimeError('Outer training requires six compositions')
                scored=[]
                for sigma,scheme in choices:
                    errors=[];qc=[]
                    try:
                        for inner in available:
                            tr=tr_outer&(~np.isclose(pack['nom'],inner))
                            te=tr_outer&np.isclose(pack['nom'],inner)
                            ratios=ratios_for(pack,tr)
                            candidate=fit_and_score(pack,tr,te,sigma,scheme,ratios,
                                include_models=('K2_PLUS_K8',))['K2_PLUS_K8']
                            errors.append(candidate['abs_error_pp'])
                            qc.append(candidate.get('min_raw_denominator_support_pct',np.nan))
                    except Exception as exc:
                        log(f'Nested inner candidate failed DP{dp}/NCE{e:g}/{scheme}/sigma={sigma}: {str(exc)[:120]}')
                        continue
                    scored.append((float(np.mean(errors)),float(np.max(errors)),sigma,scheme,
                                   float(np.nanmin(qc))))
                if not scored:raise RuntimeError('No valid nested candidates with eight reliable ratios')
                scored.sort(key=lambda x:(x[0],x[1],x[2],x[3]))
                best=scored[0]
                inner_mae,inner_max,sigma,scheme,minqc=best
                ratios=ratios_for(pack,tr_outer)
                models=fit_and_score(pack,tr_outer,te_outer,sigma,scheme,ratios)
                outputs=[]
                for name,row in models.items():
                    row.update(dict(outer_holdout_nominal_pct=int(round(outer*100)),
                        selected_sigma_Da=sigma,selected_scheme=scheme,
                        inner_selected_joint_MAE_pp=inner_mae,
                        inner_max_error_pp=inner_max,
                        inner_min_den_support_pct=minqc,
                        n_inner_compositions=6,n_predeclared_choices=len(choices),
                        stage='nested_outer_test',selection_stage='selection_nested_within_outer_training_only'))
                    outputs.append(row)
                atomic_csv(outputs,cp)
            except Exception:
                fp=cp.with_suffix('.failed.txt');fp.parent.mkdir(parents=True,exist_ok=True)
                fp.write_text(traceback.format_exc())
                log(f'Nested failed DP{dp}/NCE{e:g}/outer{outer:g}')
            log(f'DP{dp}/NCE{e:g}: nested outer {outer*100:.0f}% complete')


def ratio_lists_and_groups(packs,dp):
    outputs=[]; members=[]
    for e,pack in packs.items():
        keep=np.ones(len(pack['nom']),bool)
        try:
            ids=ratios_for(pack,keep)
            _,_,names,info=v66.ratio_data(pack,
                preprocessing(pack,1.,'ORIGINAL7',train=keep))
            for rank,idx in enumerate(ids,1):
                meta=info[idx]
                outputs.append(dict(dp=dp,nce=e,ratio_rank=rank,ratio_index=idx,
                                    ratio_name=str(names[idx]),numerator_mz=meta['p1'],
                                    denominator_mz=meta['p2'],
                                    selected_on='all_7_calibration_compositions'))
        except Exception as exc:log(f'Full ratio list DP{dp}/NCE{e:g} unavailable: {exc}')
    for dp_ in (dp,):
        p=next(iter(packs.values()))
        _,m=extended_membership(dp_,p['grid']);members.extend(m)
    return outputs,members


def compile_report(packs_by_dp=None,scan_rows=None,fixed=None):
    fixed=fixed or DEFAULT_FIXED
    grid_files=sorted((OUT_DIR/'GRID_CHECKPOINTS').rglob('*.csv'))
    nest_files=sorted((OUT_DIR/'NESTED_CHECKPOINTS').rglob('*.csv'))
    gr=pd.concat([pd.read_csv(p) for p in grid_files],ignore_index=True) if grid_files else pd.DataFrame()
    ne=pd.concat([pd.read_csv(p) for p in nest_files],ignore_index=True) if nest_files else pd.DataFrame()
    if scan_rows:
        count_path=OUT_DIR/'V67_SCAN_COUNTS.csv'
        parts=[pd.DataFrame(scan_rows)]
        if count_path.exists():parts.insert(0,pd.read_csv(count_path))
        joined=pd.concat(parts,ignore_index=True)
        joined=joined.drop_duplicates(['dp','nce','nominal_pct'],keep='last')
        atomic_csv(joined.to_dict('records'),count_path)
    if not gr.empty:
        gr.to_csv(OUT_DIR/'V67_GRID_ALL_FOLDS.csv',index=False)
        keys=['dp','nce','gaussian_sigma_Da','group_scheme','model']
        sm=gr.groupby(keys,as_index=False).agg(MAE_pp=('abs_error_pp','mean'),
            max_error_pp=('abs_error_pp','max'),n_holdouts=('abs_error_pp','count'),
            mean_denominator_support_pct=('min_raw_denominator_support_pct','mean'),
            min_denominator_support_pct=('min_raw_denominator_support_pct','min'),
            total_retained_technical_scans=('n_scans','sum'))
        sm['complete_seven_folds']=sm.n_holdouts.eq(7)
        sm.to_csv(OUT_DIR/'V67_GRID_BY_DP_NCE.csv',index=False)
        baseline=sm[(sm.group_scheme=='ORIGINAL7')&np.isclose(sm.gaussian_sigma_Da,1.)]
        baseline.to_csv(OUT_DIR/'V67_ORIGINAL7_BASELINE.csv',index=False)
        # Locked protocol has no DP5-7 tuning; configuration set before their results.
        locked=[]
        for e in NCES:
            f=fixed[str(int(e))]
            part=sm[np.isclose(sm.nce,e)&np.isclose(sm.gaussian_sigma_Da,float(f['sigma']))&
                    (sm.group_scheme==f['scheme'])]
            locked.append(part)
        fixed_df=pd.concat(locked,ignore_index=True)
        fixed_df['protocol_status']='FIXED_BEFORE_DP5_DP7_MODEL_SELECTION'
        fixed_df.to_csv(OUT_DIR/'V67_FIXED_PROTOCOL_FIG3F.csv',index=False)
        # The best-of-grid is diagnostic only; no post-selection accuracy claim.
        joint=sm[(sm.model=='K2_PLUS_K8')&sm.complete_seven_folds]
        selected=joint.sort_values(['dp','nce','MAE_pp','max_error_pp']).groupby(['dp','nce']).head(1)
        selected['NOTICE']='Best-of-grid MAE is a selection-biased development metric'
        selected.to_csv(OUT_DIR/'V67_DEVELOPMENT_GRID_WINNERS.csv',index=False)
        compare=sm.pivot_table(index=['dp','nce','gaussian_sigma_Da','group_scheme'],
            columns='model',values='MAE_pp',aggfunc='first').reset_index()
        if set(MODELS).issubset(compare.columns):
            compare['joint_minus_k8_MAE_pp']=compare['K2_PLUS_K8']-compare['K8_ONLY']
        compare.to_csv(OUT_DIR/'V67_K8_VS_JOINT_COMPARISON.csv',index=False)
    if not ne.empty:
        ne.to_csv(OUT_DIR/'V67_NESTED_OUTER_ALL_FOLDS.csv',index=False)
        summary=ne.groupby(['dp','nce','model'],as_index=False).agg(
            nested_outer_MAE_pp=('abs_error_pp','mean'),
            worst_outer_error_pp=('abs_error_pp','max'),
            n_outer_holdouts=('abs_error_pp','count'),
            min_outer_denom_support_pct=('min_raw_denominator_support_pct','min'),
            mean_scan_SD_pp=('scan_sd_pp','mean'))
        summary['complete_seven_outer_folds']=summary.n_outer_holdouts.eq(7)
        summary.to_csv(OUT_DIR/'V67_NESTED_BY_DP_NCE.csv',index=False)
        choices=ne[ne.model=='K2_PLUS_K8'].groupby(
            ['dp','nce','selected_sigma_Da','selected_scheme'],as_index=False).size()
        choices=choices.rename(columns={'size':'selected_outer_folds'})
        choices.to_csv(OUT_DIR/'V67_NESTED_SELECTION_COUNTS.csv',index=False)
    if packs_by_dp:
        ratios=[];groups=[]
        for dp,packs in packs_by_dp.items():
            r,g=ratio_lists_and_groups(packs,dp);ratios.extend(r);groups.extend(g)
        atomic_csv(ratios,OUT_DIR/'V67_FULL_CALIBRATION_K8_RATIOS.csv')
        atomic_csv(groups,OUT_DIR/'V67_GROUP_MEMBERSHIP.csv')
    draw_heatmaps()
    report=['Alginate V6.7 DP5-DP7 evidence audit',f'Version: {VERSION}',
        'Objective: MAE on independent held-out calibration COMPOSITIONS, not scan-level MAE.',
        'NO usable DP5-DP7 external spectra: do NOT report external DP-specific accuracy.',
        f'Grid candidates: {len(grid_factors())} per DP/NCE; inner-nested candidates: {len(NESTED_CHOICES)}.',
        'Nested outer MAE can support model-selection evaluation; grid winner MAE is optimistic.',
        'Fixed-protocol heatmap is also available, using a recipe locked before DP5-7 results.',
        'Technical spectra (scan counts) are not independent biological or preparation replicates.',
        'Group extension excludes intact precursor bins from newly normalized fragment families.',
        'Normalizations are matched in calibration training and held-out inference.',
        'No external labels used for model training or selection.',
        'When V6.6 finishes, set --transfer-protocol to a predeclared NCE-specific JSON and use a NEW output directory.',
        f'Grid checkpoints: {len(grid_files)}; nested checkpoints: {len(nest_files)}.']
    (OUT_DIR/'V67_EVIDENCE_AUDIT.txt').write_text('\n'.join(report)+'\n')
    log(f'Compiled reports: {OUT_DIR}')


def draw_heatmaps():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:return
    figs=OUT_DIR/'FIGURES';figs.mkdir(parents=True,exist_ok=True)
    for inp,title,name,metric in ((
        OUT_DIR/'V67_ORIGINAL7_BASELINE.csv','Original baseline','BASELINE','MAE_pp'),(
        OUT_DIR/'V67_FIXED_PROTOCOL_FIG3F.csv','Fixed transferred protocol','FIXED_PROTOCOL','MAE_pp'),(
        OUT_DIR/'V67_NESTED_BY_DP_NCE.csv','Nested selection; outer held-out error','NESTED','nested_outer_MAE_pp')):
        if not inp.exists():continue
        df=pd.read_csv(inp);df=df[df.model=='K2_PLUS_K8']
        if 'complete_seven_folds' in df.columns:df=df[df.complete_seven_folds]
        if 'complete_seven_outer_folds' in df.columns:df=df[df.complete_seven_outer_folds]
        if df.empty:continue
        mat=df.pivot_table(index='dp',columns='nce',values=metric,aggfunc='first')
        if not all(d in mat.index for d in DPS) or not all(e in mat.columns for e in NCES):continue
        mat=mat.loc[list(DPS),list(NCES)]
        fig,ax=plt.subplots(figsize=(6.0,4.0))
        img=ax.imshow(mat.values,cmap='Greens',aspect='auto',origin='upper')
        for i in range(len(DPS)):
            for j in range(len(NCES)):
                value=mat.values[i,j]
                ax.text(j,i,f'{value:.2f}' if np.isfinite(value) else 'NA',
                        ha='center',va='center',color='white' if np.isfinite(value) and value>np.nanmedian(mat.values) else 'black')
        ax.set_xticks(range(3),[f'{int(e)}%' for e in NCES]);ax.set_yticks(range(3),[f'DP{d}' for d in DPS])
        ax.set(xlabel='HCD normalized collision energy',ylabel='Degree of polymerization',
               title=title+' (MAE; M percentage points)')
        fig.colorbar(img,ax=ax,label='MAE (percentage points)')
        fig.tight_layout();fig.savefig(figs/f'V67_DP5_DP7_{name}_HEATMAP.png',dpi=190)
        plt.close(fig)


def selftest():
    # Baseline must reproduce source implementation on positive spectral arrays.
    rng=np.random.default_rng(9)
    for dp in DPS:
        grid=np.arange(100.,base.DP_TARGET_MZ[dp]+5.,.2)
        centers=np.asarray(base.get_preprocess_centers(dp),float)
        x=np.zeros((8,len(grid)),np.float32)
        for c in centers:
            x+=rng.uniform(.1,2,size=(len(x),1))*np.exp(-.5*((grid-c)/.6)**2).astype(np.float32)
        pack=dict(dp=dp,grid=grid,masked={1.:x,.5:x,.25:x,0.:x})
        out=preprocessing(pack,1.,'ORIGINAL7')
        expected=np.stack([FractionSpectra.group_area_normalization(grid,row,base.GROUP_BOUNDARIES)
                           for row in x])
        assert np.allclose(out,expected,rtol=1e-6,atol=1e-6)
        ms,rows=extended_membership(dp,grid)
        assert len(ms)>=5
        precursor=float(base.DP_TARGET_MZ[dp])
        assert all(not np.any(mask & (abs(grid-precursor)<base.PEAK_WINDOW)) for _,mask,_ in ms)
        assert len(rows)>0
        p2=preprocessing(pack,1.,'EXTENDED7')
        p3=preprocessing(pack,1.,'GUARDED_EXTENDED7',np.ones(len(x),bool))
        assert np.all(np.isfinite(p2)) and np.all(np.isfinite(p3))
        assert np.min(p2)>=0 and np.min(p3)>=0
    log('Numerical selftest passed: baseline exact; DP5-7 extended groups and guard finite')


def main():
    global DATA_DIR,OUT_DIR,NMFI
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-mode',choices=('selftest','pilot','grid','nested','overnight','report'),
                    default='overnight')
    ap.add_argument('--data-dir',type=Path,default=None)
    ap.add_argument('--out-dir',type=Path,default=None)
    ap.add_argument('--transfer-protocol',type=Path,default=None,
                    help='NCE-> {sigma, scheme} locked before DP5-7 results; see JSON template')
    ap.add_argument('--dp',nargs='+',type=int,choices=DPS,default=list(DPS))
    ap.add_argument('--nces',nargs='+',type=float,choices=NCES,default=list(NCES))
    ap.add_argument('--threads',type=int,default=2)
    ap.add_argument('--nmf-max-iter',type=int,default=NMFI)
    args=ap.parse_args()
    if args.data_dir:DATA_DIR=args.data_dir.expanduser().resolve()
    if args.out_dir:OUT_DIR=args.out_dir.expanduser().resolve()
    NMFI=int(args.nmf_max_iter)
    v66.OUT_DIR=OUT_DIR
    v66.NMF_ITER=NMFI
    v66.SIGMAS=SIGMAS
    core.NMF_MAX_ITER=NMFI
    fixed=fixed_protocol_from_json(args.transfer_protocol)
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=max(1,args.threads))
    except ImportError:pass
    selftest()
    if args.run_mode=='selftest':return
    ensure_manifest(fixed)
    if args.run_mode=='report':compile_report(fixed=fixed);return
    scan_rows=[];packs_by_dp={}
    for dp in args.dp:
        packs,rows=cache_dp(dp);packs_by_dp[dp]=packs;scan_rows.extend(rows)
        if args.run_mode in ('pilot','grid','overnight'):
            run_grid(packs,dp,pilot=args.run_mode=='pilot',nces=args.nces)
        if args.run_mode in ('pilot','nested','overnight'):
            run_nested(packs,dp,pilot=args.run_mode=='pilot',nces=args.nces)
        compile_report(packs_by_dp,scan_rows,fixed)
    log('COMPLETE DP5-7 study')

if __name__=='__main__':main()
