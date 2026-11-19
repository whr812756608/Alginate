from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from .config import selected_model
from .publication_math import (
    NCES, COMPOSITIONS, Rank2Model, SelectedK8, fixed_k8_from_grid,
    read_processed, chosen_nmf_input,
    ratio_signals, transform_ratios, fit_rank2, composition_weights,
    matrix_fingerprint, candidate_nmf_ion_pool,
)

CONFIG_PATH = Path(__file__).resolve().parents[2]/'configs'/'models.json'


def load_publication_config(path=None):
    path = CONFIG_PATH if path is None else Path(path)
    doc = json.loads(path.read_text(encoding='utf8'))
    if set(doc['models']) != {str(i) for i in range(3, 8)}:
        raise ValueError('Configuration must contain exactly DP3–DP7')
    for dp, entry in doc['models'].items():
        c = entry['model']
        if c.get('nmf_method') != 'learned' or c.get('ratio_transform') not in ('raw', 'log1p_raw'):
            raise ValueError(f'DP{dp} violates frozen learned-NMF/raw-or-log1p protocol')
        if c.get('ratio_policy') != 'frozen' or len(c.get('fixed_ratios',()))!=8 or len(set(c['fixed_ratios']))!=8:
            raise ValueError(f'DP{dp} requires exactly eight fixed SI ratio identities')
        expected_mode = {3:'centered_ions',4:'centered_ions',5:'training_peak_pool',
                         6:'fixed_ratio_ions',7:'training_peak_pool'}[int(dp)]
        if c.get('nmf_input_mode') != expected_mode:
            raise ValueError(f'DP{dp} NMF input mode does not match the model definition')
        if float(c['ridge_alpha']) <= 0 or float(c['nmf_gamma']) <= 0:
            raise ValueError('Invalid Ridge/feature-weight settings')
    return doc


def publication_model(dp):
    return load_publication_config()['models'][str(int(dp))]['model']


def configuration_sha256():
    return hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()


def load_calibration(input_dir, dp):
    arrays={}
    for nce in NCES:
        name=f'dp{int(dp)}_nce{nce}_calibration.npz'
        path=Path(input_dir)/name
        if not path.exists():raise FileNotFoundError(f'Missing Stage 1 file: {path}')
        data=read_processed(path)
        if int(data['dp'])!=int(dp) or int(data['nce'])!=nce:
            raise ValueError(f'Metadata mismatch {path}')
        arrays[nce]=data
    return arrays


def select_k8(arrays, dp, held=None):
    if held is not None and int(held) not in COMPOSITIONS:
        raise ValueError('Invalid held-out composition')
    dp=int(dp)
    fixed=tuple(publication_model(dp)['fixed_ratios'])
    if dp<=4:
        return None
    if set(arrays) != set(NCES):
        raise ValueError('Expected three NCE Stage 1 inputs')
    grid=np.asarray(arrays[18]['grid_mz'],float)
    if any(not np.array_equal(grid,arrays[n]['grid_mz']) for n in NCES):
        raise ValueError('NCE Stage 1 grids differ')
    selection=fixed_k8_from_grid(grid,dp,fixed)
    if dp in (5,7):
        nmf_bins=candidate_nmf_ion_pool(arrays,dp,held=held)
        selection=replace(selection,nmf_ion_bins=nmf_bins)
    return selection


@dataclass
class FittedPipeline:
    dp: int
    nce: int
    nmf: Rank2Model
    selection: SelectedK8 | None
    ratio_names: tuple[str, ...]
    transform: str
    gamma: float
    alpha: float
    scaler: StandardScaler
    ridge: Ridge
    converged: bool
    config_hash: str

    def predict(self, data):
        if int(data['dp'])!=self.dp or int(data['nce'])!=self.nce:
            raise ValueError('External DP/NCE differs from fitted model')
        N,D,names=ratio_signals(data,self.dp,{'selection':self.selection})
        if tuple(names)!=self.ratio_names:
            raise RuntimeError('External K8 ion identities do not match final model')
        R=transform_ratios(N,D,self.dp,self.transform)
        H=self.nmf.transform(chosen_nmf_input(data,self.dp,self.selection))
        X=np.column_stack((H,R))
        Z=self.scaler.transform(X)
        Z[:,:2] *= self.gamma
        raw=self.ridge.predict(Z)*100.
        return np.clip(raw,0.,100.),raw,R,H


def fit_pipeline(data, dp, nce, selection, held=None):
    cfg=publication_model(dp)
    if int(data['dp'])!=dp or int(data['nce'])!=nce:
        raise ValueError('Stage-1 data mismatch')
    nom=np.asarray(data['nominal_m_pct'],int)
    y=np.asarray(data['nmr_m_pct'],float)
    if set(nom.tolist())!=set(COMPOSITIONS):
        raise ValueError('Seven calibration compositions are required')
    train=np.ones(len(nom),bool) if held is None else nom!=int(held)
    test=~train
    if held is not None and (not test.any() or set(nom[train].tolist()) != set(COMPOSITIONS)-{held}):
        raise ValueError('Missing held-out composition')
    N,D,names=ratio_signals(data,dp,{'selection':selection})
    if tuple(names)!=tuple(cfg['fixed_ratios']):
        raise RuntimeError('K8 ratio names differ from SI frozen configuration')
    R=transform_ratios(N,D,dp,cfg['ratio_transform'])
    fit,H=fit_rank2(chosen_nmf_input(data,dp,selection),train,nom,y,
                    method='learned',seed=int(cfg['nmf_seed']),
                    max_iter=int(cfg['nmf_max_iter']))
    X=np.column_stack((H,R))
    if X.shape!=(len(nom),10) or not np.isfinite(X).all():
        raise RuntimeError('Expected finite K2+K8 feature matrix')
    weights=composition_weights(nom[train])
    scaler=StandardScaler().fit(X[train],sample_weight=weights)
    Z=scaler.transform(X)
    Z[:,:2]*=float(cfg['nmf_gamma'])
    ridge=Ridge(alpha=float(cfg['ridge_alpha']))
    ridge.fit(Z[train],y[train]/100.,sample_weight=weights)
    proc=FittedPipeline(dp,nce,fit,selection,tuple(names),cfg['ratio_transform'],
                        float(cfg['nmf_gamma']),float(cfg['ridge_alpha']),
                        scaler,ridge,bool(fit.converged),configuration_sha256())
    metadata=dict(features=X,nmf_coefficients=H,ion_ratios=R,
          ratio_names=np.asarray(names,str),
          nmf_method=np.asarray('learned'),ratio_transform=np.asarray(cfg['ratio_transform']),
          ratio_policy=np.asarray(cfg['ratio_policy']),
          nominal_m_pct=nom,nmr_m_pct=y,training_mask=train,heldout_mask=test,
          heldout_nominal_pct=np.asarray(-1 if held is None else held),
          dp=np.asarray(dp),nce=np.asarray(nce),
          nmf_n_iter=np.asarray(int(fit.n_iter)),
          nmf_max_iter=np.asarray(int(fit.max_iter)),
          nmf_converged=np.asarray(bool(fit.converged)),
          nmf_seed=np.asarray(int(cfg['nmf_seed'])),
          nmf_components=np.asarray(fit.components),
          nmf_active_columns=np.asarray(fit.active_columns),
          nmf_input_scale=np.asarray(fit.input_scale),
          ratio_fingerprint=np.asarray(matrix_fingerprint(R)),
          model_config_sha256=np.asarray(proc.config_hash),
          sample_id=np.asarray(data.get('sample_id', nom.astype(str)),str),
          nmf_sigma_da=np.asarray(float(cfg['nmf_sigma_da'])),
          ratio_sigma_da=np.asarray(float(cfg['ratio_sigma_da'])),
          gamma=np.asarray(proc.gamma),ridge_alpha=np.asarray(proc.alpha))
    if selection is not None:
        if selection.nmf_ion_bins is not None:
            metadata['nmf_ion_bins']=np.asarray(selection.nmf_ion_bins,int)
        metadata['k8_ion_bins']=np.asarray(selection.ion_bins,int)
        metadata['k8_ion_mz']=np.asarray(selection.ion_mz,float)
        metadata['k8_ratio_pairs']=np.asarray(selection.pairs,int)
    return proc,metadata


def evaluate_fold(arrays, dp, held):
    selection=select_k8(arrays,dp,held)
    out=[]
    for nce in NCES:
        item=arrays[nce]
        model,meta=fit_pipeline(item,dp,nce,selection,held)
        y=np.asarray(meta['nmr_m_pct'],float)
        nom=np.asarray(meta['nominal_m_pct'],int)
        te=np.asarray(meta['heldout_mask'],bool)
        Z=model.scaler.transform(meta['features'])
        Z[:,:2] *= model.gamma
        raw=model.ridge.predict(Z)*100.
        pred=np.clip(raw,0.,100.)
        truth=float(np.median(y[te]))
        result=dict(dp=dp,nce=nce,held_nominal_pct=int(held),
            ratio_transform=model.transform,truth_NMR_pct=truth,
            predicted_pct=float(np.mean(pred[te])),
            absolute_error_pp=float(abs(np.mean(pred[te])-truth)),
            clipped_scans_pct=float(np.mean((raw[te]<0)|(raw[te]>100))*100),
            n_test_scans=int(te.sum()),
            ratio_names=';'.join(model.ratio_names),
            k8_digest=str(meta['ratio_fingerprint']),
            nmf_converged=bool(model.converged),nmf_iterations=int(model.nmf.n_iter),
            model_config_sha256=model.config_hash)
        out.append(result)
    return out


def save_pipeline(model,path):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    fit=model.nmf
    contents={
        'format_version':np.asarray('alginate-ms2-fixed-k8-v3'),
        'dp':np.asarray(model.dp),'nce':np.asarray(model.nce),
        'method':np.asarray('learned'),
        'ratio_transform':np.asarray(model.transform),
        'ratio_names':np.asarray(model.ratio_names,str),
        'gamma':np.asarray(model.gamma),'alpha':np.asarray(model.alpha),
        'nmf_active_columns':np.asarray(fit.active_columns,bool),
        'nmf_input_scale':np.asarray(fit.input_scale,float),
        'nmf_components':np.asarray(fit.components,float),
        'nmf_max_iter':np.asarray(fit.max_iter,int),
        'nmf_n_iter':np.asarray(fit.n_iter,int),
        'nmf_converged':np.asarray(model.converged),
        'scaler_mean':np.asarray(model.scaler.mean_,float),
        'scaler_scale':np.asarray(model.scaler.scale_,float),
        'ridge_coef':np.asarray(model.ridge.coef_,float),
        'ridge_intercept':np.asarray(model.ridge.intercept_,float),
        'model_config_sha256':np.asarray(model.config_hash),
        'k8_ion_bins':np.asarray(model.selection.ion_bins,int) if model.selection is not None else np.array([],int),
        'k8_ion_mz':np.asarray(model.selection.ion_mz,float) if model.selection is not None else np.array([],float),
        'k8_ratio_pairs':np.asarray(model.selection.pairs,int) if model.selection is not None else np.zeros((0,2),int),
        'nmf_ion_bins':np.asarray(model.selection.nmf_ion_bins,int) if model.selection is not None and model.selection.nmf_ion_bins is not None else np.array([],int),
    }
    np.savez_compressed(p,**contents)
    return p


def load_pipeline(path,*,verify_config=True):
    with np.load(Path(path),allow_pickle=False) as z:
        d={k:z[k] for k in z.files}
    if str(d['format_version'])!='alginate-ms2-fixed-k8-v3' or str(d['method'])!='learned':
        raise ValueError('Unsupported or unsafe model format')
    h=str(d['model_config_sha256'])
    if verify_config and h!=configuration_sha256():
        raise ValueError('Publication configuration differs from model training snapshot')
    dp=int(d['dp'])
    maxiter=int(d['nmf_max_iter'])
    nmf=Rank2Model('learned',np.asarray(d['nmf_active_columns'],bool),
                   float(d['nmf_input_scale']),np.asarray(d['nmf_components'],float),
                   int(d['nmf_n_iter']),bool(d['nmf_converged']),maxiter)
    sel=None
    if dp>=5:
        sel=SelectedK8(ion_bins=np.asarray(d['k8_ion_bins'],int),
                       ion_mz=np.asarray(d['k8_ion_mz'],float),
                       pairs=np.asarray(d['k8_ratio_pairs'],int),
                       names=tuple(map(str,d['ratio_names'].tolist())),
                       nmf_ion_bins=np.asarray(d['nmf_ion_bins'],int) if dp in (5,7) else None)
    if dp in (5,7) and (sel.nmf_ion_bins is None or len(sel.nmf_ion_bins)<5):
        raise ValueError('Saved model NMF ion pool is missing')
    if tuple(map(str,d['ratio_names'].tolist())) != tuple(publication_model(dp)['fixed_ratios']):
        raise ValueError('Saved model K8 identities differ from frozen SI configuration')
    scaler=StandardScaler()
    scaler.mean_=np.asarray(d['scaler_mean'],float)
    scaler.scale_=np.asarray(d['scaler_scale'],float)
    scaler.var_=scaler.scale_**2
    scaler.n_features_in_=len(scaler.mean_)
    ridge=Ridge(alpha=float(d['alpha']))
    ridge.coef_=np.asarray(d['ridge_coef'],float)
    ridge.intercept_=float(d['ridge_intercept'])
    ridge.n_features_in_=len(scaler.mean_)
    return FittedPipeline(dp,int(d['nce']),nmf,sel,
                          tuple(map(str,d['ratio_names'].tolist())),
                          str(d['ratio_transform']),float(d['gamma']),float(d['alpha']),
                          scaler,ridge,bool(d['nmf_converged']),h)
