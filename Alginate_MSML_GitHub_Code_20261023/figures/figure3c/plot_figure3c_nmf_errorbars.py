#!/usr/bin/env python3
"""Figure 3c | DP3 HCD-MS/MS NMF activations, seven NMR-calibrated mixtures.

Scientific intent
-----------------
Fit rank-2 NMF once to the complete, FIXED-PROTOCOL DP3/NCE26 calibration
spectra using the identical V6.6/V7.0 matrix construction. Plot the raw
sample-level activations as MEAN ± SAMPLE SD of technical MS/MS scans, with
large visible caps and a separate SD-only detail plot. This is a DESCRIPTIVE
full-calibration visualization, not cross-validated prediction accuracy.

Modes:
  real     load original DP3 calibration mzML via the shipped V6.6 modules;
           no invented NMF values; use --data-dir on Mac.
  csv      replot real scan-level activations previously exported by this script
           (or another CSV with required column names).
  demo     visually test plot with clearly watermarked SYNTHETIC data.
  selftest numerical and graphical sanity test, no original mzML required.

All raw mzML scans grouped by a NOMINAL MIXTURE are technical observations.
Error bars are the within-mixture scan-level SD, not a biological-replicate CI.
Do not interpret x=8% and x=85% as chemically pure endpoints.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parent
MIX_NOMINAL_PCT = (0, 20, 40, 50, 60, 80, 100)
NMR_CALIBRATED_DP3_M = np.array([8.0, 23.4, 38.8, 46.5, 54.2, 69.6, 85.0])
DEFAULT_DATA_DIR = Path.home() / 'Desktop' / 'alginate_experiment'

COLORS = ('#0C9277', '#D16B2D')


def load_real(dp:int, nce:float, data_dir:Path, cache_dir:Path, nmf_iter:int):
    """Exactly V7.0 reference matrix and label-independent rank-two NMF fit.

    V6.6 raw mzML cache collects all NCEs once; repeated runs reuse cache.
    A full-calibration NMF is a descriptive plot, not cross-validation.
    """
    module_dir = HERE.parents[1]  # package root: shared V6.6 and v62_core modules
    sys.path.insert(0,str(module_dir))
    try:
        import alginate_V66_ALL_NCE_OVERNIGHT_IDE as v66
        import v62_core as core
    except ModuleNotFoundError as exc:
        raise RuntimeError('Install requirements.txt before processing mzML. '
                           'The CSV and demo modes do not need pyteomics.') from exc
    v66.DATA_DIR = Path(data_dir).expanduser().resolve()
    v66.OUT_DIR = Path(cache_dir).expanduser().resolve()
    v66.OUT_DIR.mkdir(parents=True,exist_ok=True)
    core.NMF_MAX_ITER = int(nmf_iter)
    v66.NMF_ITER = int(nmf_iter)
    packs = v66.cache_dp(int(dp),external=False)
    e = next((key for key in packs if np.isclose(float(key), float(nce))),None)
    if e is None:
        raise RuntimeError(f'No DP{dp}/NCE{nce:g} pack found in original mzML')
    pack = packs[e]
    if len(np.unique(pack['nom'])) != 7:
        raise RuntimeError('Expected seven nominal calibration compositions')
    # EXACT fixed-protocol configuration in V7.0 Figure 3f:
    #   sigma parameter=1 (legacy Gaussian kernel), ORIGINAL7, lambda=1.
    X = v66.matrix(pack,1.,'ORIGINAL7',1.)
    # V7.0 NMF uses all-zero labels to prevent true M% from determining the
    # component ordering. raw_coeff=True exposes actual NMF activations W.
    tr = np.ones(len(pack['y']),dtype=bool)
    model = core.fit_rank2(X, np.zeros_like(pack['y']),pack['nom'],tr,
                           weighted=False,raw_coeff=True)
    activation,_ = model.features_residual(X)
    if activation.shape != (len(pack['y']),2):
        raise RuntimeError('Expected two NMF activation columns')
    if not np.all(np.isfinite(activation)):
        raise RuntimeError('NMF activations contain nonfinite values')

    df = pd.DataFrame({
        'scan_index': np.arange(len(pack['y']),dtype=int),
        'nominal_M_rich_stock_pct': np.rint(pack['nom']*100).astype(int),
        'nmr_calibrated_M_pct': np.asarray(pack['y'],float)*100,
        'component_1_activation': activation[:,0],
        'component_2_activation': activation[:,1],
    })
    meta = dict(data_source='Original calibration mzML; V6.6/V7.0 reference modules',
                dp=int(dp),nce=float(nce),
                nominal_sigma_parameter=1.0,
                gaussian_width_note='Legacy kernel effective physical width differs from nominal sigma; do not silently reinterpret 1.0 as 1 Da.',
                group_scheme='ORIGINAL7',lambda_group_norm=1.0,
                nmf_components=2,nmf_solver='coordinate descent',
                nmf_init='nndsvda',nmf_max_iter=int(nmf_iter),
                nmf_feature_type='raw component activations',
                model_fitting='one unsupervised NMF on all calibration spectra; no external labels',
                n_scans=int(len(df)),n_independent_composition_groups=7)
    return df,meta


def synthetic_demo():
    # Only a visual/wiring preview. Never use for scientific claims.
    rng=np.random.default_rng(20261021)
    rows=[]
    for nominal, m in zip(MIX_NOMINAL_PCT,NMR_CALIBRATED_DP3_M):
        for i in range(75):
            c1=.024+.00023*(m-8)+rng.normal(0,.00065)
            c2=.031-.00038*(m-8)+rng.normal(0,.0008)
            rows.append((i,nominal,m,max(0,c1),max(0,c2)))
    df=pd.DataFrame(rows,columns=['scan_index','nominal_M_rich_stock_pct',
                 'nmr_calibrated_M_pct','component_1_activation','component_2_activation'])
    return df,dict(data_source='SYNTHETIC DEMONSTRATION ONLY — NOT EXPERIMENTAL DATA',
                   dp=3,nce=26,synthetic=True)


def validate(df:pd.DataFrame):
    needed=['nominal_M_rich_stock_pct','nmr_calibrated_M_pct',
            'component_1_activation','component_2_activation']
    missing=[col for col in needed if col not in df]
    if missing:raise ValueError('Missing columns in activation CSV: '+', '.join(missing))
    copy=df.copy()
    for c in needed:copy[c]=pd.to_numeric(copy[c],errors='raise')
    if not np.isfinite(copy[needed].to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite mixture label or NMF activation values')
    if (copy[['component_1_activation','component_2_activation']]<0).any().any():
        raise ValueError('NMF activations must be nonnegative')
    groups=copy.groupby('nominal_M_rich_stock_pct',sort=True)
    if len(groups)!=7:
        raise ValueError(f'Expected seven calibration groups; found {len(groups)}')
    # Each mixture has one chemically assigned NMR-calibrated label.
    for group, g in groups:
        if not np.allclose(g.nmr_calibrated_M_pct, g.nmr_calibrated_M_pct.iloc[0],atol=1e-6):
            raise ValueError(f'Mixture {group} has varying NMR labels')
        if len(g)<2:
            raise ValueError('At least two technical scans are required for sample SD')
    return copy


def summarize(df):
    df=validate(df)
    records=[]
    for nominal,g in df.groupby('nominal_M_rich_stock_pct',sort=True):
        row={'nominal_M_rich_stock_pct':float(nominal),
             'nmr_calibrated_M_pct':float(g.nmr_calibrated_M_pct.iloc[0]),
             'n_technical_scans':int(len(g))}
        for i in (1,2):
            s=g[f'component_{i}_activation'].to_numpy(float)
            row[f'component_{i}_mean']=float(np.mean(s))
            row[f'component_{i}_sd']=float(np.std(s,ddof=1))
            row[f'component_{i}_median']=float(np.median(s))
            row[f'component_{i}_q10']=float(np.quantile(s,.1))
            row[f'component_{i}_q90']=float(np.quantile(s,.9))
        records.append(row)
    out=pd.DataFrame(records).sort_values('nmr_calibrated_M_pct').reset_index(drop=True)
    if len(np.unique(out.nmr_calibrated_M_pct))!=7:
        raise ValueError('NMR composition labels not uniquely ordered')
    # Descriptive orientation only; fitting never uses labels.
    x=out.nmr_calibrated_M_pct.to_numpy(float)
    correlations=[]
    for i in (1,2):
        z=out[f'component_{i}_mean'].to_numpy(float)
        correlations.append(float(np.corrcoef(x,z)[0,1]) if np.std(z)>0 else 0.)
    # Sort on positive Pearson association. A semantic M/G label is only used
    # if signs differ, since NMF component identity is not physically fixed.
    order=list(np.argsort(correlations)[::-1]+1)
    return out,order,correlations


def _add_error_bar(ax,x,y,sd,color):
    # Draw BLACK error bars first (not obscured by the colored line/markers).
    ax.errorbar(x,y,yerr=sd,fmt='none',ecolor='#303743',elinewidth=2.2,
                capsize=6.5,capthick=2.2,zorder=4)
    ax.plot(x,y,'-o',color=color,lw=2.0,markersize=7.1,
            markerfacecolor='white',markeredgecolor=color,
            markeredgewidth=2.2,zorder=5)
    # The translucent band is a secondary cue; the error bars retain true SD.
    ax.fill_between(x,y-sd,y+sd,color=color,alpha=.10,linewidth=0,zorder=2)


def plot_publication(summary,order,corr,dp,nce,out_base,synthetic=False):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,
                         'axes.titlesize':12.0,'axes.labelsize':11.5,
                         'xtick.labelsize':10,'ytick.labelsize':10,
                         'pdf.fonttype':42,'ps.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    x=summary.nmr_calibrated_M_pct.to_numpy(float)
    fig,axs=plt.subplots(1,2,figsize=(10.6,3.9),constrained_layout=True)
    xticks=np.arange(10,81,10)
    for col, (ax,j) in enumerate(zip(axs,order)):
        y=summary[f'component_{j}_mean'].to_numpy(float)
        sd=summary[f'component_{j}_sd'].to_numpy(float)
        color=COLORS[col]
        _add_error_bar(ax,x,y,sd,color)
        # Neutral labels: no chemical M/G assignment to mathematical factors.
        # W1/W2 refer to the DISPLAY order; raw component ids are saved
        # separately in the metadata's component_display_order mapping.
        ax.set_title(f'NMF component (W{col+1})', loc='left', weight='bold', pad=10)
        ax.set_ylabel('NMF weight (a.u.)')
        ax.set_xlabel('Ground truth (M fraction%)')
        # Tick LABELS stay at 10,20,...,80 as requested, but axis LIMITS
        # must include the actual end compositions (8% and 85%).
        # Previously set_xlim(10, 80) silently clipped two measured points.
        ax.set_xlim(min(5.0, float(np.min(x))-3.0),
                     max(88.0, float(np.max(x))+3.0))
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(v) for v in xticks], rotation=0)
        ax.grid(axis='y',alpha=.18,linestyle='--')
        ax.set_axisbelow(True)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        # All actual error bars are shown on the same y-axis as their means;
        # do not magnify or manufacture variability.
        ymin=np.min(y-sd);ymax=np.max(y+sd)
        delta=max(ymax-ymin,1e-8)
        ax.set_ylim(max(0,ymin-.10*delta),ymax+.10*delta)
    if synthetic:
        fig.text(.5,.49,'SYNTHETIC PREVIEW — NOT REAL DATA',ha='center',va='center',
                 fontsize=22,color='#AA2525',alpha=.32,rotation=14,weight='bold')
    files=[]
    for fmt in ('png','pdf','svg'):
        f=out_base.with_name(out_base.name+'_mean_SD').with_suffix('.'+fmt)
        fig.savefig(f,dpi=600 if fmt=='png' else None,facecolor='white',bbox_inches='tight',pad_inches=.15)
        files.append(f)
    plt.close(fig)
    return files
def plot_sd_detail(summary,order,dp,nce,out_base,synthetic=False):
    """Standalone uncertainty-only plot makes even tiny genuine SDs readable."""
    x=summary.nmr_calibrated_M_pct.to_numpy(float)
    fig,axs=plt.subplots(1,2,figsize=(10.6,2.6),constrained_layout=True)
    for k, (ax,j) in enumerate(zip(axs,order)):
        sd=summary[f'component_{j}_sd'].to_numpy(float)
        ax.bar(range(7),sd,color=COLORS[k],width=.62,alpha=.82)
        ax.set_title(f'W{k+1}: technical-scan SD',loc='left',fontsize=12,weight='bold')
        ax.set_ylabel('NMF weight SD (a.u.)')
        ax.set_xlabel('NMR-calibrated M fraction (%)')
        ax.set_xticks(range(7),[f'{a:g}' for a in x],rotation=0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_ylim(0,max(sd)*1.24 if max(sd)>0 else 1.0)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.19,linestyle='--')
    fig.suptitle(f'DP{dp} / NCE{int(nce)}% — uncertainty detail',fontsize=13,weight='bold')
    if synthetic:fig.text(.5,.48,'SYNTHETIC DATA',ha='center',color='#A52A2A',fontsize=24,alpha=.25,rotation=13)
    files=[]
    for fmt in ('png','pdf','svg'):
        f=out_base.with_name(out_base.name+'_SD_detail').with_suffix('.'+fmt)
        fig.savefig(f,dpi=600 if fmt=='png' else None,facecolor='white',bbox_inches='tight',pad_inches=.15)
        files.append(f)
    plt.close(fig)
    return files


def run(df,meta,out_dir,dp,nce,synthetic=False):
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    summary,order,corr=summarize(df)
    prefix=f'Figure3c_DP{dp}_NCE{int(nce)}'
    base=out_dir/prefix
    df.to_csv(out_dir/(prefix+'_scan_level.csv'),index=False)
    summary.to_csv(out_dir/(prefix+'_summary_mean_SD.csv'),index=False)
    files=plot_publication(summary,order,corr,dp,nce,base,synthetic)
    files+=plot_sd_detail(summary,order,dp,nce,base,synthetic)
    meta=dict(meta,component_display_order=[int(a) for a in order],
              figure_component_labels={'W1':'display-first neutral NMF component', 'W2':'display-second neutral NMF component'},
              component_pearson_correlations_with_NMR_M_pct=[float(a) for a in corr],
              count_by_nominal_stock_mix={str(int(row.nominal_M_rich_stock_pct)):int(row.n_technical_scans) for row in summary.itertuples()},
              errorbars='mean ± sample SD across technical MS/MS scans',
              caveat='technical scan SD is NOT independent sample replicate uncertainty',
              files=[p.name for p in files])
    (out_dir/(prefix+'_metadata.json')).write_text(json.dumps(meta,indent=2))
    captions=(f'Fig. 3c | Global rank-two NMF weight trends for DP{dp} at NCE {nce:g}%. '
              f'Each marker represents the mean NMF weight of the selected precursor spectra '
              f'from one of seven NMR-calibrated fraction mixtures. Error bars indicate ±1 '
              f'standard deviation across the {int(summary.n_technical_scans.min())}–'
              f'{int(summary.n_technical_scans.max())} retained technical MS/MS scans per mixture '
              f'(counts listed in source-data table). Components are ordered descriptively by '
              f'correlation with M fraction; W1 and W2 are display labels, not chemical species. NMF fitting itself uses no composition labels. '
              f'Technical-scan error bars do not quantify between-preparation variability.\n')
    if synthetic:captions='SYNTHETIC PREVIEW ONLY. DO NOT USE IN MANUSCRIPT.\n'+captions
    (out_dir/(prefix+'_caption.txt')).write_text(captions)
    print(captions)
    print(f'Output folder: {out_dir.resolve()}')
    for f in files:print(f'  {f.name}')
    return summary


def selftest():
    df,meta=synthetic_demo()
    summary,order,corr=summarize(df)
    assert len(summary)==7 and len(order)==2
    assert (summary.n_technical_scans==75).all()
    assert (summary[['component_1_sd','component_2_sd']].to_numpy()>0).all()
    assert (summary.nmr_calibrated_M_pct.to_numpy()==NMR_CALIBRATED_DP3_M).all()
    return df,meta


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode',choices=('real','csv','demo','selftest'),default='real')
    p.add_argument('--data-dir',type=Path,default=DEFAULT_DATA_DIR,
                   help='Folder containing original Raw_F3_M0G100.mzML etc.')
    p.add_argument('--cache-dir',type=Path,default=HERE/'cache_v66',
                   help='Signature-checked legacy mzML intermediate cache')
    p.add_argument('--input-csv',type=Path,
                   help='Scan-level activation CSV; --mode csv')
    p.add_argument('--out-dir',type=Path,default=HERE/'results')
    p.add_argument('--dp',type=int,choices=tuple(range(3,8)),default=3)
    p.add_argument('--nce',type=float,choices=(18.,22.,26.),default=26.)
    p.add_argument('--nmf-max-iter',type=int,default=650)
    a=p.parse_args()
    if a.mode=='selftest':
        selftest();print('PASS: grouping, technical-scan SD and composition labels');return
    if a.mode=='demo':
        if a.dp!=3 or a.nce!=26.:
            p.error('Demo represents DP3/NCE26 only')
        df,meta=selftest()
    elif a.mode=='csv':
        if not a.input_csv or not a.input_csv.exists():
            p.error('--mode csv requires a valid --input-csv')
        df=pd.read_csv(a.input_csv)
        meta={'data_source':str(a.input_csv),'plot_only':True,
              'note':'NMF source/model should be verified against source data'}
    else:
        df,meta=load_real(a.dp,a.nce,a.data_dir,a.cache_dir,a.nmf_max_iter)
    run(df,meta,a.out_dir,a.dp,a.nce,synthetic=(a.mode=='demo'))


if __name__=='__main__':main()
