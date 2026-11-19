"""Numerical primitives for the predefined DP3–DP7 NMF + K8 + Ridge models.

The eight fragment-ion ratios per DP are defined in configs/models.json.
NMF and regression parameters are fitted only on training compositions.
"""
from __future__ import annotations
from pathlib import Path
import hashlib
import re
import warnings
from dataclasses import dataclass
import numpy as np
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from .config import selected_model
from .preprocessing import CENTERS_ALL, fullrange_smooth, tic_normalize

NCES = (18, 22, 26)
NCE = NCES
COMPOSITIONS = (0, 20, 40, 50, 60, 80, 100)
GROUPS = COMPOSITIONS
EPS = 1e-12
DROP_MZ = (193.035, 351.057, 369.067)

def centered_nmf_ion_indices(grid_mz, dp, precursor_mz):
    """DP3/4 ion-center NMF input with three excluded fragment centers."""
    if dp not in (3, 4):
        raise ValueError("Centered-window NMF ion input applies to DP3/DP4")
    centers = CENTERS_ALL[CENTERS_ALL <= precursor_mz + .001]
    kept = np.array([c for c in centers if not any(abs(c - d) < .3 for d in DROP_MZ)])
    grid_mz = np.asarray(grid_mz, float)
    if len(kept) < 2:
        raise ValueError("Too few retained ion centers")
    idx = np.array([int(np.argmin(np.abs(grid_mz - c))) for c in kept], int)
    if len(np.unique(idx)) != len(idx):
        raise ValueError("Selected ion centers collapse to the same input bin")
    if np.max(np.abs(grid_mz[idx] - kept)) > .05:
        raise ValueError("m/z grid does not contain the selected ion centers")
    return idx, kept


def extract_fullrange_ions(matrix, bin_indices):
    """Fragment-ion signal: sum each SI-fixed 0.5-Da center and ±1 neighbors."""
    X = np.asarray(matrix, float)
    i = np.asarray(bin_indices, int)
    if X.ndim != 2 or len(i) < 2 or i.min() < 0 or i.max() >= X.shape[1]:
        raise ValueError("Invalid full-range ion input/bin positions")
    return sum(X[:, np.clip(i + off, 0, X.shape[1]-1)] for off in (-1, 0, 1))


@dataclass
class Rank2Model:
    """Frozen fit object: transform must not learn from evaluation scans."""
    method: str
    active_columns: np.ndarray
    input_scale: float
    components: np.ndarray
    n_iter: int
    converged: bool
    max_iter: int

    def transform(self, inputs):
        X = np.maximum(np.asarray(inputs, float), 0.)
        if X.ndim != 2 or X.shape[1] != len(self.active_columns):
            raise ValueError("NMF feature input has different number of columns")
        A = X[:, self.active_columns] / self.input_scale
        if self.method == "learned":
            # Apply the fitted NMF basis without changing its solver settings.
            decoder = NMF(n_components=2, init='nndsvda', solver='cd',
                          max_iter=self.max_iter, tol=1e-4, random_state=42)
            decoder.components_ = self.components.copy()
            decoder.n_features_in_ = self.components.shape[1]
            H = decoder.transform(A)
        else:
            raise ValueError("Unknown NMF method")
        return np.maximum(H, 0.)


def fit_rank2(inputs, train_mask, nominal_m_pct, nmr_m_pct, method,
              seed=42, max_iter=2500):
    """Fit training-only learned rank-2 NMF.

    Orient coefficients by their correlation with training-set M fraction.
    Labels orient components but do not fit the NMF basis. All NMR fractions
    are expressed in percentage points.
    """
    X = np.asarray(inputs, float)
    train = np.asarray(train_mask, bool)
    nominal = np.asarray(nominal_m_pct, int)
    y = np.asarray(nmr_m_pct, float)
    if X.ndim != 2 or len(train) != len(X) or len(nominal) != len(X) or len(y) != len(X):
        raise ValueError("Inconsistent NMF matrices and training labels")
    if not train.any() or not np.isfinite(y[train]).all() or not np.isfinite(X).all():
        raise ValueError("Invalid training set for NMF")
    if not np.all(X >= 0.):
        raise ValueError("NMF input has negative values")
    if method != 'learned':
        raise ValueError('Publication code supports learned rank-2 NMF only')
    active = np.any(X[train] > 1e-15, axis=0)
    if active.sum() < 2:
        raise ValueError("Fewer than 2 NMF-active columns")
    A = np.maximum(X[:, active], 0.)
    scale = max(float(A[train].mean()), EPS)
    A = A / scale
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = NMF(n_components=2, init="nndsvda", solver="cd",
                    tol=1e-4, max_iter=max_iter, random_state=seed)
        model.fit(A[train])
        W = np.maximum(model.components_, 0.)
        H = np.maximum(model.transform(A), 0.)
        nit = int(model.n_iter_)
        converged = nit < max_iter
    areas = np.maximum(W.sum(axis=1), EPS)
    shares = H * areas
    shares /= np.maximum(shares.sum(axis=1, keepdims=True), EPS)
    corr = (float(np.corrcoef(shares[train, 0], y[train])[0, 1])
            if np.std(shares[train, 0]) > EPS else 0.)
    order = [0, 1] if np.isfinite(corr) and corr >= 0 else [1, 0]
    fit = Rank2Model(method=method, active_columns=active, input_scale=scale,
                     components=W[order], n_iter=nit, converged=converged,
                     max_iter=int(max_iter))
    # Reuse coefficients from the fitted NMF to avoid numerical re-projection drift.
    return fit, np.asarray(H[:, order], float)



# --- K8 ion ratios: predefined SI identities, never selected during LOO ---

NCE = (18, 22, 26)
GROUPS = (0, 20, 40, 50, 60, 80, 100)
RATIO_PATTERN = re.compile(r'^I_(\d+(?:\.\d+)?)/I_(\d+(?:\.\d+)?)$')


@dataclass(frozen=True)
class SelectedK8:
    """Eight SI-fixed ion ratio definitions; never inferred from sample data."""
    ion_bins: np.ndarray
    ion_mz: np.ndarray
    pairs: np.ndarray
    names: tuple[str, ...]
    nmf_ion_bins: np.ndarray | None = None


def fixed_k8_from_grid(grid_mz, dp, ratio_names):
    """Map 8 named SI ratios to a precise, shared 0.5-Da grid.

    Names, directions, and order come from configs/models.json. The current
    fold, NMR labels, and spectral intensities are never consulted.
    """
    if int(dp) not in (5, 6, 7):
        raise ValueError('Full-range SI fixed ratios apply to DP5–DP7')
    names=tuple(ratio_names)
    if len(names)!=8 or len(set(names))!=8:
        raise ValueError('Expected exactly eight distinct fixed K8 ratio names')
    grid=np.asarray(grid_mz,float)
    if grid.ndim!=1 or len(grid)<20 or not np.isfinite(grid).all():
        raise ValueError('Invalid full-range grid')
    width=float(selected_model(dp)['bin_width_da'])
    if not np.allclose(np.diff(grid), width, rtol=0, atol=1e-8):
        raise ValueError('Full-range grid spacing does not match Stage 1')
    ion_masses=[];pairs_mz=[]
    for name in names:
        match=RATIO_PATTERN.fullmatch(name)
        if match is None:raise ValueError('Invalid fixed SI ratio: '+name)
        a,b=map(float,match.groups())
        if a==b:raise ValueError('Cannot divide ion intensity by itself')
        pairs_mz.append((a,b));ion_masses.extend((a,b))
    # Ratio ion positions are independent of the NMF input definition.
    centers=sorted(set(ion_masses))
    indices=np.asarray([int(np.argmin(np.abs(grid-mz))) for mz in centers],int)
    if len(set(indices.tolist()))!=len(centers) or np.max(np.abs(grid[indices]-centers))>0.005:
        raise ValueError('A fixed SI fragment center is absent from Stage 1 bins')
    lookup={mz:i for i,mz in enumerate(centers)}
    pairs=np.asarray([(lookup[a],lookup[b]) for a,b in pairs_mz],int)
    return SelectedK8(ion_bins=indices,ion_mz=grid[indices],pairs=pairs,names=names)


def grouped_scan_normalize(raw, mids, dp, *, scheme, lam=1.0, cap=6., min_area=1e-5):
    """Per-scan capped group weighting with five-band or 176-Da partitions."""
    X = tic_normalize(raw)
    if not 0 <= lam <= 1:
        raise ValueError('lambda must be between 0 and 1')
    if lam == 0:
        return X
    lo, hi = 100., float(selected_model(dp)['precursor_mz']) - 4.
    if scheme == 'EQ5':
        edges = np.linspace(lo, hi, 6)
    elif scheme == 'LADDER176':
        edges = [lo]
        p = lo
        while p + 176 < hi - 30:
            p += 176
            edges.append(p)
        edges.append(hi)
        edges = np.asarray(edges, float)
    else:
        raise ValueError(f'Unknown SCAN group scheme: {scheme}')
    group_idx = np.minimum(len(edges)-2, np.searchsorted(edges[1:-1], mids, side='right'))
    area = np.column_stack([X[:, group_idx == g].sum(axis=1) for g in range(len(edges)-1)])
    target = 1./(len(edges)-1)
    boost = np.clip(target/np.maximum(area, min_area), 0., cap)
    boost[area < min_area] = 0.
    weighted = X*np.take(boost, group_idx, axis=1)
    return (1-lam)*X + lam*tic_normalize(weighted)


def ratio_source_matrix(arrays, dp):
    """Return the DP-specific physical K8 spectrum; ratios are defined separately."""
    if dp <= 4:
        return np.asarray(arrays['ratio_input_spectra'], float)
    if dp in (5,6):
        # DP5/6 K8 inputs use the Stage 1 NMF spectral branch.
        return np.asarray(arrays['nmf_spectra'], float)
    if dp == 7:
        # DP7 uses separate NMF (GLOBAL) and K8 (LADDER176) normalizations.
        cfg = selected_model(7)
        # Keep K8 settings independent from the NMF input normalization.
        model = cfg['model']
        frozen_k8 = dict(ratio_sigma_da=float(model['ratio_sigma_da']),
                         ratio_group_scheme=str(model['ratio_normalization']).removesuffix('_SCAN'),
                         ratio_lambda=float(model['ratio_lambda']))
        if frozen_k8 != dict(ratio_sigma_da=0.0, ratio_group_scheme='LADDER176', ratio_lambda=1.0):
            raise ValueError('DP7 K8 must use frozen LADDER176 SCAN lambda=1')
        for key, expected in frozen_k8.items():
            if key in cfg and cfg[key] != expected:
                raise ValueError(f'DP7 K8 config mismatch: {key}={cfg[key]!r}, '
                                 f'expected publication comparison value {expected!r}')
        smoothed = fullrange_smooth(arrays['ratio_input_spectra'],
                                    frozen_k8['ratio_sigma_da'], float(cfg['bin_width_da']))
        return grouped_scan_normalize(smoothed, np.asarray(arrays['grid_mz'], float), 7,
                                      scheme=frozen_k8['ratio_group_scheme'],
                                      lam=frozen_k8['ratio_lambda'])
    raise ValueError(f'Unexpected DP{dp}')


def composition_weights(group):
    group=np.asarray(group)
    w=np.zeros(len(group),float)
    for val in np.unique(group):
        m=group==val
        w[m]=1./m.sum()
    return w*len(w)/w.sum()


def read_processed(path):
    """Read preprocessed spectral arrays without permitting Python pickle loading."""
    with np.load(Path(path), allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}
    required = {"nmf_spectra", "ratio_input_spectra", "grid_mz", "nominal_m_pct",
                "nmr_m_pct", "dp", "nce", "sample_set"}
    missing = sorted(required - set(out))
    if missing:
        raise ValueError(f"{path}: missing Stage 1 arrays {missing}")
    if str(out['sample_set']) != 'calibration':
        raise ValueError("Stage 2 fold fitting requires calibration spectra")
    if out['nmf_spectra'].shape != out['ratio_input_spectra'].shape:
        raise ValueError("NMF and ratio spectra must use same raw grid")
    if out['nmf_spectra'].shape[1] != len(out['grid_mz']):
        raise ValueError("Incorrect saved m/z grid")
    if len(out['nominal_m_pct']) != len(out['nmr_m_pct']) or len(out['nominal_m_pct']) != len(out['nmf_spectra']):
        raise ValueError("Spectrum/label count mismatch")
    if not all(np.isfinite(out[k]).all() for k in ('nmf_spectra', 'ratio_input_spectra', 'nmr_m_pct')):
        raise ValueError("Nonfinite Stage 1 matrix or labels")
    if any((out[k] < -1e-12).any() for k in ('nmf_spectra', 'ratio_input_spectra')):
        raise ValueError("Negative spectral matrix values")
    if set(np.asarray(out['nominal_m_pct']).tolist()) != set(COMPOSITIONS):
        raise ValueError("Stage 2 requires all seven calibration composition groups")
    return out



# --- Fixed K8 signals and learned-NMF inputs ---

def centered_signals(spectra, grid_mz, dp):
    """Extract DP3/4 numerator and denominator signals at fixed ion centers.

    The denominator pseudocount is applied in the ratio transformation.
    """
    if dp not in (3,4):
        raise ValueError('Centered-window ratios apply to DP3/DP4')
    X = np.asarray(spectra,float)
    mz = np.asarray(grid_mz,float)
    centers = CENTERS_ALL[CENTERS_ALL <= selected_model(dp)['precursor_mz'] + .001]
    centers = centers[np.abs(centers - selected_model(dp)['precursor_mz']) > 2.]
    lookup = {float(f'{v:g}'):i for i,v in enumerate(centers)}
    indices = [int(np.argmin(abs(mz-c))) for c in centers]
    if np.max(abs(mz[indices]-centers)) > .05:
        raise ValueError('DP3/DP4 centered m/z grid mismatch')
    ions = X[:,indices]
    aa,bb=[],[]
    ratio_names=tuple(selected_model(dp)["model"]["fixed_ratios"])
    for label in ratio_names:
        m=RATIO_PATTERN.fullmatch(label)
        if m is None: raise RuntimeError('Invalid frozen ratio label '+label)
        a,b=float(m.group(1)),float(m.group(2))
        aa.append(ions[:,lookup[a]]);bb.append(ions[:,lookup[b]])
    return np.column_stack(aa),np.column_stack(bb),ratio_names


def ratio_signals(item,dp,shared):
    """Extract the same 8 numerator/denominator ion signals for every transform."""
    source=ratio_source_matrix(item,dp)
    if dp<=4:
        N,D,names=centered_signals(source,item['grid_mz'],dp)
    else:
        selection=shared['selection']
        if selection is None:
            raise RuntimeError('DP5–DP7 require fixed K8 definitions')
        ions=extract_fullrange_ions(source,selection.ion_bins)
        a,b=selection.pairs.T
        N,D=ions[:,a],ions[:,b]
        names=selection.names
    if N.shape!=(len(item['nominal_m_pct']),8) or D.shape!=N.shape:
        raise ValueError('Wrong dimension of eight ratio features')
    if not np.isfinite(N).all() or not np.isfinite(D).all():
        raise ValueError('Nonfinite ion intensities')
    if (N<0).any() or (D<0).any():
        raise ValueError('Negative fragment-ion intensities')
    return N,D,names


def transform_ratios(N,D,dp,mode):
    eps=1e-9 if dp<=4 else 1e-6
    if mode=='raw':
        R=N/(D+eps)
    elif mode=='log1p_raw':
        R=np.log1p(N/(D+eps))
    else:
        raise ValueError('Unsupported ratio transform '+str(mode))
    if not np.isfinite(R).all() or R.shape[1]!=8:
        raise ValueError('Invalid transformed eight-ratio matrix')
    return R


def candidate_nmf_ion_pool(arrays, dp, held=None):
    """Find NMF input peaks from calibration spectra at all three NCEs.

    LOO selection uses only the six training compositions. Full-model fitting
    uses all seven compositions. The eight ratio identities remain fixed.
    """
    if int(dp) not in (5, 7):
        raise ValueError('Candidate ion-pool selection is configured for DP5 and DP7')
    if set(arrays) != set(NCES):
        raise ValueError('Three NCE datasets are required for shared ion-pool discovery')
    grid = np.asarray(arrays[18]['grid_mz'], float)
    if any(not np.array_equal(grid, arrays[nce]['grid_mz']) for nce in NCES):
        raise ValueError('Mass grids differ between NCE datasets')
    entry = selected_model(dp)
    cfg = selected_model_config()['nmf_peak_pool']
    width = float(entry['bin_width_da'])
    if not np.isclose(np.median(np.diff(grid)), width):
        raise ValueError('Stage 1 mass-bin spacing differs from the model configuration')
    edges = np.linspace(100., float(entry['precursor_mz']) - float(cfg['upper_precursor_offset_da']), 4)
    bands = np.minimum(2, np.searchsorted(edges[1:-1], grid, side='right'))
    groups = [int(g) for g in COMPOSITIONS if held is None or int(g) != int(held)]
    qualities, supports = [], []
    for nce in NCES:
        item = arrays[nce]
        B = np.asarray(item['ratio_input_spectra'], float)
        nom = np.asarray(item['nominal_m_pct'], int)
        if set(nom.tolist()) != set(COMPOSITIONS):
            raise ValueError('All seven calibration compositions are required')
        q = np.stack([np.quantile(B[nom == g], .7, axis=0) for g in groups])
        support = np.stack([(B[nom == g] > float(cfg['ion_detection_tic'])).mean(axis=0) for g in groups])
        qualities.append(np.maximum(q.max(axis=0), 1e-15))
        supports.append(support.max(axis=0))
    intensity = np.median(qualities, axis=0)
    support = np.mean(supports, axis=0)
    quality = gaussian_filter1d(intensity, sigma=max(.6, float(cfg['quality_smoothing_da']) / width))
    locations, _ = find_peaks(quality, distance=max(1, int(round(float(cfg['peak_separation_da']) / width))))
    selected = []
    for band in range(3):
        eligible = [int(j) for j in locations if bands[j] == band and support[j] >= float(cfg['minimum_ion_support'])]
        eligible.sort(key=lambda j: (
            -quality[j], -support[j],
            hashlib.sha256(f"{cfg['peak_rank_seed']}:ion:{grid[j]:.4f}".encode()).hexdigest()
        ))
        selected.extend(eligible[:int(cfg['max_ions_per_mass_band'])])
    indices = np.asarray(sorted(selected), dtype=int)
    if len(indices) < 5 or len(set(indices.tolist())) != len(indices):
        raise RuntimeError(f'DP{dp}: too few qualified NMF input peaks')
    return indices


def selected_model_config():
    """Read the single configuration defining calibration and model settings."""
    from .config import load_config
    return load_config()


def chosen_nmf_input(item, dp, selection):
    if dp<=4:
        ix,_=centered_nmf_ion_indices(item['grid_mz'],dp,selected_model(dp)['precursor_mz'])
        return np.asarray(item['nmf_spectra'],float)[:,ix]
    if selection is None:
        raise RuntimeError('Fixed K8 ion definitions are required')
    if dp==6:
        # NMF and ion ratios use different Gaussian widths for DP6.
        raw=np.asarray(item['ratio_input_spectra'],float)
        recipe=selected_model(dp)['model']
        if str(recipe['nmf_normalization']) != 'EQ5_SCAN':
            raise ValueError('DP6 requires five-region per-scan normalization')
        smooth=fullrange_smooth(raw,float(recipe['nmf_sigma_da']),
                                float(selected_model(dp)['bin_width_da']))
        physical=grouped_scan_normalize(smooth,np.asarray(item['grid_mz'],float),
                                        dp,scheme='EQ5',lam=float(recipe['nmf_lambda']))
        indices=selection.ion_bins
    else:
        physical=np.asarray(item['nmf_spectra'],float)
        indices=selection.nmf_ion_bins
        if indices is None:
            raise RuntimeError('The fitted NMF input pool is missing')
    return extract_fullrange_ions(physical, indices)


def matrix_fingerprint(X):
    return hashlib.sha256(np.ascontiguousarray(X,dtype=np.float64).tobytes()).hexdigest()[:16]


