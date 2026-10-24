#!/usr/bin/env python3
"""Checks reported CSV consistency; does not substitute for raw-data rerun."""
from pathlib import Path
import ast, re
import numpy as np
import pandas as pd
root=Path(__file__).resolve().parents[1]
source=pd.read_csv(root/'results/reported/v70/V70_COMPOSITION_LEVEL_OOF_PREDICTIONS.csv')
primary=source.loc[source.model=='K2_PLUS_K8']
assert len(primary)==105
assert primary.groupby(['dp','nce']).size().eq(7).all()
assert sorted(primary.dp.unique())==[3,4,5,6,7]
assert sorted(primary.nce.unique())==[18.,22.,26.]
assert primary.n_scans.gt(0).all()
si=pd.read_csv(root/'source_data/supplementary_scan_counts_S9.csv')
assert len(si)==105
assert int(si.included_technical_spectra.sum())==int(primary.n_scans.sum())
fig=pd.read_csv(root/'source_data/figure3f_validation_summary.csv')
assert len(fig)==15
assert fig.observed_MAE_pp.ge(0).all()
for row in fig.itertuples():
    held=primary[(primary.dp==row.dp)&(primary.nce==row.nce)]
    mae=float(held.abs_error_pp.mean())
    assert np.isclose(mae,row.observed_MAE_pp,atol=1e-6), (row.dp,row.nce,mae,row.observed_MAE_pp)
centers=pd.read_csv(root/'source_data/supplementary_fragment_center_annotations.csv')
assert len(centers)==34 and (centers.retained_by_processing_code=='Yes').sum()==24
print('PASS: 15 condition MAEs, 105 composition predictions, scan counts, and 24 of 34 centers')
