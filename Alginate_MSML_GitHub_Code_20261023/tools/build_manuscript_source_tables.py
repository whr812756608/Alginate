#!/usr/bin/env python3
"""Export manuscript-support tables from checked-in analysis exports; NOT a raw-mzML rerun."""
from pathlib import Path
import argparse, ast, re, csv
import pandas as pd

repo=Path(__file__).resolve().parents[1]

def generate(out_dir:Path):
    out_dir.mkdir(parents=True,exist_ok=True)
    heldout=pd.read_csv(repo/'results/reported/v70/V70_COMPOSITION_LEVEL_OOF_PREDICTIONS.csv')
    rows=heldout.loc[heldout.model=='K2_PLUS_K8', ['dp','nce','heldout_nominal_pct','true_M_pct','n_scans']]
    assert len(rows)==105 and rows.groupby(['dp','nce']).size().eq(7).all()
    rows.rename(columns={'heldout_nominal_pct':'nominal_MM_rich_pct','true_M_pct':'NMR_calibrated_M_pct','n_scans':'included_technical_spectra'}).to_csv(out_dir/'supplementary_scan_counts_S9.csv',index=False)
    df=pd.read_csv(repo/'results/reported/v70/V70_FIG3F_LOCKED_15_CONDITIONS.csv')
    assert len(df)==15 and set(df.dp)=={3,4,5,6,7}
    df.to_csv(out_dir/'figure3f_validation_summary.csv',index=False)
    s=(repo/'alginate_ratio_transfer_IDE.py').read_text()
    m=re.search(r'(?ms)^ALL_FILTER_CENTERS\s*=\s*(\[[^\]]+\])',s)
    assert m, 'Missing ALL_FILTER_CENTERS'
    centers=ast.literal_eval(m.group(1))
    assert len(centers)==24
    excluded=[307.0671,333.0463,483.0992,509.0784,527.0890,587.1101,643.0999,659.1313,685.1105,861.1426]
    with (out_dir/'supplementary_fragment_center_annotations.csv').open('w',newline='') as fh:
        w=csv.writer(fh);w.writerow(['theoretical_mz','retained_by_processing_code','source','note'])
        for x in sorted(centers+excluded):
            retained=x in centers
            w.writerow([f'{x:.4f}','Yes' if retained else 'No', 'alginate_ratio_transfer_IDE.py:ALL_FILTER_CENTERS' if retained else 'S4 theoretical-only set', 'Retained center; may be DP-filtered' if retained else 'Not in 24-center code list'])
    print('Created Figure 3f, Supplementary S9 and S4 audit CSVs from existing data and source code.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir',type=Path,default=repo/'outputs/source_tables')
    generate(p.parse_args().out_dir)
