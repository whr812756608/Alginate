#!/usr/bin/env python3
"""Synthetic/layout and data-integrity checks; does not fit real mzML."""
from pathlib import Path
import subprocess,sys,tempfile
import pandas as pd
repo=Path(__file__).resolve().parents[1]
p3c=repo/'figures/figure3c/plot_figure3c_nmf_errorbars.py'
p3ef=repo/'figures/figure3e_f/generate_clean_panel_e_f.py'
subprocess.run([sys.executable,str(p3c),'--mode','selftest'],check=True)
csv=repo/'figures/figure3e_f/source_data/V70_FIG3F_LOCKED_15_CONDITIONS.csv'
df=pd.read_csv(csv)
assert len(df)==15 and set(df.dp)=={3,4,5,6,7} and set(df.nce)=={18.,22.,26.}
with tempfile.TemporaryDirectory() as tmp:
    out=Path(tmp)
    demo=out/'demo';demo.mkdir()
    subprocess.run([sys.executable,str(p3c),'--mode','demo','--out-dir',str(demo)],check=True,stdout=subprocess.DEVNULL)
    svg=(demo/'Figure3c_DP3_NCE26_mean_SD.svg').read_text()
    for label in ('NMF component (W1)','NMF component (W2)','NMF weight (a.u.)','SYNTHETIC PREVIEW'):
        assert label in svg, label
    for stale in ('M-correlated NMF component','G-correlated NMF component'):
        assert stale not in svg

    subprocess.run([sys.executable,str(p3ef),'--out-dir',str(out)],check=True)
    for panel in ('e','f'):
        for ext in ('svg','png','pdf'):
            p=out/f'Figure3{panel}_clean_no_title.{ext}'
            assert p.exists() and p.stat().st_size>1000, p
    e=(out/'Figure3e_clean_no_title.svg').read_text()
    assert '<svg ' in e and '100%' in e
    # No panel letter, title or extra illustration text in the exported artwork.
    assert 'Seven-fold leave-one' not in e
    f=(out/'Figure3f_clean_no_title.svg').read_text()
    assert '<svg ' in f and '9.41' in f
print('PASS: manuscript figure scripts and exported files')
