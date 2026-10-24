#!/usr/bin/env python3
"""Print a read-only calibration / external input availability report."""
import argparse
from pathlib import Path
ap=argparse.ArgumentParser()
ap.add_argument('data_dir',type=Path)
a=ap.parse_args()
for dp in range(3,8):
    for m in (0,20,40,50,60,80,100):
        p=a.data_dir / f'Raw_F{dp}_M{m}G{100-m}.mzML'
        print(('FOUND  ' if p.is_file() else 'MISSING')+' '+str(p))
for file in ('AlgG_LowDPMix_1_051625.mzML','AlgMG_LowDPMix_1_051625.mzML','AlgM_LowDPMix_1_051625.mzML'):
    p=a.data_dir/file
    print(('FOUND  ' if p.is_file() else 'MISSING')+' '+str(p))
