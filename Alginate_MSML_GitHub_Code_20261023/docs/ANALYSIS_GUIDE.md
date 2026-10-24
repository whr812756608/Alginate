# Reproducible experiments and suggested manuscript use

All scripts use the original codebase and do not require notebooks. Run from the repository root with `--data-dir` pointing to the raw-data directory.

| Version | Primary role | Run |
|---|---|---|
| V6.6 | DP3/DP4 3-NCE smoothing/group-strength grid and calibration-selected models | `python alginate_V66_ALL_NCE_OVERNIGHT_IDE.py --run-mode overnight --data-dir /path/to/data` |
| V6.7 | DP5–DP7 fixed, development grid, nested preprocessing selection | `python experiments/alginate_V67_DP5_DP7_OVERNIGHT_IDE.py --run-mode overnight --data-dir /path/to/data` |
| V6.8 | DP3/NCE26 ratio selection, log-ratio and precursor-NMF ablations | `python experiments/alginate_V68_DP3_NCE26_TRANSFER_IDE.py --run-mode overnight --data-dir /path/to/data` |
| V6.9 | DP3 all-NCE per-ratio and ion-support / domain-shift diagnostics | `python experiments/alginate_V69_DP3_ALL_NCE_DIAGNOSTIC_IDE.py --run-mode overnight --data-dir /path/to/data` |
| V7.0 | DP3–DP7 observed vs 199 grouped-label permutations and locked external QC | `python experiments/alginate_V70_EVIDENCE_EXISTING_DATA_IDE.py --run-mode overnight --data-dir /path/to/data` |

All V6.x grid improvements are **model development**, not unbiased out-of-sample performance estimates. V6.7 outer nested validation applies only to DP5–DP7. The recommended primary 15-condition heatmap uses **fixed** `sigma=1.0`, `ORIGINAL7`, `lambda=1.0` and composition-held-out estimates. Compare processing implementations before pooling outputs from separate scripts.

The bundled reference CSVs were supplied as separate versioned result exports. Their provenance is described in `results/reported/README.md`. Model fitting from raw mzML has NOT been re-executed in this packaging step.

## Synthetic acceptance tests

```bash
python tests/test_v66_synthetic.py
python tests/test_v67_synthetic.py
python tests/test_v68_synthetic.py
python tests/test_v69_synthetic.py
python tests/test_v70_synthetic.py
```

Tests validate software integration with **simulated signals only**; their numerical values are not manuscript results. Run each main script in `selftest` mode for additional checks.

## External comparison

External NMR M fractions 12%, 51%, and 82% are measured over the *entire* oligomer population. DP3/DP4 MS–ML estimates are conditional on isolated precursor ions. Differences between the two are **descriptive bulk agreement**, not independent DP-specific external MAE. The exploratory DP4 model selected based on bulk agreement has different preprocessing and should be labeled retrospective if reported.
