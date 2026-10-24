# Selected existing result exports (not re-run)

These CSVs were copied unchanged from user-supplied V6.6, V6.7 and V6.8 outputs. No original mzML datasets were available for re-running in the package creation environment.

- **v66** — original DP3/DP4 LOCO baseline, calibration-selected preprocessing and common NCE summary.
- **v67** — DP5–DP7 fixed-preprocessing results used to complete Figure 3f, nested selection, ratio-only ablation, scan counts, and DP-specific eligible normalization groups.
- **v68** — DP3/NCE26 calibration ablations and external model summaries. These are *exploratory development data*; external labels are not DP-specific truth.

Calibration LOCO is based on 7 composition-level units per DP/NCE (technical MS/MS scans are not independent labeled samples). Do not combine post-hoc search minima with fixed-protocol or nested estimates without clear separation.

All three external material bulk NMR references are DP-unresolved; do not compute a DP-specific accuracy or external MAE from them.


**V7.0 update (2026-10-22):** `v70/` now contains source CSVs supplied from completed real-data V7.0 runs. Results are included as provided, not independently regenerated. `V70_GLOBAL_PERMUTATION_TEST.csv` and `V70_ALL_PERMUTATION_RESULTS.csv` include null permutations with a high rate of fallback behavior; see manuscript limitations and do not treat p-value as proof of externally validated accuracy. External bulk NMR tables are contextual, DP-unresolved measurements.
