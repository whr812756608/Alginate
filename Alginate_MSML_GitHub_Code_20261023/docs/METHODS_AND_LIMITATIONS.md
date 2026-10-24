# Methods audit and interpretation limits

This repository reflects successive development versions, not one retrospectively invented uniform fitted model.

### Main methodological framework

1. Use NMR-characterized DP-enriched MM- and GG-rich source fractions to calculate the M fraction of seven mixed-proportion samples per nominal DP. The actual calibration labels are **not** 0, 20, ..., 100% M.
2. Acquire HCD-MS/MS of the DP3–DP7 precursor ions at NCE18/22/26, with technical scan filtering and fragment-informed preprocessing.
3. Extract two nonnegative per-scan NMF activations and eight training-selected fragment intensity ratios; use standardized Ridge regression for M-fraction prediction.
4. Use leave-**one-composition**-out validation (withhold all scans for a held-out mixing proportion; train feature selection, NMF, scaling and Ridge on remaining compositions).
5. Refit a frozen DP/NCE model on all seven calibration mixtures before external predictions; report technical scan variability and spectral QC.

### Essential checks before publication

- **Smoothing units:** review the actual custom Gaussian kernel and its effective bandwidth; a parameter called `sigma=1.0` is not yet independently verified to be a physical 1-Da standard deviation. Do not silently change published values without rerunning.
- **Normalized fragment groups:** fixed ORIGINAL7 group sums, DP-aware extensions and optional partial strength must match the chosen experiment; do not call group-sum normalization local base-peak normalization.
- **Fragment centers:** 34 originally annotated ions versus 24 implemented fragment centers, eligible according to DP.
- **NMF solver:** confirm the exact NNDSVDA initialization, coordinate-descent solver, convergence tolerance, maximum iterations and precursor-region handling in the final script; avoid mixing versions.
- **Scan counts:** the instrument Methods mention 20 scans per precursor, but extracted V6.7 files report hundreds of target MS/MS scans per composition/NCE. Resolve acquisition scheduling and report retained spectra separately from independent mixture units.
- **Leakage:** if selection of sigma/groups/regularization used a large development grid, its minimum LOCO MAE is not an unbiased error for the selection process. Refer to nested selection where actually implemented.
- **Reference mismatch:** DP-resolved M fraction is not measured by external bulk NMR. Do not use 12%, 51%, 82% as DP3/4 training labels or target-specific validation.
- **Null test:** V7.0 permutations preserve whole-sample labels across technical scans. They test a composition-associated signal within the available calibration stocks, not independent source-batch transfer.
- **Independent reference limitations:** SEC size-enriched fractions contain adjacent DPs; source-stock NMR labels are fraction-derived proxies assigned to the selected precursor, not sequence-pure DP-specific standard measurements.

### Manuscript status

The 101926 manuscript and SI are *not* included in the repository and still require edited Methods/SI tables, figure updates and harmonization with the final locked model. These code and result exports should be treated as a research-development snapshot until those checks are resolved.
