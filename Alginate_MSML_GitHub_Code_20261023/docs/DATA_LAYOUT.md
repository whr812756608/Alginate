# Input data layout (real scientific runs)

Place original mzML files under the directory specified by `--data-dir` (or the legacy default `~/Desktop/alginate_experiment`). Do **not** put raw mzML in this Git repository.

The calibration filenames, for each DP in `3,4,5,6,7`, are:

```
Raw_F3_M0G100.mzML
Raw_F3_M20G80.mzML
Raw_F3_M40G60.mzML
Raw_F3_M50G50.mzML
Raw_F3_M60G40.mzML
Raw_F3_M80G20.mzML
Raw_F3_M100G0.mzML
```

Replace `F3` by `F4`, `F5`, `F6`, and `F7` (35 calibration mzML files in total). The `M0` etc. numbers describe **mixing proportions**, NOT the true NMR-calibrated M fraction.

External (if available):

```
AlgG_LowDPMix_1_051625.mzML
AlgMG_LowDPMix_1_051625.mzML
AlgM_LowDPMix_1_051625.mzML
```

Each original external mzML includes multiple precursor DPs and energies. Do not infer an absent energy or precursor from another scan. V7.0 requires only the existing DP3/DP4 external datasets for external predictions, while V6.7 calibrations are DP5–DP7 only.

NMR-derived calibration labels are computed by the original shared processing code from reference stock-fraction compositions. Compare the code values with the final SI calibration tables before publication; do not replace these labels with the nominal mixing percentages.

Make sure the *relative file paths* and `--data-dir` option match the exact script you run. Several old scripts have different defaults, described in `docs/legacy_readmes/`.
