# Alginate V6.9 — DP3 all-NCE spectral-transfer diagnostic

**Standalone PyCharm/Terminal package.** Investigates NCE **18%, 22%, and 26%**, independently. Reads your original DP3 calibration and three independent AOS mzML files; no new external labels or DP-specific NMR measurements are needed.

**This is a diagnostic, not an automatic optimization to match bulk NMR.** The external GG/MG/MM M% values are unknown at DP3. Bulk NMR (12%, 51%, 82%) is not used anywhere in this script for fitting, selecting, filtering, ordering models, or producing uncertainty bounds.

## 1. Folder and requirements

Unzip the entire package into your Mac `~/Desktop/alginate_experiment/` folder. **Keep all five Python source files together**; the V6.6 and legacy processors are included unchanged.

Required local input files, default `DATA_DIR = ~/Desktop/alginate_experiment`:

| Calibration (seven files) | External (three files) |
|---|---|
| `Raw_F3_M0G100.mzML` | `AlgG_LowDPMix_1_051625.mzML` |
| `Raw_F3_M20G80.mzML` | `AlgMG_LowDPMix_1_051625.mzML` |
| `Raw_F3_M40G60.mzML` | `AlgM_LowDPMix_1_051625.mzML` |
| `Raw_F3_M50G50.mzML` | |
| `Raw_F3_M60G40.mzML` | |
| `Raw_F3_M80G20.mzML` | |
| `Raw_F3_M100G0.mzML` | |

Every mzML file should contain targeted DP3 MS/MS scans at the three NCE values. If only one NCE was actually acquired/retained, other NCEs cannot be recovered from it; the runner warns about missing external energies.

```bash
conda activate alginate_experiment
cd ~/Desktop/alginate_experiment/Alginate_V69_DP3_AllNCE_Diagnostics
python -m pip install -r requirements.txt
python alginate_V69_DP3_ALL_NCE_DIAGNOSTIC_IDE.py --run-mode selftest
python test_v69_synthetic.py
python alginate_V69_DP3_ALL_NCE_DIAGNOSTIC_IDE.py --run-mode overnight --threads 2
```

**PyCharm:** open `alginate_V69_DP3_ALL_NCE_DIAGNOSTIC_IDE.py`; choose your existing Conda interpreter, edit `DATA_DIR` near the top if needed, and click **Run**. The default is `overnight`.

Other modes:

- `--run-mode pilot`: only NCE18, both preprocessing references, seven held-out composition folds + external inference. Resumes into overnight.
- `--run-mode external`: fast full-calibration fits and external diagnostic outputs for all NCEs (skips seven-fold LOCO; not suitable for LOCO-based decisions).
- `--run-mode report`: regenerates reports and figures from saved checkpoints, without rereading mzML.
- `--run-mode selftest`: synthetic numerical unit test that needs no mzML.
- `--out-dir /path/to/new/results`: select a clean result folder if inputs/settings change. Checkpoints automatically resume otherwise.

## 2. Frozen reference preprocessing for each NCE

The same eight ratio identities are selected in each composition-training fold from the **V6.6 original preprocessing (σ=1, ORIGINAL7, λ=1)** and are *frozen* across the two arms being compared. The final external fit selects those eight from all seven calibration compositions. There is no external-feature-selection leakage.

| NCE | `ORIGINAL_REFERENCE` | `V66_CAL_SELECTED` |
|---|---|---|
| 18% | σ1.00 / ORIGINAL7 / λ1 | σ0.00 / TIC-only / λ0 |
| 22% | σ1.00 / ORIGINAL7 / λ1 | σ0.25 / ORIGINAL7 / λ0.5 |
| 26% | σ1.00 / ORIGINAL7 / λ1 | σ0.25 / ORIGINAL7 / λ1 |

The V6.6 selected settings come from **calibration-only development**, but their selected-configuration LOCO errors are not unbiased post-selection validation.

The model uses K=2 global NMF scores and the eight local ion ratios for standardized Ridge regression (α=1). A K8-only ablation is included. All eight drop-one-ratio experiments are **refitted**, not simply applied to the existing Ridge coefficients. NMF fitting, ratio selection, and Ridge/scaling are restricted to six training compositions in each held-out evaluation.

## 3. Primary diagnostic questions

1. **Which selected fragment ratios shift on transfer?** The eight ratios, with calibration median/IQR at each of the seven known NMR-derived M% labels, are compared with external GG/MG/MM median/IQR **on a separate categorical axis**, not at invented DP3 M% values. Cal/external shifts outside the pooled calibration 1st–99th percentile are flags, not estimates of error.
2. **Which numerator/denominator ion is unreliable?** Reports show raw-centroid detectability for each ion, joint support, near-zero processed denominators, and local intensity medians.
3. **Which single ratio affects external predictions?** Eight separate ratio-deletion refits per NCE/preprocessing, with seven-fold calibration error, external prediction changes, and a ratio influence summary. No deletion is promoted based on closeness to bulk NMR.
4. **Where do external spectra differ globally?** K2 NMF residual hotspots and q99-based OOD exceedances are computed under the corresponding calibration-fitted NMF basis; model-specific thresholds are not directly comparable as absolute spectral quality.
5. **How stable are the measurements?** Technical scan-level SD, half-file difference, raw regression clipping frequency, and moving-block bootstrap intervals on the **mean technical-scan prediction**. These are **not** uncertainty intervals for the true DP3 M%.
6. **Does the diagnostic change across NCE?** A single table and figure compare all three NCEs for the same external materials.

## 4. Output files (under `V69_DP3_ALL_NCE_DIAGNOSTICS`)

| CSV/figure | Contents |
|---|---|
| `V69_RATIO_ION_DISTRIBUTIONS.csv` | 8 ratios × 7 calibrators + 3 external materials, each NCE and preprocessing; medians, quartiles, ion raw support |
| `V69_EXTERNAL_RATIO_DOMAIN_SHIFT.csv` | External out-of-range fractions and median displacement in calibration IQR units |
| `V69_SELECTED_RATIO_IDENTITY_ALL_NCE.csv` | Full-fit and held-out-fold ion-ratio identities |
| `V69_LOCO_ABLATION_SUMMARY.csv` | Eight drop-one refits + joint and K8-only LOCO MAE per NCE/preprocessing |
| `V69_ABLATION_EXTERNAL_INFLUENCE_VS_LOCO.csv` | External prediction difference for every ablation; matched LOCO error |
| `V69_RATIO_ABLATION_INFLUENCE_SUMMARY.csv` | Per-ratio maximum/mean absolute change in external prediction, and LOCO cost |
| `V69_EXTERNAL_PREDICTIONS_AND_SCAN_BOOTSTRAP.csv` | Joint, K8-only, ablated external predictions, OOD fraction, clipping, technical uncertainty |
| `V69_NCE_CONSISTENCY_AND_OOD.csv` | Same external sample across energies; observed energy range and model-specific OOD |
| `V69_NMF_RESIDUAL_HOTSPOTS.csv` | Top 25 m/z bins per specimen with highest normalized NMF reconstruction residuals |
| `V69_RETAINED_SCAN_COUNTS_BY_SAMPLE_NCE.csv` | Exact retained scans for each calibration/external material at each NCE; these **are not acquired scan counts** |
| `V69_SCAN_COUNTS_AUDIT.csv` | Per-checkpoint retained scans and preprocessor-reported rejection totals, not a complete instrument audit |
| `V69_ALL_PREDICTIONS.csv` | Held-out and external full prediction table |
| `FIGURES/RATIO_OVERVIEW_NCE*_*.png` | Eight paired calibration/external ratio panels, one pair per NCE and preprocessing |
| `FIGURES/ABLATION_EXTERNAL_NCE*_*.png` | Impact of each deletion on GG/MG/MM predictions |
| `FIGURES/EXTERNAL_NCE_TRENDS_TECHNICAL_BOOTSTRAP.png` | NCE-dependent mean and technical bootstrap intervals |
| `V69_REPORT_AND_LIMITATIONS.txt` | Completion audit and interpretation cautions |

### Nature manuscript note: scan counts

The original acquisition method states 20 scheduled scans per precursor, but the exported mzML can yield a larger number of retained DP3 MS/MS spectra aggregated across cycles/files. `n_retained_MS2_spectra` reports the actual number fed to analysis, **not** the number of acquired scans, injections, biological replicates, or independent calibration samples. For the Figure 3 caption, `n = 7` refers to held-out compositions per DP/NCE, while the actual technical spectrum counts should be listed separately by condition. Instrument-level acquisition counts require a distinct audit of raw scan metadata.

## 5. How to interpret the results responsibly

DP3 M% on external samples is **not independently known**. The bulk-NMR values are whole-distribution averages. A 31% DP3 estimate can be physically compatible with 12% whole-sample M if DP3 contributes a sufficiently small residue fraction. Conversely, stable predicted M% does **not** guarantee accuracy if the external fragment relationships differ from calibration.

Good diagnostic progress = stronger matched-calibrator feature support; fewer severe ratio distribution anomalies attributable to technical factors; better held-out calibration behavior; and stable external inference. If the external fragment patterns are outside the calibration domain, flag the estimate rather than choosing a lower percentage because it resembles bulk NMR. Repeated scan bootstraps cannot resolve systematic calibration-to-external bias.

## 6. Troubleshooting

- `No MS2 spectra found at NCE 18/22/26`: check precursor isolation target ~545.0996, ms level 2, collision energy metadata and actual mzML content. Missing energies cannot be inferred.
- `Missing calibration mzML`: check `DATA_DIR`. Do not rename files without updating `v66.mzml_sources`.
- `Experiment inputs or parameters changed`: use `--out-dir` to start a new experiment so old checkpoints are not mixed.
- `pyteomics` import error: use the Conda interpreter where `pip install -r requirements.txt` was executed.
- Initial mzML caching is slow: expected; subsequent runs reuse `CACHE/` as long as file fingerprints and V6.6 version match.

## Tests performed

Synthetic tests validate all three energies and both preprocessing branches, fold-frozen ratio selection, eight leave-one-ratio-out Ridge refits, feature-support/shift tables, NMF hotspot calculations, scan-block bootstrap, checkpoint resume, and generated figures/CSVs. Real mzML predictions can only be verified after running on the Mac with your raw data.
