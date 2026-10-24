# V6.8 — DP3 / NCE 26% targeted spectral-transfer experiment

A **standalone, PyCharm-friendly** experiment for the alginate HCD-MS/MS manuscript. It targets the outstanding high DP3 external M% predictions **without using external bulk NMR values to fit models, select ratios, or tune preprocessing**.

## Why this experiment?

V6.6 tested 54 Gaussian/group/normalization combinations at DP3/NCE26. Its closest retrospective K2+K8 result was still approximately **31.24% (GG-rich), 75.91% (MG-rich), and 93.01% (MM-rich)**, compared with **12%, 51%, and 82% bulk NMR**. Those NMR values describe the **whole material**, **not DP3-specific ground truth**, so the difference is descriptive, not a verified DP3 error.

V6.8 keeps the two-component NMF + eight selected peak ratios framework and tests changes to the actual predictors rather than another smoothing/grid expansion.

## Experiment matrix

Runs only **DP3 at NCE26** with all seven calibration compositions and the three previously acquired independent external materials.

- **Two fixed preprocessing references:** V6.6 selected σ=0.25 Da, ORIGINAL7, λ=1.0; and original manuscript σ=1.0 Da, ORIGINAL7, λ=1.0.
- **Four training-only ratio selection policies:** `REFERENCE` (exact V6.6 reference selection from σ=1), `RAW_SUPPORT` (calibration-composition-wise 98% raw centroid denominator support), `CONSENSUS` (six inner training-only composition omissions), and `PERTURB_ROBUST` (training-only synthetic ion-intensity jitter/dropout penalty, plus 98% raw support).
- **Two local feature transforms:** ordinary intensity ratios (`RAW`) and training-derived-pseudocount logarithmic ratios (`LOG_RATIO`).
- **Two K2 NMF representations:** unmodified full range (`FULL`) and precursor-window-excluded (`NO_PRECURSOR`), using the same two components and coordinate-descent NMF.
- **Matched K8-only control** for every ratio policy and transformation (NMF setting is irrelevant).
- **Four sample-level reporting aggregations:** arithmetic mean, median, 10% trimmed mean, and prediction-blind QC weights based on raw ratio-denominator support and NMF reconstruction residual, with quality weights fixed by rules and calibration residual thresholds.

**48 unique model arms × seven held-out compositions = 336 calibration-fold model evaluations**, plus external inference. Four reporting aggregations are **not** four independently trained models.

All selection/ridge/NMF/scalers are fitted from training compositions only. No scan from the held-out nominal composition enters any fitted object. Calibration spectra are technical scans, not independent materials.

## Installation and PyCharm run

Extract the ZIP into:

`~/Desktop/alginate_experiment/Alginate_V68_DP3_NCE26_Transfer/`

The ZIP includes the original processing dependencies from the V6.6 package.

Use the existing Conda interpreter (`/opt/anaconda3/envs/alginate_experiment/bin/python`).

```bash
conda activate alginate_experiment
cd ~/Desktop/alginate_experiment/Alginate_V68_DP3_NCE26_Transfer
python -m pip install -r requirements.txt
python alginate_V68_DP3_NCE26_TRANSFER_IDE.py --run-mode selftest
python test_v68_synthetic.py
```

First run a small real-data pilot:

```bash
python alginate_V68_DP3_NCE26_TRANSFER_IDE.py --run-mode pilot --threads 2
```

Then the full DP3/NCE26 grid:

```bash
python alginate_V68_DP3_NCE26_TRANSFER_IDE.py --run-mode overnight --threads 2
```

**PyCharm green Run button** defaults to `overnight`; for a pilot enter `--run-mode pilot` in Run > Edit Configurations > Parameters. The default data directory is `~/Desktop/alginate_experiment`, set as `DATA_DIR` near the top of the main script. If your data live elsewhere, use `--data-dir /path/to/mzML`.

The script discovers these original files in your data directory:

- `Raw_F3_M0G100.mzML`, `Raw_F3_M20G80.mzML`, ..., `Raw_F3_M100G0.mzML` (seven mixtures).
- `AlgM_LowDPMix_1_051625.mzML`, `AlgMG_LowDPMix_1_051625.mzML`, `AlgG_LowDPMix_1_051625.mzML` (three external materials; as defined by `alginate_ratio_transfer_IDE.py`).

For **resume**, simply rerun the same command. Results and raw-spectrum interpolated/smoothed caches are stored in `V68_DP3_NCE26_RESULTS`. If input mzML file size/mtime or experiment settings change, use a new `--out-dir` so incompatible checkpoints are never mixed.

To regenerate reports only:

```bash
python alginate_V68_DP3_NCE26_TRANSFER_IDE.py --run-mode report
```

To only refit/freeze external models:

```bash
python alginate_V68_DP3_NCE26_TRANSFER_IDE.py --run-mode external
```

## Outputs

| File | Meaning |
|---|---|
| `V68_LORO_ALL.csv` | Seven calibration-held-out composition predictions for all 48 arms, with four aggregations. |
| `V68_LORO_SUMMARY.csv` | Calibration MAE, maximum fold error, completion and QC for each arm. |
| `V68_LORO_RANKED_VS_BASELINE.csv` | Calibration performance relative to exact V6.6 DP3/NCE26 joint K2+K8 baseline. |
| `V68_EXTERNAL_ALL.csv` | External GG-/MG-/MM-rich M%, scan SD, clipping, raw-ion denominator support, NMF OOD, effective QC-weighted sample size. |
| `V68_EXTERNAL_COMPACT_M_PERCENT.csv` | Three populations side-by-side for each arm. |
| `V68_CALIBRATION_SHORTLIST_NOT_EXTERNAL_OPTIMAL.csv` | Calibration-ranked shortlist based on seven-fold MAE and held-out raw-denominator QC; external QC is reported as an independent flag, not used to rank models. |
| `V68_CALIBRATION_EXTERNAL_JOINT_REVIEW.csv` | Joins calibration and external QC without selecting on external NMR. |
| `V68_POSTHOC_BULK_AGREEMENT_EXPLORATORY.csv` | Retrospective bulk-NMR comparisons. **NOT external DP3 accuracy.** |
| `V68_EXTERNAL_BULK_CONTEXT_ONLY.csv` | Explicit raw deviations from bulk NMR for all predictions. |
| `RATIO_SELECTION/H*/...` | Selected eight ratio identities for each fold/selection approach. |
| `RATIO_SELECTION/EXTERNAL/...` | Selected final calibration-derived eight ratios. |
| `FROZEN_MODELS/` | Frozen full-calibration model objects. |
| `FIGURES/` | Calibration model comparison and external sample comparison for calibration-shortlisted models (if full seven-fold results exist). |
| `V68_EVIDENCE_AUDIT.txt` | Validation boundaries and result completeness information. |

Any failed configuration generates a `.error.txt` checkpoint alongside the intended output rather than silently dropping its diagnostics. If a ratio policy cannot find eight eligible ratios, it is marked failed; the script does **not** silently substitute unsafe features.

## Interpretation and manuscript use

The V6.6 selected **unchanged joint baseline** is `V66_SELECTED / REFERENCE / RAW / FULL / K2_PLUS_K8 / MEAN`. The test suite checks numeric parity against V6.6 fitting on synthetic data. A genuinely improved result should (a) preserve reasonable seven-composition calibration accuracy, (b) avoid weak or absent raw ratio denominators, (c) avoid severe prediction clipping or instability, and (d) be described honestly with external domain-shift QC. Raw external bulk-NMR disagreement is interesting but **cannot validate** DP3 accuracy, and any apparent minimum among many candidates is post-hoc, not an independent result.

A limitation of the `QC_WEIGHTED` arm is that quality weights can concentrate heavily on very few scans. Use `qc_effective_n` and scan-half stability to identify such cases. The `CONSENSUS` selection uses six inner omissions but does not confer a nested unbiased estimate for the **overall choice among the 48 model arms**.

## Testing scope

Synthetic self-tests passed for all four selector policies, both transforms, both NMF variants, K8-only and joint fits, sample aggregations, checkpointing, CSV reports, and numerical parity with V6.6 on synthetic spectra. The local test environment lacked the `pyteomics` mzML reader and no raw experimental mzML data were available; these synthetic tests used a test-only in-memory mzML import stub. Install `requirements.txt` in your Mac Conda environment and run the real files there.
