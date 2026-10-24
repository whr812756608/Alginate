# Alginate V6.6 — all-NCE Gaussian × group × partial-normalization overnight study

## Scientific objective

Use **one shared HCD collision energy** (18, 22, or 26) for the final DP3 and DP4 external results, while allowing distinct DP-specific preprocessing if calibration evidence supports it. Preserve the manuscript's **rank-2 NMF (global spectral features) + eight peak-intensity ratios (local features)**; compare with the eight-ratio-only control.

We test:
- DP: 3, 4; NCE: 18, 22, 26 **in the same run**.
- Gaussian sigma (Da): **0, 0.10, 0.25, 0.50, 0.75, 1.00**.
- Group definitions: **ORIGINAL7, G291, G545, G721, G291_G545, G291_G721, G545_G721, GALL**.
- Group-normalization strength: **0 (global TIC), 0.5 (partial), 1 (full original-style group-area normalization)**.
- Two models at each setting: **K8_ONLY** and **K2_PLUS_K8**.
- Seven full nominal-composition holdouts per DP/NCE; all scans from the held-out nominal composition excluded from calibration training. Each NCE separately fitted; a common NCE can be selected afterward based on **calibration** evidence. There is no across-energy training leak.

Equivalent combinations are deduplicated: DP3 has 4 effective fragment group choices because m/z 721 is outside its processed m/z range, while DP4 has all 8. `lambda=0` has one shared TIC-normalized condition. This gives **54 preprocessing settings per DP3/NCE and 102 per DP4/NCE**; **468 distinct DP/NCE/preprocessing settings**, **936 model configurations**, **6,552 model-by-heldout calibration evaluations**, plus **936 full-calibration final fits** for external predictions. Expect more than one night on some Macs; checkpoint/resume is built in.

## Install and run in PyCharm

Extract ZIP into `~/Desktop/alginate_experiment/` and open:

`alginate_V66_ALL_NCE_OVERNIGHT_IDE.py`

Verify at the top of the script:

```python
DATA_DIR = Path.home() / 'Desktop' / 'alginate_experiment'
OUT_DIR = HERE / 'V66_ALL_NCE_RESULTS'
```

Select Python interpreter `/opt/anaconda3/envs/alginate_experiment/bin/python` (or your own working environment). Dependencies:

```bash
conda activate alginate_experiment
cd ~/Desktop/alginate_experiment/Alginate_V66_AllNCE_GroupGaussian_Overnight
python -m pip install -r requirements.txt
```

Run checks first:

```bash
python alginate_V66_ALL_NCE_OVERNIGHT_IDE.py --run-mode selftest
python test_v66_synthetic.py
```

Small real-mzML pilot (DP3 NCEs 18/22/26, 2 holdouts and 4 settings; full calibration frozen external predictions):

```bash
python alginate_V66_ALL_NCE_OVERNIGHT_IDE.py --run-mode pilot --dp 3 --threads 2
```

Full **DP3 + DP4, all NCEs** overnight:

```bash
python alginate_V66_ALL_NCE_OVERNIGHT_IDE.py --run-mode overnight --threads 2
```

`overnight` is the **IDE default**, so clicking PyCharm's Run button also runs the full experiment. To resume, run the same command again. Completed fold and external checkpoint CSVs are skipped automatically.

If it takes several nights, keep the same `OUT_DIR` and data unchanged. The cache and settings manifest include source file fingerprints. For a different dataset or model configuration, set `--out-dir` to a new folder; do not mix old and new checkpoint results.

Only DP4 while you debug:

```bash
python alginate_V66_ALL_NCE_OVERNIGHT_IDE.py --run-mode overnight --dp 4 --threads 2
```

Only rebuild summary outputs (no model refitting):

```bash
python alginate_V66_ALL_NCE_OVERNIGHT_IDE.py --run-mode report
```

**Do not** use `--run-mode external` with missing calibration-derived ratios: it will still compute those ratios from the full calibration spectra, but does not perform held-out validation.

## Specific preprocessing choices

The original manuscript's `FractionSpectra.interpolate_to_grid_without_norm`, original sigma-1 smoothing implementation, annotated peak filtering, and original seven fragment groups are preserved. **Sigma 1, ORIGINAL7, strength 1 reproduces original group-area normalization**. All other sigma values use the original `FractionSpectra.smooth_spectrum` function on the same interpolated grid.

The intermediate normalization is deliberately defined to interpolate **spectral shape** and **total scale**:

- `T = x/sum(x)` = TIC normalized spectrum, after Gaussian smoothing and peak-mask filtering.
- `G = group_area_normalization(x, scheme)` = original-style grouped spectrum.
- `g = sum(G)`, `S = (1-λ)T + λ(G/g)`, `A = (1-λ) + λg`.
- Output = `A*S`.

This produces TIC exactly at λ=0 and original group normalization exactly at λ=1. At λ=0.5 it blends their normalized spectral shapes and smoothly restores the natural group-normalized total scale. **It is a newly defined intermediate preprocessing scheme, not an established analytical convention.** The formula and group assignments must be disclosed if a selected model is reported in the manuscript.

The original seven windows are taken directly from `alginate_ratio_transfer_IDE.py`. G291 extends group 2 lower bound to 290; G545 extends group 3 upper bound to 550; G721 extends group 4 upper bound to 730. The combinations apply the corresponding extensions simultaneously. No external bulk labels are used in choosing group boundaries.

## Ratio-selection protection

In each outer training fold, the reference eight ratios are selected using the **ORIGINAL7, sigma 1, lambda 1** calibration-train spectra only. They are reused at all candidate sigma/group/strength settings, so normalization effects are **not confounded by ratio-identity changes**. Full-calibration frozen ratios are selected again using all calibration scans before evaluating the three external samples. Each NCE has its own training-derived reference ratio set. This protocol intentionally differs from a script that freely reselects eight ratios under every normalization scheme.

All fitted scalers, NMF bases, and Ridge regressions use calibration training rows only. Composition-balanced sample weights match the previous scripts. Individual MS2 scans are technical replicates, **not independent material samples**.

## Important output CSVs

Results are written to `V66_ALL_NCE_RESULTS/`:

| File | Meaning |
|---|---|
| `V66_LORO_ALL_FOLDS.csv` | Every held-out calibration composition for every candidate/model |
| `V66_LORO_BY_CONDITION.csv` | Mean MAE, worst error, seven-fold completeness and denominator QC |
| `V66_ORIGINAL_BASELINE.csv` | Original sigma1/ORIGINAL7/lambda1 controls |
| `V66_CALIBRATION_VS_ORIGINAL.csv` | Candidate-vs-original calibration comparison |
| `V66_CALIBRATION_SELECTED_PER_DP_NCE.csv` | Lowest-MAE calibrated joint-model candidate for each DP/NCE |
| `V66_COMMON_NCE_CALIBRATION_SELECTION.csv` | Joint DP3+DP4 held-out MAE for each shared NCE |
| `V66_COMMON_PREPROCESSING_BOTH_DP_RANKING.csv` | Also rank SAME preprocessing for both DPs |
| `V66_EXTERNAL_ALL_MODELS.csv` | GG-rich, MG-rich, MM-rich predicted M% + scan variability and QC at every NCE |
| `V66_PRIMARY_DP_NCE_M_PERCENT_TABLE.csv` | External predictions of calibration-selected DP/NCE candidates only |
| `V66_PRIMARY_COMPACT_ALL_NCE_TABLE.csv` | One compact row per selected DP/NCE with three M% columns |
| `V66_EXTERNAL_NCE_CONSISTENCY.csv` | Energy dependence of the same DP/preprocessing/model/sample |
| `V66_BULK_NMR_CONTEXT_NOT_ACCURACY.csv` | All predictions vs DP-unresolved bulk NMR (descriptive only) |
| `V66_POSTHOC_BULK_RANKING_EXPLORATORY.csv` | Retrospective agreement ranking—NOT independent accuracy |
| `V66_EVIDENCE_AUDIT.txt` | Completion summary and evidence limitations |

Figures are under `FIGURES/`. Partial runs also compile results, but models with fewer than 7 folds are **not eligible for calibration-based selection**.

## Quality caveats

- External MS/MS M% is **DP-specific**; bulk 1H NMR values (GG 12, MG 51, MM 82% M) describe **bulk samples**, not DP3 or DP4 ground truth. Difference from bulk NMR is not DP-specific external prediction error.
- Group-dependent NMF reconstruction OOD thresholds are not an identical absolute scale across different preprocessing methods. Treat comparisons cautiously.
- The m/z 703.121 ratio denominator can be poorly supported at DP4/NCE22. The script keeps denominator QC next to every sample prediction and flags <80% detection. It does **not** overwrite invalid ratios or use bulk NMR values to choose ratios.
- Because this is a large experiment, the smallest calibration MAE after searching many configurations is a **development-selected estimate**; independent test performance cannot be inferred from that minimum. For a final manuscript accuracy claim, add nested CV or describe that the comparison is an ablation/development study.
- The script writes failed checkpoint stack traces under `FOLD_CHECKPOINTS` or `EXTERNAL_CHECKPOINTS` with `.failed.txt`. Review these if the expected seven folds are not complete.

## Verification performed on this package

- Python source compilation.
- Exact sigma1/ORIGINAL7/full-group-normalization mathematical equivalence check on constructed spectra.
- Input grid deduplication (54 unique DP3 settings; 102 unique DP4 settings).
- Synthetic **DP3 and DP4 × all three NCEs × seven held-out compositions** using original ratio-selection and real NMF/Ridge code, with a reduced test-only preprocessing set.
- Checkpoint/resume, external full-calibration inference, and the common-NCE report generation.

Actual mzML spectra **cannot be run here**; use your Mac with the original data. Do not interpret synthetic test predictions as real experimental results.
