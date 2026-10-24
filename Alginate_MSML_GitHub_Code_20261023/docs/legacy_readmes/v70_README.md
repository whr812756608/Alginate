# Alginate V7.0 — stronger manuscript evidence from existing data ONLY

**Purpose:** Demonstrate that DP3–DP7 HCD-MS/MS spectra carry genuine M/G composition-associated information without requiring sequence-pure AOS standards or additional material. The calibration labels remain calculated from **NMR-calibrated mixed-composition fractions**. There are just three independently prepared unfractionated external AOS materials; their **bulk NMR is not DP-specific truth**.

## What is new and why this test matters

**Primary experiment: whole-composition label permutation** across DP3–DP7 at NCE 18%, 22%, and 26%, evaluating a fixed, common preprocessing protocol, rather than selecting the most favorable preprocessing or external-bulk agreement.

1. First, estimate held-out calibration MAE on the authentic seven composition labels, with **all scans for the held-out mixture withheld together**. Feature selection, NMF, scaling and Ridge training use the other six mixtures.
2. Repeat the entire supervised prediction procedure with 199 unique rearrangements of the seven mixture composition labels **as intact groups**. The same permutation is applied to every DP/NCE condition in each repetition, preserving important cross-condition dependence. Critically, the eight local fragment ratios are re-selected from the six training mixtures under the permuted labels, and Ridge is refitted. The K2 unsupervised global spectra are fit once per fold and cached; their ordering is deliberately independent of any M% labels.
3. Evaluate two matched model arms: **K2+K8** (the manuscript pipeline) and **K8-only** (a control). Fixed preprocessing is Gaussian σ=1.0 Da, ORIGINAL7 group-area normalization, λ=1.0 for **all 15 conditions**.
4. Apply the **authentic-label, full-calibration frozen models** (not permuted models) to the existing GG-, MG-, and MM-rich external samples using DP3 and DP4. Check predictions, raw centroid ratio-denominator support, NMF OOD flags, scan variability, and cross-NCE consistency. Present bulk NMR alongside them **only as DP-unresolved descriptive context**.
5. Report **endpoint versus interior LOCO errors** to distinguish composition interpolation from 0%/100% nominal-mixture extrapolation.

### Statistical inference

Each condition has seven held-out composition estimates. For each observed or permuted dataset, the primary score is the mean of the **seven absolute composition-level errors**, not an average over thousands of technical scans.

Permutation *p* is calculated as `(1 + number of permuted MAEs no greater than the observed MAE) / (1 + number of permutations)`. Smaller observed errors indicate stronger association. A synchronized 15-condition **macro-average permutation test** is the main aggregate inferential test; per-condition permutation results are supplementary and correlated. With 199 permutations, the smallest attainable one-sided Monte Carlo p is 0.005.

If random labels result in fewer than eight eligible diagnostic ratios, the predefined model falls back to the **mean of the six training-composition labels**. This rule is applied identically to authentic and randomized labels and failures are counted in the output. The null therefore evaluates the *complete predeclared analysis procedure*, including its failure behavior.

No model parameter, threshold, NCE or group definition is chosen based on the GG/MG/MM identities, their expected ordering, or bulk NMR 12%, 51%, 82%. This test does **not** establish accuracy on independently characterized DP3/DP4 samples or prove that NMF fragments represent pure MM/GG sequence components.

## Directory and requirements

Extract the ZIP so all these files are together:

- `alginate_V70_EVIDENCE_EXISTING_DATA_IDE.py` — V7.0 main script
- `alginate_V66_ALL_NCE_OVERNIGHT_IDE.py` — exact legacy preprocessing and model wrapper
- `alginate_ratio_transfer_IDE.py`, `v62_core.py`, `DP_spectrum_process_Class.py` — original fragment and NMF helpers
- `test_v70_synthetic.py` — synthetic acceptance test; never use its data as scientific evidence
- `requirements.txt` and this README

On your Mac, set `DATA_DIR` at the beginning of the main script or provide `--data-dir`. The default is `~/Desktop/alginate_experiment`. The directory must contain `Raw_F3_M0G100.mzML` ... `Raw_F7_M100G0.mzML` for all seven nominal ratios **0,20,40,50,60,80,100**, and the external files `AlgG_LowDPMix_1_051625.mzML`, `AlgMG_LowDPMix_1_051625.mzML`, `AlgM_LowDPMix_1_051625.mzML` if external inference is enabled. One external file contains MS2 scans for multiple precursor DPs and energies.

Install in your `alginate_experiment` environment:

```bash
conda activate alginate_experiment
cd ~/Desktop/alginate_experiment/Alginate_V70_ExistingData_Evidence
python -m pip install -r requirements.txt
```

## Run steps

```bash
# 1. No real mzML needed: checks group-wise shuffling, fitting and statistics.
python alginate_V70_EVIDENCE_EXISTING_DATA_IDE.py --run-mode selftest
python test_v70_synthetic.py

# 2. Small real-data pilot: DP3/NCE26, 9 permutations, separate PILOT folder.
python alginate_V70_EVIDENCE_EXISTING_DATA_IDE.py \
  --run-mode pilot --dp 3 --nce 26 --threads 2

# 3. Full experiment: 5 DPs × 3 NCEs × 199 permutations.
python alginate_V70_EVIDENCE_EXISTING_DATA_IDE.py \
  --run-mode overnight --threads 2

# 4. Resume same full experiment by rerunning the identical command.
# 5. Rebuild reports from existing checkpoints without parsing mzML:
python alginate_V70_EVIDENCE_EXISTING_DATA_IDE.py \
  --run-mode report --n-permutations 199

# 6. Optional: external frozen-model predictions only.
python alginate_V70_EVIDENCE_EXISTING_DATA_IDE.py \
  --run-mode external --dp 3 4 --threads 2 \
  --out-dir V70_EXTERNAL_ONLY
```

**IDE default = `overnight`.** You can click PyCharm's Run button if the working environment, mzML files, and paths are correct. The full run may require several hours; the scripts checkpoint every 10 permutations per condition. Each DP/NCE job refits K2 NMF only seven times because that unsupervised part is independent of all permuted composition labels. If you want a faster exploratory test, specify a smaller `--n-permutations` and use a **new** `--out-dir`. Different permutation counts or input fingerprints cannot share a checkpoint directory.

**Note:** A pilot cannot generate a complete 15-cell figure or valid full-data global permutation *p*. Full scientific results require the real mzML files; the ZIP does **not** contain raw data.

## Key output files

| Output | Interpretation |
|---|---|
| `V70_GLOBAL_PERMUTATION_TEST.csv` | The 15-condition macro-MAE compared with synchronized whole-composition null permutations; **primary statistical result** |
| `V70_PERMUTATION_PVALUES_BY_DP_NCE.csv` | Per-DP/NCE K2+K8 and K8-only observed error vs permuted-label distribution |
| `V70_FIG3F_LOCKED_15_CONDITIONS.csv` | Matched fixed-preprocessing DP3–DP7 × 3 NCE heatmap values |
| `V70_ALL_PERMUTATION_RESULTS.csv` | One summary per permutation, DP/NCE and model arm; includes fallback counts |
| `V70_COMPOSITION_LEVEL_OOF_PREDICTIONS.csv` | True vs predicted M% from all seven held-out mixture groups |
| `V70_ENDPOINT_VS_INTERIOR_ERRORS.csv` | Nominal endpoint versus interior calibration-mixture errors |
| `V70_EXTERNAL_FROZEN_PREDICTIONS.csv` | Unseen-sample DP3/DP4 predictions at available NCEs, with spectral QC |
| `V70_EXTERNAL_CROSS_NCE_QC.csv` | Cross-energy ranges and important detection/OOD fields |
| `V70_EXTERNAL_ORDER_DIAGNOSTIC_ONLY.csv` | GG<MG<MM ordering as an unlabeled plausibility check |
| `V70_EXTERNAL_BULK_NMR_CONTEXT_ONLY.csv` | Descriptive side-by-side DP3/DP4 and whole-sample NMR; **not external DP-specific accuracy** |
| `V70_EXTERNAL_BULK_DISCREPANCY_CONTEXT_ONLY.csv` | Mean absolute bulk discrepancy; **never use to select model or claim DP-specific MAE** |
| `V70_MANUSCRIPT_EVIDENCE_AUDIT.md` | Interpretation, limitations, copy-ready conditional manuscript wording |
| `FIGURES/*.png` | Publication-oriented calibration heatmap, label-permutation comparison, external cross-NCE, and bulk-context plots |

## How this strengthens the manuscript

**If the true-label error is substantially below the permuted-label null:** The learned MS/MS–composition relationship is unlikely to be an artifact of arbitrary labeling within the seven-mixture system. This is meaningful supporting evidence for the **NMR-calibrated, sequence-standard-free composition modeling** claim across DP3–DP7. The K8 control tests whether two global NMF features improve supervised prediction, without assuming that they always do.

**If the true-label result is not distinguishable from null:** Do not claim strong composition discrimination for that DP/NCE. Inspect endpoints, selected ratio support, and calibration-label heterogeneity before changing models.

**For Fig. 3:** retain the fixed-preprocessing 15-cell LOCO heatmap and add one supplementary permutation null panel plus endpoint/interior results. **For Fig. 4:** show all three external materials at DP3 and DP4 with a fixed-calibration model and available NCEs; their DP-unresolved bulk NMR values are helpful contextual comparators, not DP-specific ground truth. The previously observed DP4 GALL model may remain a clearly labeled retrospective preprocessing sensitivity result; V7.0 does not try to manufacture further agreement with it.

### Limitations we should explicitly state

- The seven nominal mixtures share precursor-fraction source materials; each LOCO fold is a new *mixing proportion*, not a newly prepared independent chemical source batch.
- High technical scan counts do not increase the number of independently labeled compositions beyond seven per condition.
- Permutations test within-calibration label association rather than independent-sample external prediction error. They do not distinguish every possible preparation artifact from composition-driven spectral chemistry.
- The three external unfractionated materials have no DP-specific truth, so no defensible numerical external DP3/DP4 MAE is available. Cross-NCE consistency and correct ordering alone do not prove accuracy.
- Model development has already involved examining these three external materials, so they are not a previously untouched prospective validation set. Model choice must not be optimized against their bulk NMR values.
