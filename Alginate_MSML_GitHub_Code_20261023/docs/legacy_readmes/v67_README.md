# V6.7 — DP5–DP7 calibration-only overnight experiment

## Objective and scientific scope

**Primary endpoint:** absolute error in **M percentage points** on calibration **compositions completely withheld from model training**. DP5–DP7 do **not** have adequate independent external MS/MS signal, so the analysis **cannot estimate external DP-specific accuracy**. The 7 mixtures per DP provide seven independent composition-level validation units per energy; individual MS/MS scans are **technical observations**, not independent reference materials.

The script reuses the same nominal composition calibration labels and mzML source structure as the original manuscript scripts. DP5–DP7 NMR-derived labels are read from the included `DP_spectrum_process_Class.py`; they are not synthesized or inferred from external bulk NMR. The script preserves **two-component NMF + eight selected intensity ratios (K2+K8)**, and includes a matched **K8-only** regression control.

## Three DP-aware normalization schemes

1. `ORIGINAL7`: **exact original seven group-area windows**, Gaussian smoothing, source peak mask. This is the locked baseline at **sigma=1 Da**.
2. `EXTENDED7`: seven candidate fragment families centered on chemically annotated ions throughout the available DP5–DP7 m/z range: 100–205, 275–385, 450–565, 585–755, 815–930, 1015–1100, 1170–1270. Groups are applied **only to annotated fragment-ion windows**, require at least two eligible centers, and **exclude the modeled intact precursor** even if its mass falls inside a group's nominal bounds.
3. `GUARDED_EXTENDED7`: same fragment membership, but only normalizes groups with at least **90% training-scan support**, and uses training-derived group-area denominator floors with maximum scaling **25×**. Guard thresholds are computed from **training scans only** in every held-out fold.

The extended/guarded group definitions are *experimental*, not yet established manuscript preprocessing; final method selection must be documented in the paper. For all configurations we test Gaussian **sigma = 0, 0.10, 0.25, 0.50, 0.75, 1.00 Da**, matching V6.6's available widths. NCE **18%, 22%, and 26%** are analyzed separately for each DP. There are **18 preprocessing settings × 2 models × 9 DP/NCE conditions × 7 held-out compositions = 2,268 model-fold predictions**. Several jobs may be required; resume is automatic.

## Validation and leakage controls

**Development grid:** 7-fold leave-one-*composition*-out (LOCO), where all retained spectra of the held-out nominal mixture are excluded from NMF, ratio selection, scaling, group guards and Ridge fitting. Ratios are selected once per training fold from the `ORIGINAL7, sigma=1` representation and held fixed for candidate preprocessing comparisons. The best-of-18 grid MAE is a **development-selected estimate**, and should **not** be used as independent post-selection accuracy.

**Nested LOCO:** seven outer held-out composition folds, with a six-composition inner cross-validation to choose among four *predeclared* candidate recipes: original σ1, extended σ0.5, guarded extended σ0.5, guarded extended σ1. Inner ratio selection is re-run using only its five inner-training compositions. The entire selected model is refitted on six outer-training compositions before evaluating the seventh. Report the outer-fold MAE as **performance of the model-selection procedure within the calibration-fraction system** (not performance on independently sourced samples).

**Fixed protocol:** The default fixed baseline remains `ORIGINAL7, sigma=1` at all three NCEs. Once V6.6 finishes, choose a **single scientifically justified DP-transfer rule** (one sigma and group scheme per NCE); save it in a JSON file **before** examining the V6.7 DP5–DP7 results, and use a fresh output directory. This yields a clean fixed-protocol comparison for Figure 3f. The source `locked_protocol_template.json` documents the JSON shape. Some V6.6 group codes are specific to DP3/4 and must be translated explicitly into the *DP-aware group rules* here rather than assumed identical.

## Installation and usage (Mac PyCharm)

1. Extract the ZIP under `~/Desktop/alginate_experiment/`.
2. Open `alginate_V67_DP5_DP7_OVERNIGHT_IDE.py` in PyCharm.
3. Set the interpreter to `/opt/anaconda3/envs/alginate_experiment/bin/python`.
4. Check that `DATA_DIR` points to the directory containing `Raw_F5_M0G100.mzML` ... `Raw_F7_M100G0.mzML`.
5. Install: `python -m pip install -r requirements.txt`.

Run:

```bash
python alginate_V67_DP5_DP7_OVERNIGHT_IDE.py --run-mode selftest
python test_v67_synthetic.py

# Complete DP5–DP7 study: default IDE mode is also overnight
python alginate_V67_DP5_DP7_OVERNIGHT_IDE.py --run-mode overnight --threads 2

# Smaller experiment to validate real mzML parsing
python alginate_V67_DP5_DP7_OVERNIGHT_IDE.py --run-mode pilot --dp 5 --nces 22 --threads 2

# Later re-run additional cases / resume after interruptions
python alginate_V67_DP5_DP7_OVERNIGHT_IDE.py --run-mode overnight --threads 2

# Rebuild reports from saved checkpoints, no fitting
python alginate_V67_DP5_DP7_OVERNIGHT_IDE.py --run-mode report

# Fixed protocol imported only after V6.6 is decided; use separate output dir
python alginate_V67_DP5_DP7_OVERNIGHT_IDE.py --run-mode overnight \
    --transfer-protocol locked_protocol_template.json \
    --out-dir V67_LOCKED_FROM_V66_RESULTS --threads 2
```

Modes: `pilot` = first two holdouts, two Gaussian levels and two groups in the grid; the full four-choice inner selection on two outer folds; `grid` = full grid only; `nested` = nested selection only; `overnight` = grid and nested; `report` = reporting only; `selftest` = local numerical checks.

`--run-mode overnight` is the **IDE default**, so clicking Run starts the complete study. Pilot outputs are **partial** and must never be presented as seven-composition validation. To avoid rerunning completed work, repeat the same command and output directory. A settings/data fingerprint guards against incompatible reuse.

## Output files

`V67_DP5_DP7_RESULTS/` includes:

| Output | What it provides |
|---|---|
| `V67_SCAN_COUNTS.csv` | Number of matching target MS/MS scans acquired and retained, rejected counts, **per DP × NCE × mixture** |
| `V67_GRID_ALL_FOLDS.csv` | All grid held-out composition predictions and errors |
| `V67_GRID_BY_DP_NCE.csv` | All 18 preprocessing settings × both regression architectures; seven-fold MAE and denominator support |
| `V67_ORIGINAL7_BASELINE.csv` | Locked original σ1 group-area reference |
| `V67_FIXED_PROTOCOL_FIG3F.csv` | Fixed, predeclared protocol values for the 3 × 3 Figure 3f submatrix |
| `V67_DEVELOPMENT_GRID_WINNERS.csv` | Best-of-grid (selection-biased) candidates, only for development |
| `V67_K8_VS_JOINT_COMPARISON.csv` | Added-value assessment of NMF beyond ratios |
| `V67_NESTED_OUTER_ALL_FOLDS.csv` | Outer held-out predictions after inner preprocessing selection |
| `V67_NESTED_BY_DP_NCE.csv` | Nested outer MAE for 9 DP × NCE conditions |
| `V67_NESTED_SELECTION_COUNTS.csv` | Which preprocessing strategies inner CV selected |
| `V67_FULL_CALIBRATION_K8_RATIOS.csv` | K8 identities selected on full calibration for each DP/NCE |
| `V67_GROUP_MEMBERSHIP.csv` | Exact annotated-ion memberships per DP; intact precursor excluded |
| `V67_EVIDENCE_AUDIT.txt` | Scope and limitations |
| `FIGURES/*.png` | Baseline, fixed-protocol and nested 3 × 3 heatmaps |

**For publication**, use a consistent validation definition for all 15 heatmap cells (DP3–DP7 × three NCEs). If DP5–DP7 use a different selection protocol than DP3/DP4, either recompute DP3/DP4 under the matched protocol or distinguish the two evaluation designs in the manuscript and supplementary materials. Do not combine best-of-grid MAE for some DPs with frozen-protocol MAE for other DPs as if they were comparable unbiased estimates.

## Verification limitations

The numerical self-test and synthetic model-fitting/checkpoint tests can run without any real mzML files. Raw mzML parsing still needs verification in your real Conda environment (`pyteomics` required); this environment used for building the package does not contain the mzML reader. Synthetic results must never be reported as scientific observations.
