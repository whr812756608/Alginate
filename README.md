# Alginate MS–ML — NMR-calibrated AOS M/G ratio prediction

Reproducible **research-development code** for learning composition-associated fragmentation patterns in alginate oligosaccharides (AOS) without requiring sequence-pure AOS standards.

**Scope:** DP3–DP7 calibration mixtures at HCD NCE 18%, 22%, and 26%, using NMR-calibrated mixture labels, two NMF-derived coefficients and eight diagnostic fragment-ion intensity ratios, standardized Ridge regression, and leave-one-out cross-validation (LOO-CV), holding out all spectra from one calibration composition per fold.

## Quick start

Requires Python 3.10+ (3.11 recommended). Run the following commands from the code directory shown below; real-data analyses require the original mzML files, while synthetic tests do not.

```bash
cd Alginate_MSML_GitHub_Code_20261023
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt

# Run the fixed-protocol calibration analysis self-test (no mzML required):
python experiments/alginate_V70_EVIDENCE_EXISTING_DATA_IDE.py --run-mode selftest

# Synthetic integration tests — results are NOT scientific measurements:
python tests/test_v70_synthetic.py

# Full existing-data experiment (specify location of private mzML data):
python experiments/alginate_V70_EVIDENCE_EXISTING_DATA_IDE.py \
  --run-mode overnight --threads 2 \
  --data-dir /path/to/alginate_experiment \
  --out-dir ./outputs/v70
```

The fixed-protocol analysis applies the same preprocessing configuration across DP3–DP7 and NCE 18%, 22%, and 26%. For each DP/NCE condition, two NMF-derived coefficients and eight selected fragment-ion intensity ratios are used in Ridge regression. Performance is evaluated by LOO-CV, withholding one complete calibration composition per fold. This assesses prediction within the calibration-mixture system, not independently measured DP-specific external accuracy.

## Repository structure

| Location | Content |
| --- | --- |
| `alginate_ratio_transfer_IDE.py`, `v62_core.py`, `DP_spectrum_process_Class.py` | Original shared fragment-processing and MS model dependencies |
| `alginate_V66_ALL_NCE_OVERNIGHT_IDE.py` | Shared V6.6 pipeline and experiment code used by later versions |
| `experiments/alginate_V67_*.py` | Higher DP5–DP7 preprocessing sensitivity and nested selection |
| `experiments/alginate_V68_*.py` | DP3/NCE26 calibration/external transfer ablations |
| `experiments/alginate_V69_*.py` | DP3 three-NCE ratio-distribution and model-domain diagnostics |
| `experiments/alginate_V70_*.py` | Fixed-protocol LOO-CV, feature comparisons, and grouped-label permutation experiments |
| `tests/` | Synthetic acceptance tests (not scientific validation) |
| `results/reported/` | Selected CSV exports from earlier real-data runs, identified by script version |
| `docs/legacy_readmes/` | Original experiment-specific instructions and changelogs |
| `figures/figure3c/` | DP3/NCE26 NMF-weight and error-bar figure generator (experimental plot requires mzML) |
| `figures/figure3e_f/` | Figure 3e LOO-CV schematic and Figure 3f calibration MAE heatmap generators, local source CSV, SVG/PNG |
| `figures/editable/` | Editable PowerPoint source artwork for Figures 3e and 3f |
| `results/reported/v70/` | Archived analysis outputs: calibration LOO-CV, feature comparisons, permutation controls, and exploratory diagnostics |
| `docs/` | Data organization, interpretation, limitations, reproducibility checklist |

V6.6 is imported by downstream scripts and is intentionally located at the repository root. Each experiment script additionally contains a small import-path adapter needed because the repo now groups experiment files under `experiments/`. No scientific computations were changed as part of packaging.

## Reproducibility

This repository provides the Python analysis workflow and selected source-data tables for NMR-calibrated HCD-MS/MS prediction of alginate M/G composition. The original mzML files are required to rerun model fitting and validation; they are not included in the repository.

The calibration dataset comprises seven nominal stock-mixture proportions (0%, 20%, 40%, 50%, 60%, 80%, and 100%) for each degree of polymerization (DP3–DP7). These percentages describe the preparation of the mixtures, **not their measured M fractions**. Regression targets are calculated from the NMR-characterized stock fractions and their mixing proportions.

Model performance is assessed using **leave-one-out cross-validation (LOO-CV)** at each of the 15 DP/NCE conditions (five DPs × three collision energies). Each fold withholds all spectra from one calibration composition and uses the remaining six compositions for model fitting. The seven folds per condition yield 105 composition-level held-out predictions in total.

To reproduce the analysis, follow **Quick start** from the code directory and organize the mzML inputs as specified in the [data layout guide](Alginate_MSML_GitHub_Code_20261023/docs/DATA_LAYOUT.md). The [analysis guide](Alginate_MSML_GitHub_Code_20261023/docs/ANALYSIS_GUIDE.md) describes the available experiments, and [reported results](Alginate_MSML_GitHub_Code_20261023/results/reported/) contain selected outputs from prior real-data runs. The included synthetic tests check software operation; **they are not experimental validation or substitutes for reprocessing the original spectra**.

## Data and code availability

The manuscript reports HCD-MS/MS deposition at MassIVE accession **MSV000103648**; verify that its actual files and access permissions match the input naming conventions before citing this repository as a complete reproduction. Raw mzML, instrument-native files, calibration stocks' NMR raw datasets, unpublished manuscript Word documents, and checkpoint/prediction caches are **not distributed in this repository**. Do not commit them without the data owners' approval.

No open-source license is assigned by this package. Publication permissions and a license should be agreed with all code contributors before making this repository public.
