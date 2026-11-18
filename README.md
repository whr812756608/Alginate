# Alginate MS/MS Composition Prediction

Research code for predicting the mannuronic acid (M) fraction of alginate oligosaccharide (AOS) mixtures from higher-energy collisional dissociation tandem mass spectrometry (HCD-MS/MS), using NMR-calibrated composition references.

The study covers **DP3–DP7** at normalized collision energies (**NCE**) of **18, 22, and 26**. Each model combines two nonnegative matrix factorization coefficients and eight fragment-ion intensity ratios with Ridge regression.



## Requirements and Installation

Python 3.10 or later is required.

```bash
git clone https://github.com/whr812756608/Alginate.git
cd Alginate
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
```

The current checkout supports configuration checks and the composition-grouped cross-validation splitter:

```bash
python -m pytest -q
```

These are software tests, not a reproduction of the experimental results.

## Data

Calibration measurements comprise seven nominal M-rich compositions (0, 20, 40, 50, 60, 80, and 100%) for each DP. The corresponding NMR-derived composition values are used as regression targets. The analysis also includes external GG-rich, MG-rich, and MM-rich AOS samples for DP3/DP4 prediction.

Raw HCD-MS/MS data were previously referenced under **MassIVE accession MSV000103648**. File access, coverage, and correspondence with the analysis inputs should be confirmed against the data record. Original mzML files are not included here.

## Usage

The intended reproducible analysis consists of:

1. Reading and preprocessing calibration HCD-MS/MS spectra.
2. Extracting two NMF coefficients and eight selected ion-intensity ratios.
3. Evaluating models with seven-fold **leave-one-out cross-validation (LOO-CV)**, holding out all spectra from one calibration composition per fold.
4. Fitting final DP × NCE models using all calibration compositions.
5. Applying the frozen DP3/DP4 models to external AOS spectra without external adaptation.

The corresponding preprocessing, model-training, evaluation, and external-prediction implementations are **under numerical verification** and are not yet operational in this public release. Commands to reproduce the manuscript's tables and figures will be documented here after validation.

## Interpretation

The selected-model LOO results are retrospective: the calibration compositions were also used during method development, so the scores do not represent fully independent model-selection validation.

The GG/MG/MM external NMR reference values describe **bulk composition**, not DP3- or DP4-specific ground truth. Agreement with those references should not be reported as DP-specific external prediction accuracy.

## Citation

The manuscript citation and tagged software version will be added upon publication. No open-source license has yet been assigned; public availability alone does not grant reuse rights.
