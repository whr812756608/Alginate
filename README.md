# NMR-calibrated prediction of alginate oligosaccharide composition from HCD-MS/MS

This repository contains research code and associated analysis resources for estimating the mannuronic acid (M) fraction of alginate oligosaccharides (AOS) from higher-energy collisional dissociation tandem mass spectrometry (HCD-MS/MS). The study covers degrees of polymerization **DP3–DP7** at normalized collision energies **18, 22, and 26**, using composition references obtained by NMR.

## Method

The approach combines **two nonnegative NMF-derived coefficients (K = 2)** with **eight diagnostic fragment-ion intensity ratios (K = 8)** as inputs to Ridge regression. Spectral selection, m/z processing, smoothing, and intensity normalization are applied according to the DP-specific analysis settings.

Model evaluation uses composition-grouped leave-one-out cross-validation (**LOO-CV**). For each fold, all scans from one of the seven calibration compositions are withheld, and fitted transformations and regression are learned from the remaining compositions. Final models are trained on the full calibration dataset after evaluation. Models for DP3 and DP4 are also applied to external GG-rich, MG-rich, and MM-rich AOS samples without external calibration or adaptation.

The computational settings are recorded in [`configs/models.json`](configs/models.json). The implementation is organized as a Python package under `src/alginate_ms2/`, with analysis entry points under `scripts/`.

## Installation

Python 3.10 or newer is required. To install the current package and run its software tests:

```bash
python -m pip install -e '.[test]'
python -m pytest
```

**Current reproducibility status:** The scientific implementation is still being consolidated and checked against the selected research analyses. The current checkout includes model metadata, a composition-grouped LOO splitting utility, and structural tests, but its raw-spectrum preprocessing, NMF/ratio extraction, final model-fitting, and external-inference entry points are **not yet validated or operational**. The tests above are software checks, **not** a reproduction of the experimental results. Please do not treat this checkout as a completed manuscript reproduction package.

## Data availability

Reproduction from original spectra requires the calibration and external mzML files, the corresponding sample metadata, and NMR-derived composition references. Earlier project documentation identifies MassIVE accession **MSV000103648**; the accessibility and completeness of the deposited files must be confirmed before raw-data reproduction can be claimed.

Original raw files and unpublished data are not included in this GitHub repository.

## Interpretation and limitations

The reported LOO results are based on seven calibration composition groups, with an independent regression for each DP × NCE condition. Because the analysis configurations were developed retrospectively using the same calibration series, fixed-recipe LOO results do **not** measure fully selection-independent generalization.

The external GG-rich, MG-rich, and MM-rich materials have **bulk**, DP-unresolved NMR references (12%, 51%, and 82% M, respectively). Comparisons with those references are descriptive agreement with bulk composition, not independently measured DP3- or DP4-specific prediction accuracy.

## Citation and use

Manuscript citation details and an approved software license will be added after release preparation. The code is publicly viewable, but no open-source reuse license has yet been assigned.
