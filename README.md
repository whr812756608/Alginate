# Alginate HCD-MS/MS composition prediction

Python implementation for MS-ML method estimating the NMR-calibrated mannuronic acid (M) fraction of alginate oligosaccharides (DP3–DP7) from HCD-MS/MS spectra. Each DP × normalized collision energy (NCE 18, 22, or 26) model combines **two learned nonnegative matrix factorization (NMF) coefficients**, **eight fragment-ion intensity ratios**, and **Ridge regression**.


## Pipeline Workflow

![Alginate MS/MS workflow: mzML loading, preprocessing, NMF and ion-ratio features, LOO validation, final training, and external inference](assets/workflow.svg)


## Installation

Python 3.10 or newer is required.

```bash
git clone https://github.com/whr812756608/Alginate.git
cd Alginate

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[mzml]'
```

## Input data

Raw HCD-MS/MS datasets are referenced under **MassIVE accession MSV000103648**.

Each DP has seven nominal calibration compositions (M0, M20, M40, M50, M60, M80, M100). Their DP-specific NMR-calibrated target values are specified in `configs/models.json`.

Place the calibration mzML files in one input directory, using the naming pattern:

```text
Raw_F{dp}_M{m}G{100-m}.mzML
```

where `dp` is 3–7 and `m` is 0, 20, 40, 50, 60, 80, or 100.

For external inference, the same directory must also contain:

```text
AlgG_LowDPMix_1_051625.mzML
AlgMG_LowDPMix_1_051625.mzML
AlgM_LowDPMix_1_051625.mzML
```

## Run the pipeline

From the repository root, run the full calibration and external-inference workflow:

```bash
python scripts/run_pipeline.py --data-dir /path/to/mzML_directory
```

To reuse existing preprocessed calibration spectra:

```bash
python scripts/run_pipeline.py \
  --data-dir /path/to/mzML_directory \
  --reuse-preprocessed
```

To run calibration evaluation and final-model training without external files, add `--skip-external`.

The same pipeline can be executed step by step:

```bash
python scripts/run_preprocessing.py --data-dir /path/to/mzML_directory --dp all --nce all --out-dir outputs/preprocessed
python scripts/run_features.py --input-dir outputs/preprocessed --out-dir outputs/features
python scripts/run_loo.py --input-dir outputs/preprocessed --out-dir outputs/loo
python scripts/train_final_models.py --input-dir outputs/preprocessed --out-dir outputs/models
python scripts/predict_external.py --data-dir /path/to/mzML_directory --model-dir outputs/models --out-csv outputs/external_dp3_dp4_nce26.csv
```

## Outputs

| Output | Description |
| --- | --- |
| `outputs/preprocessed/` | Fifteen preprocessed calibration spectral datasets |
| `outputs/features/` | Ten-feature matrices: two NMF coefficients and eight fixed ratios |
| `outputs/loo/loo_folds.csv` | 105 composition-level LOO predictions |
| `outputs/loo/loo_by_dp_nce.csv` | LOO MAE reported separately for each DP and NCE |
| `outputs/models/` | Fifteen saved final models and model manifest |
| `outputs/external_dp3_dp4_nce26.csv` | Six predictions: GG, MG, MM for DP3 and DP4 at NCE26 |


## Citation

A manuscript citation and tagged software release will be added when available.
