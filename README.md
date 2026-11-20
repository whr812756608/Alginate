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

Run the following commands from the repository root after completing the installation. Replace `/path/to/mzML_directory` with the directory containing your experimental mzML files.

### Complete workflow from raw mzML files

```bash
python scripts/run_pipeline.py \
  --data-dir /path/to/mzML_directory
```

This command preprocesses all DP3–DP7 calibration spectra at NCE18, NCE22 and NCE26, generates features, performs composition-grouped leave-one-out (LOO) evaluation, trains 15 final models, and predicts the three external samples for DP3 and DP4 at NCE26. Results are written under `outputs/`.

### Reuse preprocessed calibration spectra

If all 15 preprocessed calibration files already exist under `outputs/preprocessed/`, skip preprocessing:

```bash
python scripts/run_pipeline.py \
  --data-dir /path/to/mzML_directory \
  --reuse-preprocessed
```

To reuse preprocessed files stored elsewhere (for example, on a local cache drive), provide their directory explicitly:

```bash
python scripts/run_pipeline.py \
  --data-dir /path/to/mzML_directory \
  --reuse-preprocessed \
  --preprocessed-dir /path/to/stage1_cache
```

The preprocessed directory must contain all 15 files named `dp{dp}_nce{nce}_calibration.npz` for DP3–DP7 and NCE18, NCE22 and NCE26. The `--preprocessed-dir` option is valid only with `--reuse-preprocessed`. The `--data-dir` directory is still used for external mzML inference.

### Calibration-only workflow

If external mzML files are unavailable, use `--skip-external` to run the calibration workflow through final-model training:

```bash
python scripts/run_pipeline.py \
  --data-dir /path/to/mzML_directory \
  --skip-external
```

You can combine `--skip-external` with `--reuse-preprocessed` and `--preprocessed-dir`. To place generated results outside the repository, add `--output-dir /path/to/output_directory`.

### Run each stage separately

```bash
python scripts/run_preprocessing.py --data-dir /path/to/mzML_directory --dp all --nce all --out-dir outputs/preprocessed
python scripts/run_features.py --input-dir outputs/preprocessed --out-dir outputs/features
python scripts/run_loo.py --input-dir outputs/preprocessed --out-dir outputs/loo
python scripts/train_final_models.py --input-dir outputs/preprocessed --out-dir outputs/models
python scripts/predict_external.py --data-dir /path/to/mzML_directory --model-dir outputs/models --out-csv outputs/external_dp3_dp4_nce26.csv
```

When using an external preprocessed directory, use the same directory with `--input-dir` for feature extraction, LOO evaluation, and final-model training.

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
