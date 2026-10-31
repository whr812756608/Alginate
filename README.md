# Alginate MS–ML — NMR-calibrated AOS composition modeling

Reproducible **research-development code** for learning composition-associated fragmentation patterns in alginate oligosaccharides (AOS) without requiring sequence-pure AOS standards.

**Scope:** DP3–DP7 calibration mixtures at HCD NCE 18%, 22%, and 26%, using NMR-calibrated mixture labels, two-component nonnegative matrix factorization (K2) and eight diagnostic fragment-ion ratios (K8), standardized Ridge regression, and leave-one-out cross-validation (LOO-CV), holding out all spectra from one calibration composition per fold.

## Quick start

Requires Python 3.10+ (3.11 recommended) and the original mzML files for real-data runs. Synthetic tests require no real mzML.

```bash
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

The fixed-protocol analysis applies the same preprocessing configuration across DP3–DP7 and NCE 18%, 22%, and 26%. It evaluates prediction of held-out calibration compositions using LOO-CV and includes K8-only versus K2+K8 comparisons and 199 grouped-label permutations. These analyses assess prediction within the calibration-mixture system, not independently measured DP-specific external accuracy.

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

## Reproduce analysis

The primary training dataset contains seven nominal mixture ratios (0, 20, 40, 50, 60, 80, 100% MM-enriched source fraction) for each nominal DP, with true **calibration M% calculated from measured stock-fraction NMR M fractions and mixing proportions**. Nominal 0% and 100% do not mean pure-G or pure-M AOS.

See [`docs/DATA_LAYOUT.md`](docs/DATA_LAYOUT.md) for filenames, [`docs/ANALYSIS_GUIDE.md`](docs/ANALYSIS_GUIDE.md) for the experiments and manuscript-relevant outputs, and [`docs/METHODS_AND_LIMITATIONS.md`](docs/METHODS_AND_LIMITATIONS.md) for scientific interpretation and open checks.

For complete per-version options and version-specific output files, see the five `docs/legacy_readmes/` entries. The default run mode of most experiments is **overnight**; explicitly pass `--run-mode selftest` first.


## Data and code availability

The manuscript reports HCD-MS/MS deposition at MassIVE accession **MSV000103648**; verify that its actual files and access permissions match the input naming conventions before citing this repository as a complete reproduction. Raw mzML, instrument-native files, calibration stocks' NMR raw datasets, unpublished manuscript Word documents, and checkpoint/prediction caches are **not distributed in this repository**. Do not commit them without the data owners' approval.

No open-source license is assigned by this package. Publication permissions and a license should be agreed with all code contributors before making this repository public.


## Figure 3 source code and publication exports (2026-10-22 update)

Figure 3e and 3f are provided as **editable/vector SVG**, PNG, and native PowerPoint
source objects. They intentionally omit the panel letters and titles; details belong
in the figure caption. Figure 3e retains the training/held-out legend and the nominal
mixing labels (0%, 20%, 40%, 50%, 60%, 80%, 100%). Its full 100% column is visible.
Figure 3f summarizes fixed-protocol LOO-CV mean absolute error (15 DP/NCE conditions; one complete calibration composition held out per fold).

To regenerate panels 3e and 3f from the included CSV:

```bash
python figures/figure3e_f/generate_clean_panel_e_f.py \
  --out-dir outputs/figure3ef
```

Figure 3c requires the **real DP3/NCE26 calibration mzML** for actual NMF mean and
scan-level SD. Do not use the included watermarked synthetic preview in the manuscript.
The full-fit NMF in Figure 3c is descriptive, not held-out validation:

```bash
python figures/figure3c/plot_figure3c_nmf_errorbars.py --mode selftest
python figures/figure3c/plot_figure3c_nmf_errorbars.py \
  --mode real --data-dir /path/to/private/alginate_experiment \
  --dp 3 --nce 26 --out-dir outputs/figure3c
```


## October 23 code-release update

This release incorporates the latest neutral Figure 3c component labels; verifies the two
user-approved Figure 3e/f SVGs; and adds independently checkable source-data exports:
`source_data/figure3f_validation_summary.csv`, `source_data/supplementary_scan_counts_S9.csv`,
and `source_data/supplementary_fragment_center_annotations.csv`.
`tools/build_manuscript_source_tables.py` regenerates these tables from **existing
reported CSVs / implemented code**, without implying raw mzML was reprocessed.
**See [RELEASE_NOTES_20261023.md](RELEASE_NOTES_20261023.md) before making this repository public.**
