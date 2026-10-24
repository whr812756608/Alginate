# Research code-sharing package — October 23, 2026

## Intended audience

The authors and collaborators preparing a GitHub code release for the alginate
MS–ML *Nature Methods* manuscript. This is a **candidate release**, not an
assertion that the original code has been published or comprehensively audited.

## Contains

1. **Primary V7.0 analysis** with shared V6.6 base modules, seven-composition
   holdout evaluation, whole-composition permutation tests, matched K8 vs K2+K8
   comparison, and frozen external inference.
2. **Supporting V6.7–V6.9 experiments** illustrating historical optimization
   and robustness checks. They are not substituted for primary figures.
3. **Figure 3**: DP3/NCE26 technical-scan NMF weight plot source (experimental
   data must be supplied separately), Figure 3e cross-validation schematic and
   3f V7.0 MAE heatmap source code, their user-approved SVG exports.
4. **Figure 4**: existing vector art and numerical source CSVs, with an explicit
   warning that it includes an exploratory DP4 comparison and cannot certify
   external quantitative accuracy.
5. **Source data helpers** for S4 24/34 fragment center audit, S9 seven-fold
   composition scan counts, and Fig. 3f 15-condition heatmap.
6. Synthetic tests and methodology/limitation documents.

## Main changes versus October 22 package

- Replaced previous Figure 3c role-based titles with **neutral NMF component
  (W1/W2)** labels; corrected y-axis to **NMF weight (a.u.)**.
- Figure 3c source uses repository-local shared dependencies; no hardcoded
  location to a developer's desktop is required (supply `--data-dir`).
- Synced Figure 3e and 3f SVGs with the author's latest approved files.
- Added scripts and CSV tables to support SI fragment-center annotation
  (34 theoretical, 24 implemented) and reported scan counts.

## Before public GitHub publication

- **Obtain explicit coauthor/code-owner approval.** This download does not
  upload anything to GitHub. Choose an appropriate license after approval.
- Verify code/data match the manuscript version. For full reproducibility,
  provide accessible raw mzML (the manuscript cites MassIVE MSV000103648)
  and a final-data-derived Figure 3c scan-level CSV.
- Do not treat synthetic demo art, synthetic tests, or retrospective DP4
  tuning as experimental validation.
- The ML runs were not recomputed from raw mzML during this packaging step.
  The included V7.0 CSVs are **previously reported exports**.
- V7.0 permutation results contain feature-selection fallback for many null
  folds and an approximately tied reversed-label ordering; see Methods notes.
- One script `alginate_ratio_transfer_IDE.py` defines 24 implemented centers
  whereas the original annotated SI contains 34 theoretical assignments;
  consult the source workbook for assignment labels/DP applicability.
- Acquisition '20 scans each' text has not yet been reconciled with the
  321–349 technical scans used per composition in V7.0.
- No automatic broad license is granted by this package.

## Fast reproducibility checks

```bash
python -m pip install -r requirements.txt
python experiments/alginate_V70_EVIDENCE_EXISTING_DATA_IDE.py --run-mode selftest
python figures/figure3c/plot_figure3c_nmf_errorbars.py --mode selftest
python tests/test_figure3_scripts.py
python tools/build_manuscript_source_tables.py --out-dir outputs/source_tables
```

See `README.md`, `docs/ANALYSIS_GUIDE.md`, and `docs/PUBLICATION_CHECKLIST.md`.

## Packaging-time checks

- Python syntax compilation: passed.
- Figure 3c neutral-title demonstration, figure 3e/3f exports: passed.
- Publication-source CSV integrity: passed for 15 MAE cells, 105 held-out composition predictions, and 24 retained centers.
- V7.0 built-in synthetic self-test: passed.
- V6.6 built-in synthetic integration test: passed.
- V6.7 test recorded PASS but was computationally expensive.
- Remaining full synthetic test suite, raw-mzML full pipeline, and raw Figure 3c numerical reproduction have not all been run in this environment.
