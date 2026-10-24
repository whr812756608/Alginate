# Publication / GitHub release checklist

- [ ] Confirm with every coauthor/code owner that this code can be published; existing GitHub `Alginate` repository is PUBLIC.
- [ ] Decide on a software license (none supplied here).
- [ ] Confirm MassIVE accession and availability of original raw spectra to readers.
- [ ] Confirm calibration NMR labels and their fraction-derived uncertainty against SI tables.
- [ ] Verify exact Gaussian kernel effective sigma; retained scan counts vs acquisition settings; NMF solver.
- [ ] Freeze primary figure-generation experiment and regenerate reported tables from raw mzML.
- [ ] Add eventual V6.9 and V7.0 real-data outputs only when the experiments finish (do not substitute synthetic tests).
- [ ] Explicitly describe the retrospective nature of any external-bulk-agreement-selected model.
- [ ] Ensure no proprietary raw datasets, patient/user personal data or unfinished manuscript drafts are committed.


## Figure 3 and V7.0 release checks (2026-10-22)

- [ ] Generate **real** Figure 3c NMF activation mean±SD (seven compositions, no endpoint clipping) from private mzML and deposit CSV source data before describing it as measured. Never submit the bundled synthetic preview.
- [ ] Verify figure 3f heatmap CSV and figure caption match the one fixed-preprocessing V7.0 analysis (15 conditions, n=7 compositions per cell).
- [ ] Figure 3e column percentages reflect **nominal stock-mixing proportions**, not actual NMR-calibrated M fraction.
- [ ] Report the global permutation p-value with the null-model fallback caveat, not as proof of arbitrary unseen AOS accuracy.
- [ ] Decide which items are approved for a **public** GitHub release, agree on code license, then create a release tag/commit SHA and cite that immutable version.
