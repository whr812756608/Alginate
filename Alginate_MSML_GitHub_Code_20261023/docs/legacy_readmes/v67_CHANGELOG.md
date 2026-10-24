# V6.7 changes relative to V6.6

- Restricts analysis to DP5, DP6, DP7 with no unfractionated external dataset assumed.
- Preserves original 7-window group-area normalization as a locked comparison.
- Adds DP-aware extended fragment families through the DP7 mass range, omitting each DP's intact precursor from new group-normalization regions.
- Adds calibration-train-only group signal guards to avoid amplifying weak high-mass fragment noise.
- Compares Gaussian sigma 0, 0.10, 0.25, 0.50, 0.75, 1.00 Da across all three collision energies.
- Adds nested leave-one-composition-out protocol selection distinct from exploratory minimum-MAE grid search.
- Adds scan-acquisition/retention counts per DP × NCE × calibration mixture for manuscript sample-size reporting.
- Provides Figure 3f 3×3 matrices, original-baseline and locked transfer protocol summaries, and K8 versus K2+K8 controls.
- No new/external NMR labels; never infers DP5–DP7 external accuracy.
