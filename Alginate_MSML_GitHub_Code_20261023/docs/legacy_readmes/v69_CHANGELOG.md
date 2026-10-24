# V6.9 — DP3 all-NCE diagnostic (2026-10-19)

- Added DP3 spectral-domain-transfer inspection at NCE 18, 22 and 26 simultaneously.
- Preserved unmodified V6.6 data loading, DP-specific NMR calibration labels, Gaussian/group normalization, train-only K8 selection, K2 NMF and Ridge feature construction.
- Frozen two preprocessing references per NCE: original (σ1/ORIGINAL7/λ1) and V6.6 calibration-selected.
- Added scan-level 8-ratio and individual numerator/denominator support diagnostics for seven known calibration mixtures and three unlabeled external materials.
- Added 8 individually refitted drop-one-ratio models at each fold; same NMF basis, refit scaler + Ridge without the deleted ratio.
- Added K8-only comparison, global NMF reconstruction hotspots, q99 OOD flags, scan clipping, moving-block bootstrap for technical scan mean and across-NCE consistency.
- Added complete 48-checkpoint path (3 NCE × 2 preprocessing × [7 held-out folds + final full fit]), caching, restart-safe settings signature, per-sample retained-scan audit and readable diagnostic figures.
- Bulk NMR is never treated as DP3 truth, and is never used to rank/tune/drop ratios. No automatic correction to external predictions.
- Verified synthetic three-NCE full grid and per-NCE output using included acceptance tests; production mzML run is not claimed.
