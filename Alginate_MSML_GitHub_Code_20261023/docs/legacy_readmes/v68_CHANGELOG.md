# Alginate V6.8 changelog

- Reuses the V6.6 DP3 data loader and exact DP3/NCE26 σ0.25 ORIGINAL7 λ1.0 K2+K8 reference.
- Adds calibration-only higher-support, inner-consensus, and ion-perturbation-robust alternative K8 selection.
- Adds log-ratio processing with pseudocounts determined exclusively from calibration training scans.
- Adds a precursor-excluded rank-two NMF option without changing K8 features.
- Adds four output aggregation methods, raw-ion detection QC, reconstruction OOD, scan-halves, and effective QC-weight sample counts.
- Outputs both calibration-based shortlist and separately flagged descriptive bulk-NMR disagreement.
- Resumable checkpoints per held-out composition / arm, frozen-model outputs, input fingerprints and settings signature.
- Fixed full-calibration `EXTERNAL` checkpoint RNG-label handling for simulated perturbation selection during testing.
- Verified DP3 baseline numeric parity with V6.6 using synthetic data. No real mzML evaluation claimed.
