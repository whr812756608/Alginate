# V6.6 additions over V6.3–V6.5

- Full factorial sweep for **all NCE 18, 22 and 26 in one run**, DP3 and DP4.
- **Six** Gaussian sigma values: 0, 0.10, 0.25, 0.50, 0.75, 1.00 Da.
- **Eight** group definitions: ORIGINAL7 and all seven nonempty combinations of the m/z 291/545/721 extensions.
- **Three** group-normalization strengths: TIC-only, partial, full.
- Cache only six smoothed/masked arrays per DP (and external), not hundreds of copies of group-normalized arrays; normalization is built on demand.
- Freeze the eight ratio **identities** using training-fold ORIGINAL7 reference per DP/NCE, preventing an uncontrolled ratio-reselection confound.
- Matched K8-only and K2+K8 models with the same ratio identities, seven-composition held-out CV.
- Save per-composition-per-preprocessing CSV checkpoints and input-file fingerprints; resume after interruption.
- Automatically report calibration-selected preprocessing per DP/NCE and rank all three shared NCEs for a combined DP3+DP4 figure.
- Add raw-centroid denominator support, clipping, OOD, scan-half repeatability, cross-energy variation and descriptive bulk-NMR context.
