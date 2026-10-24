# V7.0 changes from V6.6/V6.7/V6.9

- New synchronized label-permutation experiment over all five DPs and three NCEs using the original NMR-calibrated mixture labels.
- Retains V6.6 ORIGINAL7, Gaussian sigma 1 Da, full-strength group-area normalization, train-only reference K8 selection, NMF rank 2 and Ridge alpha 1; source dependencies are included without modifications.
- Reuses an unsupervised, label-neutral NMF basis across permutations within each held-out composition, maintaining exact V6.6 predictions for authentic labels (verified by numerical test).
- Never shuffles individual scans, never fits on external NMR context.
- Explicit fallback to composition-training-mean when eight reliable ratios are unavailable under a permuted label arrangement.
- Adds synchronized global across-condition permutation significance, held-out interior versus endpoint checks, frozen-model DP3/DP4 external NCE consistency and descriptive bulk NMR comparison.
- Checkpoint/resume, source fingerprints, and no-mzML synthetic end-to-end acceptance tests (numerical fitting and report figures).

Synthetic checks do not replace real mzML verification. Exact NMF solver settings and peak definitions are inherited from user-provided V6.6 source.  External predicted M% from this workflow must not be optimized toward the three bulk NMR values.
