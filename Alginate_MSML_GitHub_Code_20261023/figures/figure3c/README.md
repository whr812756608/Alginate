# Figure 3c: neutral NMF component labeling

The updated code produces two panels labeled **NMF component (W1)** and **NMF component (W2)**, with y-axis **NMF weight (a.u.)**. These are data-derived latent spectral features, **not chemically isolated M and G species**. In the Results, discuss their observed positive/negative trends with NMR-calibrated M fraction rather than assigning chemical identities.

## Reproduce

Run in the repository root after dependencies are installed:

```bash
python figures/figure3c/plot_figure3c_nmf_errorbars.py --mode selftest
python figures/figure3c/plot_figure3c_nmf_errorbars.py --mode real \
  --data-dir /path/to/private/calibration_mzML --dp 3 --nce 26 \
  --out-dir outputs/figure3c
```

The self-test checks grouping/SD calculations only. Real-data mode requires the original mzML source. **No real experimental Figure 3c NMF weights are included.** `examples/SYNTHETIC_PREVIEW_*` illustrates the layout and is watermarked synthetic; do not publish as an experimental figure.

Each marker represents the mean across technical scans within a calibration composition and its error bar is ±1 scan-level SD, not an independent biological-replicate uncertainty. The V7.0 analysis includes 333–336 technical DP3/NCE26 scans per composition, but verify counts against the mzML used for the final Figure 3c plot.

## Important mapping

NMF factors can be permuted between fits. To keep the same presentation order, the script displays the component more positively correlated with M% first; the export's metadata maps display labels **W1/W2** to the raw NMF columns. The factor labels should not be interpreted as molecular identities.
