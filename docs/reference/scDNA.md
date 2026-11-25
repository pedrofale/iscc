::: iscc.data.scDNA

## Technology presets

Single-cell DNA platforms differ mainly in the amplification chemistry and the depth.
The amplification concentration `kappa` sets the coverage spread: the excess squared coefficient of
variation of a locus is about `kappa_loci / kappa`, whatever the number of loci, so each `kappa` is
given with the number of loci it was fitted on.
We fitted every value below with [`estimate_dna`](estimate_dna.md) on a public reference of the
platform, and `platform=` selects the fit.
Any keyword argument overrides the preset.

| Platform | `platform` | `breadth` | Depth per locus | `kappa` (`kappa_loci`) | `ado_rate` | Notes |
|---|---|---|---|---|---|---|
| Default | `None` | `"wgs"` | 9 (0.3 × the WGS depth) | 204,222 (5,787) | 0.20 (prior) | The DLP+ dispersion at the breadth depth: on a 6,000-locus genome, a median coverage CV of 0.40 at two copies and under 0.1% of loci without reads. |
| DLP+ | `"dlp"` | `"wgs"` | 830 | 204,222 (5,787) | 0.20 (prior) | Fitted on 40 cells of OV2295 (Laks et al. 2019) in 500 kb bins. A locus counts the reads of a whole bin, not the reads over its variant site. Coverage CV 0.22 at two copies, against 0.23 in the reference. The reference has no allele counts, so the allele layer keeps the priors. |
| Mission Bio Tapestri | `"tapestri"` | `"panel"` | 42.7 | 5,757 (584) | 0.325 | Fitted on 800 cells of the KaryoTap panel (Mays et al. 2024), with copy number called per chromosome arm, `capture_sigma` 0.326 and `beta_binom_conc` 4.5. Coverage CV 0.49 at two copies, against 0.43 in the reference, and close to it at one to five copies. |

We ship no fit for MALBAC, MDA or 10x Chromium single-cell CNV.
For these platforms, fit `estimate_dna` on data from the platform and pass the result as
`scDNA(breadth=est.breadth, depth_model=est.depth_model, **est.dna_kwargs())`.
