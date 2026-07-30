::: iscc.data.scRNA

## Technology presets

Pass `protocol=` to select a platform preset, or override individual knobs. Two presets are built
in; the rest are parameter starting points you can set explicitly, or fit from a real dataset with
[`estimate_rna`](estimate_rna.md).

| Platform | `protocol` | Profile |
|---|---|---|
| 10x Chromium 3′ / 5′ | `"10x"` | Droplet UMI: `mu_lib` ≈ 4000, `dropout_mid` = 1.5 (real dropout), `ambient_frac` = 0.05, `doublet_rate` = 0.05, no per-well term. |
| Smart-seq3 | `"smartseq3"` | Plate UMI: deeper (`mu_lib` ≈ 8000), no dropout (`dropout_mid` = 0), low ambient (0.005), `well_sigma` = 0.15. |
| Smart-seq3xpress | `"smartseq3"` + lower `mu_lib` (~3000–5000) | Miniaturised Smart-seq3 — same plate profile, shallower per cell. |
| Smart-seq2 | `"smartseq3"` + `dispersion` ≈ 0.4 | Plate, no UMIs → read counts carry more amplification noise (raise `dispersion`); full-length gene-body coverage is not modelled. |
