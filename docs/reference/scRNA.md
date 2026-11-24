::: iscc.data.scRNA

## Technology presets

Pass `protocol=` to select a platform preset, or override individual knobs. Two presets are built
in; the rest are parameter starting points you can set explicitly, or fit from a real dataset with
[`estimate_rna`](estimate_rna.md).

| Platform | `protocol` | Profile |
|---|---|---|
| 10x Chromium 3′ / 5′ | `"10x"` | Droplet UMI: `mu_lib` ≈ 4000, `dispersion` = 0.35 (`estimate_rna` on PBMC3k, within cell types), no dropout (`dropout_mid` = 0), `ambient_frac` = 0.05, `doublet_rate` = 0.05, no per-well term. |
| Smart-seq3 | `"smartseq3"` | Plate UMI: deeper (`mu_lib` ≈ 8000), no dropout (`dropout_mid` = 0), low ambient (0.005), `well_sigma` = 0.15. |
| Smart-seq3xpress | `"smartseq3"` + lower `mu_lib` (~3000–5000) | Miniaturised Smart-seq3 — same plate profile, shallower per cell. |
| Smart-seq2 | `"smartseq3"` + `dispersion` ≈ 0.4 | Plate, no UMIs → read counts carry more amplification noise (raise `dispersion`); full-length gene-body coverage is not modelled. |

Neither preset adds dropout. UMI counts are not zero-inflated beyond the negative binomial
(Svensson 2020): within the cell types of PBMC3k, the negative binomial alone gives the zero
fraction at every mean, and leaves no excess zeros for a dropout to explain. `dropout_mid`
switches on a logistic dropout for data that need one. Its midpoint is in counts per gene, so the
same midpoint deletes far more of the genes of a cell over the 36,601 human genes than over the
growth genes, and it acts on every molecule in the droplet, ambient ones included.

The ambient soup is a share of each library (`ambient_frac` of it), not counts on top, so it does
not add depth. With `count_model="dm"` the Dirichlet-multinomial concentration `kappa` is 1,500 at
10x, fitted by moments within the cell types of PBMC3k; it inflates the variance of every gene by
the same factor, so the negative binomial, whose excess grows with the mean, stays the default.
