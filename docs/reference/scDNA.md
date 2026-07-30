::: iscc.data.scDNA

## Technology presets

Single-cell DNA platforms differ mainly in the whole-genome-amplification (WGA) chemistry and depth.
Set `breadth` for the locus set / depth regime, and override the amplification knobs — `kappa`
(lower = lumpier amplification), `ado_rate` (allelic dropout), and `mu_depth` (coverage) — via the
constructor. These are starting points; [`estimate_dna`](estimate_dna.md) fits them from real data.

| Platform | `breadth` | `kappa` | `ado_rate` | Depth | Notes |
|---|---|---|---|---|---|
| MALBAC (default) | `"wgs"` | ~5 | ~0.20 | shallow (~9×) | Quasi-linear WGA, moderate bias — the built-in single-cell defaults. |
| MDA | `"wgs"` | ~2–3 | ~0.30 | shallow | Stronger lumpiness and higher dropout than MALBAC. |
| DLP / DLP+ | `"wgs"` | ~500 (near-uniform) | ~0.05 | very shallow (low `mu_depth`; DLP+ is ~0.01–0.1× the genome) | Direct single-cell library, no WGA bias: near-uniform coverage, low dropout. |
| Mission Bio Tapestri | `"panel"` | ~5–10 | ~0.15 | very deep | Targeted single-cell SNV amplicon panel. |
| 10x CNV (Chromium) | `"wgs"` | ~50 | ~0.05 | very shallow | Droplet shallow WGS for copy number. |
