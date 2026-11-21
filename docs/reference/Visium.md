::: iscc.data.Visium

## Technology presets

| Platform | Parameters | Notes |
|---|---|---|
| 10x Visium | defaults (`spot_pitch` = 2, `spot_radius` = 0.55, `mu_counts` ≈ 20000) | 55 µm spots at a 100 µm pitch with a deme of ~50 µm, so a spot pools about one deme's share of the section. |
| Visium HD | smaller `spot_pitch` / `spot_radius`, lower `mu_counts` | High-resolution (near-single-cell): many more, smaller spots. |
