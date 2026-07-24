# iscc pipeline & I/O schema

iscc simulates the full chain from tumor biology to observed molecular data as three
composable command-line stages. Each stage reads the previous stage's output directory
and writes its own, so pipelines are just directories on disk.

```
isccsim  ──>  <tumor>/        (grow a tumor: ground-truth state)
isccsample ──> <sample>/      (biopsy / dissociation: a subset of cells)
isccdata ──>  <data>/         (sequencing / spatial assay: observed data)
```

(`isccfig` / `isccgif` render figures/animations from a `<tumor>/` directory.)

---

## Stage 1 — `isccsim` (growth)

```
isccsim --sim-config CONFIG.yaml -s STEPS -o <tumor>
```

The sim-config selects a spatial `mode` (currently `glandular`; `mixed` is reserved) and
carries `spatial_params`, `genome_params`, `selection_params`, `deme_params`, and
per-type `cell_params` (cancer / epithelial / stromal / immune). See
`src/iscc/tumor/tumorconfigs/glandular.yaml`.

**Output `<tumor>/`:**

| Path | Contents |
|---|---|
| `trace_counts.csv` | genotype counts per step (Muller-plot input) |
| `parents.csv` | genotype → parent genotype |
| `genotypes.csv` | genotype list |
| `grid.csv` | most-frequent genotype per deme (spatial models) |
| `genotype_counts_demes.csv` | per-deme genotype counts (spatial models) |
| `gene_data/*.csv` | ground-truth gene annotations: `driver_types`, `dispersal_types`, `treatment_resistance_types`, `immune_resistance_types` |
| `cell_data/*.csv` | **per-cell ground truth** (the substrate for sampling + assays) |

**`cell_data/` schema** (rows = cells, indexed `C0…Cn`):

| File | Columns | Meaning |
|---|---|---|
| `cell_snv.csv` | genes | SNV VAF / indicator per gene |
| `cell_cnv.csv` | genes | copy number per gene (total) |
| `cell_exp.csv` | genes | transcriptional activity per gene |
| `cell_evo.csv` | evolutionary params + driver tallies | division/death/dispersal rates, resistances, `n_mut_*` |
| `cell_crd.csv` | `row`, `col` | deme coordinates on the grid |
| `cell_type.csv` | `cell_id` | genotype id (cancer) or cell type (normal) |
| `cell_deme.csv` | `deme_id` | which deme the cell is in |
| `cell_rna_vaf.csv` | genes | RNA-level B-allele VAF per gene |

The following are written **only when the corresponding feature is enabled** (the base schema above is
unchanged otherwise), so a `cell_data/` directory may additionally contain:

| File | Written when | Meaning |
|---|---|---|
| `cell_wgd.csv` | `wgd_rate > 0` | per-cell `is_wgd` flag (whole-genome-duplication status) |
| `cell_program.csv` | the gene-program layer is on (`expression_params`) | per-cell program activity (cells × programs; R13) |
| `cell_exp_p.csv`, `cell_exp_m.csv` | allele-specific dosage on (`allele_specific=True`) | paternal / maternal homolog expression per gene |
| `cell_rna_baf.csv` | allele-specific dosage on | RNA B-allele fraction per gene (allele imbalance) |
| `cell_gland.csv` | a ductal field is seeded (`n_glands` set) | `gland_id` per cell (which gland, `-1` = stroma) |
| `cell_compartment.csv` | a metastatic deposit is seeded (`met_grid_size > 0`) | per-cell `compartment` (0 = primary, 1 = metastasis; R9) |

When a metastatic deposit is enabled (`met_grid_size > 0`, R9) the tumor directory additionally
carries per-compartment abundance and a discrete-event log — both keyed to the **same** `parents.csv`
genealogy, so the primary and the metastasis share one clone identity/colour (the 2-band Muller). The
metastasis is a second deme-grid appended to the primary's lattice; `cell_compartment` disambiguates
the two, and met cells' `cell_crd` are in the met grid's own coordinate space.

| Path | Written when | Contents |
|---|---|---|
| `trace_counts_primary.csv`, `trace_counts_met.csv` | `met_grid_size > 0` | genotype counts per step, split by compartment |
| `events.csv` | any discrete event occurred | seeding / resection / chemo-window annotations (`step`, `time`, `event`, …) |

## Stage 2 — `isccsample` (biopsy / dissociation)

```
isccsample <tumor> --method {dissociation,biopsy} --fraction F -o <sample>
```

Selects a subset of cells (preserving the full per-cell ground truth) and records how the
sample was taken. **Output `<sample>/`:** a `cell_data/` directory using the **same
schema** as stage 1 (subset of rows), plus `sample_meta.yaml`
(`method`, `fraction`, `n_input`, `n_sampled`, `seed`, `source`).

> Current sampling is a random subset of the requested size for both methods; spatially
> realistic biopsy, physical slicing, and dissociation dropout/doublets are the next
> milestone. The on-disk contract above is stable.

## Stage 3 — `isccdata` (assay)

```
isccdata <sample> -a {scrna,bdna,scdna,visium} --assay-config CONFIG.yaml -o <data>
```

Runs the chosen assay on `<sample>/cell_data/`. Assay configs live in
`src/iscc/data/assayconfigs/`. **Output `<data>/`** depends on the assay:

| Assay | Files |
|---|---|
| `scrna` | `umis.csv` (cells × genes UMI counts) |
| `bdna` | `counts.csv` (per-gene coverage + alt counts) |
| `scdna` | `coverage.csv`, `alt_counts.csv` (cells × genes), or `observed_snvs.csv` in binary mode |
| `visium` | `spot_umi.csv`, `spot_crd.csv`, `spot_cell_counts.csv`, `spot_cell_ids.csv` |

For `visium`, the grid side is inferred from `cell_crd` unless `--grid-side` is given.
