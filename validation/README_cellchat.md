# CellChat on iscc data — env setup and verified constraints

Written 2026-08-27 from a round-trip experiment run BEFORE any implementation, to settle whether
CellChat will accept a ligand-receptor database iscc invents over its own abstract gene
identifiers (`G_1_10`, `G_3_30`, ...) rather than real human symbols. **It does**, for both the
RNA and the spatial pipeline. That is what allows the naming layer (W1) to stay dead.

Design context: `DESIGN_cci_spatial.md` sections W0/W3. Sibling of `README_integration.md`.

---


**Verdict: PROVEN.** CellChat accepts a fully invented ligand–receptor database over
iscc-style identifiers (`G_1_10`, `G_3_30`, …). The complete pipeline — RNA *and*
spatial — runs end to end and returns a communication table naming our own pairs.
No gene renaming to human symbols is required. The risk is retired.

There is exactly **one hard constraint** (§4.1) and a handful of gotchas (§8) that
will otherwise cost a future implementer hours, because they fail *silently* or with
misleading errors.

---

## 1. Environment

Built: `iscc-cellchat` at `/Users/pedroferreira/miniconda3/envs/iscc-cellchat` (osx-arm64, 1.6 GB).

| | |
|---|---|
| R | 4.5.3 |
| **CellChat** | **2.2.0.9001** (`jinworks/CellChat`, branch `main`) |
| NMF | 0.28 (CRAN source build) |
| presto | 1.1.0 (`immunogenomics/presto`) |
| ComplexHeatmap | 2.26.1 |
| BiocNeighbors | 2.4.0 |

### Install route that worked

```bash
CONDA=/Users/pedroferreira/miniconda3/bin/conda
$CONDA create -y -n iscc-cellchat -c conda-forge -c bioconda \
  'r-base=4.5' r-remotes r-devtools r-biocmanager \
  r-matrix r-igraph r-dplyr r-ggplot2 r-ggrepel r-reshape2 r-scales r-rcolorbrewer \
  r-patchwork r-future r-future.apply r-pbapply r-expm r-shape r-sna r-svglite \
  r-ggalluvial r-circlize r-reticulate r-fnn r-irlba r-rtsne r-rcpp r-rcppeigen r-rcpparmadillo \
  r-registry r-rngtools r-gridbase r-foreach r-doparallel r-cluster r-digest r-stringr r-colorspace \
  r-ggpubr r-cowplot r-plyr r-data.table r-r.utils \
  bioconductor-complexheatmap bioconductor-biocgenerics bioconductor-biobase

$CONDA install -y -n iscc-cellchat -c conda-forge -c bioconda \
  r-rspectra r-magrittr r-ggnetwork r-plotly r-shiny r-bslib r-collapse \
  r-network r-statnet.common bioconductor-biocneighbors

E=/Users/pedroferreira/miniconda3/envs/iscc-cellchat
export PATH=$E/bin:$PATH          # required: conda's clang/gfortran must be on PATH
$E/bin/Rscript -e 'install.packages("NMF", repos="https://cloud.r-project.org")'

# GitHub tarballs (see §1.1 — remotes::install_github times out here)
curl -L -o CellChat.tar.gz https://codeload.github.com/jinworks/CellChat/tar.gz/refs/heads/main
curl -L -o presto.tar.gz   https://codeload.github.com/immunogenomics/presto/tar.gz/refs/heads/master
tar xzf CellChat.tar.gz && tar xzf presto.tar.gz
$E/bin/R CMD INSTALL presto-master
$E/bin/R CMD INSTALL CellChat-main
```

### 1.1 Install obstacles hit (all resolved)

1. **`r-nmf` does not exist on conda-forge or bioconda.** NMF is a hard CellChat
   `Imports:` and must come from CRAN as a source build. (`r-nmfn` in the search
   results is a different, unrelated package.)
2. **R 4.3 is not viable.** `bioconductor-biocneighbors` (a CellChat `Imports:`)
   only has builds requiring `r-base >=4.4`; the current one requires `>=4.5`.
   Pin `r-base=4.5`.
3. **`remotes::install_github()` failed** — `Timeout was reached [api.github.com]`
   after 10 s. Downloading the tarball from `codeload.github.com` and using
   `R CMD INSTALL` worked. Note `install_github` ignores `options(timeout=)`; it uses
   its own curl handle.
4. **CellChat's default branch is `main`, not `master`.** A `refs/heads/master`
   tarball URL returns a 14-byte body containing `404: Not Found` — `curl` exits 0,
   so this fails *silently* and only shows up as a corrupt-tarball error later.
5. Source compilation into conda R works with the env's own toolchain
   (`arm64-apple-darwin20.0.0-clang`), provided `$E/bin` is on `PATH`. A benign
   `SafetyError` about `r-base .../packages.html` size appears during env creation
   and can be ignored.

---

## 2. The synthetic database

8 interactions, strict 1:1 (no complexes), 4 pathways, deliberately mixed
`annotation` so both spatial branches get exercised:

| ligand | receptor | pathway_name | annotation |
|---|---|---|---|
| G_1_10 | G_3_30 | PW_A | Secreted Signaling |
| G_1_25 | G_3_51 | PW_A | Secreted Signaling |
| G_2_11 | G_4_40 | PW_B | Secreted Signaling |
| G_2_44 | G_4_62 | PW_B | Secreted Signaling |
| G_5_7  | G_6_12 | PW_C | Secreted Signaling |
| G_5_19 | G_6_77 | PW_C | Cell-Cell Contact |
| G_7_3  | G_8_5  | PW_D | Cell-Cell Contact |
| G_9_88 | G_10_2 | PW_D | Cell-Cell Contact |

`complex` and `cofactor` are **empty**; `geneInfo` is a **single-column**
`data.frame(Symbol = <all 16 fake ids>)`.

```r
db.iscc <- updateCellChatDB(
  db         = interaction_input,                     # data.frame
  gene_info  = geneInfo_input,                        # data.frame(Symbol=...)
  other_info = list(complex = data.frame(), cofactor = data.frame())
)
cellchat@DB <- db.iscc
```

Accepted. `extractGene(db.iscc)` returns all 16 fake ids. The CSV round trip
(`read.csv` → `updateCellChatDB`), i.e. exactly the `Update-CellChatDB` tutorial
shape, also works.

---

## 3. Full RNA pipeline — WORKS

300 cells, 3 labels (TumorA/TumorB/Stroma), 196 genes (16 L-R + 180 filler, all
`G_x_y`), NB counts → `normalizeData()`.

```
createCellChat -> @DB<-db.iscc -> subsetData -> identifyOverExpressedGenes
  -> identifyOverExpressedInteractions -> computeCommunProb -> filterCommunication
  -> computeCommunProbPathway -> aggregateNet -> subsetCommunication
```

- `subsetData` recovers **16/16** signalling genes.
- `identifyOverExpressedInteractions`: "The number of highly variable
  ligand-receptor pairs used for signaling inference is **8**".
- `subsetCommunication()` returns **39 rows naming our pairs**, e.g.

  ```
  source target ligand receptor interaction_name pathway_name      prob pval
  TumorA Stroma G_1_10   G_3_30    G_1_10_G_3_30         PW_A 0.5793536    0
  Stroma TumorB G_2_11   G_4_40    G_2_11_G_4_40         PW_B 0.5778449    0
  TumorB TumorA  G_5_7   G_6_12     G_5_7_G_6_12         PW_C 0.5704412    0
  TumorA TumorB  G_7_3    G_8_5      G_7_3_G_8_5         PW_D 0.5777510    0
  ```

  The four planted directions are recovered as the maximum-probability entry of
  their pathway. `subsetCommunication(slot.name="netP")` gives the pathway-level table.
- `netAnalysis_computeCentrality`, `netVisual_bubble`, `netVisual_circle`,
  `netVisual_heatmap`, `netAnalysis_signalingRole_network`,
  `showDatabaseCategory` all work on the custom DB.

---

## 4. Constraints discovered

### 4.1 THE hard constraint — `geneInfo$Symbol` is the gene whitelist

`extractGeneSubset()` (`R/database.R`) does:

```r
geneSet <- intersect(geneSet, geneIfo$Symbol)
```

Any ligand/receptor **not** present in `geneInfo$Symbol` and not a rowname of
`complex` is **silently dropped from the analysis**. It is not an error. The
interaction table keeps all its rows; the pair simply never appears in the output.

Demonstrated: removing `G_1_10` and `G_3_30` from `geneInfo` gave
`checkGeneSymbol` printing a `cat()` notice, then
`"...ligand-receptor pairs used for signaling inference is 7"` and **0 rows** for
`G_1_10_G_3_30` in `subsetCommunication()`.

**Implication for iscc: emit `geneInfo` containing every gene the DB references.**
Never call `updateCellChatDB(gene_info = NULL, species_target = "human")` — that
substitutes the 26 827-symbol human table, and we measured
`extractGene()` returning **0 genes** with our identifiers. Everything downstream
would then be empty with no error.

`checkGeneSymbol()` only `cat`s "Issue identified!!" and returns `FALSE`; its
return value is discarded by callers. Treat that message as fatal in any wrapper.

### 4.2 Required / optional columns of `interaction`

- **Required:** `ligand`, `receptor`. That is genuinely all `updateCellChatDB`
  enforces.
- **Auto-generated if absent:** `pathway_name` (empty + warning),
  `interaction_name` (= `toupper(ligand)_toupper(receptor)`), `interaction_name_2`
  (= `"ligand - receptor"`), `agonist`, `antagonist`, `co_A_receptor`, `co_I_receptor`.
- **`annotation` is NOT auto-added for RNA data** and is *not* in the auto-fill list.
  Supply it explicitly. Legal values are exactly
  `"Secreted Signaling"`, `"ECM-Receptor"`, `"Non-protein Signaling"`,
  `"Cell-Cell Contact"` — CellChat `factor()`s the column against those levels and
  any other string becomes `NA`, which silently reorders/corrupts the pair ordering
  used to split diffusive vs contact-dependent pairs in spatial mode.
- `interaction_name` must be unique (duplicates are dropped with a warning) and
  becomes the rownames. Underscores in gene ids are fine — nothing parses
  `interaction_name` by splitting on `_`.
- Extra columns (we added `evidence`) are preserved and passed through.

### 4.3 Naming rules for the gene identifiers

None beyond §4.1. `G_1_10`-style ids with digits and underscores work throughout,
including in plot labels, heatmap rownames and `spatialFeaturePlot(features=...)`.
No species annotation, Ensembl lookup, or symbol validation is applied anywhere.

### 4.4 Empty `complex` / `cofactor`

Allowed — but **the shape matters** (see §8.1). Use a bare `data.frame()`
(0 rows × 0 cols) or a table with **≥ 2 columns**.

Non-empty complexes with invented names also work: a receptor named `RC_alpha`
with `complex["RC_alpha", ] = (G_3_30, G_3_51)` expands correctly, so iscc can
model receptor heterodimers later if wanted.

---

## 5. Spatial mode — WORKS

440 spots on an ideal Visium-like hex grid (100 µm pitch), coordinates authored
directly **in micrometres**, three spatial domains (TumorA | Stroma | TumorB).

```r
spatial.factors <- data.frame(ratio = 1, tol = 65/2)   # ratio=1 because coords are already µm
cc <- createCellChat(object = data.input, meta = meta, group.by = "labels",
                     datatype = "spatial", coordinates = coords,
                     spatial.factors = spatial.factors)
...
cc <- computeCommunProb(cc, type = "triMean", distance.use = TRUE,
                        interaction.range = 250, scale.distance = 0.01,
                        contact.dependent = TRUE, contact.range = 100, nboot = 100)
```

### Inputs the spatial mode demands

| input | requirement |
|---|---|
| `coordinates` | 2-column data.frame/matrix; CellChat **renames** the columns to `x_cent`, `y_cent`. Anything other than exactly 2 columns → `stop()`. Rownames should be the cell/spot barcodes. |
| `spatial.factors` | data.frame with **both** `ratio` and `tol`, else `stop()`. `ratio` = multiplier converting coordinate units → µm (1.0 if you author in µm; `spot.size/spot.size.fullres` for real Visium pixel coords). `tol` = distance tolerance in µm, tutorial uses `spot.size/2`. Both are indexed per sample (`ratio[k]`), so one row per sample for multi-sample. |
| `meta$samples` | needed; auto-added as `"sample1"` with a warning if missing. Must be a factor. |
| `contact.range` **or** `contact.knn.k` | **mandatory** whenever `contact.dependent=TRUE`, which is the default — see §8.2. |
| `interaction.range` | µm, default 250. |
| `scale.distance` | must satisfy `min(d.spatial)*scale.distance >= 1` — see §8.3. |

### Result

`subsetCommunication()` returns **31 rows over our pairs**, and the spatial
constraint visibly does its job — the aggregated count matrix has **0** for
TumorA↔TumorB (381 µm apart via Stroma, beyond the 250 µm interaction range),
while all Stroma-adjacent routes survive:

```
       TumorA TumorB Stroma          spatial distance (µm):
TumorA      5      0      4                 TumorA TumorB Stroma
TumorB      0      6      4          TumorA    NaN    NaN    381
Stroma      4      4      4          TumorB    NaN    NaN    381
                                     Stroma    381    381    NaN
```

CellChat correctly reported: *"The input L-R pairs have both secreted signaling and
contact-dependent signaling. Run CellChat in a contact-dependent manner for
`Cell-Cell Contact` signaling, and in a diffusion manner based on the
`interaction.range` for other L-R pairs."* — i.e. our custom `annotation` column
drove the intended branch.

Spatial visualisation on fake ids also works:
`netVisual_aggregate(layout="spatial")`, `spatialFeaturePlot(features=c("G_1_10","G_3_30"))`.

---

## 6. What CellChat never does

It never consults `CellChatDB.human` / `CellChatDB.mouse` once `object@DB` is
replaced, never queries an external annotation service, and never validates gene
identifiers against anything except the `geneInfo` you hand it. There is no
species field on the object.

---

## 7. Recommendation for iscc

Have iscc emit four objects and write them as CSVs:

1. `interaction` — columns `interaction_name, pathway_name, ligand, receptor,
   agonist, antagonist, co_A_receptor, co_I_receptor, annotation, interaction_name_2`.
   Fill the four cofactor columns with `""`. Set `annotation` to one of the four
   legal strings.
2. `complex` — 0 rows, but write **≥ 2** `subunit_N` columns (or omit entirely).
3. `cofactor` — same, `cofactor1..N`, N ≥ 2 (or omit entirely).
4. `geneInfo` — at minimum `Symbol`, listing **every** gene named in `interaction`
   and `complex`. This is the whitelist and the single point of failure.

Then in R: `updateCellChatDB(db=interaction, gene_info=geneInfo,
other_info=list(complex=..., cofactor=...))` and `cellchat@DB <- db`.

A wrapper should assert `setequal(extractGene(db), <expected gene set>)`
immediately after building the DB. That one assertion catches the entire class of
silent-drop failures.

---

## 8. Gotchas that would cost hours

**8.1 A single-column `complex` or `cofactor` table crashes with a nonsense error.**
`extractGene()` does `complex_input[match(...), ]` without `drop=FALSE`. With one
column R drops the data.frame to a vector, and the next `dplyr::select()` dies with
`no applicable method for 'select' applied to an object of class "character"` —
which points nowhere near the real cause. Measured:

| complex | cofactor | result |
|---|---|---|
| `data.frame()` | `data.frame()` | OK |
| 0×2 | `data.frame()` | OK |
| 0×2 | 0×2 | OK |
| 0×5 | 0×16 (CellChat-native shape) | OK |
| `data.frame()` | **0×1** | **FAIL** |
| **0×1** | `data.frame()` | **FAIL** |

So: 0 columns or ≥2 columns. Never 1. This bites precisely when you write a
"minimal" cofactor CSV.

**8.2 `computeCommunProb` in spatial mode errors out on its own defaults.**
`contact.dependent = TRUE` is the default but `contact.range` and `contact.knn.k`
both default to `NULL`, and `computeRegionDistance` then does
`stop("Please check the documentation of computeCommunProb and provide the value of
either contact.range or contact.knn.k")`. Verified: the plain
`computeCommunProb(cc)` call fails on spatial data. Always pass `contact.range`
(µm, e.g. one spot pitch) or `contact.knn.k`.

**8.3 `scale.distance` is coupled to your coordinate units.**
CellChat requires `min(scaled off-diagonal distance) >= 1` and otherwise
`stop()`s with *"Please increase the value of `scale.distance` and use a value that
is slightly smaller than N"*. With µm coordinates and inter-domain distances of a
few hundred µm, the default `0.01` is fine, but the safe margin is thin — if iscc
emits coordinates in deme units or mm, this must be recomputed. The error message
does tell you the right value, so it fails loudly rather than silently.

**8.4 `netVisual(..., layout="spatial")` is broken in 2.2.0.9001.**
Genuine upstream bug, unrelated to custom DBs: `netVisual()`'s spatial branch
(`R/visualization.R` lines ~321–369) passes `idents.use = idents.use`, but
`idents.use` is not one of `netVisual()`'s formals. Result:
`Error in netVisual(...): object 'idents.use' not found`. You cannot work around it
by supplying `idents.use` yourself — it would then be matched twice through `...`.
**Use `netVisual_aggregate(object, signaling=..., layout="spatial")` instead**;
that function does declare `idents.use` and works fine.

**8.5 `checkGeneSymbol()` warns via `cat()`, not `warning()`/`stop()`.**
Its output ("Issue identified!! Please check the official Gene Symbol of the
following genes:") goes to stdout, so it is easy to lose in a log and impossible to
trap with `tryCatch`/`withCallingHandlers`. Its return value is discarded by every
caller. See §4.1.

**8.6 `updateCellChatDB` uppercases auto-generated `interaction_name`s.**
If you let it build them, `G_1_10` + `G_3_30` → `G_1_10_G_3_30` (no visible change
for our ids, but it would mangle mixed-case identifiers). Supplying
`interaction_name` explicitly avoids any surprise.

**8.7 Background communication is easy to manufacture.**
`computeCommunProb` uses `dataLR^n/(Kh^n + dataLR^n)` on *group-averaged* expression,
so any gene with non-zero baseline in every group yields a non-zero probability for
every group pair, and permutation p-values are near 0 because the *pattern* is real.
In our RNA run all 39 source/target/pair combinations with non-zero expression
survived `filterCommunication(min.cells=10)`; the designed edges were distinguished
only by having the highest `prob` (~0.58 vs ~0.40 background). If iscc's emitted
counts have a non-zero floor for L-R genes, expect a dense network. Rank by `prob`,
and consider `population.size`, `thresh`, or a sparser expression floor.

---

## Files

All under
`/private/tmp/claude-502/-Users-pedroferreira-projects-cce-scPhyTr/bdf4fe7b-35a3-4399-88ec-04a4b8ed0871/scratchpad/`

| file | purpose |
|---|---|
| `build_db.R` | builds the synthetic DB; tests acceptance, the human-geneInfo trap, `extractGene` round trip |
| `pipeline_rna.R` | full RNA pipeline end to end |
| `pipeline_spatial.R` | full spatial pipeline end to end |
| `final_checks.R` | CSV round trip, silent-drop failure mode, non-empty complex, all plotting functions |
| `probe_empty.R` | the 0×1 complex/cofactor matrix of §8.1 |
| `inspect_db.R` | dump of `CellChatDB.human`'s schema |
| `iscc_interaction_input.csv`, `iscc_geneInfo_input.csv`, `iscc_complex_input.csv`, `iscc_cofactor_input.csv` | the emitted DB in tutorial CSV form |
| `db_iscc.rds`, `cc_rna.rds`, `cc_spatial.rds`, `synth_expr.rds` | saved objects |
| `rna_bubble.pdf`, `spatial_plots.pdf` | rendered plots over the fake ids |
| `CellChat-main/` | unpacked CellChat source (for reading) |

The iscc repo was not touched.
