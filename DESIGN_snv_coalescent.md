# DESIGN — Coalescent SNV overlay: full, lineage-faithful per-cell genomes at cm-scale

Status: proposed (2026-08-05). Supersedes the per-cell-independent passenger reconstruction in
`GenotypeTumor._reconstruct_passengers`. Owner decision: **route 2** — one cm-scale ground truth, full
genomes for every sampled cell.

## 1. Why

The simulator's thesis is *one simulation → all ground truth*: a user grows a single realistic tumour
and every modality (scDNA, bulk DNA, scRNA, Visium, imaging, reads) is a view of the **same** cells. For
DNA that means the non-negotiable requirement:

> **Every sampled cell must carry a complete genome — all SNVs + allele-specific copy number — and it
> must be the SAME genome the RNA assays see.** Mutation calling scores against it; DNA↔RNA clone
> matching (clonealign/cardelino-style) only works if a cell's genotype is one shared object, not
> re-drawn per modality.

Today this holds for the **clone-defining** layer (drivers, CNAs, WGD) — which is exact — but not for the
**neutral passenger** layer. At cm-scale we run with `coarsen_passengers=True`: a pure-passenger division
is folded into a per-clone counter `_pass_load[gid]` instead of spawning a genotype (that is what keeps
`#genotypes ≈ #clones` instead of `#cells`). `_reconstruct_passengers` then re-emits each cell's
passengers as an **independent** `Poisson(load/count)` draw over neutral sites, at a placeholder
`VAF = 1/cn`. Consequences:

- **No shared clades.** Two cells in the same lineage share no passenger SNVs → the passenger genealogy
  (the bulk of a real tumour's mutational signal) is destroyed. Measured: cm-scale cophenetic
  correlation (single-cell distance vs true lineage) is carried almost entirely by the sparse driver
  SNVs; the passenger layer is noise.
- **Multiplicity is fiction.** `1/cn` ignores which homolog the SNV sits on and how amplifications/WGD
  changed its copy count — so VAF, B-allele frequency, and mutation timing are all wrong.

Turning coarsening **off** fixes fidelity but caps scale (`#genotypes → #cells`), so it cannot reach the
grid-170 tissue the spatial modalities need. We keep coarsening on and **reconstruct faithful genomes for
the sampled biopsy at materialisation** — which is cheap because we only ever materialise `max_cells`
(~50k), never the whole population.

## 2. Precedent (this is an extension of accepted methods, not a gamble)

- **CellCoal** (Posada, *MBE* 2020): generates a **coalescent genealogy for a sample of somatic cells
  from a growing population, drops mutations along it, and emits single-cell genotypes** (infinite/finite
  sites, deletion, copy-neutral LOH, cancer signatures) plus reads with ADO/amplification/sequencing
  error and doublets. This is exactly the "coalescent → per-cell SNV genome → scDNA noise" pipeline. Its
  LOH/deletion support shows allele-aware SNV-on-a-genealogy is standard.
- **Multispecies coalescent** (*BEAST/StarBEAST): gene lineages coalescing *within* branches of a fixed
  species tree, with per-branch population sizes. Our construction is identical with **clones = species,
  the forward clone tree = species tree, cells = gene lineages, and `traces` = the population sizes.**
- **MutationTimeR** (Gerstung et al.): the SNV-multiplicity-under-copy-number relationship we invert here
  to *place* mutations (they *infer* timing from multiplicity; we *set* multiplicity from timing).

CellCoal's gap — which is our contribution — is that its genealogy is a **neutral** coalescent from a
demographic model with a **diploid+LOH+deletion** CN model. We condition the coalescent on iscc's
**selection-driven, spatially-structured forward clone tree** and evolve SNV multiplicity through
**arbitrary allele-specific amplifications + WGD**, so every modality of one spatial sim shares the result.

## 3. What the engine already gives us (build on these, do not recompute)

| object | what it is | use |
|---|---|---|
| `genotypes_parents[gid] → parent` , `founder_id` | the exact **clone tree** | coalescent backbone |
| `genotype.ord` | creation ordinal (birth order) | orders clone births; birth *time* from first trace appearance |
| `self.traces` (`genotypes_counts` per snapshot) + `trace_times` | per-clone **size trajectory** `N_g(t)` | coalescent rates (variable-size) |
| `_pass_load[gid]` | **realized** count of passenger SNVs folded into clone g | total neutral mutations to place on g's within-clone genealogy → ties reconstruction to *this run* |
| `genotype.genome[seg][hap]` = list of per-copy bitsets; `genome_summary['seg_cns']` | exact **allele-specific CN** per clone | homolog choice + multiplicity trajectory |
| `_neutral_gene_ids`, `_gene_segment` | neutral sites + gene→segment map | infinite-sites site pool |
| `_materialize_plan` (max_cells / region / depth_frac) | the sampled cells per (deme, clone) | the coalescent's leaf set |

The clone-tree edges already carry the exact clonal mutations (drivers/CNAs/WGD, per-copy). The overlay
only adds the **neutral** layer, consistent with that backbone.

## 4. Algorithm

Run once inside `make_cell_data`, after the sampled cells and their per-clone assignment are known, before
the assay layers read `cell_snv`/`cell_cnv`. Dedicated seeded RNG (`seed + <const>`) → fully reproducible.

### 4.1 Backbone genealogy — multispecies coalescent on the clone tree

Leaves = the materialised (sampled) cells, each labelled with its clone `g` and deme. Build the cell
genealogy bottom-up over the clone tree (post-order):

1. For clone `g`, the lineages entering its edge from below are the **sampled cells of `g`** plus **one
   lineage per child clone** (each child was founded by a single cell of `g`, so its whole subtree hangs
   off one lineage that entered `g` at the child's birth time).
2. Coalesce those lineages backward through `g`'s size trajectory `N_g(t)` (pairwise rate
   `C(k,2)/N_g(t)`, standard variable-size coalescent), from the present (or child-birth times, added as
   the lineage enters) back to `g`'s birth `t_g`.
3. Because clone `g` descends from a **single founding cell**, force any lineages still un-coalesced at
   `t_g` to coalesce there; the resulting single lineage enters the parent clone as one of its cells.
4. Recurse to `founder_id`.

Result: a binary genealogy of the sampled cells whose topology respects the clone tree and whose branch
lengths (in cell-generations) come from the real per-clone sizes. Normal (non-cancer) cells are attached
at their clone as usual and carry no somatic passengers.

*v1 simplification (optional):* skip the within-clone coalescent and treat each clone's passengers as a
star at the clone's founding (clade-clonal) + a per-cell private tail. Cheaper, gives correct clades but a
less realistic within-clone site-frequency spectrum. The full coalescent (above) is the target.

### 4.2 Dropping neutral SNVs

On each branch of the genealogy place neutral mutations. Two calibrations (pick per §7):

- **(a) realized:** distribute clone `g`'s `_pass_load[g]` mutations across the branch-length *within
  `g`'s within-clone genealogy* ∝ branch length (multinomial). Faithful to *this run's* realized burden.
- **(b) rate:** `Poisson(μ_neutral · ℓ_branch)` with `μ_neutral` from the config
  (`mutation_rate × neutral_fraction`). Simpler; ignores the realized `_pass_load`.

Each mutation picks a **neutral site** under infinite-sites (unused on that lineage; see §4.4). A mutation
on a branch is inherited by exactly the sampled cells below it → shared clades emerge for free, and the
sample's site-frequency spectrum (singletons → clonal) is coalescent-correct.

### 4.3 Allele placement + multiplicity through CN events

Each neutral SNV, at birth in clone `X` on segment `s`:

- **homolog** `h ∈ {p, m}` chosen ∝ `cn_h(X)` (a copy is equally likely to be the mutated one); initial
  **multiplicity 1** at total `seg_cn_X(s)`. If `cn_h = 0`, pick the other; if both 0 the site is
  unavailable.

Propagate multiplicity down each descendant cell-lineage as it threads through clone transitions
(`X → child`), applying that transition's CN event on `(s, h)` — the marginal of the engine's exact
per-copy operations:

- **WGD:** `mult ×= 2`, `cn_h ×= 2` (every copy duplicated — deterministic).
- **Amplification** `cn_h → cn_h + 1` (a uniform existing copy duplicated): `mult += 1` with probability
  `mult / cn_h`.
- **Deletion** `cn_h → cn_h − 1` (a uniform copy removed): `mult −= 1` with probability `mult / cn_h`;
  the SNV is **lost on that lineage** if `mult` hits 0 (LOH).

Because every clone stores exact `seg_cns`, the CN state along any lineage is known — no extra
bookkeeping. Observed value written to the genome: **VAF (in a pure cell) = mult / seg_cn** at that cell's
clone. Bulk VAF and RNA BAF then aggregate the *same* per-cell multiplicities → automatically consistent.

### 4.4 Infinite sites, allele resolution, homoplasy

Infinite-sites is enforced by the engine **per copy** (`Cell._available_sites`: "unmutated ... infinite
sites per allele"), so a mutation at position *g* on homolog-p-copy-1, another at *g* on
homolog-p-copy-2, and another at *g* on homolog *m* are three **distinct** slots. The overlay MUST mirror
this: a passenger is a `(segment, homolog, copy, position)` slot, drawn unused **on its copy**. At the
truth level there is therefore **no saturation and no homoplasy** — every mutation has a unique physical
slot — and driver↔passenger consistency (§6) holds regardless of `segment_size`.

Saturation is a property of the **encoding/observation**, not the truth:
- Storing SNVs as a cells × *position* matrix (today's `snv_mat[i, p]`) collapses all copies of a
  position into one column — that is where distinct mutations merge into apparent homoplasy. Empirically
  the current per-position matrix is already ~93% saturated at cm-scale (**559/600** informative sites),
  which is what caps tree resolution today. The fix is **allele-resolved storage** (§4.5, §7.4), not a
  bigger genome.
- **p vs m at a shared position is observable** (haplotype phasing / BAF, which the allele-resolved
  scDNA/RNA readouts already expose) → allele-aware tools see distinct sites, no homoplasy.
- **Same-homolog, different-copy at a shared position is NOT phaseable** from short reads → it merges into
  one higher-multiplicity variant. That is *real* sequencing homoplasy the sim should reproduce while
  keeping the underlying truth clean — a feature, not a bug.

`segment_size` therefore drops from a correctness gate to a **secondary headroom knob** (the per-copy
budget); allele-resolution is the primary lever. Still log the per-copy site budget vs. tree depth.

### 4.5 Output — one shared genome

Write the reconstructed neutral SNVs into the same `cell_snv` (allele-resolved where the frame supports
it) and reconcile `cell_cnv` / `cell_rna_baf` from the shared multiplicities. **All** assays (scDNA, bulk,
scRNA allelic expression, reads) read this one object → DNA↔RNA matching holds by construction.

## 5. Integration & invariants

- **Hook:** replace the body of `_reconstruct_passengers` (called from `make_cell_data`, line ~1855);
  same call site, same "modifies `snv_mat` in place" contract, extended to set multiplicity/allele.
- **Sampling-agnostic:** operates on whatever `_materialize_plan` selected (full, `max_cells` subsample,
  `region`/`depth_frac` biopsy, met compartment). The coalescent leaf set = the materialised cells.
- **Byte-identical fallbacks (protect existing tests):** no-op when `coarsen_passengers` is off, when
  `_pass_load` is empty, or when there are no neutral sites — so every current small-tumour run/test is
  unchanged. Gated behind the same `coarsen_passengers` path it replaces.
- **Determinism:** dedicated seeded RNG; same tumour + same sample → same genomes.
- **Met (R9):** use per-compartment trace counts for `N_g(t)`; migrations are already clone-tree edges.

## 6. Faithfulness — what is exact vs. approximated

- **Exact:** clone tree, clonal driver/CNA/WGD mutations, per-clone allele-specific CN, and (calibration
  a) the realized per-clone passenger *count*.
- **Statistically faithful (marginal of the exact per-copy process):** SNV multiplicity trajectories
  (§4.3) and the sample's site-frequency spectrum / clade structure (§4.1–4.2).
- **Approximated:** the *within-clone* coalescent is reconstructed from clone sizes, not the true (thrown-
  away) within-clone genealogy; so within-clone timing is a coalescent draw, not the realized history.
  This is the price of coarsening and is exactly what CellCoal-style methods accept.

**Invariant — driver↔passenger consistency.** Drivers and passengers are both mutations on the ONE cell
lineage tree, so a passenger-reconstructed tree must be a *refinement* of the driver/clone tree, never a
contradiction. The design guarantees this: (i) the coalescent is nested in the clone tree (§4.1), and
(ii) every clone is founded by a single cell (a size-1 bottleneck at each new genotype), so each clone is
monophyletic in the true cell tree → **no incomplete lineage sorting**. Any homoplasy a *tool* sees is an
artifact of the assay's allele resolution (§4.4), not of the ground truth. The naïve per-cell-independent
reconstruction violates this (passenger tree ≈ noise, uncorrelated with the clone tree) — the core reason
to replace it.

## 7. Open decisions (for review before coding)

1. **Calibration (a) realized `_pass_load` vs (b) rate `μ_neutral`.** (a) ties to the run and needs no new
   parameter; recommend (a), with (b) as a fallback when `_pass_load` is unavailable.
2. **Full within-clone coalescent (§4.1) vs v1 star-at-founding.** Recommend shipping v1 to unblock DNA
   notebooks, then the coalescent — both share §4.3/§4.5.
3. **`segment_size` for the phylo regime — RESOLVED: secondary.** Now a headroom knob (§4.4), not a
   correctness gate. Keep the default (600) unless the per-copy budget logs saturation; a config preset
   can raise it for very deep trees.
4. **Allele-resolved SNV storage — RESOLVED (build into v1):** the reconstruction's PRIMARY output is
   allele-resolved (per **homolog** at minimum; the genome's `genome[seg][hap]` is already per-copy),
   carrying multiplicity — this is what buys the per-copy infinite-sites budget and keeps the ground-truth
   tree homoplasy-free. The existing position-collapsed `cell_snv` (VAF = total_mult / seg_cn) is
   **derived** from it for back-compat, so current assay readers/tests are unchanged in shape while new
   allele-aware readers can consume the resolved form.

## 8. Validation plan

- **Driver↔passenger consistency (HEADLINE, §6 invariant):** a tree reconstructed from the reconstructed
  passenger SNVs must recover **every split of the true clone tree** (`to_lineage_tree`) — Robinson–Foulds
  ≈ 0 up to within-clone refinement. This certifies the overlay invents no lineage structure; the current
  per-cell-independent reconstruction fails it. Primary acceptance test.
- **Cophenetic jump:** cm-scale + overlay cophenetic (single-cell distance vs `to_lineage_tree`) should
  rise from ~0.5 toward the coarsen-off exact (~0.6+), and keep rising with allele resolution.
- **Site-frequency spectrum:** the reconstructed SFS should match a coalescent expectation (∝ 1/i tail)
  and the coarsen-off SFS at matched scale — not the flat/independent shape today.
- **Multiplicity/timing:** SNVs pre-WGD show multiplicity 2 (high VAF) vs post-WGD 1 — recover the
  MutationTimeR early/late split on a known WGD tumour.
- **DNA↔RNA consistency:** a cell's SNV set is identical in `cell_snv` and in what scRNA allelic
  expression reports; clone assignment from RNA recovers the DNA clone.
- **Guards:** byte-identical outputs with coarsening off; existing 664 tests stay green; perf — overlay is
  `O(#sampled cells)`, target < a few seconds for a 50k-cell biopsy.

## 9. Phased implementation

1. `v1` star-at-founding + §4.3 multiplicity + §4.5 shared genome → unblocks DNA notebooks; validate
   cophenetic + DNA/RNA consistency.
2. `v2` full within-clone multispecies coalescent (§4.1) from traces → correct SFS.
3. `segment_size` lever + a DNA/phylo config preset in `realistic_regime`.
4. Regenerate the DNA notebooks (`tree_inference_dna`, `combining_scdna_scrna`, `wgd_allele_cna`) and DNA
   benchmarks (`validate_dna/snv/cna/scrna_snv/multiregion_phylo/clonealign/infercnv/numbat`) on the new
   genomes; update numbers.

## 10. Risks

- **Coalescent cost** if a biopsy has very many clones (each a small within-clone coalescent) — bounded by
  #sampled cells; profile at 50k.
- **Genome saturation** silently capping deep trees — mitigate with the `segment_size` lever + a logged
  site-budget warning.
- **Assay-reader churn** if allele-resolved SNV storage changes (decision §7.4) — keep the `cell_snv`
  contract stable; add rather than reshape where possible.
