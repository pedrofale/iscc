# DESIGN: simulation features (sampling, reads, assay batch effects)

Status: design / scoping (2026-06-25). Complements `DESIGN_inference.md` (that doc is the
*validation/estimation* roadmap; this one is the *feature-generation* roadmap). The two meet at
one point: the batch/technical parameters defined here are exactly what the Splatter-style
`estimate()` (DESIGN_inference §B / M2) fits from real data.

## Guiding principle: separate biology from technical; make technical hierarchical & instance-keyed

Every observed dataset = **biological ground truth** (tumor → sample) passed through an **assay
instance** = (protocol, batch realization keyed by a seed).

```
tumor (true per-cell state)  ──>  sample (biopsy/dissociation: biological/prep bias)
                                      └──> Assay instance = protocol × Batch(seed)  ──> observed data
```

Same tumor + two assay instances ⇒ **same biology, two technical signatures**. Because iscc
keeps the ground truth (clone, cell type, spatial coords) and labels every observation with its
`batch`, a multi-batch dataset is a *labeled* benchmark for batch correction / data integration /
replicate-concordance methods — the cell↔cell correspondence across batches is known exactly,
which real data can never provide. Support both **balanced** (one tumor split across batches) and
**confounded** (different tumors → different batches) designs, since confounding is the hard case
integration methods must handle.

## A. Sampling layer (`isccsample`)

Solid + liquid biopsy (F1) and dissociation biases (F2) are **DONE**; slice stays a stub. The CLI
keeps the `load_cell_data` / write-`cell_data` / `sample_meta.yaml` contract and adds two methods
(`--method {biopsy,dissociation}`) on top of the legacy uniform `--fraction` thinning.

- **Solid biopsy — spatial region selection** over the cell grid (`cell_crd` row/col), in
  `sample/biopsy/biopsy.py` (`Biopsy`): **DONE**
  - *needle core*: a narrow strip (perpendicular distance ≤ width/2 to a line through the tumour at
    a chosen `--angle`); *punch*: a disk (`--center` + `--radius`); *multi-region*: `--n-regions` k
    disjoint disks (centres drawn from occupied cells, kept ≥ 2·radius apart), EACH tagged with its
    own `region_i` label. This reproduces multi-region heterogeneity (you only see the clones present
    in the sampled region) — the substrate for phylogeny studies.
- **Liquid biopsy (blood / CTC)**: `--biopsy-type liquid` — `--n-liquid` circulating cells,
  restricted to cancer clones and weighted by a per-cell dispersal signal from `cell_evo`
  (`n_mut_disp`, else `dispersal_rate`; degrades to uniform if absent); region label `blood`. **DONE**
- **Dissociation biases** (sample-prep, *biological* not sequencing-technical), in
  `sample/dissociation/dissociation.py` (`Dissociation`): **DONE**
  - cell-type-dependent recovery probability per biological type (cancer / immune / stromal /
    epithelial, derived from the genotype id) → **composition bias** (`DEFAULT_RECOVERY` under-recovers
    immune/stromal; override with `--recovery 'immune=0.4,...'`). Optional dissociation-stress
    expression signature (`--stress-strength`, multiplicative bump on a small stress-gene set of
    recovered cells' `cell_exp`). Droplet doublets / ambient release are NOT here — they live in the
    F3 assay.
- **Slicing** (for spatial): select the 2D section presented to the spatial assay. iscc is 2D so
  the section = grid or sub-grid; a true 3D tumor + arbitrary cut plane is future work. *(stub)*
- Output: subset of cells preserving per-cell ground truth + `cell_data/cell_region.csv` (per-cell
  `region` label, biopsy only) + geometry / recovery probs / realized composition shift recorded in
  `sample_meta.yaml`. The extra `cell_region.csv` is ignored by `isccdata`'s fixed-key loader, so the
  region labels flow through to the assay's `cell_data` for downstream multi-region analyses.
  Validation: `validation/validate_sampling.py` → `manuscript/figures/validation_sampling.png`.
  Tests: `tests/test_sample_biopsy.py`, `tests/test_sample_dissociation.py`.

## B. Assay batch-effect model (the core new piece)

### B.1 Hierarchy
`hyper-parameters` (protocol-typical magnitudes) → `per-batch realization` (one seed = one
reaction/plate/run) → `per-cell / per-gene observation`. The batch effect is a **property of the
assay instance**, applied consistently across all cells of that instance, and reproducible from
its seed.

### B.2 scRNA model
For cell *c*, gene *g*, batch *b*, with biological expression `λ_cg` (from the tumor's `cell_exp`
ground truth):

- **per-gene batch factor** `β_gb ~ LogNormal(0, σ_batch²)` — *shared across all cells of batch b*,
  varies by gene. `σ_batch` is the batch-strength hyper-parameter (Splatter's `batch.facScale`).
  This is the canonical batch effect: a per-gene fold-change common to a run.
- **per-batch depth** `μ_lib,b`; **per-cell library factor** `ℓ_c ~ LogNormal(μ_lib,b, σ_lib²)`.
- **per-batch dispersion** `φ_b`, **dropout**, **ambient profile**, **doublet rate**.
- Counts: `y_cg ~ NB(ℓ_c · β_gb · λ_cg, φ_b)`, then add ambient counts (from a batch ambient
  profile) and doublets (merge two cells' profiles at the batch doublet rate).
- **The count emission is a pluggable step** — keep everything above (biology → library → batch)
  shared and swap only the final draw. **Default `count_model="nb"`** (field standard: Splatter/
  DESeq2/edgeR, composes cleanly with the multiplicative factors, what `estimate()` fits). Optional
  **`"dm"` (Dirichlet-Multinomial)**: total `N_c = ℓ_c`, proportions `p_c ~ Dirichlet(β_gb·λ_cg·κ_b)`
  — the compositional model (fixed library total, gene–gene competition for reads) for later
  robustness studies. (A scDesign-style copula over NB marginals is a third future option for
  realistic gene–gene correlation.)

**Protocol presets** set which components dominate:
- *10x (droplet, UMI, 3′)*: significant ambient RNA + doublets; moderate depth; UMI suppresses
  amplification noise.
- *Smart-seq3 (plate, full-length, 5′ UMIs)*: higher sensitivity and lower dropout than 10x;
  the 5′ UMIs suppress amplification noise on UMI-counted molecules (unlike Smart-seq2), with
  full-length non-UMI coverage available for allele/isoform-level signal; extra nesting level —
  **plate** (= batch) **+ well** (per-cell position) effects.

### B.3 Running batches
Two `Assay` instances with identical hyper-parameters but different seeds → two batches; write
each as its own AnnData with `batch` in `.obs`, concatenate for integration benchmarking. The
biological signal (`λ`) is shared; only the `β`, depth, ambient draws differ.

## C. Read simulation (read-count matrices → FASTQ / BAM)

Follow the standard 3-stage DNA-seq pipeline (as SISTEM, Weiner 2025): **per-cell reference →
coverage model → third-party read simulator**. Only the coverage model is bespoke; do NOT
reimplement the sequencer error model — delegate stage 3 to a validated tool (**ART**, DWGSIM,
pIRS, InSilicoSeq). Phased:

- **C1 — allele-specific *count / coverage* realism** (highest value; needs NO nucleotide sequence):
  per-locus coverage scaled by **copy number** (amplified region → proportionally more reads — how
  CNA callers work), depth dispersion, GC/mappability, + binomial VAF sampling with an error rate
  → **read-count / coverage / VAF matrices**. This is what most CNA/SNV callers actually consume
  (SISTEM emits these *separately* from raw reads), and it works on the abstract bitset genome
  today. **Breadth-aware** (WGS/WES/panel, §D); for single-cell, apply ADO + amplification bias
  first. Extends `data/dna.py` (already has a `data_mode='reads'` stub).
  - **Depth model is pluggable; default to Dirichlet-Multinomial (compositional) for DNA** (the
    reverse of scRNA's NB default — the physics differ). DNA coverage is a fixed read budget `N`
    partitioned across segments with proportions `p_seg ∝ CN_seg · length_seg · GC/mappability_seg`;
    counts `~ Multinomial(N, p)` with `p ~ Dirichlet(κ · p̄)` for over-Poisson coverage lumpiness
    (`κ` = concentration). DM is the right default because (i) copy number is the natural proportion
    driver, (ii) with coarse segments + **large segmental CN**, the compositional coupling
    (an amplicon stealing reads at fixed total depth) is non-negligible, and (iii) it correctly
    makes depth **relative** not absolute (a WGD/high-ploidy genome is not uniformly deeper — what
    CNA callers normalise for); independent NB-per-segment misses (ii)–(iii). Offer NB-per-segment
    as the simpler alternative. VAF is a separate binomial layer: alt `~ Binomial(depth_seg,
    true_alt_fraction)` + error. (`κ` is M4's estimate() target alongside depth/GC/error.)
  - **Sampling distribution depends on the data type** — the dominant noise is the *amplification*
    regime (un-amplified bulk ≈ Poisson; whole-cell-amplified single-cell = wildly lumpy), and
    capture (WES/panel) adds a *systematic* per-target mean, not stochastic overdispersion. Two
    layers (depth + allele) choose separately. The single unifying knob is **κ = the amplification
    regime** (large κ ≈ multinomial/Poisson for bulk; small κ = lumpy for single-cell):

    | Data type | Depth / coverage | Allele / VAF |
    |---|---|---|
    | single-cell WGS | **DM, low κ** (amplification overdispersion + compositional) | **Beta-Binomial + ADO** |
    | single-cell panel (Tapestri) | DM, low κ, few loci | Beta-Binomial + **heavy ADO** |
    | single-cell WES | DM, low κ + capture | Beta-Binomial + ADO |
    | bulk WGS | NB-per-bin (field convention; mild overdisp.) **or DM high κ** | Binomial |
    | bulk WES | NB-per-target + **systematic per-target capture mean** | Binomial |
    | bulk panel | very deep; per-amplicon efficiency mean (depth near-trivial) | Binomial / Beta-Binomial (deep, low-freq) |

    Notes: (a) **ADO is a separate Bernoulli layer** (one allele lost at a locus), the dominant
    single-cell allele artifact — model explicitly on the allele layer, not via the depth
    distribution. (b) Offer **NB-per-segment for bulk** to match field tools (HMMcopy/CNVkit/
    Control-FREEC). (c) WES/panel capture bias is a per-target/per-amplicon *mean* multiplier,
    orthogonal to the depth distribution. (d) κ, per-target/amplicon efficiencies, and the ADO rate
    are all M4 `estimate()` targets.
- **C2 — raw reads (FASTQ)**: needs a nucleotide reference. **Follow SISTEM (Weiner & Bansal 2025)
  exactly** — verified from their docs/source: SISTEM *"constructs a full reference sequence for each
  mutated cell and computes a coverage distribution before calling a third-party short-read
  simulator"*, the reference is **user-supplied** (*"pass one or multiple reference genomes"*), and
  the external tools are **DWGSIM (≥0.1.13)** + **samtools** (read-the-docs install page). So the
  iscc design is:
  - **Pluggable reference, not hardwired** (SISTEM's model). A `Reference` interface with two
    backends: *(A) synthetic* — generate a random per-segment FASTA once, the default that works on
    today's abstract genome (lacks real sequence context → benchmark transfer is open,
    RESEARCH_QUESTIONS R5); *(B) real-genome* — supply a real FASTA, anchored via the M3b real-genome
    mode (segments = chromosome arms → hg38 coordinates), the drop-in once that mode is mature.
    Same interface; the reference is swappable input exactly as in SISTEM.
  - **Per-cell reference construction**: apply each cell's CNAs (duplicate / delete the segment
    sequence) and SNVs (substitute bases) to the chosen reference → per-cell FASTA.
  - **Coverage distribution**: reuse the C1 copy-number-scaled coverage model already in
    `data/dna.py` (DM depth, GC/mappability, breadth) — this is the bespoke part; do NOT reimplement
    the sequencer error model.
  - **Read simulator = pluggable adapter, default DWGSIM** (matches SISTEM), **ART as alternative**.
    A thin wrapper writes the per-cell FASTA + per-segment coverage, shells out to the binary,
    collects FASTQ. The external binary is an optional runtime dependency.
- **C3 — BAM**: align the FASTQ back to the chosen reference (bwa) + **samtools** sort/index (SISTEM's
  toolchain) for tools that consume BAMs.

Gate C2/C3 behind C1; C1 alone serves the many methods that take count/coverage matrices, not reads.
**CI stays light**: DWGSIM/ART/samtools are optional binaries — tests skip the shell-out gracefully
when absent and instead assert the *bespoke* layers (per-cell FASTA reflects CNAs/SNVs; coverage
allocation ∝ copy number; adapter command is built correctly, mocked).

**Read emission is per-modality** — the count matrix is the universal interface (what most methods
consume *and* the input these read simulators take), but the read tool + reference differ:
- **DNA** (genome reference): ART / DWGSIM / pIRS / InSilicoSeq.
- **scRNA, droplet/10x** (count matrix → reads, barcodes+UMIs): **scReadSim** (Yan 2023, learns from
  real data; current best) ≫ minnow (Sarkar 2019, count→reads but poor coverage realism).
- **scRNA full-length / Smart-seq3** (5′ UMI + full-length): less standardised — polyester-style bulk
  RNA read sim + a UMI layer. **Known gap**, not a blocker.
- **spatial (Visium)**: count-level (F6 spot×gene matrix) suffices for most spatial-method
  benchmarking, but reads are supported (F6 reads component) — a 10x-style sim with **spatial**
  barcodes, reusing the F7b scRNA read machinery (`write_scrna_fastq`) with spot-level
  clone-mixture RNA-VAF.

**Keep the read-realism backends separate; unify at the count matrix + a shared variant layer.**
DNA (DWGSIM/ART) and scRNA (scReadSim) place reads differently — DNA from a per-cell **nucleotide
FASTA** (apply CNAs/SNVs to a reference), scRNA from a **count matrix** mapped onto read profiles —
so the `Reference`/per-cell-FASTA path stays DNA-only and they live in separate `reads/{dna,rna}.py`
adapters. **Asymmetry:** scReadSim needs a **real** reference/template BAM even though iscc's counts
are synthetic, so the *synthetic-reference* backend is DNA-only (scRNA = synthetic counts → real read
templates; R5 lands differently per modality). They unify at:
- (i) the upstream **count/coverage matrix** — the universal interface both tools ingest;
- (ii) a downstream `ReadEmitter` protocol + `run_binary()` util in `reads/base.py`;
- (iii) **`reads/variants.py` — a shared variant-injection seam** keyed on `(total, alt_fraction)`:
  given the reads/UMIs at a locus and a per-cell alt-fraction, stochastically assign alt vs ref base
  (+ error) **preserving the total**. DNA: `total`=coverage(∝CN), `alt_fraction`=DNA-VAF (`cell_snv`).
  RNA: `total`=UMI count from the matrix, `alt_fraction`=observed RNA-VAF (next paragraph).

**Mutation-aware scRNA reads (the scDNA-vs-scRNA calling benchmark).** Real scRNA reads carry the
cell's expressed somatic mutations; modelling this lets iscc show that **SNV calling from scRNA is
intrinsically hard**. Design constraints (do not violate):
- **UMI totals are conserved.** scRNA reads are *driven by* the F3 expression count matrix — we do
  NOT invent coverage. For each cell-gene the UMI total is fixed by the matrix; the read layer only
  adds **sequence content**, partitioning those UMIs into alt-carrying vs ref-carrying. Total reads
  out == count matrix in, always (scReadSim already matches a given matrix, so it fits this exactly).
- **One abstraction parameter.** A single scalar `obs_fidelity ∈ [0,1]` folds monoallelic
  expression, transcriptional bursting, RNA editing and RT error into one knob mapping
  **true VAF → observed VAF** (`v_obs = distort(v_true, obs_fidelity)`); not a mechanistic ASE model.
  `v_true` is the allele-expression-weighted RNA-VAF (`(m·e)/(m·e+w)` from `get_exp`'s per-allele
  effects; engine gap below) — or DNA-VAF as the simple first cut. Then `alt ~ Binomial(n_umi, v_obs)`.
- **Deliver count-level first.** The observed alt/ref UMI matrix already demonstrates the effect
  (observed VAF distorted/dropped vs true DNA-VAF); actual reads (scReadSim) are the realism layer on
  top, totals identical. Output both the **observed-VAF matrix** and the **true DNA-VAF** for the
  benchmark figure.
- **Engine gap:** expose per-cell per-locus `cell_rna_vaf` (allele-expression-weighted; ingredients
  already in `CancerCell.get_exp`). Small addition to `make_cell_data`, not a new subsystem; the
  `obs_fidelity` distortion lives in the read/assay layer, not the engine.

## D. Per-modality technical / batch considerations

| Modality | Batch unit | Dominant technical factors | iscc model |
|---|---|---|---|
| **scRNA 10x** | reaction / lane | ambient RNA, doublets, depth, per-gene batch factor, capture efficiency | §B.2 |
| **scRNA Smart-seq3** | plate (+ well) | plate + well effects; 5′ UMIs (low amplification noise) + full-length coverage; higher sensitivity / lower dropout than 10x | §B.2 + plate/well nesting |
| **bulk DNA** (WGS / WES / panel) | library / run (+ capture kit) | coverage depth, **GC-bias curve**, mappability, PCR duplication, sequencing error, FFPE C>T deamination; breadth-dependent capture bias (see below) | per-batch GC→coverage curve + depth + error; per-bin CNA log-ratio noise |
| **single-cell DNA** (WGS / WES / panel; DLP, 10x-CNV, Tapestri…) | chip / run (+ per-cell amplification) | very low coverage, **allelic dropout (ADO)**, MDA/MALBAC uneven amplification (per-cell), GC bias, doublets; same breadth axis | per-cell amplification profile nested in run-batch; ADO rate |
| **spatial (Visium)** | slide / section | per-spot capture efficiency (**spatially autocorrelated**, edge effects), spot library size, lateral mRNA diffusion/bleed, spot→cell aggregation (~1–10 cells / 55µm), ambient, per-gene batch factor, **section plane** | spatially-correlated capture field + spot aggregation + diffusion kernel + §B.2 per-gene factor |

Key modality-specific notes:
- **Capture breadth (WGS / WES / targeted panel) is orthogonal to bulk-vs-single-cell** — both
  bulk and single-cell DNA can be any of the three. Breadth sets three things: (i) the **observed
  locus set** — WGS = all genome positions; WES = an exon/gene subset; panel = a small designated
  set (e.g. the driver genes); (ii) the **depth regime** — WGS low, WES moderate on-target, panel
  very high; (iii) the **dominant capture bias** — WGS GC bias, WES per-target/probe capture
  variability, panel per-amplicon bias. In iscc this is a `breadth` + `target_genes` parameter on
  the DNA assay (reusing `dna.py`'s existing `target_genes`), applied identically to bulk and sc.
  Downstream implication: panel → deep VAF on few drivers but poor genome-wide CNA; WGS → good CNA,
  shallow VAF; WES → coding SNVs + coarse CNA.
- **Bulk/sc DNA**: the batch effect that matters for *CNA calling* is the GC/coverage bias and the
  per-bin log-ratio noise level (what panel-of-normals normalization targets); for *VAF* it is
  depth-driven binomial noise + error rate. Model batch = GC-curve + depth + error (+ ADO for sc).
- **scDNA** has a *nested* technical structure: a per-cell amplification profile (the dominant
  noise) sits inside the run-level batch — analogous to Smart-seq3's well-in-plate nesting.
- **Spatial**: the distinctive requirement is that the technical noise is **spatially correlated**
  (capture efficiency varies smoothly across the slide), and that a "batch" also includes which
  2D **section** of the tumor was placed on the slide — part technical, part biological sampling.

## E. Architecture / API

```
src/iscc/data/
  assay.py        # Assay base: protocol + batch hyper-params + seed -> Batch realization
  batch.py        # Batch: per-gene factor draw, depth, dispersion, ambient, doublets (per modality)
  rna.py          # scRNA: protocol presets 10x / smart-seq3, applies Batch
  dna.py          # bulk + single-cell DNA × {WGS, WES, panel} breadth; GC/coverage/ADO, applies Batch
  visium.py       # spatial: capture field + aggregation, applies Batch
  reads/          # C2 FASTQ -> C3 BAM. Separate read-realism ADAPTERS per modality, shared seams.
    base.py       #   shared: ReadEmitter protocol emit(matrix, reference, outdir)->FASTQ/BAM +
                  #   run_binary() util (PATH detect, skip-if-absent, FASTQ/BAM collection)
    variants.py   #   shared variant-injection seam: inject(total, alt_fraction, error) -> alt/ref
                  #     content PRESERVING total. DNA: total=coverage,alt=DNA-VAF; RNA: total=UMI,alt=v_obs
    dna.py        #   DWGSIM(default)|ART: Reference{synthetic|real} -> per-cell FASTA -> C1 coverage
                  #     -> variants.inject(DNA-VAF) -> reads -> bwa/samtools BAM   [F7 now]
    rna.py        #   scReadSim: count matrix (totals conserved) -> reads on REAL template ->
                  #     variants.inject(v_obs from cell_rna_vaf x obs_fidelity)   [later]
```
CLI: `isccdata --protocol 10x --batches 2 [--batch-seed ...]` writes one labelled AnnData per
batch. Batch hyper-parameters are exactly what `estimate()` (DESIGN_inference §B) fits from real
data — so realistic defaults can be *learned*, not guessed.

## F. Milestones (features)
- **F1 — DONE** — spatial solid biopsy (needle / punch / multi-region) + region labels, plus liquid
  biopsy. `sample/biopsy/biopsy.py` (`Biopsy`): geometries over `cell_crd` (punch disk, needle strip,
  k disjoint multi-region disks each with its own `region_i` label, liquid CTC draw biased by the
  `cell_evo` dispersal signal). CLI `isccsample --method biopsy --biopsy-type {needle,punch,
  multiregion,liquid}` (+ `--radius/--width/--angle/--n-regions/--n-liquid/--center/--grid-size`)
  writes the subset + `cell_data/cell_region.csv` (region labels flow to `isccdata`) + geometry in
  `sample_meta.yaml`. Tests `tests/test_sample_biopsy.py`; validation panels (a)/(c) in
  `validation/validate_sampling.py`.
- **F2 — DONE** — dissociation biases (composition bias + optional stress signature) + liquid biopsy
  (liquid lives in F1's `Biopsy`). `sample/dissociation/dissociation.py` (`Dissociation`):
  cell-type-dependent recovery → composition bias (`DEFAULT_RECOVERY`; `--recovery`, `--stress-strength`),
  realized composition shift in `sample_meta.yaml`. Droplet doublets / ambient release deferred to the
  F3 assay (sample-prep biology only here). Tests `tests/test_sample_dissociation.py`; validation panel
  (b) in `validation/validate_sampling.py`. Demo notebooks `sampling_biopsy.ipynb` / `dissociation.ipynb`
  pending (§G).
- **F3 — DONE** — **scRNA batch model + 10x / Smart-seq3 presets** (the user's headline ask);
  multi-batch output with ground-truth labels. `data/batch.py` (`BatchHyperParams` + `Batch`:
  per-gene factor β_gb, per-batch depth/dispersion, ambient soup, doublets, dropout; **pluggable
  count emission** `COUNT_MODELS={"nb",...}`, `"dm"` Dirichlet-Multinomial stubbed as a drop-in
  seam). `data/rna.py` rewritten: `PROTOCOL_PRESETS` (10x: ambient+doublets+dropout; smartseq3:
  deeper, low dispersion, well-in-plate nesting, dropout off), Splatter-style composition
  (β reshapes per-gene, library sets total), AnnData output with clone/coords/batch in `.obs`,
  `run_scrna_batches` (shared / split designs) + `concat_batches`. CLI: `isccdata -a scrna
  --protocol {10x,smartseq3} --batches N --design {shared,split} --count-model nb` → one labelled
  h5ad per batch + `scrna_all_batches.h5ad`. `load_cell_data` now also loads `cell_type` (clone
  ground truth). Tests: `tests/test_assay_scrna.py`. Demo: `notebooks/assay_scrna.ipynb`
  (sweeps depth/dispersion/dropout/batch-strength/#batches/protocol → counts, UMAP, library-size
  dist, batch mixing). Hyper-param names = M2 `estimate()` targets.
- **F4 — DONE** — **bulk DNA** technical realism across **WGS/WES/panel** breadth (= read-counts C1).
- **F5 — DONE** — **single-cell DNA** (ADO + per-cell amplification, nested in run-batch), same breadth.
  F4+F5 built together on a **shared coverage core** (`data/batch.py` `DNABatch` + `data/dna.py`),
  replacing the old uniform-multinomial + fpr/fnr placeholders. **Copy-number-scaled coverage** from
  the engine's `cell_cnv` (per-locus CN) / `cell_snv` (per-locus true alt fraction). **Pluggable depth
  model** `DNA_DEPTH_MODELS={"dm","nb"}`: default **Dirichlet-Multinomial** (fixed budget `N`,
  `p_seg ∝ CN·length·GC/mappability`, `p~Dir(κ·p̄)`) — `κ` = amplification regime (BULK large κ ≈
  multinomial/Poisson; SINGLE-CELL small κ = lumpy), plus independent **NB-per-bin** as the field-tool
  alternative (HMMcopy/CNVkit). **Allele layer** is data-type-dependent: BULK `Binomial(depth, true_af)`
  + error; SINGLE-CELL **Beta-Binomial + explicit ADO** (separate Bernoulli, het locus → homozygous) +
  doublets; optional FFPE C>T. **Capture breadth** (`DNA_BREADTH_PRESETS`, orthogonal to bulk/sc) sets
  observed loci (wgs=all / wes=fraction / panel=`target_genes` drivers), depth regime (wgs low → panel
  very high), and a systematic per-target/amplicon **capture-efficiency mean**. **DNA batch** (`DNABatch`,
  mirrors §B): per-batch GC→coverage curve + depth shift + per-locus error; `run_dna_batches`
  (shared/split) → multi-batch labelled output, ground truth preserved. Outputs: per-locus
  coverage/alt/VAF/**CNA log2-ratio** + true CN/alt-fraction/ADO (bulk DataFrame + AnnData; sc matrices
  + AnnData). CLI: `isccdata -a {bdna,scdna} --breadth {wgs,wes,panel} --depth-model {dm,nb} [--batches N]`.
  Tests `tests/test_assay_dna.py` (coverage∝CN, compositional read-stealing, VAF recovery, ADO allele
  loss, breadth, NB, multi-batch). Demo `notebooks/assay_dna.ipynb`. κ / capture efficiencies / ADO rate
  are named M4 `estimate()` targets.
- **F6 — DONE** — **spatial (Visium) assay** — upgraded the `visium.py` stub (grid spots → averaged
  expression → fixed multinomial) to a proper assay matching F3/F4/F5 (`Assay` + `VisiumBatch` batch
  model + `VisiumBatchHyperParams` + AnnData + `estimate_visium` + `validate_visium`). The
  **distinctive requirement is that the technical noise is SPATIALLY CORRELATED** (§D spatial row).
  Built: the capture field is a squared-exponential GP over the spot coords through a LogNormal link
  × an `edge_sigma` boundary falloff, normalized to mean 1, achieving Moran's I ≈ 0.5–0.9 (rising
  with `field_lengthscale`); the DM count seam (`batch._emit_dm`) is now implemented (Visium's
  default). Spot-barcoded reads in `reads.emit_visium_reads`; `estimate_visium` fits
  `mu_counts`/`sigma_counts` + `field_sigma` + `field_lengthscale` + the count overdispersion
  (carrying ambient/edge/diffusion + single-section `sigma_batch` `_PRIOR_ONLY`). Tests
  `tests/test_assay_visium.py` + `test_estimate_visium.py` + `test_reads_visium.py`; demo
  `notebooks/assay_spatial.ipynb`; CLI `isccdata --assay visium`. Components:
  1. **Spot layout** — Visium-like spots over the deme grid (~55µm, **1–10 cells/spot**); a
     `spot_radius`/pitch sets aggregation. Keep iscc's single 2D section (no z).
  2. **Spot→cell aggregation** — pool the `cell_exp` of cells whose `cell_crd` falls in the spot;
     surface ground truth per spot (n_cells, **dominant clone / cell-type fractions**).
  3. **Lateral mRNA diffusion/bleed** — a Gaussian kernel spreading expression to neighbouring
     spots (one `diffusion_sigma` knob); the spatial-mixing artifact spatial-deconvolution methods fight.
  4. **Spatially-correlated capture-efficiency field** — a smooth positive random field over the
     spots (low-pass-filtered noise / squared-exponential GP, `field_lengthscale` = autocorrelation
     scale) × **edge effects** (lower efficiency near the tissue boundary). This is the headline
     piece — it makes Moran's I of the capture field > 0.
  5. Reuse the §B.2 **per-gene batch factor**, **ambient**, and the pluggable **NB/DM count model**
     (`COUNT_MODELS`); **per-spot library size** (lognormal). Output **AnnData** with
     `obsm["spatial"]` = spot coords, `.obs` ground truth, `.uns["hyperparams"]`. CLI: wire
     `isccdata --assay visium`. Demo `notebooks/assay_spatial.ipynb` (§G).
  - **`VisiumBatchHyperParams`** (the estimate targets): `spot_radius`/pitch, `mu_counts`/
    `sigma_counts` (per-spot library), `field_lengthscale`, `field_sigma`, `edge_sigma`,
    `diffusion_sigma`, `sigma_batch` (per-gene), `ambient_frac`, `kappa`/`nb_dispersion`.
  - **M4 Visium estimate (`DESIGN_inference §C.2`)**: `estimate_visium()` mirroring
    `estimate()`/`estimate_dna()` (`.fitted` map, `_PRIOR_ONLY`) — fit spots-per-tissue,
    counts-per-spot, and the capture-field autocorrelation (Moran's I / variogram length-scale)
    from a real Visium AnnData. Validation `validate_visium.py`: posterior-predictive overlay of
    **Moran's I** (spatial autocorrelation) + spot-count distribution + spots-per-tissue.
  - **Visium reads (optional, reuses F7b)**: spot-barcoded 10x-style FASTQ — Visium reads are scRNA
    reads barcoded by SPOT not cell, so `observed_allele_counts` / `write_scrna_fastq` /
    `SyntheticTranscriptome` (reads/rna.py) apply unchanged on the spot×gene axis (the per-"cell"
    barcode becomes the spatial barcode). NEW piece: the spot pools several cells of possibly
    different clones, so spot-level RNA-VAF = the **expression-weighted mixture** of member cells'
    `cell_rna_vaf` (the clone-mixture ground truth spatial deconvolution must untangle). Same
    invariants as scRNA: spot UMI totals conserved, single-source error (error-free molecular split
    + per-base read error).
- **F7** — read emission. **C1 count/coverage matrices** (copy-number-scaled, breadth-aware) **done**
  (in `dna.py`). Remaining: **C2 FASTQ → C3 BAM, SISTEM-faithful** — pluggable `Reference`
  (synthetic default / real-genome drop-in), per-cell FASTA from CNAs/SNVs, C1 coverage, **DWGSIM**
  (default; ART alternative) + bwa/**samtools**, **plus the shared `reads/variants.py` seam**
  (inject alt/ref preserving total; DNA uses it with DNA-VAF). **DNA-first** (bulk + single-cell).
- **F7b ✅ DONE** — **mutation-aware scRNA reads**: `reads/rna.py` drives off the F3 count matrix
  (UMI totals conserved) → `variants.inject` with observed RNA-VAF = `cell_rna_vaf × obs_fidelity`
  (single abstraction parameter). Engine adds `cell_rna_vaf` = the EXPECTED allele *fraction*
  (allele-expression-weighted; per-gene baseline cancels — the OBSERVED scRNA-VAF additionally
  depends on per-gene expression depth, so it does NOT match DNA-VAF at neutral loci). Count-level
  observed alt/ref UMI matrix is the primary output; **actual reads** via `emit_fastq=True` — a
  self-contained 10x-style paired FASTQ (synthetic transcriptome, barcodes+UMIs, one read per UMI
  so totals are conserved, alt base carried) with no external binary; the scReadSim seam is the
  real-template option. Benchmark `validation/validate_scrna_snv.py`: even at `obs_fidelity=1`
  only ~31% of true mutations are scRNA-detectable (expression gating), falling with fidelity —
  the **scDNA-vs-scRNA SNV-calling-is-hard** result. External binaries optional; CI asserts the bespoke layers.
- **F8 ✅ DONE — microenvironment-driven expression (the integration keystone).** A cell's
  expression now depends on its **spatial microenvironment**, not only genotype/CNA/cell-type.
  Implemented in the count engine (`models/count.py`) as a **per-deme × gene modifier applied at
  materialisation** (`_o2_field` hypoxia + `_cci_field` cell-cell communication → `_microenv_deme_mod`
  → `make_cell_data`), NOT inside `get_exp` (which is per-genotype/cached). OPTIONAL via a
  `microenv_params` config block; OFF ⇒ output bit-identical, and even ON the **growth is
  byte-identical** (modifier draws from a dedicated rng) — F8 modulates the READOUT only.
  Ground truth surfaced (`self.microenv_truth` + per-cell `cell_microenv`). Tests
  `tests/test_microenvironment.py` (12); validation `validation/validate_microenvironment.py` →
  `validation_microenvironment.png`. **Full design: section H.** Makes iscc the benchmark substrate
  for **DNA+RNA+spatial integration** and gives Visium/imaging real *niche* structure.
  **Future extension (deferred):** coupling the microenvironment to FITNESS (hypoxia slowing
  division, etc.) — v1 is expression-only so growth stays invariant.

- **F9 ✅ DONE — single-cell spatial assay (imaging-based; MERFISH/Xenium-like).** `data/imaging.py`
  `scSpatial` (registered as `ASSAYS["scspatial"]`): per-cell counts at the cell's `(row, col)` (NO
  spot aggregation), over a **targeted gene panel** (the *transcriptome coverage* knob — `panel` list
  or `n_panel_genes` top-N; whole-transcriptome default), drawn from the shared `batch.Batch` NB/DM
  count machinery with imaging *data-distribution* params (`IMAGING_PRESETS` merfish/xenium: lower
  depth, near-zero dropout, no droplet doublets). Surfaces per-cell coords + clone/type ground truth;
  output AnnData (cells × panel genes), coords in `obsm["spatial"]`. Platform-agnostic (presets, not
  a hardcoded name). Tests `tests/test_assay_scspatial.py` (14). Enables validation against
  single-cell spatial + lineage datasets (**PEtracer**, Science 2025 — spatial clonal architecture +
  tree statistics now; the intrinsic-vs-extrinsic module decomposition once F8 lands).

## Validation hooks (per DESIGN_inference conventions)
- Batch model: *recover* injected batch params via `estimate()`; show two same-tumor batches share
  biology but differ technically (e.g. kBET / iLISI-style mixing only after correction).
- Benchmark utility: integrate two batches, measure recovery of the *known* clone/type labels.
- DNA/spatial: validate coverage/GC and spatial-autocorrelation distributions vs real references.

## G. Deliverables — per-module demo notebooks (required for every feature)

Every feature module ships, alongside its **code + tests**, a **dedicated demo notebook** under
`notebooks/` that (a) shows how to use the module and (b) demonstrates the **impact of its key
parameters**, starting from a *realistic upstream input*. These complement the end-to-end pipeline
notebooks (`01_pipeline_walkthrough`, `02_tumor_growth`, `03_data_overview`), which show the whole
chain; these are single-module deep-dives. (They also become the reference for "what each knob does"
in the eventual web platform — see `DESIGN_platform.md`.)

Each notebook should: take a realistic input from the previous stage (e.g. the scRNA notebook starts
from a tumour **dissociation/sample**, not a raw tumour); **sweep the module's key parameters** and
visualize their effect on the output; and surface the ground truth (clone / type / coords) so the
parameter effect is interpretable.

| Notebook | Module (milestone) | Sweeps / shows |
|---|---|---|
| `sampling_biopsy.ipynb` | biopsy/sampling (F1) | region geometry (needle/punch/multi-region) → observed clonal heterogeneity |
| `dissociation.ipynb` | dissociation (F2) | composition bias, doublet rate → cell-type proportions vs ground truth |
| `assay_scrna.ipynb` | scRNA assay (F3) | **given a dissociation:** depth, dispersion, dropout, **batch strength / #batches**, protocol (10x vs Smart-seq3) → counts, UMAP, library-size dist, batch mixing |
| `assay_dna.ipynb` | bulk + sc DNA (F4/F5) | **breadth (WGS/WES/panel)**, depth, GC bias, ADO → coverage / VAF / CNA profiles |
| `assay_spatial.ipynb` | Visium (F6) | capture field, diffusion, spot size, section → spot counts + spatial autocorrelation |
| `treatment.ipynb` | treatment (shipped) | chemo/targeted/immuno, dose schedule, adaptive vs continuous → burden over time (backfill) |
| `reads.ipynb` | read emission (F7) | coverage, error → FASTQ / VAF tables |
| `microenvironment_expression.ipynb` | microenvironment expression (F8) | O₂ supply/consumption, CCI strength → hypoxia-core gradient, clone-boundary crosstalk, spatial niches vs ground truth |

## H. Microenvironment-driven expression (F8 — the integration keystone)

**Problem.** iscc expression is currently `cell.get_exp = baseline(cell_type) × dosage(CNA) ×
mut_effects(SNV)` — a function of *genotype and cell type only*. So a cell's transcriptome is
independent of where it sits in the tumour. Real tumour expression is strongly shaped by the
**microenvironment**: hypoxia in the poorly-perfused core, paracrine signalling at cell-type/clone
boundaries, immune crosstalk. Without this, iscc's Visium reproduces clonal/CNA structure but **no
niche structure**, and cannot serve as ground truth for spatial-niche / cell–cell-communication
methods — the exact tools the integration story needs to run on.

**Two coupled mechanisms, both at deme resolution** (lattice, not per-cell → cheap; recomputed per
snapshot so it stays tau-leaping-compatible; per-cell expression still materialized on demand):

1. **Diffusible fields (hypoxia / secreted factors)** — *translate PhysiCell/BioFVM*. Solve a
   steady-state reaction–diffusion on the deme grid: O₂ supplied at the tumour boundary (or seeded
   "vessels"), consumed ∝ local cell density → a hypoxia field with a tunable diffusion lengthscale.
   Generic secreted-factor fields (e.g. a stromal morphogen) work the same way. Each cell reads its
   deme's field value. Recapitulates the classic **viable rim / hypoxic-necrotic core**.
2. **Cell–cell communication (ligand–receptor)** — *translate scMultiSim / sCCIgen*. A cell's
   receptor-target genes are modulated by the **ligand-emitting cell density in its deme
   neighbourhood** (immune, stromal, or clone-specific ligands). Deme-neighbourhood aggregation, the
   spatial analogue of scMultiSim's grid-neighbour L–R model, but emitted by the *evolving* cell
   populations rather than a fixed input tree.

**The change to `get_exp`** (new multiplicative terms on designated gene programs):
```
cell_exp = baseline(cell_type) · dosage(CNA) · mut_effects(SNV)
                               · g_hypoxia(O2_field@deme,  hypoxia_program)
                               · g_CCI(neighbour_ligands@deme, receptor_program)
```
New parameters: a hypoxia-responsive gene program + field params (supply/consumption/lengthscale);
L–R pairs + a CCI-strength knob. All optional (off → current behaviour, backwards-compatible).

**Why it is the keystone for paper 1.**
- **Completes the "coupled > bolt-on" argument.** A bolt-on pipeline (take CNAs from SISTEM/CNAsim,
  generate expression on top) has *no microenvironment term at all*, and its CNA→expression coupling
  is whatever model you hand-impose — which, if it's the dosage model `clonealign`/`inferCNV` assume,
  makes the benchmark **circular** (you test a method on data generated by its own assumption). iscc's
  coupling emerges from evolution + microenvironment, so it is a **non-circular** benchmark, and it
  additionally provides niche/CCI structure a bolt-on cannot.
- **Unlocks the integration tools to run** (paper-1 demonstrations, scored against ground truth):
  DNA↔RNA — `clonealign` (CNA-dosage clone assignment), `cardelino` (SNV-from-RNA), `Numbat`/`copyKAT`/`inferCNV`
  (CNA-from-RNA); RNA↔spatial — `cell2location`/`Tangram` (deconvolution), `CellPhoneDB`/`COMMOT`/`NICHES`
  (CCI). The CCI/niche tools are *only meaningful once F8 exists*.

**Prior art & differentiator.**
- **scMultiSim** (Nat Methods 2025): expression guided by GRN + spatial L–R CCI on a grid — but
  takes a differentiation tree as *input*; no genome evolution, no CNA-driven fitness, no diffusible
  hypoxia, no DNA/treatment modality.
- **sCCIgen** (2025): high-fidelity SRT simulator for CCI (colocalization, neighbour gene–gene,
  spatial expression dependence) — but **reference-based**, no evolutionary process.
- **PhysiCell / BioFVM**: mechanistic O₂ diffusion → phenotype (viable rim, necrotic core) — but a
  biophysical cell simulator, not a sequencing-data generator, and not genotype-evolution-driven.
- **iscc (F8)** = combine the diffusible-field idea (BioFVM) and L–R CCI (scMultiSim/sCCIgen) but
  drive both from the **same evolving spatial tumour**, so microenvironment→expression is mutually
  consistent with genotype, CNA, spatial geography, DNA reads, and treatment. **No tool couples
  microenvironment-driven expression to genome evolution and a multi-modal readout.**

**Validation hooks.** Hypoxia program rises in low-O₂ core demes and is depleted at the rim
(recapitulating PhysiCell's rim/core); CCI-target genes elevated at clone/cell-type boundaries;
spatial-niche methods recover the known niches; `clonealign`-type DNA↔RNA assignment still succeeds
(CNA dosage preserved) **and** niche/CCI methods now have a ground truth to be scored against.

## I. External-simulator adapters (a recipe for plugging in scMultiSim, simATAC, …)

Rather than natively implement every modality (ATAC, GRN-driven expression, …), let iscc be the
**evolutionary + spatial substrate provider** and let mature external simulators be **modality
plugins** conditioned on that substrate. This gives users e.g. ATAC/GRN "on our tumor-evolution
process" without iscc reimplementing scMultiSim.

**The seam.** iscc already holds, per cell: a **lineage tree**, **spatial coordinates**, **clone /
cell-type identity**, expression, CNA. Expose it in standard interchange formats:
- lineage tree → **Newick** (the cell-division tree external tools consume);
- coords + identity + expression → **AnnData** (`obsm["spatial"]`, `obs["clone"]/["cell_type"]`).

**Per-simulator adapter.** A thin function maps the export to the tool's expected input, runs it, and
re-attaches the output to the same cell ids for a consistent multi-modal object. Worked example —
**scMultiSim**: feed iscc's cell-division tree as its input tree, iscc's cell states/coords as its
CIF / spatial inputs → scMultiSim emits RNA / ATAC / GRN-driven data on iscc's cells, inheriting
iscc's clonal + spatial structure. Same pattern for SymSim (tree), simATAC/simPIC (cell groups),
SRTsim (spatial reference).

**Honest limitation — structure-conditioned, not genome-consistent.** The plugged-in modality
inherits iscc's *tree / space / identity* but NOT the *genome* (the external tool doesn't know iscc's
CNAs/SNVs). So scMultiSim-via-adapter ATAC has iscc's clonal/spatial structure but not the
CNA→accessibility coupling a **native** iscc ATAC (the deferred "regulatory layer") would. Adequate
for "I want ATAC support"; not for "I want genome-consistent ATAC ground truth". Document this
tradeoff wherever adapters are offered.

**Deliverable.** An `iscc.integrations` seam: `to_newick(tumor)`, `to_anndata(cell_data)`, and
adapter stubs (`adapters/scmultisim.py`, …). Low-risk, additive; each adapter is optional and guarded
on the external package being installed (mirrors the reads.py binary-optional pattern).
