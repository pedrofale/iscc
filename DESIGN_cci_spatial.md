# DESIGN — cell–cell communication, intra-deme layout, and a named genome

Scope: make F8 a genuine cell–cell-communication substrate, give cells a real within-deme
arrangement, and put real gene symbols on the abstract genome. Prompted by a comparison against
**sCCIgen** (real-data-based SRT simulator, purpose-built CCI ground truth) and **scSpatialSIM**
(point-pattern generator). Neither models time, clones or copy number; both beat iscc on spatial
texture and count fidelity.

## Decisions taken (2026-08-27)

1. **F8 target = the clone-vs-interaction confound**, not parity with sCCIgen on planted-interaction
   detection. Competing on "detect the planted L–R pair" is their turf: their ground truth is
   purpose-built and their counts are fitted to a real reference. The question only iscc can pose is
   that neighbouring cells are also *clonally related*, so interaction-driven co-expression is
   entangled with clone-driven co-expression. sCCIgen has no clones.
2. **Intra-deme layout = informative**, not texture-only. Cell placement within a deme carries
   information (same-clone proximity, type-pair attraction/inhibition).
3. **Genome = naming layer**, not marginal-matching. Real gene symbols on real arms; fitting real
   per-gene marginals/dispersion is deferred.

## Ordering (this is a dependency chain, not three parallel tracks)

    W1 gene symbols  ->  W2 intra-deme layout  ->  W3 L-R channels  ->  W4 the confound benchmark

CellChat and CellPhoneDB look up ligand–receptor pairs **by gene symbol**, so without W1 they cannot
be run at all and a planted channel has nothing to match against. Contact- and neighbourhood-based
CCI needs cell-level positions, so W3 needs W2.

---

## W1 — a named genome (naming layer only) — **ON HOLD 2026-08-27**

**Why it is on hold.** Naming imports a prior-knowledge graph the abstract genome does not satisfy,
and the failure is in the generative model rather than in the mapping:

- Roles are directional and would be contradicted by a wrong label: `driver_types` is
  `{-1 TSG, 0 passenger, +1 oncogene}` and a mutated oncogene gets 2x expression while a mutated TSG
  gets 0.5x (`selection.py`). Calling a `+1` gene TP53 would have the engine up-regulating it on
  mutation. A role-aware mapping fixes this one.
- **It does not fix the per-arm content contradiction.** iscc draws roles UNIFORMLY at random over
  the whole genome — `rng.choice([-1,0,1], p=[prop_driver/2, 1-prop_driver, prop_driver/2])`, the
  same rate on every segment — so simulated per-arm oncogene/TSG counts are binomial noise around a
  constant. Real arms are strongly uneven (17p TSG-rich, 8q oncogene-rich), and `GenomeSpec` already
  carries those COSMIC/Davoli counts. The naming layer would be importing an annotation that the
  engine's own driver layout contradicts.
- `prop_driver` is user-set. At 0.5 a named genome asserts that half of all human genes are drivers.
- Two further consistency debts naming would expose: gene programmes are random gene sets, so any
  pathway tool reading real symbols would find no coherence (or spurious enrichment); and per-gene
  expression levels were deliberately left unmatched, so a named gene would sit at an implausible
  level.

**Narrower replacement under consideration.** W1's only purpose was to unblock W3/W4, and that does
not need a named genome. It needs the ligand/receptor genes of the planted channels to carry real
symbols so CellChat/CellPhoneDB can match them; every other gene can keep its abstract name, and the
tools simply ignore unmatched genes. Choosing those L-R genes from the PASSENGER set
(`driver_types == 0`) makes no role claim at all, no arm-content claim, and imports no annotation.
Decide this before any implementation.

**Independent finding worth keeping** (true regardless of naming): the uniform-across-arms driver
layout is a real divergence from the arm content `GenomeSpec` records, and it is relevant to the
real-genome / Charm work, where per-arm selection is inferred against exactly that annotation.

### Original spec (retained for when/if this resumes)

**Goal.** Real gene symbols in `var_names` of the scRNA / Visium / imaging outputs, placed on real
chromosome arms consistently with the CNA structure, so that a gene's copy number is its arm's copy
number and CNA contiguity maps to genomic contiguity.

**Where.** A mapping layer applied at assay/materialisation. The engine stays abstract — evolution
acts on segments and a small driver set, and genes are ride-alongs; simulating 20k of them buys
nothing but 20k correlated ride-alongs.

**Reuse what exists.** `iscc/inference/genome.py` already carries `GenomeSpec` over ~39 human
autosomal arms with real lengths (UCSC cytoBand), COSMIC/Davoli oncogene and TSG counts and Charm
scores, and arm length already sets `segment_sizes`. Today this is wired only into the inference
side. This is the piece that makes an upscaled genome coherent rather than decorative relabelling.

**Requirements.**
- A reproducible bijection `abstract gene index -> gene symbol`, stable within a run, seeded.
- Symbols ordered by genomic position within their arm, so contiguous abstract segments stay
  contiguous in the real coordinate system.
- **The mapping MUST include the ligand/receptor genes used by the planted channels.** Reserve those
  positions rather than assigning symbols arbitrarily, or W3's channels are unexpressible.

**Explicit non-goal.** Matching real per-gene marginals, dispersion or gene–gene correlation. That
is the heavier "anchor to a reference at the assay layer" design; revisit only if the fidelity gap
turns out to cost something measurable.

## W2 — informative intra-deme layout

**Today.** `sample/section.py::spatialize(jitter=0.5)` scatters each deme's cells uniformly in its
unit cell, and the docstring says the layout is "cosmetic" because the engine tracks only per-deme
counts.

**Measured, 2026-08-27** (pure geometry, Visium v1: `spot_radius=0.55`, `spot_pitch=2.0` demes):

    cells/spot 11.2      distinct demes/spot 1.62 (max 4)
    fraction of a spot's cells from its DOMINANT deme: 0.94
    re-jitter only, identical deme composition:
      cells/spot correlation 0.10     mean |change| 1.5 of 11.2     deme-set Jaccard 0.61

So spot COMPOSITION is deme-resolution and robust, but spot DEPTH is jitter-determined, and *which
cell types co-occur within a spot* is currently a random draw from the deme's composition. With 75%
of spots multi-type in the rctd dataset, the mixtures the deconvolution benchmark measures are
presently unstructured.

**Model.** A labelled point process at materialisation, conditioned on the deme's composition (counts
by clone and by type — already tracked by the engine). Sequential placement or a Gibbs/Strauss
process with:
- a **hardcore radius** (cells cannot overlap),
- **same-clone attraction** (clonal patches within a deme),
- a **type-pair interaction matrix** (attraction/inhibition, e.g. immune at the gland–stroma
  interface).

This is sCCIgen's model applied *inside* a deme, conditioned on a composition that came from
evolution — their spatial realism, our provenance, engine untouched. Cost is not a concern:
dart-throwing is O(n^2) per deme and n is carrying-capacity-sized.

**Contract change, stated.** Composition stays exact; the layout is no longer cosmetic. Keep the
uniform mode behind a flag so published numbers remain reproducible. **This restates the RCTD /
deconvolution results** and needs a deliberate re-run.

## W3 — ligand–receptor channels in F8

**Today.** `_cci_field(emitter_type, lengthscale)` is a Gaussian-smoothed density of ONE emitter type
over deme coordinates, multiplying a fixed target-gene set:
`mod[:, cci_targets] *= 1 + strength * signal[deme]`. One emitter type, one target set, one strength,
one lengthscale — and the emitter contributes purely by being present. That is a NICHE field, not a
communication model. Against sCCIgen's four ground truths: expression–distance ~yes, regional
variable genes yes (hypoxia), colocalization no, neighbour expression–expression no.

**Model.** N typed channels, each
`(emitter type, receiver type, ligand gene, receptor gene, target gene set, strength, lengthscale)`:
- ligand availability at a position = kernel-weighted sum of nearby emitter cells' **ligand
  expression** (not merely their density),
- the effect on a receiver = `strength * ligand_available * receiver's own receptor expression`,
- evaluated at **cell** level using W2's coordinates, not per deme.

Receptor-dependence is the load-bearing change: every tool in this space scores L–R pairs, so without
it they are being tested on a signal that is not in the data.

**Invariant to preserve.** F8 modifies the READOUT only — growth stays byte-identical, the modifier
draws from a dedicated RNG, and OFF is bit-identical. Ligand and receptor levels are read at
materialisation and never feed back into fitness.

**Ground truth to surface.** The channel table; per-cell received signal per channel; and each cell's
clone (needed by W4).

## W4 — the benchmark: interaction vs clonal relatedness

**Question.** Can a CCI method separate interaction-driven co-expression from co-expression that is
simply clonal relatedness in space?

**Why it is ours.** Neighbouring cells in a real tumour are usually clonally related. A CCI simulator
without clones cannot construct the confound, so no existing benchmark measures whether a method is
fooled by it.

**Design.** Plant channels (W3) in a tumour whose clones are spatially structured, and sweep the
**clonal territoriality** knob — `dispersal_rate`, the same knob that drives the PEtracer
lineage–space confound. Low dispersal gives clonal patches and maximal confounding; high dispersal
intermixes clones and should clean it up. Score a real tool (CellChat / CellPhoneDB / COMMOT) on
recovery of the planted channels AND on how many false channels it reports that are in fact
clone-driven. The expectation is a false-positive rate that tracks territoriality — same shape as the
PEtracer and CNA-dosage-vs-gene-programme results, and a natural fourth member of that cluster.

## Open questions / prerequisites

- **Which CCI database** (CellChatDB human?) — it determines the L–R symbols W1 must reserve.
- **Emitters must exist.** `_emitter_density("immune")` needs an immune compartment present in the
  tumour; the rctd dataset added one, but the default configs need checking.
- **W2 restates the deconvolution numbers.** Budget the re-run.
- Whether the naming layer alone is enough for the tools to run end-to-end, or whether pathway/marker
  databases also need plausible expression LEVELS (the deferred marginal-matching work).
