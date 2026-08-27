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

    W0 database  ->  W3 L-R channels  ->  W4 confound benchmark        (Visium-resolution: ships alone)
    W2 intra-deme layout                                              (needed only for imaging resolution)

**Ordering corrected 2026-08-27.** An earlier draft made W2 a prerequisite for W3. It is not, and the
question that actually decides the order is **which assay the benchmark targets**.

- **W3's load-bearing change does not need cell positions.** The effect is
  `strength x ligand_available x the receiver's own receptor expression`. Ligand availability can stay
  a per-deme field exactly as F8 computes it today, while the RECEPTOR term is already per-cell,
  because expression varies cell to cell within a deme by clone and by programme activity. So W3
  delivers genuine per-cell response heterogeneity at deme-resolution geometry.
- **Visium geometry is exact.** `visium.py::_layout()` returns the fixed v1 slide (or a fixed
  hex/square array); `_place_section` translates and rotates the tissue ONTO that lattice. Spot
  coordinates — what a spatial CCI method reads — never depend on cell jitter. Jitter only shifts
  which cells fall in a spot, and spot composition is 94% dominated by a single deme (measured, W2).
- **W4's confound is a territory-scale phenomenon.** Clonal patches span tens of demes and are tuned
  by `dispersal_rate`; the confound does not live below deme scale.
- **W2 becomes a prerequisite only at single-cell spatial (imaging) resolution**, where co-deme cells
  are currently emitted at identical coordinates (see W2), or for contact-range methods operating
  below ~20-25 um.

**Practical argument for starting with W0+W3.** F8 is optional and default-off, so W3 is additive and
restates NOTHING. W2 restates the deconvolution results. And W0+W3+W4 is where the novelty sits — the
decoy classes and the clone-vs-interaction confound — so doing it first de-risks the line: if the
confound does not separate cleanly, that is learned before investing in layout infrastructure.

**Caveat if W3 ships alone.** At deme resolution the planted signal is piecewise-constant over
~20-25 um blocks. Visium spots are ~55 um, so that is below the observation scale and invisible to the
benchmark. It would show at imaging resolution — which is the same condition that makes W2 required.

## W1 — a named genome — **REJECTED 2026-08-27**

Considered and dropped. Superseded by **W0**: iscc emits its own L–R database over its own abstract
gene identifiers, so real gene symbols are not needed at all.

It was rejected on its own merits too, and the reason is in the generative model rather than in any
mapping. Roles are directional — `driver_types` is `{-1 TSG, 0 passenger, +1 oncogene}`, and a mutated
oncogene gets 2x expression while a mutated TSG gets 0.5x — so a wrong label would have the engine
up-regulating a gene named TP53 on mutation. A role-aware mapping fixes that, but not the deeper
problem: iscc draws roles UNIFORMLY at random across the genome, so simulated per-arm oncogene/TSG
counts are binomial noise around a constant, while real arms are strongly uneven and `GenomeSpec`
already carries the real counts a naming layer would import. `prop_driver` is user-set, so at 0.5 a
named genome would assert that half of all human genes are cancer drivers. Naming would also expose
two debts we deliberately deferred: gene programmes are random gene sets with no pathway coherence,
and per-gene expression levels are unmatched.

That uniform-across-arms driver layout is a real divergence from the annotation the real-genome /
Charm inference is scored against, independent of naming. Tracked in `BACKLOG.md` under
*Engine / inference follow-ups*.

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

## W0 — iscc emits its OWN ligand–receptor database (decision, 2026-08-27)

**Decision.** Rather than name genes so a curated database can match them, `iscc` **generates its own
L–R database** and the analysis tools are pointed at it. This replaces W1 entirely: gene identifiers
stay abstract, and no real cancer-gene annotation is imported — so none of the consistency debts that
put W1 on hold arise.

**Verified feasible for both primary tools.**
- **CellChat** ships a documented `Update-CellChatDB` path taking `interaction_input`,
  `complex_input`, `cofactor_input` and `geneInfo` CSVs.
- **CellPhoneDB** generates custom databases from `gene_input` / `protein_input` / `complex_input` /
  `interaction_input`, and **`--user-interactions-only`** makes it use ONLY the supplied
  interactions, so no real pairs leak in. This is the flag that makes the substitution total.
- **COMMOT** takes an L–R dataframe directly — **NOT verified**; treat as an assumption until checked.

**The database is the instrument, and it needs decoys.** A database containing only the planted
channels makes recovery trivial: there is nothing to discriminate. CellChatDB carries ~2,000 human
interactions of which a handful are active in any given tissue, and a benchmark has to reproduce that
selection problem. So the emitted database holds N candidate pairs of which only K are wired into F8,
in three deliberate classes:

1. **Active channels** — wired into F8; genuinely spatially driven signalling.
2. **Neutral decoys** — ligand and receptor genes exist and are expressed comparably, but there is no
   downstream effect. These measure the false-positive rate against expression alone.
3. **Clone-confounded decoys** — ligand/receptor expression co-varies with CLONE, and therefore with
   space (clones are territorial), but no signalling exists. **These are the traps.** A method blind
   to clonal relatedness should call them.

The W4 score is then: can the method separate class 1 from class 3? That is the confound benchmark,
and this database structure is what makes it constructible. No CCI simulator without clones can build
class 3 — which is precisely the gap against sCCIgen.

**Hard constraint.** Decoy ligand/receptor genes MUST be expressed comparably to the active ones. If
they are silent or flat they are rejected on expression grounds rather than on communication
evidence, and the benchmark measures nothing.

**Caveat to state in the paper, not to discover in review.** Supplying our own database means we test
the tools' statistical and spatial inference, NOT their curated prior knowledge (pathways, complexes,
cofactors). That is a legitimate and arguably the correct scoping — annotation quality is a separate
axis, and holding it fixed is what isolates the inference — but it must be said plainly, or a
reviewer can object that half of CellChat was switched off. CellPhoneDB's `--user-interactions-only`
makes the scoping explicit and auditable.

**Open question.** Multi-subunit receptor complexes: both tools support them. The simplest v1 is
strict 1:1 pairs with empty complex/cofactor tables. Decide whether complexes are in scope before
building.

**Revised chain.** `W2 intra-deme layout -> W3 channels + database -> W4 confound benchmark`. The
database and the F8 wiring are two outputs of ONE object and must be generated together — a pair is
"active" precisely because F8 is wired to it.

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
