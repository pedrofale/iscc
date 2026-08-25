# DESIGN — therapy-induced senescence, polyploidy, and unequal division [design-first; NOT built]

Status: **DESIGN-FIRST (2026-08-24), NOT built.** Motivated by an experimental collaboration (Pedro +
Jeff, cell-line imaging) with matching published evidence. Companion to `DESIGN_focal_cna.md` (owns
WGD), `DESIGN_phenotype_plasticity.md` §3.3 (the carried drug-induced state). **This is NOT a fifth
escape mode** — see §2. No engine code until sign-off.

## 1. The observation

From live imaging of treated cancer cell lines (all starting **diploid**):
1. Under treatment, **all** cells become senescent — not a subpopulation.
2. After treatment is **withdrawn**, some arrested cells resume dividing.
3. Those divisions are **genomically unstable**: visibly unequal genome splitting, sometimes
   **three-way (multipolar)** divisions.
4. Escape often follows a few rounds of **endoreplication**.
5. **In a subsequent treatment window the escapers become senescent again.**

Point 5 is the load-bearing one and is easy to miss.

The literature name for 1-4 is **therapy-induced senescence (TIS) → polyploid giant cancer cells
(PGCC) → neosis**: mitotic slippage or endoreplication produces giant polyploid cells, which then
depolyploidise by nuclear budding and asymmetric cytokinesis into aneuploid progeny that re-enter the
cycle. Paclitaxel/vincristine promote PGCC formation directly via mitotic stress.

## 2. This is NOT resistance — and that reclassifies the whole thing

The escapers are not drug-insensitive: re-treat them and they arrest again. What the population has is
the ability to **survive and repopulate**, not to divide under drug. Clinically that is the difference
between "the drug stopped working" and "the drug never eradicated anything".

**A burden curve cannot distinguish this from acquired resistance.** Response then relapse draws the
same shape as Kane/Maley modes III and IV. The discriminator is the **second treatment window**: a
resistant relapse keeps growing through re-treatment; this one re-arrests.

**The published protocols mostly cannot detect resistance**, because they remove the drug before the
interesting phase. Verified in Ahmadinejad et al. (*Therapy-induced senescence is a transient drug
resistance mechanism in breast cancer*, PMC12044945): doxorubicin applied **5 days then washed out**;
cells sit senescent ~7 further days with **no drug present** before repopulating. They did re-treat —
"re-TIS cells displayed almost identical resistance profiles to original TIS cells", i.e. they arrested
again. The field's own data therefore shows TOLERANCE while its language says RESISTANCE (the paper's
own title says "transient drug resistance"). Growth under drug was never tested.

Clinically the washout protocol is realistic: MTD chemotherapy is cycled because toxicity forces it, so
real treatment supplies the drug-free windows this mechanism needs. It predicts re-challenge with the
same agent should work — which is what Kuczynski et al. (*Drug rechallenge and treatment beyond
progression*, Nat Rev Clin Oncol 2013;10:571) documents, and which the Kane/Maley review cites as
evidence that tumours regain sensitivity after a holiday.

**Same structural point as the reversion work** (memory `maley-escape-modes`), reached from the other
direction: revertants only take over during the drug-free tail, and dosing to progression eliminates
them. Different mechanism entirely — copy-number reversion vs senescence escape — but the same
principle: **the drug-free window is where non-resistant escape lives, and whether a relapse is called
"resistant" depends almost entirely on whether anyone re-treated it.**

## 3. What it IS: a karyotype-evolution engine

Whole-genome doubling occurs in **>25% of human tumours, early in tumorigenesis** (after an antecedent
transforming driver), and tetraploidy is proposed as **the intermediate en route to aneuploidy**. WGD+
tumours are more chromosomally unstable and more permissive to aneuploidy — more events and a wider
variety. WGD is linked to increased tumour-cell diversity, accelerated genome evolution and worse
prognosis.

So diploid → endoreplication → messy division → aneuploid progeny is the canonical route by which
cancer genomes *become* cancer genomes. The aneuploidy it generates is a **substrate**, not resistance:
each arrest/escape cycle deals a fresh hand of karyotypes, and over enough cycles one might be
genuinely resistant. (That predicts resistance emerges FASTER under intermittent dosing than
continuous — an uncomfortable implication for drug holidays, and testable.)

There is an initiation-side version with the same architecture: **oncogene-induced senescence** is a
tumour-suppressive barrier, and escape from it is a step toward transformation. Same structure
(arrest → escape → instability), different trigger, different stage.

**Why the cell-line system is valuable:** WGD is currently reconstructed *retrospectively* from patient
genomes. This is an inducible, observable model of the tetraploid→aneuploid transition with the
divisions visible — the event everyone infers and nobody watches.

## 4. Evidence in real tumours (stratified honestly)

- **Patient, mechanistic (strongest).** Bone-marrow aspirates from 44 advanced prostate cancer
  patients: circulating tumour cells with increased genomic content (CTC-IGC) associated with poorer
  progression-free survival, and **single-cell copy-number profiling showed CTC-IGC share clonal origin
  with ordinary CTCs** — which rules out fusion with host cells, the main competing explanation for
  polyploidy in patient material, and supports genuine endoreplication. (*Oncogene* 2024,
  s41388-024-03212-z.)
- **Patient, observational.** Senescence-like resilient phenotypes enriched in relapsed patient
  tumours; TIS documented in patient tumours generally.
- **Mouse (the causal step).** Cells enriched for the senescence-like phenotype formed tumours in both
  immunodeficient AND immunocompetent hosts — senescent cells escaping and regaining tumorigenicity
  in vivo.
- **In vitro (where the mechanism is established).** Escape from TIS reported as *universal* across
  breast cancer cell lines.
- **NOT established:** that PGCC/TIS escapers *cause* patient relapse rather than mark it. No
  longitudinal patient data tracks lineages through arrest and escape — and cannot, without watching
  divisions.

## 5. What iscc has, and the one structural gap

**Has:** `wgd_rate` as its own per-division event channel (`components/cell.py`, DESIGN_focal_cna.md
v1) — duplicates every copy on both homologs, carrying SNVs into duplicates; `is_wgd` tracked as
monotone ground truth; copy-number-resolved per-cell output.

**Gap 1 — no arrest state.** No alive-but-not-dividing state that can be exited. NOTE this is nearly
free under `kill_mode: proliferation`: `added_death = intensity * kill_rate * division_rate`, so a cell
at `division_rate = 0` takes **exactly zero** drug-induced death, with no resistance trait anywhere —
and it is not resistance, because re-treatment arrests it again. This also **corrects** the conclusion
in memory `iscc-drug-persisters` that "a division-rate discount cannot hold a floor": true for a
discount, FALSE for arrest. At `division_rate = 0` the cell neither divides nor is killed and the floor
holds exactly. The dormancy formulation those notes said was needed is this.

**Gap 2 — WGD is welded to division.** It fires inside `mutate()`, which only runs on a dividing cell.
Endoreplication is replication WITHOUT division and cannot be expressed. `max_ploidy: 6` would also
reject the cells of interest (PGCCs reach 8n-32n).

**Gap 3 — every division is exactly equal. THIS IS THE STRUCTURAL ONE.** `Cell.divide()` is
`copy(self)`; the daughter *aliases* the parent's genome under copy-on-write and only diverges if
`mutate()` changes one. There is no asymmetric genome segregation and no way to express a three-way
division. **So iscc can double a genome but cannot represent what WGD DOES** — the tetraploid→aneuploid
conversion — which matters for CIN, karyotype evolution and prognosis modelling well beyond drug
resistance. Breaking copy-on-write aliasing is also the same scalability wall flagged for the §3.3
state feature; check whether the two compound.

Gaps 1-2 give the phenotype (survive, regrow, re-arrest). Gap 3 gives the karyotypes.

## 6. What to measure (identifiability, not a shopping list)

The model is a state machine: proliferating → arrested → endoreplicating → escape division → aneuploid
progeny. Ordered by how much each buys:

1. **Escape rate as a function of ploidy.** Decides whether endoreplication is CAUSAL (8n escapes
   faster than 4n → extra genome is why escape works) or incidental (flat in ploidy → ignore it).
   Nothing else matters until this is answered; "often after a few rounds" is not yet an answer.
2. **Division-geometry distribution** — fraction bipolar/tripolar/higher, and how unequal the split is.
   The genuinely rare measurement; most of this literature infers neosis from endpoint karyotypes.
3. **Progeny fate** — what fraction of escaper daughters divide again vs die (imaging gives this
   directly as lineage-or-not).
4. **Escape latency** after withdrawal.

**The gap imaging cannot close:** polarity and gross asymmetry are visible; the GENOMIC content of each
daughter is not. That needs scDNA-seq of the regrown population, or a karyotype reporter.

**Why this ordering gives identifiability:** measure escape-rate-vs-ploidy and the geometry
distribution from imaging, and the **karyotype distribution of escapers becomes a PREDICTION**, with
sequencing as the test rather than the calibration. That is the opposite of the Gatenbee trap
(`DESIGN_phenotype_plasticity.md` §0) where every parameter was unmeasurable — here every parameter is
directly observable.

### 6.1 Sequencing design — bulk WGS on single-cell expansions from escapers

Bulk WGS of a **clonal expansion** is the right instrument here: ~100% purity, high effective depth per
allele, no single-cell amplification artefacts. But nearly all the power comes from one design choice.

**SEQUENCE THE PARENT, AND MANY INDEPENDENT ESCAPER CLONES.** Then you are not inferring history from
one genome — you difference each escaper against a KNOWN ancestral karyotype, and the distribution of
outcomes across independent clones **IS the segregation kernel, measured directly**. That is the thing
nobody has. **n ≈ 10-20 clones**, more if outcomes are heterogeneous. Depth buys resolution WITHIN one
clone; the founder karyotype is clonal by construction, so it does not need it.

**Core readout: ALLELE-SPECIFIC copy number, not total.** The parental line is heterozygous at millions
of SNPs, giving a phasing backbone — so per chromosome per escaper you get not just *how many copies*
but *which parental homolog*.

**The sharpest signature — homolog identity.** From 2n (1 maternal + 1 paternal) you can lose a copy,
but you cannot obtain **two copies of the SAME homolog** without a duplication event. From 4n
(2M + 2P) a random three-way split readily hands a daughter 2M + 0P — copy number 2, normal on TOTAL
CN, but copy-neutral LOH **with homolog identity**. Many chromosomes showing two copies of one parental
homolog in a single clone, with no duplication signature, is hard to reach from a diploid lineage and
easy from a tetraploid one. **A footprint of the ROUTE, not the endpoint.**

**Supporting readouts:** whole-chromosome vs arm-level event ratio (multipolar mis-segregation moves
WHOLE chromosomes; replication stress / BFB give arm-level and focal); nullisomies and homozygous losses
of regions lethal from 2n (the R15 test); SNV multiplicity to time the WGD against the mutation clock;
SV burden and pattern, mainly to EXCLUDE chromothripsis (clustered breakpoints + oscillating CN rather
than clean whole-chromosome changes).

**Two controls not to skip.** (1) **Single-cell expansions from UNTREATED cells** — without a baseline
drift rate nothing is attributable, cell lines drift. (2) **Escaper clones that were never polyploid**,
identified by imaging before picking — the internal control separating "escape" from "escape VIA
polyploidy". The imaging is what makes (2) possible at all.

**COVERAGE REALITY: ~10-20x is available, and that is mostly fine.**
- **Holds.** Allele-specific CN is a SEGMENT-level statistic aggregated over thousands of het SNPs (a
  10 Mb segment has ~10,000), so per-SNP noise at 15x averages out — LOH calling, copy-neutral-LOH vs
  hemizygous loss, and the homolog-identity test all survive. Total CN/ploidy is robust (a 1 Mb bin has
  ~1,500 reads at 15x). ~100% purity + a sequenced parent removes the two things that make tumour CN
  calling hard.
- **Degrades.** Subclonal CN resolvable only above ~25-30% cell fraction; subclonal SNV clusters
  essentially not at all. WGD timing by SNV multiplicity becomes marginal (per-mutation VAF too noisy at
  15x; only the aggregate distribution is usable, and it depends on mutation burden). SV
  characterisation limited — enough to notice gross chromothripsis, not to rule it out properly.
- **The trade: spend on CLONES, not depth.** 20 clones at 15x beats 3 at 100x, because the kernel is a
  distribution ACROSS independent events.
- **For the ongoing-instability question, deep bulk is the wrong tool.** "One-off event or persistent
  state" is measured directly by **low-pass single-cell CNV sequencing** (0.01-0.1x per cell, hundreds
  of cells) — mature, cheap, and how the punctuated-evolution work was done. Bulk answers it only by
  inference at any depth.
- **Pick clones early** regardless: fewer generations means less drift, which at 15x you cannot see but
  which still shifts segment-level calls.

**The unique asset:** each sequenced clone can be paired with imaging of the actual division that
produced it — the segregation kernel AND its genomic consequence for the SAME event. That pairing does
not appear to exist anywhere.

## 7. Cheap things to do first, before building anything

- **Re-treatment as a diagnostic, in the simulator, today.** Add a second chemotherapy phase to the
  schedule (`arc.py` already builds a fresh `Chemotherapy` per entry) and re-treat a mode IV relapse.
  A genuinely resistant relapse keeps growing; a tolerance relapse would arrest. Config-only, and it
  makes the escape-mode figures assert something a burden curve alone cannot.
- Decide whether the arrest state is another level of the §3.3 carried state (likely) or its own
  object. If shared, the marginal cost here is endoreplication + asymmetric division only.
