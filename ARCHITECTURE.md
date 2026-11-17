# iscc architecture

`iscc` simulates a tumour growing in a tissue and then measures it the way an experiment would.
This document describes how the package is built: what each part models, how the parts hand data to each other, and which module implements each one.
The constructor parameters of each part are documented in the API reference of the package documentation, and `notebooks/example_config.yaml` sets them for the tutorials.

## The flow: growth, sampling, assays

```
GenotypeTumor ──grow()──▶ per-deme genotype counts ──Resection / make_cell_data──▶ cell_data ──assay.run()──▶ observed data
 (tumor/models)                                          (sample/, materialise)       (dict of       (data/)       (+ ground truth)
                                                                                       per-cell tables)
```

1. **Growth** (`iscc.tumor.models.count.GenotypeTumor`).
   The tumour lives on a lattice of demes.
   Each deme holds counts of cells per genotype, and a shared registry holds one representative cell per genotype.
   `grow()` advances the counts; nothing per cell exists yet.
2. **Sampling** (`iscc.sample`).
   A `Resection` (`sample/resection/resection.py`) cuts the grown specimen: `bisect` splits it in the plane, `dissociate` turns a part into a single-cell suspension for the sequencing assays, and `slice` thins a part to a section for imaging and Visium.
   `Biopsy` (`sample/biopsy/biopsy.py`) selects punch, needle or multi-region samples, `dissociation.py` adds the composition bias and stress signature of tissue dissociation, and `section.py` places each deme's cells inside the deme for spatial assays.
   Each sample is materialised on demand (below), so a centimetre-scale tumour is never expanded whole.
3. **Assays** (`iscc.data`).
   Each assay takes the per-cell ground truth of a sample (`cell_data`) and returns observed data together with the truth it came from.

The command-line tools follow the same three stages: `isccsim` (`tumor/main.py`), `isccsample` (`sample/main.py`) and `isccdata` (`data/main.py`).

## The engine: state and events

`GenotypeTumor` (`tumor/models/count.py`) is the one engine.
Its state is:

- `demes`: one dict per deme, genotype id to cell count.
  The primary lattice comes first; a metastatic lattice, when enabled, is appended after it.
- `genotypes`: the registry, genotype id to a representative cell (`tumor/components/cell.py`) that carries the genome, the genome summary and the evolutionary parameters (division, death and dispersal rates and the heritable traits).
  Normal cells are registered under their type name (`"epithelial"`, `"stromal"`, `"immune"`, `"host"`) and never divide.
- `genotypes_parents`: the genealogy, and per-genotype birth records (`genotype_births()`).
- `traces`, `trace_times` and optional `spatial_history`: the population over time.

The events are per clone and deme: a cancer cell divides at its division rate and dies at its death rate.
A division may mutate the daughter and may send a daughter to another deme (demon's rule, below); a birth needs a free slot where it lands.
Immune killing, compartment hazards and treatment add to the death rate.
Normal cells change only by eviction, when a cancer birth displaces one, and by drug toxicity.
Mutation and dispersal are outcomes of a division, not channels of their own.

`update_mode` selects the algorithm that advances this state; see the next section.
All randomness comes from per-run streams keyed by (seed, purpose, step) (`_run_rng`), so no two seeds, purposes or steps share a stream.

The constructor takes the nested parameter blocks of a config (or the config path).
`tumor/models/params.py` resolves them into two typed groups, `RunSettings` and `ModelBlocks`, and the construction steps (`_init_*`) read those in a fixed order, which is the order of every random draw at construction.

The engine is split by concern into mixins of `GenotypeTumor`, all in `tumor/models/`:

| Module | What it holds |
|---|---|
| `count.py` | the class, construction, the registry, count bookkeeping, the rates, the exact and tau-leaping steps, `grow` |
| `tissue.py` | the ductal field, slots and eviction, compartment composition, origin confinement, dispersal routes |
| `metastasis.py` | the metastatic lattice and the transit of seeding cells |
| `therapy.py` | dosing, the label and mechanistic drug responses, surgery, the resistance state |
| `materialise.py` | counts to per-cell tables, the microenvironment fields, passenger re-creation |
| `plotting.py` | Muller plots, tissue maps, phylogenies, H&E images |
| `params.py` | the parameter groups and their resolution |

## Simulation algorithms

`iscc` simulates its tumour model with two standard algorithms and introduces none of its own.
This section is written for a reader who knows CINner's or demon's algorithm; the model (the rates) is described elsewhere in this document and is kept apart from the algorithm that steps it.

### The process

The state is, for every deme, the number of cells of each genotype (`GenotypeTumor.demes`, a list of `{genotype id: count}`), over a registry of genotypes shared by all demes (`genotypes`).
Every cancer clone in every deme has two event channels with per-cell rates: division at `b` and death at `d` (`_div_rate`, `_death_rate` in `count.py`).
Mutation and dispersal are not channels of their own: they are outcomes of a division, decided when it fires (demon's invasion rule: `_divide_noble` in the exact mode, `_split_births_noble` in the tau mode, and `_mutate_daughter` for the new genome in both), and a dispersing daughter is routed by `_route_one` (exact) or `_route_births` (tau) in `tissue.py`.
The rates depend on the clone's genome and on its deme (crowding, the resident and stromal hazards, the compartment), so the population is a continuous-time Markov jump process whose propensities are piecewise constant between events.
Time is measured in generations, the unit of the rates.

### The exact mode: Gillespie's direct method, in demon's hierarchical form

`update_mode="exact"`; one event per call of `update()` (`count.py`).

1. A deme is chosen with probability proportional to its summed rate (`_update_event`; `deme_rates` is kept per deme by `_deme_rate` and refreshed for every deme an event touches by `_refresh_rate`).
2. Within the deme, a clone is chosen in proportion to its count times `b + d`, and the event is death with probability `d / (b + d)`, else division (`_event_in_deme`).
3. Time advances by an exponential waiting time with the total rate, `Exp(sum(deme_rates))`, drawn after the event (`update()`), and the founding acinus's generation clock counts that time (`_origin_tick` in `tissue.py`).

This is Gillespie's direct method (Gillespie 1977, *J Phys Chem* 81:2340), organised hierarchically as demon organises it (Noble et al. 2022, `demon.cpp`): the hierarchy only factorises the choice of the next event, so the event probabilities and the waiting-time distribution are those of the direct method.
Each event draws from its own random stream (`_run_rng`, keyed by seed, purpose and event index), so runs at different seeds never share a stream.
The exact mode is the reference against which the tau mode is checked.

### The tau mode: tau-leaping, as in CINner

`update_mode="tau"`; records are taken every `tau` generations (0.5 in the paper), and each record interval is covered by one or more leaps (`_tau_generation`).
Within a leap of length `dt`, every channel of every clone in every deme fires a random number of times at its rate at the start of the leap (Gillespie 2001, *J Chem Phys* 115:1716), in `_tau_substep`:

- divisions are `Poisson(b c dt)`;
- deaths are `Binomial(c, min(1, d dt))` (the default, `tau_deaths="binomial"`), binomial tau-leaping for a first-order channel (Tian and Burrage 2004, *J Chem Phys* 121:10356; Chatterjee et al. 2005, *J Chem Phys* 122:024112), so no count goes negative.

**How CINner does it, for comparison.**
CINner (Dinh et al. 2025, `CINner_step_1.r`) uses a fixed step, draws the total number of events of each clone from a Poisson distribution (a normal one above 1,000), redraws a count that exceeds the clone, and splits it multinomially into deaths, divisions and divisions with a new genotype.
`iscc` draws divisions and deaths as separate channels, uses the binomial for deaths instead of a redraw, and chooses the step adaptively, because of its crowding step (below).

**The step.**
In the paper's crowding law the death rate of every cancer cell in a deme jumps from `d0` to `d1` (0.05 to 5) when the deme passes its capacity `K`, so a leap that freezes the rates lets a full deme lose several cells where the process loses one.
The step is therefore chosen by a published rule, set by `deme_params["tau_step"]` and applied in `_tau_generation`:

- `"cgp"` (the default): the species-based ε-bounded step selection of Cao, Gillespie and Petzold (2006, *J Chem Phys* 124:044109), with each clone in each deme as a species of highest reaction order 1: `dt = min over species of max(ε c, 1) / |(b - d) c|` and `max(ε c, 1)^2 / ((b + d) c)` (`_cgp_dt`, ε = `tau_epsilon`, default 0.01, read by `_init_tau`);
- `"cgp_critical"`: tau-leaping with critical reactions (Cao, Gillespie and Petzold 2005, *J Chem Phys* 123:054104): the channels of demes at or above `K` are critical (`_classify_critical`); each leap takes the ε-bounded step over the other channels and an exponential time for the critical ones, and fires one critical event exactly, through the exact mode's `_event_in_deme`, when its time comes first (`_tau_leaps_critical`).

- `"fixed"`, with `tau_deaths="poisson"`: the earlier scheme, which splits each record interval into equal sub-steps bounded by the largest per-cell rate and caps Poisson deaths at the clone size. It is not a leap condition and is biased near `K`; it stays for tumours grown with it, which keep it when pickled and grown further.

Against the exact mode, on small systems: `"cgp_critical"` reproduces a deme at capacity, neutral fixation and 5,000-cell tumours, but in a paper tumour most demes are full, so it costs more than the exact mode; `"cgp"` reproduces neutral fixation and 5,000-cell tumours and holds a full deme about 1% below the exact occupancy, at about five times the cost of the fixed step at ε 0.01.
While every clone holds fewer than about 1/ε cells, the rule allows one expected event per leap, so growth from a few founder cells is slower than in the exact mode: from one cell, 8 generations of well-mixed growth end with about 40% fewer cells on average (27 against 45 over 200 seeds; the fixed step, about 24% fewer), with the right extinction probability.
Once any clone is large, every leap is short and new single-cell mutants are resolved.
The paper runs the defaults, `"cgp"` at ε 0.01 with binomial deaths, set explicitly in `validation/paper/regime.py`; the critical-channel rule stays available for runs that need exact-quality crowding at a deme and can afford its cost.

**Births and free slots.**
A deme holds at most `ρK` cells; a birth that reaches a full deme displaces a resident normal cell if there is one, and is lost otherwise.
In the exact mode births arrive one at a time, so the first ones take the free slots.
In a leap, the births routed to a deme (`_tau_substep`; the cap and eviction are `_slot_cap_of`, `_evict` and `_place_one` in `tissue.py`) arrive in the same interval; conditional on their numbers, the births of independent Poisson processes arrive in a uniformly random order, so the accepted ones are a multivariate hypergeometric draw over the candidates, and the leftovers compete in the same way for the slots that eviction frees.
This is the conditional law of the exact process, not a separate model rule.

**Stopping at a size.**
`grow(..., target_cancer=)` (and `regime.grow`, which passes it) stops after the first event (exact) or leap (tau) at which the cancer population reaches the target, and advances the clock only by the time simulated.

## The genome and the mutation process

A cell's genome (`tumor/components/cell.py`) is `n_segments` segments, each carried on two homologs (`"p"`, `"m"`), each homolog a list of copies, each copy a bitset of point mutations over the segment's genes.
Copy number is the number of copies; an SNV is a set bit on one copy.
A genome summary caches what fitness reads (copy number per segment, ploidy, mutated drivers and trait genes per segment).
Daughters share their parent's genome until one of them mutates (copy-on-write per segment).

`CancerCell.mutate` makes one event per mutating division:

- with probability `wgd_rate`, a whole-genome duplication;
- otherwise a copy-number event with weight `cnv_prob`: one copy of one segment, chosen in proportion to the copies, is duplicated (probability `amp_prob`) or deleted;
- or an SNV event with weight `snv_prob`: Poisson(`n_snvs_per_allele`) new mutations on every copy of every segment.
  An SNV event whose draws are all 0 leaves the genome unchanged.

A daughter that exceeds the viability limits (ploidy, copies of a segment, nullisomic segments, driver load; `Selection.update_viability`) is never born.

Before growth, `tumor/germline.py` writes the patient's germline variants into every cell and the founder's truncal mutations into the founder.

## Fitness and the traits

`tumor/components/selection.py` draws the gene layout once per configuration (from `layout_seed`, so a cohort shares it): which genes are oncogenes and tumour suppressors, and which carry each trait (dispersal, immune resistance, treatment resistance, breach, stromal survival, metastatic survival, drug tolerance).
The division rate follows CINner's dosage model: each oncogene and suppressor copy contributes a factor relative to the diploid wild type, with the exponent scaled by ploidy, and the product multiplies the baseline division rate, capped at `max_birth_rate` (`update_division_rate`).
Each trait is a value in [0, 1] computed from the copies and mutations of its genes; each costs a fraction of division wherever it is carried and pays off only in its compartment (`proliferation_cost`, and the hazards in `_death_rate`).
`selection_mode="arm"` replaces the gene model with per-arm copy-number fitness (the real-genome mode, `inference/genome.py`, `inference/realgenome.py`).
An optional epistasis network (`tumor/components/epistasis.py`) adds pairwise interactions and dependency gating between planted events.

## The tissue, the compartments and the metastasis

The tissue (`tumor/models/tissue.py`, `tumor/geometry.py`) is a lattice of demes, each a column of tissue with a carrying capacity K.
The ductal field places small glands (an epithelial ring around a lumen) in stroma seeded at moderate density, with their own K for ducts and stroma.
Residents (epithelium, stroma, immune, host) are seeded once and never divide.

- **Crowding.** A deme has `floor(crowding_overfill * K)` slots.
  A birth that finds no free slot displaces an evictable resident (epithelium, stroma, host; never immune cells) or fails.
  A cancer cell dies at its baseline rate while the deme holds at most K cells and at `crowding_d1` above it, the same for every clone in the deme (demon's two-valued step), so clones in a full deme keep competing through their division rates.
  Residents add a death hazard at the fixed reference rate `resident_pressure_ref`.
- **Dispersal.** A division sends a daughter out with probability `dispersal_rate`, independently of whether it mutates (demon's rule).
  The daughter goes to a random von Neumann neighbour, hops to another gland's lumen (`cross_gland_kappa`, the intraductal route), or, from a primary stroma deme, attempts to seed the metastasis.
- **Compartments.** Leaving a duct for the stroma crosses the basement membrane: it succeeds with probability equal to the breach trait of the clone, and a daughter that fails stays in the duct (`breach_gated_invasion`).
  The stroma adds a death hazard attenuated by stromal survival (`stromal_hazard`), the epithelium one attenuated by breach (`epithelial_barrier`), and immune cells one attenuated by immune resistance.
- **Origin confinement.** The founding acinus holds its cells in for `generations` generations after it is half full, so an early alteration can become truncal.
- **Metastasis** (`tumor/models/metastasis.py`). An optional second lattice of host parenchyma with one vessel deme.
  A seeding cell survives transit with probability `f + (1 - f) m`, where `m` is the metastatic-survival trait of the clone and `f` the floor `met_transit_floor`, and the host adds a death hazard attenuated by the same trait.

Treatment (`tumor/models/therapy.py`, `iscc.treatment`) acts during growth through the death rate (chemotherapy, targeted therapy), immune resistance (immunotherapy) or by removing cells (surgery), with either label-based or mechanistic responses.

## Passenger folding and the coalescent re-creation

With `coarsen_passengers`, a mutating division whose daughter differs from its parent only at neutral positions is not registered as a new genotype.
It joins its parent's count, and its passenger count is added to the parent's folded burden (`_pass_load`).
The dynamics are unchanged in distribution, since a passenger daughter has its parent's rates, and the number of genotypes then tracks clones rather than cells.

At materialisation, `_reconstruct_passengers` (`tumor/models/materialise.py`) re-creates the folded passengers per sampled cell along the genealogy.
A share of each clone's burden (`F_STEM_DEFAULT`) is placed on the clone's founding edge and inherited by its whole sampled subtree; the rest is private to single cells.
Each sampled clone gets at least `STEM_FLOOR` stem markers, so every clone split among sampled cells is marked.
The draws come from their own stream, so the re-created passengers depend only on the tumour and the sample.

## Materialisation

`make_cell_data` (`tumor/models/materialise.py`) turns the counts into `cell_data`, a dict of per-cell tables indexed `C0…Cn`: SNVs and copy number per gene (total and per homolog), expression (total and allele-resolved), RNA VAF, coordinates, cell type, deme, gland, compartment, WGD state, program activities and microenvironment levels.

- **Plan** (`_materialize_plan`).
  Without a cap every cell is built.
  With `max_cells`, each (deme, genotype) bucket is subsampled binomially, which preserves spatial, clonal and cell-type proportions.
  `region` restricts to a set of demes at full density, and `depth_frac` keeps a fraction of each deme's column (a section).
- **Layers.** `layers` selects which cells × genes tables to build (or names an assay, standing for the tables it reads).
- **Chunks.** The tables, the program multiplier included, are built in chunks of cells, so memory does not grow with the number of cells beyond the tables returned.
- **Expression.** A cell's expression is its type's baseline, scaled by gene programs (`tumor/programs.py`: program activities driven by genotype and niche, allele-specific dosage, SNV effects) and by the microenvironment (a hypoxia field from oxygen diffusion and a cell-cell communication field; `_o2_field`, `_cci_field`).
  Both layers are readouts: growth never reads them.

## The assays and their biases

Every assay subclasses `data/assay.py` and draws its technical noise from a seeded batch realisation (`data/batch.py`), whose hyper-parameters are the ones the estimators fit.

- **Bulk and single-cell DNA** (`data/dna.py`, `bulkDNA`, `scDNA`).
  Coverage is proportional to copy number times a per-locus efficiency, drawn compositionally (Dirichlet-multinomial) or per locus (negative binomial).
  The biases are GC and capture efficiency per breadth (WGS, WES, panel), a per-batch depth shift, sequencing error and FFPE damage.
  Single-cell DNA adds lumpy amplification, allelic dropout and doublets.
  Bulk DNA reports the pool's true purity.
- **Single-cell RNA** (`data/rna.py`, `scRNA`).
  Per-gene batch factors, per-cell library sizes, negative-binomial (or Dirichlet-multinomial) counts, ambient soup, doublets and optional dropout, by protocol.
- **Visium** (`data/visium.py`).
  Hexagonal spots aggregate the section's cells, lateral diffusion mixes neighbouring spots, a spatially correlated batch field and spot swapping (`data/visium_background.py`, SpotClean's bleed model) add spatial artefacts, and spots outside the tissue receive bled counts.
- **Imaging** (`data/imaging.py`, `scSpatial`).
  A targeted panel measured cell by cell, reusing the scRNA count machinery.
  Cell positions are resolved only to the deme and scattered at random within it, so the assay does not model structure below the deme.
- **Reads** (`data/reads/`).
  FASTQ and BAM emission from the count matrices, through external simulators (DWGSIM, ART) and a synthetic or real reference.

## The estimators

`data/estimate.py` (`estimate_rna`), `data/estimate_dna.py` (`estimate_dna`) and `data/estimate_visium.py` (`estimate_visium`) fit each assay's batch hyper-parameters from real data by moments and one-dimensional likelihoods, so the technical magnitudes are learned rather than guessed.
They are simulation-free.
Inference of the biology uses `iscc.inference`: approximate Bayesian computation over the tumour parameters (`abc.py`, `tumor.py`, `summaries.py`) and the evolutionary-mode indices (`indices.py`).

## The human genome layer

Growth runs on a coarse genome; `iscc.data.human` lifts it to the human genome when cells are measured.
`arm_genome` (`data/human/genome.py`) builds a growth genome with one segment per autosomal arm and drivers placed from Cancer Gene Census content.
`HumanGenomeLayer` (`data/human/layer.py`) maps each growth gene to a stretch of its arm, so every human gene there inherits that growth gene's copy number per homolog, and gives the patient a germline of real heterozygous SNPs with per-haplotype allele counts.
`HumanExpression` (`data/human/expression.py`) gives each cell type a reference-atlas baseline and Hallmark programmes on the human genes, reading only what genotype and niche do off the growth genes.
The assays take the lift through `gene_lift`, and `data/human/writers.py` writes tool-ready outputs.

## Integrations

`iscc.integrations` exports the substrate other tools read: the clone tree (Newick), AnnData with spatial coordinates, clone calls from clades (`clones.py`), multi-region trees (`multiregion.py`), lineage-against-space decompositions (`lineage.py`, `petracer.py`), progression-model inputs (`progression.py`) and ligand-receptor ground truth (`cci.py`).
