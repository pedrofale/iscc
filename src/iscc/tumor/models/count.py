"""
Genotype-level (count-based) tumor engine.

Instead of one Python object per cell, the population is represented as **per-deme genotype
counts** over a shared **genotype registry**. A "cell" is a unit of count in a (genotype,
deme) bucket; cells of the same genotype in the same deme are statistically exchangeable, so
counting them loses no information.

Each birth/death/mutation/dispersal event draws "how many of the count" rather than "which
cell", so it is O(1) integer arithmetic on counts. A genotype is carried by a representative cell
(reusing its genome / mutate / get_snvs / get_cnvs / get_exp and CINner fitness).

Normal cells (epithelial/stromal/immune) are seeded as **static** genotype counts (they don't
divide or mutate); they provide tissue structure, cell-type labels, and the local immune density
that the cancer death rate reads. Treatment runs on this engine (see grow/_apply_treatment) and
immune killing is additive contact pressure (see _death_rate); the immune compartment itself is
static (recruitment/migration are future work).
"""

import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from ..components.selection import Selection
from ..components.epistasis import bits_to_events
from ..components.cell import CancerCell, EpithelialCell, StromalCell, ImmuneCell, HostCell
from ..germline import draw_germline_sites, apply_germline, seed_founder_mutations
from ..programs import ProgramModel
from ...constants import normal_names
from .tissue import TissueMixin
from .metastasis import MetastasisMixin
from .therapy import TherapyMixin
from .materialise import (  # noqa: F401  (re-exported: the module-level names tests and callers use)
    MaterialiseMixin, CELL_LAYERS, ASSAY_LAYERS, PASSENGER_RNG_OFFSET, _STREAM_GROW, _STREAM_PASSENGERS,
    _STREAM_DEPTH, _STREAM_SUBSAMPLE)
from .plotting import PlottingMixin
from .params import BREAST_SUBTYPES, ModelBlocks, RunSettings, resolve_parameters  # noqa: F401


CELLTYPES = ["cancer", "epithelial", "stromal", "immune"]

# --- Coalescent passenger overlay -------------------------------------------------------------------

class GenotypeTumor(TissueMixin, MetastasisMixin, TherapyMixin, MaterialiseMixin, PlottingMixin):
    """Genotype-level (count-based) tumour engine — the default, scalable iscc growth model.

    The tumour grows on a grid of demes (lattice patches); each deme is represented as
    **genotype counts** over a shared registry rather than one Python object per cell, so it
    is spatially explicit at the deme level while every birth / death / mutation / dispersal
    event is O(1) integer arithmetic on counts. It is the class most users should reach for.
    Typical flow: construct, call ``grow`` to run the dynamics, then read the per-cell
    ground truth off ``make_cell_data`` and quality-check it with ``diagnose``.

    The constructor takes **either** a single ``config`` YAML path (the usual entry point;
    see ``notebooks/example_config.yaml``) **or** the individual nested-dict parameter
    blocks below. Only the top-level blocks are summarised here — every knob, with its
    default, valid range and effect, is covered in the parameter documentation.

    Parameters
    ----------
    config : str or pathlib.Path, optional
        Path to a YAML config that defines all of the parameter blocks below (keys
        ``genome_params``, ``selection_params``, ``deme_params``, ``spatial_params``,
        ``cell_params`` and any optional layers). When given it **overrides** the
        individual arguments. This is the recommended entry point.
    seed : int, optional
        EVOLUTION seed (default 42): drives the per-run stochastic dynamics and the
        spatial seeding. Two runs differing only in ``seed`` are two patients with the
        same genome landscape but different evolution.
    genome_params : dict, optional
        Genome geometry — ``n_segments`` × ``segment_size`` set the gene count
        (default 5 × 200 = 1000 genes).
    selection_params : dict, optional
        The CINner fitness model — driver / dispersal / resistance proportions and
        effect sizes, viability limits, and the optional ``epistasis_params`` /
        ``dependency_params`` network layers.
    cancer_cell_params : dict, optional
        Cancer-cell dynamics — ``division_rate``, ``death_rate``, ``mutation_rate``,
        ``n_snvs_per_allele``, ``snv_prob`` / ``cnv_prob``, ``amp_prob``, ``wgd_rate``.
    deme_params : dict, optional
        Per-deme demography — ``initial_cancer_cells``, ``carrying_capacity``,
        ``maximum_death_rate``, and the crowding law's ``crowding_overfill`` (the slot cap as a
        multiple of K, default 1.5) and ``crowding_d1`` (the death rate above K, default
        ``maximum_death_rate``). Crowding is demon's two-valued death step under a structural slot
        cap, and division is demon's invasion rule: a division attempts a mutation with probability
        ``mutation_rate`` and, independently, one of its two daughters disperses with probability
        ``dispersal_rate``.
    spatial_params : dict, optional
        Grid and tissue geometry, given as keys of this dict (not as individual
        constructor arguments): ``grid_size``, ``dispersal_rate``, ``structure_radius`` /
        ``n_structures`` (glandular substrate), and the optional compartment-selection
        hazards (``epithelial_barrier``, ``stromal_hazard``).
    epithelial_cell_params, stromal_cell_params, immune_cell_params, host_cell_params : dict, optional
        Seeding parameters for each non-cancer (static) cell type — the tissue
        structure, cell-type labels and local immune density that the cancer death
        rate reads.
    genome_mode : {"abstract", "real"}, optional
        ``"abstract"`` (default) uses a random gene-driver layout; ``"real"`` wires the
        genome from human chromosome-arm data and requires ``genome_spec``.
    genome_spec : GenomeSpec, optional
        Real-genome specification (required when ``genome_mode="real"``).
    update_mode : {"exact", "tau"}, optional
        ``"exact"`` (default) is the reference one-event-per-update Gillespie engine;
        ``"tau"`` is tau-leaping (``tau`` sets the generation length), whose cost scales
        with number-of-clones × number-of-generations instead of number-of-cells.
    tau : float, optional
        Tau-leaping generation length, used only when ``update_mode="tau"`` (default 1.0).
    snapshot_every : int, optional
        Under tau-leaping, record a full per-clone snapshot every ``k`` generations
        (default 1) so the Muller / grid plots keep working.
    spatial_snapshot_every : int, optional
        Also record the spatial state of the tumour — every genotype's count in every deme — at
        every ``k``-th trace snapshot (default 0, off). Read the record back with
        [`spatial_snapshots`][iscc.tumor.GenotypeTumor.spatial_snapshots], and each clone's birth
        with [`genotype_births`][iscc.tumor.GenotypeTumor.genotype_births]. Recording draws no
        random numbers, so the tumour grows identically with it on or off. Under the exact engine
        a trace snapshot is taken after every single event, so choose a large ``k`` there.
    spatial_snapshot_budget : int, optional
        Upper bound on the number of stored (deme, genotype) entries across all spatial
        snapshots (default 5,000,000, about 60 MB). When it is exceeded, every other snapshot is
        dropped and the interval doubles, so the record stays evenly spaced in time.
    coarsen_passengers : bool, optional
        Fold a division that adds only neutral passenger mutations into its parent clone instead
        of minting a new genotype (default False). The growth dynamics are unchanged in
        distribution, and the cost of a run tracks the number of clones rather than of cells;
        the passengers are re-emitted per cell when cell data are materialised. Forced off when
        epistasis is on.
    max_cells : int, optional
        Cap on the number of cells materialised as cell data (default None, no cap). Above it, a
        representative subsample that preserves spatial, clonal and cell-type proportions is
        materialised instead.
    microenv_params : dict, optional
        Microenvironment layer (hypoxia field + cell-cell communication). Off by
        default; a pure readout that never changes growth.
    expression_params : dict, optional
        Gene-program expression layer (``program_params``, ``activity_params``,
        ``coupling_params``, ``dosage_params``, ``snv_effect_params``). Off by default;
        a readout only. ``cell_of_origin`` (a normal cell type, e.g. ``"epithelial"``)
        gives cancer cells that type's baseline expression instead of an unrelated one.
    layout_seed : int, optional
        GENOME-LAYOUT seed (config-determined, shared across runs of the same config):
        fixes driver identities and baseline expression so a cohort is comparable by
        construction. Defaults to the fixed ``DEFAULT_LAYOUT_SEED``.
    germline_params : dict, optional
        Inherited (germline) variants — ``het_frac``, ``hom_frac`` and an optional ``seed``.
        Off by default; when given, neutral loci are seeded into every cell of the patient
        (tumour and normal alike) before growth, giving a DNA assay the constant background
        it reads purity and allelic imbalance against.
    founder_mutations : dict, optional
        The mutations the founder cell already carried when it transformed — ``n_passenger``
        (the clock-like burden the normal lineage accumulated first), ``n_driver`` (the
        transforming hits) and an optional ``seed``. Off by default; when given, they are
        written into the founder genome before growth, so every cancer cell inherits them and
        the tumour has the clonal peak real sequencing shows.
    """

    # Tau-leaping options (see ``_init_tau``). Class-level defaults, so tumours pickled before the
    # options existed read today's behaviour.
    _tau_deaths = "poisson"
    _tau_step = "fixed"
    _tau_epsilon = 0.03
    # Demes whose cells changed since the last reset; recorded by _add/_remove only while a
    # tau_step "cgp_critical" generation runs (None otherwise: no bookkeeping).
    _dirty = None

    def __init__(self, config=None, seed=42, genome_params=None, selection_params=None,
                 cancer_cell_params=None, deme_params=None, spatial_params=None,
                 epithelial_cell_params=None, stromal_cell_params=None, immune_cell_params=None,
                 host_cell_params=None,
                 genome_mode="abstract", genome_spec=None,
                 update_mode="exact", tau=1.0, snapshot_every=1, microenv_params=None,
                 layout_seed=None, expression_params=None, coarsen_passengers=False,
                 max_cells=None, germline_params=None, founder_mutations=None,
                 spatial_snapshot_every=0, spatial_snapshot_budget=5_000_000):
        r, p = resolve_parameters(
            config, genome_mode, genome_spec,
            RunSettings(update_mode=update_mode, tau=tau, snapshot_every=snapshot_every,
                        spatial_snapshot_every=spatial_snapshot_every,
                        spatial_snapshot_budget=spatial_snapshot_budget,
                        coarsen_passengers=coarsen_passengers, max_cells=max_cells,
                        germline_params=germline_params, founder_mutations=founder_mutations,
                        layout_seed=layout_seed),
            ModelBlocks(genome=genome_params, selection=selection_params, cancer=cancer_cell_params,
                        deme=deme_params, spatial=spatial_params,
                        epithelial=epithelial_cell_params, stromal=stromal_cell_params,
                        immune=immune_cell_params, host=host_cell_params,
                        microenv=microenv_params, expression=expression_params))
        # The order of these steps is the order of every random draw at construction (the layout
        # stream, the seeding stream): keep it.
        self._init_run(seed, r, genome_mode, genome_spec, p)
        self._init_genome(p)
        self._init_microenv_genes()
        self._init_cell_of_origin()
        self._init_registry()
        self._init_therapy_state(p)
        self._init_compartment_hazards(p)
        self._init_founder(seed, p)
        self._init_crowding(p)
        self._init_tau(p)
        self._init_field(p)
        self._init_met(p)
        self._init_grid(p)

    # --- construction ---------------------------------------------------------
    def _init_run(self, seed, r, genome_mode, genome_spec, p):
        """The run's seeds, clocks, records and the settings resolved from the constructor."""
        self.seed = seed
        # EVOLUTION rng (per-run: spatial seeding; grow() draws its own fresh default_rng(seed+step)).
        self.rng = np.random.default_rng(seed)
        # GENOME LAYOUT rng (config-determined, SHARED across same-config runs): the gene-role
        # layout (Selection) and shared per-cell-type baseline expression. Defaulting layout_seed to
        # DEFAULT_LAYOUT_SEED makes any two same-config runs share their driver identities by
        # construction (comparability by default) — recurrence / cohort analysis is meaningful — while
        # they still differ in evolution.
        self.layout_seed = r.layout_seed
        self.layout_rng = np.random.default_rng(self.layout_seed)
        self.type = "genotype"

        # Update mode. "exact" = the reference one-birth/death-per-update Gillespie engine.
        # "tau" = r.tau-leaping: advance ALL clones once per
        # discrete generation of length `tau` by drawing Poisson(rate*count*tau) births/deaths per
        # (deme, genotype), so wall-time scales with #clones x #generations rather than #cells.
        # `snapshot_every` records a full per-clone count snapshot every k generations (so
        # plot_muller/plot_grid keep working, on a real-time x-axis -- see self.trace_times).
        self.update_mode = r.update_mode
        self.tau = r.tau
        self.snapshot_every = r.snapshot_every
        self.time = 0.0
        self.trace_times = []
        # Spatiotemporal record. Every genotype gets its birth (trace index, step, time, deme) in
        # _register / _add; when spatial_snapshot_every > 0, _spatial_snapshot also stores every
        # deme's genotype counts at every k-th trace snapshot as compact int32 arrays keyed by the
        # genotype's creation ordinal. Neither draws a random number, so growth is the same with the
        # record on or off. The traces list and the step counter are created here, not
        # beside the deme grid below, because the founder is registered before the grid exists.
        self.spatial_snapshot_every = int(r.spatial_snapshot_every)
        self.spatial_snapshot_budget = int(r.spatial_snapshot_budget)
        self.spatial_history = []
        self._spatial_entries = 0
        self.traces = []
        self.step = 0
        self._clock = 0     # the current exact-engine event or tau generation, for birth records

        # Passenger coarsening DECOUPLES growth cost from the distinct-genotype count. Under a high
        # mutation rate almost every division fixes a NEUTRAL (passenger) SNV and so, in the exact
        # model, spawns a brand-new genotype whose fitness is identical to its parent's — making
        # #genotypes track #cells and the whole growth loop super-linear. With ``coarsen_passengers`` on, a mutating
        # division whose daughter is genome_summary-identical to the parent (a pure passenger: no
        # driver hit, no copy-number change, no WGD) is NOT registered as a new genotype; it grows the
        # parent clone in place and its neutral burden is tallied in ``self._pass_load`` for lazy
        # reconstruction at materialisation. Only FITNESS/COPY-NUMBER-CHANGING events (driver SNVs,
        # CNVs, WGD) still create genotypes, so #genotypes tracks the number of distinct *clones*, not
        # cells. The dynamics are unchanged in distribution — a passenger daughter has identical
        # division/death/dispersal rates, so folding it into the parent leaves every future event rate
        # the same (selection fully intact). Passengers are re-emitted per sampled cell in
        # ``make_cell_data``. Force-off when epistasis is on (an SNV anywhere in an event module can fire an event, so
        # "genome_summary unchanged" no longer implies "dynamically neutral").
        self.coarsen_passengers = bool(r.coarsen_passengers)

        # Assay-memory cap. ``make_cell_data`` materialises one row per cell
        # across ~a dozen dense (n_cells × n_genes) frames; at cm-scale (millions of cells) that is
        # tens of GB. ``max_cells`` bounds it: when the tumour exceeds the cap, a REPRESENTATIVE
        # subsample (Binomial(count, cap/total) per (deme, genotype) bucket, so spatial + clonal +
        # cell-type proportions are preserved) is materialised instead of every cell — a biopsy of one
        # tissue, which is exactly what an assay samples. ``None`` (default) materialises all cells.
        self.max_cells = r.max_cells

        # Germline (inherited) variants: the patient's constant genetic background, carried by every
        # cell — tumour and normal alike — and so the purity / cancer-cell-fraction anchor a DNA
        # analysis calibrates against. A dict of the ``draw_germline_sites`` knobs, e.g.
        # ``{"het_frac": 5e-3, "hom_frac": 0.33, "seed": 7}``; ``seed`` defaults to this run's seed,
        # since a germline background is private to one individual. ``None`` (the default) means no
        # germline layer at all. The variants are drawn from a dedicated rng and written into the
        # founder genomes before any growth rng is touched, so a tumour grows the same with the layer
        # on or off at a given seed.
        self.germline_params = r.germline_params

        # Founder (truncal) mutations: what the transformed cell already carried at t=0 — the
        # clock-like burden of its normal lineage plus the driver hits that transformed it. A dict of
        # the ``seed_founder_mutations`` knobs, e.g. ``{"n_passenger": 60, "n_driver": 2, "seed": 7}``;
        # ``seed`` defaults to this run's seed. ``None`` (the default) leaves the founder genome blank,
        # which means NOTHING in the tumour can ever be clonal — the most recent common ancestor of
        # every cancer cell is the founder, so an empty founder genome is an empty clonal layer.
        self.founder_mutations = r.founder_mutations

        self.genome_mode = genome_mode
        self.genome_spec = genome_spec
        # Microenvironment-driven expression: optional, off when None. When enabled, a per-deme
        # expression modifier (hypoxia field + cell-cell communication) is applied at
        # materialisation only; GROWTH is untouched, so a tumour grows the same with the layer on or
        # off at the same seed (the modifier draws from a dedicated rng). The microenvironment does
        # not feed back into fitness.
        self.microenv_params = p.microenv
        # Gene-program expression: optional, off when None. GROWTH never reads it, so a tumour grows
        # the same with the program layer on or off at a given seed. Programs are a READOUT: the
        # genotype and the niche drive them, they never feed back into fitness. Built after
        # `self.selection` below, since the model needs the gene count. See `iscc.tumor.programs`.
        self.expression_params = p.expression
        self.programs = None
        self.subtype = p.subtype

    def _init_tau(self, p):
        """The tau-leaping options, read from ``deme_params``.

        ``tau_deaths``: ``"poisson"`` (default) draws Poisson(d c dt) deaths per (deme, clone), capped
        at c; ``"binomial"`` draws Binomial(c, min(1, d dt)), binomial tau-leaping for a first-order
        channel (Tian and Burrage 2004; Chatterjee et al. 2005), which cannot overshoot c.

        ``tau_step``: ``"fixed"`` (default) splits a generation into ceil(r_max tau / 0.34) equal
        sub-steps; ``"cgp"`` chooses each sub-step by the epsilon-bounded rule of Cao, Gillespie and
        Petzold (2006) with ``tau_epsilon`` (default 0.03; ``_cgp_dt``); ``"cgp_critical"`` is
        tau-leaping with critical reactions (Cao, Gillespie and Petzold 2005): the channels of demes at
        or above K fire one event at a time and the rest are leaped with the "cgp" step
        (``_tau_leaps_critical``). It needs a carrying capacity."""
        self._tau_deaths = str(p.deme.get("tau_deaths", "poisson"))
        if self._tau_deaths not in ("poisson", "binomial"):
            raise ValueError(f"tau_deaths must be 'poisson' or 'binomial', got {self._tau_deaths!r}")
        self._tau_step = str(p.deme.get("tau_step", "fixed"))
        if self._tau_step not in ("fixed", "cgp", "cgp_critical"):
            raise ValueError(f"tau_step must be 'fixed', 'cgp' or 'cgp_critical', got {self._tau_step!r}")
        if self._tau_step == "cgp_critical" and not self._crowding:
            raise ValueError("tau_step 'cgp_critical' needs a carrying capacity")
        self._tau_epsilon = float(p.deme.get("tau_epsilon", 0.03))

    def _init_genome(self, p):
        """The genome layout (shared by every run of a config): segments, the gene roles, the
        passenger masks, the program layer and the per-cell-type baseline expression."""
        self.genome_params = p.genome
        self.n_segments = p.genome["n_segments"]
        self.segment_size = p.genome.get("segment_size", 1000)
        # Per-segment sizes (real-genome mode: proportional to chromosome-arm length); falls
        # back to a uniform scalar in the abstract mode.
        self.segment_sizes = p.genome.get("segment_sizes")
        self._normal_params = {
            "epithelial": (EpithelialCell, p.epithelial or {}),
            "stromal": (StromalCell, p.stromal or {}),
            "immune": (ImmuneCell, p.immune or {}),
            "host": (HostCell, p.host or {}),
        }

        # ``layout_seed`` is handed over (as well as ``layout_rng``) so the optional epistasis network
        # can draw from its own layout SUB-STREAM — part of the shared landscape, but independent of
        # the gene-role layout drawn from layout_rng. See constants.py.
        self.selection = Selection(n_segments=self.n_segments, segment_size=self.segment_size,
                                   segment_sizes=self.segment_sizes, rng=self.layout_rng,
                                   layout_seed=self.layout_seed, **p.selection)
        self.n_genes = self.selection.n_genes
        self._cancer_params = p.cancer

        # Passenger coarsening state (see the flag docstring above). Epistasis makes any SNV in an
        # event module fitness-relevant, so coarsening is force-disabled there.
        if self.coarsen_passengers and self.selection.epistasis is not None:
            self.coarsen_passengers = False
        # Per-segment mask of NEUTRAL (passenger) gene positions: everything NOT in any selection
        # axis (drivers = onc∪tsg, dispersal, immune/treatment resistance, breach, stromal/met
        # survival). A mutating division that touches only these leaves genome_summary untouched, so
        # it is a pure passenger. Used to draw reconstructed passenger SNVs at materialisation.
        sel = self.selection
        self._neutral_pos = []
        self._func_bits = []           # per-segment boolean mask of FITNESS-relevant gene positions
        for seg in range(self.n_segments):
            func = np.zeros(int(sel.segment_sizes[seg]), dtype=bool)
            for idx in (sel.drivers[seg], sel.dispersal[seg], sel.immune_resistance[seg],
                        sel.treatment_resistance[seg], sel.breach[seg], sel.stromal_survival[seg],
                        sel.met_survival[seg], sel.drug_tolerance[seg]):
                if len(idx):
                    func[np.asarray(idx, dtype=int)] = True
            self._func_bits.append(func)
            self._neutral_pos.append(np.flatnonzero(~func))
        # Global (genome-wide) neutral position ids and per-clone folded-passenger burden.
        self._neutral_gene_ids = np.concatenate(
            [self.selection._seg_offsets[s] + self._neutral_pos[s] for s in range(self.n_segments)]
        ) if self.n_segments else np.array([], dtype=int)
        # gene id -> segment index (for passenger reconstruction: a neutral site's VAF is 1/cn of the
        # segment it sits in).
        self._gene_segment = (np.concatenate(
            [np.full(int(sel.segment_sizes[s]), s, dtype=int) for s in range(self.n_segments)])
            if self.n_segments else np.array([], dtype=int))
        self._pass_load = Counter()     # gid -> cumulative passenger SNVs folded into that clone
        self._n_folded = 0              # diagnostics: passenger divisions folded this run

        # Program layer. The dictionary (gene->program map, `loading`, regulators, `s_g`) is part
        # of the SHARED landscape and so draws from `layout_seed`'s program sub-stream — two patients
        # with the same config get the SAME programs, exactly as they already get the same oncogenes.
        # The per-cell `z` and each SNV's functional class are per-RUN events and draw from `seed`.
        if self.expression_params is not None:
            self.programs = ProgramModel(
                n_genes=self.n_genes, segment_sizes=self.selection.segment_sizes,
                layout_seed=self.layout_seed, run_seed=self.seed,
                expression_params=self.expression_params)

        # per-cell-type baseline expression (part of the SHARED landscape -> layout_rng, so a shared
        # cell state has the same baseline profile across patients; see make_celltype_exps note in tumor.py)
        self.celltype_exps = {}
        for ct in CELLTYPES:
            exp = self.layout_rng.beta(0.1, 1.0, size=self.n_genes)
            exp[self.selection.get_tsgs()] = 0.8
            exp[self.selection.get_oncogenes()] = 0.01
            self.celltype_exps[ct] = exp

    def _init_cell_of_origin(self):
        """Give cancer cells the baseline of their cell of origin, when one is configured."""
        # Cell of origin: a cancer cell is a transformed cell of one normal type, so with
        # `expression_params["cell_of_origin"]` it shares that type's baseline and differs from it only
        # by what evolution did (copy number, SNVs, programs). Without it, the cancer baseline is an
        # unrelated draw, and malignant cells separate from every normal type on baseline alone. The
        # cancer draw above is still made, and the gene-set designations above still read the drawn
        # baselines, so every layout draw is the same either way; only the materialised cancer
        # expression (and, under targeted therapy, the target-expression check that reads it) moves.
        origin = (self.expression_params or {}).get("cell_of_origin")
        if origin is not None:
            if origin not in CELLTYPES or origin == "cancer":
                raise ValueError(f"cell_of_origin must be a normal cell type "
                                 f"{[c for c in CELLTYPES if c != 'cancer']}, got {origin!r}")
            self.celltype_exps["cancer"] = self.celltype_exps[origin].copy()

    def _init_registry(self):
        """The genotype registry and the per-genotype running state."""
        # genotype registry: id -> representative cell. Normals keyed by their type name.
        self.genotypes = {}
        self.genotypes_parents = {}
        self.genotypes_counts = Counter()
        self._next_ord = 0
        # Drug-induced resistance STATE: the (genotype x epistate)
        # sublineage registry. Maps a genotype id -> a SHARED dict {state_level -> gid} holding every
        # state-variant of the SAME genome; a state transition of g to level s' looks up the twin here
        # and mints it (via _state_twin) only once, reusing it across generations, so the state costs a
        # small multiplier on the genotype count rather than one new id per transitioning cell. Empty and
        # never touched unless the feature is on (all population happens in the gated transition path).
        self._state_families = {}
        # Running max of (division + dispersal) over cancer genotypes, maintained incrementally in
        # _register so the tau-leap substep count doesn't need an O(#genotypes) scan every generation
        # (only under coarsen_passengers, where #genotypes can still be large; otherwise the live-max
        # scan is used). Monotone upper bound -> conservative (never too-few substeps).
        self._max_div_disp = 0.0

    def _init_founder(self, seed, p):
        """Register the founder cancer genotype."""
        # founder cancer genotype
        founder = CancerCell(
            n_segments=self.n_segments, segment_size=self.segment_size,
            segment_sizes=self.segment_sizes, seed=seed,
            n_onc=len(self.selection.get_oncogenes()), n_tsg=len(self.selection.get_tsgs()),
            n_disp=len(self.selection.get_dispersal_genes()),
            n_ir=len(self.selection.get_immune_resistant()),
            n_tr=len(self.selection.get_treatment_resistant()),
            n_breach=len(self.selection.get_breach()),
            n_ss=len(self.selection.get_stromal_survival()),
            n_ms=len(self.selection.get_met_survival()),
            n_dt=len(self.selection.get_drug_tolerance()),
            **p.cancer,
        )
        founder.set_genotype_id()
        self._register(founder)
        self.founder_id = founder.genotype_id

    def _init_grid(self, p):
        """Build the deme grid and seed it: the tissue, the founder, immune cells, the met, the germline
        and founder mutations; then the initial event rates."""
        # Discrete-event annotation log (met seeding, resection, chemo windows) -> events.csv, used to
        # annotate the 2-band Muller. Each entry is a dict with at least step/time/event.
        self.events = []
        # Ground truth of each PARTIAL resection (Surgery(region=...)): what was removed, the margin
        # rim, and what was left behind. See `_resect_region`.
        self.resections = []
        self._has_host = False

        n_demes = self.grid_size * self.grid_size
        self.n_primary_demes = n_demes
        self.demes = [dict() for _ in range(n_demes)]
        # deme index -> its live CANCER cell count, present only while positive. Maintained by
        # _add/_remove (the only two routes cells take in or out of a deme) so a tau substep can
        # iterate the cancer-occupied demes instead of all grid_size**2 of them (_active_demes).
        self._deme_cancer_n = {}
        self.deme_coords = [(i // self.grid_size, i % self.grid_size) for i in range(n_demes)]
        # Per-deme carrying capacity: uniform = carrying_capacity by default; _seed_structure overwrites duct demes with
        # K_duct and stroma demes with K_stroma. Only consulted when crowding is on.
        self._deme_capacity = (np.full(n_demes, self.carrying_capacity, dtype=float)
                               if self._crowding else None)

        if self.structure_radius > 0:
            self._seed_structure()
        else:
            center = (self.grid_size // 2) * self.grid_size + (self.grid_size // 2)
            self._add(center, self.founder_id, self._n_founder)
            self._origin_demes = (center,)

        # Optional immune microenvironment: seed immune cells in every deme so that
        # cancer growing into them experiences local immune pressure (and so that
        # immunotherapy has a substrate to act on). The count scales with carrying
        # capacity (immune_density = immune cells per capacity unit) so the immune
        # fraction is not washed out once a deme fills with cancer. Static: immune cells
        # neither divide nor migrate.
        self.immune_density = p.spatial.get("immune_density", 0.0)
        n_immune = int(round(self.immune_density * self._cap))
        if n_immune > 0:
            imm = self._normal_genotype("immune")
            for i in range(n_demes):
                self._add(i, imm, n_immune)
        # Metastasis: seed the second (met) grid AFTER primary immune seeding, so immune stays
        # primary-only and the met demes are appended at the tail. A no-op when met is off.
        self._seed_met()
        # Whether any immune cells exist (static: only seeded here). When false, the immune-killing
        # term is always zero, so the per-genotype death-rate can skip the O(#genotypes-in-deme)
        # immune sum entirely — a big win for the clone-heavy tau-leaping path (see _immune_fraction).
        self._has_immune = "immune" in self.genotypes
        # Whether the metastatic deposit's host parenchyma exists (seeded iff met_grid_size>0). Same
        # fast-path role as _has_immune for the met host hazard / _host_fraction scan.
        self._has_host = "host" in self.genotypes
        # Whether the gland's resident compartments exist (seeded iff structure_radius>0). When a
        # compartment is absent its fraction is identically zero, so the matching death term can skip
        # the O(#genotypes-in-deme) composition scan — the same optimisation as _has_immune.
        self._has_epithelial = "epithelial" in self.genotypes
        self._has_stromal = "stromal" in self.genotypes

        # Germline layer. Applied HERE — after every normal genotype has been registered, so the
        # inherited variants reach the whole tissue and not just the tumour, and before the per-deme
        # event rates below, so nothing has to be recomputed. It draws from its own seeded rng and
        # never touches the growth stream, so growth is the same with the layer on or off.
        self.germline_sites = np.array([], dtype=int)
        self.germline_zygosity = np.array([], dtype="<U4")
        self.germline_homolog = np.array([], dtype="<U4")
        if self.germline_params:
            gp = dict(self.germline_params)
            gp.setdefault("seed", self.seed)
            apply_germline(self, *draw_germline_sites(self.selection, **gp))

        # Founder (truncal) mutations. Applied to the founder genotype AFTER it is registered and the
        # germline is in place (so the two layers can be kept off each other's positions) and BEFORE
        # any growth, which is what makes them clonal by descent rather than by reconstruction. Like
        # the germline it uses its own seeded rng, so the growth stream is the same with or without it.
        self.seeded_truncal_sites = np.array([], dtype=int)
        self.seeded_truncal_homolog = np.array([], dtype="<U4")
        self.seeded_truncal_is_driver = np.array([], dtype=bool)
        if self.founder_mutations:
            fm = dict(self.founder_mutations)
            fm.setdefault("seed", self.seed)
            seed_founder_mutations(self, **fm)

        self.deme_rates = np.array([self._deme_rate(i) for i in range(len(self.demes))], dtype=float)
        self.cell_data = None
        # True once `grow(materialize=False)` has advanced the counts past `cell_data`; cleared by
        # `make_cell_data`. Read with getattr so older pickled tumours without the flag still load.
        self._cell_data_stale = False


    # --- genotype registry ---------------------------------------------------
    def _is_viable(self, rep):
        """Whether a freshly mutated genotype satisfies the CINner viability limits
        (``max_ploidy`` / ``max_cn`` / ``max_nullisomy`` / ``max_mut_drivers``; see
        Selection.update_viability). A non-viable daughter is REJECTED AT BIRTH: the division is
        still consumed, but it yields no cell and no genotype is registered.

        WHY REJECT-AT-BIRTH. The alternative is to check lazily, when a cell is next *sampled* for
        an event. Both rules agree on the long-run dynamics: the division yields no surviving descendant either way, and the event is consumed
        either way, so the parent pays for it identically. They differ only in a transient: under
        the lazy rule the doomed cell occupies a carrying-capacity slot until it happens to be
        picked. That dwell time is not a modelled quantity -- it is set by the deme's total event
        rate, so it has no parameter behind it and no biological meaning -- and while it lasts the
        cell is counted by ``get_tumor_size`` and can be sampled into ``cell_data``, i.e. genomes
        that breach the configured limits leak into the emitted assay data. Rejecting at birth
        instead gives the invariant the limits are documented to provide: no cell breaching them
        ever exists. It is also the complete seam -- ``update_evolutionary_parameters`` (the only
        thing that computes viability) runs only inside ``Cell.mutate``, so a successful mutation is
        the ONLY way a genotype can become non-viable. Founders and normal cells are viable by
        construction.
        """
        return self.selection.update_viability(rep.genome_summary) != 0

    def _register(self, rep):
        rep.ord = self._next_ord
        self._next_ord += 1
        # Birth record: the first trace snapshot that can contain the genotype, the exact-engine
        # event or tau generation it was minted in, the time (tau only; the exact engine keeps
        # none) and, set by _add on first placement, the deme its first cell landed in. A child is
        # a copy of its parent's representative, so every field is overwritten here, never inherited.
        rep.birth_trace = len(self.traces)
        rep.birth_step = self._clock
        rep.birth_time = self.time if self.update_mode == "tau" else float("nan")
        rep.birth_deme = None
        self.genotypes[rep.genotype_id] = rep
        if self.coarsen_passengers and rep.type == "cancer":
            ep = rep.evolutionary_parameters
            dd = ep["division_rate"] + ep["dispersal_rate"]
            if dd > self._max_div_disp:
                self._max_div_disp = dd
        return rep.genotype_id

    def _normal_genotype(self, type_name):
        """Return the (cached) representative genotype id for a normal cell type."""
        if type_name in self.genotypes:
            return type_name
        cls, params = self._normal_params[type_name]
        rep = cls(n_segments=self.n_segments, segment_size=self.segment_size,
                  segment_sizes=self.segment_sizes, **params)
        rep.genotype_id = type_name
        self._register(rep)
        return type_name

    def _is_cancer(self, gid):
        return self.genotypes[gid].type == "cancer"

    # --- count bookkeeping ---------------------------------------------------
    def _add(self, deme_idx, gid, n):
        if self._dirty is not None:
            self._dirty.add(deme_idx)
        deme = self.demes[deme_idx]
        deme[gid] = deme.get(gid, 0) + n
        self.genotypes_counts[gid] += n
        rep = self.genotypes[gid]
        if rep.type == "cancer":
            self._deme_cancer_n[deme_idx] = self._deme_cancer_n.get(deme_idx, 0) + n
            if rep.birth_deme is None:
                rep.birth_deme = deme_idx

    def _remove(self, deme_idx, gid, n):
        if self._dirty is not None:
            self._dirty.add(deme_idx)
        deme = self.demes[deme_idx]
        deme[gid] -= n
        if deme[gid] <= 0:
            del deme[gid]
        self.genotypes_counts[gid] -= n
        if self.genotypes_counts[gid] <= 0:
            del self.genotypes_counts[gid]
        if self.genotypes[gid].type == "cancer":
            left = self._deme_cancer_n.get(deme_idx, 0) - n
            if left > 0:
                self._deme_cancer_n[deme_idx] = left
            else:
                self._deme_cancer_n.pop(deme_idx, None)

    # --- rates -----------------------------------------------------------------
    def _div_rate(self, gid, deme_idx, rep=None):
        """The clone's division rate here and now, including the drug-tolerance cost while a dose is
        active in this deme's compartment.

        Every OTHER trait cost is folded into ``division_rate`` once, at genotype level, because those
        traits are permanent. Tolerance is not: it is a drug-induced state, so its cost has to be
        applied at the point of use and gated to the treated compartment, exactly like the treatment's
        death hazard. ``_tx_div_mult`` is empty unless a dose is active AND some clone actually carries
        the trait, so with the feature off this is a dict lookup returning the untouched rate."""
        rep = rep if rep is not None else self.genotypes[gid]
        div = rep.evolutionary_parameters["division_rate"]
        if not self._tx_div_mult:
            return div
        m = self._tx_div_mult.get(gid)
        if m is None or not self._tx_applies(deme_idx):
            return div
        return div * m

    def _death_rate(self, gid, deme_idx, total=None, comp=None):
        """Cancer death rate = crowding-modulated baseline + local immune killing + treatment.

        Immune killing is additive contact pressure (more local immune cells -> higher death, attenuated by immune resistance),
        and active therapy adds a death hazard (chemo/targeted) or strips immune resistance
        (immunotherapy, via the per-step _tx_* overrides). ``total`` (the deme's cell count) may be
        passed in when the caller already computed it, so a per-deme loop doesn't recompute it once
        per genotype.

        ``comp`` is the deme's precomputed composition tuple from ``_deme_comp``
        ``(total, n_normal, n_immune, n_epithelial, n_stromal, n_host)``. When given,
        the immune / compartment fractions and ``n_normal`` are read from it in O(1) instead of
        re-scanning the deme per genotype; the numbers are identical to ``comp=None``, but the
        per-deme cost drops from O(#genotypes²) to O(#genotypes)."""
        rep = self.genotypes[gid]
        deme = self.demes[deme_idx]
        if comp is None and self._crowding:
            comp = self._deme_comp(deme)
        if comp is not None:
            total, n_normal_c, n_immune_c, n_epi_c, n_stro_c, n_host_c = comp
        elif total is None:
            total = sum(deme.values())
        base = rep.evolutionary_parameters["death_rate"]
        if self._crowding:
            steep = 1.0 + self.crowding_margin
            # Per-deme carrying capacity: duct demes cap at K_duct, stroma at K_stroma, otherwise the
            # scalar carrying_capacity.
            K = self.carrying_capacity if self._deme_capacity is None else self._deme_capacity[deme_idx]
            # demon's two-valued step (see __init__), the same for every clone in the deme: d0 at or
            # below K; above it a fixed d1 chosen to exceed any attainable division rate. The
            # structural slot cap (`_slot_cap_of`) bounds density; the step sets turnover.
            if total <= K:
                death = base
            else:
                d1 = (self.maximum_death_rate if self._crowding_d1 is None
                      else float(self._crowding_d1))
                death = max(base, d1)
            # Resident pressure: the fitness gate on entering normal-occupied tissue. The deme's
            # immortal normal cells add death at the fixed reference rate `_resident_ref`; it carries
            # no `div` either.
            n_normal = (n_normal_c if comp is not None
                        else sum(deme.get(nm, 0) for nm in normal_names))
            if n_normal:
                death += max(0.0, self._resident_ref - base) * steep * (n_normal / K)
        else:
            # well-mixed regime (carrying_capacity None/0): no crowding ceiling -> unbounded growth.
            death = base
        death = min(death, self.maximum_death_rate)

        # Immune resistance may be transiently stripped by immunotherapy; that override is gated to the
        # treated compartment(s), so an untreated site keeps the clone's baseline resistance.
        base_ir = rep.evolutionary_parameters["immune_resistance"]
        ir = self._tx_immune_resist_for(gid, rep, base_ir) if self._tx_applies(deme_idx) else base_ir
        ir = min(max(ir, 0.0), 1.0)
        imm_frac = ((n_immune_c / total if total else 0.0) if comp is not None
                    else self._immune_fraction(deme, total))
        death += self._immune_prob_kill * imm_frac * (1.0 - ir)

        # Compartment-dependent selection: each resident compartment adds a local hazard,
        # attenuated by the clone's matching heritable trait — the exact shape of the immune term.
        # The trait is clamped to [0,1] like `ir`. Both coefficients default to 0.0 (off).
        if self._epithelial_barrier:
            b = min(max(rep.evolutionary_parameters["breach"], 0.0), 1.0)
            epi_frac = ((n_epi_c / total if total else 0.0) if comp is not None
                        else self._epithelial_fraction(deme, total))
            death += self._epithelial_barrier * epi_frac * (1.0 - b)
        if self._stromal_hazard:
            ss = min(max(rep.evolutionary_parameters["stromal_survival"], 0.0), 1.0)
            stro_frac = ((n_stro_c / total if total else 0.0) if comp is not None
                         else self._stromal_fraction(deme, total))
            death += self._stromal_hazard * stro_frac * (1.0 - ss)
        # Metastatic host-tissue hazard: the met analogue of the stromal hazard — the deposit's
        # immortal host parenchyma adds a local death hazard to invading cancer, attenuated by the
        # clone's heritable met_survival trait (clamped [0,1] like the others). Default 0.0 (off).
        if self._met_hazard:
            ms = min(max(rep.evolutionary_parameters["met_survival"], 0.0), 1.0)
            host_frac = ((n_host_c / total if total else 0.0) if comp is not None
                         else self._host_fraction(deme, total))
            death += self._met_hazard * host_frac * (1.0 - ms)

        # Chemo/targeted death hazard, gated to the treated compartment(s): systemic ('both')
        # hits everywhere, 'met'/'primary' only their site.
        # A genotype minted mid-step is absent from _tx_death_add (frozen by this step's
        # _apply_treatment); _tx_death_add_for computes and memoises its hazard on demand rather than
        # letting a bare `.get(..., 0.0)` grant it a drug-free step.
        if self._tx_applies(deme_idx):
            death += self._tx_death_add_for(gid, rep)
        return death

    def _cancer_gids(self, deme):
        return [gid for gid in deme if self._is_cancer(gid)]

    @staticmethod
    def _run_rng(seed, offset, purpose):
        """A per-run random stream, keyed by ``(seed, purpose, offset)`` through a ``SeedSequence``,
        so no two seeds, purposes or steps share a stream."""
        return np.random.default_rng([int(seed), purpose, int(offset)])

    def _deme_rate(self, deme_idx):
        """Total event rate of a deme. Cancer genotypes are always dynamic; NORMAL genotypes become
        (death-only) dynamic in a treated compartment under the off-target chemo toxicity hazard.
        Untreated (``_tx_death_add`` empty) -> the normal term vanishes."""
        deme = self.demes[deme_idx]
        comp = self._deme_comp(deme)                # ONE composition scan reused by every genotype
        total = comp[0]
        rate = 0.0
        for gid in self._cancer_gids(deme):
            div = self._div_rate(gid, deme_idx)
            rate += deme[gid] * (div + self._death_rate(gid, deme_idx, total, comp=comp))
        if self._tx_death_add and self._tx_applies(deme_idx):
            for gid, cnt in deme.items():
                if gid in self._tx_death_add and not self._is_cancer(gid):
                    rate += cnt * self._tx_death_add[gid]
        return rate

    def _refresh_rate(self, deme_idx):
        self.deme_rates[deme_idx] = self._deme_rate(deme_idx)

    def _active_demes(self):
        """Ascending indices of the demes a tau substep can draw an event in.

        A deme with no cancer cell and no treatment-sensitive resident draws NOTHING in a tau
        substep — the genotype loop is empty — but would still pay a ``_deme_comp`` scan. In the
        ductal field EVERY deme is non-empty (it is pre-filled with normal tissue), so iterating all
        of them costs ~grid_size**2 scans per substep whatever the lesion's size. Iterating the
        cancer-occupied set instead makes the per-substep cost O(occupied demes). Purely an omission
        of no-op work: the skipped demes consume no rng draw and change no state.

        Under an active therapy normal cells DO die (``_tx_death_add``), so every deme is live and
        the full range is returned.
        """
        if self._tx_death_add:
            return range(len(self.demes))
        return sorted(self._deme_cancer_n)

    # --- simulation ----------------------------------------------------------
    def is_extinct(self):
        """Whether the tumour has died out — ``True`` when the total deme event rate is zero."""
        return self.deme_rates.sum() == 0

    def get_tumor_size(self):
        """Total number of live cells across all genotypes (cancer and normal)."""
        return int(sum(self.genotypes_counts.values()))

    def get_cancer_size(self):
        return int(sum(c for g, c in self.genotypes_counts.items() if self._is_cancer(g)))

    def _trace_snapshot(self):
        """One trace snapshot: the global genotype counts plus, when the met is enabled, the
        per-compartment counts (``primary_counts`` / ``met_counts``). Met-off snapshots carry only the
        `genotypes_counts` key."""
        snap = dict(genotypes_counts=dict(self.genotypes_counts))
        if self._met_enabled:
            snap["primary_counts"], snap["met_counts"] = self._compartment_counts()
        return snap

    def _spatial_snapshot(self):
        """Store every deme's genotype counts if the trace snapshot just appended is a spatial one.

        Called right after each trace append. A no-op unless ``spatial_snapshot_every > 0``; draws
        no random number either way. When the stored entries exceed ``spatial_snapshot_budget``,
        snapshots off the doubled interval are dropped, so the record stays evenly spaced."""
        k = self.spatial_snapshot_every
        if k <= 0:
            return
        ti = len(self.traces) - 1
        if ti % k:
            return
        genotypes = self.genotypes
        demes, ords, counts = [], [], []
        for di, deme in enumerate(self.demes):
            for gid, c in deme.items():
                demes.append(di)
                ords.append(genotypes[gid].ord)
                counts.append(c)
        self.spatial_history.append(dict(
            trace=ti, step=self._clock,
            time=self.time if self.update_mode == "tau" else float("nan"),
            deme=np.asarray(demes, dtype=np.int32), genotype=np.asarray(ords, dtype=np.int32),
            count=np.asarray(counts, dtype=np.int32)))
        self._spatial_entries += len(demes)
        while self._spatial_entries > self.spatial_snapshot_budget and len(self.spatial_history) > 1:
            self.spatial_snapshot_every *= 2
            k = self.spatial_snapshot_every
            self.spatial_history = [s for s in self.spatial_history if s["trace"] % k == 0]
            self._spatial_entries = sum(len(s["deme"]) for s in self.spatial_history)

    def _deme_location(self, deme_idx):
        """``(compartment, row, col)`` of a deme; met demes use their position in the met grid."""
        r, c = self.deme_coords[deme_idx]
        return ("met" if deme_idx >= self.n_primary_demes else "primary"), int(r), int(c)

    def genotype_births(self):
        """When and where every cancer genotype arose.

        Recorded for every run, whether or not spatial snapshots are on. Together with
        [`spatial_snapshots`][iscc.tumor.GenotypeTumor.spatial_snapshots] and the genealogy, it
        gives the tumour's spatiotemporal history: where each clone first appeared, when, and from
        which parent.

        Returns
        -------
        pandas.DataFrame
            One row per cancer genotype ever created, including clones that later died out,
            indexed by genotype id in creation order, with columns

            - ``parent``: the parent genotype id (missing for the founder);
            - ``ord``: the creation ordinal, the key ``spatial_snapshots`` reports genotypes by
              before mapping them back to ids;
            - ``birth_trace``: index of the first entry of ``traces`` that can contain it;
            - ``birth_step``: the exact-engine event, or tau-leaping generation, it arose in;
            - ``birth_time``: when it arose, in generations (tau-leaping only, NaN otherwise);
            - ``birth_deme``: the deme its first cell was placed in (-1 if it was never placed);
            - ``compartment``: ``"primary"`` or ``"met"``;
            - ``row``, ``col``: the deme's position within its compartment's grid (-1 if never
              placed).
        """
        rows = []
        for gid, rep in self.genotypes.items():
            if rep.type != "cancer":
                continue
            di = rep.birth_deme
            comp, r, c = self._deme_location(di) if di is not None else (None, -1, -1)
            rows.append(dict(genotype=gid, parent=self.genotypes_parents.get(gid), ord=rep.ord,
                             birth_trace=rep.birth_trace, birth_step=rep.birth_step,
                             birth_time=rep.birth_time, birth_deme=-1 if di is None else di,
                             compartment=comp, row=r, col=c))
        cols = ["parent", "ord", "birth_trace", "birth_step", "birth_time", "birth_deme",
                "compartment", "row", "col"]
        if not rows:
            return pd.DataFrame(columns=cols).rename_axis("genotype")
        return pd.DataFrame(rows).set_index("genotype")[cols]

    def spatial_snapshots(self):
        """The recorded spatial history: every genotype's count in every deme over time.

        Empty unless the tumour was built with ``spatial_snapshot_every > 0``. One snapshot is
        stored at every ``k``-th entry of ``traces``; if the memory budget forced thinning, the
        interval in effect is ``spatial_snapshot_every`` and the stored snapshots stay evenly
        spaced. Join on ``genotype`` with
        [`genotype_births`][iscc.tumor.GenotypeTumor.genotype_births] for where and when each
        clone arose.

        Returns
        -------
        pandas.DataFrame
            One row per (snapshot, deme, genotype) with a positive count, with columns
            ``snapshot`` (position in the record), ``trace`` (index into ``traces``), ``step``,
            ``time`` (generations; NaN under the exact engine), ``deme``, ``compartment``
            (``"primary"`` or ``"met"``), ``row``, ``col``, ``genotype`` (genotype id),
            ``cell_type`` (``"cancer"`` or the normal cell type) and ``count``.
        """
        cols = ["snapshot", "trace", "step", "time", "deme", "compartment", "row", "col",
                "genotype", "cell_type", "count"]
        if not self.spatial_history:
            return pd.DataFrame(columns=cols)
        by_ord = {rep.ord: (gid, rep.type) for gid, rep in self.genotypes.items()}
        n = np.array([len(s["deme"]) for s in self.spatial_history])
        deme = np.concatenate([s["deme"] for s in self.spatial_history])
        ords = np.concatenate([s["genotype"] for s in self.spatial_history])
        coords = np.asarray(self.deme_coords, dtype=int)
        out = pd.DataFrame(dict(
            snapshot=np.repeat(np.arange(len(n)), n),
            trace=np.repeat([s["trace"] for s in self.spatial_history], n),
            step=np.repeat([s["step"] for s in self.spatial_history], n),
            time=np.repeat([s["time"] for s in self.spatial_history], n),
            deme=deme,
            compartment=np.where(deme >= self.n_primary_demes, "met", "primary"),
            row=coords[deme, 0], col=coords[deme, 1],
            genotype=[by_ord[o][0] for o in ords],
            cell_type=[by_ord[o][1] for o in ords],
            count=np.concatenate([s["count"] for s in self.spatial_history])))
        return out[cols]

    def update(self, rng):
        """One event of Gillespie's direct method (Gillespie 1977), in demon's hierarchical form: a deme
        in proportion to its summed rate, a clone within it and the event (``_update_event``); time then
        advances by an exponential waiting time with the total rate. The waiting time is drawn after
        the event, from the same per-event stream, so the event draws do not depend on it; the founding
        acinus's confinement clock counts this time."""
        if self.is_extinct():
            return
        R = float(self.deme_rates.sum())
        self._update_event(rng)
        dt = float(rng.exponential(1.0 / R))
        self.time += dt
        if self._origin_confined:
            self._origin_tick(dt)

    def _update_event(self, rng):
        """Pick a deme in proportion to its summed rate and fire one event in it."""
        di = int(rng.choice(len(self.demes), p=self.deme_rates / self.deme_rates.sum()))
        self._event_in_deme(di, rng)

    def _event_in_deme(self, di, rng, refresh=True):
        """Fire one event in deme ``di``, the second and third levels of the direct method: a clone in
        proportion to count x (division + death) (or a resident under off-target toxicity), then death
        with probability death / (division + death), else a division with its routing, mutation, slot
        cap and eviction. ``refresh`` updates ``deme_rates`` for the demes the event touched; the
        critical-reaction tau-leaping (``_tau_leaps_critical``) keeps its own rates and passes False."""
        deme = self.demes[di]
        # pick a cancer genotype in the deme proportionally to count * (div + death),
        # ordered by creation ordinal (NOT the id()-based genotype_id) for reproducibility
        gids = sorted(self._cancer_gids(deme), key=lambda g: self.genotypes[g].ord)
        comp = self._deme_comp(deme)                # ONE composition scan reused by every genotype
        total = comp[0]
        weights = [
            deme[gid] * (self._div_rate(gid, di)
                         + self._death_rate(gid, di, total, comp=comp))
            for gid in gids
        ]
        # NORMAL cells become death-only candidates under off-target chemo toxicity in a treated deme.
        # tox_gids is empty unless a therapy is active.
        tox_gids = []
        if self._tx_death_add and self._tx_applies(di):
            tox_gids = sorted((g for g in deme if g in self._tx_death_add and not self._is_cancer(g)),
                              key=lambda g: self.genotypes[g].ord)
            weights += [deme[g] * self._tx_death_add[g] for g in tox_gids]
        weights = np.array(weights)
        pick = int(rng.choice(len(gids) + len(tox_gids), p=weights / weights.sum()))
        if pick >= len(gids):
            self._remove(di, tox_gids[pick - len(gids)], 1)   # a normal cell dies from toxicity
            if refresh:
                self._refresh_rate(di)
            return
        gid = gids[pick]
        rep = self.genotypes[gid]
        div = self._div_rate(gid, di, rep)
        death = self._death_rate(gid, di, total, comp=comp)

        affected = [di]
        if rng.random() < death / (div + death):
            self._remove(di, gid, 1)                       # death
        else:
            self._divide_noble(di, gid, rep, rng, affected)

        if refresh:
            for idx in affected:
                self._refresh_rate(idx)

    # --- growth --------------------------------------------------------------
    def grow(self, n_steps=1000, seed=None, treatment=None, materialize=True, target_cancer=None,
             **kwargs):
        """Advance the tumour by ``n_steps`` update steps and materialise the result.

        Runs the birth / death / mutation / dispersal process (exact Gillespie or
        tau-leaping, per ``update_mode``), appending a population snapshot to
        ``self.traces`` each step, and finally calls ``make_cell_data`` to build the
        per-cell ground truth (unless ``materialize=False``).

        Parameters
        ----------
        n_steps : int, optional
            Number of update steps to run (default 1000).
        seed : int, optional
            Evolution seed for this call; defaults to the tumour's construction ``seed``.
        treatment : Treatment, optional
            A therapy (e.g. [`Chemotherapy`][iscc.treatment.Chemotherapy],
            [`TargetedTherapy`][iscc.treatment.TargetedTherapy],
            [`Immunotherapy`][iscc.treatment.Immunotherapy],
            [`Surgery`][iscc.treatment.Surgery]) whose dosing schedule and rate modifiers are
            applied each step; ``None`` (default) grows the tumour untreated.
        materialize : bool, optional
            Rebuild ``cell_data`` at the end of the call (default ``True``). ``False`` skips it
            and marks the per-cell tables stale, for callers that step the tumour in a loop and
            only need cells when they sample: call
            [`ensure_cell_data`][iscc.tumor.GenotypeTumor.ensure_cell_data] before sampling.
            Materialisation draws only from its own seeded generators and writes nothing the
            dynamics read, so the trajectory is the same either way.
        target_cancer : int, optional
            Stop as soon as the cancer population reaches this size: after the event that reaches
            it (exact) or the sub-step or leap that reaches it (tau), with ``self.time`` advanced
            only by the time simulated. Growth below the target is unchanged (the check draws
            nothing).

        Returns
        -------
        list
            ``self.traces`` — the list of per-step population snapshots.
        """
        if self.update_mode == "tau":
            return self._grow_tau(n_steps=n_steps, seed=seed, treatment=treatment,
                                  materialize=materialize, target_cancer=target_cancer, **kwargs)
        if seed is None:
            seed = self.seed
        self._clear_treatment_overrides()
        self.traces.append(self._trace_snapshot())
        self._spatial_snapshot()
        prev_dose = 0.0
        done = n_steps
        for local_step in range(n_steps - 1):
            if target_cancer is not None and self.get_cancer_size() >= target_cancer:
                done = local_step
                break
            self._clock = self.step + local_step
            rng = self._run_rng(seed, self.step + local_step, _STREAM_GROW)
            if treatment is not None:
                treatment.discrete_event(self, self.step + local_step)  # e.g. surgical resection
                dose = treatment.get_dosage(self.step + local_step, self.get_tumor_size())
                self._apply_treatment(treatment, dose)
                # death rates changed for every genotype while a dose is on (or just turned
                # off); refresh the whole rate vector so event sampling stays correct.
                if dose > 0 or prev_dose > 0:
                    self.deme_rates = np.array(
                        [self._deme_rate(i) for i in range(len(self.demes))], dtype=float)
                prev_dose = dose
            self.update(rng)
            self.traces.append(self._trace_snapshot())
            self._spatial_snapshot()
        self.step += done
        self._clock = self.step
        if materialize:
            self.make_cell_data()
        else:
            self._cell_data_stale = True
        return self.traces

    # --- one division (exact engine) -----------------------------------------
    def _mutate_daughter(self, gid, rng):
        """Divide ``gid`` and mutate the daughter. Returns ``(joins, folded)``: the genotype the
        daughter joins (``gid`` for a purely neutral daughter, which folds into the parent and adds
        its passenger burden, or for a saturated allele; a newly registered genotype for a functional
        daughter; ``None`` for a daughter that breaches the viability limits and is never born) and
        whether it was folded."""
        rep = self.genotypes[gid]
        child = rep.divide()
        res = child.mutate(rng, self.selection,
                           func_mask=self._func_bits if self.coarsen_passengers else None)
        if res == "passenger":
            self._pass_load[gid] += child._n_new_snv
            self._n_folded += 1
            return gid, True
        if res:
            if not self._is_viable(child):
                return None, False
            self._register(child)
            self.genotypes_parents[child.genotype_id] = rep.genotype_id
            return child.genotype_id, False
        return gid, False

    def _realise_one_mutant(self, tgt, gid, rng):
        """Realise ONE mutated daughter of ``gid`` in deme ``tgt``, where the caller has already made
        sure a slot can be had: a functional daughter becomes a new genotype unless it breaches the
        viability limits, a purely neutral one folds into the parent, and a saturated allele grows the
        parent. Returns the genotype the cell joined, or ``None`` when nothing was born."""
        joins, _ = self._mutate_daughter(gid, rng)
        if joins is None:
            return None
        return joins if self._place_one(tgt, joins, rng) else None

    def _divide_noble(self, di, gid, rep, rng, affected):
        """One division under demon's invasion rule.

        The division attempts a mutation with probability ``mutation_rate``. Independently, one of its
        two daughters, picked at random, disperses with probability ``dispersal_rate``; inside a
        still-intact founding acinus only an ``_escape_factor`` share of those leave, and the rest stay
        home. The parent copy and the new daughter are interchangeable unless the new one mutated, so
        the pick is drawn only then: the mutant itself leaves, or the parent copy leaves and the mutant
        takes its place. Every placement goes through the structural cap where it is on, and a birth
        whose moving cell finds no slot fails, so no genotype is registered without a cell. A mutated
        daughter is routed with its parent's traits (breach, met survival): the new mutation acts from
        its next division on.
        """
        mutate = rng.random() < min(rep.mutation_rate, 1.0)
        disp = rep.evolutionary_parameters["dispersal_rate"]
        move = disp > 0.0 and rng.random() < disp
        if move and (esc := self._escape_factor(di)) < 1.0 and (esc <= 0.0 or rng.random() >= esc):
            move = False                                   # held in by the intact acinus
        if not move:
            if mutate:
                if not self._crowding or self._can_place(di):
                    self._realise_one_mutant(di, gid, rng)
            else:
                self._place_one(di, gid, rng)
            return
        mutant_leaves = mutate and rng.random() < 0.5
        route = self._route_one(di, rep, rng)
        if route is None:                                  # the moving cell died in transit
            if mutate and not mutant_leaves:
                self._remove(di, gid, 1)                   # the parent copy left; the mutant stays
                self._realise_one_mutant(di, gid, rng)
            return
        tgt, is_met = route
        if mutant_leaves:
            if self._crowding and not self._can_place(tgt):
                return
            placed = self._realise_one_mutant(tgt, gid, rng)
        else:
            placed = gid if self._place_one(tgt, gid, rng) else None
            if placed is not None and mutate:
                self._remove(di, gid, 1)                   # the parent copy left; the mutant stays
                self._realise_one_mutant(di, gid, rng)
        if placed is not None:
            affected.append(tgt)
            if is_met:
                self.events.append(dict(step=self.step, time=self.time, event="seeding",
                                        from_deme=int(di), genotype=placed, n=1))

    # --- tau-leaping ---------------------------------------------------------
    def _split_births_noble(self, rng, di, gid, rep, n, esc, mutants, stayers, dispersals,
                            mut_disp, swaps):
        """Split ``n`` births of ``gid`` in deme ``di`` under demon's invasion rule (see
        ``_divide_noble``): each birth mutates with probability ``mutation_rate`` and, independently,
        sends one daughter out with probability ``dispersal_rate``; inside an intact founding acinus
        only an ``esc`` share leaves. When a mutated birth sends a daughter out, the mutant itself
        leaves (``mut_disp``) or the parent copy leaves and the mutant takes its place (``swaps``),
        with equal probability. Births that stay home unmutated join ``stayers``."""
        mu = min(max(float(rep.mutation_rate), 0.0), 1.0)
        d = min(max(float(rep.evolutionary_parameters["dispersal_rate"]), 0.0), 1.0)
        n_md, n_m, n_d, n_0 = (int(k) for k in rng.multinomial(
            n, [mu * d, mu * (1.0 - d), (1.0 - mu) * d, (1.0 - mu) * (1.0 - d)]))
        if esc < 1.0:
            out = int(rng.binomial(n_md, esc)) if (esc > 0.0 and n_md) else 0
            n_m += n_md - out
            n_md = out
            out = int(rng.binomial(n_d, esc)) if (esc > 0.0 and n_d) else 0
            n_0 += n_d - out
            n_d = out
        n_leave = int(rng.binomial(n_md, 0.5)) if n_md else 0
        if n_m:
            mutants.append((di, gid, n_m))
        if n_0:
            stayers.append((di, gid, n_0))
        if n_d:
            dispersals.append((di, gid, n_d))
        if n_leave:
            mut_disp.append((di, gid, n_leave))
        if n_md - n_leave:
            swaps.append((di, gid, n_md - n_leave))

    def _convert_in_place(self, di, gid, n, rng):
        """``n`` parent copies of ``gid`` have left deme ``di`` and the mutated daughters take their
        slots: remove them and realise the mutants in place (capped at the cells still there)."""
        m = min(int(n), self.demes[di].get(gid, 0))
        if m:
            self._remove(di, gid, m)
            self._realise_mutation_births(di, gid, m, rng)

    def _realise_mutation_births(self, di, gid, n, rng):
        """Realise ``n`` accepted mutation-branch births of ``gid`` in deme ``di``: each is one
        division that attempts a mutation, spawning a new genotype (unless it breaches the viability
        limits, in which case nothing is born), folding into the parent when the daughter is purely
        neutral, or growing the parent in place on a saturated allele. Adds AT MOST ``n`` cells."""
        n_fold = 0
        for _ in range(n):
            joins, folded = self._mutate_daughter(gid, rng)
            if folded:
                n_fold += 1                              # folds join the parent in one batch below
            elif joins is not None:
                self._add(di, joins, 1)
        if n_fold:
            self._add(di, gid, n_fold)

    def _tau_substep(self, rng, dt, demes=None):
        """One synchronous tau-leap of length `dt`.

        ``demes`` (ascending indices) restricts the leap to the channels of those demes; ``None`` leaps
        every active deme. Births leaped from them still compete for slots wherever they land.

        The exact engine needs no allocation rule — it fires one event at a time, so a slot is taken
        by whichever birth happens next and the Gillespie sampler already draws that birth with
        probability ``b_i c_i / sum_k b_k c_k``. Tau-leaping cannot block individual events, so it
        needs one, in four phases:

        0/1. draw births ~ Poisson(division_rate*c*dt) and deaths ~ Poisson(death_rate*c*dt) per
             (deme, clone) from the PRE-step state, split the births by the division law, and apply
             the DEATHS first;
        2.   route every candidate birth to its TARGET deme (in place for the mutation branch; the
             drawn neighbour / cross-gland lumen / metastatic vessel for the dispersal branch, and
             back to the source duct for a failed basement-membrane crossing), accumulating a
             per-target candidate vector keyed by (channel, genotype);
        3.   per target, in ascending deme order, allocate the ``S`` free slots (``_slot_cap_of``
             minus the current occupancy) among
             the candidates with a MULTIVARIATE HYPERGEOMETRIC draw, then offer the leftovers a
             second, nested draw for the slots that displacing residents would free;
        4.   realise ONLY the accepted births — so a mutation-branch birth that loses the lottery
             never reaches ``mutate()`` and registers no genotype.

        The hypergeometric is not an expectation match, it is the exact conditional law: in the
        event-by-event path the births in an interval are a superposition of Poisson processes, so
        conditional on their counts their ORDER is a uniformly random permutation of the multiset,
        the exact rule accepts the first ``S`` of that order, and the composition of the first ``S``
        items of a uniformly random permutation of a multiset IS ``MVH(S; a_1..a_m)``. The residual
        difference from the exact engine is the one tau-leaping already makes everywhere — births
        and deaths treated as simultaneous within a substep — which biases slot supply by ~half a
        substep's deaths. That is a RATE bias, never a cap violation: ``S`` is recomputed from the
        real occupancy every substep and clipped at zero.
        """
        deaths, dispersals, mutants, stayers = [], [], [], []
        mut_disp, swaps = [], []     # mutated births that leave, or whose parent copy leaves
        binomial = self._tau_deaths == "binomial"
        for di in (self._active_demes() if demes is None else demes):
            deme = self.demes[di]
            if not deme:
                continue
            comp = self._deme_comp(deme)        # ONE composition scan reused by every genotype
            total = comp[0]
            esc = self._escape_factor(di)
            for gid in sorted(self._cancer_gids(deme), key=lambda g: self.genotypes[g].ord):
                c = deme[gid]
                rep = self.genotypes[gid]
                div = self._div_rate(gid, di, rep)
                death = self._death_rate(gid, di, total, comp=comp)
                n_div = int(rng.poisson(div * c * dt))
                if binomial:
                    n_death = int(rng.binomial(c, min(1.0, death * dt)))
                else:
                    n_death = int(rng.poisson(death * c * dt))
                if n_death:
                    deaths.append((di, gid, n_death))
                if n_div:
                    self._split_births_noble(rng, di, gid, rep, n_div, esc, mutants, stayers,
                                             dispersals, mut_disp, swaps)
            if self._tx_death_add and self._tx_applies(di):
                for gid in sorted((g for g in deme if g in self._tx_death_add and not self._is_cancer(g)),
                                  key=lambda g: self.genotypes[g].ord):
                    n_death = int(rng.poisson(self._tx_death_add[gid] * deme[gid] * dt))
                    if n_death:
                        deaths.append((di, gid, n_death))

        # --- Phase 1: deaths, capped at the pre-step count so a clone never goes negative --------
        for di, gid, n in deaths:
            n = min(n, self.demes[di].get(gid, 0))
            if n:
                self._remove(di, gid, n)

        # --- Phase 2: route candidate births to their target demes -------------------------------
        cand = {}                       # target deme -> {(channel, ...): n candidates}, insertion-ordered

        def _cand(tgt, key, n):
            if n > 0:
                slots = cand.setdefault(tgt, {})
                slots[key] = slots.get(key, 0) + int(n)

        for di, gid, n in mutants:      # the mutation branch divides IN PLACE: target is the source
            _cand(di, ("mut", gid), n)
        # Daughters the intact founding acinus held in: ordinary candidates for a slot at HOME,
        # exactly like a failed basement-membrane crossing. Always empty without confinement.
        for di, gid, n in stayers:
            _cand(di, ("plain", gid), n)
        # Dispersing daughters: a failed basement-membrane crossing leaves the daughter in its own
        # duct, where it still has to win a slot; seeding the met has its own channel, keyed by
        # source, so the event log reports the cells that WON a slot in the vessel deme.
        for di, gid, n in dispersals:
            routes, _ = self._route_births(di, gid, n, rng)
            for tgt, cnt, is_met in routes:
                _cand(tgt, ("met", di, gid) if is_met else ("plain", gid), cnt)
        # A leaving mutant competes for a slot where it lands
        # and is realised only if it wins one; a leaving parent copy competes the same way, and only
        # when it wins does the mutant take its place at home. A parent copy that dies in transit has
        # left all the same, so its mutant takes its place.
        lost_swaps = []
        for di, gid, n in mut_disp:
            routes, _ = self._route_births(di, gid, n, rng)
            for tgt, cnt, is_met in routes:
                _cand(tgt, ("metmut", di, gid) if is_met else ("mut", gid), cnt)
        for di, gid, n in swaps:
            routes, n_lost = self._route_births(di, gid, n, rng)
            for tgt, cnt, is_met in routes:
                _cand(tgt, ("swapmet" if is_met else "swap", di, gid), cnt)
            if n_lost:
                lost_swaps.append((di, gid, n_lost))

        # --- Phase 3 + 4: allocate the slots, then realise only the accepted births ---------------
        for tgt in sorted(cand):
            items = cand[tgt]
            keys = list(items)
            a = np.array([items[k] for k in keys], dtype=np.int64)
            n_slots = (max(0, self._slot_cap_of(tgt) - sum(self.demes[tgt].values()))
                       if self._crowding else int(a.sum()))       # well-mixed: no slot cap
            if int(a.sum()) <= n_slots:
                acc = a
            else:
                acc = (rng.multivariate_hypergeometric(a, n_slots) if n_slots > 0
                       else np.zeros(len(a), dtype=np.int64))
                leftover = a - acc
                n_evict = min(int(leftover.sum()), self._n_evictable(tgt))
                if n_evict > 0:
                    # The next `n_evict` items of the SAME uniformly random order — exact for the
                    # same reason the first draw is.
                    acc = acc + rng.multivariate_hypergeometric(leftover, n_evict)
                    self._evict(tgt, n_evict, rng)
            for key, n in zip(keys, acc):
                n = int(n)
                if n <= 0:
                    continue
                if key[0] == "mut":
                    self._realise_mutation_births(tgt, key[1], n, rng)
                elif key[0] == "met":
                    self._add(tgt, key[2], n)
                    self.events.append(dict(step=self.step, time=self.time, event="seeding",
                                            from_deme=int(key[1]), genotype=key[2], n=n))
                elif key[0] == "metmut":
                    self._realise_mutation_births(tgt, key[2], n, rng)
                    self.events.append(dict(step=self.step, time=self.time, event="seeding",
                                            from_deme=int(key[1]), genotype=key[2], n=n))
                elif key[0] in ("swap", "swapmet"):
                    self._add(tgt, key[2], n)
                    if key[0] == "swapmet":
                        self.events.append(dict(step=self.step, time=self.time, event="seeding",
                                                from_deme=int(key[1]), genotype=key[2], n=n))
                    self._convert_in_place(key[1], key[2], n, rng)
                else:
                    self._add(tgt, key[1], n)
        for di, gid, n in lost_swaps:
            self._convert_in_place(di, gid, n, rng)

    def _cgp_dt(self, eps, demes=None):
        """The species-based leap of Cao, Gillespie and Petzold (2006, J Chem Phys 124:044109, eq. 33).

        Each (deme, cancer clone) count c is a species, changed by first-order channels (division at b,
        death at d, both per cell), so its highest reaction order is 1. With mean drift
        mu = (b - d) c and variance sigma^2 = (b + d) c, the leap that keeps the expected change and its
        standard deviation within max(eps c, 1) is
            min(max(eps c, 1) / |mu|, max(eps c, 1)^2 / sigma^2),
        and the step is the minimum over all species. Inflow by dispersal is not credited, which only
        shortens the step. ``demes`` restricts the species to those demes (default: every active one)."""
        best = np.inf
        for di in (self._active_demes() if demes is None else demes):
            deme = self.demes[di]
            if not deme:
                continue
            comp = self._deme_comp(deme)
            for gid in self._cancer_gids(deme):
                c = deme[gid]
                b = self._div_rate(gid, di, self.genotypes[gid])
                d = self._death_rate(gid, di, comp[0], comp=comp)
                bound = max(eps * c, 1.0)
                mu, var = abs(b - d) * c, (b + d) * c
                if mu > 0:
                    best = min(best, bound / mu)
                if var > 0:
                    best = min(best, bound * bound / var)
        return best

    def _classify_critical(self, di):
        """Mark deme ``di`` critical when it holds a channel (a cancer cell, or a resident under
        off-target toxicity) and its total, residents included, is at or above K, where the d1 crowding
        channel acts; record its total propensity in ``_crit_rate`` (0 for a non-critical deme)."""
        live = di in self._deme_cancer_n or (self._tx_death_add and self.demes[di])
        if live and sum(self.demes[di].values()) >= self._cap_of(di):
            self._crit_rate[di] = self._deme_rate(di)
            self._crit.add(di)
        else:
            self._crit_rate[di] = 0.0
            self._crit.discard(di)

    def _tau_leaps_critical(self, rng, tau, target_cancer=None):
        """One generation of length ``tau`` by explicit tau-leaping with critical reactions (Cao,
        Gillespie and Petzold 2005, J Chem Phys 123:054104, "Avoiding negative populations in explicit
        Poisson tau-leaping"). Returns the time simulated (``tau`` unless ``target_cancer`` stops it
        after a leap).

        The channels (division and death of every cancer clone, and toxicity deaths of residents under
        therapy) of a deme whose total is at or above K, where the d1 crowding channel acts, are
        critical; all others are non-critical. Each leap, from the current state:

        1. tau1 = the epsilon-bounded step of Cao, Gillespie and Petzold (2006) over the non-critical
           species only (``_cgp_dt`` on the non-critical demes);
        2. tau2 ~ Exp(a0c), a0c the summed propensity of the critical channels;
        3. if tau1 < tau2 the non-critical channels are leaped by tau1 and no critical channel fires;
           otherwise exactly one critical event fires, its deme chosen in proportion to its propensity
           and the clone and event within it as in the exact engine (``_event_in_deme``), and the
           non-critical channels are leaped by tau2;
        4. a step that would pass the end of the generation is clipped to it; the non-critical channels
           are leaped by the remainder and no critical event fires, which is exact for the critical
           channels because their waiting time is memoryless.

        The leap is applied first, from the rates at the leap's start, and the critical event at its
        end: its deme is drawn from the propensities at the leap's start, and the clone and event within
        that deme from the deme as it is after the leap, which differs only by births leaped into it
        from its non-critical neighbours. Demes are re-classified after every leap (the demes whose
        cells changed, recorded by ``_add`` / ``_remove``), and from scratch at the start of every
        generation. Births leaped from a non-critical deme into a critical one are channels of the
        source deme and stay in the leap."""
        n = len(self.demes)
        self._crit_rate = np.zeros(n)
        self._crit = set()
        for di in self._active_demes():
            self._classify_critical(di)
        self._dirty = set()
        left, elapsed = tau, 0.0
        try:
            while left > 1e-12:
                noncrit = [di for di in self._active_demes() if di not in self._crit]
                tau1 = self._cgp_dt(self._tau_epsilon, noncrit) if noncrit else np.inf
                a0c = float(self._crit_rate.sum())
                tau2 = float(rng.exponential(1.0 / a0c)) if a0c > 0 else np.inf
                fire = tau2 <= tau1
                dt = tau2 if fire else tau1
                if dt >= left:
                    dt, fire = left, False
                if fire:                                # the deme, by the propensities at the leap's start
                    cum = np.cumsum(self._crit_rate)
                    di = min(int(np.searchsorted(cum, rng.random() * cum[-1], side="right")), n - 1)
                    while self._crit_rate[di] <= 0.0:   # float edge at a zero-rate tail entry
                        di -= 1
                self._tau_substep(rng, dt, demes=noncrit)   # an empty list leaps nothing, draws nothing
                if fire:
                    self._event_in_deme(di, rng, refresh=False)
                    self._n_crit_events = getattr(self, "_n_crit_events", 0) + 1
                self._n_substeps = getattr(self, "_n_substeps", 0) + 1
                left -= dt
                elapsed += dt
                for di in self._dirty:
                    self._classify_critical(di)
                self._dirty.clear()
                if target_cancer is not None and self.get_cancer_size() >= target_cancer:
                    break
        finally:
            self._dirty = None
            del self._crit_rate, self._crit
        return elapsed if left > 1e-12 else tau

    def _tau_generation(self, rng, tau, target_cancer=None):
        """Advance the whole tumour by one generation of length `tau`, sub-stepped by the
        ``tau_step`` rule (see ``_init_tau``): equal sub-steps that keep the largest single-cell event
        probability per sub-step at most 0.34 (``"fixed"``), the Cao-Gillespie-Petzold step
        (``"cgp"``), or tau-leaping with critical reactions (``"cgp_critical"``). Death rates are
        re-read every sub-step. Records a full per-clone snapshot every `snapshot_every`
        generations.

        ``target_cancer``, if given, stops the generation after the first sub-step (or leap) at which
        the cancer population reaches it; ``self.time`` then advances only by the time simulated, and
        the once-per-generation resistance-state transitions are skipped. The check draws nothing, so a
        generation that does not reach the target is unchanged."""
        ACCURACY = 0.34  # keep rate*dt <= this so Poisson tau-leaping stays accurate
        if self.coarsen_passengers:
            # incremental monotone upper bound (see _max_div_disp) — avoids the O(#genotypes) scan
            max_cell_rate = self._max_div_disp + self.maximum_death_rate
        else:
            max_cell_rate = 0.0
            for gid in self.genotypes_counts:
                if self._is_cancer(gid):
                    ep = self.genotypes[gid].evolutionary_parameters
                    max_cell_rate = max(max_cell_rate,
                                        ep["division_rate"] + ep["dispersal_rate"])
            max_cell_rate += self.maximum_death_rate
        self._clock = self.step
        reached = (lambda: target_cancer is not None and self.get_cancer_size() >= target_cancer)
        stopped = None                  # the time simulated when the target stopped the generation
        if self._tau_step == "cgp_critical":
            elapsed = self._tau_leaps_critical(rng, tau, target_cancer)
            if reached():
                stopped = elapsed
        elif self._tau_step == "cgp":
            left = tau
            while left > 1e-12:
                dt = min(left, self._cgp_dt(self._tau_epsilon))
                self._tau_substep(rng, dt)
                self._n_substeps = getattr(self, "_n_substeps", 0) + 1
                left -= dt
                if reached():
                    stopped = tau - left if left > 1e-12 else tau
                    break
        else:
            n_sub = max(1, int(np.ceil(max_cell_rate * tau / ACCURACY)))
            dt = tau / n_sub
            for k in range(n_sub):
                self._tau_substep(rng, dt)
                if reached():
                    stopped = (k + 1) * dt if k + 1 < n_sub else tau
                    break
        partial = stopped is not None and stopped < tau
        # Drug-induced resistance-state entry/exit, once per generation on the settled counts, before
        # the snapshot. Gated: no draw / no genotype minted when the feature is off.
        if self.selection.resistance_state_on and not partial:
            self._apply_state_transitions(rng)
        self.step += 1
        self.time += stopped if partial else tau
        if self._origin_confined:
            self._origin_tick(stopped if partial else tau)
        self._clock = self.step
        if self.step % self.snapshot_every == 0:
            self.traces.append(self._trace_snapshot())
            self.trace_times.append(self.time)
            self._spatial_snapshot()

    def _grow_tau(self, n_steps=1000, seed=None, treatment=None, tau=None, materialize=True,
                  target_cancer=None, **kwargs):
        """Tau-leaping growth: `n_steps` is the number of generations to advance. A full per-clone
        snapshot is recorded every `snapshot_every` generations, so plot_muller/plot_grid work on a
        REAL-TIME x-axis (self.trace_times, in generation units).
        `materialize=False` skips the closing `make_cell_data` and marks the tables stale (see
        `grow`). `target_cancer` stops growth at the first sub-step reaching it (see `grow`)."""
        if seed is None:
            seed = self.seed
        if tau is None:
            tau = self.tau
        rng = self._run_rng(seed, self.step, _STREAM_GROW)
        self._clear_treatment_overrides()
        # snapshot at t=0 (matches the exact engine's initial trace)
        self.traces.append(self._trace_snapshot())
        self.trace_times.append(self.time)
        self._spatial_snapshot()
        prev_dose = 0.0
        for _ in range(n_steps):
            if self.get_cancer_size() == 0:
                break
            if target_cancer is not None and self.get_cancer_size() >= target_cancer:
                break
            if treatment is not None:
                treatment.discrete_event(self, self.step)  # e.g. surgical resection
                dose = treatment.get_dosage(self.step, self.get_tumor_size())
                self._apply_treatment(treatment, dose)
                prev_dose = dose
            self._tau_generation(rng, tau, target_cancer=target_cancer)
        if materialize:
            self.make_cell_data()
        else:
            self._cell_data_stale = True
        return self.traces

    def get_genotype_frequencies(self, normalize=True):
        """Cancer-genotype ids and their frequencies (normal cell types excluded).

        Parameters
        ----------
        normalize : bool, optional
            If ``True`` (default) the counts are divided by their sum to give proportions;
            otherwise raw cell counts are returned.

        Returns
        -------
        tuple of (list, numpy.ndarray)
            The cancer-genotype ids and their matching counts / frequencies.
        """
        gids = [g for g in self.genotypes_counts if g not in normal_names]
        counts = np.array([self.genotypes_counts[g] for g in gids], dtype=float)
        if normalize and counts.sum() > 0:
            counts = counts / counts.sum()
        return gids, counts

    def get_gene_data(self, **kwargs):
        return self.selection.get_gene_data(**kwargs)

    # --- epistasis ground truth -----------------------------------------------
    def epistasis_ground_truth(self):
        """The planted network — the answer key for MHN/TreeMHN/CBN/REVOLVER edge + order recovery.

        Returns the true ``E`` matrix, the marginal effects ``beta``, the interaction edges, the
        conjunctive dependency DAG, the mutually-exclusive pairs and the event->gene modules; or
        ``None`` when epistasis is off. Shared by every patient of a cohort (drawn from the layout
        stream), which is what makes pooling patients to recover ONE network well-posed.
        """
        if self.selection.epistasis is None:
            return None
        return self.selection.epistasis.ground_truth()

    def event_table(self):
        """Per-clone event sets and the ORDER this lineage acquired them (empty when epistasis is off).

        One row per cancer genotype: its cell count, the events it carries, and the realised order
        along its lineage — the per-patient ordering ground truth. ``event_groups`` is the truthful
        form (one tuple per mutating division; events inside a group are TIED, having been acquired
        together, so no order between them exists); ``event_order`` is that flattened for exports
        needing a linear path. **Score ordering against ``event_groups``** — ``event_order`` breaks
        ties arbitrarily.
        """
        net = self.selection.epistasis
        rows = []
        cols = ["genotype_id", "n_cells", "events", "event_order", "event_groups"]
        if net is None:
            return pd.DataFrame(columns=cols)
        for gid, count in self.genotypes_counts.items():
            if gid in normal_names:
                continue
            rep = self.genotypes[gid]
            rows.append(dict(genotype_id=gid, n_cells=int(count),
                             events=tuple(bits_to_events(rep.event_bits, net.n_events)),
                             event_order=tuple(rep.event_order),
                             event_groups=tuple(rep.event_groups)))
        return pd.DataFrame(rows, columns=cols)

    # --- operating-envelope QC (read-only) ------------------------------------
    def diagnose(self, thresholds=None, verbose=False):
        """Flag degenerate ("crappy") tumour regimes after growth, with actionable hints.

        Computes phenotype metrics (size, clonal diversity, TMB, clone spatial confinement,
        fraction-genome-altered, hypoxia core–rim contrast, 1/f fit) and checks each against an
        overridable threshold: extinct / monoclonal / hypermutated / well-mixed /
        no-microenvironment-gradient / CNA-runaway / trivial-genome. Returns a ``TumorDiagnosis``.
        Read-only — it never draws from ``self.rng`` or changes the counts, so it cannot alter
        simulation output (like the microenvironment ground truth)."""
        from ..diagnostics import diagnose
        return diagnose(self, thresholds=thresholds, verbose=verbose)

    # --- output --------------------------------------------------------------
    def write(self, output_path):
        """Write the canonical iscc layout (so isccsample/isccdata can consume it)."""
        Path(output_path).mkdir(parents=True, exist_ok=True)
        if self.traces:
            pd.DataFrame([t["genotypes_counts"] for t in self.traces]).fillna(0).to_csv(
                os.path.join(output_path, "trace_counts.csv"))
            # Per-compartment abundance: only present when the met is enabled (the snapshots then
            # carry primary_counts / met_counts). Both share the one parents.csv genealogy -> the
            # 2-band Muller.
            if self._met_enabled and "primary_counts" in self.traces[0]:
                pd.DataFrame([t["primary_counts"] for t in self.traces]).fillna(0).to_csv(
                    os.path.join(output_path, "trace_counts_primary.csv"))
                pd.DataFrame([t.get("met_counts", {}) for t in self.traces]).fillna(0).to_csv(
                    os.path.join(output_path, "trace_counts_met.csv"))
            pd.DataFrame([self.genotypes_parents]).to_csv(os.path.join(output_path, "parents.csv"))
            gids, _ = self.get_genotype_frequencies()
            pd.DataFrame(gids).to_csv(os.path.join(output_path, "genotypes.csv"))
        # Discrete-event annotations (seeding / resection / chemo windows), for the Muller/animation.
        if self.events:
            pd.DataFrame(self.events).to_csv(os.path.join(output_path, "events.csv"), index=False)
        # Spatiotemporal record: the birth table always (one row per cancer genotype), the
        # per-deme snapshots only when they were recorded.
        self.genotype_births().to_csv(os.path.join(output_path, "genotype_births.csv"))
        if self.spatial_history:
            self.spatial_snapshots().to_csv(os.path.join(output_path, "spatial_history.csv.gz"),
                                            index=False)

        gene_data = self.selection.get_gene_data()
        Path(os.path.join(output_path, "gene_data")).mkdir(parents=True, exist_ok=True)
        for mat in gene_data:
            pd.DataFrame(gene_data[mat]).to_csv(os.path.join(output_path, "gene_data", f"{mat}.csv"))

        self.ensure_cell_data()
        Path(os.path.join(output_path, "cell_data")).mkdir(parents=True, exist_ok=True)
        for mat in self.cell_data:
            self.cell_data[mat].to_csv(os.path.join(output_path, "cell_data", f"{mat}.csv"))

    def write_h5ad(self, path, compression="gzip", **kwargs):
        """Write the whole tumour — cells, genes, clones and parameters — to one ``.h5ad`` file.

        The standard single-cell container instead of a directory of CSVs: expression in ``X``, the
        other per-cell matrices in layers, per-cell annotation in ``obs``, tissue coordinates in
        ``obsm``, gene roles in ``var``, and the clone tree / programs / run parameters in ``uns``.
        Reads back anywhere ``anndata`` is installed (``anndata.read_h5ad``). Extra keywords go to
        :func:`iscc.integrations.to_anndata` (e.g. ``spatial="rowcol"``).
        """
        from ...integrations.anndata import to_anndata
        adata = to_anndata(self, **kwargs)
        adata.write_h5ad(str(path), compression=compression)
        return path
