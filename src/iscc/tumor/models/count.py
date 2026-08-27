"""
Genotype-level (count-based) tumor engine.

Instead of one Python object per cell, the population is represented as **per-deme genotype
counts** over a shared **genotype registry**. A "cell" is a unit of count in a (genotype,
deme) bucket; cells of the same genotype in the same deme are statistically exchangeable, so
counting them loses no information.

This realises the same birth/death/mutation/dispersal process as the cell-level
`GlandularTumor` (validated by statistical equivalence — it is NOT byte-identical, since it
draws "how many of the count" rather than "which cell"), but each event is O(1) integer
arithmetic on counts. A genotype is carried by a representative cell (reusing its genome /
mutate / get_snvs / get_cnvs / get_exp and CINner fitness).

Normal cells (epithelial/stromal/immune) are seeded as **static** genotype counts (they don't
divide or mutate); they provide tissue structure, cell-type labels, and the local immune density
that the cancer death rate reads. Treatment runs on this engine (see grow/_apply_treatment) and
immune killing is additive contact pressure (see _death_rate); the immune compartment itself is
static (recruitment/migration are future work).
"""
import os
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..components.selection import Selection
from ..components.epistasis import bits_to_events
from ..components.cell import CancerCell, EpithelialCell, StromalCell, ImmuneCell, HostCell
from ..germline import draw_germline_sites, apply_germline, seed_founder_mutations
from .glandular import bresenham_circumference, get_inside
from ..programs import ProgramModel
from ...constants import normal_names, DEFAULT_LAYOUT_SEED, LAYOUT_OFFSET_F8_PROGRAMS, LAYOUT_OFFSET_MET

CELLTYPES = ["cancer", "epithelial", "stromal", "immune"]

# --- Coalescent passenger overlay (v1, DESIGN_snv_coalescent.md §9 step 1) --------------------------
# Fraction of a clone's realized folded passenger burden (``_pass_load``) that is treated as
# CLADE-CLONAL (a "stem" born on the clone's founding edge, inherited by the whole subtree) vs.
# PRIVATE (per-cell singletons). 0.35 was chosen empirically: high enough that the shared clade
# signal dominates the per-cell noise (so a passenger-only tree recovers the clone-tree clades),
# low enough to keep a realistic per-cell singleton tail (the SFS is not collapsed to pure clonal
# markers). The stem layer is what the old per-cell-independent reconstruction entirely lacked.
F_STEM_DEFAULT = 0.35
# Every clone-split that separates sampled cells should be MARKABLE, so each clone on a sampled
# lineage gets at least this many stem markers even when its folded burden rounds to zero.
STEM_FLOOR = 2
# Per-clone cap on stem markers, so the finite neutral-site budget spreads across many splits
# instead of saturating on one early high-burden clone (the genome-saturation headroom lever of
# DESIGN_snv_coalescent.md §4.4 / §7.3; raise the genome for deeper trees).
STEM_CAP = 6
# Dedicated RNG offset for the passenger reconstruction (added to ``self.seed``) — a fixed constant
# so the overlay is reproducible and independent of the growth stream.
PASSENGER_RNG_OFFSET = 909090

# --- Breast-cancer subtype presets ------------------------------------------------------------
# Selected with ``cell_params.cancer.subtype`` (absent -> no preset -> nothing changes). A preset
# only supplies DEFAULTS: any of the three knobs written explicitly in the config still wins, so a
# config can pick a subtype and then override one number.
#
# The receptor subtypes of breast cancer differ in how chromosomally unstable they are, and that
# difference shows up in exactly three measurable quantities, which are the three knobs here:
#
#   * ``cnv_prob``     — how much of the genome ends up at a non-diploid copy number. Oestrogen-
#                        receptor-positive (luminal) tumours are the quieter end of the spectrum
#                        (roughly a fifth of the genome altered); triple-negative (basal-like)
#                        tumours are the loud end (roughly twice that).
#   * ``wgd_rate``     — how often a whole-genome doubling happens. Doublings are common in breast
#                        cancer and markedly commoner in triple-negative disease than in luminal.
#   * ``mutation_rate`` — ongoing instability: the rate at which new clone-defining events arise at
#                        all, so how much subclonal structure the tumour keeps generating.
#
# Nothing else about the subtypes is modelled here — no receptor signalling, no subtype-specific
# driver genes, no treatment response. These are three instability dials and that is all.
BREAST_SUBTYPES = {
    # cnv_prob is the share of mutating divisions that change copy number. These values were briefly
    # tuned down while a defect quadrupled the mutation rate inside the confined founding patch; with
    # that removed they are back where they belong, and give the aneuploidy burden a real breast tumour
    # carries (about a third of the genome altered for the luminal subtype).
    "ER+": dict(cnv_prob=0.045, wgd_rate=0.001, mutation_rate=0.30),
    "TNBC": dict(cnv_prob=0.100, wgd_rate=0.006, mutation_rate=0.45),
}


class GenotypeTumor:
    """Genotype-level (count-based) tumour engine — the default, scalable iscc growth model.

    The tumour grows on a grid of demes (lattice patches); each deme is represented as
    **genotype counts** over a shared registry rather than one Python object per cell, so it
    is spatially explicit at the deme level while every birth / death / mutation / dispersal
    event is O(1) integer arithmetic on counts. It realises the same evolutionary process as
    the cell-level [`GlandularTumor`][iscc.tumor.GlandularTumor] (statistically equivalent,
    roughly two orders of magnitude faster) and is the class most users should reach for.
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
        ``maximum_death_rate`` and the density-dependent crowding law.
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
    microenv_params : dict, optional
        Microenvironment layer (hypoxia field + cell-cell communication). Off by
        default; a pure readout that never changes growth.
    expression_params : dict, optional
        Gene-program expression layer (``program_params``, ``activity_params``,
        ``coupling_params``, ``dosage_params``, ``snv_effect_params``). Off by default;
        a readout only.
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

    def __init__(self, config=None, seed=42, genome_params=None, selection_params=None,
                 cancer_cell_params=None, deme_params=None, spatial_params=None,
                 epithelial_cell_params=None, stromal_cell_params=None, immune_cell_params=None,
                 host_cell_params=None,
                 genome_mode="abstract", genome_spec=None,
                 update_mode="exact", tau=1.0, snapshot_every=1, microenv_params=None,
                 layout_seed=None, expression_params=None, coarsen_passengers=False,
                 max_cells=None, germline_params=None, founder_mutations=None):
        self.seed = seed
        # EVOLUTION rng (per-run: spatial seeding; grow() draws its own fresh default_rng(seed+step)).
        self.rng = np.random.default_rng(seed)
        # GENOME LAYOUT rng (config-determined, SHARED across same-config runs): the gene-role
        # layout (Selection) and shared per-cell-type baseline expression. Defaulting layout_seed to
        # DEFAULT_LAYOUT_SEED makes any two same-config runs share their driver identities by
        # construction (comparability by default) — recurrence / cohort analysis is meaningful — while
        # they still differ in evolution. See DESIGN_cohort.md §1. Byte-identical to the previous
        # single-rng plumbing at the default seed 42 (structure_radius=0 never consumes self.rng at
        # construction, and evolution never reads it).
        self.layout_seed = DEFAULT_LAYOUT_SEED if layout_seed is None else layout_seed
        self.layout_rng = np.random.default_rng(self.layout_seed)
        self.type = "genotype"

        # Update mode (DESIGN_scalability §7). "exact" = the reference one-birth/death-per-update
        # Gillespie engine (default, unchanged). "tau" = tau-leaping: advance ALL clones once per
        # discrete generation of length `tau` by drawing Poisson(rate*count*tau) births/deaths per
        # (deme, genotype), so wall-time scales with #clones x #generations rather than #cells.
        # `snapshot_every` records a full per-clone count snapshot every k generations (so
        # plot_muller/plot_grid keep working, now on a real-time x-axis -- see self.trace_times).
        self.update_mode = update_mode
        self.tau = tau
        self.snapshot_every = snapshot_every
        self.time = 0.0
        self.trace_times = []

        # Passenger coarsening (DESIGN_scalability §8): the lever that DECOUPLES growth cost from the
        # distinct-genotype count. Under a high mutation rate almost every division fixes a NEUTRAL
        # (passenger) SNV and so, in the exact model, spawns a brand-new genotype whose fitness is
        # identical to its parent's — making #genotypes track #cells and the whole growth loop
        # super-linear (see the grid-90 profile in §7). With ``coarsen_passengers`` on, a mutating
        # division whose daughter is genome_summary-identical to the parent (a pure passenger: no
        # driver hit, no copy-number change, no WGD) is NOT registered as a new genotype; it grows the
        # parent clone in place and its neutral burden is tallied in ``self._pass_load`` for lazy
        # reconstruction at materialisation. Only FITNESS/COPY-NUMBER-CHANGING events (driver SNVs,
        # CNVs, WGD) still create genotypes, so #genotypes tracks the number of distinct *clones*, not
        # cells. The dynamics are unchanged in distribution — a passenger daughter has identical
        # division/death/dispersal rates, so folding it into the parent leaves every future event rate
        # the same (selection fully intact). Passengers are re-emitted per sampled cell in
        # ``make_cell_data``. OFF by default -> byte-identical to the exact per-genotype engine.
        # Force-off when epistasis is on (an SNV anywhere in an event module can fire an event, so
        # "genome_summary unchanged" no longer implies "dynamically neutral").
        self.coarsen_passengers = bool(coarsen_passengers)

        # Assay-memory cap (DESIGN_scalability §9). ``make_cell_data`` materialises one row per cell
        # across ~a dozen dense (n_cells × n_genes) frames; at cm-scale (millions of cells) that is
        # tens of GB. ``max_cells`` bounds it: when the tumour exceeds the cap, a REPRESENTATIVE
        # subsample (Binomial(count, cap/total) per (deme, genotype) bucket, so spatial + clonal +
        # cell-type proportions are preserved) is materialised instead of every cell — a biopsy of one
        # tissue, which is exactly what an assay samples. ``None`` (default) materialises all cells, so
        # every existing small-tumour run/test is byte-identical. The re-scaled example_config sets it.
        self.max_cells = max_cells

        # Germline (inherited) variants: the patient's constant genetic background, carried by every
        # cell — tumour and normal alike — and so the purity / cancer-cell-fraction anchor a DNA
        # analysis calibrates against. A dict of the ``draw_germline_sites`` knobs, e.g.
        # ``{"het_frac": 5e-3, "hom_frac": 0.33, "seed": 7}``; ``seed`` defaults to this run's seed,
        # since a germline background is private to one individual. ``None`` (the default) means no
        # germline layer at all. The variants are drawn from a dedicated rng and written into the
        # founder genomes before any growth rng is touched, so a tumour is byte-identical with the
        # layer on or off at a given seed.
        self.germline_params = germline_params

        # Founder (truncal) mutations: what the transformed cell already carried at t=0 — the
        # clock-like burden of its normal lineage plus the driver hits that transformed it. A dict of
        # the ``seed_founder_mutations`` knobs, e.g. ``{"n_passenger": 60, "n_driver": 2, "seed": 7}``;
        # ``seed`` defaults to this run's seed. ``None`` (the default) leaves the founder genome blank,
        # which means NOTHING in the tumour can ever be clonal — the most recent common ancestor of
        # every cancer cell is the founder, so an empty founder genome is an empty clonal layer.
        self.founder_mutations = founder_mutations

        # Real-genome mode (DESIGN_inference A.5): the genome is wired from a GenomeSpec built
        # from human chromosome-arm data (arm lengths -> segment_sizes, per-arm oncogene/TSG
        # content) and selection uses the per-arm copy-number model (selection_mode="arm",
        # s_arm vector) instead of the abstract random gene-driver layout. The engine is
        # otherwise unchanged. ``selection_params`` (notably ``s_arm``) still wins, so the ABC
        # layer can inject the per-arm coefficients it is inferring.
        self.genome_mode = genome_mode
        self.genome_spec = genome_spec
        if genome_mode == "real":
            if genome_spec is None:
                raise ValueError("genome_mode='real' requires genome_spec")
            genome_params, selection_params = genome_spec.engine_params(selection_params)

        if config is not None:
            with open(config) as f:
                cfg = yaml.safe_load(f)
            genome_params = cfg["genome_params"]
            selection_params = cfg["selection_params"]
            deme_params = cfg["deme_params"]
            spatial_params = cfg["spatial_params"]
            cp = cfg["cell_params"]
            cancer_cell_params = cp["cancer"]
            epithelial_cell_params = cp.get("epithelial", {})
            stromal_cell_params = cp.get("stromal", {})
            immune_cell_params = cp.get("immune", {})
            host_cell_params = cp.get("host", {})
            self.update_mode = cfg.get("update_mode", update_mode)
            self.tau = cfg.get("tau", tau)
            self.snapshot_every = cfg.get("snapshot_every", snapshot_every)
            self.coarsen_passengers = bool(cfg.get("coarsen_passengers", coarsen_passengers))
            self.max_cells = cfg.get("max_cells", max_cells)
            self.germline_params = cfg.get("germline_params", germline_params)
            self.founder_mutations = cfg.get("founder_mutations", founder_mutations)
            microenv_params = cfg.get("microenv_params", microenv_params)
            expression_params = cfg.get("expression_params", expression_params)
            if cfg.get("layout_seed") is not None:
                self.layout_seed = cfg["layout_seed"]
                self.layout_rng = np.random.default_rng(self.layout_seed)
        # F8 microenvironment-driven expression (DESIGN_features §H): OPTIONAL and OFF by default
        # (absent -> None -> `make_cell_data` output is bit-identical to the base engine). When
        # enabled, a per-deme expression modifier (hypoxia field + cell-cell communication) is
        # applied at materialisation only; GROWTH is untouched, so a tumour is byte-identical with
        # F8 on or off at the same seed (the modifier draws from a dedicated rng). Fitness-coupling
        # of the microenvironment (hypoxia slowing division, etc.) is a FUTURE EXTENSION.
        self.microenv_params = microenv_params
        # R13 gene-program expression (DESIGN_expression.md): OPTIONAL and OFF by default, on exactly
        # the F8 discipline — absent -> None -> `make_cell_data` is bit-identical to the base engine,
        # and GROWTH never reads it at all, so a tumour is byte-identical with the program layer on or
        # off at a given seed. Programs are a READOUT: the genotype and the niche drive them, they
        # never feed back into fitness (that loop is R8b/R12-v3). Built after `self.selection` below,
        # since the model needs the gene count. See `iscc.tumor.programs`.
        self.expression_params = expression_params
        self.programs = None

        # Subtype preset (see BREAST_SUBTYPES). ``subtype`` is a cancer-cell config key that stands in
        # for three instability numbers; it is popped here so the cell class never sees it. Absent ->
        # the params dict is copied unchanged -> byte-identical.
        cancer_cell_params = dict(cancer_cell_params or {})
        subtype = cancer_cell_params.pop("subtype", None)
        if subtype is not None:
            if subtype not in BREAST_SUBTYPES:
                raise ValueError(
                    f"unknown cancer subtype {subtype!r}; choose one of "
                    f"{sorted(BREAST_SUBTYPES)} (or drop the key and set cnv_prob / wgd_rate / "
                    f"mutation_rate directly)")
            for k, v in BREAST_SUBTYPES[subtype].items():
                cancer_cell_params.setdefault(k, v)   # an explicit config value still wins
        self.subtype = subtype

        self.genome_params = genome_params
        self.n_segments = genome_params["n_segments"]
        self.segment_size = genome_params.get("segment_size", 1000)
        # Per-segment sizes (real-genome mode: proportional to chromosome-arm length); falls
        # back to a uniform scalar so the abstract mode is unchanged.
        self.segment_sizes = genome_params.get("segment_sizes")
        self._normal_params = {
            "epithelial": (EpithelialCell, epithelial_cell_params or {}),
            "stromal": (StromalCell, stromal_cell_params or {}),
            "immune": (ImmuneCell, immune_cell_params or {}),
            "host": (HostCell, host_cell_params or {}),
        }

        # ``layout_seed`` is handed over (as well as ``layout_rng``) so the optional epistasis network
        # can draw from its own layout SUB-STREAM — part of the shared landscape, but independent of
        # the gene-role layout drawn from layout_rng. See DESIGN_epistasis.md / constants.py.
        self.selection = Selection(n_segments=self.n_segments, segment_size=self.segment_size,
                                   segment_sizes=self.segment_sizes, rng=self.layout_rng,
                                   layout_seed=self.layout_seed, **selection_params)
        self.n_genes = self.selection.n_genes
        self._cancer_params = cancer_cell_params

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

        # R13 program layer. The dictionary (gene->program map, `loading`, regulators, `s_g`) is part
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

        # F8 program designation: pick the hypoxia-responsive and CCI-receptor-target gene sets
        # (the ground-truth cell-extrinsic modules). A DEDICATED rng (not self.rng) keeps the
        # growth trajectory byte-identical whether F8 is on or off at a given seed.
        # WHICH genes respond to hypoxia is a property of the GENOME, not of a run, so the stream is
        # the LAYOUT one: two patients in a cohort must share their niche programs exactly as they
        # share their oncogenes. (This previously read `self.seed + 9973` — the per-run EVOLUTION
        # seed — which silently gave every patient a different hypoxia programme; F8 predates
        # `layout_seed` and was never migrated. The dedicated-stream intent was right, the seed
        # source was wrong.)
        self._hypoxia_genes = np.array([], dtype=int)
        self._cci_target_genes = np.array([], dtype=int)
        # W0 (DESIGN_cci_spatial.md): iscc's OWN ligand-receptor database. `_cci_pairs` is
        # (n_candidate_pairs x 2) of gene indices (col 0 = ligand, col 1 = receptor); row 0 is the
        # WIRED pair F8's CCI channel is driven by, every other row is an unwired decoy. Empty (and
        # ligand/receptor None) unless a CCI channel is active, so an F8-off tumour is untouched.
        self._cci_pairs = np.empty((0, 2), dtype=int)
        self._cci_ligand = None
        self._cci_receptor = None
        if self.microenv_params:
            prog_rng = np.random.default_rng(self.layout_seed + LAYOUT_OFFSET_F8_PROGRAMS)
            hyp = ((self.microenv_params.get("hypoxia") or {}) if isinstance(self.microenv_params, dict) else {})
            cci = ((self.microenv_params.get("cci") or {}) if isinstance(self.microenv_params, dict) else {})
            n_hyp = int(hyp.get("n_genes", 0))
            n_cci = int(cci.get("n_target_genes", 0))
            if n_hyp > 0:
                self._hypoxia_genes = prog_rng.choice(self.n_genes, size=min(n_hyp, self.n_genes),
                                                      replace=False)
            if n_cci > 0:
                self._cci_target_genes = prog_rng.choice(self.n_genes, size=min(n_cci, self.n_genes),
                                                         replace=False)
            # Draw the L-R database from the SAME layout sub-stream, AFTER the target-gene draws so
            # those stay byte-identical (the pooling of patients onto ONE landscape is only justified
            # against a shared network — the epistasis-network argument). `n_candidate_pairs` is the
            # ONE new parameter (default 1 -> a database holding only the wired pair).
            # The WIRED pair's ligand and receptor are themselves MODULATED by the CCI field (see
            # `_microenv_deme_mod`): that is what puts the L-R signal in the two genes every CCI tool
            # actually reads. Only the hypoxia set is excluded from the pool, so a candidate's
            # expression is never moved by the unrelated hypoxia programme (which would boost a DECOY
            # and blur the benchmark); overlap with the CCI target set is harmless and allowed.
            if float(cci.get("strength", 0.0)) != 0.0 and n_cci > 0:
                n_pairs = max(1, int(cci.get("n_candidate_pairs", 1)))
                pool = np.setdiff1d(np.arange(self.n_genes), self._hypoxia_genes)
                k = min(2 * n_pairs, len(pool) - (len(pool) % 2))     # even: 2 distinct genes/pair
                if k >= 2:
                    picks = prog_rng.choice(pool, size=k, replace=False)
                    self._cci_pairs = picks.reshape(-1, 2).astype(int)
                    self._cci_ligand = int(self._cci_pairs[0, 0])
                    self._cci_receptor = int(self._cci_pairs[0, 1])

        # genotype registry: id -> representative cell. Normals keyed by their type name.
        self.genotypes = {}
        self.genotypes_parents = {}
        self.genotypes_counts = Counter()
        self._next_ord = 0
        # Drug-induced resistance STATE (DESIGN_phenotype_plasticity.md §3.3): the (genotype x epistate)
        # sublineage registry. Maps a genotype id -> a SHARED dict {state_level -> gid} holding every
        # state-variant of the SAME genome; a state transition of g to level s' looks up the twin here
        # and mints it (via _state_twin) only once, reusing it across generations, so the state costs a
        # small multiplier on the genotype count rather than one new id per transitioning cell. Empty and
        # never touched unless the feature is on (all population happens in the gated transition path).
        self._state_families = {}
        # Running max of (division + dispersal) over cancer genotypes, maintained incrementally in
        # _register so the tau-leap substep count doesn't need an O(#genotypes) scan every generation
        # (only under coarsen_passengers, where #genotypes can still be large; the exact-engine path
        # keeps its live-max scan so it stays byte-identical). Monotone upper bound -> conservative
        # (never too-few substeps).
        self._max_div_disp = 0.0

        # per-immune-cell contact kill hazard, and per-step treatment overrides
        # (gid -> extra death hazard / overridden immune resistance), refreshed each
        # treated step by _apply_treatment so they never corrupt the shared genotype.
        self._immune_prob_kill = (immune_cell_params or {}).get("prob_kill", 0.01)
        self._tx_death_add = {}
        self._tx_immune_resist = {}
        # The active dose's context, kept alongside the override dicts so the hazard for a genotype
        # MINTED MID-STEP (by a division-mutation during update / _tau_generation, after this step's
        # _apply_treatment already froze the dicts) can be computed on demand rather than defaulting
        # to zero. Without this a newborn clone escapes the drug for one whole step -- worst exactly
        # in therapy-induced-mutagenesis runs, where mid-step minting is most frequent. None => no
        # active treatment => the on-demand paths return the untreated defaults (byte-identical).
        self._tx_treatment = None
        self._tx_dosage = 0.0
        # Per-genotype division multiplier that applies ONLY while the drug is on (drug-tolerant
        # persisters). The tolerant state is DRUG-INDUCED: off drug a tolerant cell is an ordinary
        # cell and pays nothing, so this cost cannot live in `division_rate`, which is baked at
        # genotype level and would apply everywhere and always -- purging the tolerant pool long
        # before the first dose. Empty dict => no override => byte-identical.
        self._tx_div_mult = {}
        self._tx_sites = "both"  # active treatment's target compartment(s): both / met / primary (R9)
        # Compartment-dependent selection (v1, DESIGN_phenotype_plasticity.md §2). Two local hazards,
        # each contributed by a resident normal compartment the gland geometry seeds and each
        # attenuated by a MATCHING heritable trait (breach / stromal_survival), exactly the shape of
        # the immune term above. This is the whole "payoff table": edit these two coefficients (+ the
        # two prop_/effects axes in selection_params) in config, not the engine. DEFAULT 0.0 -> the
        # terms vanish -> growth is byte-identical to before (the F8 off-by-default discipline).
        self._epithelial_barrier = (spatial_params or {}).get("epithelial_barrier", 0.0)
        self._stromal_hazard = (spatial_params or {}).get("stromal_hazard", 0.0)
        # Breach-GATED invasion (DESIGN_ductal_field.md §3.2): with this on, crossing the basement
        # membrane — a dispersal hop from a DUCT deme into a STROMA deme — succeeds only in proportion to
        # the clone's heritable `breach` trait; a cell that fails stays confined in the duct (DCIS). This
        # makes breach a true GATE (only breach clones enter the stroma) rather than the epithelial_barrier
        # death-hazard, which merely disadvantages non-breach cells in the duct and so leaks. DEFAULT
        # False -> the duct->stroma hop is an ordinary neighbour dispersal (byte-identical to before).
        self._breach_gated_invasion = bool((spatial_params or {}).get("breach_gated_invasion", False))

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
            **cancer_cell_params,
        )
        founder.set_genotype_id()
        self._register(founder)
        self.founder_id = founder.genotype_id

        # grid of demes; each deme is a dict {genotype_id: count}
        self.grid_size = spatial_params["grid_size"]
        # carrying_capacity is a real per-deme CAP now (DESIGN_crowding.md, Option A): crowding
        # death rises RELATIVE to each clone's own evolved division rate, so demes cap near K even
        # for fast-evolved clones. Setting it to None or 0 disables crowding entirely -> the
        # "well-mixed" regime (unbounded growth in a deme), used by the single-deme SISTEM benchmark.
        self.carrying_capacity = deme_params.get("carrying_capacity", 10)
        self._crowding = self.carrying_capacity is not None and self.carrying_capacity > 0
        # Positive fill/normalisation capacity: used for structure filling, immune seeding and the
        # microenvironment density fields. Falls back to 1 when crowding is off (no capacity).
        self._cap = int(self.carrying_capacity) if self._crowding else 1
        # Firmness of the cap: the crowding slope is steepened by (1 + crowding_margin) so the
        # per-deme fixed point sits at K/(1+margin) (slightly below K) and the restoring force above
        # K is firm rather than marginal. 0 recovers the plain fixed-point-at-K form. See _death_rate.
        self.crowding_margin = deme_params.get("crowding_margin", 0.1)
        # maximum_death_rate MUST be >= max_birth_rate for the cap to bind: a clone whose division
        # evolved up to max_birth_rate needs its crowding death to be able to reach (and exceed) that
        # rate. Default 1.0 (>= the 0.8 max_birth_rate default). A lower value re-opens the overfill bug.
        self.maximum_death_rate = deme_params.get("maximum_death_rate", 1.0)
        # Resident-pressure reference division rate (DESIGN_crowding.md, invasion gate). A deme's
        # immortal normal cells (epithelial/stromal) contribute to a cancer cell's crowding death at
        # this FIXED reference rate rather than at the cancer cell's own evolved division rate — so
        # their contribution does NOT cancel in the survival condition (net = div - death > 0) and
        # becomes a genuine FITNESS THRESHOLD a cancer clone must clear to establish (and then invade)
        # a normal-occupied gland deme. Defaults to the founder cancer division rate: a clone must be
        # at least as fit as a baseline cancer cell to hold a slot against a resident. Cancer-only
        # demes (no normal cells) are byte-identical to before (the term is zero there). Raise it for
        # a stricter invasion gate.
        self._resident_ref = deme_params.get(
            "resident_pressure_ref",
            self.genotypes[self.founder_id].baseline_rates["division_rate"])
        # Crowding law (DESIGN_crowding.md). "own" = Option A (own-division cancer term + fixed-ref
        # resident term). "fixed" = a single UNIFIED law: crowding death rises with TOTAL occupancy
        # relative to a FIXED reference (`crowding_ref`, default max_birth_rate), NOT the clone's own
        # division. => near-neutral where there is space (low density) and fitness-selective where
        # crowded (the invasion border), from one context-free rule; nothing exceeds the reference so
        # demes still cannot overfill. Subsumes the resident-pressure gate (normals count in `total`).
        # "lottery" = the structural-cap law (DESIGN_crowding_v2.md), PROTOTYPE, off by default:
        # the crowding death is UNIFORM across the clones of a deme (so the fitness term cannot
        # cancel) and sits BELOW the deme's mean net growth, while the density cap is enforced
        # STRUCTURALLY -- a deme has floor(K) slots, a birth that cannot get one simply fails, and
        # when more births are drawn than there are slots the slots go to clones in proportion to the
        # births they drew. Rate law sets TURNOVER, the cap sets DENSITY, and neither has to
        # compromise for the other, so demes sit at K *and* clones compete inside them.
        self._crowding_mode = deme_params.get("crowding_mode", "own")
        self._crowding_ref = deme_params.get(
            "crowding_ref", getattr(self.genotypes[self.founder_id], "max_birth_rate", 0.98))
        # rho: the uniform crowding pressure as a FRACTION of the deme's mean net growth. Must be < 1
        # -- the rate law alone has to want to overfill, otherwise the structural cap never binds and
        # the whole law is inert (DESIGN_crowding_v2.md §8, amendment 1).
        self._crowding_turnover = float(deme_params.get("crowding_turnover", 0.6))
        # Deme-model crowding (Noble et al. 2022, Nat Ecol Evol, "Within-deme dynamics"): the
        # within-deme death rate is a TWO-VALUED STEP, not a continuous function of density —
        #   "When the deme population size is less than or equal to the carrying capacity, the death
        #    rate takes a fixed value d0 that is less than the initial division rate. When the deme
        #    population size exceeds carrying capacity, the death rate takes a different fixed value
        #    d1 that is much greater than the largest attainable division rate."
        # d1 being above the LARGEST ATTAINABLE division rate is what makes the cap unbreakable: it
        # is the exact property the 2026-07 overfill bug lacked, where the death cap sat BELOW the
        # evolved division rate. Both values are constants, so the law carries no `div` and cannot
        # cancel out of net = div - death — clone-vs-clone competition survives for free.
        # `crowding_law="logistic"` (default) keeps the shipped ramp; "noble" selects the step.
        self._crowding_law = str(deme_params.get("crowding_law", "logistic"))
        if self._crowding_law not in ("logistic", "noble"):
            raise ValueError("crowding_law must be 'logistic' or 'noble'")
        # d0 defaults to the clone's own baseline death rate (Noble uses a single global constant,
        # and sets it to 0 in the spatial simulations); d1 to maximum_death_rate, which the shipped
        # configs already hold above max_birth_rate.
        self._crowding_d0 = deme_params.get("crowding_d0")          # None -> the clone's own base
        self._crowding_d1 = deme_params.get("crowding_d1")          # None -> maximum_death_rate
        self._crowding_overfill = float(deme_params.get("crowding_overfill", 1.0))
        if self._crowding_overfill < 1.0:
            raise ValueError("crowding_overfill must be >= 1 (1 = the slot cap sits exactly on K)")
        if self._crowding_law == "noble":
            # d1 MUST exceed the largest attainable division rate, which is what makes the cap
            # unbreakable however far selection pushes birth rates up. This is not a style check:
            # the 2026-07 overfill bug was exactly a density cap expressed as a rate that sat BELOW
            # the evolved division rate (`maximum_death_rate` 0.5 against an evolved 0.8), and the
            # tumour piled 1,200-4,200 cells into demes of nominal capacity 10. Fail loudly at
            # construction rather than silently overfill after thousands of generations.
            _d1 = self.maximum_death_rate if self._crowding_d1 is None else float(self._crowding_d1)
            # The EFFECTIVE d1 is what survives the `min(death, maximum_death_rate)` clamp at the end
            # of _death_rate. Checking the requested value alone would wave through exactly the 2026-07
            # configuration — a large d1 silently clipped back below the evolved division rate.
            _eff = min(_d1, self.maximum_death_rate)
            _max_b = float(getattr(self.genotypes[self.founder_id], "max_birth_rate", 0.0))
            if _eff <= _max_b:
                raise ValueError(
                    f"crowding_law='noble' needs d1 > the largest attainable division rate, but the "
                    f"EFFECTIVE d1 is {_eff:g} <= max_birth_rate={_max_b:g} "
                    f"(requested d1={_d1:g}, clamped by maximum_death_rate={self.maximum_death_rate:g}). "
                    f"Raise BOTH crowding_d1 and maximum_death_rate above max_birth_rate — a density "
                    f"cap expressed as a rate below the evolved division rate cannot bound a deme.")
        # Which rate the uniform pressure references: the deme's own cell-weighted mean cancer
        # division rate ("deme_mean", RELATIVE purifying selection -- a costly clone is disadvantaged,
        # not lethal) or a fixed config scalar ("fixed" = `crowding_ref`, ABSOLUTE purifying selection).
        self._crowding_reference = deme_params.get("crowding_reference", "deme_mean")
        # Cancer replaces normal tissue: a birth with no free slot evicts one resident. Without this a
        # duct WALL deme (seeded full of immortal epithelium at K_duct) has zero slots forever and the
        # DCIS->IDC arc is structurally impossible. Immune cells hold slots but are NOT evictable by
        # default (that would make immune escape structurally free).
        self._evict_residents = bool(deme_params.get("evict_residents", True))
        self._evict_immune = bool(deme_params.get("evict_immune", False))
        # The well-mixed regime (carrying_capacity None/0) has no capacity, hence no slots: it stays
        # unbounded whatever crowding_mode says.
        self._lottery = bool(self._crowding and self._crowding_mode == "lottery")
        if self._lottery and not 0.0 < self._crowding_turnover < 1.0:
            raise ValueError("crowding_turnover must be in (0, 1) — at rho >= 1 the uniform crowding "
                             "pressure exceeds the deme's mean net growth and demes UNDER-fill; at "
                             "rho <= 0 there is no turnover and nothing ever competes.")
        self.structure_radius = spatial_params.get("structure_radius", 0)
        # Ductal-field substrate (DESIGN_ductal_field.md): the structured case is a FIELD of many small
        # epithelial-ring glands at 2D positions in moderate-density stroma (an island model), not one
        # central ring. OFF-BY-DEFAULT: n_glands=1 + gland_radius=structure_radius + stroma_fill_frac=1.0 +
        # K_duct=K_stroma=carrying_capacity reproduces the single-central-ring seeding byte-identically.
        self.n_glands = int(spatial_params.get("n_glands", 1))
        self.gland_radius = int(spatial_params.get("gland_radius", self.structure_radius))
        self.min_gland_sep = spatial_params.get("min_gland_sep", 2 * self.gland_radius + 2)
        self.stroma_fill_frac = float(spatial_params.get("stroma_fill_frac", 1.0))
        # Per-compartment carrying capacity (K captures the duct's/stroma's 3D depth — a deme is a 3D
        # column, so K is MODERATE-TO-LARGE, not a handful). Default to the scalar carrying_capacity so
        # a uniform-K field is byte-identical to the current single-K crowding law.
        self._cap_duct = int(spatial_params.get("K_duct", self._cap)) if self._crowding else 1
        self._cap_stroma = int(spatial_params.get("K_stroma", self._cap)) if self._crowding else 1
        # Cross-gland (island) dispersal: a low rate that seeds one gland's lumen from another's,
        # abstracting intraductal spread through the out-of-plane ductal tree. kappa=0 -> OFF ->
        # byte-identical. lambda (distance kernel over gland centres); None -> uniform targeting.
        self.cross_gland_kappa = float(spatial_params.get("cross_gland_kappa", 0.0))
        self.cross_gland_lambda = spatial_params.get("cross_gland_lambda", None)
        # Ground-truth ductal-field labels, populated by _seed_structure (None when unstructured).
        self.gland_id = None
        self.gland_centers = None
        self.gland_lumen_demes = None

        # --- Origin confinement (OFF by default) ------------------------------------------------
        # A breast lesion starts inside ONE intact acinus: the myoepithelial layer is continuous and
        # the basement membrane is unbroken, so a daughter cell has essentially nowhere to go and the
        # founding patch recycles its cells in place. That containment is a property of EARLY,
        # architecturally intact tissue: as the lesion grows it distends the duct and the myoepithelial
        # layer is progressively lost (Risom et al., Cell 2022, measured that disruption across 79
        # resections), after which cells leave the patch as freely as anywhere else.
        #
        # Modelling it matters because it is what lets an early alteration become TRUNCAL. A
        # copy-number change that fixes in the founding patch before any cell has left is carried by
        # every cell that ever exists, because every later cell descends from that patch. At the
        # ordinary dispersal rate cells leave within a couple of generations, long before anything can
        # fix, and the tumour ends up with no clonal copy-number layer at all.
        #
        # The containment is deliberately ORIGIN-SPECIFIC and TRANSIENT. Applying it to every duct
        # would be wrong biology and wrong data: in-situ breast lesions are commonly POLYCLONAL (in
        # Casasent et al., Cell 2018, 6 of 10 lesions profiled at single-cell resolution), so a duct
        # must stay free to be colonised by several clones.
        #
        # Config (``spatial_params.origin_confinement``), a dict; absent -> None -> nothing at all
        # happens and growth is byte-identical:
        #   dispersal_factor : dispersal rate inside the founding patch, as a MULTIPLE of the
        #                      configured ``dispersal_rate`` (0 => a closed acinus, nothing gets out).
        #                      A multiplier, not an absolute rate, so an evolved dispersal trait still
        #                      shows through. It acts on the DAUGHTER'S DESTINATION, not on the cell's
        #                      rates: a would-be disperser that cannot leave competes for a slot where
        #                      it is (``_escape_factor``). Deliberately so — confinement must not
        #                      change how often a division mutates. Scaling the dispersal rate inside
        #                      the mutate-or-disperse denominator instead would send the mutation
        #                      probability to 1.0 in a closed acinus (a silent 4x at the shipped
        #                      rates), which is an artefact of how the fate is drawn, not biology.
        #   established_at   : the countdown does not start until the founding patch is at least this
        #                      fraction of its carrying capacity — an acinus with a handful of cells in
        #                      it is not distending anything yet. Fractions of K, so it means the same
        #                      thing at every grid size.
        #   generations      : generations of distension after that before the architecture gives way
        #                      and the patch reverts to the ordinary dispersal rate. One-way: once
        #                      released, never re-confined.
        _oc = spatial_params.get("origin_confinement", None)
        self._origin_confinement = _oc
        self._origin_demes = ()          # deme(s) holding the founder at t=0; set by the seeding below
        self._origin_confined = _oc is not None
        self._origin_age = None          # generations since the patch filled; None = not filled yet
        if _oc is not None:
            self._origin_disp_factor = float(_oc.get("dispersal_factor", 1e-4))
            self._origin_established_at = float(_oc.get("established_at", 0.5))
            self._origin_generations = float(_oc.get("generations", 200))
            if self._origin_disp_factor < 0:
                raise ValueError("origin_confinement.dispersal_factor must be >= 0")
        # Number of founder cancer cells to seed (an established micro-lesion). A single founder
        # has P(extinction) ≈ death/division (~7% for the defaults) regardless of carrying
        # capacity, so a one-cell start makes runs/demos randomly cancer-free; seeding a small
        # cluster removes that founder bottleneck. Default 1 preserves prior behaviour.
        self.initial_cancer_cells = deme_params.get("initial_cancer_cells", 1)
        # Founder cells actually seeded: the requested cluster, capped by K when crowding is on
        # (can't over-seed a deme past its capacity) and left uncapped in the well-mixed regime.
        self._n_founder = max(1, min(self.initial_cancer_cells, self._cap)
                              if self._crowding else self.initial_cancer_cells)
        # Metastasis module (R9): a SECOND deme-grid — the metastatic deposit — appended to
        # self.demes, sharing the genotype registry + genotypes_parents genealogy + Selection with the
        # primary (so a clone keeps ONE identity/colour across both grids, enabling the 2-band Muller).
        # The met is a homogeneous met_grid_size x met_grid_size lattice filled with immortal `host`
        # parenchyma; one fixed `vessel` deme is the migration entry point (and O2 point-source).
        # OFF by default (met_grid_size=0 -> _seed_met is a no-op -> nothing appended -> byte-identical).
        self.met_grid_size = int(spatial_params.get("met_grid_size", 0))
        self._met_enabled = self.met_grid_size > 0
        self._cap_met = int(spatial_params.get("K_met", self._cap)) if self._crowding else 1
        self._host_fill_frac = float(spatial_params.get("host_fill_frac", 1.0))
        _mv = spatial_params.get("met_vessel")
        self.met_vessel = (tuple(_mv) if _mv is not None
                           else (self.met_grid_size // 2, self.met_grid_size // 2))
        self.met_vessel_idx = None
        # Host-tissue death hazard (attenuated by met_survival; used in _death_rate, Step 3) and the
        # migration knobs (seed rate + transit-survival floor; used in the dispersal branch, Step 4).
        # All default to 0 -> inert -> byte-identical.
        self._met_hazard = spatial_params.get("met_hazard", 0.0)
        self._met_seed_kappa = float(spatial_params.get("met_seed_kappa", 0.0))
        self._met_transit_floor = float(spatial_params.get("met_transit_floor", 0.0))
        # Discrete-event annotation log (met seeding, resection, chemo windows) -> events.csv, used to
        # annotate the 2-band Muller. Each entry is a dict with at least step/time/event.
        self.events = []
        self._has_host = False

        n_demes = self.grid_size * self.grid_size
        self.n_primary_demes = n_demes
        self.demes = [dict() for _ in range(n_demes)]
        # deme index -> its live CANCER cell count, present only while positive. Maintained by
        # _add/_remove (the only two routes cells take in or out of a deme) so a tau substep can
        # iterate the cancer-occupied demes instead of all grid_size**2 of them (_active_demes).
        self._deme_cancer_n = {}
        self.deme_coords = [(i // self.grid_size, i % self.grid_size) for i in range(n_demes)]
        # Per-deme carrying capacity (DESIGN_ductal_field.md §3): uniform = carrying_capacity by
        # default (byte-identical to the scalar law); _seed_structure overwrites duct demes with
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
        # fraction is not washed out once a deme fills with cancer. Static for now
        # (no division/migration yet).
        self.immune_density = spatial_params.get("immune_density", 0.0)
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
        # never touches the growth stream, so `germline_params=None` leaves this an early return and
        # the run byte-identical to one built before the layer existed.
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
        # the germline it uses its own seeded rng, so `founder_mutations=None` is an early return and
        # the run is byte-identical to one built before the layer existed.
        self.seeded_truncal_sites = np.array([], dtype=int)
        self.seeded_truncal_homolog = np.array([], dtype="<U4")
        self.seeded_truncal_is_driver = np.array([], dtype=bool)
        if self.founder_mutations:
            fm = dict(self.founder_mutations)
            fm.setdefault("seed", self.seed)
            seed_founder_mutations(self, **fm)

        self.deme_rates = np.array([self._deme_rate(i) for i in range(len(self.demes))], dtype=float)
        self.traces = []
        self.step = 0
        self.cell_data = None

    # --- genotype registry ---------------------------------------------------
    def _is_viable(self, rep):
        """Whether a freshly mutated genotype satisfies the CINner viability limits
        (``max_ploidy`` / ``max_cn`` / ``max_nullisomy`` / ``max_mut_drivers``; see
        Selection.update_viability). A non-viable daughter is REJECTED AT BIRTH: the division is
        still consumed, but it yields no cell and no genotype is registered.

        WHY REJECT-AT-BIRTH. The alternative is to check lazily, when a cell is next *sampled* for
        an event -- which is what the cell-level engine used to do. Both rules agree on the long-run
        dynamics: the division yields no surviving descendant either way, and the event is consumed
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
        construction, in this engine and the cell engine alike.

        BOTH ENGINES NOW REJECT AT BIRTH. ``Deme.apply_event`` applies the same rule for the
        cell-level engine (its ``sample_event`` viability check survives only as a backstop), so the
        limits are a true invariant everywhere rather than a property of this engine alone. That
        change was a no-op at the shipped defaults, where the limits are never reached -- see
        tests/test_count_engine.py::test_cell_engine_default_limits_are_a_noop.
        """
        return self.selection.update_viability(rep.genome_summary) != 0

    def _register(self, rep):
        rep.ord = self._next_ord
        self._next_ord += 1
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

    # --- structure seeding: the ductal field (DESIGN_ductal_field.md §2) -------
    def _place_glands(self):
        """Gland centres in the grid interior, >= min_gland_sep apart, ring fully inside the grid.

        n_glands==1 short-circuits to the grid centre and draws NO rng, so the single-gland field is
        byte-identical to the old single central ring (the founder draw below is then the first rng
        consumption, exactly as before). n_glands>1 rejection-samples from ``self.rng`` (the LAYOUT rng,
        so the field layout is cohort-comparable like every other make_*)."""
        c0 = self.grid_size // 2
        if self.n_glands <= 1:
            return [(c0, c0)]
        lo = self.gland_radius + 1
        hi = self.grid_size - self.gland_radius - 1
        centers = []
        attempts = 0
        max_attempts = 200 * self.n_glands
        while len(centers) < self.n_glands and attempts < max_attempts:
            attempts += 1
            r = int(self.rng.integers(lo, hi)) if hi > lo else c0
            c = int(self.rng.integers(lo, hi)) if hi > lo else c0
            if all((r - rr) ** 2 + (c - cc) ** 2 >= self.min_gland_sep ** 2 for (rr, cc) in centers):
                centers.append((r, c))
        return centers

    def _seed_structure(self):
        """Seed a FIELD of small epithelial-ring glands in moderate-density stroma, one cancer founder in gland
        0's lumen. n_glands=1 + gland_radius=structure_radius + stroma_fill_frac=1.0 +
        K_duct=K_stroma=carrying_capacity reproduces the old single-central-ring seeding byte-for-byte."""
        n_demes = len(self.demes)
        self.gland_centers = self._place_glands()
        self.gland_id = np.full(n_demes, -1, dtype=int)
        self.gland_lumen_demes = []
        epi = self._normal_genotype("epithelial")
        occupied = set()
        for gi, (cr, cc) in enumerate(self.gland_centers):
            border = bresenham_circumference(cr, cc, self.gland_radius)
            for (r, c) in border:
                if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                    di = r * self.grid_size + c
                    self._add(di, epi, self._cap_duct)
                    self.gland_id[di] = gi
                    if self._deme_capacity is not None:
                        self._deme_capacity[di] = self._cap_duct
            circle = get_inside(border)
            lumen = []
            for (r, c) in circle:
                if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                    di = r * self.grid_size + c
                    self.gland_id[di] = gi
                    if self._deme_capacity is not None:
                        self._deme_capacity[di] = self._cap_duct
                    lumen.append(di)
            self.gland_lumen_demes.append(lumen)
            occupied |= set(border) | set(circle)

        # founder: one micro-lesion in gland 0's lumen (same rng draw as the old single ring)
        cr, cc = self.gland_centers[0]
        in_border = bresenham_circumference(cr, cc, self.gland_radius - 1)
        in_border = [(r, c) for (r, c) in in_border if 0 <= r < self.grid_size and 0 <= c < self.grid_size]
        if in_border:
            pos = in_border[int(self.rng.choice(len(in_border)))]
            origin = pos[0] * self.grid_size + pos[1]
            self._add(origin, self.founder_id, self._n_founder)
            # The founding acinus — the only deme origin confinement ever applies to.
            self._origin_demes = (origin,)

        # stroma fills the rest, seeded at MODERATE density (stroma_fill_frac≈0.3-0.5: real stromal
        # cells that carry the stromal hazard as a LIVE fraction, DESIGN_ductal_field.md §5, with
        # headroom for an invasive mass). stroma_fill_frac=1.0 (the default) recovers the old behaviour.
        stroma = self._normal_genotype("stromal")
        n_stroma = int(round(self.stroma_fill_frac * self._cap_stroma))
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                di = r * self.grid_size + c
                if (r, c) not in occupied and not self.demes[di]:
                    if self._deme_capacity is not None:
                        self._deme_capacity[di] = self._cap_stroma
                    if n_stroma > 0:
                        self._add(di, stroma, n_stroma)

    def _seed_met(self):
        """Seed the metastatic deposit (R9): a met_grid_size x met_grid_size lattice appended to
        self.demes, every deme filled to K_met with immortal/static `host` parenchyma, plus one fixed
        `vessel` deme (the migration entry point and O2 point-source). No internal structure. A no-op
        when met_grid_size == 0 (nothing appended -> byte-identical)."""
        if not self._met_enabled:
            return
        G = self.met_grid_size
        n_met = G * G
        base = self.n_primary_demes
        self.demes.extend(dict() for _ in range(n_met))
        # met demes carry met-LOCAL coords (their position in the met lattice), so _neighbors and
        # make_cell_data address them within the met grid, not the concatenated global index.
        self.deme_coords.extend((i // G, i % G) for i in range(n_met))
        if self._deme_capacity is not None:
            self._deme_capacity = np.concatenate(
                [self._deme_capacity, np.full(n_met, self._cap_met, dtype=float)])
        # Host tissue has no glands: extend the ground-truth gland label with -1 for every met deme so
        # gland_id stays index-aligned with self.demes (make_cell_data and the stromal-gating in the
        # migration operator both index it).
        if self.gland_id is not None:
            self.gland_id = np.concatenate([self.gland_id, np.full(n_met, -1, dtype=int)])
        vr, vc = self.met_vessel
        self.met_vessel_idx = base + vr * G + vc
        # Host baseline expression from a DEDICATED layout sub-stream (NOT by adding "host" to
        # CELLTYPES, which would draw an extra layout_rng.beta and shift the shared landscape). Off-met
        # runs never reach here, so the landscape is byte-identical whether or not the met exists.
        mrng = np.random.default_rng(self.layout_seed + LAYOUT_OFFSET_MET)
        exp = mrng.beta(0.1, 1.0, size=self.n_genes)
        exp[self.selection.get_tsgs()] = 0.8
        exp[self.selection.get_oncogenes()] = 0.01
        self.celltype_exps["host"] = exp
        # Fill each met deme with immortal/static host cells at host_fill_frac * K_met (leave headroom
        # for the invading deposit, like stroma_fill_frac).
        host = self._normal_genotype("host")
        n_host = int(round(self._host_fill_frac * self._cap_met))
        if n_host > 0:
            for i in range(base, base + n_met):
                self._add(i, host, n_host)

    # --- count bookkeeping ---------------------------------------------------
    def _add(self, deme_idx, gid, n):
        deme = self.demes[deme_idx]
        deme[gid] = deme.get(gid, 0) + n
        self.genotypes_counts[gid] += n
        if self.genotypes[gid].type == "cancer":
            self._deme_cancer_n[deme_idx] = self._deme_cancer_n.get(deme_idx, 0) + n

    def _remove(self, deme_idx, gid, n):
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

    # --- structural density cap: slots + eviction (DESIGN_crowding_v2.md §3.2/§3.4) -----------
    #: Resident cell types a cancer birth may DISPLACE to take a slot. Invasive carcinoma replaces
    #: normal tissue — a duct effaced by DCIS is the canonical picture — so epithelium, stroma and
    #: metastatic host parenchyma are evictable. Immune cells are not (see `evict_immune`): letting a
    #: tumour structurally clear its own immune pressure would trivialise immune escape.
    _EVICTABLE = ("epithelial", "stromal", "host")

    def _cap_of(self, deme_idx):
        """This deme's integer slot count — its carrying capacity (per-deme K where the ductal field
        set one, the scalar ``carrying_capacity`` otherwise)."""
        K = (self.carrying_capacity if self._deme_capacity is None
             else self._deme_capacity[deme_idx])
        return int(K)

    def _slot_cap_of(self, deme_idx):
        """The deme's structural slot count — ``crowding_overfill`` x its carrying capacity.

        Separate from :meth:`_cap_of`, which stays the DENSITY REFERENCE K used by the crowding law.
        At the default overfill of 1.0 the two coincide and a deme is hard-capped at K. Above 1.0 the
        deme may overshoot into the accelerating region of the density term, which is what makes the
        Noble-style over-capacity response reachable instead of dead code."""
        return int(self._cap_of(deme_idx) * self._crowding_overfill)

    def _evictable_gids(self, deme_idx):
        """The deme's displaceable resident genotypes, in creation-ordinal order (deterministic)."""
        if not self._evict_residents:
            return []
        deme = self.demes[deme_idx]
        types = self._EVICTABLE + (("immune",) if self._evict_immune else ())
        return sorted((g for g in deme if self.genotypes[g].type in types),
                      key=lambda g: self.genotypes[g].ord)

    def _n_evictable(self, deme_idx):
        """How many cells in this deme a birth could displace. Zero when eviction is off — the slot
        allocator sizes its second (eviction) draw from this, so it MUST agree with ``_evict``."""
        if not self._evict_residents:
            return 0
        deme = self.demes[deme_idx]
        types = self._EVICTABLE + (("immune",) if self._evict_immune else ())
        return sum(c for g, c in deme.items() if self.genotypes[g].type in types)

    def _evict(self, deme_idx, n, rng):
        """Displace ``n`` resident cells from ``deme_idx``, drawn across the resident genotypes in
        proportion to their counts. Returns the number actually removed (< n only if the deme runs
        out of residents). One resident genotype — the overwhelmingly common case — costs no rng
        draw."""
        if n <= 0 or not self._evict_residents:
            return 0
        gids = self._evictable_gids(deme_idx)
        if not gids:
            return 0
        deme = self.demes[deme_idx]
        if len(gids) == 1:
            k = min(n, deme[gids[0]])
            self._remove(deme_idx, gids[0], k)
            return k
        counts = np.array([deme[g] for g in gids], dtype=np.int64)
        take = min(n, int(counts.sum()))
        drawn = rng.multivariate_hypergeometric(counts, take)
        for g, k in zip(gids, drawn):
            if k:
                self._remove(deme_idx, g, int(k))
        return take

    def _can_place(self, deme_idx, total=None):
        """Whether a newborn cell can acquire a slot in this deme — either a free one or one taken
        from a displaceable resident."""
        if total is None:
            total = sum(self.demes[deme_idx].values())
        if total < self._slot_cap_of(deme_idx):
            return True
        return self._evict_residents and self._n_evictable(deme_idx) > 0

    def _place_one(self, deme_idx, gid, rng):
        """Place ONE newborn cell, taking a free slot or displacing one resident. Returns whether it
        was placed; a birth into a full deme with no displaceable resident simply FAILS (the division
        is consumed, no cell is added), exactly like a daughter that breaches the viability limits.
        There is deliberately NO redraw of a rejected dispersal target — that would be free-space-
        biased directional dispersal, a different model."""
        if not self._lottery:
            self._add(deme_idx, gid, 1)
            return True
        if sum(self.demes[deme_idx].values()) >= self._slot_cap_of(deme_idx) and not self._evict(deme_idx, 1, rng):
            return False
        self._add(deme_idx, gid, 1)
        return True

    def _cross_gland_target(self, src_gland, rng):
        """A lumen deme of ANOTHER gland, for an island (cross-gland) dispersal hop
        (DESIGN_ductal_field.md §4). Glands are chosen distance-weighted (prob ~ exp(-d/lambda) over
        gland centres from the source) or uniformly when ``cross_gland_lambda`` is None; the daughter
        lands in a random lumen deme of the chosen gland — lumen->lumen, bypassing the wall, so a
        confined intraductal hop needs no breach. Returns None if no other gland has a lumen deme."""
        if self.gland_centers is None or len(self.gland_centers) <= 1:
            return None
        others = [g for g in range(len(self.gland_centers))
                  if g != src_gland and self.gland_lumen_demes[g]]
        if not others:
            return None
        if self.cross_gland_lambda is None:
            g = others[int(rng.integers(0, len(others)))]
        else:
            sr, sc = self.gland_centers[src_gland]
            d = np.array([np.hypot(self.gland_centers[g][0] - sr, self.gland_centers[g][1] - sc)
                          for g in others])
            w = np.exp(-d / max(float(self.cross_gland_lambda), 1e-9))
            g = others[int(rng.choice(len(others), p=w / w.sum()))]
        lumen = self.gland_lumen_demes[g]
        return lumen[int(rng.integers(0, len(lumen)))]

    def _transit_prob(self, met_survival):
        """Probability a migrating daughter survives transit to the met (R9), biased by the clone's
        heritable met_survival: met_transit_floor + (1 - met_transit_floor) * met_survival. A low floor
        makes wild-type cells (met_survival 0) rarely survive and high-met_survival clones survive often,
        so the deposit's founders are ENRICHED for met_survival — the transit bottleneck that makes the
        met a non-representative, often-minor sample of the primary's clones."""
        return self._met_transit_floor + (1.0 - self._met_transit_floor) * met_survival

    def _neighbors(self, deme_idx):
        # Compartment-aware von Neumann neighbourhood: neighbours stay within the SAME grid. Met demes
        # are a contiguous tail block with their own met_grid_size lattice and met-local coords, so a
        # cancer cell dispersing inside the deposit never leaks into the primary (and vice versa). When
        # met is off every deme_idx < n_primary_demes, so this is the original primary-grid computation.
        if deme_idx < self.n_primary_demes:
            G, base = self.grid_size, 0
        else:
            G, base = self.met_grid_size, self.n_primary_demes
        r, c = self.deme_coords[deme_idx]
        out = []
        for rr, cc in [(r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1)]:
            if 0 <= rr < G and 0 <= cc < G:
                out.append(base + rr * G + cc)
        return out

    # --- rates (mirror Deme.get_cancer_death_rate) ---------------------------
    def _immune_fraction(self, deme, total=None):
        # No immune cells anywhere -> the fraction is identically zero; skip the per-genotype scan
        # (this is O(#genotypes-in-deme) and dominates the clone-heavy tau-leaping loop otherwise).
        if not getattr(self, "_has_immune", True):
            return 0.0
        if total is None:
            total = sum(deme.values())
        if total == 0:
            return 0.0
        immune = sum(cnt for gid, cnt in deme.items() if self.genotypes[gid].type == "immune")
        return immune / total

    def _epithelial_fraction(self, deme, total=None):
        # Fraction of the deme that is (still) epithelial. Live composition, exactly like
        # _immune_fraction: as cancer accumulates and dilutes the resident normals the epithelial
        # barrier a cancer clone feels here falls, so the compartment is never a fixed deme label.
        if not getattr(self, "_has_epithelial", True):
            return 0.0
        if total is None:
            total = sum(deme.values())
        if total == 0:
            return 0.0
        epi = sum(cnt for gid, cnt in deme.items() if self.genotypes[gid].type == "epithelial")
        return epi / total

    def _stromal_fraction(self, deme, total=None):
        # Fraction of the deme that is (still) stromal — the stromal analogue of _epithelial_fraction.
        if not getattr(self, "_has_stromal", True):
            return 0.0
        if total is None:
            total = sum(deme.values())
        if total == 0:
            return 0.0
        stro = sum(cnt for gid, cnt in deme.items() if self.genotypes[gid].type == "stromal")
        return stro / total

    def _host_fraction(self, deme, total=None):
        # Fraction of the deme that is (still) metastatic-host parenchyma — the met analogue of
        # _stromal_fraction. Guarded by _has_host so the O(#genotypes) scan is skipped entirely when no
        # met deposit exists (the same fast-path optimisation as _immune_fraction / _stromal_fraction).
        if not getattr(self, "_has_host", False):
            return 0.0
        if total is None:
            total = sum(deme.values())
        if total == 0:
            return 0.0
        host = sum(cnt for gid, cnt in deme.items() if self.genotypes[gid].type == "host")
        return host / total

    def _deme_comp(self, deme):
        """One O(#genotypes-in-deme) pass over a deme returning the per-compartment cell counts a
        death-rate evaluation needs:
        ``(total, n_normal, n_immune, n_epithelial, n_stromal, n_host, b_bar, d_bar)``.

        ``b_bar`` / ``d_bar`` are the deme's cell-weighted mean CANCER division and baseline death
        rates — the reference the ``"lottery"`` crowding law's uniform pressure is scaled by
        (DESIGN_crowding_v2.md §3.1). They are accumulated in this same pass (two extra floats, no
        extra scan) and left at 0.0 under every other crowding mode, which never reads them.

        Passing the result into ``_death_rate`` (via its ``comp`` argument) lets every genotype in a
        deme reuse ONE composition scan instead of each re-running the O(#genotypes) immune /
        epithelial / stromal / host ``_*_fraction`` scans. That turns the per-deme death-rate work
        from O(#genotypes²) to O(#genotypes) — the dominant per-substep cost once a high mutation
        rate makes #genotypes track #cells (see the grid-90 profile in DESIGN_scalability.md §7).
        ``n_normal`` matches ``sum(deme.get(nm, 0) for nm in normal_names)`` because a normal
        genotype's id IS its type name."""
        total = n_normal = n_immune = n_epi = n_stro = n_host = 0
        b_sum = d_sum = 0.0
        lottery = self._lottery
        G = self.genotypes
        for gid, c in deme.items():
            total += c
            rep = G[gid]
            tp = rep.type
            if tp == "cancer":
                if lottery:
                    ep = rep.evolutionary_parameters
                    b_sum += c * ep["division_rate"]
                    d_sum += c * ep["death_rate"]
                continue
            n_normal += c
            if tp == "immune":
                n_immune += c
            elif tp == "epithelial":
                n_epi += c
            elif tp == "stromal":
                n_stro += c
            elif tp == "host":
                n_host += c
        if lottery:
            n_cancer = total - n_normal
            if n_cancer > 0:
                b_sum /= n_cancer
                d_sum /= n_cancer
        return (total, n_normal, n_immune, n_epi, n_stro, n_host, b_sum, d_sum)

    def _compartment_fields(self):
        """Per-deme epithelial / stromal LIVE fractions, as two (n_demes,) arrays — the compartment
        as a niche field for R13 route-3 (DESIGN_phenotype_plasticity.md §2, Part D). The SAME clone
        then expresses the niche-driven (e.g. invasive) program more at the epithelial interface than
        in the stroma: an env-responsive phenotype and the genetic-vs-niche attribution confound, with
        iscc knowing both contributions. Zero everywhere when no gland structure was seeded."""
        n_demes = len(self.demes)
        epi = np.zeros(n_demes)
        stro = np.zeros(n_demes)
        for i, deme in enumerate(self.demes):
            total = sum(deme.values())
            if total == 0:
                continue
            epi[i] = self._epithelial_fraction(deme, total)
            stro[i] = self._stromal_fraction(deme, total)
        return epi, stro

    def _tx_applies(self, deme_idx):
        """Whether the active treatment's per-genotype override acts in this deme's compartment
        (Treatment.sites: 'both' / 'met' / 'primary'). 'both' (the default) is byte-identical to the
        pre-metastasis, ungated behaviour."""
        if self._tx_sites == "both":
            return True
        is_met = deme_idx >= self.n_primary_demes
        return is_met if self._tx_sites == "met" else (not is_met)

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

    def _state_protection(self, rep):
        """Drug protection conferred by the carried resistance STATE (DESIGN_phenotype_plasticity.md
        §3.3): the cell's state level x the configured ``resistance_state_effect``. Folded into the
        ``max(treatment_resistance, drug_tolerance, ...)`` protection everywhere the drug's
        ``(1 - protection)`` factor is computed. Returns 0.0 when the feature is off (effect 0.0) or the
        cell is not in the state, so every ``max(tr, dt, self._state_protection(rep))`` is byte-identical
        to ``max(tr, dt)`` when off."""
        return getattr(rep, "resistance_state", 0.0) * self.selection.resistance_state_effect

    def _kill_amount(self, rep, intensity, treatment=None):
        """Extra death hazard a dose of ``intensity`` imposes on this clone.

        ``kill_mode="additive"`` (default) adds the SAME absolute hazard to every clone, so whether a
        cell survives is decided by how fast it divides: at kill_rate 1.5 an ordinary met clone
        (b=1.02) sits at net -0.72 and dies, while a driver-saturated one (b=1.70, at the
        max_birth_rate cap) sits at net -0.04 and merely hovers. The drug then SELECTS for the fast
        clones and the deposit regrows mid-course; the only escape is a dose large enough to
        annihilate everything, which leaves too few divisions for resistance to arise. One absolute
        number cannot serve clones whose birth rates differ threefold.

        ``kill_mode="proliferation"`` scales the hazard by the clone's own division rate, so
        ``net = b(1 - kill_rate) - death``: at kill_rate 1.0 every clone declines at exactly its death
        rate regardless of fitness, and above 1.0 the FASTER clones die faster. Dividing quickly stops
        being an escape and starts being a liability -- which is also the biology, since cytotoxic
        chemotherapy damages cells as they replicate (hence its sparing of quiescent tissue).

        NOTE the CELL engine is a THIRD model, not either of these: ``Chemotherapy._apply``
        (iscc/treatment/chemotherapy.py) does ``death *= rate_multiplier ** (1 - treatment_resistance)``
        -- multiplicative on the death rate, with no division-rate term at all. So a clone's hazard
        there scales with how fast it was ALREADY dying, not with how fast it divides, and neither
        count-engine mode reproduces it. The two engines therefore do NOT agree under treatment;
        "proliferation" is a new model rather than a restoration of parity. (An earlier version of
        this docstring claimed the cell engine already did the proliferation scaling. It does not.)"""
        tx = treatment if treatment is not None else self._tx_treatment
        kill_rate = getattr(tx, "kill_rate", 0.8)
        if getattr(tx, "kill_mode", "additive") == "proliferation":
            return intensity * kill_rate * rep.evolutionary_parameters["division_rate"]
        return intensity * max(kill_rate - rep.evolutionary_parameters["death_rate"], 0.0)

    def _tx_death_add_for(self, gid, rep=None):
        """The active treatment's extra death hazard for ``gid`` -- the value ``_apply_treatment``
        stored, or, for a genotype minted AFTER that step's rebuild, the same value computed on demand
        and memoised so it is charged from its very first step instead of leaking a drug-free step.

        The formula mirrors ``_apply_treatment``'s death-rate branch EXACTLY (same target test, same
        ``max(treatment_resistance, drug_tolerance)`` protection, same ``kill_rate - death_rate``
        intensity), so a memoised entry and an on-demand one are identical -- a newborn clone and its
        already-registered siblings take the same hazard. Returns 0.0 (and memoises nothing) when no
        death-rate dose is active, so untreated / immunotherapy-only runs are byte-identical."""
        val = self._tx_death_add.get(gid)
        if val is not None:
            return val
        tx = self._tx_treatment
        if tx is None or getattr(tx, "affects", "death_rate") == "immune_resistance":
            return 0.0
        rep = rep if rep is not None else self.genotypes[gid]
        if self._is_cancer(gid):
            target = self._is_treatment_target(tx, rep)
            tr = max(rep.evolutionary_parameters["treatment_resistance"],
                     rep.evolutionary_parameters.get("drug_tolerance", 0.0),
                     self._state_protection(rep))            # §3.3 carried resistance state
            p = tx.effectiveness if target else tx.toxicity
        else:
            tr, p = 0.0, tx.toxicity
        intensity = self._tx_dosage * p * (1.0 - min(max(tr, 0.0), 1.0))
        if intensity <= 0:
            return 0.0                          # matches _apply_treatment's `continue` (no entry)
        val = self._kill_amount(rep, intensity, tx)
        self._tx_death_add[gid] = val           # memoise: charged identically on every later step
        return val

    def _tx_immune_resist_for(self, gid, rep, base_ir):
        """The immunotherapy-stripped immune resistance for ``gid``: the stored override, or -- for a
        genotype minted after this step's rebuild -- the same ``ir * (1 - intensity)`` computed on
        demand and memoised. Mirrors ``_apply_treatment``'s immune_resistance branch. Returns the
        clone's untouched ``base_ir`` (and memoises nothing) when no immunotherapy dose reaches it, so
        untreated / death-rate-therapy runs are byte-identical to the old ``.get(gid, base_ir)``."""
        val = self._tx_immune_resist.get(gid)
        if val is not None:
            return val
        tx = self._tx_treatment
        if (tx is None or getattr(tx, "affects", "death_rate") != "immune_resistance"
                or not self._is_cancer(gid)):
            return base_ir
        target = self._is_treatment_target(tx, rep)
        tr = rep.evolutionary_parameters["treatment_resistance"]
        p = tx.effectiveness if target else tx.toxicity
        intensity = self._tx_dosage * p * (1.0 - min(max(tr, 0.0), 1.0))
        if intensity <= 0:
            return base_ir                      # matches _apply_treatment's `continue` (no entry)
        val = rep.evolutionary_parameters["immune_resistance"] * (1.0 - intensity)
        self._tx_immune_resist[gid] = val       # memoise: read identically on every later step
        return val

    def _death_rate(self, gid, deme_idx, total=None, comp=None):
        """Cancer death rate = crowding-modulated baseline + local immune killing + treatment.

        Mirrors the corrected Deme.get_cancer_death_rate: immune killing is additive contact
        pressure (more local immune cells -> higher death, attenuated by immune resistance),
        and active therapy adds a death hazard (chemo/targeted) or strips immune resistance
        (immunotherapy, via the per-step _tx_* overrides). ``total`` (the deme's cell count) may be
        passed in when the caller already computed it, so a per-deme loop doesn't recompute it once
        per genotype.

        ``comp`` is the deme's precomputed composition tuple from ``_deme_comp``
        ``(total, n_normal, n_immune, n_epithelial, n_stromal, n_host)``. When given, the immune /
        compartment fractions and ``n_normal`` are read from it in O(1) instead of re-scanning the
        deme per genotype; the numbers are identical, so this is byte-identical to ``comp=None`` (the
        old per-call scans) but drops the per-deme cost from O(#genotypes²) to O(#genotypes)."""
        rep = self.genotypes[gid]
        deme = self.demes[deme_idx]
        if comp is None and self._lottery:
            comp = self._deme_comp(deme)        # the lottery law needs the deme's mean cancer rates
        if comp is not None:
            total, n_normal_c, n_immune_c, n_epi_c, n_stro_c, n_host_c, b_bar_c, d_bar_c = comp
        elif total is None:
            total = sum(deme.values())
        base = rep.evolutionary_parameters["death_rate"]
        if self._crowding:
            # Density-dependent death split into two crowding sources (DESIGN_crowding.md).
            #
            # (1) Co-resident CANCER cells crowd RELATIVE to this clone's OWN evolved division rate
            #     (Option A): the slope is the clone's net growth (div - base) steepened by
            #     (1 + crowding_margin), so a cancer-only deme caps at K/(1+margin) INDEPENDENT of
            #     fitness — no overfill even for a clone evolved up to max_birth_rate.
            # (2) The deme's immortal NORMAL cells (epithelial/stromal) crowd at a FIXED reference
            #     rate `_resident_ref`, NOT the clone's own division. Because it is not scaled by
            #     (div - base), this term does NOT cancel in the survival condition net = div - death:
            #     net > 0 becomes div > base + (ref - base)(1+margin)(n_normal/K), a genuine FITNESS
            #     THRESHOLD rising with the resident count. Only fitter cancer survives (and then
            #     disperses onward from) a normal-occupied gland deme; the normal cells never die.
            #
            # A cancer-only deme has n_normal = 0, so term (2) vanishes and the result is byte-
            # identical to the previous `slope * (total / K)` form.
            div = self._div_rate(gid, deme_idx, rep)
            steep = 1.0 + self.crowding_margin
            # Per-deme carrying capacity (DESIGN_ductal_field.md §3): duct demes cap at K_duct, stroma
            # at K_stroma. Uniform (= carrying_capacity) reproduces the scalar law byte-identically.
            K = self.carrying_capacity if self._deme_capacity is None else self._deme_capacity[deme_idx]
            if self._crowding_mode == "lottery":
                # Structural-cap law (DESIGN_crowding_v2.md §3.1). The crowding pressure is UNIFORM
                # over the deme's cancer clones — it carries no `div`, so it cannot cancel out of
                # net = div - death, and the selection differential between two clones in the deme is
                # their bare (b_i - d_i) - (b_k - d_k), the SAME at the front as in the packed
                # interior. It is deliberately set BELOW the deme's mean net growth (rho < 1), so the
                # rate law alone would overfill and the STRUCTURAL slot cap is what sets density.
                if self._crowding_reference == "fixed":
                    slope = max(0.0, self._crowding_ref - base)
                else:
                    slope = max(0.0, b_bar_c - d_bar_c)
                # Density term. Below K it is linear in occupancy; ABOVE K it accelerates, so an
                # over-capacity deme is pushed back rather than merely held. Crucially this stays
                # UNIFORM over the deme's cancer clones (it carries no `div`), so it cannot cancel
                # out of net = div - death and clone-vs-clone competition survives — the shape of
                # the density response and the preservation of selection are independent properties.
                if self._crowding_law == "noble":
                    # Two-valued step (see __init__): no density term at or below K; above it a
                    # fixed rate chosen to exceed any attainable division rate. `death` then falls
                    # through to the resident / immune / treatment terms and the clamp below,
                    # exactly as the logistic branch does.
                    if total <= K:
                        death = base if self._crowding_d0 is None else float(self._crowding_d0)
                    else:
                        d1 = (self.maximum_death_rate if self._crowding_d1 is None
                              else float(self._crowding_d1))
                        death = max(base, d1)
                else:
                    # float(): K may be a numpy scalar (per-deme capacity array), and the old
                    # `min(1.0, total/K)` returned a PYTHON float at the cap. Keeping the type
                    # identical matters — a numpy float here turns downstream `is True` comparisons
                    # into numpy.bool_ identity checks that silently fail.
                    x = float(total) / float(K)
                    death = base + self._crowding_turnover * slope * min(1.0, x)
                # Resident pressure is UNCHANGED: it is the fitness gate on entering normal-occupied
                # tissue, and it does not cancel either (it carries no `div`).
                n_normal = (n_normal_c if comp is not None
                            else sum(deme.get(nm, 0) for nm in normal_names))
                if n_normal:
                    death += max(0.0, self._resident_ref - base) * steep * (n_normal / K)
            elif self._crowding_mode == "fixed":
                # Unified fixed-reference law: death ~ TOTAL occupancy relative to a fixed reference
                # (not the clone's own division), so survival under crowding is fitness-DEPENDENT
                # everywhere. Near-neutral at low density; at the crowded border only clones with
                # div near the reference persist. No overfill (nothing exceeds the reference).
                death = base + max(0.0, self._crowding_ref - base) * steep * (total / K)
            else:
                n_normal = (n_normal_c if comp is not None
                            else sum(deme.get(nm, 0) for nm in normal_names))
                n_cancer = total - n_normal
                death = base + max(0.0, div - base) * steep * (n_cancer / K)
                if n_normal:
                    death += max(0.0, self._resident_ref - base) * steep * (n_normal / K)
        else:
            # well-mixed regime (carrying_capacity None/0): no crowding ceiling -> unbounded growth.
            death = base
        death = min(death, self.maximum_death_rate)

        # Immune resistance may be transiently stripped by immunotherapy; that override is gated to the
        # treated compartment(s) (R9), so an untreated site keeps the clone's baseline resistance.
        base_ir = rep.evolutionary_parameters["immune_resistance"]
        ir = self._tx_immune_resist_for(gid, rep, base_ir) if self._tx_applies(deme_idx) else base_ir
        ir = min(max(ir, 0.0), 1.0)
        imm_frac = ((n_immune_c / total if total else 0.0) if comp is not None
                    else self._immune_fraction(deme, total))
        death += self._immune_prob_kill * imm_frac * (1.0 - ir)

        # Compartment-dependent selection (v1): each resident compartment adds a local hazard,
        # attenuated by the clone's matching heritable trait — the exact shape of the immune term.
        # The trait is clamped to [0,1] like `ir`. Both coefficients default to 0.0 (off), so these
        # additions are `+= 0.0` and the death rate is byte-identical to before.
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
        # Metastatic host-tissue hazard (R9): the met analogue of the stromal hazard — the deposit's
        # immortal host parenchyma adds a local death hazard to invading cancer, attenuated by the
        # clone's heritable met_survival trait (clamped [0,1] like the others). Default 0.0 -> += 0.0
        # -> byte-identical whether met is off or on-with-hazard-0.
        if self._met_hazard:
            ms = min(max(rep.evolutionary_parameters["met_survival"], 0.0), 1.0)
            host_frac = ((n_host_c / total if total else 0.0) if comp is not None
                         else self._host_fraction(deme, total))
            death += self._met_hazard * host_frac * (1.0 - ms)

        # Chemo/targeted death hazard, gated to the treated compartment(s) (R9): systemic ('both')
        # hits everywhere, 'met'/'primary' only their site. 'both' is byte-identical to before.
        # A genotype minted mid-step is absent from _tx_death_add (frozen by this step's
        # _apply_treatment); _tx_death_add_for computes and memoises its hazard on demand rather than
        # letting a bare `.get(..., 0.0)` grant it a drug-free step.
        if self._tx_applies(deme_idx):
            death += self._tx_death_add_for(gid, rep)
        return death

    def _cancer_gids(self, deme):
        return [gid for gid in deme if self._is_cancer(gid)]

    # --- origin confinement --------------------------------------------------
    def _origin_tick(self, generations):
        """Advance the founding acinus's confinement clock by ``generations`` and release it once the
        lesion has filled the acinus and then distended it for long enough.

        Called once per generation by the tau engine and once per event by the exact engine (which has
        no generation notion, so a generation there is one event per cancer cell). A no-op — and not
        even reached, since every caller guards on ``_origin_confined`` — when the feature is off.
        """
        if not self._origin_confined:
            return
        if self._origin_age is None:
            # Without crowding a deme has no capacity to fill, so there is nothing to distend: the
            # clock simply starts at once and `generations` alone sets the confinement window.
            filled = not self._crowding or all(
                sum(self.demes[di].values()) >= self._origin_established_at * self._cap_of(di)
                for di in self._origin_demes)
            if not filled:
                return
            self._origin_age = 0.0
        self._origin_age += generations
        if self._origin_age >= self._origin_generations:
            self._origin_confined = False
            self.events.append(dict(step=self.step, time=self.time, event="origin_release",
                                    from_deme=int(self._origin_demes[0]) if self._origin_demes else -1,
                                    genotype=self.founder_id, n=len(self._origin_demes)))

    def _escape_factor(self, deme_idx):
        """Fraction of would-be dispersing daughters that actually LEAVE ``deme_idx``.

        1.0 everywhere — and everywhere at all times once the feature is off or released — except
        inside the founding acinus while that acinus is still architecturally intact, where the
        intact myoepithelial layer holds the daughter in (see the ``origin_confinement`` block in
        ``__init__``). A daughter that cannot leave is not lost: it competes for a slot in its own
        deme exactly like a daughter whose basement-membrane crossing failed.

        This is deliberately a property of the *destination step*, not of the cell's rates. The
        per-division MUTATION probability must not depend on whether a cell is allowed to move
        (see the fate split in ``update`` / ``_tau_substep``).
        """
        if self._origin_confined and deme_idx in self._origin_demes:
            return self._origin_disp_factor
        return 1.0

    def _dispersal_of(self, rep, deme_idx):
        """This clone's EFFECTIVE dispersal rate as seen from ``deme_idx``: its own rate everywhere,
        scaled by ``_escape_factor`` inside a still-intact founding acinus. Reported/diagnostic —
        the engines split the division fate on the clone's own (unconfined) dispersal rate and apply
        the escape factor to the dispersal branch, so that confinement changes where daughters go
        without changing how often they mutate."""
        return rep.evolutionary_parameters["dispersal_rate"] * self._escape_factor(deme_idx)

    def _deme_rate(self, deme_idx):
        """Total event rate of a deme. Cancer genotypes are always dynamic; NORMAL genotypes become
        (death-only) dynamic in a treated compartment under the off-target chemo toxicity hazard.
        Untreated (``_tx_death_add`` empty) -> the normal term vanishes -> byte-identical."""
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

        A deme with no cancer cell and no treatment-sensitive resident draws NOTHING in
        ``_tau_substep`` — the genotype loop is empty — yet the old ``range(len(self.demes))``
        still paid a ``_deme_comp`` scan for each of them. In the ductal field EVERY deme is
        non-empty (it is pre-filled with normal tissue), so that was ~grid_size**2 wasted scans
        per substep whatever the lesion's size, and a lesion confined to ONE acinus paid all
        28,900 of them at grid 170. Iterating the cancer-occupied set instead makes the per-substep
        cost O(occupied demes). Purely an omission of no-op work: the skipped demes consume no rng
        draw and change no state, so the trajectory is unchanged.

        Under an active therapy normal cells DO die (``_tx_death_add``), so every deme is live and
        the full range is returned — byte-identical to the old behaviour on the treated path.
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

    def _compartment_counts(self):
        """Split the live genotype counts by compartment for the 2-band Muller: returns
        (primary_counts, met_counts) dicts over the primary vs met deme blocks (which share the one
        genotypes_parents genealogy, so a clone keeps its identity/colour across both)."""
        prim, met = Counter(), Counter()
        for i, deme in enumerate(self.demes):
            tgt = met if i >= self.n_primary_demes else prim
            for gid, c in deme.items():
                tgt[gid] += c
        return dict(prim), dict(met)

    def _trace_snapshot(self):
        """One trace snapshot: the global genotype counts (back-compat) plus, when the met is enabled,
        the per-compartment counts. Met-off snapshots keep the single `genotypes_counts` key, so
        existing plot_muller / write / traces consumers are byte-identical."""
        snap = dict(genotypes_counts=dict(self.genotypes_counts))
        if self._met_enabled:
            snap["primary_counts"], snap["met_counts"] = self._compartment_counts()
        return snap

    def update(self, rng):
        if self.is_extinct():
            return
        if self._origin_confined:
            # The exact engine fires ONE cell event per call and tracks no time, so a generation here
            # is defined as one event per cancer cell.
            self._origin_tick(1.0 / max(1, self.get_cancer_size()))
        di = int(rng.choice(len(self.demes), p=self.deme_rates / self.deme_rates.sum()))
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
        # tox_gids is empty unless a therapy is active, so this is byte-identical to before untreated.
        tox_gids = []
        if self._tx_death_add and self._tx_applies(di):
            tox_gids = sorted((g for g in deme if g in self._tx_death_add and not self._is_cancer(g)),
                              key=lambda g: self.genotypes[g].ord)
            weights += [deme[g] * self._tx_death_add[g] for g in tox_gids]
        weights = np.array(weights)
        pick = int(rng.choice(len(gids) + len(tox_gids), p=weights / weights.sum()))
        if pick >= len(gids):
            self._remove(di, tox_gids[pick - len(gids)], 1)   # a normal cell dies from toxicity
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
            # The division fate is split on the clone's OWN rates. Using the confinement-scaled
            # dispersal rate here would make a cell inside a closed acinus mutate on every division
            # (mut_prob -> 1) purely because the denominator lost its dispersal term — an artefact of
            # how the fate is drawn, not biology. Confinement acts on the dispersal branch instead
            # (``_escape_factor``), so mut_prob is identical confined or not.
            disp = rep.evolutionary_parameters["dispersal_rate"]
            denom = rep.mutation_rate + disp
            mut_prob = rep.mutation_rate / denom if denom > 0 else 0.0
            if rng.random() < mut_prob:
                # Under the structural cap the slot is contested BEFORE the mutation is drawn: a
                # mutation-branch birth that cannot get a slot must never reach mutate(), or the
                # registry would accumulate genotypes with no cells (DESIGN_crowding_v2.md §3.3).
                if self._lottery and not self._can_place(di, total):
                    pass                                       # birth fails; the division is consumed
                else:
                    child = rep.divide()
                    res = child.mutate(rng, self.selection,
                                       func_mask=self._func_bits if self.coarsen_passengers else None)
                    if res == "passenger":
                        # pure neutral daughter: identical dynamics, fold into the parent clone and tally
                        # its passenger burden for lazy reconstruction. No genotype registered.
                        self._place_one(di, gid, rng)
                        self._pass_load[gid] += child._n_new_snv
                        self._n_folded += 1
                    elif res:
                        # functional (driver / CNV / WGD) daughter -> a new genotype, unless it breaches
                        # the viability limits, in which case it is never born (see _is_viable): the
                        # division is consumed but adds no cell.
                        if self._is_viable(child):
                            self._register(child)
                            self.genotypes_parents[child.genotype_id] = rep.genotype_id
                            self._place_one(di, child.genotype_id, rng)
                    else:
                        # no-op mutation (saturated allele): same genotype, grows in place
                        self._place_one(di, rep.genotype_id, rng)
            elif (esc := self._escape_factor(di)) < 1.0 and (
                    esc <= 0.0 or rng.random() >= esc):
                # The daughter set out but the intact acinus holds it in: it stays put and competes
                # for a slot in its own deme, exactly like a failed basement-membrane crossing.
                # Unreachable (no rng draw, no branch) whenever the escape factor is 1.0, i.e.
                # always when origin confinement is absent or released -> byte-identical.
                self._place_one(di, gid, rng)
            else:
                # dispersal. Three routes, mutually exclusive by SOURCE compartment:
                #  - met seeding (primary STROMAL deme = an invaded IDC cell): a fraction ~met_seed_kappa
                #    of daughters attempt the hop to the metastatic deposit;
                #  - cross-gland island hop (gland-resident cell): a fraction ~cross_gland_kappa;
                #  - local: a uniformly random von-Neumann neighbour in the same compartment.
                # Each guarded branch short-circuits before its rng draw when off -> byte-identical.
                src_g = self.gland_id[di] if self.gland_id is not None else -1
                took_met = False
                # Met seeding is eligible ONLY from a primary stromal deme (src_g == -1 and di in the
                # primary block), so DCIS confined to glands cannot seed but IDC in the stroma can — the
                # clinical fact emerges from geometry, not a flag.
                if (self._met_enabled and self._met_seed_kappa > 0 and di < self.n_primary_demes
                        and src_g == -1
                        and rng.random() < self._met_seed_kappa / (1.0 + self._met_seed_kappa)):
                    took_met = True
                    ms = min(max(rep.evolutionary_parameters["met_survival"], 0.0), 1.0)
                    if rng.random() < self._transit_prob(ms) and self._place_one(
                            self.met_vessel_idx, gid, rng):
                        self.events.append(dict(step=self.step, time=self.time, event="seeding",
                                                from_deme=int(di), genotype=gid, n=1))
                        affected.append(self.met_vessel_idx)
                    # else: the daughter dies in transit — parent stays, the division is still consumed.
                if not took_met:
                    tgt = None
                    if (self.cross_gland_kappa > 0 and src_g != -1
                            and rng.random() < self.cross_gland_kappa / (1.0 + self.cross_gland_kappa)):
                        tgt = self._cross_gland_target(src_g, rng)
                    if tgt is None:
                        nbrs = self._neighbors(di)
                        tgt = nbrs[int(rng.choice(len(nbrs)))] if nbrs else di
                    # Basement-membrane gate: a duct(gland)->stroma hop is an INVASION event and needs
                    # `breach`. It succeeds with probability = the clone's breach trait; a cell that fails
                    # stays confined in its duct (DCIS). Off unless breach_gated_invasion is set, and only
                    # for the duct->stroma direction (intraductal/cross-gland and stroma->stroma hops are
                    # untouched), so it is byte-identical when the flag is off.
                    if (self._breach_gated_invasion and self.gland_id is not None
                            and self.gland_id[di] >= 0 and self.gland_id[tgt] < 0):
                        b = min(max(rep.evolutionary_parameters["breach"], 0.0), 1.0)
                        if rng.random() >= b:
                            tgt = di                      # crossing fails -> daughter stays in the duct
                    # The daughter still has to find a slot where it lands (including back in its own
                    # duct after a failed crossing); if the target is full and holds no displaceable
                    # resident, the birth fails. No redraw.
                    if self._place_one(tgt, gid, rng):
                        affected.append(tgt)

        for idx in affected:
            self._refresh_rate(idx)

    # --- treatment (genotype-level) -----------------------------------------
    def _is_treatment_target(self, treatment, rep):
        """Is this cancer genotype sensitive to the therapy?

        Broad therapies (chemo, or gene-targeted therapy with no checkpoints) hit every
        cancer cell; gene-specific therapies delegate to the treatment's expression-based
        `is_target`. The representative's baseline expression is set so `expresses` is
        deterministic.
        """
        if rep.type != "cancer":
            return False
        targets = getattr(treatment, "targets", None)
        if not targets:
            return True
        rep.baseline_exp = self.celltype_exps["cancer"]
        return bool(treatment.is_target(rep, mut_effects=self.selection.mut_effects))

    def _apply_treatment(self, treatment, dosage):
        """Recompute per-genotype treatment effects for the current dosage.

        Treatment is a transient, dose-dependent modifier, not a genotype property, so the
        effect lives in self._tx_* (refreshed every treated step) rather than mutating the
        shared genotype. The per-cell stochastic effectiveness/toxicity of the cell engine
        becomes an expected, dose-scaled intensity over the exchangeable count.
        """
        self._tx_death_add = {}
        self._tx_immune_resist = {}
        self._tx_div_mult = {}
        self._tx_sites = getattr(treatment, "sites", "both") if treatment is not None else "both"
        if treatment is None or dosage <= 0:
            self._tx_treatment, self._tx_dosage = None, 0.0
            return
        # Keep the active dose's context so a mid-step-minted genotype's hazard can be recomputed
        # on demand (see _tx_death_add_for / _tx_immune_resist_for) with the exact inputs used below.
        self._tx_treatment, self._tx_dosage = treatment, dosage
        affects = getattr(treatment, "affects", "death_rate")
        kill_rate = getattr(treatment, "kill_rate", 0.8)
        # Persister cost, charged only for the duration of the dose (see Selection.proliferation_cost
        # for why it cannot be baked into division_rate). 0.0 by default -> dict stays empty.
        dt_cost = float(getattr(self.selection, "drug_tolerance_cost", 0.0) or 0.0)
        # Therapy-induced mutator phenotype: raise an EXPOSED clone's mutation rate ONCE and leave it
        # raised (see Treatment.mutagenicity). Applied here because this runs every treated step with
        # the live genotype set; `_tx_mutagenized` makes it idempotent so a 100-step course does not
        # compound the factor 100 times. Descendants inherit it through `rep.divide()`, so the
        # phenotype outlives the drug — which is the point: the de novo resistant clone must arise
        # AFTER exposure, and the elevated rate is what makes that likely.
        mutagenicity = float(getattr(treatment, "mutagenicity", 1.0) or 1.0)
        if mutagenicity != 1.0:
            if self._tx_sites == "both":
                exposed = [g for g in self.genotypes_counts if self._is_cancer(g)]
            else:                       # only clones with cells in a treated compartment are exposed
                exposed = set()
                for di, deme in enumerate(self.demes):
                    if deme and self._tx_applies(di):
                        exposed.update(deme)
                exposed = [g for g in exposed if self._is_cancer(g)]
            # Treatment.mutagenicity_mode "dose": scale the boost by the dose the clone actually
            # receives, the same (1 - tr) factor `_kill_amount` uses, instead of handing every clone
            # in a treated compartment the full multiplier. Under "uniform" a clone taking ZERO drug
            # still mutates `mutagenicity`x faster forever — which mostly serves to multiply the CNA
            # rate that deletes its own resistance allele. Default "uniform" -> byte-identical.
            dose_scaled = getattr(treatment, "mutagenicity_mode", "uniform") == "dose"
            snv_only = getattr(treatment, "mutagenicity_target", "all") == "snv"
            for gid in exposed:
                rep = self.genotypes[gid]
                # The flag lives on the REP, not in a gid set: `Cell.divide()` shallow-copies, so a
                # daughter inherits BOTH the raised mutation_rate and the flag. Keying off a gid set
                # would re-multiply every newly minted child that is still under drug, compounding
                # the factor once per generation instead of applying it once per lineage.
                if getattr(rep, "_tx_mutagenized", False):
                    continue
                factor = mutagenicity
                if dose_scaled:
                    tr = max(rep.evolutionary_parameters["treatment_resistance"],
                             rep.evolutionary_parameters.get("drug_tolerance", 0.0),
                             self._state_protection(rep))    # §3.3 carried resistance state
                    factor = 1.0 + (mutagenicity - 1.0) * (1.0 - min(max(tr, 0.0), 1.0))
                if factor == 1.0:
                    # Fully protected: no drug reaches the DNA, so no mutator phenotype -- and NO
                    # flag, so a descendant that later loses resistance is mutagenized on its own
                    # terms rather than inheriting an exemption it no longer earns.
                    continue
                if snv_only:
                    # POINT MUTAGEN. `mutation_rate` is not a mutation rate: it is the mutate-vs-
                    # disperse FATE probability, and mutate() picks the event type from
                    # cnv_prob/snv_prob BEFORE drawing SNVs -- so scaling it scales point mutations
                    # and copy-number events in lockstep and cannot shift the balance between them
                    # (measured acquisition:reversion = 0.14 at mutagenicity 1.0 AND at 4.0).
                    # `n_snvs_per_allele` is the actual per-division point-mutation count and feeds
                    # ONLY the SNV branch, so scaling it raises the chance of an SNV landing on a
                    # resistance locus while leaving the CNA rate that DELETES one untouched --
                    # by construction, not by a compensating adjustment. Measured at x20:
                    # acquisition 6.25e-5 -> 7.46e-4, reversion 4.63e-4 -> 4.29e-4 (flat),
                    # ratio 0.14 -> 1.74. It also has no ceiling, unlike mut_prob = m/(m + dispersal)
                    # which saturates at 1 (a nominal 4x realises only 2.29x).
                    rep.n_snvs_per_allele = float(rep.n_snvs_per_allele) * factor
                else:
                    rep.mutation_rate = float(rep.mutation_rate) * factor
                rep._tx_mutagenized = True
        for gid in list(self.genotypes_counts):
            is_cancer = self._is_cancer(gid)
            rep = self.genotypes[gid]
            if affects == "immune_resistance":
                # immunotherapy strips a clone's immune resistance -- only meaningful for cancer.
                if not is_cancer:
                    continue
                target = self._is_treatment_target(treatment, rep)
                tr = rep.evolutionary_parameters["treatment_resistance"]
                p = treatment.effectiveness if target else treatment.toxicity
                intensity = dosage * p * (1.0 - min(max(tr, 0.0), 1.0))
                if intensity <= 0:
                    continue
                ir = rep.evolutionary_parameters["immune_resistance"]
                self._tx_immune_resist[gid] = ir * (1.0 - intensity)
            else:
                # Death-rate therapy (chemo / targeted). Cancer TARGET cells take `effectiveness`;
                # everything OFF-TARGET -- non-target cancer clones AND normal/host tissue -- takes
                # `toxicity` (one knob: toxicity = off-target anything). Normal cells carry no treatment
                # resistance; giving them a `_tx_death_add` is what makes chemo's systemic toxicity kill
                # normal tissue (they become death-only participants in the dynamics -- see _deme_rate /
                # update / _tau_substep). Cancer entries are unchanged, so the cancer path is byte-identical.
                if is_cancer:
                    target = self._is_treatment_target(treatment, rep)
                    tr = max(rep.evolutionary_parameters["treatment_resistance"],
                             rep.evolutionary_parameters.get("drug_tolerance", 0.0),
                             self._state_protection(rep))    # §3.3 carried resistance state
                    p = treatment.effectiveness if target else treatment.toxicity
                else:
                    tr, p = 0.0, treatment.toxicity
                # A persister pays its division cost for as long as the drug is present, INCLUDING
                # when it is tolerant enough to take no kill at all -- so this must precede the
                # `intensity <= 0` short-circuit below, or the fully tolerant cell (the one the
                # mechanism exists for) would be the only one that escapes the cost.
                if is_cancer and dt_cost:
                    dt = rep.evolutionary_parameters.get("drug_tolerance", 0.0)
                    # ...but only while tolerance is what is actually keeping the cell alive. The
                    # protection above is max(resistance, tolerance), so the cost follows the same
                    # max: a cell whose RESISTANCE already covers the drug has no reason to sit in a
                    # stress-induced persister state, and charging it anyway is what makes the de
                    # novo resistant clone the most heavily taxed thing in the deposit -- it would
                    # inherit its persister parent's cost on top of its own resistance cost and
                    # crawl at 0.41 of normal division, so the relapse ends up made of tolerant
                    # cells instead of resistant ones (measured: resistance reached only 0.69% of
                    # the deposit at chemo end).
                    if dt > rep.evolutionary_parameters["treatment_resistance"]:
                        self._tx_div_mult[gid] = max(0.0, 1.0 - dt_cost * dt)
                intensity = dosage * p * (1.0 - min(max(tr, 0.0), 1.0))  # in [0, 1]
                if intensity <= 0:
                    continue
                base = rep.evolutionary_parameters["death_rate"]
                self._tx_death_add[gid] = self._kill_amount(rep, intensity, treatment)

    def _resect(self, site="primary"):
        """Surgical resection (R9): remove EVERY cell (cancer + immortal residents) from the target
        compartment's demes and refresh their rates to 0, so the shared-deme_rates Gillespie fires only
        the remaining compartment afterward (and primary->met seeding stops with no primary cancer
        left). Deterministic — no rng draw. The genealogy is NOT pruned, so a 2-band Muller still shows
        the resected band cliff to 0. ``site`` is "primary" or "met"."""
        if site == "primary":
            lo, hi = 0, self.n_primary_demes
        elif site == "met":
            lo, hi = self.n_primary_demes, len(self.demes)
        else:
            raise ValueError(f"unknown resection site {site!r}")
        removed = 0
        for di in range(lo, hi):
            deme = self.demes[di]
            for gid in list(deme):
                removed += deme[gid]
                self._remove(di, gid, deme[gid])
            self._refresh_rate(di)
        self.events.append(dict(step=self.step, time=self.time, event="resection",
                                site=site, n=int(removed)))
        return removed

    def _clear_treatment_overrides(self):
        """Drop any residual per-step treatment overrides left by a PRIOR treated grow, so a subsequent
        grow with no active treatment (or an inactive window) sees no stale hazard on cancer OR normal
        tissue. Refreshes the rate vector only when it actually clears something, so it is a byte-
        identical no-op whenever there was nothing left over (e.g. a fresh or never-treated tumour)."""
        if self._tx_death_add or self._tx_immune_resist or self._tx_div_mult:
            self._tx_death_add, self._tx_immune_resist, self._tx_sites = {}, {}, "both"
            self._tx_div_mult = {}
            self._tx_treatment, self._tx_dosage = None, 0.0
            self.deme_rates = np.array([self._deme_rate(i) for i in range(len(self.demes))], dtype=float)

    def grow(self, n_steps=1000, seed=None, treatment=None, **kwargs):
        """Advance the tumour by ``n_steps`` update steps and materialise the result.

        Runs the birth / death / mutation / dispersal process (exact Gillespie or
        tau-leaping, per ``update_mode``), appending a population snapshot to
        ``self.traces`` each step, and finally calls ``make_cell_data`` to build the
        per-cell ground truth.

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

        Returns
        -------
        list
            ``self.traces`` — the list of per-step population snapshots.
        """
        if self.update_mode == "tau":
            return self._grow_tau(n_steps=n_steps, seed=seed, treatment=treatment, **kwargs)
        if seed is None:
            seed = self.seed
        self._clear_treatment_overrides()
        self.traces.append(self._trace_snapshot())
        prev_dose = 0.0
        for local_step in range(n_steps - 1):
            rng = np.random.default_rng(seed + self.step + local_step)
            if treatment is not None:
                treatment.discrete_event(self, self.step + local_step)  # e.g. surgical resection (R9)
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
        self.step += n_steps
        self.make_cell_data()
        return self.traces

    # --- tau-leaping (generation-batched clonal update, DESIGN §7) -----------
    def _tau_substep(self, rng, dt):
        """One synchronous tau-leap of length `dt`.

        Per (deme, genotype) with count c, draw the two reaction channels independently from
        the CURRENT (pre-step) state -- births ~ Poisson(division_rate*c*dt) and
        deaths ~ Poisson(death_rate*c*dt) -- then split births into a mutation branch
        (in-place division, attempts a mutation exactly as the exact engine) and a dispersal
        branch (daughter placed in a neighbour deme) by Binomial(n_births, mut_prob), and apply
        everything in batch. The per-channel rates match the exact engine's
        (event rate = c*(div+death), death share = death/(div+death)), so the two engines agree
        in distribution as dt -> 0. Genotype ids are iterated in creation-ordinal order and all
        randomness comes from the seeded `rng`, so runs are reproducible. Death rates are read
        from current deme totals, so carrying-capacity crowding self-limits across substeps.
        """
        if self._lottery:
            return self._tau_substep_lottery(rng, dt)
        deaths, dispersals, mutants, stayers = [], [], [], []
        for di in self._active_demes():
            deme = self.demes[di]
            if not deme:
                continue
            comp = self._deme_comp(deme)            # ONE composition scan reused by every genotype
            total = comp[0]
            esc = self._escape_factor(di)
            for gid in sorted(self._cancer_gids(deme), key=lambda g: self.genotypes[g].ord):
                c = deme[gid]
                rep = self.genotypes[gid]
                div = self._div_rate(gid, di, rep)
                death = self._death_rate(gid, di, total, comp=comp)
                # Fate is split on the clone's OWN dispersal rate, never the confinement-scaled one
                # (see _escape_factor): a cell's mutation probability must not depend on whether it
                # is allowed to move.
                disp = rep.evolutionary_parameters["dispersal_rate"]
                n_div = int(rng.poisson(div * c * dt))
                n_death = int(rng.poisson(death * c * dt))
                if n_death:
                    deaths.append((di, gid, n_death))
                if n_div:
                    denom = rep.mutation_rate + disp
                    mut_prob = rep.mutation_rate / denom if denom > 0 else 0.0
                    n_mut = int(rng.binomial(n_div, mut_prob))
                    n_disp = n_div - n_mut
                    if n_mut:
                        mutants.append((di, gid, n_mut))
                    if n_disp and esc < 1.0:
                        # Confined: only an `esc` fraction gets out; the rest stay and take a slot
                        # at home. Skipped entirely (no draw) when esc == 1.0 -> byte-identical.
                        n_out = int(rng.binomial(n_disp, esc)) if esc > 0.0 else 0
                        if n_disp - n_out:
                            stayers.append((di, gid, n_disp - n_out))
                        n_disp = n_out
                    if n_disp:
                        dispersals.append((di, gid, n_disp))
            # NORMAL cells die under off-target chemo toxicity in a treated compartment (they are
            # otherwise static). Empty unless a therapy is active -> no extra draws -> byte-identical.
            if self._tx_death_add and self._tx_applies(di):
                for gid in sorted((g for g in deme if g in self._tx_death_add and not self._is_cancer(g)),
                                  key=lambda g: self.genotypes[g].ord):
                    n_death = int(rng.poisson(self._tx_death_add[gid] * deme[gid] * dt))
                    if n_death:
                        deaths.append((di, gid, n_death))

        # deaths first, capped at the pre-step count so a clone never goes negative
        for di, gid, n in deaths:
            n = min(n, self.demes[di].get(gid, 0))
            if n:
                self._remove(di, gid, n)
        # dispersal-branch daughters the intact acinus held in: plain in-place births of the SAME
        # genotype (no mutation attempt — that is the mutation branch's job). Always empty when
        # origin confinement is absent or released.
        for di, gid, n in stayers:
            self._add(di, gid, n)
        # mutation-branch births: each is one division that attempts a mutation. A successful
        # mutate() spawns a new genotype (count 1) unless it breaches the viability limits, in
        # which case nothing is born; a saturated allele grows the parent in place.
        # This per-mutation genotype creation is intrinsic to the infinite-sites model and costs
        # exactly the same as in the exact engine (the separate #genotypes concern of §3).
        fmask = self._func_bits if self.coarsen_passengers else None
        for di, gid, n in mutants:
            rep = self.genotypes[gid]
            n_fold = 0
            for _ in range(n):
                child = rep.divide()
                res = child.mutate(rng, self.selection, func_mask=fmask)
                if res == "passenger":
                    # pure neutral daughter: fold into the parent clone (batched via n_fold),
                    # tally its passenger burden. No genotype registered.
                    n_fold += 1
                    self._pass_load[gid] += child._n_new_snv
                elif res:
                    # non-viable daughter: division consumed, no cell added (see _is_viable)
                    if self._is_viable(child):
                        self._register(child)
                        self.genotypes_parents[child.genotype_id] = rep.genotype_id
                        self._add(di, child.genotype_id, 1)
                else:
                    self._add(di, gid, 1)
            if n_fold:
                self._add(di, gid, n_fold)
                self._n_folded += n_fold
        # dispersal-branch births: same genotype. A low fraction (~kappa) of gland-resident daughters
        # take an island hop to another gland's lumen; the rest go to a uniformly random neighbour.
        # kappa=0 draws no binomial and leaves the neighbour draw untouched -> byte-identical.
        for di, gid, n in dispersals:
            src_g = self.gland_id[di] if self.gland_id is not None else -1
            # met seeding: a fraction (~met_seed_kappa) of daughters from a primary STROMAL deme attempt
            # the hop to the met, each surviving transit w.p. _transit_prob(met_survival). A disabled met
            # (or a non-stromal / met source) draws no binomial and leaves n untouched -> byte-identical.
            n_met = 0
            if self._met_enabled and self._met_seed_kappa > 0 and di < self.n_primary_demes and src_g == -1:
                n_met = int(rng.binomial(n, self._met_seed_kappa / (1.0 + self._met_seed_kappa)))
            if n_met:
                ms = min(max(self.genotypes[gid].evolutionary_parameters["met_survival"], 0.0), 1.0)
                n_survive = int(rng.binomial(n_met, self._transit_prob(ms)))
                if n_survive:
                    self._add(self.met_vessel_idx, gid, n_survive)
                    self.events.append(dict(step=self.step, time=self.time, event="seeding",
                                            from_deme=int(di), genotype=gid, n=int(n_survive)))
            n_rest = n - n_met
            n_cross = 0
            if self.cross_gland_kappa > 0 and src_g != -1:
                n_cross = int(rng.binomial(n_rest, self.cross_gland_kappa / (1.0 + self.cross_gland_kappa)))
            n_local = n_rest - n_cross
            if n_local > 0:
                nbrs = self._neighbors(di)
                if not nbrs:
                    self._add(di, gid, n_local)
                else:
                    idx = rng.integers(0, len(nbrs), size=n_local)
                    u, cnts = np.unique(idx, return_counts=True)
                    gate = (self._breach_gated_invasion and self.gland_id is not None
                            and self.gland_id[di] >= 0)
                    b = None
                    for k, cnt in zip(u, cnts):
                        tgt = nbrs[int(k)]; cnt = int(cnt)
                        # Basement-membrane gate: a duct->stroma hop is an INVASION event needing breach.
                        # A Binomial(cnt, breach) fraction cross into the stroma; the rest stay confined
                        # in the duct (DCIS). Off / non-duct->stroma hops -> add all cnt as before.
                        if gate and self.gland_id[tgt] < 0:
                            if b is None:
                                b = min(max(self.genotypes[gid].evolutionary_parameters["breach"], 0.0), 1.0)
                            n_ok = int(rng.binomial(cnt, b)) if b > 0 else 0
                            if n_ok:
                                self._add(tgt, gid, n_ok)
                            if cnt - n_ok:
                                self._add(di, gid, cnt - n_ok)  # crossing fails -> stays in the duct
                        else:
                            self._add(tgt, gid, cnt)
            for _ in range(n_cross):
                tgt = self._cross_gland_target(src_g, rng)
                if tgt is None:
                    nbrs = self._neighbors(di)
                    tgt = nbrs[int(rng.integers(0, len(nbrs)))] if nbrs else di
                self._add(tgt, gid, 1)

    # --- tau-leaping under the structural cap (DESIGN_crowding_v2.md §3.3) ---
    def _realise_mutation_births(self, di, gid, n, rng):
        """Realise ``n`` accepted mutation-branch births of ``gid`` in deme ``di``: each is one
        division that attempts a mutation, spawning a new genotype (unless it breaches the viability
        limits, in which case nothing is born), folding into the parent when the daughter is purely
        neutral, or growing the parent in place on a saturated allele. Adds AT MOST ``n`` cells."""
        rep = self.genotypes[gid]
        fmask = self._func_bits if self.coarsen_passengers else None
        n_fold = 0
        for _ in range(n):
            child = rep.divide()
            res = child.mutate(rng, self.selection, func_mask=fmask)
            if res == "passenger":
                n_fold += 1
                self._pass_load[gid] += child._n_new_snv
            elif res:
                if self._is_viable(child):
                    self._register(child)
                    self.genotypes_parents[child.genotype_id] = rep.genotype_id
                    self._add(di, child.genotype_id, 1)
            else:
                self._add(di, gid, 1)
        if n_fold:
            self._add(di, gid, n_fold)
            self._n_folded += n_fold

    def _tau_substep_lottery(self, rng, dt):
        """One synchronous tau-leap of length `dt` under the ``crowding_mode="lottery"`` law.

        The exact engine needs no allocation rule — it fires one event at a time, so a slot is taken
        by whichever birth happens next and the Gillespie sampler already draws that birth with
        probability ``b_i c_i / sum_k b_k c_k``. Tau-leaping cannot block individual events, so it
        needs one, in four phases:

        0/1. draw births and deaths per (deme, clone) from the PRE-step state exactly as the
             unconstrained substep does, and apply the DEATHS first;
        2.   route every candidate birth to its TARGET deme (in place for the mutation branch; the
             drawn neighbour / cross-gland lumen / metastatic vessel for the dispersal branch, and
             back to the source duct for a failed basement-membrane crossing), accumulating a
             per-target candidate vector keyed by (channel, genotype);
        3.   per target, in ascending deme order, allocate the ``S = floor(K) - n`` free slots among
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
        for di in self._active_demes():
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
                # The clone's OWN dispersal rate, never the confinement-scaled one — confinement
                # must not change how often a division mutates (see _escape_factor).
                disp = rep.evolutionary_parameters["dispersal_rate"]
                n_div = int(rng.poisson(div * c * dt))
                n_death = int(rng.poisson(death * c * dt))
                if n_death:
                    deaths.append((di, gid, n_death))
                if n_div:
                    denom = rep.mutation_rate + disp
                    mut_prob = rep.mutation_rate / denom if denom > 0 else 0.0
                    n_mut = int(rng.binomial(n_div, mut_prob))
                    n_disp = n_div - n_mut
                    if n_mut:
                        mutants.append((di, gid, n_mut))
                    if n_disp and esc < 1.0:
                        # Confined: only an `esc` fraction gets out, the rest stay and compete for a
                        # slot at home. No draw at all when esc == 1.0 -> byte-identical.
                        n_out = int(rng.binomial(n_disp, esc)) if esc > 0.0 else 0
                        if n_disp - n_out:
                            stayers.append((di, gid, n_disp - n_out))
                        n_disp = n_out
                    if n_disp:
                        dispersals.append((di, gid, n_disp))
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
        for di, gid, n in dispersals:
            src_g = self.gland_id[di] if self.gland_id is not None else -1
            n_met = 0
            if self._met_enabled and self._met_seed_kappa > 0 and di < self.n_primary_demes and src_g == -1:
                n_met = int(rng.binomial(n, self._met_seed_kappa / (1.0 + self._met_seed_kappa)))
            if n_met:
                ms = min(max(self.genotypes[gid].evolutionary_parameters["met_survival"], 0.0), 1.0)
                n_survive = int(rng.binomial(n_met, self._transit_prob(ms)))
                # Own channel, keyed by source, so the seeding event log can report the number that
                # actually WON a slot in the vessel deme rather than the number that set out.
                _cand(self.met_vessel_idx, ("met", di, gid), n_survive)
            n_rest = n - n_met
            n_cross = 0
            if self.cross_gland_kappa > 0 and src_g != -1:
                n_cross = int(rng.binomial(n_rest, self.cross_gland_kappa / (1.0 + self.cross_gland_kappa)))
            n_local = n_rest - n_cross
            if n_local > 0:
                nbrs = self._neighbors(di)
                if not nbrs:
                    _cand(di, ("plain", gid), n_local)
                else:
                    idx = rng.integers(0, len(nbrs), size=n_local)
                    u, cnts = np.unique(idx, return_counts=True)
                    gate = (self._breach_gated_invasion and self.gland_id is not None
                            and self.gland_id[di] >= 0)
                    b = None
                    for k, cnt in zip(u, cnts):
                        tgt = nbrs[int(k)]; cnt = int(cnt)
                        if gate and self.gland_id[tgt] < 0:
                            if b is None:
                                b = min(max(self.genotypes[gid].evolutionary_parameters["breach"], 0.0), 1.0)
                            n_ok = int(rng.binomial(cnt, b)) if b > 0 else 0
                            _cand(tgt, ("plain", gid), n_ok)
                            # A failed crossing leaves the daughter in its own duct — where it still
                            # has to win a slot like everyone else.
                            _cand(di, ("plain", gid), cnt - n_ok)
                        else:
                            _cand(tgt, ("plain", gid), cnt)
            for _ in range(n_cross):
                tgt = self._cross_gland_target(src_g, rng)
                if tgt is None:
                    nbrs = self._neighbors(di)
                    tgt = nbrs[int(rng.integers(0, len(nbrs)))] if nbrs else di
                _cand(tgt, ("plain", gid), 1)

        # --- Phase 3 + 4: allocate the slots, then realise only the accepted births ---------------
        for tgt in sorted(cand):
            items = cand[tgt]
            keys = list(items)
            a = np.array([items[k] for k in keys], dtype=np.int64)
            n_slots = max(0, self._slot_cap_of(tgt) - sum(self.demes[tgt].values()))
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
                else:
                    self._add(tgt, key[1], n)

    def _state_twin(self, gid, target_state):
        """Get-or-create the genotype with ``gid``'s genome but ``resistance_state == target_state`` —
        the (genotype x epistate) sublineage the state feature needs (DESIGN_phenotype_plasticity.md §3,
        §3.3).

        A state transition cannot be derived from the genome (the whole point), so it MINTS A NEW
        GENOTYPE ID: two cells with identical genomes and different states are different engine entities.
        The twin is minted ONCE per (genome, state) and REUSED across generations via a per-genome
        ``_state_families`` dict shared between every state-variant of that genome, so the state costs a
        small multiplier on the genotype count, not one id per transitioning cell. The twin shares the
        source's genome by copy-on-write (``divide()``), recomputes its rates for the new state
        (``update_evolutionary_parameters`` applies the state cost/effect), and is recorded as the
        source's genealogy child so viz and the trace subtree walks stay complete."""
        fam = self._state_families.get(gid)
        if fam is None:
            fam = {self.genotypes[gid].resistance_state: gid}
            self._state_families[gid] = fam
        dest = fam.get(target_state)
        if dest is not None:
            return dest
        rep = self.genotypes[gid]
        twin = rep.divide()                                  # shares genome (COW); inherits attributes
        twin.resistance_state = target_state
        twin.update_evolutionary_parameters(self.selection)  # rates reflect the new state (cost/effect)
        twin.set_genotype_id()
        self._register(twin)
        self.genotypes_parents[twin.genotype_id] = gid       # genealogy child of the split-from genotype
        fam[target_state] = twin.genotype_id
        self._state_families[twin.genotype_id] = fam         # share the SAME family dict both ways
        return twin.genotype_id

    def _apply_state_transitions(self, rng):
        """Drug-induced ENTRY into, and off-drug EXIT from, the carried resistance state — once per
        generation (DESIGN_phenotype_plasticity.md §3.3). Only called when ``resistance_state_on``.

        ENTRY (plasticity): while a death-rate dose reaches a deme, each sensitive (state 0) cell has a
        per-generation probability ``induction * intensity`` of being reprogrammed into the state, where
        ``intensity = dosage * p * (1 - protection)`` is the dose the cell actually receives — the same
        factor ``_tx_death_add_for`` / ``_kill_amount`` use — so a cell the drug cannot reach is not
        reprogrammed by it. EXIT (relaxation): where no dose reaches, each state cell with NO genetic
        anchor (``n_mut_tr == 0``) relaxes back to sensitive with probability ``relax`` (= 1/tau_relax);
        a cell that still carries the resistance allele keeps its genetic attractor and does not relax
        (which also stops the genetic-entry clause from re-firing on an exit twin). NOISE:
        env-independent switching at ``noise``. Every transition moves cells between the source genotype
        and its ``_state_twin`` (a new/reused genotype id). Moves are collected first and applied after,
        mirroring the substep's death/mutation batching, so the deme/count dicts are not mutated mid-scan.

        Tau-engine only (called from ``_tau_generation``); the genetic-entry arm and the drug-protection
        term live in shared code and so are active under the exact engine too, but the reversible
        induction/relaxation transitions are not."""
        sel = self.selection
        induction, relax, noise = (sel.resistance_state_induction,
                                   sel.resistance_state_relax, sel.resistance_state_noise)
        tx = self._tx_treatment
        dose_active = (tx is not None and self._tx_dosage > 0.0
                       and getattr(tx, "affects", "death_rate") != "immune_resistance")
        moves = []                                           # (deme_idx, src_gid, dst_gid, n)
        for di, deme in enumerate(self.demes):
            if not deme:
                continue
            treated = dose_active and self._tx_applies(di)
            for gid in sorted(self._cancer_gids(deme), key=lambda g: self.genotypes[g].ord):
                c = deme[gid]
                if c <= 0:
                    continue
                rep = self.genotypes[gid]
                s = getattr(rep, "resistance_state", 0.0)
                if s <= 0.0:
                    # ENTRY: a sensitive cell is reprogrammed only where the drug actually reaches it
                    # (plus env-independent noise), scaled by the dose it receives.
                    p_in = noise
                    if treated and induction > 0.0:
                        target = self._is_treatment_target(tx, rep)
                        prot = max(rep.evolutionary_parameters["treatment_resistance"],
                                   rep.evolutionary_parameters.get("drug_tolerance", 0.0),
                                   self._state_protection(rep))
                        p_eff = tx.effectiveness if target else tx.toxicity
                        intensity = self._tx_dosage * p_eff * (1.0 - min(max(prot, 0.0), 1.0))
                        p_in += induction * intensity
                    p_in = min(max(p_in, 0.0), 1.0)
                    if p_in > 0.0:
                        n = int(rng.binomial(c, p_in))
                        if n:
                            moves.append((di, gid, self._state_twin(gid, 1.0), n))
                elif rep.genome_summary.get("n_mut_tr", 0) == 0:
                    # EXIT: only a cell with the resistance allele already gone can relax back (a
                    # still-anchored cell keeps its genetic attractor). Relaxation only where the drug
                    # is not reaching it; noise switches regardless.
                    p_out = noise + (relax if (not treated and relax > 0.0) else 0.0)
                    p_out = min(max(p_out, 0.0), 1.0)
                    if p_out > 0.0:
                        n = int(rng.binomial(c, p_out))
                        if n:
                            moves.append((di, gid, self._state_twin(gid, 0.0), n))
        for di, src, dst, n in moves:
            n = min(n, self.demes[di].get(src, 0))           # never drive a clone negative
            if n:
                self._remove(di, src, n)
                self._add(di, dst, n)

    def _tau_generation(self, rng, tau):
        """Advance the whole tumour by one generation of length `tau`, adaptively sub-stepping so
        the largest single-cell event probability per substep stays in the accurate Poisson
        regime (this also prevents carrying-capacity overshoot, since death rates are re-read each
        substep). Records a full per-clone snapshot every `snapshot_every` generations."""
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
        n_sub = max(1, int(np.ceil(max_cell_rate * tau / ACCURACY)))
        dt = tau / n_sub
        for _ in range(n_sub):
            self._tau_substep(rng, dt)
        # Drug-induced resistance-state entry/exit (DESIGN_phenotype_plasticity.md §3.3), once per
        # generation on the settled counts, before the snapshot. Gated: no draw / no genotype minted
        # when the feature is off, so the growth stream is byte-identical.
        if self.selection.resistance_state_on:
            self._apply_state_transitions(rng)
        self.step += 1
        self.time += tau
        if self._origin_confined:
            self._origin_tick(1.0)
        if self.step % self.snapshot_every == 0:
            self.traces.append(self._trace_snapshot())
            self.trace_times.append(self.time)

    def _grow_tau(self, n_steps=1000, seed=None, treatment=None, tau=None, **kwargs):
        """Tau-leaping growth: `n_steps` is the number of generations to advance. A full per-clone
        snapshot is recorded every `snapshot_every` generations, so plot_muller/plot_grid keep
        working unchanged -- now on a REAL-TIME x-axis (self.trace_times, in generation units)."""
        if seed is None:
            seed = self.seed
        if tau is None:
            tau = self.tau
        rng = np.random.default_rng(seed + self.step)
        self._clear_treatment_overrides()
        # snapshot at t=0 (matches the exact engine's initial trace)
        self.traces.append(self._trace_snapshot())
        self.trace_times.append(self.time)
        prev_dose = 0.0
        for _ in range(n_steps):
            if self.get_cancer_size() == 0:
                break
            if treatment is not None:
                treatment.discrete_event(self, self.step)  # e.g. surgical resection (R9)
                dose = treatment.get_dosage(self.step, self.get_tumor_size())
                self._apply_treatment(treatment, dose)
                prev_dose = dose
            self._tau_generation(rng, tau)
        self.make_cell_data()
        return self.traces

    # --- F8 microenvironment fields (DESIGN_features §H) ---------------------
    # All operate on the regular grid_size x grid_size deme lattice (deme i at
    # (i//G, i%G)); computed once per make_cell_data (a snapshot), so tau-leaping-compatible.
    # NOTE (future extension): these modulate the expression READOUT only. Coupling the
    # microenvironment to FITNESS (hypoxia slowing division, etc.) is deliberately left for later.
    def _deme_density(self):
        """Per-deme occupancy = total cells / carrying capacity (drives O2 consumption)."""
        cc = self._cap
        return np.array([sum(d.values()) / cc for d in self.demes], dtype=float)

    def _emitter_density(self, emitter_type, weight=None):
        """Per-deme density of a ligand-emitting cell type (e.g. immune).

        ``weight`` (a ``gid -> per-cell level`` map) turns the bare count into per-deme LIGAND
        availability: each emitter cell contributes its ligand EXPRESSION instead of 1 (W3,
        DESIGN_cci_spatial.md). ``None`` keeps the original count fraction — the O2 perfusion source
        and the field/emitter tests use it, so those stay byte-identical."""
        cc = self._cap
        out = np.zeros(len(self.demes))
        for i, deme in enumerate(self.demes):
            acc = 0.0
            for gid, c in deme.items():
                if self.genotypes[gid].type == emitter_type:
                    acc += c * (1.0 if weight is None else float(weight.get(gid, 0.0)))
            out[i] = acc / cc
        return out

    def _o2_field(self, D=1.0, k=1.0, s=0.2, source="uniform", n_iter=500, tol=1e-5):
        """Steady-state O2 on the deme grid (BioFVM-style), returned as hypoxia = 1 - O2.

        Solves D∇²O2 + supply·(1-O2) - k·density·O2 = 0 by Jacobi relaxation with zero-flux edges.
        O2 ∈ [0,1] (a weighted average of neighbours and the supply target 1), so hypoxia ∈ [0,1].

        `source` sets where O2 comes from:
          * ``"uniform"`` — supplied everywhere (a well-vascularised-tissue assumption). Gives a
            hypoxic core only when the tumour has an oxygenated (empty/normal) margin.
          * ``"perfused"`` — supplied ONLY by non-cancer (perfused stroma / empty) tissue:
            supply ∝ (1 - cancer density). A solid cancer mass or a cancer-filled duct then develops
            a hypoxic core from the O2 diffusion limit (comedonecrosis in DCIS) even with no empty
            margin, because the vasculature does not live inside the tumour.
        """
        G = self.grid_size
        dens = self._deme_density().reshape(G, G)
        if source == "perfused":
            perfusion = np.clip(1.0 - self._emitter_density("cancer"), 0.0, 1.0).reshape(G, G)
            supply = s * perfusion
        else:
            supply = s
        nn = np.full((G, G), 4.0)
        nn[0, :] -= 1; nn[-1, :] -= 1; nn[:, 0] -= 1; nn[:, -1] -= 1
        O2 = np.ones((G, G))
        for _ in range(n_iter):
            nb = np.zeros((G, G))
            nb[1:, :] += O2[:-1, :]; nb[:-1, :] += O2[1:, :]
            nb[:, 1:] += O2[:, :-1]; nb[:, :-1] += O2[:, 1:]
            new = (D * nb + supply) / (D * nn + supply + k * dens)
            if np.max(np.abs(new - O2)) < tol:
                O2 = new
                break
            O2 = new
        return np.clip(1.0 - O2.ravel(), 0.0, 1.0)

    def _cci_field(self, emitter_type="immune", lengthscale=2.0, weight=None):
        """Per-deme ligand signal = neighbourhood-averaged emitter LIGAND availability (Gaussian
        `lengthscale`). ``weight`` (gid -> ligand level) weights each emitter by its ligand
        expression rather than its bare density (W3); ``None`` -> the original density field."""
        emit = self._emitter_density(emitter_type, weight=weight)
        coords = np.array(self.deme_coords, dtype=float)
        diff = coords[:, None, :] - coords[None, :, :]
        d2 = np.sum(diff * diff, axis=-1)
        L = max(float(lengthscale), 1e-9)
        W = np.exp(-d2 / (2.0 * L * L))
        wsum = W.sum(axis=1)
        return np.divide(W @ emit, wsum, out=np.zeros_like(emit), where=wsum > 0)

    def _microenv_deme_mod(self, exp_cache=None):
        """F8 expression modifier: the HYPOXIA per-deme x gene matrix (n_demes x n_genes), or None.

        Hypoxia stays a genuine per-deme row: `mod[deme, hypoxia_genes] *= 1 + strength·hypoxia[deme]`.

        CCI (W3, DESIGN_cci_spatial.md) is now RECEPTOR-DEPENDENT, so it is NOT folded into this
        matrix — its multiplier is per-(deme, GENOTYPE) and is applied at materialisation. Here we
        compute the two inputs that multiplier needs and stash them on `self.microenv_truth`:
          * ``cci`` — ligand availability per deme: the smoothed emitter field, but each emitter
            weighted by its LIGAND expression (the wired pair's ligand gene) instead of a bare count;
          * ``cci_receptor_level`` — each genotype's expression of the wired RECEPTOR gene.
        The received signal on cell c is ``cci[deme(c)]·receptor_level[gid(c)]`` and the CCI
        multiplier on its target genes is ``1 + strength·received``.

        NORMALISATION (a definition, not a knob — getting it wrong silently rescales the calibrated
        ``strength``). ``_cci_field`` returned a density in [0,1] (a fraction of carrying capacity),
        which is what gives ``strength`` a stable meaning; weighting by RAW ligand expression and
        multiplying by RAW receptor expression breaks that bound. So each term is divided by its
        population mean: the ligand weight by the emitter cells' mean ligand expression (so ``cci``
        reduces EXACTLY to the old density field when ligand is uniform across genotypes), the
        receptor by ALL cells' mean receptor expression (so ``receptor_level`` averages ~1). Both
        then average to ~1 and ``1 + strength·received`` recovers the old ``1 + strength·cci_signal``
        in magnitude, so the existing ``strength`` calibration still applies.
        """
        mp = self.microenv_params
        if not mp:
            return None
        hyp = (mp.get("hypoxia") or {}) if isinstance(mp, dict) else {}
        cci = (mp.get("cci") or {}) if isinstance(mp, dict) else {}
        n_demes = len(self.demes)
        mod = np.ones((n_demes, self.n_genes))
        hypoxia = np.zeros(n_demes)
        if len(self._hypoxia_genes) and float(hyp.get("strength", 0.0)) != 0.0:
            hypoxia = self._o2_field(D=float(hyp.get("o2_diffusion", 1.0)),
                                     k=float(hyp.get("o2_consumption", 1.0)),
                                     s=float(hyp.get("o2_supply", 0.2)),
                                     source=hyp.get("o2_source", "uniform"))
            mod[:, self._hypoxia_genes] *= (1.0 + float(hyp["strength"]) * hypoxia[:, None])

        # --- W3 CCI: receptor-dependent channel (the wired pair) ------------------------------
        cci_signal = np.zeros(n_demes)                    # per-deme ligand availability (normalised)
        receptor_level = {}
        cci_strength = float(cci.get("strength", 0.0))
        cci_on = (len(self._cci_target_genes) and cci_strength != 0.0
                  and self._cci_ligand is not None and exp_cache)
        if cci_on:
            lig, rec = self._cci_ligand, self._cci_receptor
            etype = cci.get("emitter_type", "immune")
            lig_raw = {gid: float(exp_cache[gid][lig]) for gid in exp_cache}
            rec_raw = {gid: float(exp_cache[gid][rec]) for gid in exp_cache}
            # count-weighted population means over the FULL deme composition (unmaterialised
            # emitters still shape the field; genotypes we hold no expression for are skipped).
            lig_s = lig_n = rec_s = rec_n = 0.0
            for deme in self.demes:
                for gid, c in deme.items():
                    if gid not in lig_raw:
                        continue
                    rec_s += c * rec_raw[gid]; rec_n += c
                    if self.genotypes[gid].type == etype:
                        lig_s += c * lig_raw[gid]; lig_n += c
            mean_lig = (lig_s / lig_n) if lig_n > 0 else 1.0
            mean_rec = (rec_s / rec_n) if rec_n > 0 else 1.0
            mean_lig = mean_lig if mean_lig > 0 else 1.0
            mean_rec = mean_rec if mean_rec > 0 else 1.0
            weight = {gid: lig_raw[gid] / mean_lig for gid in lig_raw}
            receptor_level = {gid: rec_raw[gid] / mean_rec for gid in rec_raw}
            cci_signal = self._cci_field(etype, float(cci.get("lengthscale", 2.0)), weight=weight)
            # THE L-R SIGNAL ITSELF. The wired pair's ligand and receptor are up-regulated in the
            # signalling niche — where emitters are dense, both the ligand and its receptor rise
            # together. This is the whole point of the channel: every CCI tool (CellChat,
            # CellPhoneDB, COMMOT) scores a pair from the LIGAND and RECEPTOR genes' own expression,
            # so a channel that leaves them untouched is invisible no matter how strong its
            # downstream effect. An unwired decoy gets no such boost, so "ligand-high cells sitting
            # near receptor-high cells" is true of the wired pair and of nothing else.
            # Per-DEME (field only, not receptor-gated) so it lives in this matrix; the receptor-gated
            # part stays on the downstream TARGET genes at materialisation.
            mod[:, [lig, rec]] *= (1.0 + cci_strength * cci_signal[:, None])

        self.microenv_truth = dict(
            hypoxia_genes=np.asarray(self._hypoxia_genes),
            cci_target_genes=np.asarray(self._cci_target_genes),
            hypoxia=hypoxia, cci=cci_signal,
            # W0/W3 ground truth: the candidate L-R database, which pair is wired, and the
            # per-genotype receptor level that (with `cci`) makes the received signal per cell.
            cci_pairs=np.asarray(self._cci_pairs, dtype=int),
            cci_wired_pair=(0 if len(self._cci_pairs) else -1),
            cci_ligand=self._cci_ligand, cci_receptor=self._cci_receptor,
            cci_strength=cci_strength, cci_receptor_level=receptor_level)
        return mod

    # --- materialisation: counts -> per-cell matrices ------------------------
    def _hap_cns(self, gid):
        """Per-segment ``(cn_p, cn_m)`` allele-specific copy numbers of clone ``gid`` — the number of
        physical copies on each homolog, read straight off the genome copy lists (their lengths)."""
        g = self.genotypes[gid]
        return [(len(g.genome[s]['p']), len(g.genome[s]['m'])) for s in range(self.n_segments)]

    def _reconstruct_passengers(self, snv_mat, types):
        """Overlay a LINEAGE-CONSISTENT, allele-resolved neutral (passenger) SNV layer on the sampled
        cells, replacing the old per-cell-independent reconstruction (DESIGN_snv_coalescent.md v1,
        "star-at-founding").

        For each cancer clone ``Y`` whose subtree is a PROPER subset of the sample (a split between
        sampled cells) we place a set of STEM passengers "born on ``Y``'s founding edge"
        (count ≈ ``F_STEM``·``_pass_load[Y]``, floored so every split is markable, capped so the
        finite neutral-site budget spreads across splits). This layer is deliberately SUBCLONAL only:
        the clonal (truncal) burden of a tumour is carried by the founder genome itself (see
        ``founder_mutations``), so that it is present whether or not passengers are coarsened. A
        marker picks a
        neutral position (globally unique so clades stay homoplasy-free in the collapsed view), a
        homolog ∝ its copy number, and starts at multiplicity 1. Its multiplicity is then propagated
        down every descendant clone through the exact per-clone allele-specific CN events (WGD ×2;
        amplification +1 w.p. mult/cn; deletion −1 w.p. mult/cn, lost at 0 — the §4.3 marginal of the
        engine's per-copy process). A materialised cell of clone ``Z`` therefore carries the stem
        passengers of ``Z`` AND all its ancestors (each at its propagated multiplicity) plus a private
        per-cell tail (``Poisson((1−F_STEM)·load/count)`` singletons). Cells in the same clade now
        SHARE passenger SNVs, so a passenger-only tree recovers the clone-tree clades — which the old
        independent draw (≈ noise) could not.

        Modifies ``snv_mat`` in place (the collapsed VAF = total variant copies / seg_cn, so existing
        readers are unchanged in shape/semantics) and RETURNS the allele-resolved ``(snv_p, snv_m)``
        homolog-VAF frames (the primary truth; ``snv_p + snv_m == snv_mat``), or ``None`` for the
        byte-identical no-op fallback (coarsening off / nothing folded / no neutral sites / no sampled
        cancer). Dedicated seeded rng -> reproducible."""
        if (not self.coarsen_passengers or not self._pass_load or not len(self._neutral_gene_ids)
                or snv_mat.shape[0] == 0):
            return None
        neutral = self._neutral_gene_ids
        # Keep the somatic overlay off the loci that are already spoken for: the patient's inherited
        # (germline) variants, and the mutations the founder cell carried when it transformed (which
        # are in every cancer cell and form the clonal peak). A reconstructed passenger never shares a
        # position with either, so a caller can tell the layers apart. Neither present -> the pool is
        # untouched -> byte-identical to a run without them.
        for taken in (getattr(self, "germline_sites", None), getattr(self, "seeded_truncal_sites", None)):
            if taken is not None and len(taken):
                neutral = neutral[~np.isin(neutral, taken)]
        seg_of = self._gene_segment
        n_neutral = len(neutral)
        if not n_neutral:
            return None

        # ---- materialised cancer clones + their ancestor closure (the sampled clone sub-tree) ------
        per_clone_cells = Counter(types)
        mat_cancer = [g for g in per_clone_cells if self._is_cancer(g)]
        if not mat_cancer:
            return None
        parents = self.genotypes_parents
        closure = set()
        for g in mat_cancer:
            cur = g
            while cur is not None and cur not in closure:
                closure.add(cur)
                cur = parents.get(cur)
        # children within the closure + roots (founder, or any closure node whose parent is outside)
        children = {g: [] for g in closure}
        roots = []
        for g in closure:
            p = parents.get(g)
            if p in closure:
                children[p].append(g)
            else:
                roots.append(g)
        for g in closure:                       # deterministic child order (by birth ordinal)
            children[g].sort(key=lambda c: self.genotypes[c].ord)
        roots.sort(key=lambda c: self.genotypes[c].ord)

        hap_cn = {g: self._hap_cns(g) for g in closure}   # gid -> [(cn_p, cn_m), ...] per segment
        seg_cn = {g: [cp + cm for (cp, cm) in hap_cn[g]] for g in closure}

        # Pre-order (root-first) traversal of the closure, built iteratively so tree DEPTH never hits
        # the recursion limit (a long linear driver chain can be deep). Reused for subtree sums and
        # multiplicity descent below.
        order = []
        stack = list(reversed(roots))
        while stack:
            g = stack.pop()
            order.append(g)
            stack.extend(reversed(children[g]))

        # subtree sampled-cell counts (a clone whose subtree is ALL sampled cancer cells marks no
        # observable split, so it is skipped to save the site budget for informative splits).
        total_cancer = sum(per_clone_cells[g] for g in mat_cancer)
        subtree = {g: per_clone_cells.get(g, 0) for g in closure}
        for g in reversed(order):                 # children precede parents in reverse pre-order
            p = parents.get(g)
            if p in subtree:
                subtree[p] += subtree[g]

        prng = np.random.default_rng(self.seed + PASSENGER_RNG_OFFSET)

        # ---- allocate stem markers to clones, drawing globally-unique neutral positions -----------
        # Position-level global uniqueness (one stem SNV per neutral position across the whole tree)
        # keeps the collapsed clades homoplasy-free; homolog is still resolved per SNV. A shuffled
        # position order + pointer makes the draw O(1) and deterministic. Informative splits are
        # marked first (by how balanced the split is) so a tight budget covers the big clades.
        pos_order = prng.permutation(neutral)
        pos_ptr = 0
        f_stem = F_STEM_DEFAULT
        born = {g: [] for g in closure}          # gid -> list of (pos, hom, seg)

        def _mark(g, n_markers, limit):
            """Place ``n_markers`` markers born on clone ``g``: an unused neutral position each (taken
            from the shuffled order, stopping at ``limit``), on a homolog drawn ∝ its copy number."""
            nonlocal pos_ptr
            cn_g = hap_cn[g]
            for _ in range(int(n_markers)):
                if pos_ptr >= limit:
                    return
                pos = int(pos_order[pos_ptr]); pos_ptr += 1
                s = int(seg_of[pos])
                cp, cm = cn_g[s]
                if cp + cm == 0:
                    continue                     # nullisomic segment: no copy to sit on, skip
                if cp == 0:
                    hom = 'm'
                elif cm == 0:
                    hom = 'p'
                else:
                    hom = 'p' if prng.random() < cp / (cp + cm) else 'm'
                born[g].append((pos, hom, s))

        # ---- stems: one budget per clone-split that separates sampled cells ------------------------
        # A clone whose subtree is EVERY sampled cancer cell marks no observable split, so it is
        # skipped and its sites stay available for the splits that do carry information. The clonal
        # layer of a real tumour is not built here at all — it is inherited from the founder genome
        # (``founder_mutations``), which is why it survives with coarsening switched off.
        # Total order (final `ord` tiebreaker) so allocation is independent of set-iteration order
        # (string hashing is per-process randomised) -> reproducible.
        informative = [g for g in closure if 0 < subtree[g] < total_cancer]
        informative.sort(key=lambda g: (min(subtree[g], total_cancer - subtree[g]),
                                        self._pass_load.get(g, 0), self.genotypes[g].ord),
                         reverse=True)
        for g in informative:
            if pos_ptr >= n_neutral:
                break
            load = self._pass_load.get(g, 0)
            _mark(g, min(max(int(round(f_stem * load)), STEM_FLOOR), STEM_CAP), n_neutral)

        # ---- propagate stem multiplicity down the closure (§4.3), record per-clone stem genome ----
        # A stem SNV is (pos, hom, seg, mult). At each clone we hold the surviving inherited SNVs +
        # those born there (mult 1); each child edge applies that edge's exact CN event to every
        # active SNV on the affected (seg, hom). clone_stem[Z] is the shared stem layer of clone Z.
        clone_stem = {}

        def _edge_event(parent_g, child_g):
            """The CN event on edge parent->child as ``('wgd',)`` or ``('cn', changes)`` where changes
            is a list of ``(seg, hom, delta, cn_before)``. Each registered child is ONE mutate() = one
            event: a CNV changes exactly one (seg, hom) by ±1, a WGD doubles EVERY homolog (so it
            changes many), an SNV changes none. Reading it off the exact per-homolog copy numbers, more
            than one changed (seg, hom) therefore uniquely identifies a WGD — no reliance on the
            monotone ``is_wgd`` flag (which cannot flag a repeat WGD)."""
            cp, cc = hap_cn[parent_g], hap_cn[child_g]
            changes = []
            for s in range(self.n_segments):
                dp = cc[s][0] - cp[s][0]
                dm = cc[s][1] - cp[s][1]
                if dp:
                    changes.append((s, 'p', dp, cp[s][0]))
                if dm:
                    changes.append((s, 'm', dm, cp[s][1]))
            if len(changes) > 1:
                return ('wgd',)
            return ('cn', changes)

        def _apply(active, event):
            """Return the child's active stem SNVs after applying `event` (mult propagation)."""
            ev = event[0]
            out = []
            if ev == 'wgd':
                for (pos, hom, s, mult) in active:
                    out.append((pos, hom, s, mult * 2))
                return out
            changes = event[1]
            for (pos, hom, s, mult) in active:
                m = mult
                for (cs, chom, delta, cn_before) in changes:
                    if cs != s or chom != hom:
                        continue
                    step = 1 if delta > 0 else -1
                    cn = cn_before
                    for _ in range(abs(delta)):
                        if cn <= 0:
                            break
                        if step > 0:             # amplification: duplicated copy is a mutant w.p. m/cn
                            if prng.random() < m / cn:
                                m += 1
                            cn += 1
                        else:                    # deletion: removed copy is a mutant w.p. m/cn
                            if prng.random() < m / cn:
                                m -= 1
                            cn -= 1
                if m > 0:
                    out.append((pos, hom, s, m))
            return out

        active_in = {g: [] for g in roots}        # stem SNVs entering each clone from its parent
        for g in order:                           # pre-order: a parent is processed before its children
            active = active_in.pop(g, [])
            for (pos, hom, s) in born[g]:
                active.append((pos, hom, s, 1))
            if g in per_clone_cells:
                clone_stem[g] = active
            for c in children[g]:
                active_in[c] = _apply(active, _edge_event(g, c))

        # ---- assemble per-clone overlay vectors, then per-cell rows (base alleles + stem + private) -
        snv_p = np.zeros_like(snv_mat)
        snv_m = np.zeros_like(snv_mat)
        base_p, base_m = {}, {}          # per-clone allele-resolved base (drivers + genome passengers)
        stem_p, stem_m = {}, {}          # per-clone stem overlay (homolog VAF added on top of base)
        # per-clone allele fraction CAPS: the most a homolog can contribute at a locus is cn_h/seg_cn
        # (all its copies mutated). Enforced at the end so a collision (an overlay SNV landing where the
        # base genome already mutated the same homolog+position) merges to "all copies mutated" instead
        # of over-counting past the physical copy number -> guarantees snv_p+snv_m == snv_mat and VAF<=1.
        seg_of_arr = np.asarray(seg_of)
        capfrac_p, capfrac_m = {}, {}
        for g in per_clone_cells:
            rep = self.genotypes[g]
            vp, vm = rep.get_snvs_alleles()
            base_p[g], base_m[g] = vp, vm
            sp = np.zeros_like(vp); sm = np.zeros_like(vm)
            if g in clone_stem:
                scn = seg_cn[g]
                for (pos, hom, s, mult) in clone_stem[g]:
                    cn = scn[s]
                    if cn > 0:
                        (sp if hom == 'p' else sm)[pos] += mult / cn
            stem_p[g], stem_m[g] = sp, sm
            hp = hap_cn.get(g) or self._hap_cns(g)   # cancer closure cached; normals computed here
            cp_seg = np.asarray([cp for (cp, cm) in hp], dtype=float)
            cm_seg = np.asarray([cm for (cp, cm) in hp], dtype=float)
            cn_seg = cp_seg + cm_seg
            denom = np.maximum(cn_seg[seg_of_arr], 1e-9)
            nz = cn_seg[seg_of_arr] > 0
            capfrac_p[g] = np.where(nz, cp_seg[seg_of_arr] / denom, 0.0)
            capfrac_m[g] = np.where(nz, cm_seg[seg_of_arr] / denom, 0.0)

        for i, g in enumerate(types):
            snv_p[i] = base_p[g] + stem_p[g]
            snv_m[i] = base_m[g] + stem_m[g]

        # private per-cell singletons (Poisson((1-f_stem)*load/count)); NOT inherited, so drawn fresh
        # per cell on sites unused by that cell (per-cell infinite sites).
        for i, g in enumerate(types):
            if g not in per_clone_cells or g not in self._pass_load or not self._is_cancer(g):
                continue
            cnt = self.genotypes_counts.get(g, 0)
            if not cnt:
                continue
            mu = (1.0 - f_stem) * self._pass_load[g] / cnt
            if mu <= 0:
                continue
            k = int(prng.poisson(mu))
            if k <= 0:
                continue
            used = set(np.flatnonzero(snv_p[i] + snv_m[i]).tolist())
            pool = neutral if not used else neutral[~np.isin(neutral, list(used))]
            if not len(pool):
                continue
            picks = prng.choice(pool, size=min(k, len(pool)), replace=False)
            scn = seg_cn.get(g)
            cn_g = hap_cn.get(g)
            for pos in picks:
                pos = int(pos); s = int(seg_of[pos])
                cn = scn[s]
                if cn <= 0:
                    continue
                cp, cm = cn_g[s]
                if cp + cm == 0:
                    continue
                if cp == 0:
                    hom_p = False
                elif cm == 0:
                    hom_p = True
                else:
                    hom_p = prng.random() < cp / (cp + cm)
                inc = 1.0 / cn
                if hom_p:
                    snv_p[i, pos] += inc
                else:
                    snv_m[i, pos] += inc

        # Clip each homolog to its physical fraction, then DERIVE the collapsed frame from the alleles
        # (single source of truth) so snv_p + snv_m == snv_mat exactly and every VAF is in [0, 1].
        for i, g in enumerate(types):
            np.minimum(snv_p[i], capfrac_p[g], out=snv_p[i])
            np.minimum(snv_m[i], capfrac_m[g], out=snv_m[i])
            snv_mat[i] = snv_p[i] + snv_m[i]
        return snv_p, snv_m

    def _materialize_plan(self, max_cells, region=None, depth_frac=None):
        """Per-deme ``{gid: n_to_materialise}`` for ``make_cell_data``.

        With no cap (or a tumour under it) every cell in scope is materialised (``n = count``), so
        output is unchanged. Above the cap, each (deme, genotype) bucket keeps ``Binomial(count,
        cap/total)`` cells — a representative subsample preserving spatial + clonal + cell-type
        proportions in expectation, the biopsy an assay actually takes — so the dense per-cell frames
        never exceed ~``max_cells`` rows regardless of tumour size. Deterministic (dedicated seeded
        rng).

        ``region`` (an iterable of deme indices) restricts materialisation to those demes, at FULL
        local density unless the region itself exceeds the cap. This is an IN-PLANE cut: it selects a
        2-D patch of the tissue (e.g. one part of a resected specimen). Non-region demes materialise
        nothing.

        ``depth_frac`` is the orthogonal DEPTH cut (a "parallel-axis" cut): each deme keeps only
        ``Binomial(count, depth_frac)`` of its cells, so the 2-D structure is left intact but every
        deme-column's occupancy drops. Because a deme's ``K`` stands for the cells in its 3-D column,
        this is exactly cutting through the depth — a thin histology/Visium SLICE of a tissue block
        keeps the whole 2-D field but only ~one layer of cells. Applied once up-front (dedicated seeded
        rng) so it composes deterministically with ``region`` and ``max_cells``."""
        reg = None if region is None else set(int(d) for d in region)

        # DEPTH cut: thin every deme's column to a fraction of its cells (a 2-D slice), up-front.
        if depth_frac is not None:
            drng = np.random.default_rng(self.seed + 20240731)
            src = []
            for deme in self.demes:
                out = {}
                for g, c in deme.items():
                    n = int(drng.binomial(c, depth_frac))
                    if n:
                        out[g] = n
                src.append(out)
        else:
            src = self.demes

        if depth_frac is None and reg is None:
            total = self.get_tumor_size()
        else:
            scope = range(len(src)) if reg is None else [d for d in reg if 0 <= d < len(src)]
            total = sum(sum(src[d].values()) for d in scope)
        subsample = max_cells is not None and total > max_cells and total > 0
        if not subsample and reg is None and depth_frac is None:
            return [dict(d) for d in self.demes], False, total
        frac = (max_cells / total) if subsample else 1.0
        mrng = np.random.default_rng(self.seed + 20240730)
        plan = []
        for di, deme in enumerate(src):
            if reg is not None and di not in reg:
                plan.append({})
                continue
            if not subsample:
                plan.append(dict(deme))
                continue
            keep = {}
            for gid, c in deme.items():
                n = int(mrng.binomial(c, frac))
                if n:
                    keep[gid] = n
            plan.append(keep)
        return plan, subsample, total

    def truncal_sites(self, ccf=0.95, cell_ids=None):
        """Locus indices of the tumour's **trunk** — the somatic variants essentially every cancer
        cell carries — MEASURED from the cells rather than looked up.

        The founder starts with a blank genome, so a tumour's clonal layer is not planted: it fixes
        in the founding duct before the lesion spreads. There is nothing to look up, and
        :attr:`seeded_truncal_sites` is empty unless ``founder_mutations`` was configured. This finds
        the trunk the way a study does — the somatic sites carried by at least ``ccf`` of cancer
        cells. Germline variants are excluded: they sit in every cell, tumour and normal alike, so
        they would otherwise pass the same test.

        Parameters
        ----------
        ccf : float, default 0.95
            Minimum cancer-cell fraction for a site to count as truncal. 0.95 is the usual clonal
            cutoff. The count is sensitive to it — a convention, not a constant of the tumour — so
            report the cutoff with the number.
        cell_ids : sequence of str, optional
            Restrict to these cells (e.g. one sample's). Defaults to every materialised cell.

        Returns
        -------
        numpy.ndarray
            Sorted locus indices, i.e. positions along the gene axis of ``cell_data["cell_snv"]``.
        """
        if self.cell_data is None:
            raise ValueError("cell_data is not materialised; call make_cell_data() first")
        snv = self.cell_data["cell_snv"]
        gids = self.cell_data["cell_type"].iloc[:, 0]
        if cell_ids is not None:
            snv, gids = snv.loc[list(cell_ids)], gids.loc[list(cell_ids)]
        is_cancer = np.array([self._is_cancer(g) for g in gids], dtype=bool)
        if not is_cancer.any():
            return np.array([], dtype=int)
        carried = (snv.values[is_cancer] > 0).mean(axis=0)
        germline = np.zeros(snv.shape[1], dtype=bool)
        sites = getattr(self, "germline_sites", None)
        if sites is not None and np.size(sites):
            germline[np.asarray(sites, dtype=int)] = True
        return np.flatnonzero((carried >= ccf) & ~germline)

    def primary_window(self, side, center=None):
        """Deme indices of a rectangular window of the PRIMARY grid, centred on ``center``
        ``(row, col)`` or the grid centre. Pass to ``make_cell_data(region=...)`` to materialise a
        dense spatial patch (e.g. a Visium slide's footprint) of a cm-scale tumour at real cell
        density, rather than thinning the whole tumour to a uniform subsample.

        Parameters
        ----------
        side : int or tuple of int
            Window size in demes: an int for a ``side``×``side`` square, or ``(rows, cols)`` for a
            rectangle. A real capture area is rarely square — the Visium v1 slide is 78×64 spots,
            i.e. wider than it is tall — so a rectangle is what fills one.
        center : tuple of int, optional
            ``(row, col)`` centre. Defaults to the grid centre. A window that would overhang the
            grid is SHIFTED back onto it, so the full requested size is always returned (clipped
            only if it is larger than the grid) — size a window to a capture area and you get that
            area, not a sliver of it.
        """
        G = self.grid_size
        n_r, n_c = (side, side) if np.isscalar(side) else side
        n_r, n_c = min(int(n_r), G), min(int(n_c), G)
        cr, cc = center if center is not None else (G // 2, G // 2)
        r0 = min(max(0, int(cr) - n_r // 2), G - n_r)
        c0 = min(max(0, int(cc) - n_c // 2), G - n_c)
        return [r * G + c for r in range(r0, r0 + n_r) for c in range(c0, c0 + n_c)]

    def make_cell_data(self, cell_prefix="C", max_cells=None, region=None, depth_frac=None, **kwargs):
        """Expand the per-deme genotype counts into per-cell ground-truth tables.

        Materialises one row per cell (each genotype's count becomes that many identical
        cells) into ``self.cell_data`` — a dict of dataframes keyed by cell name:
        evolutionary parameters, expression, SNV and CNV matrices, grid coordinates,
        cell-type and deme labels, plus any optional layers that are enabled (compartment,
        allele-resolved expression / RNA BAF, program activity). Called automatically at the
        end of ``grow``.

        Parameters
        ----------
        cell_prefix : str, optional
            Prefix for the generated cell names (default ``"C"``).
        max_cells : int, optional
            Cap on the number of cells materialised (defaults to the tumour's ``self.max_cells``).
            When the tumour is larger, a representative subsample is materialised instead of every
            cell (see ``_materialize_plan``) so the dense frames stay memory-bounded at cm-scale.
            ``None`` materialises every cell (byte-identical to before this cap existed).
        region : iterable of int, optional
            Deme indices to restrict materialisation to, at FULL local density (subject to
            ``max_cells``). Use this for a SPATIAL assay (e.g. a Visium slide) that samples a small
            window of a cm-scale tumour and needs that window at real cell density — a uniform
            whole-tumour subsample would thin it to ~1 cell/spot. ``None`` materialises the whole
            tumour.
        depth_frac : float, optional
            DEPTH cut (a "parallel-axis" cut): keep only this fraction of EACH deme's cells, leaving
            the 2-D structure intact but lowering every deme-column's occupancy. Because ``K`` stands
            for the deme's 3-D column, this is a thin histology/Visium SLICE of a tissue block — the
            whole 2-D field, one thin layer. ``None`` keeps the full column; ``0.5`` is a half-depth
            slice. Composes with ``region`` (an in-plane cut) and ``max_cells``.

        Returns
        -------
        dict
            ``self.cell_data`` — the per-cell dataframes.
        """
        if max_cells is None:
            max_cells = self.max_cells
        mat_plan, subsampled, want = self._materialize_plan(max_cells, region=region,
                                                            depth_frac=depth_frac)
        # Only for an explicit SECTION (region / depth_frac). A plain whole-tumour materialisation
        # being capped is the documented representative-biopsy behaviour, and warning on it would
        # fire on every cm-scale grow(). The trap is asking for a physical section and silently
        # getting a memory budget instead: the cap, not depth_frac, then sets the density, and a
        # Visium section capped this way reads ~1 cell/spot — a budget artefact that looks like a
        # modelling result. Note `max_cells=None` means "the tumour's own cap", NOT "uncapped".
        if subsampled and (region is not None or depth_frac is not None):
            warnings.warn(
                f"max_cells={max_cells} capped this section: {want:,} cells were selected by "
                f"region/depth_frac but only ~{max_cells:,} are materialised ({max_cells / want:.1%}). "
                f"The cap, not depth_frac, is setting the cell density. Pass a larger max_cells "
                f"(~{int(want * 1.2):,}) to let the physical section govern.",
                stacklevel=2)
        gene_names = self.selection.get_gene_names()
        onc_idx, tsg_idx = self.selection.get_oncogenes(), self.selection.get_tsgs()
        disp_idx, ir_idx, tr_idx = (self.selection.get_dispersal_genes(),
                                    self.selection.get_immune_resistant(),
                                    self.selection.get_treatment_resistant())
        breach_idx, ss_idx = self.selection.get_breach(), self.selection.get_stromal_survival()
        ms_idx = self.selection.get_met_survival()
        dt_idx = self.selection.get_drug_tolerance()
        snv_cache, cnv_cache, exp_cache, evo_cache = {}, {}, {}, {}
        # R13: per-genotype allele-resolved expression + the per-clone program drive (routes 1+2).
        # Both are per-CLONE, so they belong in this cache; the per-CELL `z` is drawn later.
        exp_p_cache, exp_m_cache, drive_cache = {}, {}, {}
        P = self.programs
        # Build per-genotype caches ONLY for genotypes that will be materialised — the whole plan
        # without a cap, the subsample's genotypes under one. At cm-scale iterating every genotype
        # here is itself an O(#genotypes × n_genes) memory/time wall, so the cap must bound it too.
        mat_gids = set()
        for keep in mat_plan:
            mat_gids.update(keep)
        for gid in mat_gids:
            rep = self.genotypes[gid]
            snv = rep.get_snvs()
            snv_cache[gid] = snv
            cnv_cache[gid] = rep.get_cnvs()
            if P is None:
                # legacy path — untouched, so output is bit-identical when the layer is off
                if rep.type == "cancer":
                    rep.baseline_exp = self.celltype_exps["cancer"]
                    exp_cache[gid] = rep.get_exp(self.selection.mut_effects)
                else:
                    exp_cache[gid] = self.celltype_exps[rep.type]
            else:
                rep.baseline_exp = self.celltype_exps[rep.type]
                if rep.type == "cancer":
                    ep, em = rep.get_exp_alleles(dosage_sensitivity=P.dosage_sensitivity,
                                                 snv_exp_effect=P.snv_exp_effect,
                                                 dosage_saturation=P.dosage_saturation)
                    # Route 1 reads the EVOLVED per-clone phenotype, so whatever moved fitness —
                    # CINner drivers, and now R14's epistasis multiplier — propagates into the
                    # programs by construction. Route 2 reads the clone's SNV VAFs.
                    drive_cache[gid] = P.clone_drive(rep.evolutionary_parameters,
                                                     rep.baseline_rates, snv)
                else:
                    # Normal cells are diploid and unmutated: a flat allele split, no genotype drive.
                    half = self.celltype_exps[rep.type] / 2.0
                    ep, em = half.copy(), half.copy()
                    drive_cache[gid] = np.zeros(P.n_programs)
                drive_cache[gid] = drive_cache[gid] + P.celltype_bias(rep.type)
                exp_p_cache[gid], exp_m_cache[gid] = ep, em
                exp_cache[gid] = ep + em
            # per-genotype evolutionary parameters + unique-mutated-driver tallies
            evo = dict(rep.evolutionary_parameters)
            evo["n_mut_onc"] = int((snv[onc_idx] > 0).sum())
            evo["n_mut_tsg"] = int((snv[tsg_idx] > 0).sum())
            evo["n_mut_disp"] = int((snv[disp_idx] > 0).sum())
            evo["n_mut_ir"] = int((snv[ir_idx] > 0).sum())
            evo["n_mut_tr"] = int((snv[tr_idx] > 0).sum())
            evo["n_mut_breach"] = int((snv[breach_idx] > 0).sum())
            evo["n_mut_ss"] = int((snv[ss_idx] > 0).sum())
            evo["n_mut_ms"] = int((snv[ms_idx] > 0).sum())
            evo["n_mut_dt"] = int((snv[dt_idx] > 0).sum())
            evo_cache[gid] = evo

        # F8: per-deme expression modifier (None -> disabled -> exp is bit-identical to the base
        # engine). Cache the modified expression per (deme, genotype) since many cells share both.
        # `exp_cache` is passed so the CCI ligand/receptor levels (W3) can be read per genotype.
        deme_mod = self._microenv_deme_mod(exp_cache)
        mod_exp_cache = {}
        # W3 CCI is receptor-dependent, so its multiplier is per-(deme, genotype), not a column of
        # `deme_mod` (which now carries hypoxia only). Extract the pieces the loop needs: the target
        # genes, the calibrated strength, the per-deme ligand availability and the per-genotype
        # receptor level. `None` when no CCI channel is wired -> the loop applies hypoxia alone.
        cci_apply = None
        if deme_mod is not None and self.microenv_truth.get("cci_ligand") is not None:
            _ct = np.asarray(self.microenv_truth["cci_target_genes"], dtype=int)
            _cs = float(self.microenv_truth["cci_strength"])
            _rl = self.microenv_truth["cci_receptor_level"]
            if _ct.size and _cs != 0.0 and _rl:
                cci_apply = (_ct, _cs, self.microenv_truth["cci"], _rl)

        # R13 route 3 — niche -> program: the F8 fields drive per-deme program activity, generalising
        # F8's hard-wired hypoxia/CCI gene sets (which still apply via `deme_mod`; the two routes
        # compose, see DESIGN_expression.md §3.1). The COMPARTMENT (epithelial / stromal live
        # fraction) is a niche field too (DESIGN_phenotype_plasticity.md §2 Part D): mapping it to the
        # seeded invasive/emt program in `niche_program_map` makes the SAME clone express the invasive
        # program more at the epithelial front than in the stroma — the genetic-vs-niche confound.
        # OFF-BY-DEFAULT: `niche_drive` is a zeros matrix (or None) unless `niche_program_map` names a
        # field, so this is byte-identical to before when route-3 is unconfigured.
        niche_drive = None
        if P is not None:
            fields = {}
            # `deme_mod is not None` is the F8 signal (microenv_truth then carries hypoxia/cci).
            if deme_mod is not None:
                fields["hypoxia"] = self.microenv_truth["hypoxia"]
                fields["cci"] = self.microenv_truth["cci"]
            if self.structure_radius > 0:
                epi_frac, stro_frac = self._compartment_fields()
                fields["epithelial"] = epi_frac
                fields["stromal"] = stro_frac
                # Record the compartment field as ground truth (create microenv_truth if F8 is off).
                if getattr(self, "microenv_truth", None) is None:
                    self.microenv_truth = {}
                self.microenv_truth["epithelial"] = epi_frac
                self.microenv_truth["stromal"] = stro_frac
            if fields:
                niche_drive = P.niche_drive(fields)

        rows_snv, rows_cnv, rows_exp, rows_evo, crd, types, demes_col, names = [], [], [], [], [], [], [], []
        rows_exp_p, rows_exp_m, rows_drive = [], [], []
        i = 0
        for deme_idx, keep in enumerate(mat_plan):
            if not keep:
                continue
            r, c = self.deme_coords[deme_idx]
            for gid in sorted(keep.keys(), key=lambda g: self.genotypes[g].ord):
                # W3: the receptor-dependent CCI multiplier for this (deme, genotype) — the same
                # scalar for every cell of the clone here (ligand availability at the deme × the
                # clone's own receptor level). `deme_mod` carries hypoxia; this carries CCI.
                cci_f = None
                if cci_apply is not None:
                    _ct, _cs, _la, _rl = cci_apply
                    cci_f = 1.0 + _cs * float(_la[deme_idx]) * float(_rl.get(gid, 0.0))
                if deme_mod is None:
                    exp_row = exp_cache[gid]
                else:
                    exp_row = mod_exp_cache.get((deme_idx, gid))
                    if exp_row is None:
                        exp_row = exp_cache[gid] * deme_mod[deme_idx]     # fresh array (hypoxia)
                        if cci_f is not None:
                            exp_row[_ct] *= cci_f                          # receptor-dependent CCI
                        mod_exp_cache[(deme_idx, gid)] = exp_row
                if P is not None:
                    mod_row = 1.0 if deme_mod is None else deme_mod[deme_idx]
                    ep_row, em_row = exp_p_cache[gid] * mod_row, exp_m_cache[gid] * mod_row
                    if cci_f is not None:                                  # keep alleles in step (§L3200)
                        ep_row[_ct] *= cci_f; em_row[_ct] *= cci_f
                    drive_row = drive_cache[gid]
                    if niche_drive is not None:
                        drive_row = drive_row + niche_drive[deme_idx]
                for _ in range(keep[gid]):
                    rows_snv.append(snv_cache[gid]); rows_cnv.append(cnv_cache[gid])
                    rows_exp.append(exp_row); rows_evo.append(evo_cache[gid])
                    crd.append((r, c)); types.append(gid); demes_col.append(deme_idx)
                    names.append(f"{cell_prefix}{i}"); i += 1
                    if P is not None:
                        rows_exp_p.append(ep_row); rows_exp_m.append(em_row)
                        rows_drive.append(drive_row)

        # R13: apply the per-CELL program activity. `z` is per-cell (cycling is a cell state, not a
        # clone constant) while the genotype drive is per-clone — so the drive sets each clone's MEAN
        # and `activity_noise` supplies the within-clone spread. One matmul over all cells keeps this
        # affordable despite breaking the per-(deme, genotype) expression sharing.
        Z = None
        if P is not None and rows_exp_p:
            drive = np.asarray(rows_drive, dtype=float)
            Z = P.sample_z(drive, P.activity_rng())
            prog_mult = P.gene_multiplier(Z)                      # (n_cells, n_genes)
            ep = np.asarray(rows_exp_p) * prog_mult
            em = np.asarray(rows_exp_m) * prog_mult
            rows_exp_p, rows_exp_m = ep, em
            rows_exp = ep + em

        idx = pd.Index(names)
        empty = np.empty((0, self.n_genes))
        # SNV matrix, with coarsened-away passenger SNVs re-emitted per cell (no-op when coarsening is
        # off or nothing was folded). Built once and reused for cell_rna_vaf below.
        snv_mat = np.array(rows_snv) if rows_snv else empty
        snv_alleles = self._reconstruct_passengers(snv_mat, types)
        # DTYPES. These frames are (n_cells x n_genes) and dominate the memory at cm-scale, so they
        # are stored no wider than the values need. Copy number is a small non-negative integer
        # (int16 covers 0..32767 against a realistic ceiling of ~10): LOSSLESS at a quarter of int64.
        # Expression is float32 — a level, nowhere near float64's 15 significant digits.
        # `cell_snv` STAYS float64 on purpose. Several suites pin a golden md5 of its raw bytes to
        # assert that turning a feature off leaves the growth stream byte-identical; narrowing it
        # would force those baselines to be regenerated, and a re-blessed baseline is exactly how a
        # real perturbation would slip past them. It is also the layer that least needs it — 1.4%
        # non-zero, and a pure function of the genotype, so sparsity or de-duplication beat a
        # narrower dtype without touching the digests.
        def _frame(mat, dtype):
            return pd.DataFrame(np.asarray(mat, dtype=dtype), index=idx, columns=gene_names)

        self.cell_data = dict(
            cell_evo=pd.DataFrame(rows_evo, index=idx),
            cell_snv=_frame(snv_mat, np.float64),
            cell_cnv=_frame(np.array(rows_cnv) if rows_cnv else empty, np.int16),
            # NB `len(...)`, not truthiness: with the program layer on, `rows_exp` is an ndarray.
            cell_exp=_frame(np.array(rows_exp) if len(rows_exp) else empty, np.float32),
            cell_crd=pd.DataFrame(crd if crd else np.empty((0, 2)), index=idx, columns=["row", "col"]).astype(int),
            cell_type=pd.DataFrame(types, index=idx, columns=["cell_id"]),
            cell_deme=pd.DataFrame(demes_col, index=idx, columns=["deme_id"]),
        )
        # Allele-resolved neutral SNV layer (the PRIMARY truth of the coalescent passenger overlay,
        # DESIGN_snv_coalescent.md §4.5): per-homolog VAF frames whose sum is exactly `cell_snv`.
        # Only present when the overlay actually ran (coarsening on + folded burden + neutral sites),
        # so the base schema is unchanged otherwise — the F8 off-by-default discipline.
        if snv_alleles is not None:
            snv_p, snv_m = snv_alleles
            self.cell_data["cell_snv_p"] = _frame(snv_p, np.float64)
            self.cell_data["cell_snv_m"] = _frame(snv_m, np.float64)
        # cell_rna_vaf (F7b): the EXPECTED allele FRACTION in RNA (not an observed VAF). With m
        # mutant + w wt copies at a locus and per-locus expression effect e (selection.mut_effects:
        # oncogene=2, TSG=0.5, else 1), the fraction of expression from mutant alleles is
        # (m·e·base)/(m·e·base + w·base) = (m·e)/(m·e + w) = (v·e)/(v·e + (1-v)) with v = DNA-VAF
        # (cell_snv). The per-gene baseline expression CANCELS in this fraction, which is why at a
        # neutral locus (e=1) the *expected fraction* equals DNA-VAF; e>1 inflates, e<1 deflates.
        # This is NOT what a caller observes: the OBSERVED scRNA-VAF is this fraction SAMPLED at a
        # depth equal to the gene's expression (the F3 UMI count, which carries the per-gene
        # baseline + cell-type + CNV + SNV-effect scaling). Per-gene expression varies enormously,
        # so most loci drop out or are noisy and the observed VAF does NOT match DNA-VAF at neutral
        # loci — the read layer (reads/rna.py) does the depth-aware sampling and the obs_fidelity
        # distortion; both deliberately live outside the engine.
        flat_eff = (np.concatenate(self.selection.mut_effects)
                    if self.selection.mut_effects else np.ones(self.n_genes))
        v = self.cell_data["cell_snv"].values
        num = v * flat_eff
        denom = num + (1.0 - v)
        rna_vaf = np.divide(num, denom, out=np.zeros_like(v, dtype=float), where=denom > 0)
        self.cell_data["cell_rna_vaf"] = _frame(rna_vaf, np.float64)
        # WGD ground truth (DESIGN_focal_cna.md v1): the per-genotype `is_wgd` flag surfaced per cell,
        # so downstream CNA / BAF benchmarks (e.g. Numbat) can score WGD detection against the truth.
        # Gated on WGD being enabled — off -> the frame is absent and the base schema is unchanged
        # (the F8 discipline, mirroring cell_microenv below). `types` is the per-cell genotype id.
        if self._cancer_params is not None and self._cancer_params.get("wgd_rate", 0):
            self.cell_data["cell_wgd"] = pd.DataFrame(
                {"is_wgd": [bool(self.genotypes[g].is_wgd) for g in types]}, index=idx)
        # Drug-induced resistance-state ground truth (DESIGN_phenotype_plasticity.md §3.3): the
        # per-genotype carried state level surfaced per cell, so a downstream method can be scored on
        # telling a non-genetic tolerant STATE from a resistance DRIVER (the Whiting & Graham
        # plasticity-vs-genetic split, with iscc's lineage tracing as ground truth). Gated on the
        # feature being on, so off -> the frame is absent and the base schema is unchanged.
        if self.selection.resistance_state_on:
            self.cell_data["cell_resistance_state"] = pd.DataFrame(
                {"resistance_state": [float(getattr(self.genotypes[g], "resistance_state", 0.0))
                                      for g in types]}, index=idx)
        # F8: surface the per-cell cell-extrinsic levels (ground truth for the intrinsic-vs-extrinsic
        # decomposition benchmark). Only added when F8 is enabled, so the base schema is unchanged.
        # `cci_level` is now the per-cell RECEIVED signal (W3): ligand availability at the cell's
        # deme × the cell's own receptor level — so it varies cell-to-cell by clone within a deme,
        # not just deme-to-deme. It is exactly the quantity the CCI target-gene fold-change reads
        # (fold = 1 + strength·cci_level), and the ground truth W4 scores a CCI method against.
        if deme_mod is not None:
            dcol = np.asarray(demes_col, dtype=int)
            hyp, lig = self.microenv_truth["hypoxia"], self.microenv_truth["cci"]
            rl = self.microenv_truth.get("cci_receptor_level", {})
            if dcol.size:
                recv = lig[dcol] * np.array([float(rl.get(g, 0.0)) for g in types])
            else:
                recv = np.array([])
            self.cell_data["cell_microenv"] = pd.DataFrame(
                {"hypoxia_level": hyp[dcol] if dcol.size else np.array([]),
                 "cci_level": recv}, index=idx)

        # Ductal-field ground truth (DESIGN_ductal_field.md §2): the gland each cell sits in (-1 for
        # stroma). Only added when a gland field was seeded, so the base schema is unchanged otherwise.
        if self.gland_id is not None:
            gcol = np.asarray(demes_col, dtype=int)
            self.cell_data["cell_gland"] = pd.DataFrame(
                {"gland_id": self.gland_id[gcol] if gcol.size else np.array([], dtype=int)}, index=idx)

        # Metastasis ground truth (R9): the compartment each cell sits in (0 = primary, 1 = met). Only
        # added when the met is enabled (base schema unchanged otherwise). Flows through the sample/data
        # glob loaders automatically, so an assay can subset to a single site.
        if self._met_enabled:
            ccol = np.asarray(demes_col, dtype=int)
            self.cell_data["cell_compartment"] = pd.DataFrame(
                {"compartment": ((ccol >= self.n_primary_demes).astype(int) if ccol.size
                                 else np.array([], dtype=int))}, index=idx)

        # R13: the allele layers + the RNA BAF, and the per-cell program activity. Only added when
        # the program layer is on, so the base schema is unchanged.
        if P is not None:
            if P.allele_specific:
                ep = np.asarray(rows_exp_p) if len(rows_exp_p) else empty
                em = np.asarray(rows_exp_m) if len(rows_exp_m) else empty
                tot = ep + em
                # BAF in RNA: the fraction of a gene's expression coming from the `p` homolog. 0.5 at
                # a balanced diploid locus; an allelic imbalance (an amplified or lost homolog, or an
                # NMD SNV in cis) pushes it away from 0.5. This is the layer Numbat / CalicoST read,
                # and it is exactly what summing p+m in `get_exp` used to destroy.
                baf = np.divide(ep, tot, out=np.full_like(tot, 0.5, dtype=float), where=tot > 0)
                self.cell_data["cell_exp_p"] = pd.DataFrame(ep, index=idx, columns=gene_names)
                self.cell_data["cell_exp_m"] = pd.DataFrame(em, index=idx, columns=gene_names)
                self.cell_data["cell_rna_baf"] = pd.DataFrame(baf, index=idx, columns=gene_names)
            if Z is not None and P.n_programs:
                self.cell_data["cell_program"] = pd.DataFrame(
                    Z, index=idx, columns=list(P.dictionary.program_names))
            self.program_truth = P.truth()
        return self.cell_data

    # --- plotting (shared, engine-agnostic) ----------------------------------
    def plot_muller(self, ax=None, by_drivers=False, **kwargs):
        """Render a Muller plot of the clonal dynamics (clone frequencies over steps, nested by ancestry).

        ``by_drivers=True`` colours by distinct DRIVER-mutation combinations (Noble's demon
        convention) instead of by genotype, collapsing passenger-only diversity; combine with
        ``min_freq`` for a size threshold. Additional keywords (``ax``, ``colormap``,
        ``normalize``) select the target axes and control the styling."""
        from .. import viz
        driver_map = self._driver_signatures() if by_drivers else None
        return viz.plot_muller(self.traces, self.genotypes_parents, ax=ax,
                               driver_map=driver_map, **kwargs)

    def _driver_signatures(self):
        """Map every cancer genotype id -> a hashable signature of its mutated DRIVER genes (the
        oncogene/TSG loci, ``driver_types != 0``, carrying an SNV). Covers ancestors too (needed to
        build the driver-clone ancestry). Genotypes sharing a signature are one 'driver clone'."""
        dts = getattr(self.selection, "driver_types", None)
        if not dts:
            return {}
        driver_idx = np.flatnonzero(np.concatenate(dts) != 0)
        if driver_idx.size == 0:
            return {}
        out = {}
        for gid, rep in self.genotypes.items():
            if gid in normal_names or not hasattr(rep, "get_snvs"):
                continue
            vafs = rep.get_snvs()
            out[str(gid)] = tuple(int(i) for i in driver_idx[vafs[driver_idx] > 0])
        return out

    def _functional_signatures(self):
        """Map every cancer genotype id -> a hashable signature of its mutated FUNCTIONAL genes: ALL the
        selection axes (oncogene/TSG + dispersal + immune/treatment resistance + breach/stromal/met
        survival) carrying an SNV. Unlike ``_driver_signatures`` (onc/TSG only), sharing a signature
        means sharing the mutations selection acts on, so a sweep on ANY axis (breach, met_survival,
        treatment_resistance, ...) becomes one functional clone -> one Muller band -> visible selection."""
        s = self.selection
        parts = [s.get_oncogenes(), s.get_tsgs(), s.get_dispersal_genes(), s.get_immune_resistant(),
                 s.get_treatment_resistant(), s.get_breach(), s.get_stromal_survival(),
                 s.get_met_survival(), s.get_drug_tolerance()]
        parts = [np.asarray(p, dtype=int) for p in parts if len(p)]
        if not parts:
            return {}
        func_idx = np.unique(np.concatenate(parts))
        out = {}
        for gid, rep in self.genotypes.items():
            if gid in normal_names or not hasattr(rep, "get_snvs"):
                continue
            vafs = rep.get_snvs()
            out[str(gid)] = tuple(int(i) for i in func_idx[vafs[func_idx] > 0])
        return out

    def _stage_colors(self):
        """{gid -> rgba} colouring each CANCER genotype by its STAGE-DOMINANT driver — the highest arc
        stage it has activated (from its heritable traits), on the SAME 7-stage scheme + colours as the
        landing hero: 1 proliferation (division rate raised by an onc/TSG mutation), 2 breach (duct escape),
        3 stromal survival, 4 met survival, 5 drug tolerance, 6 chemo resistance; 0 = no driver. Normal
        cells keep their type colours. This is the todo #14 categorical view: a few colours tracking the
        selection cascade.

        Returns the map plus the sorted set of stages actually present, for the legend."""
        from .. import viz
        from ...constants import normal_cmap_rgba
        out, present = {}, set()
        for gid, rep in self.genotypes.items():
            if getattr(rep, "type", None) != "cancer":
                out[str(gid)] = normal_cmap_rgba.get(rep.type, (0.8, 0.8, 0.8, 1.0))
                continue
            s = self._stage_of(rep)
            out[str(gid)] = viz.STAGE_PALETTE[s]
            present.add(s)
        return out, sorted(present)

    def _stage_of(self, rep):
        """Stage-dominant driver of one CANCER genotype, on the SAME 7-stage scheme as the landing hero
        (``iscc.tumor.arc.stage_of``): 0 none, 1 proliferation (division raised above baseline), 2 breach
        (duct escape), 3 stromal survival, 4 met survival, 5 drug tolerance, 6 chemo resistance — the
        highest arc stage its heritable traits have activated. The return value INDEXES
        ``viz.STAGE_PALETTE`` / ``viz.STAGE_NAMES``. Robust to missing evo keys."""
        ep = rep.evolutionary_parameters
        div = ep.get("division_rate", 0.0)
        base = getattr(rep, "baseline_rates", {}).get("division_rate", div)
        # Drug escape as the ENGINE defines it: protection is max(treatment_resistance, drug_tolerance)
        # (see _apply_treatment), so a pure persister takes zero drug and must not read as sensitive;
        # tolerance is its own stage, and claims the cell only when it STRICTLY exceeds resistance —
        # the same tie-break _apply_treatment uses to decide who pays the persister cost. See
        # arc.stage_of, which this mirrors.
        tr = ep.get("treatment_resistance", 0)
        dt = ep.get("drug_tolerance", 0.0)
        if max(tr, dt) > 0:
            return 5 if dt > tr else 6
        if ep.get("met_survival", 0) > 0:
            return 4
        if ep.get("stromal_survival", 0) > 0:
            return 3
        if ep.get("breach", 0) > 0:
            return 2
        if div > base + 1e-9:
            return 1
        return 0

    def plot_tissue(self, ax=None, color="state", cmap=None, figsize=(7, 7), min_freq=0.02):
        """Deme-resolution tissue map built directly from the per-deme genotype COUNTS.

        Unlike ``plot_grid`` (which reads the possibly-subsampled ``cell_data`` and so goes holey at
        cm-scale when ``max_cells`` thins the tissue), this colours EVERY occupied deme from the full
        counts, so the spatial overview is always complete. Primary grid only.

        Parameters
        ----------
        color : {"state", "cancer_frac", "clone", "stage"}
            ``"state"`` (default): categorical — empty / stroma / duct(normal) / DCIS (cancer in a
            duct) / IDC (cancer in stroma). ``"cancer_frac"``: the per-deme cancer fraction as a
            heatmap. ``"clone"``: colour each cancer deme by its dominant clone's DRIVER-clone colour
            (the same driver-collapsed palette as ``plot_muller(by_drivers=True)`` and
            ``plot_phylogeny``), on a pale-stroma background. ``"stage"``: colour each cancer deme by
            the STAGE-dominant trait of its dominant clone (none / proliferation / invasion / …) — the
            same scheme (and palette) as the landing-animation hero — on a pale-stroma background.
        min_freq : float, optional
            For ``color="clone"``, the size threshold (fraction of the cancer population) below which
            driver clones are merged into their nearest surviving ancestor before colouring — matched
            to the Muller's ``min_freq`` so the two share a palette. Default 0.02.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        G = self.grid_size
        gid_is_can = {g: self._is_cancer(g) for g in self.genotypes}
        if color == "cancer_frac":
            img = np.full(G * G, np.nan)
            for di in range(min(len(self.demes), G * G)):
                d = self.demes[di]
                tot = sum(d.values())
                if tot:
                    img[di] = sum(c for g, c in d.items() if gid_is_can[g]) / tot
            im = ax.imshow(img.reshape(G, G), cmap=cmap or "magma", vmin=0, vmax=1, interpolation="nearest")
            plt.colorbar(im, ax=ax, fraction=0.046, label="cancer fraction")
        elif color == "clone":
            from .. import viz
            clone_colors = viz.functional_clone_colors(
                self.traces, self.genotypes_parents, driver_map=self._driver_signatures(), min_freq=min_freq)
            default = (0.6, 0.6, 0.6, 1.0)
            rgba = np.tile(np.array([0.95, 0.91, 0.91, 1.0]), (G * G, 1))   # pale-stroma background
            for di in range(min(len(self.demes), G * G)):
                d = self.demes[di]
                best_g, best_c = None, 0
                for g, c in d.items():
                    if gid_is_can[g] and c > best_c:
                        best_g, best_c = g, c
                if best_g is not None:
                    rgba[di] = clone_colors.get(str(best_g), default)
            ax.imshow(rgba.reshape(G, G, 4), interpolation="nearest")
        elif color == "stage":
            from .. import viz
            from matplotlib.colors import to_hex
            from matplotlib.patches import Patch
            img = np.full(G * G, -1, dtype=int)             # -1 = empty/stroma background
            for di in range(min(len(self.demes), G * G)):
                d = self.demes[di]
                best_g, best_c = None, 0
                for g, c in d.items():
                    if gid_is_can[g] and c > best_c:
                        best_g, best_c = g, c
                if best_g is not None:
                    img[di] = self._stage_of(self.genotypes[best_g])
            img = img.reshape(G, G)
            ax.imshow(np.zeros((G, G)), cmap=ListedColormap(["#f3e7e7"]))    # pale stroma
            cm = ListedColormap(list(viz.STAGE_PALETTE))
            ax.imshow(np.ma.masked_where(img < 0, img), cmap=cm, vmin=0,
                      vmax=len(viz.STAGE_PALETTE) - 1, interpolation="nearest")
            names = ["none", "proliferation", "breach", "stromal", "met survival", "drug tolerance",
                     "chemo resist"]
            present = sorted({int(v) for v in img.ravel() if v >= 0})
            if present:
                ax.legend(handles=[Patch(color=to_hex(viz.STAGE_PALETTE[i]), label=names[i]) for i in present],
                          loc="upper right", fontsize=8, framealpha=0.9)
        else:
            # 0 empty, 1 stroma, 2 duct(normal), 3 DCIS (cancer in duct), 4 IDC (cancer in stroma)
            img = np.zeros(G * G, dtype=int)
            for di in range(min(len(self.demes), G * G)):
                d = self.demes[di]
                ncan = sum(c for g, c in d.items() if gid_is_can[g])
                induct = self.gland_id is not None and self.gland_id[di] >= 0
                img[di] = (3 if induct else 4) if ncan > 0 else (2 if (induct and d) else (1 if d else 0))
            cm = cmap or ListedColormap(["white", "#f3d3d3", "#2e8b57", "#4b0082", "#d62728"])
            ax.imshow(img.reshape(G, G), cmap=cm, vmin=0, vmax=4, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        return ax

    def plot_phylogeny(self, ax=None, min_freq=0.02, figsize=(7, 7), linewidth=1.4):
        """Radial CLADE phylogeny drawn inline with matplotlib (Cassiopeia-style).

        Rather than subsampling cells, the whole cancer population is COLLAPSED to driver clones — the
        same functional collapse the by_drivers Muller uses, size-merged at ``min_freq`` — and the tree
        is the clade genealogy: each surviving clade is one node, placed radially (founder at the centre,
        depth = clade ancestry), coloured by its driver-clone colour (matching ``plot_muller(by_drivers)``
        and ``plot_tissue("clone")``) and sized by its cell count.

        Renders straight to ``ax`` (no image file); returns the matplotlib ``Axes``. Primary grid only.
        """
        import matplotlib.pyplot as plt
        from collections import defaultdict
        from .. import viz
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        driver_map = self._driver_signatures()
        # the SAME display basis the by_drivers Muller draws on: full population collapsed to driver
        # clones and size-merged at min_freq (no cell subsampling).
        basis_counts, basis_parents, _gmap, basis_cols = viz._display_basis(
            self.traces, self.genotypes_parents, driver_map=driver_map, min_freq=min_freq)
        present = [str(c) for c in basis_cols]
        if not present:
            raise ValueError("no cancer clones to build a phylogeny")
        clone_colors = viz.functional_clone_colors(
            self.traces, self.genotypes_parents, driver_map=driver_map, min_freq=min_freq)
        default = (0.6, 0.6, 0.6, 1.0)
        # clade genealogy: each clade -> its parent clade (from the collapsed basis)
        pmap = ({str(c): str(basis_parents.iloc[0][c]) for c in basis_parents.columns}
                if basis_parents.shape[1] else {})
        presset = set(present)
        children, roots = defaultdict(list), []
        for c in present:
            p = pmap.get(c)
            (children[p].append(c) if (p in presset and p != c) else roots.append(c))
        root = roots[0] if len(roots) == 1 else "__root__"
        if len(roots) != 1:
            for r in roots:
                children[root].append(r)
        nodes = set(present) | {root}
        peak = {c: float(basis_counts[c].max()) if c in basis_counts else 0.0 for c in present}
        pmax = max(peak.values()) or 1.0
        # radial layout: DFS leaf order -> even angles; internal angle = mean of children; radius = depth
        depth = {root: 0}
        order, post, stack = [], [], [(root, False)]
        while stack:
            n, done = stack.pop()
            if done:
                post.append(n); continue
            ch = children.get(n, [])
            if not ch:
                order.append(n); post.append(n); continue
            stack.append((n, True))
            for c in reversed(ch):
                depth[c] = depth[n] + 1
                stack.append((c, False))
        nleaf = max(len(order), 1)
        angle = {lf: 2 * np.pi * (i + 0.5) / nleaf for i, lf in enumerate(order)}
        for n in post:
            if n not in angle:
                angle[n] = float(np.mean([angle[c] for c in children[n]]))
        maxd = max(depth.values()) or 1
        col = {n: tuple(np.asarray(clone_colors.get(n, default), float).tolist()) for n in nodes}

        def r_of(n):                                # founder at the centre, clades on rings by depth
            return depth[n] / maxd
        # arc at each clade's radius spanning its children, radial spokes out to each child clade
        for n in nodes:
            ch = children.get(n, [])
            if not ch:
                continue
            rn = r_of(n)
            cangs = [angle[c] for c in ch]
            a0, a1 = min(cangs), max(cangs)
            arc = np.linspace(a0, a1, max(2, int(np.ceil((a1 - a0) / 0.02)) + 1))
            ax.plot(rn * np.cos(arc), rn * np.sin(arc), color=col[n], lw=linewidth, solid_capstyle="round")
            for c in ch:
                ac, rc = angle[c], r_of(c)
                ax.plot([rn * np.cos(ac), rc * np.cos(ac)], [rn * np.sin(ac), rc * np.sin(ac)],
                        color=col[c], lw=linewidth, solid_capstyle="round")
        # one marker per clade, sized by its cell count
        for c in present:
            s = 40 + 360 * np.sqrt(peak.get(c, 0) / pmax)
            ax.scatter([r_of(c) * np.cos(angle[c])], [r_of(c) * np.sin(angle[c])], s=s,
                       color=col[c], edgecolors="black", linewidth=0.4, zorder=3)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_xlim(-1.12, 1.12); ax.set_ylim(-1.12, 1.12)
        return ax

    def he_image(self, px=6, darkness=0.8, sigma_frac=0.5):
        """Dense H&E-like morphology image built from the per-deme cell COUNTS.

        The tissue *image* a histology/Visium slide sits on. Unlike rasterising the (possibly
        subsampled) ``cell_data`` — which is a THIN section and so renders faint and holey at
        cm-scale — this paints EVERY occupied deme from the full counts, so denser tissue (packed
        DCIS ducts, invasive masses) reads dark and looser stroma reads pale, giving a complete,
        high-contrast image. Cancer nuclei tint toward hematoxylin purple; stroma stays eosin pink.

        The image is in the raw deme-grid frame (``px`` pixels per deme). To overlay a placed Visium
        slide, shift the spot coordinates back into this frame by the placement translation
        (``spot_coords.mean(0) - section_cell_crd.mean(0)``); see notebook ``03``.

        Parameters
        ----------
        px : int, default 6
            Pixels per deme.
        darkness : float, default 0.8
            Maximum staining of the densest tissue.
        sigma_frac : float, default 0.5
            Gaussian smoothing sigma as a fraction of ``px``.

        Returns
        -------
        image : numpy.ndarray, shape (G*px, G*px, 3), float32
            RGB H&E morphology image in ``[0, 1]``.
        px : float
            Pixels per deme (the scale factor mapping deme coords onto the image).
        """
        from scipy.ndimage import gaussian_filter
        G = self.grid_size
        gid_is_can = {g: self._is_cancer(g) for g in self.genotypes}
        tot = np.zeros(G * G); can = np.zeros(G * G)
        for di in range(min(len(self.demes), G * G)):
            d = self.demes[di]
            if not d:
                continue
            tot[di] = sum(d.values())
            can[di] = sum(c for g, c in d.items() if gid_is_can[g])
        tot = tot.reshape(G, G); can = can.reshape(G, G)
        frac = np.divide(can, tot, out=np.zeros_like(can), where=tot > 0)
        ref = np.percentile(tot[tot > 0], 98) if np.any(tot > 0) else 1.0
        dens = np.clip(tot / (ref or 1.0), 0.0, 1.0)
        dens = gaussian_filter(np.kron(dens, np.ones((px, px))), sigma=px * sigma_frac)
        fr = gaussian_filter(np.kron(frac, np.ones((px, px))), sigma=px * sigma_frac)
        d = darkness * dens
        # eosin-pink stroma darkening toward hematoxylin purple where cancer nuclei are dense
        image = np.stack([1 - d * (0.35 + 0.10 * fr),
                          1 - d * (0.62 + 0.33 * fr),
                          1 - d * (0.33 + 0.05 * fr)], axis=-1)
        return np.clip(image, 0.0, 1.0).astype(np.float32), float(px)

    def plot_grid(self, color=None, ax=None, **kwargs):
        """Plot the spatial deme grid, colouring each deme by a per-cell attribute.

        Rebuilds the per-cell data if stale, then draws the grid coloured by ``color``
        (default the dominant clone / cell type). When the metastatic compartment is enabled
        and no explicit ``ax`` is passed, both the primary and met grids are drawn side by
        side with a shared clone colormap.

        Parameters
        ----------
        color : str or list of str, optional
            Attribute key(s) to colour by; defaults to the dominant clone / cell type.
        ax : matplotlib.axes.Axes, optional
            Axes to draw into; forces the single-panel primary view. A new figure is created
            when ``None``.

        Returns
        -------
        matplotlib.axes.Axes
            The axes drawn into.
        """
        from .. import viz
        if self.cell_data is None or self.cell_data["cell_type"].shape[0] != self.get_tumor_size():
            self.make_cell_data()
        # With the metastatic deposit enabled, `plot_grid()` shows BOTH grids (primary + met) side by
        # side with a shared clone colormap, so the same clone is the same colour across compartments.
        # (Pass an explicit `ax` to force the single-panel primary view instead.)
        if self._met_enabled and ax is None and "cell_compartment" in self.cell_data:
            return self.plot_grid_compartments(color=color, **kwargs)
        return viz.plot_grid(self.cell_data, self.grid_size, self.traces,
                             self.genotypes_parents, color=color, ax=ax, **kwargs)

    def plot_grid_compartments(self, color=None, axes=None, by_drivers=False, by_stage=False, **kwargs):
        """Two spatial grids (primary + met) side by side, shared clone colormap. ``by_drivers``
        colours by functional clone (matching the driver-collapsed 2-band Muller); ``by_stage`` colours
        by the stage-dominant driver; pass ``min_freq`` to also merge below-threshold clones."""
        from .. import viz
        if self.cell_data is None or self.cell_data["cell_type"].shape[0] != self.get_tumor_size():
            self.make_cell_data()
        if by_stage:
            colors, present = self._stage_colors()
            kwargs.setdefault("clone_colors", colors)
            kwargs.setdefault("color_legend", viz.stage_legend(present))
        driver_map = self._functional_signatures() if by_drivers else None
        return viz.plot_grid_compartments(self.cell_data, self.grid_size, self.met_grid_size,
                                          self.traces, self.genotypes_parents, color=color,
                                          axes=axes, driver_map=driver_map, **kwargs)

    def plot_muller_compartments(self, axes=None, by_drivers=False, by_stage=False, star_seeder=True,
                                 **kwargs):
        """Two-band Muller (primary over met) sharing one colormap with the grids. ``by_drivers``
        collapses genotypes to FUNCTIONAL clones and ``min_freq`` (a fraction) merges below-threshold
        clones, so selective sweeps (DCIS→IDC breach, the met founder, post-chemo resistant escape) show
        as bands instead of a passenger rainbow. ``by_stage`` instead colours by the STAGE-DOMINANT
        driver. ``star_seeder`` stars the first met-seeding clone's band in BOTH panels at the
        seeding moment (so a minor founder is findable)."""
        from .. import viz
        driver_map = self._functional_signatures() if by_drivers else None
        if by_stage:
            colors, present = self._stage_colors()
            kwargs.setdefault("clone_colors", colors)
            kwargs.setdefault("color_legend", viz.stage_legend(present))
        if star_seeder and "star_genotype" not in kwargs:
            seeders = [e for e in self.events if e.get("event") == "seeding"]
            if seeders:
                kwargs["star_genotype"] = str(seeders[0]["genotype"])
                for i, tr in enumerate(self.traces):
                    if any(str(g) not in normal_names for g in tr.get("met_counts", {})):
                        kwargs["star_gen"] = i
                        break
        return viz.plot_muller_compartments(self.traces, self.genotypes_parents, axes=axes,
                                            driver_map=driver_map, **kwargs)

    def plot_muller_founders(self, ax=None, **kwargs):
        """A SINGLE primary Muller highlighting the clone(s) that seeded the metastasis, so you can
        see which — usually minor — primary population founds the met. Founders are read from the
        seeding events; a primary clone is highlighted iff it shares a functional signature with one."""
        from .. import viz
        founders = {e["genotype"] for e in self.events if e.get("event") == "seeding"}
        return viz.plot_muller_founders(self.traces, self.genotypes_parents, founders,
                                        self._functional_signatures(), ax=ax, **kwargs)

    def plot_clone_tree(self, ax=None, by_stage=False, by_drivers=False, **kwargs):
        """Ground-truth clone tree: the TRUE genotypes_parents genealogy collapsed to the
        display clones, each node at its true mutational depth, sized by peak population and coloured by
        ``by_stage`` (stage-dominant driver) or ``by_drivers`` (functional clone). Pass ``min_freq`` to
        set the collapse threshold (default 0.02)."""
        from .. import viz
        if by_stage:
            colors, present = self._stage_colors()
            kwargs.setdefault("clone_colors", colors)
            kwargs.setdefault("color_legend", viz.stage_legend(present))
        driver_map = self._functional_signatures() if by_drivers else None
        return viz.plot_clone_tree(self.traces, self.genotypes_parents, driver_map=driver_map,
                                   ax=ax, **kwargs)

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

    # --- epistasis ground truth (R14, DESIGN_epistasis.md §4) ----------------
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

    # --- operating-envelope QC (read-only; DESIGN_operating_envelope.md) ------
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
            # Per-compartment abundance (R9): only present when the met is enabled (the snapshots then
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

        gene_data = self.selection.get_gene_data()
        Path(os.path.join(output_path, "gene_data")).mkdir(parents=True, exist_ok=True)
        for mat in gene_data:
            pd.DataFrame(gene_data[mat]).to_csv(os.path.join(output_path, "gene_data", f"{mat}.csv"))

        if self.cell_data is None:
            self.make_cell_data()
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
