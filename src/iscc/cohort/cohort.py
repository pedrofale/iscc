"""The multi-patient COHORT layer (DESIGN_cohort.md §2).

Runs a DIFFERENT tumour per patient — each with its own **evolution** seed (private clones, private
passenger mutations, patient-specific CNAs and spatial structure) — over ONE **shared, config-
determined** landscape: the same ``layout_seed`` gives every patient the SAME driver/oncogene/TSG
gene identities and the same shared per-cell-type baseline expression (the engine seed-decoupling
fix, DESIGN_cohort.md §1). This is exactly what makes cohort analysis meaningful — recurrence,
shared-vs-private structure — and is exactly the ground truth a real cohort can never provide.

Optional **subgroups** (personalized medicine): patients are partitioned into molecular subtypes that
share the driver landscape but differ in EFFECT scalars (e.g. ``treatment_resistant_effects``), so —
coupled to :mod:`iscc.treatment` — a therapy benefits one subgroup and not another, with a known
answer key (the subgroup label + its true therapy response).

``Cohort`` is a thin wrapper over :class:`iscc.tumor.models.GenotypeTumor`; the heavy lifting (growth,
CINner selection, materialisation) is unchanged. The batch/multiplexing emission lives in
:mod:`iscc.cohort.batch` and the ground-truth tables in :mod:`iscc.cohort.groundtruth`.
"""
from dataclasses import dataclass, field
import warnings

import numpy as np

from ..tumor.models import GenotypeTumor
from ..tumor.components.selection import Selection
from ..constants import DEFAULT_LAYOUT_SEED

# selection_params keys that set gene POSITIONS (the layout). A subgroup delta touching these desyncs
# the resistance/IR gene identities across subgroups (make_drivers, using only prop_driver, runs first
# so oncogene/TSG identities stay shared regardless — but we warn so the user keeps the landscape
# comparable on purpose). See DESIGN_cohort.md §2.1.
_LAYOUT_PROP_KEYS = {"prop_driver", "prop_dispersal", "prop_treatment_resistance", "prop_immune_resistance"}


@dataclass
class Subgroup:
    """A molecular subtype: a named per-patient override of the shared selection params.

    ``selection_delta`` is merged onto the cohort's base ``selection_params`` for patients in this
    subgroup; keep it to EFFECT scalars (``driver_effects``, ``treatment_resistant_effects``,
    ``immune_resistant_effects``, ``dispersal_effects``) so the driver landscape stays shared.
    Resistance itself is NOT seeded: it must EMERGE from mutation + selection (the whole point of a
    mechanistic simulator — a bolt-on injection would hand-impose the answer). A subtype is defined by
    an EFFECT scalar such as ``treatment_resistant_effects``; during growth, resistance mutations arise
    spontaneously at the shared resistance loci and drift as neutral standing variation, and under
    therapy they are selected in the high-effect subtype (which relapses) and inert in the low-effect
    subtype (which is eradicated). See :func:`iscc.cohort.cohort` / the personalized-medicine benchmark.

    ``founder_mutations`` (flat gene indices) seed a TRUNCAL, INHERITED alteration into the founder
    genome (present in every cancer cell) — appropriate for a *germline*/hereditary subtype-defining
    background (e.g. a germline variant), NOT for modelling acquired resistance (which must emerge).
    ``therapy_response`` is a free-form ground-truth label (e.g. "sensitive"/"resistant", or a numeric
    expected response) — the stratification / biomarker answer key.
    """
    name: str
    selection_delta: dict = field(default_factory=dict)
    founder_mutations: tuple = ()
    therapy_response: object = None


@dataclass
class PatientResult:
    """One grown patient tumour + its cohort bookkeeping."""
    patient_id: int
    seed: int
    subgroup: str
    tumor: object                     # a grown GenotypeTumor (cell_data materialised)
    selection_params: dict

    @property
    def cell_data(self):
        return self.tumor.cell_data

    def cancer_size(self):
        return int(self.tumor.get_cancer_size())


class Cohort:
    """A cohort of N patient tumours over one shared, config-determined driver landscape.

    Parameters mirror :class:`GenotypeTumor` (``genome_params`` / ``selection_params`` /
    ``cancer_cell_params`` / ``deme_params`` / ``spatial_params`` / ``update_mode`` / ``tau`` /
    ``snapshot_every``) — every patient is built with these SAME params plus:

    patient_seeds : sequence[int]
        one **evolution** seed per patient (private dynamics). ``n_patients = len(patient_seeds)``.
    layout_seed : int
        the shared **layout** seed (default :data:`DEFAULT_LAYOUT_SEED`) — every patient's driver
        identities and baseline expression come from this, so the landscape is shared.
    subgroups : list[Subgroup] | None
        molecular subtypes (personalized medicine). ``None`` -> a single implicit subgroup "all".
    subgroup_assignment : list[str] | dict[int,str] | None
        patient -> subgroup name. ``None`` -> round-robin over ``subgroups`` (or all "all").
    grow_steps : int
        generations (tau) / events (exact) to grow each patient.
    """

    def __init__(self, patient_seeds, genome_params=None, selection_params=None,
                 cancer_cell_params=None, deme_params=None, spatial_params=None,
                 subgroups=None, subgroup_assignment=None, layout_seed=DEFAULT_LAYOUT_SEED,
                 grow_steps=400, update_mode="exact", tau=1.0, snapshot_every=1,
                 immune_cell_params=None, epithelial_cell_params=None, stromal_cell_params=None,
                 n_germline_markers=0, germline_seed=98765):
        self.patient_seeds = list(patient_seeds)
        self.n_patients = len(self.patient_seeds)
        self.genome_params = genome_params or {}
        self.selection_params = selection_params or {}
        self.cancer_cell_params = cancer_cell_params or {}
        self.deme_params = deme_params or {}
        self.spatial_params = spatial_params or {}
        self.immune_cell_params = immune_cell_params
        self.epithelial_cell_params = epithelial_cell_params
        self.stromal_cell_params = stromal_cell_params
        self.layout_seed = layout_seed
        self.grow_steps = grow_steps
        self.update_mode = update_mode
        self.tau = tau
        self.snapshot_every = snapshot_every

        # subgroups + assignment
        if subgroups is None:
            subgroups = [Subgroup(name="all")]
        self.subgroups = list(subgroups)
        self._subgroup_by_name = {s.name: s for s in self.subgroups}
        self.subgroup_assignment = self._resolve_assignment(subgroup_assignment)
        self._warn_layout_desync()

        # Per-patient PRIVATE germline markers (the individual-level background genetic
        # demultiplexing relies on). Each patient gets `n_germline_markers` DISJOINT non-driver
        # founder mutations shared by all its cancer cells -> a clean patient-of-origin fingerprint
        # for the demux benchmark. Ground truth in self.germline_markers[patient_idx].
        self.n_germline_markers = int(n_germline_markers)
        self.germline_seed = germline_seed
        self.germline_markers = self._draw_germline_markers()

        self.patients = []          # list[PatientResult], filled by run()

    # -- assignment ---------------------------------------------------------------------
    def _resolve_assignment(self, assignment):
        names = [s.name for s in self.subgroups]
        if assignment is None:
            return [names[i % len(names)] for i in range(self.n_patients)]
        if isinstance(assignment, dict):
            return [assignment.get(i, names[0]) for i in range(self.n_patients)]
        assignment = list(assignment)
        if len(assignment) != self.n_patients:
            raise ValueError(f"subgroup_assignment length {len(assignment)} != n_patients {self.n_patients}")
        for a in assignment:
            if a not in self._subgroup_by_name:
                raise ValueError(f"unknown subgroup {a!r}; known: {names}")
        return assignment

    def _warn_layout_desync(self):
        for s in self.subgroups:
            bad = _LAYOUT_PROP_KEYS & set(s.selection_delta)
            if bad:
                warnings.warn(
                    f"subgroup {s.name!r} delta changes layout prop(s) {sorted(bad)}; this desyncs the "
                    "resistance/IR gene identities across subgroups (oncogene/TSG identities stay "
                    "shared). Prefer EFFECT scalars for subgroups. See DESIGN_cohort.md §2.1.",
                    stacklevel=3)

    def patient_selection_params(self, patient_idx):
        """The shared base selection params merged with this patient's subgroup delta."""
        sub = self._subgroup_by_name[self.subgroup_assignment[patient_idx]]
        return {**self.selection_params, **sub.selection_delta}

    # -- shared landscape ---------------------------------------------------------------
    def _shared_selection(self):
        """A lightweight Selection over the shared landscape (base params + layout_seed) — its
        gene-role layout is identical to every patient's, without building a full tumour."""
        return Selection(n_segments=self.genome_params["n_segments"],
                         segment_size=self.genome_params.get("segment_size", 1000),
                         segment_sizes=self.genome_params.get("segment_sizes"),
                         rng=np.random.default_rng(self.layout_seed), **self.selection_params)

    def _draw_germline_markers(self):
        """Disjoint per-patient private germline marker sets, sampled from NON-driver positions so
        they are neutral (true "germline SNPs", not somatic drivers). Returns {patient_idx: indices}
        or ``None`` if disabled."""
        if not self.n_germline_markers:
            return None
        sel = self._shared_selection()
        driver = (set(int(x) for x in sel.get_oncogenes()) | set(int(x) for x in sel.get_tsgs())
                  | set(int(x) for x in sel.get_treatment_resistant())
                  | set(int(x) for x in sel.get_immune_resistant())
                  | set(int(x) for x in sel.get_dispersal_genes()))
        pool = np.array([g for g in range(sel.n_genes) if g not in driver])
        total = self.n_patients * self.n_germline_markers
        if total > len(pool):
            raise ValueError(f"need {total} germline markers but only {len(pool)} non-driver genes; "
                             "use a larger genome or fewer markers/patients")
        chosen = np.random.default_rng(self.germline_seed).choice(pool, size=total, replace=False)
        chosen = chosen.reshape(self.n_patients, self.n_germline_markers)
        return {i: chosen[i] for i in range(self.n_patients)}

    # -- tumour construction ------------------------------------------------------------
    def _build_tumor(self, patient_idx):
        """Build (unGROWN) a patient tumour: shared layout_seed, private evolution seed, merged params."""
        seed = self.patient_seeds[patient_idx]
        kwargs = dict(
            seed=seed, layout_seed=self.layout_seed,
            genome_params=self.genome_params, selection_params=self.patient_selection_params(patient_idx),
            cancer_cell_params=self.cancer_cell_params, deme_params=self.deme_params,
            spatial_params=self.spatial_params, update_mode=self.update_mode, tau=self.tau,
            snapshot_every=self.snapshot_every,
        )
        for name, val in (("immune_cell_params", self.immune_cell_params),
                          ("epithelial_cell_params", self.epithelial_cell_params),
                          ("stromal_cell_params", self.stromal_cell_params)):
            if val is not None:
                kwargs[name] = val
        t = GenotypeTumor(**kwargs)
        sub = self._subgroup_by_name[self.subgroup_assignment[patient_idx]]
        founder_muts = list(sub.founder_mutations)
        if self.germline_markers is not None:
            founder_muts = founder_muts + list(self.germline_markers[patient_idx])
        if founder_muts:
            _apply_founder_mutations(t, founder_muts)
        return t

    def grow_patient(self, patient_idx, treatment=None, grow_steps=None):
        """Build and grow one patient (optionally under a therapy). Returns the grown GenotypeTumor.

        Because the tumour is a pure function of (layout_seed, seed, params), calling this with and
        without ``treatment`` yields two runs that share their entire pre-therapy setup and differ
        ONLY in the therapy — the clean per-patient treatment-response ground truth."""
        steps = self.grow_steps if grow_steps is None else grow_steps
        t = self._build_tumor(patient_idx)
        t.grow(n_steps=steps, seed=self.patient_seeds[patient_idx], treatment=treatment)
        t.make_cell_data()
        return t

    def run(self, treatment=None, grow_steps=None):
        """Grow every patient (optionally under a shared ``treatment``) and store PatientResults."""
        self.patients = []
        for i in range(self.n_patients):
            t = self.grow_patient(i, treatment=treatment, grow_steps=grow_steps)
            self.patients.append(PatientResult(
                patient_id=i, seed=self.patient_seeds[i], subgroup=self.subgroup_assignment[i],
                tumor=t, selection_params=self.patient_selection_params(i)))
        return self

    # -- convenience --------------------------------------------------------------------
    @property
    def selection(self):
        """The shared Selection (gene-role layout) — identical across patients. Uses a grown patient's
        if available, else a lightweight shared Selection (same layout, no full tumour build)."""
        if self.patients:
            return self.patients[0].tumor.selection
        return self._shared_selection()

    def treated_cancer_sizes(self, treatment, grow_steps=None):
        """Per-patient final cancer size under ``treatment`` (fresh runs; does not clobber ``run()``)."""
        return np.array([self.grow_patient(i, treatment=treatment, grow_steps=grow_steps).get_cancer_size()
                         for i in range(self.n_patients)], dtype=float)

    def subgroup_of(self, patient_idx):
        return self._subgroup_by_name[self.subgroup_assignment[patient_idx]]


def _apply_founder_mutations(tumor, gene_indices):
    """Seed TRUNCAL mutations into the founder genotype BEFORE growth (a subtype-defining clonal
    driver present in every cancer cell). Sets the mutated bit on one allele of each named flat gene
    index and refreshes the founder's genome summary, evolutionary parameters, and deme rates. Safe
    because only the founder exists at this point (no daughters share its genome yet).
    """
    rep = tumor.genotypes[tumor.founder_id]
    sel = tumor.selection
    offs = np.concatenate([[0], np.cumsum(sel.segment_sizes)]).astype(int)
    for g in gene_indices:
        g = int(g)
        seg = int(np.searchsorted(offs, g, side="right") - 1)
        pos = g - int(offs[seg])
        allele = rep.genome[seg]["p"][0]
        if allele[pos]:
            continue
        allele[pos] = True
        mut_bits = np.zeros(sel.segment_sizes[seg], dtype=bool)
        mut_bits[pos] = True
        rep.update_genome_summary_mutation(sel, mut_bits, seg)
    rep.update_evolutionary_parameters(sel)
    # the founder's rates changed -> refresh every deme rate so event sampling is correct
    tumor.deme_rates = np.array([tumor._deme_rate(i) for i in range(len(tumor.demes))], dtype=float)
