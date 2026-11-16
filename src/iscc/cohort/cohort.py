"""The multi-patient COHORT layer: a thin utility that grows one patient at a time.

A cohort is a list of DIFFERENT tumours, each grown with its own **evolution** seed (private clones,
private passenger mutations, patient-specific CNAs and spatial structure) over ONE **shared** landscape:
the same ``layout_seed`` gives every patient the SAME driver/oncogene/TSG gene identities, the same
baseline expression and programs, and the same epistasis network, whatever the evolution seed. That is
what makes cohort analysis meaningful — recurrence, shared-vs-private structure — and it is exactly the
ground truth a real cohort can never provide.

:class:`Cohort` owns nothing of the tumour model. It takes a ``grow`` function — any function that grows
ONE tumour from keyword arguments and returns it — and calls it once per patient with that patient's
seed, the shared layout seed and the patient's optional overrides. A cohort therefore grows exactly as
its single tumours would::

    import functools
    from iscc.cohort import Cohort, grow_tumor

    # a GenotypeTumor grown for a fixed number of steps
    grow = functools.partial(grow_tumor, genome_params=..., selection_params=..., grow_steps=300)
    cohort = Cohort(grow, patient_seeds=range(1, 21)).run()

    # the paper regime (validation/paper/regime.py), grown to a target size, one patient per task
    grow = functools.partial(RG.grow, scale="small", target_cancer=15_000)
    tumour = Cohort(grow, patient_seeds=range(101, 401)).grow_patient(task_id)

Subgroups (molecular subtypes differing in an effect scalar) and per-patient private germline markers
are per-patient OVERRIDES of the growth function's keywords: :func:`patient_overrides` builds them, and
:func:`grow_tumor` is the growth function that understands them.

Clone identity across patients
------------------------------
Genotype ids come from a process-wide counter (``Cell._genotype_counter``), so they are unique within
one process but NOT across processes: patients grown in separate Slurm tasks reuse the same ids, and
even within one process the ids a patient gets depend on what was grown before it. A clone is therefore
identified across a cohort by the PAIR ``(patient_id, genotype_id)``, never by the genotype id alone.
:func:`iscc.cohort.batch.pool_cell_data` already namespaces pooled cells as ``P{patient}::{cell}`` and
carries ``patient`` and ``clone`` as separate columns; :func:`clone_key` gives the same namespace to any
table pooled from per-patient outputs.
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
# comparable on purpose).
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
    subtype (which is eradicated).

    ``germline_mutations`` (flat gene indices) seed an INHERITED germline variant into EVERY cell of the
    patient — the founder cancer cell AND every normal cell type present (epithelial, stromal, immune,
    and a metastatic deposit's host parenchyma) — since germline
    variants are present in every cell of an individual, not only the tumour. Appropriate for a
    hereditary subtype-defining background (e.g. a germline predisposition variant), NOT for modelling
    acquired resistance (which must emerge). ``therapy_response`` is a free-form ground-truth label
    (e.g. "sensitive"/"resistant", or a numeric expected response) — the stratification answer key.

    To :class:`Cohort` a Subgroup is only a label; what it changes in growth travels in the per-patient
    overrides :func:`patient_overrides` builds from it.
    """
    name: str
    selection_delta: dict = field(default_factory=dict)
    germline_mutations: tuple = ()
    therapy_response: object = None


_ALL = Subgroup(name="all")


@dataclass
class PatientResult:
    """One grown patient tumour + its cohort bookkeeping."""
    patient_id: int
    seed: int
    subgroup: str
    tumor: object                     # whatever the cohort's ``grow`` returned
    kwargs: dict                      # the patient's overrides of ``grow``'s keywords

    @property
    def cell_data(self):
        return self.tumor.cell_data

    def cancer_size(self):
        return int(self.tumor.get_cancer_size())


def clone_key(patient_id, genotype_id):
    """The cohort-wide identity of a clone, ``"P{patient}::{genotype}"``.

    Genotype ids are unique only within one process (see the module docstring), so a clone pooled from
    per-patient outputs has to carry its patient. This is the namespace
    :func:`iscc.cohort.batch.pool_cell_data` gives pooled cells."""
    return f"P{int(patient_id)}::{genotype_id}"


class Cohort:
    """A cohort of patient tumours over one shared, layout-determined landscape.

    Parameters
    ----------
    grow : callable
        Grows ONE tumour and returns it. Called as ``grow(seed=..., layout_seed=..., **kwargs)``; any
        function of that shape works — :func:`grow_tumor` for a :class:`GenotypeTumor` grown a fixed
        number of steps, or the paper regime's ``grow`` (through ``functools.partial`` for its scale
        and target size).
    patient_seeds : sequence[int]
        One **evolution** seed per patient (private dynamics). ``n_patients = len(patient_seeds)``.
    layout_seed : int
        The shared **layout** seed (default :data:`DEFAULT_LAYOUT_SEED`), passed to every patient:
        driver identities, baseline expression, programs and the epistasis network come from it.
    patient_kwargs : sequence[dict] | None
        Per-patient overrides of ``grow``'s keywords (one dict per patient), e.g. a subgroup's
        ``selection_params`` or a patient's ``germline_mutations``; see :func:`patient_overrides`.
    subgroups : sequence[Subgroup] | None
        Per-patient subgroup LABELS, for the ground-truth tables (bookkeeping only: what a subgroup
        changes in growth is in ``patient_kwargs``). ``None`` puts every patient in one subgroup,
        ``"all"``.
    """

    def __init__(self, grow, patient_seeds, layout_seed=DEFAULT_LAYOUT_SEED, patient_kwargs=None,
                 subgroups=None):
        if not callable(grow):
            raise TypeError("Cohort's first argument is the function that grows one tumour "
                            f"(e.g. functools.partial(grow_tumor, ...)), not a {type(grow).__name__}")
        self.grow = grow
        self.patient_seeds = list(patient_seeds)
        self.n_patients = len(self.patient_seeds)
        self.layout_seed = layout_seed
        if patient_kwargs is None:
            patient_kwargs = [{} for _ in range(self.n_patients)]
        self.patient_kwargs = [dict(k or {}) for k in patient_kwargs]
        if len(self.patient_kwargs) != self.n_patients:
            raise ValueError(f"patient_kwargs has {len(self.patient_kwargs)} entries for "
                             f"{self.n_patients} patients")
        for k in self.patient_kwargs:
            clash = {"seed", "layout_seed"} & set(k)
            if clash:
                raise ValueError(f"patient_kwargs may not set {sorted(clash)}: the seed is the "
                                 "patient's own and the layout seed is shared by construction")
        self.subgroups = list(subgroups) if subgroups is not None else [_ALL] * self.n_patients
        if len(self.subgroups) != self.n_patients:
            raise ValueError(f"subgroups has {len(self.subgroups)} entries for {self.n_patients} patients")
        self.patients = []          # list[PatientResult], filled by run()

    @property
    def subgroup_assignment(self):
        """Each patient's subgroup name."""
        return [s.name for s in self.subgroups]

    def subgroup_of(self, patient_idx):
        return self.subgroups[patient_idx]

    def grow_patient(self, patient_idx, **overrides):
        """Grow patient ``patient_idx`` and return the tumour — what one Slurm array task calls.

        ``overrides`` go on top of the patient's own ``patient_kwargs`` (e.g. ``treatment=...`` for a
        treated run of the same patient: the tumour is a pure function of its seeds and keywords, so the
        untreated and treated runs share their whole pre-therapy history)."""
        kwargs = {**self.patient_kwargs[patient_idx], **overrides}
        return self.grow(seed=self.patient_seeds[patient_idx], layout_seed=self.layout_seed, **kwargs)

    def run(self, **overrides):
        """Grow every patient, one at a time, and keep a :class:`PatientResult` per patient."""
        self.patients = []
        for i in range(self.n_patients):
            t = self.grow_patient(i, **overrides)
            self.patients.append(PatientResult(
                patient_id=i, seed=self.patient_seeds[i], subgroup=self.subgroups[i].name,
                tumor=t, kwargs=self.patient_kwargs[i]))
        return self

    @property
    def selection(self):
        """The shared gene-role layout (a grown patient's ``Selection``; identical across patients).
        Before :meth:`run`, build it with :func:`shared_selection`."""
        if not self.patients:
            raise RuntimeError("no patient grown yet: run the cohort first, or build the shared "
                               "layout with iscc.cohort.shared_selection(genome_params, ...)")
        return self.patients[0].tumor.selection


# ---------------------------------------------------------------------------------- growth function
def grow_tumor(seed, layout_seed=DEFAULT_LAYOUT_SEED, grow_steps=400, treatment=None,
               germline_mutations=(), **tumor_params):
    """Build one :class:`GenotypeTumor`, seed its inherited germline variants, and grow it for
    ``grow_steps`` steps (optionally under ``treatment``); its cell data are materialised at the end.

    ``tumor_params`` are :class:`GenotypeTumor`'s keywords (``genome_params``, ``selection_params``,
    ``cancer_cell_params``, ``deme_params``, ``spatial_params``, the normal-cell blocks,
    ``expression_params``, ``microenv_params``, ``update_mode``, ...). ``germline_mutations`` (flat gene
    indices) are written into every cell of the patient before growth (see :class:`Subgroup`)."""
    t = GenotypeTumor(seed=seed, layout_seed=layout_seed, **tumor_params)
    if len(germline_mutations):
        _apply_germline_mutations(t, list(germline_mutations))
    t.grow(n_steps=grow_steps, seed=seed, treatment=treatment)
    return t


def shared_selection(genome_params, selection_params=None, layout_seed=DEFAULT_LAYOUT_SEED):
    """A lightweight :class:`Selection` over the shared landscape: the gene-role layout every patient
    of a cohort at ``layout_seed`` gets, without building a tumour."""
    return Selection(n_segments=genome_params["n_segments"],
                     segment_size=genome_params.get("segment_size", 1000),
                     segment_sizes=genome_params.get("segment_sizes"),
                     rng=np.random.default_rng(layout_seed),
                     layout_seed=layout_seed, **(selection_params or {}))


# ------------------------------------------------------------------------ per-patient overrides
def assign_subgroups(n_patients, subgroups, assignment=None):
    """Each patient's :class:`Subgroup`: ``assignment`` is a list of names, a ``{patient: name}``
    dict (unlisted patients get the first subgroup), or ``None`` for round-robin over ``subgroups``."""
    subgroups = list(subgroups)
    by_name = {s.name: s for s in subgroups}
    names = [s.name for s in subgroups]
    for s in subgroups:
        bad = _LAYOUT_PROP_KEYS & set(s.selection_delta)
        if bad:
            warnings.warn(
                f"subgroup {s.name!r} delta changes layout prop(s) {sorted(bad)}; this desyncs the "
                "resistance/IR gene identities across subgroups (oncogene/TSG identities stay "
                "shared). Prefer EFFECT scalars for subgroups.",
                stacklevel=2)
    if assignment is None:
        chosen = [names[i % len(names)] for i in range(n_patients)]
    elif isinstance(assignment, dict):
        chosen = [assignment.get(i, names[0]) for i in range(n_patients)]
    else:
        chosen = list(assignment)
        if len(chosen) != n_patients:
            raise ValueError(f"subgroup assignment length {len(chosen)} != n_patients {n_patients}")
    for a in chosen:
        if a not in by_name:
            raise ValueError(f"unknown subgroup {a!r}; known: {names}")
    return [by_name[a] for a in chosen]


def draw_germline_markers(selection, n_patients, n_markers, seed=98765):
    """Disjoint per-patient PRIVATE germline marker sets — the individual-level background genetic
    demultiplexing relies on. Drawn from NON-driver positions of the shared layout ``selection`` (see
    :func:`shared_selection`) so they are neutral germline SNPs, not somatic drivers. Returns
    ``{patient_idx: gene indices}``, the ground truth of each patient's fingerprint."""
    driver = (set(int(x) for x in selection.get_oncogenes()) | set(int(x) for x in selection.get_tsgs())
              | set(int(x) for x in selection.get_treatment_resistant())
              | set(int(x) for x in selection.get_immune_resistant())
              | set(int(x) for x in selection.get_dispersal_genes()))
    pool = np.array([g for g in range(selection.n_genes) if g not in driver])
    total = n_patients * n_markers
    if total > len(pool):
        raise ValueError(f"need {total} germline markers but only {len(pool)} non-driver genes; "
                         "use a larger genome or fewer markers/patients")
    chosen = np.random.default_rng(seed).choice(pool, size=total, replace=False)
    chosen = chosen.reshape(n_patients, n_markers)
    return {i: chosen[i] for i in range(n_patients)}


def patient_overrides(n_patients, selection_params=None, subgroups=None, germline_markers=None):
    """Per-patient ``grow`` keywords for subgroups and private germline markers.

    ``subgroups`` is one :class:`Subgroup` per patient (from :func:`assign_subgroups`): a non-empty
    ``selection_delta`` becomes the patient's ``selection_params`` (merged onto ``selection_params``,
    the cohort's base), and the subgroup's ``germline_mutations`` followed by the patient's
    ``germline_markers`` (from :func:`draw_germline_markers`) become ``germline_mutations``, which
    :func:`grow_tumor` seeds into every cell before growth. Returns one dict per patient, for
    ``Cohort(patient_kwargs=...)``."""
    out = []
    for i in range(n_patients):
        kw = {}
        sub = subgroups[i] if subgroups is not None else _ALL
        if sub.selection_delta:
            kw["selection_params"] = {**(selection_params or {}), **sub.selection_delta}
        germline = list(sub.germline_mutations)
        if germline_markers is not None:
            germline = germline + list(germline_markers[i])
        if germline:
            kw["germline_mutations"] = germline
        out.append(kw)
    return out


def _apply_germline_mutations(tumor, gene_indices):
    """Seed inherited GERMLINE variants into EVERY cell of the patient BEFORE growth — the founder
    cancer genotype AND every non-cancer genotype present (epithelial, stromal, immune, and a
    metastatic deposit's host parenchyma) — since a germline
    variant is carried by every cell of an individual, not just the tumour. This makes ALL of a
    patient's cells (tumour and normal) share the same private germline background, which is exactly
    what genetic demultiplexing (souporcell/vireo) uses to assign cells to individuals.

    On the cancer founder the genome summary + fitness are refreshed (in case a variant lands on a
    driver); on the static normal genotypes only the genome bits are set (they carry no selection and
    ``get_snvs`` reads the genome directly). Safe because only these representatives exist pre-growth.
    """
    sel = tumor.selection
    offs = np.concatenate([[0], np.cumsum(sel.segment_sizes)]).astype(int)

    def set_bits(rep, update_summary):
        for g in gene_indices:
            g = int(g)
            seg = int(np.searchsorted(offs, g, side="right") - 1)
            pos = g - int(offs[seg])
            allele = rep.genome[seg]["p"][0]
            if allele[pos]:
                continue
            allele[pos] = True
            if update_summary:
                mut_bits = np.zeros(sel.segment_sizes[seg], dtype=bool)
                mut_bits[pos] = True
                rep.update_genome_summary_mutation(sel, mut_bits, seg)

    founder = tumor.genotypes[tumor.founder_id]
    set_bits(founder, update_summary=True)
    founder.update_evolutionary_parameters(sel)
    # Every non-cancer representative, tested by exclusion rather than by listing the normal
    # types, so no normal type (e.g. a metastatic deposit's host parenchyma) is left without the
    # patient's germline background.
    for gid, rep in tumor.genotypes.items():
        cell_type = getattr(rep, "type", None)
        if gid != tumor.founder_id and cell_type is not None and cell_type != "cancer":
            set_bits(rep, update_summary=False)
    # the founder's rates may have changed -> refresh every deme rate so event sampling stays correct
    tumor.deme_rates = np.array([tumor._deme_rate(i) for i in range(len(tumor.demes))], dtype=float)
