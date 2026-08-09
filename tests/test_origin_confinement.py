"""Origin confinement + the breast-subtype instability presets.

Origin confinement models the fact that a breast lesion starts inside ONE intact acinus: while the
myoepithelial layer is continuous and the basement membrane unbroken, a daughter cell has essentially
nowhere to go. It is what lets an early copy-number alteration become TRUNCAL — an alteration that
fixes in the founding patch before any cell has left is carried by every cell that ever exists,
because every later cell descends from that patch.

The tests below check, in order: the feature is inert when absent; it holds the founding patch and
only the founding patch; it is released once and for good; the release condition really is gated on
the patch filling; a truncal alteration emerges from selection with nothing seeded; the in-situ
lesion nevertheless stays polyclonal; and the subtype presets resolve as documented.

CLONALITY here always means: over a group of demes, per segment, the largest fraction of that
group's CANCER cells sharing one NON-DIPLOID total copy number (the unaltered diploid state is
excluded). Reported as max over segments and mean over segments — they are very different numbers.
Everything is computed from per-deme genotype counts, never from a materialised subsample.
"""
import collections
import hashlib
import json
import warnings

import numpy as np
import pytest

from iscc.tumor.models.count import GenotypeTumor, BREAST_SUBTYPES


# --------------------------------------------------------------------------- helpers
GENOME = dict(n_segments=8, segment_size=120)
SELECTION = dict(prop_driver=0.04, prop_dispersal=0.0, prop_immune_resistance=0.02,
                 prop_met_survival=0.0, prop_treatment_resistance=0.0,
                 prop_breach=0.02, breach_effects=2.8,
                 prop_stromal_survival=0.02, stromal_survival_effects=2.2,
                 breach_cost=0.6, stromal_survival_cost=0.6, max_mut_drivers=12,
                 trait_source="mutation")
# A SCALED-DOWN version of the shipped ductal field (notebooks/example_config.yaml): same biology,
# same mechanism, but a small grid and a small acinus so the founding patch resolves in ~150
# generations instead of ~1200 and the whole module runs in seconds. The shipped configuration's own
# numbers are a calibration question, not something a unit test should re-derive.
CANCER = dict(division_rate=0.7, death_rate=0.05, max_birth_rate=2.0, mutation_rate=0.3,
              n_snvs_per_allele=0.1, dispersal_rate=0.9, cnv_prob=0.08, wgd_rate=0.001)
DEMES = dict(carrying_capacity=14, initial_cancer_cells=5, resident_pressure_ref=0.2,
             maximum_death_rate=2.2)
SPATIAL = dict(grid_size=20, n_glands=3, structure_radius=4, gland_radius=4, min_gland_sep=9,
               K_duct=14, K_stroma=14, stroma_fill_frac=0.3, cross_gland_kappa=0.06,
               cross_gland_lambda=30, breach_gated_invasion=True, epithelial_barrier=0.0,
               stromal_hazard=0.6)
CONFINE = dict(dispersal_factor=0.0, established_at=0.5, generations=150)


def make(seed=2, confinement=None, spatial=None, cancer=None, selection=None, drop=()):
    """A tumour with NOTHING seeded into the founder — no mutations, a diploid karyotype. Whatever
    trunk these tests find was built by the simulation, which is the whole point."""
    spatial = {**SPATIAL, **(spatial or {})}
    if confinement is not None:
        spatial["origin_confinement"] = confinement
    cp = {k: v for k, v in {**CANCER, **(cancer or {})}.items() if k not in drop}
    return GenotypeTumor(
        seed=seed, genome_params=GENOME, selection_params={**SELECTION, **(selection or {})},
        cancer_cell_params=cp, deme_params=DEMES, spatial_params=spatial,
        update_mode="tau", tau=0.5, snapshot_every=10 ** 9, coarsen_passengers=True)


def advance(T, n, seed=2):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        T._tau_generation(rng, T.tau)
    return T


def state_hash(T):
    """Whole-tumour fingerprint: every deme's genotype counts, keyed by each clone's creation
    ordinal WITHIN this tumour (``genotype_id`` is a process-global counter, so it differs between
    two tumours built in one process and cannot be used here)."""
    parts = [(i, sorted((int(T.genotypes[g].ord), int(c)) for g, c in d.items()))
             for i, d in enumerate(T.demes) if d]
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


def cn_profiles(T):
    """gid -> per-segment TOTAL copy number, cancer genotypes only."""
    return {g: tuple(p + m for p, m in T._hap_cns(g))
            for g in T.genotypes_counts if T._is_cancer(g)}


def cancer_counts(T, demes):
    c = collections.Counter()
    for di in demes:
        for gid, n in T.demes[di].items():
            if T._is_cancer(gid):
                c[gid] += n
    return c


def clonality(counts, prof):
    """(max over segments, mean over segments, #segments >= 0.9, cells) — non-diploid states only."""
    tot = sum(counts.values())
    if not tot:
        return float("nan"), float("nan"), 0, 0
    n_seg = len(next(iter(prof.values())))
    per_seg = []
    for s in range(n_seg):
        hist = collections.Counter()
        for gid, n in counts.items():
            if prof[gid][s] != 2:
                hist[prof[gid][s]] += n
        per_seg.append(max(hist.values()) / tot if hist else 0.0)
    per_seg = np.asarray(per_seg)
    return float(per_seg.max()), float(per_seg.mean()), int((per_seg >= 0.9).sum()), tot


def profile_mix(counts, prof):
    """(largest copy-number-profile fraction, #distinct copy-number profiles) — the polyclonality
    read-out. Distinct total-copy-number profiles are what single-cell copy-number studies call
    clones, so a group whose largest profile approaches 1.0 is the monoclonal failure signature."""
    tot = sum(counts.values())
    by = collections.Counter()
    for gid, n in counts.items():
        by[prof[gid]] += n
    fr = np.array(sorted((v / tot for v in by.values()), reverse=True))
    return float(fr[0]), int(len(fr))


def occupied(T):
    return sum(1 for d in T.demes[:T.n_primary_demes] if any(T._is_cancer(g) for g in d))


# --------------------------------------------------------------------------- inert when absent
def test_absent_leaves_the_engine_untouched():
    T = make(confinement=None)
    assert T._origin_confined is False
    assert len(T._origin_demes) == 1
    rep = T.genotypes[T.founder_id]
    for di in (T._origin_demes[0], 0, 5):
        assert T._dispersal_of(rep, di) == rep.evolutionary_parameters["dispersal_rate"]
    T._origin_tick(1000.0)          # must not raise and must not latch anything
    assert T._origin_confined is False


def test_neutral_factor_is_byte_identical_to_absent():
    """The code path itself must not perturb the rng stream or any rate: a confinement whose
    multiplier is exactly 1.0 has to reproduce the absent case cell for cell."""
    off = advance(make(confinement=None), 12)
    neutral = advance(make(confinement=dict(CONFINE, dispersal_factor=1.0)), 12)
    assert state_hash(off) == state_hash(neutral)
    assert off.get_cancer_size() == neutral.get_cancer_size()


# --------------------------------------------------------------------------- confinement itself
def test_confinement_holds_the_founding_patch():
    """Confined, the lesion is still one deme after 40 generations; unconfined it is not."""
    assert occupied(advance(make(confinement=CONFINE), 40)) == 1
    assert occupied(advance(make(confinement=None), 40)) > 1


def test_confinement_is_origin_specific():
    """Only the founding patch is confined. Suppressing dispersal across the whole duct would let
    one clone homogenise it and give monoclonal in-situ disease, which contradicts the data."""
    T = advance(make(confinement=CONFINE), 10)
    rep = T.genotypes[T.founder_id]
    origin = T._origin_demes[0]
    base = rep.evolutionary_parameters["dispersal_rate"]
    assert T._dispersal_of(rep, origin) == 0.0          # dispersal_factor 0 == a closed acinus
    duct = [d for d in range(T.n_primary_demes)
            if T.gland_id[d] == T.gland_id[origin] and d != origin]
    assert duct, "the founding duct should span several demes"
    for di in duct:
        assert T._dispersal_of(rep, di) == base


def test_release_is_transient_and_permanent():
    T = advance(make(confinement=dict(CONFINE, generations=30)), 60)
    assert T._origin_confined is False
    rep = T.genotypes[T.founder_id]
    assert T._dispersal_of(rep, T._origin_demes[0]) == rep.evolutionary_parameters["dispersal_rate"]
    assert any(e.get("event") == "origin_release" for e in T.events)
    advance(T, 5, seed=3)                       # never re-confines
    assert T._origin_confined is False


def test_release_clock_waits_for_the_patch_to_fill():
    """`established_at` gates the countdown: an acinus that never fills never starts distending, so
    confinement holds. (An unreachable fill fraction is the cleanest way to assert that.)"""
    T = advance(make(confinement=dict(CONFINE, established_at=5.0, generations=5)), 40)
    assert T._origin_confined is True
    assert T._origin_age is None


def test_the_exact_engine_confines_and_releases_too():
    """The exact (one-event-per-call) engine tracks no time, so a generation there is defined as one
    event per cancer cell. Confinement has to work on that path as well as on the tau path."""
    T = make(seed=3, confinement=dict(CONFINE, generations=5))
    T.update_mode = "exact"
    rng = np.random.default_rng(3)
    for _ in range(4000):
        T.update(rng)
    assert T._origin_confined is False and T._origin_age >= 5
    assert any(e.get("event") == "origin_release" for e in T.events)
    assert T.get_cancer_size() > 0


class _BinomialSpy:
    """An rng proxy that records the success probability of every Binomial draw."""

    def __init__(self, rng):
        self._rng = rng
        self.p = []

    def binomial(self, n, p, *a, **kw):
        self.p.append(float(p))
        return self._rng.binomial(n, p, *a, **kw)

    def __getattr__(self, name):
        return getattr(self._rng, name)


#: A field with every OTHER binomial-drawing route switched off (no cross-gland hop, no metastasis,
#: no basement-membrane gate), so the only Binomial a substep draws is the division's mutate-or-
#: disperse split. That makes the split probability directly observable.
BARE = dict(SPATIAL, cross_gland_kappa=0.0, breach_gated_invasion=False, met_grid_size=0)


def _split_probabilities(confinement, gens=25, seed=2):
    T = make(confinement=confinement, spatial=BARE)
    spy = _BinomialSpy(np.random.default_rng(seed))
    for _ in range(gens):
        T._tau_generation(spy, T.tau)
    return T, spy.p


def test_confinement_does_not_change_the_mutation_probability():
    """A cell's mutation rate must not depend on whether it is allowed to MOVE.

    The division fate is drawn as mutate-or-disperse with probability
    ``mutation_rate / (mutation_rate + dispersal_rate)``. Scaling the dispersal rate down for
    confinement inside that denominator would drive the probability to 1.0 in a closed acinus —
    here a silent 4x (0.3/(0.3+0.9) = 0.25 -> 1.0) — which is an artefact of how the fate is drawn,
    not biology. The split must read the clone's OWN dispersal rate; confinement acts on where the
    dispersing daughter ends up.
    """
    expected = CANCER["mutation_rate"] / (CANCER["mutation_rate"] + CANCER["dispersal_rate"])
    assert expected == pytest.approx(0.25)
    for tag, conf in (("closed acinus", CONFINE), ("no confinement", None)):
        T, ps = _split_probabilities(conf)
        assert ps, f"{tag}: no division fate was ever drawn"
        assert set(np.round(ps, 12)) == {round(expected, 12)}, (
            f"{tag}: mutate-or-disperse split drawn at {sorted(set(ps))}, expected {expected}")


def test_a_daughter_that_cannot_leave_still_takes_a_slot_at_home():
    """Confinement redirects the dispersal branch, it does not delete it. With the acinus closed the
    lesion still fills its patch — the would-be dispersers compete for a slot where they are."""
    T = advance(make(confinement=CONFINE), 40)
    origin = T._origin_demes[0]
    n_cancer = sum(c for g, c in T.demes[origin].items() if T._is_cancer(g))
    assert occupied(T) == 1
    assert n_cancer >= 0.5 * T._cap_of(origin), (
        f"the closed acinus holds only {n_cancer} cancer cells of K={T._cap_of(origin)}")


def test_release_generations_is_counted_from_the_fill():
    T = make(confinement=dict(CONFINE, generations=25))
    advance(T, 20)
    assert T._origin_confined is True and T._origin_age is not None
    advance(T, 10, seed=3)
    assert T._origin_confined is False


# --------------------------------------------------------------------------- the point of it all
#: One fixed seed for the growth fixture. The mechanism is stochastic — the founding acinus resolves
#: to a single dominant altered state in most runs but not all — so the module pins a seed rather than
#: asserting a rate. The distribution across seeds is a calibration measurement, not a unit test.
SEED = 17


@pytest.fixture(scope="module")
def grown():
    """One confined tumour and one unconfined control, same seed, grown to the same size."""
    out = {}
    for tag, conf in (("confined", CONFINE), ("control", None)):
        T = make(seed=SEED, confinement=conf)
        rng = np.random.default_rng(SEED)
        for _ in range(900):
            T._tau_generation(rng, T.tau)
            if T.get_cancer_size() >= 4000:
                break
        out[tag] = T
    return out


def test_a_truncal_alteration_emerges_with_nothing_seeded(grown):
    """A copy-number state carried by ~every cancer cell, produced by selection inside the founding
    acinus — no alteration is seeded into the founder (founder_mutations seeds SNVs only)."""
    T = grown["confined"]
    assert T._hap_cns(T.founder_id) == [(1, 1)] * GENOME["n_segments"], "founder must start diploid"
    prof = cn_profiles(T)
    mx, mean, n90, cells = clonality(cancer_counts(T, range(T.n_primary_demes)), prof)
    assert cells >= 4000
    assert n90 >= 1, f"no truncal alteration: max-over-segments {mx:.3f}, mean {mean:.3f}"
    assert mx >= 0.9
    assert mean < mx, "max and mean over segments are different numbers; do not conflate them"


def test_the_control_has_no_truncal_alteration(grown):
    """Without confinement the same seed produces no clonal copy-number layer at all — that is the
    problem origin confinement exists to solve."""
    T = grown["control"]
    prof = cn_profiles(T)
    mx, _, n90, _ = clonality(cancer_counts(T, range(T.n_primary_demes)), prof)
    assert n90 == 0 and mx < 0.9


def test_in_situ_disease_stays_polyclonal(grown):
    """Confinement must not leak beyond the founding patch. A duct that has gone monoclonal is the
    failure signature: single-cell studies of in-situ breast disease find most lesions polyclonal,
    with the same clones on both sides of the basement membrane."""
    T = grown["confined"]
    prof = cn_profiles(T)
    ducts = [g for g in range(T.n_glands)
             if sum(cancer_counts(T, [d for d in range(T.n_primary_demes)
                                      if T.gland_id[d] == g]).values()) >= 150]
    assert ducts, "no colonised duct to judge"
    for g in ducts:
        demes = [d for d in range(T.n_primary_demes) if T.gland_id[d] == g]
        top, n_profiles = profile_mix(cancer_counts(T, demes), prof)
        assert n_profiles >= 2 and top < 0.9, (
            f"duct {g} is monoclonal (largest copy-number profile {top:.2f} of "
            f"{n_profiles} profiles) — confinement has leaked beyond the founding patch")


def test_invasion_stays_polyclonal(grown):
    T = grown["confined"]
    prof = cn_profiles(T)
    stroma = [d for d in range(T.n_primary_demes) if T.gland_id[d] < 0]
    counts = cancer_counts(T, stroma)
    if sum(counts.values()) < 150:
        pytest.skip("this seed has not invaded far enough to judge stromal clonality")
    top, n_profiles = profile_mix(counts, prof)
    assert n_profiles >= 2 and top < 0.9


def test_aneuploidy_burden_stays_realistic(grown):
    """The truncal layer must not come at the price of an absurd genome."""
    T = grown["confined"]
    prof = cn_profiles(T)
    counts = cancer_counts(T, range(T.n_primary_demes))
    tot = sum(counts.values())
    fga = sum(n * float((np.asarray(prof[g]) != 2).mean()) for g, n in counts.items()) / tot
    ploidy = sum(n * float(np.asarray(prof[g]).mean()) for g, n in counts.items()) / tot
    assert 0.1 <= fga <= 0.75, f"fraction of genome altered {fga:.3f}"
    assert 1.5 <= ploidy <= 3.5, f"mean ploidy {ploidy:.2f}"


# --------------------------------------------------------------------------- subtype presets
def test_subtype_presets_are_documented_defaults():
    assert set(BREAST_SUBTYPES) == {"ER+", "TNBC"}
    er, tn = BREAST_SUBTYPES["ER+"], BREAST_SUBTYPES["TNBC"]
    # the three knobs the subtypes differ in, and the direction they differ in
    assert set(er) == set(tn) == {"cnv_prob", "wgd_rate", "mutation_rate"}
    for k in er:
        assert tn[k] > er[k], f"triple-negative should be the more unstable subtype in {k}"


#: the three knobs a subtype preset supplies; dropped so the preset is what fills them in
KNOBS = ("cnv_prob", "wgd_rate", "mutation_rate")


def _plain(**kw):
    """A tumour whose cancer params carry NO explicit instability knobs, so a preset supplies them."""
    return make(cancer=kw, drop=tuple(k for k in KNOBS if k not in kw))


def test_subtype_sets_the_instability_knobs():
    T = _plain(subtype="ER+")
    rep = T.genotypes[T.founder_id]
    assert T.subtype == "ER+"
    assert (rep.cnv_prob, rep.wgd_rate, rep.mutation_rate) == (
        BREAST_SUBTYPES["ER+"]["cnv_prob"], BREAST_SUBTYPES["ER+"]["wgd_rate"],
        BREAST_SUBTYPES["ER+"]["mutation_rate"])
    T2 = _plain(subtype="TNBC")
    r2 = T2.genotypes[T2.founder_id]
    assert r2.cnv_prob > rep.cnv_prob and r2.wgd_rate > rep.wgd_rate


def test_explicit_values_beat_the_preset():
    T = _plain(subtype="TNBC", cnv_prob=0.11)
    assert T.genotypes[T.founder_id].cnv_prob == 0.11
    assert T.genotypes[T.founder_id].wgd_rate == BREAST_SUBTYPES["TNBC"]["wgd_rate"]


def test_unknown_subtype_is_rejected():
    with pytest.raises(ValueError, match="unknown cancer subtype"):
        _plain(subtype="luminal-B")


def test_no_subtype_key_changes_nothing():
    a, b = make(), make(cancer={})
    for attr in ("cnv_prob", "wgd_rate", "mutation_rate"):
        assert getattr(a.genotypes[a.founder_id], attr) == getattr(b.genotypes[b.founder_id], attr)
    assert a.subtype is None


# ------------------------------------------------- the trunk is MEASURED, not looked up
def test_truncal_sites_measures_the_trunk_and_seeded_record_is_empty(grown):
    """With no `founder_mutations`, `seeded_truncal_sites` is empty — nothing was planted — while
    `truncal_sites()` still finds the trunk the confined acinus built. Reading the empty attribute
    as if it were the trunk is what silently produced nan purity estimates downstream."""
    T = grown["confined"]
    T.make_cell_data()
    assert len(T.seeded_truncal_sites) == 0

    trunk = T.truncal_sites(ccf=0.95)
    snv = T.cell_data["cell_snv"].values
    is_cancer = np.array([T._is_cancer(g) for g in T.cell_data["cell_type"].iloc[:, 0]], bool)
    assert is_cancer.any()
    carried = (snv[is_cancer] > 0).mean(axis=0)
    assert np.all(carried[trunk] >= 0.95)          # everything returned really is at the cutoff
    germline = np.zeros(snv.shape[1], bool)
    if np.size(getattr(T, "germline_sites", [])):
        germline[np.asarray(T.germline_sites, int)] = True
    assert set(np.flatnonzero((carried >= 0.95) & ~germline)) == set(trunk)   # and nothing missed


def test_truncal_sites_excludes_germline_and_tightens_with_the_cutoff(grown):
    """Germline variants sit in EVERY cell, so a naive frequency test would call them truncal. And
    the count is a function of the cutoff, not a constant of the tumour."""
    T = grown["confined"]
    T.make_cell_data()
    trunk = T.truncal_sites(ccf=0.95)
    if np.size(getattr(T, "germline_sites", [])):
        assert not set(trunk.tolist()) & set(np.asarray(T.germline_sites, int).tolist())
    assert len(T.truncal_sites(ccf=0.99)) <= len(trunk) <= len(T.truncal_sites(ccf=0.5))


def test_max_cells_warns_only_when_it_caps_an_explicit_section(grown):
    """The cap is a MEMORY budget applied after region/depth_frac. When it binds on a section it,
    not the physical request, sets the density — silently, until now. A plain whole-tumour
    materialisation being capped is the documented representative-biopsy behaviour, not a warning."""
    T = grown["confined"]
    small = max(1, T.get_tumor_size() // 10)

    with pytest.warns(UserWarning, match="setting the cell density"):
        T.make_cell_data(depth_frac=0.5, max_cells=small)
    with warnings.catch_warnings():          # generous cap -> depth_frac governs -> silence
        warnings.simplefilter("error")
        T.make_cell_data(depth_frac=0.5, max_cells=10 ** 9)
    with warnings.catch_warnings():          # whole-tumour subsample -> expected, not a warning
        warnings.simplefilter("error")
        T.make_cell_data(max_cells=small)
