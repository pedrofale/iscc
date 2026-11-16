"""Epistasis / evolutionary-dependency structure in selection.

iscc's selection is otherwise **additive**: abstract mode scores the *count* of mutated drivers, so
the true event x event dependency network is ~empty and a cohort-progression benchmark (MHN, TreeMHN,
CBN/H-CBN, REVOLVER) could only measure a method's false-positive rate. This module lets a
**known network be planted** — the answer key those methods are scored against.

Three interaction types, all planted:

  * **pairwise epistasis** ``E[i, j]`` — a log-additive interaction term; ``> 0`` synergy /
    co-selection, ``< 0`` antagonism;
  * **conjunctive / ordered constraints** — a dependency DAG (``A -> B``: ``B`` only counts once
    ``A`` is present), which induces temporal ORDER along a lineage;
  * **mutual exclusivity / synthetic lethality** — strongly negative ``E[i, j]``, so the two events
    look mutually exclusive across a cohort (the DISCOVER/MEGSA signal).

Fitness gains one term on top of the existing additive model::

    log fitness += sum_i gate_i * beta_i * x_i + sum_{i<j} E[i, j] * x_i * x_j

``x`` is the genotype's **event vector**, so the term is a pure function of the event set: it caches
per genotype and is safe under tau-leaping and the exact engine alike. **Off by default** — with no
events configured, :class:`EpistasisNetwork` is never constructed and selection is the additive
model.


The event alphabet (abstract driver roles)
-----------------------------------------
An **event** is a disjoint **module of driver genes** (``event_size`` genes drawn from the shared
oncogene/TSG pool); it is *acquired* when >= 1 SNV lands anywhere in the module. Modules rather than
single genes because **MHN/TreeMHN pool many tumours to fit ONE network**, which requires the same
events to RECUR across patients: with ~10^4 genes and mutation spread genome-wide, any one named gene
is hit in almost no patient, so a single-gene alphabet yields a cohort with no shared events and
nothing to recover. A module is the familiar pathway-level event, and ``event_size=1`` recovers the
single-gene case. ``event_size`` is therefore the knob that sets how fast events accrue — see
PARAMETERS.md. It is one int for every module, or a list of one int per event so that events can be
set to different frequencies; either way every module's genes come from one draw of the layout
sub-stream, cut in event order, so a single size draws the same modules as a list of that size.

Events are **monotone**: once acquired an event never reverts, even if a later deletion removes the
mutated allele. That is deliberate — it is exactly the generative assumption MHN/CBN/TreeMHN make,
so the planted network stays the network those models are defined to recover.


Ordering: fitness gating vs accessibility gating
------------------------------------------------
The two are different generative stories, and the progression models assume different ones, so the
choice is explicit (``gating_mode``) and must be stated in the paper:

  * ``"fitness"`` (**default**) — ``B`` is *accessible but inert* until its DAG parents are present:
    the mutation arises freely and drifts as neutral standing variation, and only becomes beneficial
    once ``A`` is there. Order emerges from SELECTION. This is the softer, more biological story, and
    it is the one MHN/TreeMHN assume (rates are modulated, never zero).
  * ``"accessibility"`` — ``B`` *cannot arise at all* until its parents are present: mutations in
    ``B``'s module are vetoed. Order is imposed on the MUTATION process, and the DAG becomes a hard
    constraint on which genotypes exist. This is the CBN/H-CBN story (a "conjunctive Bayesian network"
    of strictly required predecessors).

Under fitness gating the observed order is noisy (a gated event can be present but unselected);
under accessibility gating it is exact. Benchmarks that score ORDER recovery should say which.


Planting an explicit network
----------------------------
By default the network is DRAWN at random from the layout stream (``n_interactions``,
``network_topology``, ``n_exclusive_pairs``, ``n_constraints``, ...). ``epistasis_params["planted"]``
instead states it exactly, so a benchmark can name every pair before any tumour is grown::

    epistasis_params = {"n_events": 6, "event_size": 10, "lethal_death_rate": 2.5,
                        "planted": {"effects": 0.05,                  # beta_i, scalar or per event
                                    "interactions": [(0, 1, 0.4)],    # E[i, j] (log-additive)
                                    "exclusive": [(2, 3)],            # lethal together
                                    "constraints": [(4, 5)]}}         # 4 must precede 5
    dependency_params = {"gating_mode": "accessibility"}

Only the event MODULES are still drawn (from the layout stream, exactly as in the random mode, so the
same ``event_size`` and ``layout_seed`` give the same modules whichever mode is used); everything else
is what was planted and nothing more — an event's own effect is 0 unless planted, a pair interacts
only if listed. Events are indexed from 0 (``event_names()`` calls them ``E0``, ``E1``, ...). An
exclusive pair is a synthetic-lethal combination (``lethal_death_rate``) with no ``E`` term of its own;
the order constraints are gated by ``dependency_params["gating_mode"]``. The random-draw knobs cannot
be combined with ``planted`` (they would be silently ignored), so giving both is an error.


Comparability across patients
-----------------------------
The network is part of the **shared landscape**, not per-run state: MHN/TreeMHN are only well-posed
if every patient in the cohort evolved under the SAME network. It is therefore drawn from the LAYOUT
stream (``layout_seed``), in a dedicated sub-stream ``default_rng(layout_seed +
LAYOUT_OFFSET_EPISTASIS)`` so that changing ``n_interactions`` / ``network_topology`` does not
reshuffle the oncogene/TSG layout or the gene programs. Same config + same ``layout_seed`` => the
identical network for every patient and across separate simulations; a different ``layout_seed`` =>
a different network (for replicate studies).
"""
import numpy as np

# Defaults for the two parameter blocks. Kept here (not buried in signatures) because they are the
# documented contract in PARAMETERS.md.
EPISTASIS_DEFAULTS = dict(
    n_events=0,                  # 0 => OFF (no network is built; selection stays additive)
    event_size=20,               # driver genes per event module (one int, or one per event);
                                 # sets how fast events accrue
    event_effect_mean=0.1,       # beta_i ~ N(mean, sd): the event's own marginal log-fitness effect
    event_effect_sd=0.0,
    n_interactions=None,         # number of non-zero E[i, j]; None => use network_sparsity
    network_sparsity=0.2,        # fraction of the n_events*(n_events-1)/2 pairs that are non-zero
    network_topology="random",   # "random" | "hub" | "chain"
    interaction_strength=0.3,    # |E[i, j]| ~ N(interaction_strength, interaction_strength_sd)
    interaction_strength_sd=0.05,
    prop_synergy=0.5,            # P(E[i, j] > 0); the rest are antagonistic (E < 0)
    mutual_exclusivity_strength=0.0,  # magnitude of the strongly-negative exclusive edges
    n_exclusive_pairs=0,         # how many such edges (drawn from pairs not already interacting)
    # Synthetic lethality must KILL, not merely stop division. A strongly negative E only suppresses
    # the division rate, and the crowding law (count.py _death_rate) uses slope = max(0, div - death):
    # a clone that has STOPPED dividing therefore takes NO density-dependent death at all, is never
    # purged, and stays "present" forever -- so exclusivity would be invisible in every observable,
    # which is the opposite of the DISCOVER/MEGSA signal it is meant to plant. With this on, a
    # genotype carrying BOTH events of a planted exclusive pair takes `lethal_death_rate` instead of
    # its baseline death, so the combination is actually removed from the population.
    mutual_exclusivity_lethal=True,
    lethal_death_rate=1.0,       # death rate of a synthetic-lethal genotype (>= max_birth_rate)
)

DEPENDENCY_DEFAULTS = dict(
    n_constraints=0,             # 0 => OFF (no DAG; no ordering)
    dag_depth=2,                 # number of layers the events are split into; edges go layer k -> k+1
    dag_branching=2,             # max parents per gated event
    gating_mode="fitness",       # "fitness" (B inert until A) | "accessibility" (B cannot arise)
)

_TOPOLOGIES = ("random", "hub", "chain")
_GATING_MODES = ("fitness", "accessibility")

# The planted network's keys, and the random-draw knobs it replaces (an error to give alongside it).
_PLANTED_KEYS = ("effects", "interactions", "exclusive", "constraints")
_RANDOM_EPISTASIS_KEYS = ("event_effect_mean", "event_effect_sd", "n_interactions", "network_sparsity",
                          "network_topology", "interaction_strength", "interaction_strength_sd",
                          "prop_synergy", "mutual_exclusivity_strength", "n_exclusive_pairs")
_RANDOM_DEPENDENCY_KEYS = ("n_constraints", "dag_depth", "dag_branching")


class EpistasisNetwork:
    """The planted event x event network: the ground truth a progression model must recover.

    Built from a dedicated LAYOUT sub-stream (see the module docstring), so it is shared by every
    patient of a cohort. Constructed only when ``n_events > 0``; the additive model is otherwise
    untouched.

    Parameters
    ----------
    driver_pool : array of flat gene indices
        The shared driver genes (oncogenes + TSGs) the event modules are carved out of.
    seg_offsets, segment_sizes : the genome layout, used to map flat gene indices back to
        (segment, position) so mutation can be scored per segment.
    rng : the dedicated epistasis LAYOUT sub-stream.
    """

    def __init__(self, driver_pool, seg_offsets, segment_sizes, rng,
                 epistasis_params=None, dependency_params=None):
        ep = {**EPISTASIS_DEFAULTS, **(epistasis_params or {})}
        dp = {**DEPENDENCY_DEFAULTS, **(dependency_params or {})}
        self.params = ep
        self.dependency_params = dp

        self.n_events = int(ep["n_events"])
        if self.n_events <= 0:
            raise ValueError("EpistasisNetwork requires n_events > 0 (it is only built when ON)")
        # module sizes: one int for every event (as before), or one int per event
        scalar_size = np.ndim(ep["event_size"]) == 0
        self.event_sizes = self._module_sizes(ep["event_size"], self.n_events)
        self.event_size = self.event_sizes[0] if scalar_size else list(self.event_sizes)
        if ep["network_topology"] not in _TOPOLOGIES:
            raise ValueError(f"network_topology must be one of {_TOPOLOGIES}, got {ep['network_topology']!r}")
        if dp["gating_mode"] not in _GATING_MODES:
            raise ValueError(f"gating_mode must be one of {_GATING_MODES}, got {dp['gating_mode']!r}")
        self.gating_mode = dp["gating_mode"]

        driver_pool = np.asarray(driver_pool, dtype=int)
        needed = int(sum(self.event_sizes))
        if needed > len(driver_pool):
            what = (f"{self.n_events} x {self.event_size}" if scalar_size
                    else " + ".join(str(s) for s in self.event_sizes))
            raise ValueError(
                f"epistasis needs {what} = {needed} driver genes for its "
                f"event modules but the shared layout only has {len(driver_pool)}; raise prop_driver / "
                "the genome size, or lower n_events / event_size")

        # -- the event alphabet: disjoint modules of driver genes ----------------------------
        # One draw of every module's genes, cut in event order: with a single size this is the same
        # draw, with the same stream consumption, as before per-event sizes existed.
        chosen = rng.choice(driver_pool, size=needed, replace=False)
        cuts = np.concatenate([[0], np.cumsum(self.event_sizes)]).astype(int)
        self.event_genes = [np.sort(chosen[cuts[i]:cuts[i + 1]]) for i in range(self.n_events)]

        # per-segment gene -> event id lookup (-1 = not an event gene), so scoring a mutation is a
        # single fancy-index on the segment's mutation mask.
        self._seg_offsets = np.asarray(seg_offsets, dtype=int)
        self.segment_sizes = [int(s) for s in segment_sizes]
        self.event_ids = [np.full(sz, -1, dtype=int) for sz in self.segment_sizes]
        for ev, genes in enumerate(self.event_genes):
            for g in genes:
                seg = int(np.searchsorted(self._seg_offsets, g, side="right") - 1)
                self.event_ids[seg][int(g) - int(self._seg_offsets[seg])] = ev

        self.lethal = bool(ep["mutual_exclusivity_lethal"])
        self.lethal_death_rate = float(ep["lethal_death_rate"])
        planted = ep.get("planted")
        self.planted = planted is not None
        if self.planted:
            # An explicit network: nothing below is drawn, so the layout stream is consumed by the
            # event modules alone (identical to the random mode's modules at the same layout seed).
            self._plant(planted, epistasis_params or {}, dependency_params or {})
        else:
            self._draw(ep, dp, rng)
        # bitmask of required parents per event, for O(1) gating
        self._parent_mask = np.zeros(self.n_events, dtype=object)
        for ev, ps in self.dag_parents.items():
            m = 0
            for p in ps:
                m |= (1 << p)
            self._parent_mask[ev] = m
        self._has_dag = bool(self.dag_edges)

        # per-event-set caches (pure functions of the event bitmask => tau-leap safe)
        self._fitness_cache = {}
        self._blocked_cache = {}

    @staticmethod
    def _module_sizes(size, n_events):
        """``event_size`` as one size per event: a single int (every module that size, as before)
        or a sequence of ``n_events`` positive ints (module ``i`` has ``size[i]`` genes)."""
        if np.ndim(size) == 0:
            return (int(size),) * n_events
        sizes = list(np.asarray(size, dtype=object).ravel())
        if len(sizes) != n_events:
            raise ValueError(f"event_size: give one int or one per event ({n_events}), "
                             f"got {len(sizes)} sizes")
        out = []
        for s in sizes:
            if isinstance(s, (bool, np.bool_)) or not isinstance(s, (int, np.integer)) or int(s) < 1:
                raise ValueError(f"event_size: module sizes are positive integers, got {s!r}")
            out.append(int(s))
        return tuple(out)

    def _draw(self, ep, dp, rng):
        """The random network (the default): beta, E, exclusive pairs and the DAG drawn, in that
        order, from the epistasis layout sub-stream after the event modules."""
        # -- marginal effects beta_i (MHN's diagonal) ----------------------------------------
        self.beta = rng.normal(ep["event_effect_mean"], ep["event_effect_sd"], size=self.n_events)

        # -- the interaction matrix E (symmetric, zero diagonal) -----------------------------
        self.E = np.zeros((self.n_events, self.n_events))
        pairs = [(i, j) for i in range(self.n_events) for j in range(i + 1, self.n_events)]
        n_inter = ep["n_interactions"]
        if n_inter is None:
            n_inter = int(round(float(ep["network_sparsity"]) * len(pairs)))
        n_inter = int(min(max(n_inter, 0), len(pairs)))
        inter_pairs = self._pick_pairs(pairs, n_inter, ep["network_topology"], rng)
        for (i, j) in inter_pairs:
            mag = abs(rng.normal(ep["interaction_strength"], ep["interaction_strength_sd"]))
            sign = 1.0 if rng.random() < float(ep["prop_synergy"]) else -1.0
            self.E[i, j] = self.E[j, i] = sign * mag

        # -- mutual exclusivity / synthetic lethality: strongly negative edges ----------------
        # Drawn from pairs NOT already carrying an interaction, so the two signals stay distinct in
        # the answer key (self.exclusive_pairs).
        self.exclusive_pairs = []
        n_excl = int(ep["n_exclusive_pairs"])
        mex = float(ep["mutual_exclusivity_strength"])
        if n_excl > 0 and mex > 0:
            free = [p for p in pairs if p not in set(inter_pairs)]
            if n_excl > len(free):
                raise ValueError(f"n_exclusive_pairs={n_excl} exceeds the {len(free)} event pairs left "
                                 "after the interaction edges; lower it or raise n_events")
            idx = rng.choice(len(free), size=n_excl, replace=False)
            for k in np.atleast_1d(idx):
                i, j = free[int(k)]
                self.E[i, j] = self.E[j, i] = -mex
                self.exclusive_pairs.append((i, j))

        # -- the conjunctive dependency DAG --------------------------------------------------
        self.dag_parents = {i: () for i in range(self.n_events)}
        self.dag_edges = []
        self._build_dag(dp, rng)

    def _plant(self, planted, epistasis_params, dependency_params):
        """The explicit network of ``epistasis_params["planted"]`` (see the module docstring),
        validated so that what is planted is exactly what was asked for."""
        if not isinstance(planted, dict):
            raise ValueError("epistasis_params['planted'] must be a dict with keys "
                             f"{_PLANTED_KEYS}, got {type(planted).__name__}")
        unknown = set(planted) - set(_PLANTED_KEYS)
        if unknown:
            raise ValueError(f"unknown planted key(s) {sorted(unknown)}; use {_PLANTED_KEYS}")
        drawn = ([k for k in _RANDOM_EPISTASIS_KEYS if k in epistasis_params]
                 + [k for k in _RANDOM_DEPENDENCY_KEYS if k in dependency_params])
        if drawn:
            raise ValueError(f"a planted network cannot be combined with the random-draw knob(s) "
                             f"{drawn}, which it would silently ignore; drop them")
        n = self.n_events

        def event(x, what):
            if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)):
                raise ValueError(f"planted {what}: events are integer indices, got {x!r}")
            if not 0 <= int(x) < n:
                raise ValueError(f"planted {what}: event {x} is out of range for n_events={n}")
            return int(x)

        def pair(entry, what, size):
            if len(entry) != size:
                raise ValueError(f"planted {what} entries are {size}-tuples, got {tuple(entry)!r}")
            i, j = event(entry[0], what), event(entry[1], what)
            if i == j:
                raise ValueError(f"planted {what}: an event cannot pair with itself ({i})")
            return i, j

        # -- the events' own effects (MHN's diagonal) ----------------------------------------
        effects = planted.get("effects", 0.0)
        if np.ndim(effects) == 0:
            self.beta = np.full(n, float(effects))
        else:
            self.beta = np.asarray(effects, dtype=float)
            if self.beta.shape != (n,):
                raise ValueError(f"planted effects: give a scalar or one value per event ({n}), "
                                 f"got {len(self.beta)}")
        if not np.all(np.isfinite(self.beta)):
            raise ValueError("planted effects must be finite")

        # -- pairwise log-additive terms -------------------------------------------------------
        self.E = np.zeros((n, n))
        seen = set()
        for entry in planted.get("interactions", ()):
            i, j = pair(entry, "interactions", 3)
            w = float(entry[2])
            if not np.isfinite(w):
                raise ValueError(f"planted interaction ({i}, {j}) has a non-finite term {w}")
            key = (min(i, j), max(i, j))
            if key in seen:
                raise ValueError(f"planted interactions: the pair {key} is given twice")
            seen.add(key)
            self.E[i, j] = self.E[j, i] = w

        # -- mutual exclusivity: lethal combinations ------------------------------------------
        self.exclusive_pairs = []
        for entry in planted.get("exclusive", ()):
            i, j = pair(entry, "exclusive", 2)
            key = (min(i, j), max(i, j))
            if key in seen:
                raise ValueError(f"planted exclusive pair {key} is also an interaction or given "
                                 "twice; keep the two signals distinct in the answer key")
            seen.add(key)
            self.exclusive_pairs.append(key)
        if self.exclusive_pairs and not self.lethal:
            raise ValueError("planted exclusive pairs act only through lethality; they would do "
                             "nothing with mutual_exclusivity_lethal=False")
        if self.exclusive_pairs and not self.lethal_death_rate > 0:
            raise ValueError(f"lethal_death_rate must be positive, got {self.lethal_death_rate}")

        # -- order constraints: a DAG, parent -> child ----------------------------------------
        parents = {i: [] for i in range(n)}
        self.dag_edges = []
        for entry in planted.get("constraints", ()):
            parent, child = pair(entry, "constraints", 2)
            if (parent, child) in self.dag_edges:
                raise ValueError(f"planted constraint {(parent, child)} is given twice")
            parents[child].append(parent)
            self.dag_edges.append((parent, child))
        self.dag_parents = {i: tuple(sorted(v)) for i, v in parents.items()}
        # acyclic: repeatedly remove events with no remaining parent (Kahn); any left form a cycle
        remaining = {i: set(v) for i, v in parents.items()}
        while True:
            free = [i for i, ps in remaining.items() if not ps]
            if not free:
                break
            for i in free:
                del remaining[i]
            for ps in remaining.values():
                ps.difference_update(free)
        if remaining:
            raise ValueError(f"planted constraints form a cycle among events {sorted(remaining)}")

    # ------------------------------------------------------------------ construction helpers
    @staticmethod
    def _pick_pairs(pairs, n, topology, rng):
        """Choose which event pairs interact, under the requested topology.

        Every topology returns exactly ``n`` distinct pairs: the topology decides which are
        *preferred*, and any shortfall is filled at random, so ``n_interactions`` means the same
        thing across topologies (making the sparsity sweep comparable).
        """
        if n == 0:
            return []
        if topology == "random":
            preferred = []
        elif topology == "hub":
            # a single hub event (0) wired to everything else — the "one master driver" story
            preferred = [p for p in pairs if p[0] == 0]
        elif topology == "chain":
            # a linear cascade 0-1-2-... — the "stepwise progression" story
            preferred = [p for p in pairs if p[1] == p[0] + 1]
        chosen = list(preferred[:n])
        if len(chosen) < n:
            rest = [p for p in pairs if p not in set(chosen)]
            extra = rng.choice(len(rest), size=n - len(chosen), replace=False)
            chosen += [rest[int(k)] for k in np.atleast_1d(extra)]
        return chosen

    def _build_dag(self, dp, rng):
        """A layered random DAG over the events: edges only ever run from an earlier layer to a
        later one, so it is acyclic by construction and ``dag_depth`` is a real depth."""
        n_constraints = int(dp["n_constraints"])
        if n_constraints <= 0:
            return
        depth = max(2, int(dp["dag_depth"]))
        branching = max(1, int(dp["dag_branching"]))
        order = rng.permutation(self.n_events)
        layers = np.array_split(order, min(depth, self.n_events))
        # candidate edges: any node in a later layer may depend on any node in an earlier one
        candidates = []
        for li in range(1, len(layers)):
            earlier = np.concatenate(layers[:li])
            for child in layers[li]:
                for parent in earlier:
                    candidates.append((int(parent), int(child)))
        if not candidates:
            return
        if n_constraints > len(candidates):
            raise ValueError(f"n_constraints={n_constraints} exceeds the {len(candidates)} edges a "
                             f"depth-{depth} DAG over {self.n_events} events admits; lower it or "
                             "raise dag_depth / n_events")
        idx = rng.choice(len(candidates), size=n_constraints, replace=False)
        parents = {i: [] for i in range(self.n_events)}
        for k in np.atleast_1d(idx):
            parent, child = candidates[int(k)]
            if len(parents[child]) >= branching:
                continue  # dag_branching caps how many parents one event may require
            parents[child].append(parent)
            self.dag_edges.append((parent, child))
        self.dag_parents = {i: tuple(sorted(v)) for i, v in parents.items()}

    # ------------------------------------------------------------------ the fitness term
    def log_fitness(self, event_bits):
        """The interaction contribution to log division fitness for a genotype's event set.

        A pure function of ``event_bits`` (an int bitmask), hence cached per genotype and identical
        under the exact and tau-leaping engines.
        """
        cached = self._fitness_cache.get(event_bits)
        if cached is not None:
            return cached
        present = [i for i in range(self.n_events) if event_bits >> i & 1]
        total = 0.0
        for i in present:
            # Under FITNESS gating a DAG child is inert until every required parent is present;
            # under ACCESSIBILITY gating it could not have arisen at all, so no gate is applied.
            if self._has_dag and self.gating_mode == "fitness":
                need = self._parent_mask[i]
                if need and (event_bits & need) != need:
                    continue
            total += self.beta[i]
        for a, i in enumerate(present):
            for j in present[a + 1:]:
                total += self.E[i, j]
        self._fitness_cache[event_bits] = total
        return total

    def multiplier(self, event_bits):
        """``exp(log_fitness)`` — the factor that multiplies the additive model's division rate.
        The event-free genotype maps to exactly 1.0, so the network never moves the baseline."""
        if not event_bits:
            return 1.0
        return float(np.exp(self.log_fitness(event_bits)))

    def is_lethal(self, event_bits):
        """True when the genotype carries BOTH events of a planted synthetic-lethal pair.

        Read by ``Selection.update_death_rate``. Without this, ``mutual_exclusivity_strength`` is a
        no-op for every observable: negative E suppresses division, and a non-dividing clone is
        exempt from crowding death (its slope is max(0, div - death) == 0), so it persists and the
        pair still reads as co-occurring. Exclusivity has to remove cells to be exclusivity.
        """
        if not self.lethal or not self.exclusive_pairs or not event_bits:
            return False
        for (i, j) in self.exclusive_pairs:
            if (event_bits >> i & 1) and (event_bits >> j & 1):
                return True
        return False

    # ------------------------------------------------------------------ accessibility gating
    def blocked_events(self, event_bits):
        """Events whose DAG parents are not yet all present — under ``accessibility`` gating these
        cannot be acquired at all (mutations in their modules are vetoed)."""
        if not self._has_dag or self.gating_mode != "accessibility":
            return frozenset()
        cached = self._blocked_cache.get(event_bits)
        if cached is None:
            blocked = []
            for i in range(self.n_events):
                need = self._parent_mask[i]
                if need and (event_bits & need) != need:
                    blocked.append(i)
            cached = frozenset(blocked)
            self._blocked_cache[event_bits] = cached
        return cached

    def blocked_mask(self, seg, event_bits):
        """Boolean mask over segment ``seg``'s positions marking sites that may NOT mutate given the
        genotype's current event set (accessibility gating). ``None`` when nothing is blocked, so the
        caller can skip the work entirely on the common path."""
        blocked = self.blocked_events(event_bits)
        if not blocked:
            return None
        ids = self.event_ids[seg]
        return np.isin(ids, np.fromiter(blocked, dtype=int, count=len(blocked)))

    # ------------------------------------------------------------------ scoring mutations
    def events_from_mutation(self, seg, mut_bits):
        """Bitmask of the events newly touched by a segment's mutation mask."""
        hit = self.event_ids[seg][mut_bits]
        bits = 0
        for ev in np.unique(hit):
            if ev >= 0:
                bits |= (1 << int(ev))
        return bits

    # ------------------------------------------------------------------ ground truth
    def true_interaction_matrix(self):
        """The planted ``E`` — the answer key for edge precision/recall."""
        return self.E.copy()

    def true_edges(self, threshold=0.0):
        """Planted interaction edges as ``(i, j, E_ij)``, ``i < j``, ``|E_ij| > threshold``."""
        return [(i, j, float(self.E[i, j]))
                for i in range(self.n_events) for j in range(i + 1, self.n_events)
                if abs(self.E[i, j]) > threshold]

    def true_dag_edges(self):
        """Planted conjunctive constraints as ``(parent, child)`` — the ordering answer key."""
        return list(self.dag_edges)

    def true_order_constraints(self):
        """The order the DAG requires: ``parent`` must precede ``child`` along any lineage."""
        return [(p, c) for (p, c) in self.dag_edges]

    def true_exclusive_pairs(self):
        """Pairs planted as mutually exclusive / synthetic-lethal."""
        return list(self.exclusive_pairs)

    def event_names(self, prefix="E"):
        return [f"{prefix}{i}" for i in range(self.n_events)]

    def ground_truth(self):
        """Everything a benchmark needs to score against, in one dict."""
        return dict(n_events=self.n_events,
                    event_names=self.event_names(),
                    event_genes=[g.tolist() for g in self.event_genes],
                    event_sizes=list(self.event_sizes),
                    beta=self.beta.copy(),
                    E=self.true_interaction_matrix(),
                    edges=self.true_edges(),
                    dag_edges=self.true_dag_edges(),
                    exclusive_pairs=self.true_exclusive_pairs(),
                    gating_mode=self.gating_mode,
                    planted=self.planted)


def bits_to_events(event_bits, n_events=None):
    """Bitmask -> sorted list of event indices."""
    n = n_events if n_events is not None else event_bits.bit_length()
    return [i for i in range(n) if event_bits >> i & 1]


def events_to_bits(events):
    """Iterable of event indices -> bitmask."""
    bits = 0
    for e in events:
        bits |= (1 << int(e))
    return bits
