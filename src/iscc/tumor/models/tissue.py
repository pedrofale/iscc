"""The tissue: the ductal field, slots and eviction, compartment composition, origin
confinement and the dispersal routes (with the basement-membrane gate).

A mixin of [`GenotypeTumor`][iscc.tumor.GenotypeTumor]; it holds no state of its own."""

import numpy as np
from ..geometry import bresenham_circumference, get_inside


class TissueMixin:
    #: Resident cell types a cancer birth may DISPLACE to take a slot. Invasive carcinoma replaces
    #: normal tissue — a duct effaced by DCIS is the canonical picture — so epithelium, stroma and
    #: metastatic host parenchyma are evictable. Without eviction a duct wall seeded full of immortal
    #: epithelium would have no slot for cancer, ever. Immune cells are not evictable: letting a
    #: tumour structurally clear its own immune pressure would trivialise immune escape.
    _EVICTABLE = ("epithelial", "stromal", "host")
    def _init_compartment_hazards(self, p):
        """The compartment hazards and the basement-membrane gate."""
        # Compartment-dependent selection. Two local hazards,
        # each contributed by a resident normal compartment the gland geometry seeds and each
        # attenuated by a MATCHING heritable trait (breach / stromal_survival), exactly the shape of
        # the immune term above. This is the whole "payoff table": edit these two coefficients (+ the
        # two prop_/effects axes in selection_params) in config, not the engine. DEFAULT 0.0 -> the
        # terms vanish.
        self._epithelial_barrier = (p.spatial or {}).get("epithelial_barrier", 0.0)
        self._stromal_hazard = (p.spatial or {}).get("stromal_hazard", 0.0)
        # Breach-GATED invasion: with this on, crossing the basement
        # membrane — a dispersal hop from a DUCT deme into a STROMA deme — succeeds only in proportion to
        # the clone's heritable `breach` trait; a cell that fails stays confined in the duct (DCIS). This
        # makes breach a true GATE (only breach clones enter the stroma) rather than the epithelial_barrier
        # death-hazard, which merely disadvantages non-breach cells in the duct and so leaks. DEFAULT
        # False -> the duct->stroma hop is an ordinary neighbour dispersal.
        self._breach_gated_invasion = bool((p.spatial or {}).get("breach_gated_invasion", False))


    def _init_crowding(self, p):
        """Carrying capacity, crowding and the division rule's checks."""
        # grid of demes; each deme is a dict {genotype_id: count}
        self.grid_size = p.spatial["grid_size"]
        # carrying_capacity is the per-deme cap K. Setting it to None or 0 disables crowding
        # entirely -> the "well-mixed" regime (unbounded growth in a deme), used by the single-deme
        # SISTEM benchmark.
        self.carrying_capacity = p.deme.get("carrying_capacity", 10)
        self._crowding = self.carrying_capacity is not None and self.carrying_capacity > 0
        # Positive fill/normalisation capacity: used for structure filling, immune seeding and the
        # microenvironment density fields. Falls back to 1 when crowding is off (no capacity).
        self._cap = int(self.carrying_capacity) if self._crowding else 1
        # Firmness of the cap: the crowding slope is steepened by (1 + crowding_margin) so the
        # per-deme fixed point sits at K/(1+margin) (slightly below K) and the restoring force above
        # K is firm rather than marginal. 0 recovers the plain fixed-point-at-K form. See _death_rate.
        self.crowding_margin = p.deme.get("crowding_margin", 0.1)
        # maximum_death_rate MUST be >= max_birth_rate for the cap to bind: a clone whose division
        # evolved up to max_birth_rate needs its crowding death to be able to reach (and exceed) that
        # rate. Default 1.0 (>= the 0.8 max_birth_rate default); a lower value lets a deme overfill.
        self.maximum_death_rate = p.deme.get("maximum_death_rate", 1.0)
        # Resident-pressure reference division rate (invasion gate). A deme's
        # immortal normal cells (epithelial/stromal) contribute to a cancer cell's crowding death at
        # this FIXED reference rate rather than at the cancer cell's own evolved division rate — so
        # their contribution does NOT cancel in the survival condition (net = div - death > 0) and
        # becomes a genuine FITNESS THRESHOLD a cancer clone must clear to establish (and then invade)
        # a normal-occupied gland deme. Defaults to the founder cancer division rate: a clone must be
        # at least as fit as a baseline cancer cell to hold a slot against a resident. The term is
        # zero in cancer-only demes (no normal cells). Raise it for a stricter invasion gate.
        self._resident_ref = p.deme.get(
            "resident_pressure_ref",
            self.genotypes[self.founder_id].baseline_rates["division_rate"])
        # Crowding (Noble et al. 2022, Nat Ecol Evol, "Within-deme dynamics", with a structural cap).
        # The density cap is STRUCTURAL: a deme has floor(crowding_overfill * K) slots, a birth that
        # cannot get one fails, and when more births are drawn than there are free slots the slots go
        # to clones in proportion to the births they drew (`_tau_substep`). The death rate is demon's
        # two-valued step:
        #   "When the deme population size is less than or equal to the carrying capacity, the death
        #    rate takes a fixed value d0 that is less than the initial division rate. When the deme
        #    population size exceeds carrying capacity, the death rate takes a different fixed value
        #    d1 that is much greater than the largest attainable division rate."
        # d0 is the clone's own baseline death rate; d1 defaults to maximum_death_rate. The step
        # carries no `div`, so it cannot cancel out of net = div - death and clones keep competing
        # inside full demes. The slot cap has to exceed K (overfill > 1) for a deme to meet the step.
        for key, law in (("crowding_mode", "lottery"), ("crowding_law", "noble"),
                         ("dispersal_law", "noble")):
            if p.deme.get(key, law) != law:
                raise ValueError(f"{key}={p.deme[key]!r} was removed; the engine runs "
                                 f"{key}={law!r} only (drop the key)")
        self._crowding_d1 = p.deme.get("crowding_d1")          # None -> maximum_death_rate
        self._crowding_overfill = float(p.deme.get("crowding_overfill", 1.5))
        if self._crowding_overfill < 1.0:
            raise ValueError("crowding_overfill must be >= 1 (1 = the slot cap sits exactly on K)")
        if self._crowding:
            # d1 MUST exceed the largest attainable division rate, which is what makes the cap
            # unbreakable however far selection pushes birth rates up. The EFFECTIVE d1 is what
            # survives the `min(death, maximum_death_rate)` clamp at the end of _death_rate, so a
            # large d1 clipped back below the evolved division rate is caught too.
            _d1 = self.maximum_death_rate if self._crowding_d1 is None else float(self._crowding_d1)
            _eff = min(_d1, self.maximum_death_rate)
            _max_b = float(getattr(self.genotypes[self.founder_id], "max_birth_rate", 0.0))
            if _eff <= _max_b:
                raise ValueError(
                    f"crowding needs d1 > the largest attainable division rate, but the "
                    f"EFFECTIVE d1 is {_eff:g} <= max_birth_rate={_max_b:g} "
                    f"(requested d1={_d1:g}, clamped by maximum_death_rate={self.maximum_death_rate:g}). "
                    f"Raise BOTH crowding_d1 and maximum_death_rate above max_birth_rate — a density "
                    f"cap expressed as a rate below the evolved division rate cannot bound a deme.")
        # A synthetic-lethal genotype (both events of an exclusive pair) takes `lethal_death_rate` as
        # its death rate, and every death rate is clamped to maximum_death_rate (the exact engine's
        # event bound assumes it), so a lethal rate above the clamp would silently be lowered to it.
        _net = self.selection.epistasis
        if (_net is not None and _net.lethal and _net.exclusive_pairs
                and _net.lethal_death_rate > self.maximum_death_rate):
            raise ValueError(
                f"lethal_death_rate={_net.lethal_death_rate:g} exceeds maximum_death_rate="
                f"{self.maximum_death_rate:g}, which clamps every death rate; raise "
                f"deme_params['maximum_death_rate'] to at least the lethal rate")
        # Division (demon's invasion mode: `cell_division` and the migration_type-0 branch of its main
        # loop): every division attempts a mutation with probability `mutation_rate`, and
        # independently one of the two daughters, picked at random, disperses with probability
        # `dispersal_rate`. Both are per-division probabilities, so a baseline above 1 is rejected
        # rather than silently clipped.
        _base = self.genotypes[self.founder_id].baseline_rates
        _mu = float(self.genotypes[self.founder_id].mutation_rate)
        if not 0.0 <= float(_base["dispersal_rate"]) <= 1.0:
            raise ValueError(
                f"dispersal_rate is a per-division probability, but it is "
                f"{_base['dispersal_rate']:g}; give a value in [0, 1]")
        if not 0.0 <= _mu <= 1.0:
            raise ValueError(
                f"mutation_rate is a per-division probability, but it is {_mu:g}; give a value in [0, 1]")

    def _init_field(self, p):
        """The ductal field, origin confinement and the founding cluster."""
        self.structure_radius = p.spatial.get("structure_radius", 0)
        # Ductal-field substrate: the structured case is a FIELD of many small epithelial-ring glands
        # at 2D positions in moderate-density stroma (an island model). The defaults (n_glands=1,
        # gland_radius=structure_radius, stroma_fill_frac=1.0, K_duct=K_stroma=carrying_capacity) give
        # a single central ring.
        self.n_glands = int(p.spatial.get("n_glands", 1))
        self.gland_radius = int(p.spatial.get("gland_radius", self.structure_radius))
        self.min_gland_sep = p.spatial.get("min_gland_sep", 2 * self.gland_radius + 2)
        self.stroma_fill_frac = float(p.spatial.get("stroma_fill_frac", 1.0))
        # Per-compartment carrying capacity (K captures the duct's/stroma's 3D depth — a deme is a 3D
        # column, so K is MODERATE-TO-LARGE, not a handful). Default to the scalar carrying_capacity.
        self._cap_duct = int(p.spatial.get("K_duct", self._cap)) if self._crowding else 1
        self._cap_stroma = int(p.spatial.get("K_stroma", self._cap)) if self._crowding else 1
        # Cross-gland (island) dispersal: a low rate that seeds one gland's lumen from another's,
        # abstracting intraductal spread through the out-of-plane ductal tree. kappa=0 -> off.
        # lambda (distance kernel over gland centres); None -> uniform targeting.
        self.cross_gland_kappa = float(p.spatial.get("cross_gland_kappa", 0.0))
        self.cross_gland_lambda = p.spatial.get("cross_gland_lambda", None)
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
        # Config (``spatial_params.origin_confinement``), a dict; absent -> None -> no confinement:
        #   dispersal_factor : dispersal rate inside the founding patch, as a MULTIPLE of the
        #                      configured ``dispersal_rate`` (0 => a closed acinus, nothing gets out).
        #                      A multiplier, not an absolute rate, so an evolved dispersal trait still
        #                      shows through. It acts on the DAUGHTER'S DESTINATION, not on the cell's
        #                      rates: a would-be disperser that cannot leave competes for a slot where
        #                      it is (``_escape_factor``). Deliberately so — confinement must not
        #                      change how often a division mutates. Scaling the dispersal rate inside
        #                      the legacy mutate-or-disperse denominator instead would send the
        #                      mutation probability to 1.0 in a closed acinus, an artefact of how the
        #                      fate is drawn, not biology.
        #   established_at   : the countdown does not start until the founding patch is at least this
        #                      fraction of its carrying capacity — an acinus with a handful of cells in
        #                      it is not distending anything yet. Fractions of K, so it means the same
        #                      thing at every grid size.
        #   generations      : generations of distension after that before the architecture gives way
        #                      and the patch reverts to the ordinary dispersal rate. One-way: once
        #                      released, never re-confined.
        _oc = p.spatial.get("origin_confinement", None)
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
            if _oc.get("clock", "generations") != "generations":
                raise ValueError("origin_confinement counts generations; the 'steps' clock was removed")
        # Number of founder cancer cells to seed (an established micro-lesion). A single founder
        # has P(extinction) ≈ death/division (~7% for the defaults) regardless of carrying
        # capacity, so a one-cell start makes runs/demos randomly cancer-free; seeding a small
        # cluster removes that founder bottleneck. Default 1.
        self.initial_cancer_cells = p.deme.get("initial_cancer_cells", 1)
        # Founder cells actually seeded: the requested cluster, capped by K when crowding is on
        # (can't over-seed a deme past its capacity) and left uncapped in the well-mixed regime.
        self._n_founder = max(1, min(self.initial_cancer_cells, self._cap)
                              if self._crowding else self.initial_cancer_cells)


    def _place_glands(self):
        """Gland centres in the grid interior, >= min_gland_sep apart, ring fully inside the grid.

        n_glands==1 short-circuits to the grid centre and draws NO rng (the founder draw in
        ``_seed_structure`` is then the first rng consumption). n_glands>1 rejection-samples from
        ``self.rng``, the per-run rng seeded by ``seed``."""
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
        K_duct=K_stroma=carrying_capacity gives a single central ring filled to K."""
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

        # founder: one micro-lesion just inside gland 0's ring
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
        # cells that carry the stromal hazard as a LIVE fraction, with headroom for an invasive mass).
        # stroma_fill_frac=1.0 (the default) fills stroma demes to K_stroma.
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

    def _cap_of(self, deme_idx):
        """This deme's integer slot count — its carrying capacity (per-deme K where the ductal field
        set one, the scalar ``carrying_capacity`` otherwise)."""
        K = (self.carrying_capacity if self._deme_capacity is None
             else self._deme_capacity[deme_idx])
        return int(K)

    def _slot_cap_of(self, deme_idx):
        """The deme's structural slot count — ``crowding_overfill`` x its carrying capacity.

        Separate from ``_cap_of``, which stays the DENSITY REFERENCE K used by the crowding law.
        At the default overfill of 1.0 the two coincide and a deme is hard-capped at K. Above 1.0 the
        deme may overshoot into the accelerating region of the density term, which is what makes the
        Noble-style over-capacity response reachable instead of dead code."""
        return int(self._cap_of(deme_idx) * self._crowding_overfill)

    def _evictable_gids(self, deme_idx):
        """The deme's displaceable resident genotypes, in creation-ordinal order (deterministic)."""
        deme = self.demes[deme_idx]
        return sorted((g for g in deme if self.genotypes[g].type in self._EVICTABLE),
                      key=lambda g: self.genotypes[g].ord)

    def _n_evictable(self, deme_idx):
        """How many cells in this deme a birth could displace. The slot allocator sizes its second
        (eviction) draw from this, so it MUST agree with ``_evict``."""
        deme = self.demes[deme_idx]
        return sum(c for g, c in deme.items() if self.genotypes[g].type in self._EVICTABLE)

    def _evict(self, deme_idx, n, rng):
        """Displace ``n`` resident cells from ``deme_idx``, drawn across the resident genotypes in
        proportion to their counts. Returns the number actually removed (< n only if the deme runs
        out of residents). One resident genotype — the overwhelmingly common case — costs no rng
        draw."""
        if n <= 0:
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
        return self._n_evictable(deme_idx) > 0

    def _place_one(self, deme_idx, gid, rng):
        """Place ONE newborn cell, taking a free slot or displacing one resident. Returns whether it
        was placed; a birth into a full deme with no displaceable resident simply FAILS (the division
        is consumed, no cell is added), exactly like a daughter that breaches the viability limits.
        There is deliberately NO redraw of a rejected dispersal target — that would be free-space-
        biased directional dispersal, a different model."""
        if not self._crowding:
            self._add(deme_idx, gid, 1)
            return True
        if sum(self.demes[deme_idx].values()) >= self._slot_cap_of(deme_idx) and not self._evict(deme_idx, 1, rng):
            return False
        self._add(deme_idx, gid, 1)
        return True

    def _cross_gland_target(self, src_gland, rng):
        """A lumen deme of ANOTHER gland, for an island (cross-gland) dispersal hop. Glands are chosen distance-weighted (prob ~ exp(-d/lambda) over
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

    def _neighbors(self, deme_idx):
        # Compartment-aware von Neumann neighbourhood: neighbours stay within the SAME grid. Met demes
        # are a contiguous tail block with their own met_grid_size lattice and met-local coords, so a
        # cancer cell dispersing inside the deposit never leaks into the primary (and vice versa).
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

    def _deme_comp(self, deme, cancer=None):
        """One O(#genotypes-in-deme) pass over a deme returning the per-compartment cell counts a
        death-rate evaluation needs:
        ``(total, n_normal, n_immune, n_epithelial, n_stromal, n_host)``. When a list ``cancer`` is
        given, the same pass appends the deme's cancer genotypes to it, in the deme's order.

        Passing the result into ``_death_rate`` (via its ``comp`` argument) lets every genotype in a
        deme reuse ONE composition scan instead of each re-running the O(#genotypes) immune /
        epithelial / stromal / host ``_*_fraction`` scans. That turns the per-deme death-rate work
        from O(#genotypes²) to O(#genotypes) — the dominant per-substep cost once a high mutation
        rate makes #genotypes track #cells.
        ``n_normal`` matches ``sum(deme.get(nm, 0) for nm in normal_names)`` because a normal
        genotype's id IS its type name."""
        total = n_normal = n_immune = n_epi = n_stro = n_host = 0
        G = self.genotypes
        for gid, c in deme.items():
            total += c
            tp = G[gid].type
            if tp == "cancer":
                if cancer is not None:
                    cancer.append(gid)
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
        return (total, n_normal, n_immune, n_epi, n_stro, n_host)

    def _compartment_fields(self):
        """Per-deme epithelial / stromal LIVE fractions, as two (n_demes,) arrays — the compartment
        as a niche field for the program layer. The SAME clone
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

    def _origin_tick(self, generations):
        """Advance the founding acinus's confinement clock by ``generations`` and release it once the
        lesion has filled the acinus and then distended it for long enough.

        Called once per generation by the tau engine and once per event by the exact engine, with
        that event's Gillespie waiting time, so ``generations`` is time in generations either way. A no-op — and not
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

    # A dispersing daughter leaves deme ``di`` by one of three routes: metastatic seeding (only from a
    # primary stromal deme, so DCIS confined to glands cannot seed), an island hop to another gland's
    # lumen, or a random von Neumann neighbour behind the basement-membrane gate. The exact engine
    # routes one cell at a time (``_route_one``), tau-leaping routes a count (``_route_births``); the
    # probabilities below are shared, and each router keeps its own draws.
    def _p_met_seed(self, di, src_g):
        """Probability that a daughter leaving ``di`` attempts to seed the met (0 when it cannot)."""
        if self._met_enabled and self._met_seed_kappa > 0 and di < self.n_primary_demes and src_g == -1:
            return self._met_seed_kappa / (1.0 + self._met_seed_kappa)
        return 0.0

    def _p_cross_gland(self, src_g):
        """Probability that a daughter leaving gland ``src_g`` hops to another gland's lumen."""
        if self.cross_gland_kappa > 0 and src_g != -1:
            return self.cross_gland_kappa / (1.0 + self.cross_gland_kappa)
        return 0.0

    def _gated(self, di, tgt):
        """Whether a hop from ``di`` to ``tgt`` crosses the basement membrane (duct to stroma)."""
        return (self._breach_gated_invasion and self.gland_id is not None
                and self.gland_id[di] >= 0 and self.gland_id[tgt] < 0)

    @staticmethod
    def _trait(rep, name):
        """A heritable trait of ``rep`` clamped to [0, 1]."""
        return min(max(rep.evolutionary_parameters[name], 0.0), 1.0)

    def _route_one(self, di, rep, rng):
        """Where ONE dispersing cell of ``rep`` leaving deme ``di`` lands, by one of three routes:
        metastatic seeding from a primary stromal deme (only a primary stromal deme can seed, so
        DCIS confined to glands cannot), an island hop to another gland's lumen, or a random
        von Neumann neighbour behind the basement-membrane gate. Returns ``(target, is_met)``, or
        ``None`` when the cell dies in transit to the met."""
        src_g = self.gland_id[di] if self.gland_id is not None else -1
        p_met = self._p_met_seed(di, src_g)
        if p_met and rng.random() < p_met:
            ms = self._trait(rep, "met_survival")
            return (self.met_vessel_idx, True) if rng.random() < self._transit_prob(ms) else None
        tgt = None
        p_cross = self._p_cross_gland(src_g)
        if p_cross and rng.random() < p_cross:
            tgt = self._cross_gland_target(src_g, rng)
        if tgt is None:
            nbrs = self._neighbors(di)
            tgt = nbrs[int(rng.choice(len(nbrs)))] if nbrs else di
        if self._gated(di, tgt) and rng.random() >= self._trait(rep, "breach"):
            tgt = di
        return tgt, False

    def _route_births(self, di, gid, n, rng):
        """Route ``n`` dispersing cells of ``gid`` leaving deme ``di`` by the dispersal branch's three
        routes (metastatic seeding with its transit survival, island hops, random neighbours behind
        the basement-membrane gate). Returns ``(routes, n_lost)``: ``routes`` is a list of
        ``(target, count, is_met)`` and ``n_lost`` the cells that died in transit."""
        routes = []
        src_g = self.gland_id[di] if self.gland_id is not None else -1
        rep = self.genotypes[gid]
        n_met = n_lost = 0
        p_met = self._p_met_seed(di, src_g)
        if p_met:
            n_met = int(rng.binomial(n, p_met))
        if n_met:
            ms = self._trait(rep, "met_survival")
            n_survive = int(rng.binomial(n_met, self._transit_prob(ms)))
            if n_survive:
                routes.append((self.met_vessel_idx, n_survive, True))
            n_lost = n_met - n_survive
        n_rest = n - n_met
        n_cross = 0
        p_cross = self._p_cross_gland(src_g)
        if p_cross:
            n_cross = int(rng.binomial(n_rest, p_cross))
        n_local = n_rest - n_cross
        if n_local > 0:
            nbrs = self._neighbors(di)
            if not nbrs:
                routes.append((di, n_local, False))
            else:
                idx = rng.integers(0, len(nbrs), size=n_local)
                u, cnts = np.unique(idx, return_counts=True)
                b = self._trait(rep, "breach")
                for k, cnt in zip(u, cnts):
                    tgt = nbrs[int(k)]; cnt = int(cnt)
                    if self._gated(di, tgt):
                        n_ok = int(rng.binomial(cnt, b)) if b > 0 else 0
                        if n_ok:
                            routes.append((tgt, n_ok, False))
                        if cnt - n_ok:
                            routes.append((di, cnt - n_ok, False))   # crossing fails: stays home
                    else:
                        routes.append((tgt, cnt, False))
        for _ in range(n_cross):
            tgt = self._cross_gland_target(src_g, rng)
            if tgt is None:
                nbrs = self._neighbors(di)
                tgt = nbrs[int(rng.integers(0, len(nbrs)))] if nbrs else di
            routes.append((tgt, 1, False))
        return routes, n_lost
