"""Treatment on the count engine: dosing, the label and mechanistic drug responses,
surgery, and the drug-induced resistance state.

A mixin of [`GenotypeTumor`][iscc.tumor.GenotypeTumor]; it holds no state of its own."""

from collections import Counter
import numpy as np
import pandas as pd


class TherapyMixin:
    def _init_therapy_state(self, p):
        """Per-step treatment overrides and the active dose's context (empty until a dose)."""
        # per-immune-cell contact kill hazard, and per-step treatment overrides
        # (gid -> extra death hazard / overridden immune resistance), refreshed each
        # treated step by _apply_treatment so they never corrupt the shared genotype.
        self._immune_prob_kill = (p.immune or {}).get("prob_kill", 0.01)
        self._tx_death_add = {}
        self._tx_immune_resist = {}
        # The active dose's context, kept alongside the override dicts so the hazard for a genotype
        # MINTED MID-STEP (by a division-mutation during update / _tau_generation, after this step's
        # _apply_treatment already froze the dicts) can be computed on demand rather than defaulting
        # to zero, so a newborn clone never escapes the drug for a step (mid-step minting is most
        # frequent in therapy-induced-mutagenesis runs). None => no active treatment => the
        # on-demand paths return the untreated defaults.
        self._tx_treatment = None
        self._tx_dosage = 0.0
        # Per-genotype division multiplier that applies ONLY while the drug is on (drug-tolerant
        # persisters). The tolerant state is DRUG-INDUCED: off drug a tolerant cell is an ordinary
        # cell and pays nothing, so this cost cannot live in `division_rate`, which is baked at
        # genotype level and would apply everywhere and always -- purging the tolerant pool long
        # before the first dose. Empty dict => no override.
        self._tx_div_mult = {}
        self._tx_sites = "both"  # active treatment's target compartment(s): both / met / primary

    def _tx_applies(self, deme_idx):
        """Whether the active treatment's per-genotype override acts in this deme's compartment
        (Treatment.sites: 'both' / 'met' / 'primary'). 'both' (the default) acts everywhere."""
        if self._tx_sites == "both":
            return True
        is_met = deme_idx >= self.n_primary_demes
        return is_met if self._tx_sites == "met" else (not is_met)

    def _state_protection(self, rep):
        """Drug protection conferred by the carried resistance STATE: the cell's state level x the
        configured ``resistance_state_effect``. Folded into the
        ``max(treatment_resistance, drug_tolerance, ...)`` protection everywhere the drug's
        ``(1 - protection)`` factor is computed. Returns 0.0 when the feature is off (effect 0.0) or the
        cell is not in the state, so the max then reduces to ``max(tr, dt)``."""
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
        chemotherapy damages cells as they replicate (hence its sparing of quiescent tissue)."""
        tx = treatment if treatment is not None else self._tx_treatment
        kill_rate = getattr(tx, "kill_rate", 0.8)
        if getattr(tx, "kill_mode", "additive") == "proliferation":
            return intensity * kill_rate * rep.evolutionary_parameters["division_rate"]
        return intensity * max(kill_rate - rep.evolutionary_parameters["death_rate"], 0.0)

    @staticmethod
    def _is_mechanistic(tx):
        return getattr(tx, "response", "label") == "mechanistic"

    @staticmethod
    def _carries_target(rep, coord, amplification_threshold):
        """Does this clone carry the target at ``coord = (segment, position)``? Yes if any copy of the
        segment has the mutation there, or if the segment has more copies than
        ``amplification_threshold``. Copy number is segment-granular (each copy is a whole-segment
        SNV bitset), so a deletion of the mutated copy, or of copies down to the threshold, is what
        makes a lineage lose the target."""
        seg, pos = coord
        copies = rep.genome[seg]["p"] + rep.genome[seg]["m"]
        if any(c[pos] for c in copies):
            return True
        return amplification_threshold is not None and len(copies) > amplification_threshold

    def _target_expression(self, gid, rep, table):
        """The clone's mean expression of the drug's target genes, memoised per genotype.

        The same law ``make_cell_data`` materialises, read at the genotype level: the dosage-driven
        baseline (``get_exp`` with the program layer off, the allele-resolved ``get_exp_alleles``
        with it on) times the program multiplier at the clone's MEAN activity (``ProgramModel.mean_z``
        of its genotype drive). The per-cell activity draw and the per-deme niche are left out: one is
        noise around this mean, the other a property of where a cell sits, not of the genotype. A
        genotype's genome and phenotype never change once registered, so the memo never goes stale."""
        level = table["levels"].get(gid)
        if level is None:
            rep.baseline_exp = self.celltype_exps["cancer"]     # what make_cell_data sets
            P = self.programs
            if P is None:
                exp = rep.get_exp(self.selection.mut_effects)
            else:
                ep, em = rep.get_exp_alleles(dosage_sensitivity=P.dosage_sensitivity,
                                             snv_exp_effect=P.snv_exp_effect,
                                             dosage_saturation=P.dosage_saturation,
                                             dosage_law=getattr(P, "dosage_law", "haplotype"))
                drive = (P.clone_drive(rep.evolutionary_parameters, rep.baseline_rates,
                                       rep.get_snvs()) + P.celltype_bias("cancer"))
                exp = (ep + em) * P.gene_multiplier(P.mean_z(drive)[None, :])[0]
            level = float(np.mean(exp[table["genes"]]))
            table["levels"][gid] = level
        return level

    def _target_table(self, tx):
        """Per-tumour memo for an expression-targeted drug: its flat gene indices, the founding clone's
        expression of them (the reference that sets the nominal dose), and the per-genotype levels.
        Keyed by the gene list, so two drugs on the same genes share it. Created lazily, so an older
        pickled tumour without the memo still loads."""
        key = tuple(tx.target_genes)
        cache = self.__dict__.setdefault("_mech_expr", {})
        table = cache.get(key)
        if table is None:
            off = self.selection._seg_offsets
            genes = []
            for g in key:
                if isinstance(g, tuple):
                    seg, pos = g
                    if not (0 <= seg < self.n_segments and 0 <= pos < off[seg + 1] - off[seg]):
                        raise ValueError(f"target gene {g} is not on the genome")
                    g = off[seg] + pos
                if not 0 <= g < self.n_genes:
                    raise ValueError(f"target gene {g} is not on the genome ({self.n_genes} genes)")
                genes.append(int(g))
            table = {"genes": np.array(genes, dtype=int), "levels": {}}
            founder = self.genotypes[self.founder_id]
            table["ref"] = self._target_expression(self.founder_id, founder, table)
            if not table["ref"] > 0:
                raise ValueError("the founding clone does not express the target genes, so there is "
                                 "no reference level to scale the drug's dose by")
            cache[key] = table
        return table

    def _mech_dose(self, tx, gid, rep):
        """``(p, x)`` for a CANCER clone under ``response="mechanistic"``: the dose it takes is
        ``dosage * p * x``, the mechanistic stand-in for the label model's ``p * (1 - protection)``.

        No resistance label enters: ``treatment_resistance``, ``drug_tolerance`` and the carried
        resistance state are all ignored. ``p`` is ``effectiveness`` when the clone carries every
        target the drug names (or it names none) and ``toxicity`` otherwise, so the off-target effect
        is unchanged. ``x`` is 1, except for an on-target clone of an expression-targeted drug, where
        it is the clone's expression of the target genes relative to the founding clone's, and 0 --
        no effect at all -- below ``expression_threshold``."""
        targets = getattr(tx, "targets", None)
        if targets:
            amp = getattr(tx, "amplification_threshold", None)
            if not all(self._carries_target(rep, t, amp) for t in targets):
                return tx.toxicity, 1.0
        if not getattr(tx, "target_genes", None):
            return tx.effectiveness, 1.0
        table = self._target_table(tx)
        level = self._target_expression(gid, rep, table)
        if level < tx.expression_threshold:
            return tx.effectiveness, 0.0
        return tx.effectiveness, level / table["ref"]

    def _tx_death_add_for(self, gid, rep=None):
        """The active treatment's extra death hazard for ``gid`` -- the value ``_apply_treatment``
        stored, or, for a genotype minted AFTER that step's rebuild, the same value computed on demand
        and memoised so it is charged from its very first step instead of leaking a drug-free step.

        The formula mirrors ``_apply_treatment``'s death-rate branch EXACTLY (same target test, same
        ``max(treatment_resistance, drug_tolerance)`` protection, same ``kill_rate - death_rate``
        intensity), so a memoised entry and an on-demand one are identical -- a newborn clone and its
        already-registered siblings take the same hazard. Returns 0.0 (and memoises nothing) when no
        death-rate dose is active, so untreated / immunotherapy-only runs pay nothing."""
        val = self._tx_death_add.get(gid)
        if val is not None:
            return val
        tx = self._tx_treatment
        if tx is None or getattr(tx, "affects", "death_rate") == "immune_resistance":
            return 0.0
        rep = rep if rep is not None else self.genotypes[gid]
        if self._is_cancer(gid) and self._is_mechanistic(tx):
            p, x = self._mech_dose(tx, gid, rep)
            intensity = self._tx_dosage * p * x
        else:
            if self._is_cancer(gid):
                target = self._is_treatment_target(tx, rep)
                tr = max(rep.evolutionary_parameters["treatment_resistance"],
                         rep.evolutionary_parameters.get("drug_tolerance", 0.0),
                         self._state_protection(rep))            # carried resistance state
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
        clone's untouched ``base_ir`` (and memoises nothing) when no immunotherapy dose reaches it."""
        val = self._tx_immune_resist.get(gid)
        if val is not None:
            return val
        tx = self._tx_treatment
        if (tx is None or getattr(tx, "affects", "death_rate") != "immune_resistance"
                or not self._is_cancer(gid)):
            return base_ir
        mech = self._is_mechanistic(tx)
        if mech:
            p, x = self._mech_dose(tx, gid, rep)
            intensity = self._tx_dosage * p * x
        else:
            target = self._is_treatment_target(tx, rep)
            tr = rep.evolutionary_parameters["treatment_resistance"]
            p = tx.effectiveness if target else tx.toxicity
            intensity = self._tx_dosage * p * (1.0 - min(max(tr, 0.0), 1.0))
        if intensity <= 0:
            return base_ir                      # matches _apply_treatment's `continue` (no entry)
        # A mechanistic dose can exceed 1 (a clone making twice the founder's target), and stripping
        # more than all of the resistance would make it negative.
        strip = max(0.0, 1.0 - intensity) if mech else 1.0 - intensity
        val = rep.evolutionary_parameters["immune_resistance"] * strip
        self._tx_immune_resist[gid] = val       # memoise: read identically on every later step
        return val

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
        shared genotype. The per-cell stochastic effectiveness/toxicity becomes an expected,
        dose-scaled intensity over the exchangeable count.
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
        # response="mechanistic": every cancer-clone dose below comes from `_mech_dose` (genotype and
        # expression) instead of the resistance labels; normal tissue is unchanged.
        mech = self._is_mechanistic(treatment)
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
            # rate that deletes its own resistance allele. Default "uniform".
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
                if dose_scaled and mech:
                    # Mechanistic: the exposure `x` stands in for (1 - protection) -- the drug that
                    # reaches the clone, by its target expression -- capped at the full multiplier.
                    _, x = self._mech_dose(treatment, gid, rep)
                    factor = 1.0 + (mutagenicity - 1.0) * min(max(x, 0.0), 1.0)
                elif dose_scaled:
                    tr = max(rep.evolutionary_parameters["treatment_resistance"],
                             rep.evolutionary_parameters.get("drug_tolerance", 0.0),
                             self._state_protection(rep))    # carried resistance state
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
                if mech:
                    p, x = self._mech_dose(treatment, gid, rep)
                    intensity = dosage * p * x
                else:
                    target = self._is_treatment_target(treatment, rep)
                    tr = rep.evolutionary_parameters["treatment_resistance"]
                    p = treatment.effectiveness if target else treatment.toxicity
                    intensity = dosage * p * (1.0 - min(max(tr, 0.0), 1.0))
                if intensity <= 0:
                    continue
                ir = rep.evolutionary_parameters["immune_resistance"]
                strip = max(0.0, 1.0 - intensity) if mech else 1.0 - intensity  # as _tx_immune_resist_for
                self._tx_immune_resist[gid] = ir * strip
            else:
                # Death-rate therapy (chemo / targeted). Cancer TARGET cells take `effectiveness`;
                # everything OFF-TARGET -- non-target cancer clones AND normal/host tissue -- takes
                # `toxicity` (one knob: toxicity = off-target anything). Normal cells carry no treatment
                # resistance; giving them a `_tx_death_add` is what makes chemo's systemic toxicity kill
                # normal tissue (they become death-only participants in the dynamics -- see _deme_rate /
                # update / _tau_substep).
                if is_cancer and mech:
                    # No labels, so no persister cost either: drug tolerance is not consulted.
                    p, x = self._mech_dose(treatment, gid, rep)
                    intensity = dosage * p * x
                    if intensity > 0:
                        self._tx_death_add[gid] = self._kill_amount(rep, intensity, treatment)
                    continue
                if is_cancer:
                    target = self._is_treatment_target(treatment, rep)
                    tr = max(rep.evolutionary_parameters["treatment_resistance"],
                             rep.evolutionary_parameters.get("drug_tolerance", 0.0),
                             self._state_protection(rep))    # carried resistance state
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

    def _resect(self, site="primary", region=None, margin=0):
        """Surgical resection: remove EVERY cell (cancer + immortal residents) from the target
        compartment's demes and refresh their rates to 0, so the shared-deme_rates Gillespie fires only
        the remaining compartment afterward (and primary->met seeding stops with no primary cancer
        left). Deterministic — no rng draw. The genealogy is NOT pruned, so a 2-band Muller still shows
        the resected band cliff to 0. ``site`` is "primary" or "met".

        With ``region`` only those demes are cleared and the ground truth of the cut is returned (see
        ``_resect_region``). Either way the per-cell tables no longer describe the tumour, so they are
        marked stale for ``ensure_cell_data``."""
        if site == "primary":
            lo, hi = 0, self.n_primary_demes
        elif site == "met":
            lo, hi = self.n_primary_demes, len(self.demes)
        else:
            raise ValueError(f"unknown resection site {site!r}")
        self._cell_data_stale = True
        if region is not None:
            return self._resect_region(site, lo, hi, region, margin)
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

    def _resect_region(self, site, lo, hi, region, margin):
        """Partial resection: clear only the demes of ``region`` within the compartment ``[lo, hi)``
        and return the ground truth of the cut, also appended to ``self.resections``.

        ``region`` is global deme indices in the compartment, or a boolean mask over its demes (flat
        or ``(side, side)``). Distances are in demes, centre to centre, to the nearest deme across the
        cut: a region deme's distance to the nearest deme outside it (the rim is the region demes
        within ``margin``), and an outside deme's distance to the nearest region deme (the residual's
        ``distance``). Both come from an exact Euclidean distance transform on the compartment grid,
        which treats the grid's edge as tissue's end rather than a cut. Cells of the OTHER compartment
        are neither cut nor counted. Deterministic — no rng draw — and, like the whole-compartment
        resection, it leaves the genealogy unpruned."""
        from scipy.ndimage import distance_transform_edt

        side = self.grid_size if site == "primary" else self.met_grid_size
        n = hi - lo
        arr = np.asarray(region)
        if arr.dtype == bool:
            if arr.size != n:
                raise ValueError(f"a region mask must have one entry per {site} deme ({n}), "
                                 f"got shape {arr.shape}")
            mask = arr.reshape(-1).copy()
        else:
            idx = arr.reshape(-1).astype(int)
            if idx.size and (idx.min() < lo or idx.max() >= hi):
                raise ValueError(f"region has demes outside the {site} compartment "
                                 f"(global indices {lo}..{hi - 1})")
            mask = np.zeros(n, dtype=bool)
            mask[idx - lo] = True
        if not mask.any():
            raise ValueError("region selects no deme")
        grid = mask.reshape(side, side)
        # inside: a region deme's distance to the nearest outside deme; outside: the reverse. A
        # region covering the whole compartment has no cut, so nothing is in the rim.
        inside = (distance_transform_edt(grid) if not grid.all()
                  else np.full(grid.shape, np.inf)).reshape(-1)
        outside = distance_transform_edt(~grid).reshape(-1)
        rim = mask & (inside <= margin) if margin > 0 else np.zeros(n, dtype=bool)

        removed, margin_cancer, residual = Counter(), Counter(), []
        for k in range(n):
            di = lo + k
            for gid, c in self.demes[di].items():
                if c <= 0 or not self._is_cancer(gid):
                    continue
                if mask[k]:
                    removed[gid] += c
                    if rim[k]:
                        margin_cancer[gid] += c
                else:
                    r, col = self.deme_coords[di]
                    residual.append((di, r, col, float(outside[k]), gid, int(c)))
        n_cells = 0
        for k in np.flatnonzero(mask):
            di = lo + int(k)
            deme = self.demes[di]
            for gid in list(deme):
                n_cells += deme[gid]
                self._remove(di, gid, deme[gid])
            self._refresh_rate(di)

        record = dict(
            step=self.step, time=self.time, site=site, region=np.flatnonzero(mask) + lo,
            margin=margin, removed=dict(removed), margin_cancer=dict(margin_cancer),
            margin_positive=bool(margin_cancer),
            residual=pd.DataFrame(residual,
                                  columns=["deme", "row", "col", "distance", "genotype", "n"]))
        self.__dict__.setdefault("resections", []).append(record)
        self.events.append(dict(step=self.step, time=self.time, event="resection", site=site,
                                n=int(n_cells), n_cancer=int(sum(removed.values())),
                                region_demes=int(mask.sum()), margin=margin,
                                margin_cancer=int(sum(margin_cancer.values())),
                                residual=int(sum(c for *_, c in residual))))
        return record

    def _clear_treatment_overrides(self):
        """Drop any residual per-step treatment overrides left by a PRIOR treated grow, so a subsequent
        grow with no active treatment (or an inactive window) sees no stale hazard on cancer OR normal
        tissue. Refreshes the rate vector only when it actually clears something, so it is a no-op
        whenever there was nothing left over (e.g. a fresh or never-treated tumour)."""
        if self._tx_death_add or self._tx_immune_resist or self._tx_div_mult:
            self._tx_death_add, self._tx_immune_resist, self._tx_sites = {}, {}, "both"
            self._tx_div_mult = {}
            self._tx_treatment, self._tx_dosage = None, 0.0
            self.deme_rates = np.array([self._deme_rate(i) for i in range(len(self.demes))], dtype=float)

    def _state_twin(self, gid, target_state):
        """Get-or-create the genotype with ``gid``'s genome but ``resistance_state == target_state`` —
        the (genotype x epistate) sublineage the state feature needs.

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
        generation. Only called when ``resistance_state_on``.

        ENTRY (plasticity): while a death-rate dose reaches a deme, each sensitive (state 0) cell has a
        per-generation probability ``induction * intensity`` of being reprogrammed into the state, where
        ``intensity = dosage * p * (1 - protection)`` is the dose the cell actually receives — the same
        factor ``_tx_death_add_for`` / ``_kill_amount`` use — so a cell the drug cannot reach is not
        reprogrammed by it. EXIT (relaxation): where no dose reaches, each state cell with NO genetic
        anchor (``n_mut_tr == 0``) relaxes back to sensitive with probability ``relax`` (= 1/tau_relax);
        a cell that still carries the resistance allele keeps its genetic attractor and does not relax
        (which also stops the genetic-entry clause from re-firing on an exit twin). Every transition moves cells between the source genotype
        and its ``_state_twin`` (a new/reused genotype id). Moves are collected first and applied after,
        mirroring the substep's death/mutation batching, so the deme/count dicts are not mutated mid-scan.

        Tau-engine only (called from ``_tau_generation``); the genetic-entry arm and the drug-protection
        term live in shared code and so are active under the exact engine too, but the reversible
        induction/relaxation transitions are not."""
        sel = self.selection
        induction, relax = sel.resistance_state_induction, sel.resistance_state_relax
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
                    # scaled by the dose it receives.
                    p_in = 0.0
                    if treated and induction > 0.0 and self._is_mechanistic(tx):
                        # the dose the clone takes by its genotype and expression, not its labels
                        p_eff, x = self._mech_dose(tx, gid, rep)
                        p_in += induction * self._tx_dosage * p_eff * x
                    elif treated and induction > 0.0:
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
                    # is not reaching it.
                    p_out = relax if (not treated and relax > 0.0) else 0.0
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
