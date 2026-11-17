"""The metastatic deposit: a second lattice of host parenchyma, its vessel deme and the
transit of seeding cells.

A mixin of [`GenotypeTumor`][iscc.tumor.GenotypeTumor]; it holds no state of its own."""

from collections import Counter
import numpy as np
from ...constants import LAYOUT_OFFSET_MET


class MetastasisMixin:
    def _init_met(self, p):
        """The metastatic deposit's settings (the lattice itself is seeded by ``_seed_met``)."""
        # Metastasis module: a SECOND deme-grid — the metastatic deposit — appended to
        # self.demes, sharing the genotype registry + genotypes_parents genealogy + Selection with the
        # primary (so a clone keeps ONE identity/colour across both grids, enabling the 2-band Muller).
        # The met is a homogeneous met_grid_size x met_grid_size lattice filled with immortal `host`
        # parenchyma; one fixed `vessel` deme is the migration entry point (and O2 point-source).
        # OFF by default (met_grid_size=0 -> _seed_met is a no-op -> nothing appended).
        self.met_grid_size = int(p.spatial.get("met_grid_size", 0))
        self._met_enabled = self.met_grid_size > 0
        self._cap_met = int(p.spatial.get("K_met", self._cap)) if self._crowding else 1
        self._host_fill_frac = float(p.spatial.get("host_fill_frac", 1.0))
        _mv = p.spatial.get("met_vessel")
        self.met_vessel = (tuple(_mv) if _mv is not None
                           else (self.met_grid_size // 2, self.met_grid_size // 2))
        self.met_vessel_idx = None
        # Host-tissue death hazard (attenuated by met_survival; used in _death_rate) and the
        # migration knobs (seed rate + transit-survival floor; used in the dispersal paths).
        # All default to 0 -> inert.
        self._met_hazard = p.spatial.get("met_hazard", 0.0)
        self._met_seed_kappa = float(p.spatial.get("met_seed_kappa", 0.0))
        self._met_transit_floor = float(p.spatial.get("met_transit_floor", 0.0))

    def _seed_met(self):
        """Seed the metastatic deposit: a met_grid_size x met_grid_size lattice appended to
        self.demes, every deme filled to K_met with immortal/static `host` parenchyma, plus one fixed
        `vessel` deme (the migration entry point and O2 point-source). No internal structure. A no-op
        when met_grid_size == 0 (nothing appended)."""
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
        # CELLTYPES, which would draw an extra layout_rng.beta and shift the shared landscape), so
        # the landscape is the same whether or not the met exists.
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

    def _transit_prob(self, met_survival):
        """Probability a migrating daughter survives transit to the met, biased by the clone's
        heritable met_survival: met_transit_floor + (1 - met_transit_floor) * met_survival. A low floor
        makes wild-type cells (met_survival 0) rarely survive and high-met_survival clones survive often,
        so the deposit's founders are ENRICHED for met_survival — the transit bottleneck that makes the
        met a non-representative, often-minor sample of the primary's clones."""
        return self._met_transit_floor + (1.0 - self._met_transit_floor) * met_survival

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
