import numpy as np
import pandas as pd
from copy import copy, deepcopy

from .epistasis import bits_to_events


class Cell(object):
    # Process-wide monotonic counter for unique genotype ids. Using str(id(self)) is unsafe:
    # CPython recycles object ids after a cell is garbage-collected, so a new clone could
    # collide with a still-living genotype (seen as "genotype is its own parent").
    _genotype_counter = 0

    def __init__(
        self,
        n_segments=10,
        segment_size=1000,
        segment_sizes=None,
        parent=None,
        division_rate=1.,
        death_rate=0.1,
        max_birth_rate=0.3,
        dispersal_rate=0.1,
        n_onc=0,
        n_tsg=0,
        n_disp=0,
        n_ir=0,
        n_tr=0,
        n_breach=0,
        n_ss=0,
        n_ms=0,
    ):
        self.n_segments = n_segments
        # Segments may have unequal sizes (real-genome mode: segment size proportional to
        # chromosome-arm length). When ``segment_sizes`` is None we fall back to the uniform
        # scalar ``segment_size`` for every segment, so the abstract mode is unchanged
        # (byte-identical flat indexing). ``_seg_offsets`` gives the flat genome offset of each
        # segment so positions can be addressed as offset[seg] + pos.
        if segment_sizes is None:
            segment_sizes = [segment_size] * n_segments
        self.segment_sizes = [int(s) for s in segment_sizes]
        self._seg_offsets = np.concatenate([[0], np.cumsum(self.segment_sizes)]).astype(int)
        self.n_genes = int(self._seg_offsets[-1])
        # representative scalar size (uniform fallback; equals segment_sizes[0] in real mode)
        self.segment_size = segment_size
        self.n_onc = n_onc
        self.n_tsg = n_tsg
        self.n_disp = n_disp
        self.n_ir = n_ir
        self.n_tr = n_tr
        self.n_breach = n_breach
        self.n_ss = n_ss
        self.n_ms = n_ms
        self.type = "healthy"
        self.parent = parent
        if parent is None:
            # Compact genome: each allele copy is a boolean bitset of length
            # segment_size (bit i == position i is mutated, infinite-sites). A fresh
            # dict/bitset per segment avoids aliasing across segments.
            self.genome = [{'p': [np.zeros(self.segment_sizes[i], dtype=bool)],
                            'm': [np.zeros(self.segment_sizes[i], dtype=bool)]}
                           for i in range(n_segments)] # copy number = 2
            self.set_genotype_id()
            self.genome_summary = {'n_wt_onc': n_onc * 2,
                                'n_mut_onc': 0,
                                    'n_wt_tsg': n_tsg * 2,
                                    'n_mut_tsg': 0,
                                    'n_wt_disp': n_disp * 2,
                                    'n_mut_disp': 0,                                
                                    'n_wt_ir': n_ir * 2,
                                    'n_mut_ir': 0,
                                    'n_wt_tr': n_tr * 2,
                                    'n_mut_tr': 0,
                                    'n_wt_breach': n_breach * 2,
                                    'n_mut_breach': 0,
                                    'n_wt_ss': n_ss * 2,
                                    'n_mut_ss': 0,
                                    'n_wt_ms': n_ms * 2,
                                    'n_mut_ms': 0,
                                    'ploidy': 2,
                                    'highest_cn': 2,
                                    'nullisomy_count': 0,
                                    'n_mutated_drivers': 0,
                                    'seg_mut_drivers': [0] * n_segments,
                                    'seg_cns': [2] * n_segments,}
        else:
            self.genome = list(parent.genome)
            self.genotype_id = self.parent.genotype_id
            self.genome_summary = dict(parent.genome_summary)
      

        self.max_birth_rate = max_birth_rate
        # Configured baseline rates. Genome-driven selection is applied as a bounded,
        # baseline-relative multiplier on top of these (see update_evolutionary_parameters),
        # so the all-wild-type genome keeps these configured rates.
        self.baseline_rates = {'division_rate': division_rate,
                               'death_rate': death_rate,
                               'dispersal_rate': dispersal_rate}

        # Epistasis event set (R14, DESIGN_epistasis.md). ``event_bits`` is a bitmask over the shared
        # event alphabet ("event i has been acquired"); ``event_groups`` is the order this lineage
        # acquired them — the per-patient ordering ground truth TreeMHN/CBN are scored against.
        #
        # ``event_groups`` is a tuple of TUPLES, one per mutating division: events in the same group
        # were acquired together and are genuinely TIED — their relative order does not exist. Being
        # explicit about that matters. Flattening the groups (as ``event_order`` does, for the tree
        # exports that need a linear path) has to break ties somehow, and any fixed rule invents an
        # ordering the simulator never generated — which would then be scored as if it were real, at
        # whatever rate the tie-breaking happens to agree with the planted DAG. Order scoring must
        # therefore read ``event_groups`` and skip ties (see integrations.progression.score_order).
        #
        # Both are IMMUTABLE (int / nested tuple), so the shallow copy() in divide() propagates them
        # by value with no aliasing. Kept OFF genome_summary so the summary schema (and every
        # dataframe built from it) is unchanged when epistasis is off. Events are MONOTONE by design:
        # once acquired, never lost — the generative assumption MHN/CBN/TreeMHN are defined under.
        self.event_bits = 0 if parent is None else parent.event_bits
        self.event_groups = () if parent is None else parent.event_groups

        # Whole-genome-duplication ground truth (DESIGN_focal_cna.md v1): has THIS lineage ever
        # undergone a WGD. A plain immutable bool, kept OFF genome_summary exactly like event_bits —
        # so the summary schema (and every dataframe built from it) is unchanged when WGD is off — and
        # propagated by value through both the shallow copy() in divide() and the parent branch above.
        # Monotone: a WGD is never "undone", so once True it stays True down the lineage.
        self.is_wgd = False if parent is None else parent.is_wgd

        self.evolutionary_parameters = dict()
        self.evolutionary_parameters['division_rate'] = division_rate
        self.evolutionary_parameters['death_rate'] = death_rate
        self.evolutionary_parameters['dispersal_rate'] = dispersal_rate
        self.evolutionary_parameters['treatment_resistance'] = 0.  # wild-type: fully treatment-sensitive
        self.evolutionary_parameters['immune_resistance'] = 0.     # wild-type: no immune escape
        self.evolutionary_parameters['breach'] = 0.                # wild-type: cannot breach the epithelial ring
        self.evolutionary_parameters['stromal_survival'] = 0.      # wild-type: no stromal survival advantage
        self.evolutionary_parameters['met_survival'] = 0.          # wild-type: no metastatic-host survival advantage
        self.evolutionary_parameters['viability'] = 1.

    def update_evolutionary_parameters(self, selection):
        """Apply the CINner-style selection model as a bounded, baseline-relative modifier.

        division/dispersal = configured baseline x relative fitness (>=0), clamped to a valid
        rate range; resistances map a relative fitness >=1 into [0,1) (wild-type -> 0). This
        keeps the all-wild-type genome at its configured rates and prevents the unbounded
        blow-up of the raw multiplicative form for many-gene genomes.
        """
        gs = self.genome_summary
        self.evolutionary_parameters['viability'] = selection.update_viability(gs)
        self.evolutionary_parameters['division_rate'] = min(
            self.baseline_rates['division_rate'] * selection.update_division_rate(
                gs, event_bits=self.event_bits),
            self.max_birth_rate)
        # Routed through selection so the epistasis layer can make a synthetic-lethal combination
        # actually lethal. Returns `param` unchanged when epistasis is off / no exclusive pairs, so
        # the death path is bit-identical to before.
        self.evolutionary_parameters['death_rate'] = selection.update_death_rate(
            gs, self.baseline_rates['death_rate'], event_bits=self.event_bits)
        self.evolutionary_parameters['dispersal_rate'] = min(
            self.baseline_rates['dispersal_rate'] * selection.update_dispersal_rate(gs), 1.0)
        self.evolutionary_parameters['immune_resistance'] = max(
            0.0, 1.0 - 1.0 / selection.update_immune_resistance(gs))
        self.evolutionary_parameters['treatment_resistance'] = max(
            0.0, 1.0 - 1.0 / selection.update_treatment_resistance(gs))
        # Compartment-selection traits (v1): heritable resistances that attenuate a matching
        # compartment hazard in _death_rate (breach <- epithelial barrier, stromal_survival <-
        # stromal hazard), mapped from a relative fitness >=1 into [0,1) exactly like immune
        # resistance. With prop_breach/prop_stromal_survival=0 (default) N_*=0 -> update_* returns
        # 1 -> trait 0, so this is a no-op and growth is byte-identical.
        #
        # WHAT the update_* below read off the genome is the selection model's ``trait_source``:
        # "dosage" (default) uses the total copies of the axis's genes, so copy-number change moves
        # the trait; "mutation" uses only the SNV-mutated copies, so the trait has to be earned by
        # mutation. The 1 - 1/F map is the same either way, and so is the oncogene/TSG driver fitness
        # in update_division_rate above, which always reads dosage.
        self.evolutionary_parameters['breach'] = max(
            0.0, 1.0 - 1.0 / selection.update_breach(gs))
        self.evolutionary_parameters['stromal_survival'] = max(
            0.0, 1.0 - 1.0 / selection.update_stromal_survival(gs))
        self.evolutionary_parameters['met_survival'] = max(
            0.0, 1.0 - 1.0 / selection.update_met_survival(gs))
        # Go-or-grow trade-off (R15): the dissemination/niche traits cost proliferation everywhere,
        # while their benefits are compartment-gated in _death_rate, so each is net-favoured only in its
        # niche. Applied AFTER the traits are known; a no-op (x1.0) unless a *_cost is configured.
        self.evolutionary_parameters['division_rate'] *= selection.proliferation_cost(
            self.evolutionary_parameters)

    def _recount_seg_drivers(self, selection, seg):
        """Refresh this segment's contribution to ``n_mutated_drivers``: the number of DISTINCT
        driver genes (onc union tsg, i.e. driver_types != 0) mutated on AT LEAST ONE copy.

        Distinct genes, not mutated copies. n_mut_onc/n_mut_tsg cannot stand in for this: they are
        copy-weighted, because update_genome_summary_cnv scales them by `sign` as copies come and
        go, so one mutated oncogene at copy number 4 contributes 4 to them but 1 here.

        Recomputed from the genome rather than carried incrementally, because the count is not
        monotone: a deletion un-mutates a gene whose only mutated copy it removed, so an
        incremental counter would have to track every position's mutated-copy multiplicity. The
        union below is O(copies x segment_size) for the ONE segment that changed -- the same order
        as the per-category counting its callers already do -- and every caller (mutate's SNV and
        CNV branches, cohort._apply_germline_mutations) writes the genome bits BEFORE calling in,
        so it always reads the post-event state.
        """
        drivers = selection.drivers[seg]
        copies = self.genome[seg]['p'] + self.genome[seg]['m']
        if len(copies) and len(drivers):
            union = np.zeros(self.segment_sizes[seg], dtype=bool)
            for bits in copies:
                union |= bits          # a gene counts once however many copies carry it
            n = int(union[drivers].sum())
        else:
            n = 0                      # nullisomic segment: no copy, nothing mutated
        # The per-segment breakdown lives on genome_summary so it inherits the copy-on-write
        # deepcopy in mutate(); a plain attribute would alias across the shallow copy() in divide().
        self.genome_summary['n_mutated_drivers'] += n - self.genome_summary['seg_mut_drivers'][seg]
        self.genome_summary['seg_mut_drivers'][seg] = n

    def update_genome_summary_mutation(self, selection, mut_bits, seg):
        # `mut_bits` is a boolean mask over the segment marking newly mutated positions;
        # count how many fall in each driver category by indexing the mask.
        n_new_onc = int(mut_bits[selection.onc[seg]].sum())
        self.genome_summary['n_mut_onc'] += n_new_onc
        self.genome_summary['n_wt_onc'] -= n_new_onc

        n_new_tsg = int(mut_bits[selection.tsg[seg]].sum())
        self.genome_summary['n_mut_tsg'] += n_new_tsg
        self.genome_summary['n_wt_tsg'] -= n_new_tsg

        n_new_disp = int(mut_bits[selection.dispersal[seg]].sum())
        self.genome_summary['n_mut_disp'] += n_new_disp
        self.genome_summary['n_wt_disp'] -= n_new_disp

        n_new_ir = int(mut_bits[selection.immune_resistance[seg]].sum())
        self.genome_summary['n_mut_ir'] += n_new_ir
        self.genome_summary['n_wt_ir'] -= n_new_ir

        n_new_tr = int(mut_bits[selection.treatment_resistance[seg]].sum())
        self.genome_summary['n_mut_tr'] += n_new_tr
        self.genome_summary['n_wt_tr'] -= n_new_tr

        n_new_breach = int(mut_bits[selection.breach[seg]].sum())
        self.genome_summary['n_mut_breach'] += n_new_breach
        self.genome_summary['n_wt_breach'] -= n_new_breach

        n_new_ss = int(mut_bits[selection.stromal_survival[seg]].sum())
        self.genome_summary['n_mut_ss'] += n_new_ss
        self.genome_summary['n_wt_ss'] -= n_new_ss

        n_new_ms = int(mut_bits[selection.met_survival[seg]].sum())
        self.genome_summary['n_mut_ms'] += n_new_ms
        self.genome_summary['n_wt_ms'] -= n_new_ms

        # Epistasis events: an event fires when >= 1 SNV lands in its gene module. Only event_bits is
        # touched here — the TIED group is closed once per division by mutate(), because this method
        # runs once per (segment, allele) and that loop walks segments in index order: grouping here
        # would rank events by which segment their module happens to sit in, which is a property of
        # the layout, not of the evolution. That artifact is invisible in the event SET and silently
        # corrupts every ordering ground truth built from it.
        if selection.epistasis is not None:
            self.event_bits |= selection.epistasis.events_from_mutation(seg, mut_bits)

        # New SNVs can only add distinct mutated drivers, but only where the gene was not already
        # mutated on another copy -- which is exactly what the recount resolves.
        self._recount_seg_drivers(selection, seg)

    def update_genome_summary_cnv(self, selection, allele_bits, seg, sign):
        # A CNV adds/removes one allele copy of segment `seg`. `allele_bits` is that
        # allele's bitset; every driver position on it changes copy number by `sign`.
        # Wild-type copies that change = category positions in the segment minus the
        # mutated ones on this allele.
        #
        # Every category's wild-type count is maintained here whatever the selection model does with
        # it: these counts are the genome's ground truth (and are read back by the assay/clone
        # exports), not a fitness cache. Under ``trait_source="mutation"`` the n_wt_* of the four
        # TRAIT axes simply stop feeding the trait value, so this bookkeeping is unchanged and only
        # Selection's reading of it differs.
        n_mut_onc = int(allele_bits[selection.onc[seg]].sum())
        n_wt_onc = len(selection.onc[seg]) - n_mut_onc
        self.genome_summary['n_mut_onc'] += sign*n_mut_onc
        self.genome_summary['n_wt_onc'] += sign*n_wt_onc

        n_mut_tsg = int(allele_bits[selection.tsg[seg]].sum())
        n_wt_tsg = len(selection.tsg[seg]) - n_mut_tsg
        self.genome_summary['n_mut_tsg'] += sign*n_mut_tsg
        self.genome_summary['n_wt_tsg'] += sign*n_wt_tsg

        n_mut_disp = int(allele_bits[selection.dispersal[seg]].sum())
        n_wt_disp = len(selection.dispersal[seg]) - n_mut_disp
        self.genome_summary['n_mut_disp'] += sign*n_mut_disp
        self.genome_summary['n_wt_disp'] += sign*n_wt_disp

        n_mut_ir = int(allele_bits[selection.immune_resistance[seg]].sum())
        n_wt_ir = len(selection.immune_resistance[seg]) - n_mut_ir
        self.genome_summary['n_mut_ir'] += sign*n_mut_ir
        self.genome_summary['n_wt_ir'] += sign*n_wt_ir

        n_mut_tr = int(allele_bits[selection.treatment_resistance[seg]].sum())
        n_wt_tr = len(selection.treatment_resistance[seg]) - n_mut_tr
        self.genome_summary['n_mut_tr'] += sign*n_mut_tr
        self.genome_summary['n_wt_tr'] += sign*n_wt_tr

        n_mut_breach = int(allele_bits[selection.breach[seg]].sum())
        n_wt_breach = len(selection.breach[seg]) - n_mut_breach
        self.genome_summary['n_mut_breach'] += sign*n_mut_breach
        self.genome_summary['n_wt_breach'] += sign*n_wt_breach

        n_mut_ss = int(allele_bits[selection.stromal_survival[seg]].sum())
        n_wt_ss = len(selection.stromal_survival[seg]) - n_mut_ss
        self.genome_summary['n_mut_ss'] += sign*n_mut_ss
        self.genome_summary['n_wt_ss'] += sign*n_wt_ss

        n_mut_ms = int(allele_bits[selection.met_survival[seg]].sum())
        n_wt_ms = len(selection.met_survival[seg]) - n_mut_ms
        self.genome_summary['n_mut_ms'] += sign*n_mut_ms
        self.genome_summary['n_wt_ms'] += sign*n_wt_ms

        self.genome_summary['seg_cns'][seg] += sign
        seg_cns = self.genome_summary['seg_cns']
        self.genome_summary['ploidy'] = np.mean(seg_cns)
        self.genome_summary['highest_cn'] = np.max(seg_cns)
        # nullisomy = number of segments with zero copies, not the largest copy number.
        self.genome_summary['nullisomy_count'] = int(np.sum(np.asarray(seg_cns) == 0))

        # An amplification duplicates an existing copy and so never changes the DISTINCT count; a
        # deletion lowers it only for genes whose last mutated copy just went.
        self._recount_seg_drivers(selection, seg)

    @property
    def event_order(self):
        """``event_groups`` flattened to a linear acquisition sequence, for the exports that need a
        path (e.g. a mutation tree). Events tied within a division appear in index order — an
        ARBITRARY tie-break, so never score ordering off this; use ``event_groups``."""
        return tuple(e for group in self.event_groups for e in group)

    def _available_sites(self, selection, seg, allele, gate_bits):
        """Positions on this allele copy that may still take a new SNV: unmutated (infinite sites
        per allele) and, under ACCESSIBILITY gating, not inside an event module whose dependency-DAG
        parents are still absent. Accessibility is judged on ``gate_bits`` — the event set as it
        stood at the START of the division — so an event acquired earlier in the same division does
        not unblock its children until the next one."""
        available = np.where(~allele)[0]
        if selection.epistasis is not None and len(available):
            blocked = selection.epistasis.blocked_mask(seg, gate_bits)
            if blocked is not None:
                available = available[~blocked[available]]
        return available

    def set_genotype_id(self):
        Cell._genotype_counter += 1
        self.genotype_id = str(Cell._genotype_counter)

    def divide(self, new_cell_id=0):
        new_cell = copy(self)
        new_cell.parent = self
        # Copy-on-write: the daughter SHARES the parent's genome and genome_summary
        # (the shallow copy() above aliases them). These are only ever modified in
        # mutate(), which copies-first, so clonal cells can safely share one genome
        # until one of them diverges. evolutionary_parameters stays per-cell because
        # treatment mutates it per cell.
        new_cell.evolutionary_parameters = dict(self.evolutionary_parameters)
        return new_cell
    
    def get_genome_df(self):
        segs = []
        haps = []
        poss = []
        muts = []
        for seg in range(self.n_segments):
            sz = self.segment_sizes[seg]
            pos_vec = np.arange(sz)
            seg_vec = seg * np.ones((sz,))
            for hap in self.genome[seg]:
                hap_vec = np.array([hap] * sz)
                for all in self.genome[seg][hap]: # each allele copy is a boolean bitset
                    mut_vec = all.astype(int)
                    segs.extend(seg_vec)
                    haps.extend(hap_vec)
                    poss.extend(pos_vec)
                    muts.extend(mut_vec)
        df = pd.DataFrame({'seg': segs, 'hap':haps, 'pos':poss, 'mut':muts})
        return df

    def get_genome_summary_df(self):
        return pd.DataFrame(self.genome_summary, index=[0])

    def get_snvs(self):
        vafs = np.zeros((self.n_genes,))
        for seg in range(self.n_segments):
            copies = self.genome[seg]['p'] + self.genome[seg]['m']
            cn = self.genome_summary['seg_cns'][seg]
            if copies and cn > 0:
                counts = np.sum(copies, axis=0)  # mutated-copy count per position
                vafs[self._seg_offsets[seg]:self._seg_offsets[seg + 1]] = counts / cn
        return vafs

    def get_snvs_alleles(self):
        """Allele-resolved SNV VAFs: ``(vaf_p, vaf_m)`` whose sum is exactly :meth:`get_snvs`.

        ``vaf_h[gene] = (mutated copies on homolog h) / seg_cn`` — the same VAF definition as
        :meth:`get_snvs` but split by homolog, so a caller can see WHICH parental allele carries a
        variant (the B-allele signal). This is the SNV analogue of :meth:`get_exp_alleles` and the
        base layer the coalescent passenger overlay (``_reconstruct_passengers``) adds to.
        """
        vaf_p = np.zeros((self.n_genes,))
        vaf_m = np.zeros((self.n_genes,))
        for seg in range(self.n_segments):
            cn = self.genome_summary['seg_cns'][seg]
            if cn <= 0:
                continue
            lo, hi = self._seg_offsets[seg], self._seg_offsets[seg + 1]
            cp, cm = self.genome[seg]['p'], self.genome[seg]['m']
            if cp:
                vaf_p[lo:hi] = np.sum(cp, axis=0) / cn
            if cm:
                vaf_m[lo:hi] = np.sum(cm, axis=0) / cn
        return vaf_p, vaf_m

    def get_cnvs(self):
        cnvs = []
        for seg, seg_cn in enumerate(self.genome_summary['seg_cns']):
            cnvs.append([seg_cn] * self.segment_sizes[seg])
        cnvs = np.concatenate([np.asarray(c) for c in cnvs]) if cnvs else np.array([])
        return cnvs

    def set_baseline_exp(self):
        self.baseline_exp = np.random.beta(.1, 1, size=self.n_genes)

    def get_exp(self, seg_mut_effects):
        exp = np.array(self.baseline_exp)
        for seg in range(self.n_segments):
            seg_baseline = self.baseline_exp[self._seg_offsets[seg]:self._seg_offsets[seg + 1]]
            seg_exp = np.array(seg_baseline)
            copies = self.genome[seg]['p'] + self.genome[seg]['m']
            if len(copies) == 0:
                seg_exp = seg_exp * 0.
            else:
                for bits in copies:
                    allele_contrib = seg_baseline * seg_mut_effects[seg] ** bits.astype(float)
                    seg_exp = seg_exp + allele_contrib
            exp[self._seg_offsets[seg]:self._seg_offsets[seg + 1]] = seg_exp
        return exp

    def get_exp_alleles(self, dosage_sensitivity=None, snv_exp_effect=None, dosage_saturation=None):
        """Allele-resolved expression: return ``(exp_p, exp_m)`` instead of their sum.

        This is R13's hard engine prerequisite (DESIGN_expression.md §2, §6-v2). The legacy
        :meth:`get_exp` SUMS the ``p``/``m`` haplotypes, which throws away the B-allele frequency in
        RNA — precisely the signal Numbat / CalicoST / cardelino invert. Keeping the two layers
        separate emits a BAF (``exp_p / (exp_p + exp_m)``) and lets dosage and cis-SNV effects act
        per allele, so a TSG with an NMD SNV on one homolog and a CNA loss of the other comes out
        biallelically silenced *by construction* rather than by a special case.

        Composition per haplotype ``a`` (baseline = ONE copy per haplotype, so a wild-type diploid
        gene returns base/2 + base/2 = base):

            exp_{g,a} = (base_g / 2) · dosage(c_a; s_g) · snv_effect_a

        * ``dosage(c; s_g) = 1 + s_g·(min(c, sat) - 1)`` — per-gene dosage SENSITIVITY. s_g = 1 is
          full linear dosage (the law inferCNV/clonealign assume); s_g = 0 is a fully buffered gene
          whose copy number is invisible in RNA. Most genes sit in between, and that partial,
          per-gene buffering is the confounder that makes those benchmarks a fair test rather than a
          tautology. ``c == 0`` short-circuits to zero: no buffering rescues an absent gene.
        * ``snv_effect_a`` averages each copy's per-site effect over that haplotype's copies, so the
          effect is graded by how many copies actually carry the mutation.

        All three arguments default to None = disabled (identity), so this reduces to a plain
        allele-split dosage model when the R13 overlays are off.

        NOTE this is deliberately NOT the scale of :meth:`get_exp`, which returns base·(1 + copies)
        (a wild-type diploid gene -> 3·base) while normal cells bypass it entirely and sit at 1·base.
        The allele path puts cancer and normal on ONE scale (wild-type diploid -> base for both),
        which is what a malignant-vs-reference comparison (inferCNV) needs. It only ever applies when
        `expression_params` is set, so nothing existing re-baselines.
        """
        exp_p = np.zeros(self.n_genes)
        exp_m = np.zeros(self.n_genes)
        # `dosage_saturation` is quoted in TOTAL copy number (what users think in); a haplotype
        # carries half of it.
        sat_allele = None if dosage_saturation is None else max(float(dosage_saturation) / 2.0, 0.0)

        for seg in range(self.n_segments):
            lo, hi = self._seg_offsets[seg], self._seg_offsets[seg + 1]
            half_base = self.baseline_exp[lo:hi] / 2.0
            s_g = None if dosage_sensitivity is None else np.asarray(dosage_sensitivity[lo:hi])
            eff_g = None if snv_exp_effect is None else np.asarray(snv_exp_effect[lo:hi])

            for hap, out in (("p", exp_p), ("m", exp_m)):
                copies = self.genome[seg][hap]
                c = len(copies)
                if c == 0:
                    continue                      # nullisomic haplotype contributes nothing
                if s_g is None:
                    dosage = float(c)             # no sensitivity model -> plain linear dosage
                else:
                    c_eff = c if sat_allele is None else min(c, sat_allele)
                    dosage = np.maximum(1.0 + s_g * (c_eff - 1.0), 0.0)

                if eff_g is None:
                    snv_factor = 1.0
                else:
                    # mean over this haplotype's copies of (effect where mutated, else 1)
                    bits = np.asarray(copies, dtype=bool)                 # (c, seg_size)
                    snv_factor = np.where(bits, eff_g[None, :], 1.0).mean(axis=0)

                out[lo:hi] = half_base * dosage * snv_factor
        return exp_p, exp_m

    def expresses(self, coordinates, seg_mut_effects, thres=0.5):
        # coordinates: tuple (seg, pos)
        seg, pos = coordinates # unpack
        if len(self.genome[seg]['p']) + len(self.genome[seg]['m']) == 0:
            return False

        if not hasattr(self, 'baseline_exp'):
            self.set_baseline_exp()

        # Compute this single gene's expression directly (mirrors get_exp), without
        # building the whole genome's expression profile. baseline_exp is a flat
        # vector over the genome, and seg_mut_effects is indexed [segment][position].
        gene_base = self.baseline_exp[self._seg_offsets[seg] + pos]
        gene_exp = gene_base
        for hap in self.genome[seg]:
            for allele in self.genome[seg][hap]:
                mut = 1. if allele[pos] else 0.
                gene_exp += gene_base * seg_mut_effects[seg][pos] ** mut

        return gene_exp > thres
        

class EpithelialCell(Cell):
    def __init__(self, **cell_kwargs):
        super(EpithelialCell, self).__init__(**cell_kwargs)
        self.type = "epithelial"
        self.genotype_id = self.type

class StromalCell(Cell):
    def __init__(self, **cell_kwargs):
        super(StromalCell, self).__init__(**cell_kwargs)
        self.type = "stromal"        
        self.genotype_id = self.type

class ImmuneCell(Cell):
    def __init__(self, prob_kill=.01, **cell_kwargs):
        super(ImmuneCell, self).__init__(**cell_kwargs)
        self.prob_kill = prob_kill # probability of killing a neighboring cancer cell
        self.type = "immune"
        self.genotype_id = self.type

class HostCell(Cell):
    # Metastatic-deposit host parenchyma (R9): the generic resident tissue the invading cancer must
    # displace in the met grid — immortal/static like epithelial/stromal, only diluted, never cleared.
    def __init__(self, **cell_kwargs):
        super(HostCell, self).__init__(**cell_kwargs)
        self.type = "host"
        self.genotype_id = self.type

class CancerCell(Cell):
    def __init__(self, mutation_rate=0.1, snv_prob=0.5, cnv_prob=0.5,
                 n_snvs_per_allele=0.5, amp_prob=0.5, wgd_rate=0.0, genotype_id=None, seed=42, **cell_kwargs):
        super(CancerCell, self).__init__(**cell_kwargs)
        self.type = "cancer"
        # mutation_rate is the per-division probability of *any* genome change relative to
        # dispersal (read in the engine). Once a change happens, snv_prob / cnv_prob are the
        # relative weights of the two event kinds; amp_prob is P(amplification | CNA event).
        # n_snvs_per_allele is the Poisson mean number of new SNVs each allele (physical copy)
        # acquires in a mutating division: SNVs are spread genome-wide over all alleles, so the
        # expected number of new SNVs scales with ploidy/copy number (≈ n_snvs_per_allele ×
        # number of alleles). Exposed as configurable parameters so inference can estimate them
        # (DESIGN_inference A.0).
        self.mutation_rate = mutation_rate
        self.snv_prob = snv_prob
        self.cnv_prob = cnv_prob
        self.n_snvs_per_allele = n_snvs_per_allele
        self.amp_prob = amp_prob
        # wgd_rate is a SEPARATE per-division channel (DESIGN_focal_cna.md v1): the probability that
        # a mutating division is a whole-genome duplication INSTEAD of the usual SNV/CNA split, leaving
        # snv_prob/cnv_prob semantics intact. Default 0.0 -> WGD OFF -> mutate() draws no extra random
        # variable, so growth is byte-identical to before (the F8 off-by-default discipline).
        self.wgd_rate = wgd_rate
        self.seed  = seed
        self.set_params()

    def set_params(self):
        if self.parent is not None:
            self.genotype_id = self.parent.genotype_id

    def _cow_privatize(self, seg):
        """Give this cell a private copy of segment ``seg``'s dict and its two hap lists before the
        segment is mutated (the individual allele bitsets stay shared until each is itself copied on
        write, in the mutate branches). Idempotent within one division via ``self._cow_segs``. This
        is the per-segment half of the structural-sharing copy-on-write that replaces mutate()'s old
        whole-genome deepcopy; only segments a division actually writes to are copied."""
        if seg in self._cow_segs:
            return
        old = self.genome[seg]
        self.genome[seg] = {'p': list(old['p']), 'm': list(old['m'])}
        self._cow_segs.add(seg)

    def mutate(self, rng, selection, n_snvs_per_allele=None, snv_prob=None, cnv_prob=None,
               amp_prob=None, wgd_rate=None, func_mask=None):
        # ``func_mask`` (per-segment boolean array of FITNESS-relevant gene positions) turns on the
        # count engine's passenger fast-path: an SNV division that lands NO mutation on a functional
        # position leaves genome_summary unchanged (a pure passenger), so it returns the sentinel
        # ``"passenger"`` and skips the summary update + CINner recompute + genotype id — the caller
        # folds it into the parent clone. ``None`` (the default, and every non-coarsening caller such
        # as the cell engine) preserves the original behaviour byte-for-byte. Only ever set when the
        # engine's ``coarsen_passengers`` is on, which is force-disabled under epistasis (so skipping
        # the summary update can never miss an event firing from a neutral-looking SNV).
        # SNV/CNA split, per-allele SNV rate and amp/del split default to this genotype's
        # configured rates (set in __init__, inherited through divide()), but stay overridable per call.
        n_snvs_per_allele = self.n_snvs_per_allele if n_snvs_per_allele is None else n_snvs_per_allele
        snv_prob = self.snv_prob if snv_prob is None else snv_prob
        cnv_prob = self.cnv_prob if cnv_prob is None else cnv_prob
        amp_prob = self.amp_prob if amp_prob is None else amp_prob
        wgd_rate = self.wgd_rate if wgd_rate is None else wgd_rate
        # Copy-on-write: this cell is diverging into a new genotype, so take a private
        # copy of the (until now possibly shared) genome/summary before modifying them.
        # This is the only place the genome is mutated, so sharing elsewhere is safe.
        #
        # STRUCTURAL SHARING (not a whole-genome deepcopy): a mutating division usually touches
        # only a handful of the n_segments segments, so we copy the top-level list and the two
        # mutable summary lists cheaply here and copy each SEGMENT's dict / hap-list / allele bitset
        # lazily, on first write, via _cow_privatize. Untouched segments stay shared with the parent
        # (their bitsets are never mutated in place — see the per-allele .copy() below). This is
        # byte-identical to the old ``deepcopy(self.genome)`` (same bits, same rng draws) but its
        # cost scales with segments-touched, not n_segments — the dominant growth cost when almost
        # every division mutates (high mutation_rate). See DESIGN_scalability.md §5 phase 6.
        self.genome = list(self.genome)                     # private top list; seg dicts shared for now
        gs = dict(self.genome_summary)                      # private summary; two lists still shared
        gs['seg_cns'] = list(gs['seg_cns'])
        gs['seg_mut_drivers'] = list(gs['seg_mut_drivers'])
        self.genome_summary = gs
        self._cow_segs = set()                              # segments already privatised this division
        # Number of NEW SNVs this division fixes (read by the count engine's passenger-coarsening
        # path to tally a folded neutral daughter's burden; 0 for CNV/WGD, which are never folded).
        self._n_new_snv = 0
        events_before = self.event_bits   # everything acquired below is ONE tied group (see _close_event_group)
        # A fully deleted genome (every segment at nullisomy) has no allele to act on; the
        # per-segment selection weights would be all-zero (0/0 -> NaN). Nothing to mutate.
        if not any(self.genome[s]['p'] or self.genome[s]['m'] for s in range(self.n_segments)):
            return False
        hit_functional = False   # coarsening: did any SNV land on a fitness-relevant position?
        # WGD is its own event channel (DESIGN_focal_cna.md v1), fired with probability wgd_rate per
        # mutating division INSTEAD of the SNV/CNA split. The `wgd_rate > 0` guard is what keeps the
        # off-by-default path byte-identical: at the default 0.0 no random variable is drawn here, so
        # the rng.choice below sees the exact same stream as before WGD existed.
        if wgd_rate > 0 and rng.random() < wgd_rate:
            event = 'wgd'
        else:
            event = rng.choice(['cnv', 'mut'], p=np.array([cnv_prob, snv_prob])/(cnv_prob + snv_prob))
        if event == 'wgd':
            # Whole-genome duplication: duplicate EVERY copy on BOTH homologs of EVERY segment once.
            # copy number is segment-granular in this genome (a copy is a whole-segment SNV bitset),
            # so a WGD needs no sub-segment structure — it just appends an independent .copy() of each
            # existing copy (mirroring the CNV amp branch, so SNVs on a copy are carried into its
            # duplicate — a WGD preserves and doubles existing mutations). Nullisomic (seg,hap) slots
            # have no copy to double and stay at zero: a WGD never rescues a lost segment. The doubled
            # seg_cns / ploidy / highest_cn / nullisomy_count all flow through the existing
            # update_genome_summary_cnv seam, and the viability gate (update_evolutionary_parameters
            # below + the engine's reject-at-birth check) drops any daughter that busts max_ploidy/max_cn.
            for seg in range(self.n_segments):
                self._cow_privatize(seg)                     # WGD writes to every segment
                for hap in ('p', 'm'):
                    existing = list(self.genome[seg][hap])   # snapshot: don't double the new copies
                    for allele_bits in existing:
                        self.genome[seg][hap].append(allele_bits.copy())  # independent copy (no aliasing)
                        self.update_genome_summary_cnv(selection, allele_bits, seg, 1)
            self.is_wgd = True
            hit_functional = True                            # WGD changes copy number -> never folded
        elif event == 'mut':
            # A dividing cell can fix several SNVs at once, scattered across the whole genome
            # rather than clustered in one segment. Each allele (physical copy) independently
            # accrues ~Poisson(n_snvs_per_allele) new SNVs, so the expected number of new SNVs
            # scales with ploidy/copy number (≈ n_snvs_per_allele × number of alleles). Within an
            # allele, positions are drawn only from sites not yet mutated on that copy (infinite-
            # sites per allele); the same locus may still be hit independently on a different copy.
            gate_bits = self.event_bits  # accessibility judged on the PRE-division event set
            placed = 0
            for seg in range(self.n_segments):
                for hap in ('p', 'm'):
                    hap_list = self.genome[seg][hap]  # parent's list; the private copy is made on write
                    for ai in range(len(hap_list)):
                        allele = hap_list[ai]         # pristine parent bitset (only ever read here)
                        k = rng.poisson(n_snvs_per_allele)
                        if k == 0:
                            continue
                        available = self._available_sites(selection, seg, allele, gate_bits)
                        if len(available) == 0:
                            continue
                        n_mutations = min(k, len(available))
                        muts = rng.choice(available, size=n_mutations, replace=False)
                        # copy-on-write: privatise the segment, then write to a copy of THIS allele so
                        # the parent's shared bitset is never mutated in place.
                        self._cow_privatize(seg)
                        new_allele = self.genome[seg][hap][ai].copy()
                        new_allele[muts] = True  # set mutated bits on this copy
                        self.genome[seg][hap][ai] = new_allele
                        mut_bits = np.zeros(self.segment_sizes[seg], dtype=bool)
                        mut_bits[muts] = True
                        # Coarsening fast-path: a purely NEUTRAL mut_bits leaves genome_summary
                        # unchanged (all category counts += 0, driver recount += 0), so skipping the
                        # update is EXACT (epistasis, the one exception, is off whenever func_mask is
                        # set). Only a functional hit updates the summary and marks a new genotype.
                        if func_mask is None or func_mask[seg][muts].any():
                            self.update_genome_summary_mutation(selection, mut_bits, seg)
                            hit_functional = True
                        placed += n_mutations
            if placed == 0:
                # The engine already decided this division mutates, so an SNV event must change
                # the genotype. Every per-allele Poisson draw came up 0 (or the rate is tiny):
                # force a single SNV on a random allele that still has an unmutated site.
                candidates = [
                    (seg, hap, ai)
                    for seg in range(self.n_segments)
                    for hap in ('p', 'm')
                    for ai, allele in enumerate(self.genome[seg][hap])
                    if len(self._available_sites(selection, seg, allele, gate_bits))
                ]
                if not candidates:
                    return False  # genome fully saturated everywhere: nothing to mutate
                seg, hap, ai = candidates[rng.choice(len(candidates))]
                allele = self.genome[seg][hap][ai]
                pos = int(rng.choice(self._available_sites(selection, seg, allele, gate_bits)))
                self._cow_privatize(seg)                     # copy-on-write before the in-place set
                new_allele = self.genome[seg][hap][ai].copy()
                new_allele[pos] = True
                self.genome[seg][hap][ai] = new_allele
                mut_bits = np.zeros(self.segment_sizes[seg], dtype=bool)
                mut_bits[pos] = True
                if func_mask is None or func_mask[seg][pos]:
                    self.update_genome_summary_mutation(selection, mut_bits, seg)
                    hit_functional = True
            self._n_new_snv = placed if placed else 1     # fallback above forces exactly one SNV
        elif event == 'cnv':
            # Sample segment
            segment_probs = np.array([len(self.genome[seg]['p'] + self.genome[seg]['m']) for seg in range(self.n_segments)]) # can't select empty segment
            segment_probs = segment_probs / np.sum(segment_probs)
            seg = rng.choice(range(self.n_segments), p=segment_probs)
            # Sample haplotype
            n_p, n_m = len(self.genome[seg]['p']), len(self.genome[seg]['m'])
            hap = rng.choice(['p', 'm'], p=np.array([n_p, n_m])/(n_p + n_m))
            # Sample allele
            all = rng.choice(range(len(self.genome[seg][hap])))
            # Decide wether to delete or copy
            evt = rng.choice(['del', 'amp'], p=[1.0 - amp_prob, amp_prob])
            self._cow_privatize(seg)                         # copy-on-write before the list append/del
            if evt == 'amp':
                sign = 1
                self.genome[seg][hap].append(self.genome[seg][hap][all].copy()) # independent copy (no aliasing)
                allele_bits = self.genome[seg][hap][all]
            elif evt == 'del':
                sign = -1
                allele_bits = self.genome[seg][hap][all].copy()  # capture before removing
                # Remove the copy. The segment may reach 0 copies (nullisomy) — a real state the
                # viability check handles. Keeping the allele but still decrementing seg_cns (the
                # old len==1 special case) drove copy number NEGATIVE; deleting keeps seg_cns equal
                # to the actual allele count. Segments with 0 copies carry zero CNV selection
                # weight, so they are never picked for further deletion.
                del self.genome[seg][hap][all]
            self.update_genome_summary_cnv(selection, allele_bits, seg, sign) # for evolutionary parameters
            hit_functional = True                            # CNV changes copy number -> never folded

        self._close_event_group(selection, events_before)
        # Coarsening: a pure-passenger SNV division (no functional hit) yields a daughter whose
        # genome_summary — hence fitness — is identical to the parent's. Signal it so the engine folds
        # it into the parent clone, skipping the CINner recompute and the new genotype id. func_mask
        # is None for every non-coarsening caller, so this branch is never taken there.
        if func_mask is not None and not hit_functional:
            return "passenger"
        self.update_evolutionary_parameters(selection)
        self.set_genotype_id()
        return True  # a new genotype was created

    def _close_event_group(self, selection, events_before):
        """Record everything this ONE division acquired as a single tied group (see event_groups)."""
        if selection.epistasis is None:
            return
        new = self.event_bits & ~events_before
        if new:
            self.event_groups = self.event_groups + (
                tuple(bits_to_events(new, selection.epistasis.n_events)),)