import numpy as np
import pandas as pd
from copy import copy, deepcopy


class Cell(object):
    # Process-wide monotonic counter for unique genotype ids. Using str(id(self)) is unsafe:
    # CPython recycles object ids after a cell is garbage-collected, so a new clone could
    # collide with a still-living genotype (seen as "genotype is its own parent").
    _genotype_counter = 0

    def __init__(
        self,
        n_segments=10,
        segment_size=1000,
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
    ):
        self.n_segments = n_segments
        self.segment_size = segment_size
        self.n_onc = n_onc
        self.n_tsg = n_tsg
        self.n_disp = n_disp
        self.n_ir = n_ir
        self.n_tr = n_tr
        self.type = "healthy"
        self.parent = parent
        if parent is None:
            # Compact genome: each allele copy is a boolean bitset of length
            # segment_size (bit i == position i is mutated, infinite-sites). A fresh
            # dict/bitset per segment avoids aliasing across segments.
            self.genome = [{'p': [np.zeros(segment_size, dtype=bool)],
                            'm': [np.zeros(segment_size, dtype=bool)]}
                           for _ in range(n_segments)] # copy number = 2
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
                                    'ploidy': 2,
                                    'highest_cn': 2,
                                    'nullisomy_count': 0,
                                    'n_mutated_drivers': 0,
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

        self.evolutionary_parameters = dict()
        self.evolutionary_parameters['division_rate'] = division_rate
        self.evolutionary_parameters['death_rate'] = death_rate
        self.evolutionary_parameters['dispersal_rate'] = dispersal_rate
        self.evolutionary_parameters['treatment_resistance'] = 0.  # wild-type: fully treatment-sensitive
        self.evolutionary_parameters['immune_resistance'] = 0.     # wild-type: no immune escape
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
            self.baseline_rates['division_rate'] * selection.update_division_rate(gs),
            self.max_birth_rate)
        self.evolutionary_parameters['death_rate'] = self.baseline_rates['death_rate']
        self.evolutionary_parameters['dispersal_rate'] = min(
            self.baseline_rates['dispersal_rate'] * selection.update_dispersal_rate(gs), 1.0)
        self.evolutionary_parameters['immune_resistance'] = max(
            0.0, 1.0 - 1.0 / selection.update_immune_resistance(gs))
        self.evolutionary_parameters['treatment_resistance'] = max(
            0.0, 1.0 - 1.0 / selection.update_treatment_resistance(gs))

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

        # TODO: Maybe increase number of mutated drivers

    def update_genome_summary_cnv(self, selection, allele_bits, seg, sign):
        # A CNV adds/removes one allele copy of segment `seg`. `allele_bits` is that
        # allele's bitset; every driver position on it changes copy number by `sign`.
        # Wild-type copies that change = category positions in the segment minus the
        # mutated ones on this allele.
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

        self.genome_summary['seg_cns'][seg] += sign
        seg_cns = self.genome_summary['seg_cns']
        self.genome_summary['ploidy'] = np.mean(seg_cns)
        self.genome_summary['highest_cn'] = np.max(seg_cns)
        # nullisomy = number of segments with zero copies, not the largest copy number.
        self.genome_summary['nullisomy_count'] = int(np.sum(np.asarray(seg_cns) == 0))

        # TODO: maybe reduce the number of unique drivers...

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
        pos_vec = np.arange(self.segment_size)
        for seg in range(self.n_segments):
            seg_vec = seg * np.ones((self.segment_size))
            for hap in self.genome[seg]:
                hap_vec = np.array([hap] * self.segment_size)
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
        vafs = np.zeros((self.n_segments * self.segment_size,))
        for seg in range(self.n_segments):
            copies = self.genome[seg]['p'] + self.genome[seg]['m']
            cn = self.genome_summary['seg_cns'][seg]
            if copies and cn > 0:
                counts = np.sum(copies, axis=0)  # mutated-copy count per position
                vafs[seg*self.segment_size:(seg+1)*self.segment_size] = counts / cn
        return vafs

    def get_cnvs(self):
        cnvs = []
        for seg_cn in self.genome_summary['seg_cns']:
            cnvs.append([seg_cn] * self.segment_size)
        cnvs = np.array(cnvs).flatten()
        return cnvs   

    def set_baseline_exp(self):
        self.baseline_exp = np.random.beta(.1, 1, size=self.n_segments * self.segment_size) 

    def get_exp(self, seg_mut_effects):
        exp = np.array(self.baseline_exp)
        for seg in range(self.n_segments):
            seg_baseline = self.baseline_exp[seg*self.segment_size:(seg+1)*self.segment_size]
            seg_exp = np.array(seg_baseline)
            copies = self.genome[seg]['p'] + self.genome[seg]['m']
            if len(copies) == 0:
                seg_exp = seg_exp * 0.
            else:
                for bits in copies:
                    allele_contrib = seg_baseline * seg_mut_effects[seg] ** bits.astype(float)
                    seg_exp = seg_exp + allele_contrib
            exp[seg*self.segment_size:(seg+1)*self.segment_size] = seg_exp
        return exp

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
        gene_base = self.baseline_exp[seg * self.segment_size + pos]
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

class CancerCell(Cell):
    def __init__(self, mutation_rate=0.1, snv_prob=0.1, genotype_id=None, seed=42, **cell_kwargs):
        super(CancerCell, self).__init__(**cell_kwargs)
        self.type = "cancer"
        self.mutation_rate = mutation_rate
        self.seed  = seed
        self.set_params()

    def set_params(self):
        if self.parent is not None:
            self.genotype_id = self.parent.genotype_id
            
    def mutate(self, rng, selection, n_events=5, mut_prob=.1, cnv_prob=.1):
        # Copy-on-write: this cell is diverging into a new genotype, so take a private
        # copy of the (until now possibly shared) genome/summary before modifying them.
        # This is the only place the genome is mutated, so sharing elsewhere is safe.
        self.genome = deepcopy(self.genome)
        self.genome_summary = deepcopy(self.genome_summary)
        event = rng.choice(['cnv', 'mut'], p=np.array([mut_prob, cnv_prob])/sum([mut_prob, cnv_prob])) # add WGDs too...
        if event == 'mut':
            # Sample segment
            segment_probs = np.array([len(self.genome[seg]['p'] + self.genome[seg]['m']) for seg in range(self.n_segments)]) # can't select empty segment
            segment_probs = segment_probs / np.sum(segment_probs)
            seg = rng.choice(range(self.n_segments), p=segment_probs)
            # Sample haplotype
            n_p, n_m = len(self.genome[seg]['p']), len(self.genome[seg]['m'])
            hap = rng.choice(['p', 'm'], p=np.array([n_p, n_m])/(n_p + n_m))
            # Sample allele
            all = rng.choice(range(len(self.genome[seg][hap])))
            allele = self.genome[seg][hap][all]
            # Sample only from positions not yet mutated in this allele (ISA)
            available = np.where(~allele)[0]
            if len(available) == 0:
                return False  # allele fully saturated: no new mutation, genotype unchanged
            n_mutations = min(rng.poisson(n_events) + 1, len(available))
            muts = rng.choice(available, size=n_mutations, replace=False)
            allele[muts] = True  # set mutated bits
            mut_bits = np.zeros(self.segment_size, dtype=bool)
            mut_bits[muts] = True
            self.update_genome_summary_mutation(selection, mut_bits, seg) # for evolutionary parameters
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
            evt = rng.choice(['del', 'amp'])
            if evt == 'amp':
                sign = 1
                self.genome[seg][hap].append(self.genome[seg][hap][all].copy()) # independent copy (no aliasing)
                allele_bits = self.genome[seg][hap][all]
            elif evt == 'del':
                sign = -1
                allele_bits = self.genome[seg][hap][all].copy()  # capture before removing
                if len(self.genome[seg][hap]) == 1:
                    self.genome[seg][hap][all] = np.zeros(self.segment_size, dtype=bool)
                else:
                    del self.genome[seg][hap][all] # remove
            self.update_genome_summary_cnv(selection, allele_bits, seg, sign) # for evolutionary parameters

        self.update_evolutionary_parameters(selection)
        self.set_genotype_id()
        return True  # a new genotype was created