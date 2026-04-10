import numpy as np
import pandas as pd
from copy import copy, deepcopy


class Cell(object):
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
            self.genome = [{'p':[set()], 'm':[set()]}] * n_segments # copy number = 2
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
      
        # self.exp = np.random.beta(.1, 1, size=self.n_genes)  # Gene activity probability (each gene ranges from 0 to 1 indicating its prob of expression. transcripts will be sampled binomially)

        self.max_birth_rate = max_birth_rate
        self.baseline_treatment_resistance = 1.-death_rate

        self.evolutionary_parameters = dict()
        self.evolutionary_parameters['division_rate'] = division_rate
        self.evolutionary_parameters['death_rate'] = death_rate
        self.evolutionary_parameters['dispersal_rate'] = dispersal_rate
        self.evolutionary_parameters['treatment_resistance'] = self.baseline_treatment_resistance
        self.evolutionary_parameters['immune_resistance'] = 0.
        self.evolutionary_parameters['viability'] = 1.      

    def update_evolutionary_parameters(self, update_dict):
        for evo_param in self.evolutionary_parameters:
            self.evolutionary_parameters[evo_param] = update_dict[evo_param](self.genome_summary, param=self.evolutionary_parameters[evo_param])

    def update_genome_summary_mutation(self, selection, muts, seg):
        n_new_onc = sum(muts.intersection(selection.onc[seg]))
        self.genome_summary['n_mut_onc'] += n_new_onc
        self.genome_summary['n_wt_onc'] -= n_new_onc

        n_new_tsg = sum(muts.intersection(selection.tsg[seg]))
        self.genome_summary['n_mut_tsg'] += n_new_tsg
        self.genome_summary['n_wt_tsg'] -= n_new_tsg

        n_new_disp = sum(muts.intersection(selection.dispersal[seg]))
        self.genome_summary['n_mut_disp'] += n_new_disp
        self.genome_summary['n_wt_disp'] -= n_new_disp

        n_new_ir = sum(muts.intersection(selection.immune_resistance[seg]))
        self.genome_summary['n_mut_ir'] += n_new_ir
        self.genome_summary['n_wt_ir'] -= n_new_ir

        n_new_tr = sum(muts.intersection(selection.treatment_resistance[seg]))
        self.genome_summary['n_mut_tr'] += n_new_tr
        self.genome_summary['n_wt_tr'] -= n_new_tr

        # TODO: Maybe increase number of mutated drivers

    def update_genome_summary_cnv(self, selection, muts, seg, sign):
        n_mut_onc = sum(muts.intersection(selection.onc[seg]))
        n_wt_onc = self.segment_size - n_mut_onc
        self.genome_summary['n_mut_onc'] += sign*n_mut_onc
        self.genome_summary['n_wt_onc'] += sign*n_wt_onc

        n_mut_tsg = sum(muts.intersection(selection.tsg[seg]))
        n_wt_tsg = self.segment_size - n_mut_tsg
        self.genome_summary['n_mut_tsg'] += sign*n_mut_tsg
        self.genome_summary['n_wt_tsg'] += sign*n_wt_tsg

        n_mut_disp = sum(muts.intersection(selection.dispersal[seg]))
        n_wt_disp = self.segment_size - n_mut_disp
        self.genome_summary['n_mut_disp'] += sign*n_mut_disp
        self.genome_summary['n_wt_disp'] += sign*n_wt_disp

        n_mut_ir = sum(muts.intersection(selection.immune_resistance[seg]))
        n_wt_ir = self.segment_size - n_mut_ir
        self.genome_summary['n_mut_ir'] += sign*n_mut_ir
        self.genome_summary['n_wt_ir'] += sign*n_wt_ir

        n_mut_tr = sum(muts.intersection(selection.treatment_resistance[seg]))
        n_wt_tr = self.segment_size - n_mut_tr
        self.genome_summary['n_mut_tr'] += sign*n_mut_tr
        self.genome_summary['n_wt_tr'] += sign*n_wt_tr

        self.genome_summary['seg_cns'][seg] += sign
        seg_cns = self.genome_summary['seg_cns']
        self.genome_summary['ploidy'] = np.mean(seg_cns)
        self.genome_summary['highest_cn'] = np.max(seg_cns)
        self.genome_summary['nullisomy_count'] = np.max(seg_cns)

        # TODO: maybe reduce the number of unique drivers...

    def set_genotype_id(self):
        self.genotype_id = str(id(self))

    def divide(self, new_cell_id=0):
        new_cell = copy(self)
        new_cell.parent = self
        new_cell.genome = deepcopy(self.genome)
        new_cell.genome_summary = deepcopy(self.genome_summary)
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
                for all in self.genome[seg][hap]: # self.genome[seg][hap] is a list of sets [set(), set(), set()], so all is a set
                    mut_vec = np.zeros((self.segment_size,)).astype(int)
                    if len(all) > 0:
                        mut_vec[np.array(list(all))] = 1
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
            for hap in self.genome[seg]: 
                for all in self.genome[seg][hap]:
                    for mut in all:
                        vafs[mut + seg*self.segment_size] += 1
            vafs[seg*self.segment_size:(seg+1)*self.segment_size] /= self.genome_summary['seg_cns'][seg]
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
            if len(self.genome[seg]['p']) + len(self.genome[seg]['m']) == 0:
                seg_exp *= 0.
            else:
                for hap in self.genome[seg]:
                    for all in range(len(self.genome[seg][hap])):
                        seg_mut_status = np.zeros((self.segment_size,))
                        muts = self.genome[seg][hap][all]
                        if len(muts) > 0:
                            seg_mut_status[np.array(list(muts))] = 1.
                        allele_contrib = seg_baseline * seg_mut_effects[seg]**seg_mut_status
                        seg_exp += allele_contrib
            exp[seg*self.segment_size:(seg+1)*self.segment_size] = seg_exp
        return exp

    def expresses(self, coordinates, seg_mut_effects, thres=0.5):
        # coordinates: tuple (seg, pos)
        seg, pos = coordinates # unpack
        if len(self.genome[seg]['p']) + len(self.genome[seg]['m']) == 0:
            return False
        
        # Go to the genomic location, and compute its expression directly without having to cycle through the whole genome to compute total expression profile
        gene_exp = self.baseline_exp[seg+pos]
        for hap in self.genome[seg]:
            for all in range(len(self.genome[seg][hap])):
                mut = self.genome[seg][hap][all][pos]
                allele_contrib = gene_exp * seg_mut_effects[seg+pos]**mut
                gene_exp += allele_contrib
        
        if gene_exp > thres:
            return True
        else:
            return False
        

class EpithelialCell(Cell):
    def __init__(self, **cell_kwargs):
        super(EpithelialCell, self).__init__(**cell_kwargs)
        self.type = "epithelial"
        self.genotype_id = self.type
        # self.exp = np.random.beta(.1, 1, size=self.n_genes)  # Gene activity probability (each gene ranges from 0 to 1 indicating its prob of expression. transcripts will be sampled binomially)

class StromalCell(Cell):
    def __init__(self, **cell_kwargs):
        super(StromalCell, self).__init__(**cell_kwargs)
        self.type = "stromal"        
        self.genotype_id = self.type
        # self.exp = np.random.beta(.1, 1, size=self.n_genes)  # Gene activity probability (each gene ranges from 0 to 1 indicating its prob of expression. transcripts will be sampled binomially)

class ImmuneCell(Cell):
    def __init__(self, prob_kill=.01, **cell_kwargs):
        super(ImmuneCell, self).__init__(**cell_kwargs)
        self.prob_kill = prob_kill # probability of killing a neighboring cancer cell
        self.type = "immune"
        self.genotype_id = self.type
        # self.exp = np.random.beta(.1, 1, size=self.n_genes)  # Gene activity probability (each gene ranges from 0 to 1 indicating its prob of expression. transcripts will be sampled binomially)

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
            # Sample only from positions not yet mutated in this allele (ISA)
            available = np.array(
                list(set(range(self.segment_size)) - self.genome[seg][hap][all])
            )
            if len(available) == 0:
                return  # allele fully saturated; skip this mutation event
            n_mutations = min(rng.poisson(n_events) + 1, len(available))
            muts = set(rng.choice(available, size=n_mutations, replace=False))
            self.genome[seg][hap][all].update(muts) # for actual genotype and expression tracking
            self.update_genome_summary_mutation(selection, muts, seg) # for evolutionary parameters
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
                self.genome[seg][hap].append(self.genome[seg][hap][all]) # add a copy
                muts = self.genome[seg][hap][all]
            elif evt == 'del':
                sign = -1
                muts = self.genome[seg][hap][all]
                if len(self.genome[seg][hap]) == 1:
                    self.genome[seg][hap][all] = set()
                else:
                    del self.genome[seg][hap][all] # remove
            self.update_genome_summary_cnv(selection, muts, seg, sign) # for evolutionary parameters

        self.update_evolutionary_parameters(selection.update_dict)
        self.set_genotype_id()