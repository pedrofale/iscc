import numpy as np
import pandas as pd

class Selection(object):
    def __init__(self, n_segments=10, segment_size=1000,
                 prop_driver=0.1, prop_dispersal=0.1, prop_treatment_resistance=0.1, prop_immune_resistance=0.1,
                 driver_effects=1.1, dispersal_effects=1.1, treatment_resistant_effects=1.1, immune_resistant_effects=1.1,
                 max_ploidy=6, max_cn=12, max_nullisomy=2, max_mut_drivers=1000, rng=None, ):
        # Seeded generator so the driver/resistance layout is reproducible.
        self.rng = rng if rng is not None else np.random.default_rng()
        # Fixed about the genome
        self.n_segments = n_segments
        self.segment_size = segment_size
        self.n_genes = n_segments * segment_size
        self.prop_driver = prop_driver
        self.prop_dispersal = prop_dispersal
        self.prop_treatment_resistance = prop_treatment_resistance
        self.prop_immune_resistance = prop_immune_resistance
        
        # Fixed about fitness
        self.driver_effects = driver_effects
        self.dispersal_effects = dispersal_effects
        self.treatment_resistant_effects = treatment_resistant_effects
        self.immune_resistant_effects = immune_resistant_effects

        # Fixed about viability
        self.max_ploidy = max_ploidy
        self.max_cn = max_cn
        self.max_nullisomy = max_nullisomy
        self.max_mut_drivers = max_mut_drivers

        self.drivers = []
        self.passengers = []

        # Put drivers and passengers in position
        self.make_drivers()
        self.make_dispersal()
        self.make_treatment_resistant()
        self.make_immune_resistant()
        self.make_expmap()
        self.update_dict = {'viability': self.update_viability,
                            'division_rate': self.update_division_rate,
                            'dispersal_rate': self.update_dispersal_rate,
                            'immune_resistance': self.update_immune_resistance,
                            'treatment_resistance': self.update_treatment_resistance,
                            'death_rate': self.update_death_rate,}
        
        self.gene_names = self.get_gene_names()

    def get_evolutionary_parameters(self):
        return list(self.update_dict.keys())

    def make_drivers(self): 
        # supressor, notdriver, oncogene
        self.drivers = []
        self.passengers = []
        self.driver_types = []
        self.onc = []
        self.tsg = []
        for _ in range(self.n_segments):
            driver_types = self.rng.choice([-1,0,1], p=[self.prop_driver/2, 1.-self.prop_driver, self.prop_driver/2],
                                                size=self.segment_size)
            self.drivers.append(np.where(driver_types!=0)[0])
            self.passengers.append(np.where(driver_types==0)[0])
            self.driver_types.append(driver_types)
            self.onc.append(np.where(driver_types==1)[0])
            self.tsg.append(np.where(driver_types==-1)[0])

    def make_dispersal(self): # select sites that if mutated make the cell less more likely to attempt dispersal
        self.dispersal_types = []
        self.dispersal = []
        for _ in range(self.n_segments):
            confers_dispersal = self.rng.binomial(1, self.prop_dispersal, size=self.segment_size)
            self.dispersal_types.append(confers_dispersal)
            self.dispersal.append(np.where(confers_dispersal==1)[0])

    def make_treatment_resistant(self): # select sites that if mutated make the cell less likely to respond to treatment
        self.treatment_resistance_types = []
        self.treatment_resistance = []
        for _ in range(self.n_segments):
            confers_resistance = self.rng.binomial(1, self.prop_treatment_resistance, size=self.segment_size)
            self.treatment_resistance_types.append(confers_resistance)
            self.treatment_resistance.append(np.where(confers_resistance==1)[0])

    def make_immune_resistant(self): # select sites that if mutated make the cell hide from immune system
        self.immune_resistance_types = []
        self.immune_resistance = []
        for _ in range(self.n_segments):
            confers_resistance = self.rng.binomial(1, self.prop_immune_resistance, size=self.segment_size)
            self.immune_resistance_types.append(confers_resistance) 
            self.immune_resistance.append(np.where(confers_resistance==1)[0])           

    def make_expmap(self):
        # Effect of snv on exp: up or down depending on wether tsg or og, and also if immune resistance-inducing, overexpress
        self.mut_effects = []
        for seg in range(self.n_segments):
            mut_effects = np.ones((self.segment_size,)) # in general, mutation has no effect
            mut_effects[np.where(self.driver_types[seg] == 1)] = 2. # if mutated og, increase exp
            mut_effects[np.where(self.driver_types[seg] == -1)] = 0.5 # if mutated tsg, decrease exp
            mut_effects[np.where(self.immune_resistance_types[seg] == 1)] = 2. # if mutated immune resistance, increase exp
            self.mut_effects.append(mut_effects)

    def get_tsgs(self):
        tsgs = []
        for seg in range(self.n_segments):
            idx = np.where(self.driver_types[seg] == -1)[0]
            tsgs.append(idx + seg*self.segment_size)
        tsgs = np.concatenate(tsgs)
        return tsgs

    def get_oncogenes(self):
        tsgs = []
        for seg in range(self.n_segments):
            idx = np.where(self.driver_types[seg] == 1)[0]
            tsgs.append(idx + seg*self.segment_size)
        tsgs = np.concatenate(tsgs)
        return tsgs
    
    def get_dispersal_genes(self):
        tsgs = []
        for seg in range(self.n_segments):
            idx = np.where(self.dispersal_types[seg] == 1)[0]
            tsgs.append(idx + seg*self.segment_size)
        tsgs = np.concatenate(tsgs)
        return tsgs    
    
    def get_immune_resistant(self):
        tsgs = []
        for seg in range(self.n_segments):
            idx = np.where(self.immune_resistance_types[seg] == 1)[0]
            tsgs.append(idx + seg*self.segment_size)
        tsgs = np.concatenate(tsgs)
        return tsgs    
    
    def get_treatment_resistant(self):
        tsgs = []
        for seg in range(self.n_segments):
            idx = np.where(self.treatment_resistance_types[seg] == 1)[0]
            tsgs.append(idx + seg*self.segment_size)
        tsgs = np.concatenate(tsgs)
        return tsgs        

    # def update_viability(self, genome):
    #     seg_cns = []
    #     n_mutated_drivers = 0
    #     for seg in range(len(genome)):
    #         cn = 0
    #         for hap in genome[seg]:
    #             cn += len(genome[seg][hap]) # number of alleles
    #             mutated_drivers = set()
    #             for all in genome[seg][hap]:
    #                 mutated_drivers.update(all.intersection(self.drivers[seg])) # number of mutated drivers -- use set to ignore copies
    #             n_mutated_drivers += len(mutated_drivers)
    #         seg_cns.append(cn)
                    
    #     ploidy = np.mean(seg_cns) # average copy number across segments
    #     if ploidy > self.max_ploidy:
    #         return 0
    #     highest_cn = np.max(seg_cns) # maximum copy number of any segment
    #     if highest_cn > self.max_cn:
    #         return 0 
    #     nullisomy_count = np.sum(seg_cns == 0) # number of segments with zero copies
    #     if nullisomy_count > self.max_nullisomy:
    #         return 0
    #     if n_mutated_drivers > self.max_mut_drivers:
    #         return 0
    #     return 1
    
    def update_viability(self, genome_summary, **kwargs):
        if genome_summary['ploidy'] > self.max_ploidy:
            return 0
        if genome_summary['highest_cn'] > self.max_cn:
            return 0 
        if genome_summary['nullisomy_count'] > self.max_nullisomy:
            return 0
        if genome_summary['n_mutated_drivers'] > self.max_mut_drivers:
            return 0
        return 1        

    def update_death_rate(self, genome_summary, param, **kwargs):
        return param

    def update_division_rate(self, genome_summary, **kwargs):
        n_wt_tsg = genome_summary['n_wt_tsg']
        n_mut_tsg = genome_summary['n_mut_tsg']
        n_wt_onc = genome_summary['n_wt_onc']
        n_mut_onc = genome_summary['n_mut_onc']
        ploidy = genome_summary['ploidy']
        # Following CINer
        tsg_wt_effect = 1./self.driver_effects
        tsg_mut_effect = 1.
        og_wt_effect = self.driver_effects
        og_mut_effect = self.driver_effects**2
        tsg = tsg_wt_effect**(2*n_wt_tsg/ploidy) * tsg_mut_effect**(2*n_mut_tsg/ploidy)
        og = og_wt_effect**(2*n_wt_onc/ploidy) * og_mut_effect**(2*n_mut_onc/ploidy)
        return tsg * og

    def update_dispersal_rate(self, genome_summary, **kwargs):
        n_wt_disp = genome_summary['n_wt_disp']
        n_mut_disp = genome_summary['n_mut_disp']
        ploidy = genome_summary['ploidy']        
        wt_effect = self.dispersal_effects
        mut_effect = self.dispersal_effects**2
        return wt_effect**(2*n_wt_disp/ploidy) * mut_effect**(2*n_mut_disp/ploidy)

    def update_immune_resistance(self, genome_summary, **kwargs):
        n_wt_ir = genome_summary['n_wt_ir']
        n_mut_ir = genome_summary['n_mut_ir']
        ploidy = genome_summary['ploidy']                
        wt_effect = self.immune_resistant_effects
        mut_effect = self.immune_resistant_effects**2
        return wt_effect**(2*n_wt_ir/ploidy) * mut_effect**(2*n_mut_ir/ploidy)

    def update_treatment_resistance(self, genome_summary, **kwargs):
        n_wt_tr = genome_summary['n_wt_tr']
        n_mut_tr = genome_summary['n_mut_tr']
        ploidy = genome_summary['ploidy']        
        wt_effect = self.treatment_resistant_effects
        mut_effect = self.treatment_resistant_effects**2
        return wt_effect**(2*n_wt_tr/ploidy) * mut_effect**(2*n_mut_tr/ploidy)

    # def _update_division_rate(self, baseline_fitness, genome):
    #     # Make it sensible to copy number
    #     fitness = baseline_fitness
    #     for seg in range(len(genome)):
    #         for hap in genome[seg]:
    #             for all in genome[seg][hap]:
    #                 fitness += self.driver_effects * len(all.intersection(self.drivers[seg])) # more mutated copies of a gene shouldn't make a big difference though
    #     fitness = np.min([1., fitness])                    
    #     return fitness

    # def update_dispersal_rate(self, baseline_fitness, genome):
    #     # Make it sensible to copy number
    #     fitness = baseline_fitness
    #     for seg in range(len(genome)):
    #         for hap in genome[seg]:
    #             for all in genome[seg][hap]:
    #                 fitness += self.driver_effects * len(all.intersection(self.drivers[seg])) # more mutated copies of a gene shouldn't make a big difference though
    #     fitness = np.min([1., fitness])
    #     return fitness
    
    # def update_immune_resistance(self, genome):
    #     # Make it sensible to copy number
    #     for seg in range(len(genome)):
    #         for hap in genome[seg]:
    #             for all in genome[seg][hap]:
    #                 fitness += self.driver_effects * len(all.intersection(self.drivers[seg])) # more mutated copies of a gene shouldn't make a big difference though
    #     fitness = np.min([1., fitness])                    
    #     return fitness

    # def update_treatment_resistance(self, baseline_fitness, genome):
    #     # Make it sensible to copy number
    #     fitness = baseline_fitness
    #     for seg in range(len(genome)):
    #         for hap in genome[seg]:
    #             for all in genome[seg][hap]:
    #                 fitness += self.driver_effects * len(all.intersection(self.drivers[seg])) # more mutated copies of a gene shouldn't make a big difference though
    #     fitness = np.min([1., fitness])                    
    #     return fitness

    # def update_evolutionary_parameters(self, param_dict, genome):
    #     # Make it sensible to copy number
    #     for seg in range(len(genome)):
    #         for hap in genome[seg]:
    #             for all in genome[seg][hap]:
    #                 for param in param_dict:
    #                     param_dict[param] = self.driver_effects * len(all.intersection(self.drivers[seg])) # more mutated copies of a gene shouldn't make a big difference though
    #                     param_dict[param] = min(1., param_dict[param])
    #     return param_dict
        
    def get_gene_names(self, gene_prefix='G'):
        gene_names = []
        for segment in range(self.n_segments):
            gn = [f'{gene_prefix}_{segment}_{i}' for i in range(self.segment_size)]
            gene_names.extend(gn)
        return gene_names

    def gene_to_pos(self, gene_name):
        # Gene names are built by get_gene_names as f'{prefix}_{segment}_{pos}'.
        prefix, seg, pos = gene_name.split('_')
        return int(seg), int(pos)

    def get_gene_data(self, **kwargs):
        gene_names = self.get_gene_names(**kwargs)
        driver_types = np.concatenate(self.driver_types)
        dispersal_types = np.concatenate(self.dispersal_types)
        treatment_resistance_types = np.concatenate(self.treatment_resistance_types)
        immune_resistance_types = np.concatenate(self.immune_resistance_types)
        return dict(driver_types=pd.DataFrame(driver_types, index=gene_names), 
                    dispersal_types=pd.DataFrame(dispersal_types, index=gene_names),
                    treatment_resistance_types=pd.DataFrame(treatment_resistance_types, index=gene_names),
                    immune_resistance_types=pd.DataFrame(immune_resistance_types, index=gene_names))