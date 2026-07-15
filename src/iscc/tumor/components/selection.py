import numpy as np
import pandas as pd

from .epistasis import EpistasisNetwork
from ...constants import DEFAULT_LAYOUT_SEED, LAYOUT_OFFSET_EPISTASIS

class Selection(object):
    def __init__(self, n_segments=10, segment_size=1000, segment_sizes=None,
                 prop_driver=0.1, prop_dispersal=0.1, prop_treatment_resistance=0.1, prop_immune_resistance=0.1,
                 driver_effects=1.1, dispersal_effects=1.1, treatment_resistant_effects=1.1, immune_resistant_effects=1.1,
                 selection_mode="gene", s_arm=None, arm_baseline=2.0,
                 max_ploidy=6, max_cn=12, max_nullisomy=2, max_mut_drivers=1000, rng=None,
                 epistasis_params=None, dependency_params=None, layout_seed=None, ):
        # Seeded generator so the driver/resistance layout is reproducible. This ``rng`` is used
        # ONLY for the config-determined gene-role LAYOUT (make_drivers / make_dispersal /
        # make_treatment_resistant / make_immune_resistant), never for evolution — so the engines
        # hand it a dedicated LAYOUT rng seeded by a config-determined ``layout_seed`` (shared across
        # same-config runs), decoupled from the per-run evolution seed. See DESIGN_cohort.md §1.
        self.rng = rng if rng is not None else np.random.default_rng()
        # Fixed about the genome. Segments may have unequal sizes (real-genome mode: size
        # proportional to chromosome-arm length); ``segment_sizes`` overrides the uniform scalar.
        self.n_segments = n_segments
        self.segment_size = segment_size
        if segment_sizes is None:
            segment_sizes = [segment_size] * n_segments
        self.segment_sizes = [int(s) for s in segment_sizes]
        self._seg_offsets = np.concatenate([[0], np.cumsum(self.segment_sizes)]).astype(int)
        self.n_genes = int(self._seg_offsets[-1])
        self.prop_driver = prop_driver
        self.prop_dispersal = prop_dispersal
        self.prop_treatment_resistance = prop_treatment_resistance
        self.prop_immune_resistance = prop_immune_resistance

        # Fixed about fitness
        self.driver_effects = driver_effects
        self.dispersal_effects = dispersal_effects
        self.treatment_resistant_effects = treatment_resistant_effects
        self.immune_resistant_effects = immune_resistant_effects

        # Selection model. "gene" (default) = the abstract CINner gene-driver model
        # (oncogene/TSG mutation + copy-number fitness via n_wt/n_mut counts). "arm" = the
        # real-genome per-arm copy-number model (CINner's arm model): division fitness is
        # prod_seg s_arm[seg] ** (seg_cns[seg] - arm_baseline), read directly from the
        # per-segment copy numbers iscc already maintains. s_arm[seg] > 1 -> amplifying that
        # arm is beneficial (oncogene-dominated arm); s_arm[seg] < 1 -> deleting it is.
        self.selection_mode = selection_mode
        self.arm_baseline = arm_baseline
        if s_arm is None:
            s_arm = np.ones(n_segments)
        self.s_arm = np.asarray(s_arm, dtype=float)
        if self.s_arm.shape[0] != n_segments:
            raise ValueError(f"s_arm length {self.s_arm.shape[0]} != n_segments {n_segments}")
        self._log_s_arm = np.log(self.s_arm)

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

        # Total number of genes in each category, used to make fitness *relative* to the
        # all-wild-type diploid baseline (so the baseline is neutral and only deviations
        # -- mutations and copy-number changes -- shift the rate; see update_* below).
        self.N_onc = sum(len(x) for x in self.onc)
        self.N_tsg = sum(len(x) for x in self.tsg)
        self.N_disp = sum(len(x) for x in self.dispersal)
        self.N_ir = sum(len(x) for x in self.immune_resistance)
        self.N_tr = sum(len(x) for x in self.treatment_resistance)
        self.update_dict = {'viability': self.update_viability,
                            'division_rate': self.update_division_rate,
                            'dispersal_rate': self.update_dispersal_rate,
                            'immune_resistance': self.update_immune_resistance,
                            'treatment_resistance': self.update_treatment_resistance,
                            'death_rate': self.update_death_rate,}
        
        self.gene_names = self.get_gene_names()

        # Epistasis / dependency network (R14, DESIGN_epistasis.md). OFF BY DEFAULT: with no
        # ``epistasis_params`` (or ``n_events=0``) nothing is built, ``self.epistasis`` stays None and
        # every fitness path below is bit-identical to the additive model. When ON, the network is
        # drawn from a DEDICATED layout SUB-STREAM (``layout_seed + LAYOUT_OFFSET_EPISTASIS``) rather
        # than from ``self.rng``: the network is part of the SHARED landscape (every patient of a
        # cohort must evolve under the SAME network for MHN/TreeMHN to be well-posed), and using a
        # sub-stream means turning epistasis on — or changing n_interactions/topology — leaves the
        # gene-role layout drawn from ``self.rng`` above untouched.
        self.layout_seed = DEFAULT_LAYOUT_SEED if layout_seed is None else layout_seed
        self.epistasis_params = epistasis_params
        self.dependency_params = dependency_params
        self.epistasis = None
        if epistasis_params and int(epistasis_params.get("n_events", 0)) > 0:
            driver_pool = np.concatenate([self.get_oncogenes(), self.get_tsgs()])
            self.epistasis = EpistasisNetwork(
                driver_pool=driver_pool, seg_offsets=self._seg_offsets,
                segment_sizes=self.segment_sizes,
                rng=np.random.default_rng(self.layout_seed + LAYOUT_OFFSET_EPISTASIS),
                epistasis_params=epistasis_params, dependency_params=dependency_params)

    def get_evolutionary_parameters(self):
        return list(self.update_dict.keys())

    def make_drivers(self): 
        # supressor, notdriver, oncogene
        self.drivers = []
        self.passengers = []
        self.driver_types = []
        self.onc = []
        self.tsg = []
        for seg in range(self.n_segments):
            driver_types = self.rng.choice([-1,0,1], p=[self.prop_driver/2, 1.-self.prop_driver, self.prop_driver/2],
                                                size=self.segment_sizes[seg])
            self.drivers.append(np.where(driver_types!=0)[0])
            self.passengers.append(np.where(driver_types==0)[0])
            self.driver_types.append(driver_types)
            self.onc.append(np.where(driver_types==1)[0])
            self.tsg.append(np.where(driver_types==-1)[0])

    def make_dispersal(self): # select sites that if mutated make the cell less more likely to attempt dispersal
        self.dispersal_types = []
        self.dispersal = []
        for seg in range(self.n_segments):
            confers_dispersal = self.rng.binomial(1, self.prop_dispersal, size=self.segment_sizes[seg])
            self.dispersal_types.append(confers_dispersal)
            self.dispersal.append(np.where(confers_dispersal==1)[0])

    def make_treatment_resistant(self): # select sites that if mutated make the cell less likely to respond to treatment
        self.treatment_resistance_types = []
        self.treatment_resistance = []
        for seg in range(self.n_segments):
            confers_resistance = self.rng.binomial(1, self.prop_treatment_resistance, size=self.segment_sizes[seg])
            self.treatment_resistance_types.append(confers_resistance)
            self.treatment_resistance.append(np.where(confers_resistance==1)[0])

    def make_immune_resistant(self): # select sites that if mutated make the cell hide from immune system
        self.immune_resistance_types = []
        self.immune_resistance = []
        for seg in range(self.n_segments):
            confers_resistance = self.rng.binomial(1, self.prop_immune_resistance, size=self.segment_sizes[seg])
            self.immune_resistance_types.append(confers_resistance) 
            self.immune_resistance.append(np.where(confers_resistance==1)[0])           

    def make_expmap(self):
        # Effect of snv on exp: up or down depending on wether tsg or og, and also if immune resistance-inducing, overexpress
        self.mut_effects = []
        for seg in range(self.n_segments):
            mut_effects = np.ones((self.segment_sizes[seg],)) # in general, mutation has no effect
            mut_effects[np.where(self.driver_types[seg] == 1)] = 2. # if mutated og, increase exp
            mut_effects[np.where(self.driver_types[seg] == -1)] = 0.5 # if mutated tsg, decrease exp
            mut_effects[np.where(self.immune_resistance_types[seg] == 1)] = 2. # if mutated immune resistance, increase exp
            self.mut_effects.append(mut_effects)

    def get_tsgs(self):
        tsgs = []
        for seg in range(self.n_segments):
            idx = np.where(self.driver_types[seg] == -1)[0]
            tsgs.append(idx + self._seg_offsets[seg])
        tsgs = np.concatenate(tsgs)
        return tsgs

    def get_oncogenes(self):
        tsgs = []
        for seg in range(self.n_segments):
            idx = np.where(self.driver_types[seg] == 1)[0]
            tsgs.append(idx + self._seg_offsets[seg])
        tsgs = np.concatenate(tsgs)
        return tsgs

    def get_dispersal_genes(self):
        tsgs = []
        for seg in range(self.n_segments):
            idx = np.where(self.dispersal_types[seg] == 1)[0]
            tsgs.append(idx + self._seg_offsets[seg])
        tsgs = np.concatenate(tsgs)
        return tsgs

    def get_immune_resistant(self):
        tsgs = []
        for seg in range(self.n_segments):
            idx = np.where(self.immune_resistance_types[seg] == 1)[0]
            tsgs.append(idx + self._seg_offsets[seg])
        tsgs = np.concatenate(tsgs)
        return tsgs

    def get_treatment_resistant(self):
        tsgs = []
        for seg in range(self.n_segments):
            idx = np.where(self.treatment_resistance_types[seg] == 1)[0]
            tsgs.append(idx + self._seg_offsets[seg])
        tsgs = np.concatenate(tsgs)
        return tsgs

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

    def _rel_fitness(self, n_wt, n_mut, ploidy, n_total, wt_effect, mut_effect):
        """CINner-style multiplicative fitness for one gene category, expressed RELATIVE to
        the all-wild-type diploid baseline (n_wt = 2*n_total, n_mut = 0, ploidy = 2).

        Returns exp(logF(genome) - logF(baseline)) so that the baseline genome maps to 1 and
        only deviations (mutations, copy-number gains/losses) move it. Computed in log space
        to avoid the overflow of the absolute form (e.g. 1.1**200)."""
        if ploidy <= 0 or n_total == 0:
            return 1.0
        log_wt, log_mut = np.log(wt_effect), np.log(mut_effect)
        log_f = (2 * n_wt / ploidy) * log_wt + (2 * n_mut / ploidy) * log_mut
        log_base = 2 * n_total * log_wt          # baseline: 2*n_total wild-type copies, no mutations
        return float(np.exp(log_f - log_base))

    def _arm_division_rate(self, genome_summary):
        """CINner per-arm copy-number fitness: prod_seg s_arm[seg] ** (cn[seg] - baseline).

        Reads the per-segment copy numbers iscc already maintains (``seg_cns``). Computed in
        log space (sum of (cn - baseline) * log s_arm) so many-arm genomes don't overflow.
        Relative to the all-diploid baseline (cn == baseline for every arm) this is 1.0, so
        only copy-number deviations move the division rate.
        """
        cns = np.asarray(genome_summary['seg_cns'], dtype=float)
        return float(np.exp(np.sum((cns - self.arm_baseline) * self._log_s_arm)))

    def _epistasis_multiplier(self, event_bits):
        """The planted network's contribution to division fitness (1.0 when off / no events).

        A pure function of the genotype's event set, so it is cached per event set inside
        ``EpistasisNetwork`` — the same value under the exact and tau-leaping engines alike.
        """
        if self.epistasis is None or not event_bits:
            return 1.0
        return self.epistasis.multiplier(event_bits)

    def update_division_rate(self, genome_summary, event_bits=0, **kwargs):
        if self.selection_mode == "arm":
            return self._arm_division_rate(genome_summary) * self._epistasis_multiplier(event_bits)
        # Following CINner: oncogenes (effect>1) and tumour suppressors (effect<1) act in
        # opposite directions; mutating an oncogene or a TSG copy increases division fitness.
        de = self.driver_effects
        og = self._rel_fitness(genome_summary['n_wt_onc'], genome_summary['n_mut_onc'],
                               genome_summary['ploidy'], self.N_onc, de, de**2)
        tsg = self._rel_fitness(genome_summary['n_wt_tsg'], genome_summary['n_mut_tsg'],
                                genome_summary['ploidy'], self.N_tsg, 1. / de, 1.)
        # The epistasis term multiplies the additive model rather than replacing it: the additive
        # driver-count fitness stays exactly what it was, and the network adds
        # exp(sum_i beta_i x_i + sum_{i<j} E_ij x_i x_j) on top (DESIGN_epistasis.md §3).
        return og * tsg * self._epistasis_multiplier(event_bits)

    def update_dispersal_rate(self, genome_summary, **kwargs):
        e = self.dispersal_effects
        return self._rel_fitness(genome_summary['n_wt_disp'], genome_summary['n_mut_disp'],
                                 genome_summary['ploidy'], self.N_disp, e, e**2)

    def update_immune_resistance(self, genome_summary, **kwargs):
        e = self.immune_resistant_effects
        return self._rel_fitness(genome_summary['n_wt_ir'], genome_summary['n_mut_ir'],
                                 genome_summary['ploidy'], self.N_ir, e, e**2)

    def update_treatment_resistance(self, genome_summary, **kwargs):
        e = self.treatment_resistant_effects
        return self._rel_fitness(genome_summary['n_wt_tr'], genome_summary['n_mut_tr'],
                                 genome_summary['ploidy'], self.N_tr, e, e**2)

    def get_gene_names(self, gene_prefix='G'):
        gene_names = []
        for segment in range(self.n_segments):
            gn = [f'{gene_prefix}_{segment}_{i}' for i in range(self.segment_sizes[segment])]
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