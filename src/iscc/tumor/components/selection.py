import numpy as np
import pandas as pd

from .epistasis import EpistasisNetwork
from ...constants import DEFAULT_LAYOUT_SEED, LAYOUT_OFFSET_EPISTASIS

class Selection(object):
    def __init__(self, n_segments=10, segment_size=1000, segment_sizes=None,
                 prop_driver=0.1, prop_dispersal=0.1, prop_treatment_resistance=0.1, prop_immune_resistance=0.1,
                 prop_breach=0.0, prop_stromal_survival=0.0, prop_met_survival=0.0,
                 prop_drug_tolerance=0.0,
                 driver_effects=1.1, dispersal_effects=1.1, treatment_resistant_effects=1.1, immune_resistant_effects=1.1,
                 breach_effects=1.1, stromal_survival_effects=1.1, met_survival_effects=1.1,
                 drug_tolerance_effects=1.1,
                 breach_cost=0.0, stromal_survival_cost=0.0, met_survival_cost=0.0,
                 treatment_resistance_cost=0.0, drug_tolerance_cost=0.0, trait_source="dosage",
                 treatment_resistance_binary=False,
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
        # Per-gene germline (inherited) variant mask: 0 = none, 1 = heterozygous, 2 = homozygous.
        # All zeros unless a patient's germline is seeded (iscc.tumor.germline.apply_germline writes
        # it), and surfaced by get_gene_data only when non-empty, so the default annotation is
        # unchanged. It lives here because the germline is a property of the GENOME, alongside which
        # genes are drivers.
        self.germline_types = np.zeros(self.n_genes, dtype=int)
        self.prop_driver = prop_driver
        self.prop_dispersal = prop_dispersal
        self.prop_treatment_resistance = prop_treatment_resistance
        self.prop_immune_resistance = prop_immune_resistance
        # Compartment-dependent selection (v1, DESIGN_phenotype_plasticity.md §2): two more gene-based
        # heritable axes, exact analogues of immune resistance. ``breach`` attenuates the epithelial-
        # ring barrier; ``stromal_survival`` attenuates the stromal hazard. OFF by default (prop 0 ->
        # empty axis -> N_*=0 -> update_* returns 1 -> trait 0 -> zero death terms -> byte-identical).
        self.prop_breach = prop_breach
        self.prop_stromal_survival = prop_stromal_survival
        # Metastatic-host survival (R9, metastasis module): a further gene-based heritable axis, exact
        # analogue of stromal_survival, that attenuates the metastatic host-tissue hazard (_met_hazard)
        # and biases transit survival on the primary->met migration hop. OFF by default (prop 0 -> empty
        # axis -> N_ms=0 -> update_met_survival returns 1 -> trait 0 -> zero met terms -> byte-identical).
        self.prop_met_survival = prop_met_survival
        # DRUG TOLERANCE — the persister axis (Sharma et al. 2010; Hata et al. 2016). Deliberately NOT
        # a second flavour of resistance: a tolerant clone SURVIVES therapy but barely proliferates,
        # so it forms the residual-disease floor that outlasts the drug WITHOUT being able to regrow
        # under it. Resistance is what regrows; tolerance is what waits. Mode IV needs both, because a
        # de novo resistance mutation has to arise in something that is still alive and still dividing
        # once the sensitive bulk is gone (which at kill_rate 1.5 takes only ~9 generations).
        # MODELLING CAVEAT, deliberate: a real persister state is NON-genetic and reversible, whereas
        # this is a heritable trait, so the tolerant pool is standing genetic variation rather than a
        # state any cell can enter. It reproduces the population DYNAMICS that mode IV requires, not
        # the chromatin biology. NOTE ALSO that iscc draws mutation as a fate of DIVISION, so a
        # tolerant clone must keep dividing (slowly) to mutate at all — set ``drug_tolerance_cost`` to
        # SLOW persisters, never to stop them, or the pool becomes a mutational dead end.
        # OFF by default (prop 0 -> empty axis -> N_dt=0 -> update returns 1 -> trait 0 -> no effect).
        self.prop_drug_tolerance = prop_drug_tolerance

        # Fixed about fitness
        self.driver_effects = driver_effects
        self.dispersal_effects = dispersal_effects
        self.treatment_resistant_effects = treatment_resistant_effects
        self.immune_resistant_effects = immune_resistant_effects
        self.breach_effects = breach_effects
        self.stromal_survival_effects = stromal_survival_effects
        self.met_survival_effects = met_survival_effects
        self.drug_tolerance_effects = drug_tolerance_effects
        # Compartment-context fitness TRADE-OFFS (R15, go-or-grow): each niche/dissemination trait
        # carries a PROLIFERATION cost that applies EVERYWHERE, while its BENEFIT (attenuating a hazard)
        # is gated to its compartment in _death_rate. So a trait is net-favoured only where its niche
        # benefit outweighs the cost (breach at the wall, stromal in the stroma, met in the deposit,
        # resistance under chemo) and is selected AGAINST elsewhere — instead of being a free neutral
        # passenger that hitchhikes into every compartment. `cost` fraction of division lost per unit
        # trait. ALL DEFAULT 0.0 -> proliferation_cost() returns 1.0 -> division byte-identical to before.
        self.breach_cost = breach_cost
        self.stromal_survival_cost = stromal_survival_cost
        self.met_survival_cost = met_survival_cost
        self.treatment_resistance_cost = treatment_resistance_cost
        # ALL-OR-NOTHING resistance. The graded map (trait = 1 - 1/effects^(2*n_mut/ploidy)) means a
        # cell with one mutated copy at the shipped effects 2.8 is only 64% resistant -- it still
        # absorbs 36% of the kill and dies alongside the sensitive bulk, just slower, which is why a
        # "resistant" clone could not expand DURING treatment. It also leaks resistance back on a
        # whole-genome doubling, which halves the exponent (one mutated copy at effects 20 falls from
        # 0.95 to 0.78). With this flag ANY resistance mutation sets the trait to exactly 1.0: the
        # drug term (1 - trait) becomes exactly zero -- the cell is untouched by chemotherapy -- and
        # the clone pays the FULL treatment_resistance_cost, so resistance is a clean all-or-nothing
        # trade instead of a sliding scale. Default False -> the graded map, byte-identical.
        self.treatment_resistance_binary = bool(treatment_resistance_binary)
        # Tolerance is COSTLY by construction: that is what keeps persisters rare before therapy
        # (out-competed in a crowded deme) instead of taking over the tumour.
        self.drug_tolerance_cost = drug_tolerance_cost

        # What a dissemination/niche TRAIT reads off the genome — breach, stromal_survival,
        # met_survival, immune_resistance and treatment_resistance only. The oncogene/TSG DRIVER
        # fitness is untouched by this and always reads copy-number dosage (that is the CINner model
        # and it is the point of the copy-number layer).
        #
        #   "dosage"   (default, historical): the trait reads the ploidy-normalised dosage of ALL
        #              trait-gene copies, mutated or not. Because a trait axis is scattered uniformly
        #              at ``prop_*``, a segment's trait-gene count fluctuates binomially around the
        #              genome mean, so almost every copy-number change moves the trait — a gain of a
        #              segment that happens to be trait-dense switches the trait on even though no
        #              trait gene was ever mutated. Paired with a ``*_cost`` that makes the trait cost
        #              proliferation everywhere, that turns ordinary copy-number drift into a
        #              proliferation tax and leaves almost no CNA net-beneficial.
        #   "mutation" (opt-in): the trait reads only SNV-MUTATED trait-gene copies. Formally it is
        #              the dosage form divided by the same genome's all-wild-type trait fitness, so
        #              the copy-number-only component cancels exactly: an arm-level gain of unmutated
        #              invasion genes does not make a cell invasive. The trait still has to be EARNED
        #              by mutation (and, in the ductal field, still gates invasion), and it is still
        #              graded by the mutant allele's own dosage, so amplifying a mutated trait gene
        #              still strengthens the trait and a whole-genome duplication leaves it unchanged.
        if trait_source not in ("dosage", "mutation"):
            raise ValueError(
                f"trait_source must be 'dosage' or 'mutation', got {trait_source!r}")
        self.trait_source = trait_source

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
        self.make_breach()
        self.make_stromal_survival()
        self.make_met_survival()
        self.make_expmap()
        # LAST on purpose: make_drug_tolerance() draws from the layout rng even when
        # prop_drug_tolerance is 0 (binomial(1, 0.0, size=N) still consumes state), so calling it
        # before make_expmap would shift the expression-map layout of every EXISTING config. Called
        # last, an off-by-default tolerance axis leaves every other gene-role layout untouched.
        self.make_drug_tolerance()

        # Total number of genes in each category, used to make fitness *relative* to the
        # all-wild-type diploid baseline (so the baseline is neutral and only deviations
        # -- mutations and copy-number changes -- shift the rate; see update_* below).
        self.N_onc = sum(len(x) for x in self.onc)
        self.N_tsg = sum(len(x) for x in self.tsg)
        self.N_disp = sum(len(x) for x in self.dispersal)
        self.N_ir = sum(len(x) for x in self.immune_resistance)
        self.N_tr = sum(len(x) for x in self.treatment_resistance)
        self.N_breach = sum(len(x) for x in self.breach)
        self.N_ss = sum(len(x) for x in self.stromal_survival)
        self.N_ms = sum(len(x) for x in self.met_survival)
        self.N_dt = sum(len(x) for x in self.drug_tolerance)
        self.update_dict = {'viability': self.update_viability,
                            'division_rate': self.update_division_rate,
                            'dispersal_rate': self.update_dispersal_rate,
                            'immune_resistance': self.update_immune_resistance,
                            'treatment_resistance': self.update_treatment_resistance,
                            'breach': self.update_breach,
                            'stromal_survival': self.update_stromal_survival,
                            'met_survival': self.update_met_survival,
                            'drug_tolerance': self.update_drug_tolerance,
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

    def make_breach(self): # select sites that if mutated let the cell breach the epithelial ring
        self.breach_types = []
        self.breach = []
        for seg in range(self.n_segments):
            confers_breach = self.rng.binomial(1, self.prop_breach, size=self.segment_sizes[seg])
            self.breach_types.append(confers_breach)
            self.breach.append(np.where(confers_breach == 1)[0])

    def make_stromal_survival(self): # select sites that if mutated let the cell survive the stroma
        self.stromal_survival_types = []
        self.stromal_survival = []
        for seg in range(self.n_segments):
            confers_survival = self.rng.binomial(1, self.prop_stromal_survival, size=self.segment_sizes[seg])
            self.stromal_survival_types.append(confers_survival)
            self.stromal_survival.append(np.where(confers_survival == 1)[0])

    def make_met_survival(self): # select sites that if mutated let the cell survive in the metastatic host tissue
        self.met_survival_types = []
        self.met_survival = []
        for seg in range(self.n_segments):
            confers_survival = self.rng.binomial(1, self.prop_met_survival, size=self.segment_sizes[seg])
            self.met_survival_types.append(confers_survival)
            self.met_survival.append(np.where(confers_survival == 1)[0])

    def make_expmap(self):
        # Effect of snv on exp: up or down depending on wether tsg or og, and also if immune resistance-inducing, overexpress
        self.mut_effects = []
        for seg in range(self.n_segments):
            mut_effects = np.ones((self.segment_sizes[seg],)) # in general, mutation has no effect
            mut_effects[np.where(self.driver_types[seg] == 1)] = 2. # if mutated og, increase exp
            mut_effects[np.where(self.driver_types[seg] == -1)] = 0.5 # if mutated tsg, decrease exp
            mut_effects[np.where(self.immune_resistance_types[seg] == 1)] = 2. # if mutated immune resistance, increase exp
            mut_effects[np.where(self.breach_types[seg] == 1)] = 2. # if mutated breach gene, increase exp
            mut_effects[np.where(self.stromal_survival_types[seg] == 1)] = 2. # if mutated stromal-survival gene, increase exp
            mut_effects[np.where(self.met_survival_types[seg] == 1)] = 2. # if mutated met-survival gene, increase exp
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

    def get_breach(self):
        genes = []
        for seg in range(self.n_segments):
            idx = np.where(self.breach_types[seg] == 1)[0]
            genes.append(idx + self._seg_offsets[seg])
        genes = np.concatenate(genes)
        return genes

    def get_stromal_survival(self):
        genes = []
        for seg in range(self.n_segments):
            idx = np.where(self.stromal_survival_types[seg] == 1)[0]
            genes.append(idx + self._seg_offsets[seg])
        genes = np.concatenate(genes)
        return genes

    def get_met_survival(self):
        genes = []
        for seg in range(self.n_segments):
            idx = np.where(self.met_survival_types[seg] == 1)[0]
            genes.append(idx + self._seg_offsets[seg])
        genes = np.concatenate(genes)
        return genes

    def update_viability(self, genome_summary, **kwargs):
        if genome_summary['ploidy'] > self.max_ploidy:
            return 0
        if genome_summary['highest_cn'] > self.max_cn:
            return 0
        if genome_summary['nullisomy_count'] > self.max_nullisomy:
            return 0
        # Driver LOAD: distinct oncogene/TSG drivers PLUS each dissemination/niche PROGRAM the clone has
        # activated (breach / stromal_survival / met_survival / treatment_resistance, counted by
        # presence). A clone that stacks every program — the "super-clone" — thus exceeds a realistic
        # mutational-load ceiling and is non-viable, the CINner way. Counting the compartment traits is
        # what stops one lineage from being optimal in every compartment at once. NOTE off-by-default:
        # ``max_mut_drivers`` defaults to 1000, so this is a no-op (load << 1000) unless a config sets a
        # realistic limit; and with prop_*=0 the compartment terms are 0 -> byte-identical either way.
        load = (genome_summary['n_mutated_drivers']
                + int(genome_summary.get('n_mut_breach', 0) > 0)
                + int(genome_summary.get('n_mut_ss', 0) > 0)
                + int(genome_summary.get('n_mut_ms', 0) > 0)
                + int(genome_summary.get('n_mut_tr', 0) > 0)
                + int(genome_summary.get('n_mut_dt', 0) > 0))
        if load > self.max_mut_drivers:
            return 0
        return 1

    def update_death_rate(self, genome_summary, param, event_bits=0, **kwargs):
        # Synthetic lethality (R14): a genotype carrying both events of a planted mutually-exclusive
        # pair is REMOVED, not merely slowed. Suppressing its division alone would not purge it --
        # the density-death slope is max(0, div - death), so a clone that has stopped dividing takes
        # no crowding death and simply persists. Off unless exclusive pairs are configured.
        if self.epistasis is not None and self.epistasis.is_lethal(event_bits):
            return self.epistasis.lethal_death_rate
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

    def _trait_fitness(self, n_wt, n_mut, ploidy, n_total, effect):
        """Relative fitness of ONE dissemination/niche trait axis, under ``trait_source``.

        ``"dosage"`` is exactly :meth:`_rel_fitness` with (effect, effect**2) — the historical
        behaviour, kept bit-for-bit.

        ``"mutation"`` drops the wild-type dosage term, i.e. it measures the axis RELATIVE TO THE SAME
        COPY-NUMBER STATE with every trait gene wild-type::

            F_mut = F_dosage(n_wt, n_mut) / F_dosage(n_wt + n_mut, 0)
                  = exp[(2 n_mut / ploidy) (log effect**2 - log effect)]
                  = effect ** (2 n_mut / ploidy)

        so the copy-number-only part cancels identically and only mutated copies move the trait. A
        single heterozygous trait SNV in a diploid cell gives ``effect`` under BOTH modes — the two
        agree on what a mutation is worth and disagree only about what a copy-number change is worth.
        """
        if self.trait_source == "dosage":
            return self._rel_fitness(n_wt, n_mut, ploidy, n_total, effect, effect ** 2)
        if ploidy <= 0 or n_total == 0:
            return 1.0
        return float(np.exp((2.0 * n_mut / ploidy) * np.log(effect)))

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
        return self._trait_fitness(genome_summary['n_wt_ir'], genome_summary['n_mut_ir'],
                                   genome_summary['ploidy'], self.N_ir, e)

    def make_drug_tolerance(self):
        """Sites that, when mutated, let a cell ENTER the drug-tolerant persister state."""
        self.drug_tolerance_types = []
        self.drug_tolerance = []
        for seg in range(self.n_segments):
            confers = self.rng.binomial(1, self.prop_drug_tolerance, size=self.segment_sizes[seg])
            self.drug_tolerance_types.append(confers)
            self.drug_tolerance.append(np.where(confers == 1)[0])

    def get_drug_tolerance(self):
        genes = []
        for seg in range(self.n_segments):
            idx = np.where(self.drug_tolerance_types[seg] == 1)[0]
            genes.append(idx + self._seg_offsets[seg])
        return np.concatenate(genes) if genes else np.array([], dtype=int)

    def update_drug_tolerance(self, genome_summary, **kwargs):
        e = self.drug_tolerance_effects
        return self._trait_fitness(genome_summary['n_wt_dt'], genome_summary['n_mut_dt'],
                                   genome_summary['ploidy'], self.N_dt, e)

    def update_treatment_resistance(self, genome_summary, **kwargs):
        e = self.treatment_resistant_effects
        return self._trait_fitness(genome_summary['n_wt_tr'], genome_summary['n_mut_tr'],
                                   genome_summary['ploidy'], self.N_tr, e)

    def update_breach(self, genome_summary, **kwargs):
        e = self.breach_effects
        return self._trait_fitness(genome_summary['n_wt_breach'], genome_summary['n_mut_breach'],
                                   genome_summary['ploidy'], self.N_breach, e)

    def update_stromal_survival(self, genome_summary, **kwargs):
        e = self.stromal_survival_effects
        return self._trait_fitness(genome_summary['n_wt_ss'], genome_summary['n_mut_ss'],
                                   genome_summary['ploidy'], self.N_ss, e)

    def update_met_survival(self, genome_summary, **kwargs):
        e = self.met_survival_effects
        return self._trait_fitness(genome_summary['n_wt_ms'], genome_summary['n_mut_ms'],
                                   genome_summary['ploidy'], self.N_ms, e)

    def proliferation_cost(self, ep):
        """Go-or-grow trade-off multiplier on ``division_rate`` (R15): each dissemination/niche trait
        costs a fraction ``*_cost`` of division PER UNIT of the trait, applied EVERYWHERE, while the
        trait's BENEFIT is gated to its compartment in ``_death_rate``. So a trait is net-favoured only
        where the niche benefit outweighs the cost, and selected against elsewhere. Returns 1.0 when
        every cost is 0 (default) -> division byte-identical to the additive model."""
        return (max(0.0, 1.0 - self.breach_cost * ep.get("breach", 0.0))
                * max(0.0, 1.0 - self.stromal_survival_cost * ep.get("stromal_survival", 0.0))
                * max(0.0, 1.0 - self.met_survival_cost * ep.get("met_survival", 0.0))
                * max(0.0, 1.0 - self.treatment_resistance_cost * ep.get("treatment_resistance", 0.0)))
        # NOTE drug_tolerance is deliberately NOT charged here. Every other trait's cost is permanent
        # because the trait is permanent, but the persister state only exists while the drug does --
        # a tolerant cell off drug is an ordinary cell. Charging it here made the trait unusable: a
        # cost high enough to hold the pool at a residual-disease floor under drug also purged that
        # pool to ~0.1% before treatment ever started. GenotypeTumor._apply_treatment applies
        # drug_tolerance_cost as a division multiplier for the duration of the dose instead.

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
        breach_types = np.concatenate(self.breach_types)
        stromal_survival_types = np.concatenate(self.stromal_survival_types)
        met_survival_types = np.concatenate(self.met_survival_types)
        gene_data = dict(driver_types=pd.DataFrame(driver_types, index=gene_names),
                         dispersal_types=pd.DataFrame(dispersal_types, index=gene_names),
                         treatment_resistance_types=pd.DataFrame(treatment_resistance_types, index=gene_names),
                         immune_resistance_types=pd.DataFrame(immune_resistance_types, index=gene_names),
                         breach_types=pd.DataFrame(breach_types, index=gene_names),
                         stromal_survival_types=pd.DataFrame(stromal_survival_types, index=gene_names),
                         met_survival_types=pd.DataFrame(met_survival_types, index=gene_names))
        # Germline mask (0 none / 1 heterozygous / 2 homozygous), added ONLY when this patient
        # actually carries germline variants — so the annotation a plain tumour writes out, and the
        # keys a reader gets back, are exactly what they were before germline existed.
        germline_types = getattr(self, "germline_types", None)
        if germline_types is not None and np.any(germline_types):
            gene_data["germline_types"] = pd.DataFrame(np.asarray(germline_types), index=gene_names)
        return gene_data