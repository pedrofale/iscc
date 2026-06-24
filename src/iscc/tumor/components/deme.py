import numpy as np

from collections import Counter
import logging


class Deme(object):
    def __init__(
        self,
        cell=None,
        carrying_capacity=1,
        initial_death_rate=0.1,
        maximum_death_rate=0.5,
        tumor=None,
        row=None,
        col=None,
        id=None,
    ):
        if cell is None and tumor is None:
            raise ValueError(
                "Must initialise Deme with either a Cell or a Tumor object."
            )
        self.carrying_capacity = carrying_capacity
        self.initial_death_rate = initial_death_rate
        self.maximum_death_rate = maximum_death_rate
        self.tumor = tumor
        self.row = row
        self.col = col
        self.id = id
        self.types_counts = Counter()
        self.genotypes_counts = Counter()
        self.genotypes_parents = dict()
        # Insertion-ordered set of cells: a plain set iterates in object-id order,
        # which varies across processes and breaks reproducibility (the seeded RNG
        # would sample cells in a different order). A dict preserves insertion order.
        self.cells = dict()
        self.deme_rate = 0.0

        if cell is not None:
            self.add_cell(cell)
            self.deme_rate = cell.evolutionary_parameters['death_rate'] + cell.evolutionary_parameters['division_rate']

    def add_cell(self, cell, genotype_id=None):
        if genotype_id is None:
            if cell.type == 'cancer':
                cell.genotype_id = str(cell.genotype_id)
            else:
                cell.genotype_id = cell.type    
        else:
            cell.genotype_id = str(genotype_id)
        self.cells[cell] = None
        if cell.genotype_id in self.genotypes_counts:
            self.genotypes_counts[cell.genotype_id] += 1
        else:
            self.genotypes_counts[cell.genotype_id] = 1

        if cell.type in self.types_counts:
            self.types_counts[cell.type] += 1
        else:
            self.types_counts[cell.type] = 1
        if self.tumor is not None:
            self.tumor.register_birth(cell.genotype_id)

    def sample_event(self, cell, immune_cell_fraction=0., rng=None):
        if cell.evolutionary_parameters['viability'] == 0:
            return "death"

        if cell.type == 'cancer':
            events = ["death", "division"]
            rates = [self.get_cancer_death_rate(cell.evolutionary_parameters['death_rate'], 
                                                immune_cell_fraction=immune_cell_fraction, 
                                                immune_resistance=cell.evolutionary_parameters['immune_resistance']), 
                    cell.evolutionary_parameters['division_rate']]
            event = rng.choice(events, p=np.array(rates) / np.sum(rates))
        elif cell.type == 'immune':
            events = ['death', "dispersal"]
            rates = [self.get_normal_death_rate(cell.evolutionary_parameters['death_rate']), cell.evolutionary_parameters['dispersal_rate']]
            event = rng.choice(events, p=np.array(rates) / np.sum(rates))
        else: # assume non-cancer cells don't divide
            events = ['death']
            rates = [self.get_normal_death_rate(cell.evolutionary_parameters['death_rate'])]
            event = 'death'
        
        return event

    def apply_event(self, event, cell, rng=None):
        if event == "death":
            pre_death_count = self.genotypes_counts[cell.genotype_id]
            del self.cells[cell]
            self.genotypes_counts[cell.genotype_id] -= 1
            self.types_counts[cell.type] -= 1
            self.tumor.register_death(cell.genotype_id)

            self.deme_rate -= cell.evolutionary_parameters['division_rate'] + cell.evolutionary_parameters['death_rate']
            if cell.type == 'immune':
                self.deme_rate -= cell.evolutionary_parameters['dispersal_rate']
            self.deme_rate = max(0, self.deme_rate)
            self.tumor.deme_rates[self.id] = self.deme_rate
            del cell    
        elif event == "division":
            new_cell = cell.divide()
            new_cell.set_params()
            if cell.type == "cancer":
                if self.tumor.type == 'mixed':
                    mutate = 1
                else:
                    mutation_prob = cell.mutation_rate / (cell.mutation_rate + cell.evolutionary_parameters['dispersal_rate'])
                    mutate = rng.binomial(1, mutation_prob)

                if mutate:
                    new_cell.mutate(rng, self.tumor.selection)
                    # new_cell.genotype_id = f"{i:03}" + str(
                    #     new_cell.genotype_id
                    # )  # TODO: Maybe not a great idea to depend on number of cells
                    self.genotypes_parents[
                        new_cell.genotype_id
                    ] = new_cell.parent.genotype_id
                    if (
                        self.genotypes_parents[new_cell.genotype_id]
                        == new_cell.genotype_id
                    ):
                        raise Exception(
                            f"Oh no! genotype is its own parent?!: {new_cell.genotype_id}"
                        )
                    self.tumor.register_parent(new_cell.genotype_id, new_cell.parent.genotype_id)
                    self.add_cell(new_cell, genotype_id=new_cell.genotype_id)
                    self.deme_rate += new_cell.evolutionary_parameters['division_rate'] + new_cell.evolutionary_parameters['death_rate']
                    self.tumor.deme_rates[self.id] = self.deme_rate
                else:
                    possible_demes = self.tumor.get_neighboring_demes(self)
                    target_deme = rng.choice(possible_demes)                                    
                    target_deme.add_cell(new_cell, genotype_id=new_cell.genotype_id)
                    target_deme.deme_rate += new_cell.evolutionary_parameters['division_rate'] + new_cell.evolutionary_parameters['death_rate']
                    self.tumor.deme_rates[target_deme.id] = target_deme.deme_rate
        elif event == 'dispersal':
            possible_demes = self.tumor.get_neighboring_demes(self)
            if len(possible_demes) == 0:
                return
            target_deme = rng.choice(possible_demes)
            # The cell migrates: remove it from this deme and add it to the target.
            moved_rate = (cell.evolutionary_parameters['division_rate']
                          + cell.evolutionary_parameters['dispersal_rate']
                          + cell.evolutionary_parameters['death_rate'])
            del self.cells[cell]
            self.genotypes_counts[cell.genotype_id] -= 1
            self.types_counts[cell.type] -= 1
            self.tumor.register_death(cell.genotype_id)  # target_deme.add_cell re-registers the birth
            self.deme_rate = max(0, self.deme_rate - moved_rate)
            self.tumor.deme_rates[self.id] = self.deme_rate

            target_deme.add_cell(cell, genotype_id=cell.genotype_id)
            target_deme.deme_rate += moved_rate
            self.tumor.deme_rates[target_deme.id] = target_deme.deme_rate

    def get_immune_cell_fraction(self):
        return sum([cell.type == 'immune' for cell in self.cells]) / len(self.cells)

    def update(self, treat=False, treatment=None, rng=None, subset_size=10, batch_size=1):
        # Choose subset of cells randomly in this deme
        # Get immune cell fraction in deme
        immune_cell_fraction = self.get_immune_cell_fraction()
        # Sample cell *indices* (not the Cell objects) to avoid rebuilding an object
        # ndarray on every rng.choice; the random draws are identical either way.
        cell_list = list(self.cells)
        sub_idx = rng.choice(len(cell_list), size=min(subset_size, len(cell_list)), replace=False)
        cells = [cell_list[i] for i in sub_idx]
        cell_rates = []
        for cell in cells:
            cell_rate = cell.evolutionary_parameters['death_rate'] + cell.evolutionary_parameters['division_rate']
            if cell.type == 'immune':
                cell_rate += cell.evolutionary_parameters['dispersal_rate']
            cell_rates.append(cell_rate)
        cell_rates = np.array(cell_rates)
        batch_idx = rng.choice(len(cells), p=cell_rates/cell_rates.sum(), size=min(batch_size, len(cells)), replace=False)
        cells = [cells[i] for i in batch_idx]

        for cell in cells:
            if treat:
                treatment.apply(cell, mut_effects=self.tumor.selection.mut_effects)
            
            event = self.sample_event(cell, immune_cell_fraction=immune_cell_fraction, rng=rng)
            self.apply_event(event, cell, rng=rng)
            rng = rng.spawn(1)[0]

    def get_normal_death_rate(self, cell_death_rate):
        """Update prob of each cell dieing based on its own 
        rate, the deme's carrying capacity"""
        if len(self.cells) <= self.carrying_capacity:
            return min(cell_death_rate, self.maximum_death_rate)
        else:
            return min(cell_death_rate * self.carrying_capacity, self.maximum_death_rate)
        
    def get_cancer_death_rate(self, cell_death_rate, immune_cell_fraction, immune_resistance):
        """Update prob of each cell dieing based on its own 
        rate, the deme's carrying capacity and the fraction of immune cells"""
        if len(self.cells) <= self.carrying_capacity:
            return min(cell_death_rate * (immune_cell_fraction ** immune_resistance), self.maximum_death_rate)
        else:
            return min(cell_death_rate * (immune_cell_fraction ** immune_resistance) * self.carrying_capacity, self.maximum_death_rate)        

    def get_genotype_frequencies(self, normalize=True):
        # Get the genotype ids present in this deme and their frequencies.
        genotypes = list(self.genotypes_counts.keys())
        counts = np.array([self.genotypes_counts[g] for g in genotypes], dtype=float)
        freqs = counts
        if normalize and counts.sum() > 0:
            freqs = counts / counts.sum()
        return genotypes, freqs

    def get_most_frequent_genotype(self):
        return self.genotypes_counts.most_common(1)[0][0]

    def get_diversity(self):
        # Get Simpson index
        raise NotImplementedError

    def plot_grid(self):
        raise NotImplementedError

    def plot_tree(self):
        raise NotImplementedError
