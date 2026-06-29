"""Wire the genotype tumour engine to the ABC engine (DESIGN_inference A.1/A.4.1).

``TumorSimulator`` turns a parameter dict into a summary vector by running a short
``GenotypeTumor`` and compressing it with ``inference.summaries.summary_vector``. It is the
``simulate`` callback :class:`inference.abc.ABC` calls, and ``default_prior`` gives a sensible
product prior over the inferrable rates. Together with ``inference.abc`` this delivers the
parameter-recovery demonstration (``validation/validate_inference_recovery.py``) on an abstract
genome — no external data required.

Each inferrable parameter is mapped to its location in the engine config by ``PARAM_PATHS``; a
new knob is exposed to inference just by adding it there.
"""
import copy
import warnings

import numpy as np

from ..tumor.models import GenotypeTumor
from .abc import Prior
from .summaries import summary_vector

# parameter name -> (config section, key) within the simulator's config dict
PARAM_PATHS = {
    "mutation_rate": ("cancer_cell_params", "mutation_rate"),
    "amp_prob": ("cancer_cell_params", "amp_prob"),
    "snv_prob": ("cancer_cell_params", "snv_prob"),
    "cnv_prob": ("cancer_cell_params", "cnv_prob"),
    "n_snvs_per_allele": ("cancer_cell_params", "n_snvs_per_allele"),
    "dispersal_rate": ("cancer_cell_params", "dispersal_rate"),
    "driver_effects": ("selection_params", "driver_effects"),
    "prop_driver": ("selection_params", "prop_driver"),
}


def default_base_config():
    """A small, fast genotype-engine config for inference-sized runs.

    Small genome + modest grid so a single simulation is ~seconds, while still producing enough
    CNA/SNV events for the summary statistics to be informative.
    """
    return dict(
        # genome large enough that the SNV burden stays sub-saturated (so mutation_rate is
        # identifiable from the number of segregating sites, not a flat ceiling).
        genome_params={"n_segments": 10, "segment_size": 150},
        selection_params={
            "prop_driver": 0.15, "prop_dispersal": 0.0,
            "prop_immune_resistance": 0.0, "prop_treatment_resistance": 0.0,
            "driver_effects": 1.1, "dispersal_effects": 1.0,
            "immune_resistant_effects": 1.0, "treatment_resistant_effects": 1.0,
        },
        cancer_cell_params={
            "max_birth_rate": 0.95, "division_rate": 0.4, "death_rate": 0.02,
            "mutation_rate": 0.3, "dispersal_rate": 0.2,
            "snv_prob": 0.5, "cnv_prob": 0.5, "n_snvs_per_allele": 0.5, "amp_prob": 0.5,
        },
        deme_params={"carrying_capacity": 8, "maximum_death_rate": 0.5},
        spatial_params={"grid_size": 14, "structure_radius": 0, "immune_density": 0.0},
    )


def default_prior(params=("mutation_rate", "amp_prob")):
    """Product prior over the inferrable rates (uniform; ``amp_prob`` in [0,1])."""
    ranges = {
        "mutation_rate": (0.05, 0.6),
        "amp_prob": (0.1, 0.9),
        "snv_prob": (0.1, 0.9),
        "cnv_prob": (0.1, 0.9),
        "dispersal_rate": (0.0, 0.6),
        "driver_effects": (1.0, 2.0),
        "prop_driver": (0.05, 0.4),
        "n_snvs_per_allele": (0.0, 1.5),
    }
    return Prior({p: ranges[p] for p in params})


class TumorSimulator:
    """Callable ``param-dict -> summary vector`` for the ABC engine (picklable for workers)."""

    def __init__(self, base_config=None, n_steps=700, n_replicates=1,
                 include_snv=True, seed=0):
        self.base_config = base_config or default_base_config()
        self.n_steps = n_steps
        self.n_replicates = n_replicates
        self.include_snv = include_snv
        self.seed = seed

    def _config_with(self, theta):
        cfg = copy.deepcopy(self.base_config)
        for name, value in theta.items():
            if name not in PARAM_PATHS:
                raise KeyError(f"unknown inferrable parameter {name!r}")
            section, key = PARAM_PATHS[name]
            cfg[section][key] = float(value)
        return cfg

    def simulate_tumor(self, theta, seed):
        cfg = self._config_with(theta)
        t = GenotypeTumor(
            seed=seed,
            genome_params=cfg["genome_params"],
            selection_params=cfg["selection_params"],
            cancer_cell_params=cfg["cancer_cell_params"],
            deme_params=cfg["deme_params"],
            spatial_params=cfg["spatial_params"],
        )
        t.grow(n_steps=self.n_steps, seed=seed)
        return t

    def __call__(self, theta, seed=None):
        """Mean summary vector over ``n_replicates`` runs. ``nan`` if every replicate goes extinct."""
        base = self.seed if seed is None else seed
        # hash the parameter values into the seed so different theta get different trajectories
        offset = abs(hash(tuple(sorted((k, round(v, 6)) for k, v in theta.items())))) % 100000
        vecs = []
        for r in range(self.n_replicates):
            t = self.simulate_tumor(theta, seed=base + offset + r)
            if t.get_cancer_size() == 0:
                continue
            vec, self.names = summary_vector(t, include_snv=self.include_snv)
            vecs.append(vec)
        if not vecs:
            n = len(summary_vector(self.simulate_tumor(theta, seed=base))[1])
            return np.full(n, np.nan)
        with warnings.catch_warnings():        # all-nan column (e.g. sfs_rsq) -> nan, not a warning
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(np.vstack(vecs), axis=0)
