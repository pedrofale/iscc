import matplotlib.colors as mcolors
import numpy as np

# Config-determined seed for the GENOME/gene-role LAYOUT (driver/oncogene/TSG/dispersal/TR/IR
# positions in Selection + the shared per-cell-type baseline expression), decoupled from the
# per-run EVOLUTION seed that drives the stochastic dynamics. Because it defaults to a fixed
# constant, two runs of the SAME config share their driver identities by construction (so
# recurrence / cohort analysis is meaningful), differing only in evolution. Chosen as 42 — the
# ubiquitous test/default seed — so existing single-tumour fixtures are reproduced byte-for-byte
# (the abstract-mode analogue of the fixed shared genome_spec real-genome mode already has). See
# DESIGN_cohort.md §1.
DEFAULT_LAYOUT_SEED = 42

# --- LAYOUT sub-stream registry ---------------------------------------------------------------
# Pre-assigned offsets into the LAYOUT stream. Every property of the GENOME/LANDSCAPE (as opposed to
# a per-run EVENT) must be drawn from the layout stream, so that two patients sharing a config share
# it. Each optional layer of that landscape draws from its OWN sub-stream,
# ``default_rng(layout_seed + OFFSET)``, rather than from ``layout_rng`` itself. Two reasons:
#
#  1. The base ``layout_rng`` is left untouched, so Selection's gene-role layout and the per-cell-type
#     baseline expression stay byte-identical whether these layers are on or off — nothing re-baselines.
#  2. The sub-streams are INDEPENDENT, so changing one layer's parameters (``n_programs``,
#     ``n_interactions``, …) cannot shift another layer's draws. A single shared stream would couple
#     them: adding a program would silently reshuffle which genes are oncogenes, breaking
#     comparability between configs that differ only in program parameters.
#
# Offsets are permanent identifiers and are registered here — and only here — so that
# concurrently-developed layers cannot collide. Never renumber one (that would silently change the
# landscape of every existing tumour); claim a new one here before use.
LAYOUT_OFFSET_PROGRAMS = 101       # expression gene programs (R13, DESIGN_expression.md): gene->program
                                   # map, `loading`, program regulators, per-gene dosage sensitivity s_g
LAYOUT_OFFSET_F8_PROGRAMS = 102    # F8 microenvironment programs (DESIGN_features §H): the
                                   # hypoxia-responsive + CCI-target gene sets
LAYOUT_OFFSET_EPISTASIS = 201      # the epistasis E matrix / dependency DAG (R14, DESIGN_epistasis.md)
LAYOUT_OFFSET_MET = 301            # metastasis module (R9): the host-parenchyma baseline expression
                                   # profile for the metastatic deposit (celltype_exps["host"])

normal_cmap = {'epithelial': 'green', 'immune': 'yellow', 'stromal': 'pink', 'host': 'lightgray'}
normal_names = list(normal_cmap.keys())
normal_colors = list(normal_cmap.values())
normal_colors_rgba = np.array([np.array(mcolors.to_rgba(color)) for color in normal_colors])
normal_cmap_rgba = dict(zip(normal_names, normal_colors_rgba))