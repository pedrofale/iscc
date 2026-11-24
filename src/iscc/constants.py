import matplotlib.colors as mcolors
import numpy as np

# Config-determined seed for the GENOME/gene-role LAYOUT (driver/oncogene/TSG/dispersal/TR/IR
# positions in Selection + the shared per-cell-type baseline expression), decoupled from the
# per-run EVOLUTION seed that drives the stochastic dynamics. Because it defaults to a fixed
# constant, two runs of the SAME config share their driver identities by construction (so
# recurrence / cohort analysis is meaningful), differing only in evolution (the abstract-mode
# analogue of the fixed shared genome_spec of real-genome mode).
DEFAULT_LAYOUT_SEED = 42

# --- LAYOUT sub-stream registry ---------------------------------------------------------------
# Pre-assigned offsets into the LAYOUT stream. Every property of the GENOME/LANDSCAPE (as opposed to
# a per-run EVENT) must be drawn from the layout stream, so that two patients sharing a config share
# it. Each optional layer of that landscape draws from its OWN sub-stream,
# ``default_rng(layout_seed + OFFSET)``, rather than from ``layout_rng`` itself. Two reasons:
#
#  1. The base ``layout_rng`` is left untouched, so Selection's gene-role layout and the per-cell-type
#     baseline expression stay the same whether these layers are on or off.
#  2. The sub-streams are INDEPENDENT, so changing one layer's parameters (``n_programs``,
#     ``n_interactions``, …) cannot shift another layer's draws. A single shared stream would couple
#     them: adding a program would silently reshuffle which genes are oncogenes, breaking
#     comparability between configs that differ only in program parameters.
#
# Offsets are permanent identifiers and are registered here — and only here — so that
# concurrently-developed layers cannot collide. Never renumber one (that would silently change the
# landscape of every existing tumour); claim a new one here before use.
LAYOUT_OFFSET_PROGRAMS = 101       # expression gene programs: gene->program
                                   # map, `loading`, program regulators, per-gene dosage sensitivity s_g
LAYOUT_OFFSET_F8_PROGRAMS = 102    # microenvironment programs: the
                                   # hypoxia-responsive + CCI-target gene sets
LAYOUT_OFFSET_EPISTASIS = 201      # the epistasis E matrix / dependency DAG
LAYOUT_OFFSET_MET = 301            # metastasis module: the host-parenchyma baseline expression
                                   # profile for the metastatic deposit (celltype_exps["host"])
LAYOUT_OFFSET_IMMUNE_SUBTYPES = 401  # the baseline expression of the myeloid and B/plasma immune
                                   # subtypes (celltype_exps["myeloid"], ["bplasma"])
LAYOUT_OFFSET_STROMAL_SUBTYPES = 402  # the baseline expression of the endothelial and perivascular
                                   # stromal subtypes (celltype_exps["endothelial"], ["pvl"])
RUN_OFFSET_IMMUNE_SPLIT = 6101     # per run: the split of each deme's immune cells into subtypes
RUN_OFFSET_STROMAL_SPLIT = 6102    # per run: the split of each deme's stromal cells into subtypes, and
                                   # of each eviction of stromal cells among the subtypes

# Immune cells come in three subtypes, each a static normal genotype whose type is "immune": T and NK
# cells (genotype "immune", the only ones that kill cancer cells), myeloid cells (macrophages,
# monocytes, dendritic and mast cells) and B and plasma cells. Genotype id -> the subtype's label.
IMMUNE_SUBTYPES = {"immune": "T_NK", "myeloid": "myeloid", "bplasma": "B_plasma"}

# Stromal cells come in three subtypes, each a static normal genotype whose type is "stromal" and
# which all count as stroma for the stromal hazard and for eviction: fibroblasts (genotype
# "stromal"), endothelial cells and perivascular cells (pericytes and vascular smooth muscle cells).
# Genotype id -> the subtype's label.
STROMAL_SUBTYPES = {"stromal": "fibroblast", "endothelial": "endothelial", "pvl": "PVL"}

normal_cmap = {'epithelial': 'green', 'immune': 'yellow', 'stromal': 'pink', 'host': 'lightgray',
               'myeloid': 'orange', 'bplasma': 'khaki', 'endothelial': 'salmon', 'pvl': 'rosybrown'}
normal_names = list(normal_cmap.keys())
normal_colors = list(normal_cmap.values())
normal_colors_rgba = np.array([np.array(mcolors.to_rgba(color)) for color in normal_colors])
normal_cmap_rgba = dict(zip(normal_names, normal_colors_rgba))