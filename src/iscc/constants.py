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

normal_cmap = {'epithelial': 'green', 'immune': 'yellow', 'stromal': 'pink'}
normal_names = list(normal_cmap.keys())
normal_colors = list(normal_cmap.values())
normal_colors_rgba = np.array([np.array(mcolors.to_rgba(color)) for color in normal_colors])
normal_cmap_rgba = dict(zip(normal_names, normal_colors_rgba))