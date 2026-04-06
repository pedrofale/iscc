import matplotlib.colors as mcolors
import numpy as np

normal_cmap = {'epithelial': 'green', 'immune': 'yellow', 'stromal': 'pink'}
normal_names = list(normal_cmap.keys())
normal_colors = list(normal_cmap.values())
normal_colors_rgba = np.array([np.array(mcolors.to_rgba(color)) for color in normal_colors])
normal_cmap_rgba = dict(zip(normal_names, normal_colors_rgba))