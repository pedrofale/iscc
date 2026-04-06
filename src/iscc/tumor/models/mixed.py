from ..tumor import Tumor

import numpy as np
from copy import deepcopy
from tqdm import tqdm

class MixedTumor(Tumor):
    def __init__(self, **tumor_kwargs):
        super(Tumor, self).__init__(**tumor_kwargs)
        self.type = 'mixed'
        # TODO: need set up the grid of one deme only
