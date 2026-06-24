from ..tumor import Tumor

import numpy as np
from copy import deepcopy
from tqdm import tqdm

class MixedTumor(Tumor):
    def __init__(self, **tumor_kwargs):
        # NB: must be super(MixedTumor, self); super(Tumor, self) would skip
        # Tumor.__init__ entirely and run object.__init__.
        super(MixedTumor, self).__init__(**tumor_kwargs)
        self.type = 'mixed'
        # TODO: MixedTumor is still incomplete — it needs a single-deme "grid" set up
        # (make_grid / get_neighboring_demes) before grow() will work. Tracked for the
        # sampling/spatial-models milestone.
