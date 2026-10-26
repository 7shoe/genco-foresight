"""The frozen per-origin scorer and equal-weight summary."""
import numpy as np
from power_fm_foresight.scoring import UNITS, evaluate, merge_origin_scores

from power_fm_foresight.scoring import score



from power_fm_foresight.scoring import merged



def mixture_samples(blocks):
    """Concatenate equal-size member draws before joint rescoring."""
    return {k: np.concatenate([a[k] for a in blocks]) for k in blocks[0]}
