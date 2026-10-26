"""The frozen per-origin scorer and equal-weight summary."""
import numpy as np
from power_fm_foresight.scoring import UNITS, evaluate, merge_origin_scores

def score(values, target, scales, pairs, mode, identity, index):
    row, arrays = evaluate(values, target, {k: np.isfinite(target[k]) for k in UNITS},
        dict(scales=scales, variogram_pairs=pairs, mode=mode, origin_id=identity,
             seed=10042+index+(10000 if mode in ('C', 'P0') else 0)))
    row['physical_balance'] = dict(p_mean_abs_mw=float(np.abs(values['residual_p_mw']).mean()),
                                  q_mean_abs_mvar=float(np.abs(values['residual_q_mvar']).mean()))
    return row, arrays



def merged(rows):
    result = merge_origin_scores(rows)
    result['physical_balance'] = {k: float(np.mean([r['physical_balance'][k] for r in rows]))
                                 for k in rows[0]['physical_balance']}
    return result



def mixture_samples(blocks):
    """Concatenate equal-size member draws before joint rescoring."""
    return {k: np.concatenate([a[k] for a in blocks]) for k in blocks[0]}
