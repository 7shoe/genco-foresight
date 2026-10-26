"""Frozen headline aggregation over the selected coordinates."""
import numpy as np

METRICS = ("mae", "mse", "crps", "coverage_50", "coverage_90", "width_50", "width_90")

def subset_summary(result, key, indices):
    row = result["variables"][key]["by_component"]
    counts = np.asarray(row["count"])[indices]
    valid = counts > 0
    return {"count": int(counts.sum()), **{k: float(np.average(np.asarray(row[k], dtype=float)[indices][valid],
             weights=counts[valid])) if valid.any() else None for k in METRICS}}
