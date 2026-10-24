"""Named physical events and calibration-only sigmoid maps; no plotting or fitting forecasts."""
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression


RELIABILITY_EDGES = np.array([0., .1, .3, .5, .7, .9, 1.])
LOGIT_EPS = 1e-4


def event_values(values, network, bus_ids=(1, 9, 11, 13, 15, 21, 22), line_ids=(1,)):
    buses = [network["bus_ids"].tolist().index(b) for b in bus_ids]
    lines = np.array(line_ids) - 1
    b = network["bus"][buses]
    vm = np.asarray(values["vm"])[..., buses]
    loading = np.asarray(values["loading"])[..., lines]
    rated = network["branch"][lines, 5] > 0
    voltage = (vm <= b[:, 12] + .05 * (b[:, 11] - b[:, 12])) | (vm >= b[:, 11] - .05 * (b[:, 11] - b[:, 12]))
    thermal = loading > .95
    return {"voltage": (voltage, np.isfinite(vm)),
            "thermal": (thermal, np.isfinite(loading) & rated)}


def daily_union(events, available):
    """Hourly axis is penultimate; a known positive resolves a partial day."""
    events, available = np.asarray(events, bool), np.asarray(available, bool)
    positive = (events & available).any(axis=-2)
    valid = positive | available.all(axis=-2)
    return positive, valid


def event_probabilities(values, network, bus_ids=(1, 9, 11, 13, 15, 21, 22), line_ids=(1,), weights=None):
    result = {}
    m = len(values["vm"])
    weights = np.ones(m) / m if weights is None else np.asarray(weights, float)
    if weights.shape != (m,) or not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1):
        raise ValueError("Weights must describe complete joint trajectories and sum to one")
    for family, (event, valid) in event_values(values, network, bus_ids, line_ids).items():
        if not valid.all():
            raise ValueError("A forecast event trajectory is undefined; do not silently renormalize missing samples")
        daily, _ = daily_union(event, valid)
        result[family] = {"hourly": np.tensordot(weights, event, axes=1),
                          "daily_any": np.tensordot(weights, daily, axes=1)}
    return result


def event_targets(values, network, bus_ids=(1, 9, 11, 13, 15, 21, 22), line_ids=(1,)):
    result = {}
    for family, (event, valid) in event_values(values, network, bus_ids, line_ids).items():
        daily, daily_valid = daily_union(event, valid)
        result[family] = {"hourly": np.where(valid, event.astype(float), np.nan),
                          "daily_any": np.where(daily_valid, daily.astype(float), np.nan)}
    return result


def component_support(target, resolution):
    y = np.asarray(target, float)
    valid = np.isfinite(y)
    if resolution == "hourly":
        positive_days = ((y == 1) & valid).any(axis=1).sum(axis=0)
        negative_days = ((y == 0) & valid).any(axis=1).sum(axis=0)
        total = valid.sum(axis=(0, 1))
        positives = np.where(valid, y, 0).sum(axis=(0, 1))
    elif resolution == "daily_any":
        positive_days, negative_days = (y == 1).sum(0), (y == 0).sum(0)
        total = valid.sum(0)
        positives = np.where(valid, y, 0).sum(0)
    else:
        raise ValueError(resolution)
    rate = np.divide(positives, total, out=np.full(total.shape, np.nan), where=total > 0)
    supported = (positive_days >= 5) & (negative_days >= 5)
    if resolution == "daily_any":
        supported &= (rate >= .2) & (rate <= .8)
    return {"positive_days": positive_days.tolist(), "negative_days": negative_days.tolist(),
            "positive": positives.astype(int).tolist(), "total": total.tolist(),
            "prevalence": [float(v) if np.isfinite(v) else None for v in rate], "eligible": supported.tolist()}


def logit_feature(p):
    p = np.clip(np.asarray(p, float), LOGIT_EPS, 1 - LOGIT_EPS)
    return np.log(p / (1 - p))


def fit_map(probabilities, targets, resolution, partition):
    if partition != "risk_calibration":
        raise ValueError("Post-hoc event maps must fit on risk_calibration only")
    p, y = np.asarray(probabilities), np.asarray(targets)
    if p.shape != y.shape:
        raise ValueError("Event samples and targets must align")
    support = component_support(y, resolution)
    valid = np.isfinite(y) & np.asarray(support["eligible"])
    record = {"family": "single pooled sigmoid on logit(raw_probability)", "resolution": resolution,
        "fit_partition": partition, "C": 1., "logit_clip": LOGIT_EPS, "support": support,
        "weighting": "Equal valid component-hours (or component-days); one map per mode/event family, not per component",
        "count": int(valid.sum()), "supported": bool(valid.any()), "coefficient": None, "intercept": None}
    if not valid.any():
        record["reason"] = "No components satisfy calibration day support and resolution rules"
        return record
    if not np.isfinite(p[valid]).all() or ((p[valid] < 0) | (p[valid] > 1)).any():
        raise ValueError("Invalid raw probability")
    model = LogisticRegression(C=1., solver="lbfgs", max_iter=1000, random_state=42)
    model.fit(logit_feature(p[valid])[:, None], y[valid].astype(int))
    record.update(coefficient=float(model.coef_[0, 0]), intercept=float(model.intercept_[0]),
                  iterations=int(model.n_iter_[0]), raw_probability_distinct_values=len(np.unique(p[valid])))
    return record


def apply_map(probabilities, mapping):
    p = np.asarray(probabilities)
    if not mapping["supported"]:
        return np.full(p.shape, np.nan)
    mapped = expit(mapping["coefficient"] * logit_feature(p) + mapping["intercept"])
    return np.where(np.asarray(mapping["support"]["eligible"]), mapped, np.nan)


def reliability(probabilities, targets, minimum_bin_count=20):
    p, y = np.asarray(probabilities).reshape(-1), np.asarray(targets).reshape(-1)
    valid = np.isfinite(p) & np.isfinite(y)
    p, y = p[valid], y[valid]
    bins = np.minimum(np.searchsorted(RELIABILITY_EDGES, p, side="right") - 1, len(RELIABILITY_EDGES) - 2)
    rows = []
    for i, (lo, hi) in enumerate(zip(RELIABILITY_EDGES[:-1], RELIABILITY_EDGES[1:])):
        keep = bins == i
        n = int(keep.sum())
        rows.append({"lower": float(lo), "upper": float(hi), "count": n,
            "mean_probability": float(p[keep].mean()) if n else None,
            "observed_frequency": float(y[keep].mean()) if n else None,
            "plot_supported": n >= minimum_bin_count})
    return {"count": len(p), "brier": float(((p - y) ** 2).mean()) if len(p) else None,
            "observed_prevalence": float(y.mean()) if len(y) else None,
            "mean_probability": float(p.mean()) if len(p) else None,
            "minimum_plotted_bin_count": minimum_bin_count, "bins": rows}
