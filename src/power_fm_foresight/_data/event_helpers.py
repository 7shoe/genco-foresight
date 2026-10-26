"""Original masked event definitions and mechanical calibration support."""
import numpy as np

def event_values(values, network, bus_ids=None, line_ids=None):
    bus_ids = network.bus_ids if bus_ids is None else bus_ids
    line_ids = np.flatnonzero(network.rated_branches)+1 if line_ids is None else line_ids
    buses = [network["bus_ids"].tolist().index(b) for b in bus_ids]
    lines = np.array(line_ids) - 1
    b = network["bus"][buses]
    vm = np.asarray(values["vm"])[..., buses]
    loading = np.asarray(values["loading"])[..., lines]
    rated = network.rated_branches[lines]
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
    supported = (positive_days >= 5) & (negative_days >= 5) & (rate < .95)
    if resolution == "daily_any":
        supported &= (rate >= .2) & (rate <= .8)
    return {"positive_days": positive_days.tolist(), "negative_days": negative_days.tolist(),
            "positive": positives.astype(int).tolist(), "total": total.tolist(),
            "prevalence": [float(v) if np.isfinite(v) else None for v in rate], "eligible": supported.tolist()}
