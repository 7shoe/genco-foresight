"""Physical coordinate identities and masks derived from a GridSpec."""
import numpy as np

from power_fm_foresight._solver.checks import physical_quantities


def bus_qg(grid, qg):
    """Aggregate solver generator labels in Graphkit's bus-Qg convention."""
    qg = np.asarray(qg, dtype=float)
    result = np.zeros((*qg.shape[:-1], grid.n_bus))
    for i in np.flatnonzero(grid.online_gen):
        result[..., grid.gen_bus[i]] += qg[..., i]
    return result[..., grid.qg_buses]


def record_values(grid, record):
    state = record.get("target_state") or {k: np.full(n, np.nan) for k, n in
        (("vm", grid.n_bus), ("va_rad", grid.n_bus), ("pg_mw", grid.n_gen), ("qg_mvar", grid.n_gen))}
    state = {k: np.asarray(v, dtype=float) for k, v in state.items()}
    physics = physical_quantities(grid, record["pd_mw"], record["qd_mvar"], state)
    values = dict(pd_mw=np.asarray(record["pd_mw"]), qd_mvar=np.asarray(record["qd_mvar"]), **state,
                  **{k: physics[k] for k in ("pf_mw", "qf_mvar", "pt_mw", "qt_mvar", "loading")})
    values["qg_mvar"] = bus_qg(grid, state["qg_mvar"])
    values["pg_mw"] = np.where(grid.online_gen, state["pg_mw"], np.nan)
    for key in ("pf_mw", "qf_mvar", "pt_mw", "qt_mvar"):
        values[key] = np.where(grid.online_branch, values[key], np.nan)
    return values


def fit_scales(grid, records):
    """Fit the released scale rule to training labels only, keeping offline masks."""
    values = [record_values(grid, r) for r in records]
    result = dict(features={}, coordinate_order=[], scale_vector_one_hour=[], generator_history={})
    for key in values[0]:
        if key == "loading": continue
        array = np.asarray([v[key] for v in values])
        mask = np.isfinite(array).any(0)
        mean = np.divide(np.nansum(array, axis=0), np.isfinite(array).sum(0),
                         out=np.zeros(array.shape[1]), where=mask)
        std = np.sqrt(np.divide(np.nansum((array-mean)**2, axis=0), np.isfinite(array).sum(0),
                                out=np.zeros(array.shape[1]), where=mask))
        floor = .01 if key in ("vm", "va_rad") else 1.
        scale = np.maximum(std, floor)
        if key == "pd_mw": mask &= grid.active_demand
        elif key == "qd_mvar": mask[:] = False
        elif key == "va_rad": mask[grid.reference_bus] = False
        elif key == "pg_mw": mask &= grid.online_gen & (grid.gen[:, 8] > grid.gen[:, 9])
        result["features"][key] = dict(mean=mean.tolist(), population_std=std.tolist(), floor=floor,
                                        scale=scale.tolist(), joint_score_mask=mask.tolist())
        for i in np.flatnonzero(mask):
            result["coordinate_order"].append(f"{key}:{i+1}")
            result["scale_vector_one_hour"].append(float(scale[i]))
    for key in ("pg_mw", "qg_mvar"):
        array = np.array([(r.get("target_state") or {}).get(key, [np.nan]*grid.n_gen) for r in records], dtype=float)
        valid = np.isfinite(array).sum(0)
        mean = np.divide(np.nansum(array, axis=0), valid, out=np.zeros(grid.n_gen), where=valid>0)
        std = np.sqrt(np.divide(np.nansum((array-mean)**2, axis=0), valid, out=np.zeros(grid.n_gen), where=valid>0))
        result["generator_history"][key] = dict(mean=mean.tolist(), scale=np.maximum(std, 1.).tolist())
    result.update(dimension_one_hour=len(result["coordinate_order"]),
                  dimension_24_hours=24*len(result["coordinate_order"]),
                  qg_coordinate_bus_ids=grid.bus_ids[grid.qg_buses].tolist(), fit_partition="train")
    return result


def event_support(grid, records):
    """Apply component support rules using valid hours and complete calendar days."""
    voltage = []; thermal = []; hours = []
    for r in records:
        if not r["state_available"]: continue
        v = record_values(grid, r)
        lower, upper = grid.bus[:, 12], grid.bus[:, 11]
        voltage.append((v["vm"] <= lower+.05*(upper-lower)) | (v["vm"] >= upper-.05*(upper-lower)))
        thermal.append(v["loading"] > .95); hours.append(r["hour"])
    result = {}
    for name, array, ids in (("voltage", voltage, grid.bus_ids), ("thermal", thermal, np.arange(grid.n_branch)+1)):
        if not array: result[name] = []; continue
        array = np.asarray(array); days = np.array(hours)//24; rows = []
        for i, identity in enumerate(ids):
            if name == "thermal" and not grid.rated_branches[i]: continue
            positives = sum(array[days==d, i].any() for d in np.unique(days))
            negatives = sum((~array[days==d, i]).any() for d in np.unique(days))
            rate = float(array[:, i].mean())
            rows.append(dict(id=int(identity), positive_hours=int(array[:, i].sum()), valid_hours=len(array),
                             prevalence=rate, positive_days=int(positives), negative_days=int(negatives),
                             supported=bool(positives>=5 and negatives>=5 and rate<.95)))
        result[name] = rows
    return result
