"""Masked event audit and train-only scales for variants with unresolved labels."""
from pathlib import Path
import numpy as np

from power_fm_foresight._config import load_config, read_json, sha256, write_json
from power_fm_foresight._data.network import load_network
from power_fm_foresight._data.coordinates import fit_scales
from .event_helpers import event_values, daily_union, component_support
from power_fm_foresight._training.physics import target_values
from .event_policy import reporting_policy


def counts(x):
    valid = np.isfinite(x)
    n = int(valid.sum())
    positive = int(np.nansum(x))
    return {"positive": positive, "negative": n - positive, "denominator": n,
            "unknown": int(x.size - n), "rate": positive / n if n else None}


def run(cfg):
    root = Path(cfg["paths"]["temporal_root"])
    out = Path(cfg["paths"]["stage_a_run"])
    dest = out / "stage_a_followup"
    manifest, net = read_json(root / "manifest.json"), load_network(config=cfg)
    groups, train, train_hashes, unresolved = {}, {}, {}, []
    train_records = []
    for e in manifest["episodes"]:
        path = root / f"episodes/{e['episode_id']}.json"
        records = read_json(path)["records"]
        values = target_values(records, net)
        events = event_values(values, net, bus_ids=tuple(net["bus_ids"].tolist()), line_ids=tuple(range(1, net.n_branch+1)))
        record = {}
        for f, (event, valid) in events.items():
            hourly = np.where(valid, event.astype(float), np.nan)
            day, known = daily_union(event[72:].reshape(18, 24, -1), valid[72:].reshape(18, 24, -1))
            record[f] = {"hourly_all": hourly, "hourly": hourly[72:], "daily_any": np.where(known, day.astype(float), np.nan)}
        record["loading"] = values["loading"]
        record["vm"] = values["vm"]
        groups.setdefault(e["split"], []).append(record)
        for r in records:
            if not r["state_available"]:
                unresolved.append({"episode_id": e["episode_id"], "hour": r["hour"], "status": r["solver"]["status"]})
        if e["split"] == "train":
            train_records.extend(records)
            train_hashes[f"episodes/{e['episode_id']}.json"] = sha256(path)
            for key in ("pd_mw", "qd_mvar", "vm", "va_rad", "pg_mw", "qg_mvar", "pf_mw", "qf_mvar", "pt_mw", "qt_mvar"):
                train.setdefault(key, []).append(values[key])
    combined = {s: {f: {r: np.concatenate([e[f][r] for e in episodes]) for r in ("hourly_all", "hourly", "daily_any")}
                   for f in ("voltage", "thermal")} for s, episodes in groups.items()}
    scales = {"benchmark_id": manifest["benchmark_id"], "fit_partition": "train", "unique_hours": sum(len(x) for x in train["pd_mw"]),
        "training_shards_sha256": train_hashes, "rule": "max(population std of finite unique train coordinates, physical floor); tile over 24 hours",
        "coordinate_order": [], "scale_vector_one_hour": [], "features": {}}
    scales.update(fit_scales(net, train_records))
    write_json(dest / "train_coordinate_scales.json", scales)
    support = {f: component_support(combined["risk_calibration"][f]["hourly"].reshape(-1, 24, combined["risk_calibration"][f]["hourly"].shape[-1]), "hourly") for f in ("voltage", "thermal")}
    daily_support = {f: component_support(combined["risk_calibration"][f]["daily_any"], "daily_any") for f in ("voltage", "thermal")}
    events = {"dataset_id": cfg["dataset"]["id"], "benchmark_id": manifest["benchmark_id"], "unresolved_labels": unresolved,
        "mask_policy": "Unavailable labels remain unknown, never negative; daily-any is known positive if any resolved positive, known negative only if all hours resolved negative.",
        "support": support, "daily_support": daily_support, "buses": [], "lines": [], "partitions": {}}
    for f, key, n in (("voltage", "buses", net.n_bus), ("thermal", "lines", net.n_branch)):
        for j in range(n):
            row = {"id": int(net.bus_ids[j]) if f == "voltage" else j + 1, "partitions": {s: {r: counts(v[f][r][:, j]) for r in ("hourly_all", "hourly", "daily_any")} for s, v in combined.items()},
                "calibration_hourly_support": {k: v[j] for k, v in support[f].items()},
                "calibration_daily_support": {k: v[j] for k, v in daily_support[f].items()}}
            if f == "thermal":
                loading = np.concatenate([e["loading"][:, j] for eps in groups.values() for e in eps])
                row.update(from_bus=int(net["bus_ids"][int(net["branch"][j, 0])]), to_bus=int(net["bus_ids"][int(net["branch"][j, 1])]),
                    max_loading=float(loading[np.isfinite(loading)].max()) if np.isfinite(loading).any() else None,
                    binding_hours=int((abs(loading[np.isfinite(loading)] - 1) <= 1e-6).sum()),
                    rated=bool(net.rated_branches[j]))
            events[key].append(row)
    for s, vals in combined.items():
        events["partitions"][s] = {}
        for r, x in vals["thermal"].items():
            # Union across lines: unresolved if no known positive and not all lines known.
            any_event = np.where((x == 1).any(1), 1., np.where(np.isfinite(x).all(1), 0., np.nan))
            events["partitions"][s]["thermal_" + r] = counts(any_event)
    write_json(dest / "events.json", events)
    settings = cfg["stage_b"]["calibration"]
    ids = settings.get("headline_voltage_bus_ids", [int(net.bus_ids[i]) for i, keep in enumerate(support["voltage"]["eligible"]) if keep])
    line_ids = settings.get("continuous_loading_line_ids", [i+1 for i, keep in enumerate(support["thermal"]["eligible"]) if keep and net.rated_branches[i]])
    write_json(dest/"headline_components.json", dict(voltage_bus_ids=ids, thermal_line_ids=line_ids, selection="calibration support only"))
    rows, lookup = [], {}
    for f, names, indices in (("voltage", [f"voltage_bus_{b}" for b in ids], [net.bus_ids.tolist().index(b) for b in ids]), ("thermal", [f"thermal_line_{i}" for i in line_ids], [i-1 for i in line_ids])):
        for event, j in zip(names, indices):
            for s, vals in combined.items():
                for resolution, x in vals[f].items():
                    c = counts(x[:, j])
                    row = {"event": event, "resolution": resolution, "partition": s, "positive": c["positive"], "total": c["denominator"], "unknown": c["unknown"], "prevalence": c["rate"]}
                    rows.append(row)
                    lookup[event, resolution, s] = row
    roles = []
    for event in [f"voltage_bus_{b}" for b in ids] + [f"thermal_line_{i}" for i in line_ids]:
        for r in ("hourly", "daily_any"):
            c, t = lookup[event, r, "risk_calibration"], lookup[event, r, "test"]
            roles.append({"event": event, "resolution": r, "calibration_prevalence": c["prevalence"], "test_prevalence": t["prevalence"],
                **reporting_policy(c, t, r, cfg["stage_b"]["calibration"])})
    write_json(out / "stage_b_prevalence.json", {"prevalence": rows, "reporting_roles": roles, "unresolved_labels": unresolved,
        "manifest_sha256": sha256(root / "manifest.json"), "training_blocker": False, "selection": "User-frozen; no event/split tuning"})
    lines = ["# V4 event prevalence", "", "Unavailable solved labels remain masked. Rows are positive / resolved count (unknown count). No prevalence-based tuning.", "", "| Event | Resolution | Train | Selection | Calibration | Test |", "|---|---|---:|---:|---:|---:|"]
    for event in [f"voltage_bus_{b}" for b in ids] + [f"thermal_line_{i}" for i in line_ids]:
        for r in ("hourly", "daily_any"):
            cells = [lookup[event, r, s] for s in ("train", "model_selection", "risk_calibration", "test")]
            lines.append(f"| {event} | {r} | " + " | ".join(f"{v['positive']}/{v['total']} ({v['unknown']})" for v in cells) + " |")
    (out / "stage_b_prevalence.md").write_text("\n".join(lines) + "\n")
