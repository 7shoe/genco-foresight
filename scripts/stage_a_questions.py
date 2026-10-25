"""Component event support and train-only scales before the Stage B implementation."""
from pathlib import Path
import csv

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common import ROOT, load_config, read_json, sha256, write_json
from data.loader.dataset import daily_origins
from data.loader.network import load_network
from solver.checks import physical_quantities


def count(values):
    positive = int(np.sum(values))
    return {"positive": positive, "negative": int(np.size(values))-positive,
            "denominator": int(np.size(values)), "rate": positive/np.size(values)}


def figure_support(report, daily_range=None):
    excluded = [r["id"] for r in report["buses"] if r["train_hourly"]["rate"] > .95
                or r["calibration"]["hourly_forecast"]["rate"] > .95]
    return {"rule": "Exclude voltage components over95% in train or calibration forecast hours; also require5positive/5negative distinct calibration days; no test-based selection.",
        "voltage_prevalence_exclusions": excluded,
        "daily_voltage_eligible": [r["id"] for r in report["buses"] if r["id"] not in excluded and r["calibration"]["daily_5positive_5negative"]
            and (daily_range is None or daily_range[0] <= r["calibration"]["daily_any"]["rate"] <= daily_range[1])],
        "hourly_voltage_eligible": [r["id"] for r in report["buses"] if r["id"] not in excluded and r["calibration"]["hourly_5days_each_class"]],
        "daily_thermal_eligible": [r["id"] for r in report["lines"] if r["calibration"]["daily_5positive_5negative"]
            and (daily_range is None or daily_range[0] <= r["calibration"]["daily_any"]["rate"] <= daily_range[1])],
        "daily_calibration_prevalence_range": daily_range,
        "hourly_thermal_eligible": [r["id"] for r in report["lines"] if r["calibration"]["hourly_5days_each_class"]]}


def main():
    cfg = load_config()
    if "grid" in cfg:
        from scripts.variant_events import run
        return run(cfg)
    root = Path(cfg["paths"]["temporal_root"])
    manifest = read_json(root/"manifest.json")
    hashes = {r["file"]: r["sha256"] for r in manifest["shards"]}
    net = load_network(config=load_config())
    buses, edges = net["bus"], net["branch"]
    bus_ids = net["bus_ids"].tolist()
    profile = pd.read_csv(Path(cfg["paths"]["assets_root"])/"profiles/eia_erco_demand_2025.csv")
    ref = cfg["generation"]["scale_low"] + (profile.demand_mw.to_numpy()-manifest["train_profile_min"])*(
        cfg["generation"]["scale_high"]-cfg["generation"]["scale_low"])/(manifest["train_profile_max"]-manifest["train_profile_min"])
    groups = {}
    train_values = {}
    training_shards = {}
    for e in manifest["episodes"]:
        name = f"episodes/{e['episode_id']}.json"
        assert sha256(root/name) == hashes[name]
        records = read_json(root/name)["records"]
        assert len(records) == 504 and all(r["state_available"] for r in records)
        vm = np.array([r["target_state"]["vm"] for r in records])
        q = [physical_quantities(net, r["pd_mw"], r["qd_mvar"], r["target_state"]) for r in records]
        loading = np.array([x["loading"] for x in q])
        thermal = loading > .95
        voltage = (vm <= buses[:,12]+.05*(buses[:,11]-buses[:,12])) | (vm >= buses[:,11]-.05*(buses[:,11]-buses[:,12]))
        voltage_binding = np.minimum(abs(vm-buses[:,12]), abs(vm-buses[:,11])) <= 1e-6
        thermal_binding = abs(loading-1.) <= 1e-6
        origins = daily_origins()
        day_t = np.array([thermal[o:o+24].any(0) for o in origins])
        day_v = np.array([voltage[o:o+24].any(0) for o in origins])
        # Hourly-event fitting may have both classes on a single distinct day.
        day_t_negative_present = np.array([(~thermal[o:o+24]).any(0) for o in origins])
        day_v_negative_present = np.array([(~voltage[o:o+24]).any(0) for o in origins])
        values = {"thermal": thermal, "voltage": voltage, "vm_binding": voltage_binding,
            "thermal_binding": thermal_binding, "loading": loading, "day_t": day_t, "day_v": day_v,
            "day_t_negative_present": day_t_negative_present, "day_v_negative_present": day_v_negative_present,
            "forecast_thermal": np.concatenate([thermal[o:o+24] for o in origins]),
            "forecast_voltage": np.concatenate([voltage[o:o+24] for o in origins]),
            "ref_max": np.array([ref[e["source_start_hour"]+o:e["source_start_hour"]+o+24].max() for o in origins]),
            "episode_id": e["episode_id"]}
        groups.setdefault(e["split"], []).append(values)
        if e["split"] == "train":
            training_shards[name] = hashes[name]
            for key in ("pd_mw", "qd_mvar"):
                train_values.setdefault(key, []).extend(r[key] for r in records)
            for key in ("vm", "va_rad", "pg_mw", "qg_mvar"):
                train_values.setdefault(key, []).extend(r["target_state"][key] for r in records)
            for key in ("pf_mw", "qf_mvar", "pt_mw", "qt_mvar"):
                train_values.setdefault(key, []).extend(x[key] for x in q)
    joined = {split: {key: np.concatenate([e[key] for e in episodes]) for key in episodes[0] if key != "episode_id"}
              for split, episodes in groups.items()}
    all_values = {key: np.concatenate([part[key] for part in joined.values()]) for key in joined["train"]}
    out = Path(cfg["paths"].get("evidence_root", ROOT/"reports"))/"stage_a_followup"
    out.mkdir(parents=True, exist_ok=True)
    report = {"benchmark_id": manifest["benchmark_id"], "dataset_manifest_sha256": sha256(root/"manifest.json"),
        "thresholds": {"voltage_outer_fraction": .05, "thermal_gt": .95, "binding_absolute_tolerance": 1e-6},
        "denominators": "All hours include the first 72 observed hours; forecast hours/days exclude that history.",
        "bus_ids": bus_ids, "line_ids": "1-based original MATPOWER branch row; endpoints use original bus IDs",
        "partitions": {}, "buses": [], "lines": []}
    for split, values in joined.items():
        report["partitions"][split] = {
            "thermal_hourly_all": count(values["thermal"].any(1)),
            "thermal_hourly_forecast": count(values["forecast_thermal"].any(1)),
            "thermal_daily_any": count(values["day_t"].any(1)),
            "component_counts_by_episode": [{"episode_id": e["episode_id"],
                "thermal_daily_positive": e["day_t"].sum(0).tolist(),
                "voltage_daily_positive": e["day_v"].sum(0).tolist()} for e in groups[split]]}
    for family, output, width, hourly, daily in (("voltage", "buses", net.n_bus, "voltage", "day_v"),
                                               ("thermal", "lines", len(edges), "thermal", "day_t")):
        for i in range(width):
            item = {"id": bus_ids[i] if family == "voltage" else i+1,
                "all_hourly": count(all_values[hourly][:,i]), "train_hourly": count(joined["train"][hourly][:,i])}
            if family == "thermal":
                item.update({"from_bus": bus_ids[int(edges[i,0])], "to_bus": bus_ids[int(edges[i,1])],
                    "rating_mva": float(edges[i,5]), "rated": bool(edges[i,5]>0),
                    "max_loading": float(all_values["loading"][:,i].max()),
                    "binding_hours": int(all_values["thermal_binding"][:,i].sum()),
                    "exactly_one_hours": int((all_values["loading"][:,i] == 1.).sum()),
                    "binding_fraction_among_stressed": float(all_values["thermal_binding"][:,i].sum()/max(1,all_values[hourly][:,i].sum()))})
            else:
                item.update({"bound_pinned_hours": int(all_values["vm_binding"][:,i].sum()),
                    "degenerate_over_95pct_all_hours": bool(all_values[hourly][:,i].mean()>.95),
                    "exclude_calibration_train_over_95pct": bool(joined["train"][hourly][:,i].mean()>.95)})
            cal = joined["risk_calibration"]
            day_counts = count(cal[daily][:,i])
            negative_present = int(cal[daily+"_negative_present"][:,i].sum())
            item["calibration"] = {"hourly_all": count(cal[hourly][:,i]),
                "hourly_forecast": count(cal["forecast_"+hourly][:,i]), "daily_any": day_counts,
                "days_with_negative_hour": negative_present,
                "daily_5positive_5negative": day_counts["positive"]>=5 and day_counts["negative"]>=5,
                "hourly_5days_each_class": day_counts["positive"]>=5 and negative_present>=5}
            report[output].append(item)
    tr, cal = joined["train"], joined["risk_calibration"]
    mean, std = float(tr["ref_max"].mean()), float(tr["ref_max"].std())
    train_labels, cal_labels = tr["day_t"].any(1), cal["day_t"].any(1)
    classifier = None
    probabilities = None
    if len(np.unique(train_labels)) == 2 and std > 0:
        classifier = LogisticRegression(C=1., solver="lbfgs", random_state=42, max_iter=1000)
        classifier.fit(((tr["ref_max"]-mean)/std)[:,None], train_labels)
        probabilities = classifier.predict_proba(((cal["ref_max"]-mean)/std)[:,None])[:,1]
    auc_supported = probabilities is not None and len(np.unique(cal_labels)) == 2
    report["profile_diagnostic"] = {"model": "L2 logistic C=1, standardized max realized daily ref_t only",
        "fit_partition": "train", "score_partition": "risk_calibration", "train_days": len(tr["ref_max"]),
        "calibration_days": len(cal["ref_max"]), "auroc": float(roc_auc_score(cal_labels, probabilities)) if auc_supported else None,
        "auroc_supported": auc_supported,
        "unavailable_reason": None if auc_supported else "Training/profile variance or calibration class support is insufficient",
        "feature_mean": mean, "feature_std": std, "coefficient": classifier.coef_.tolist() if classifier else None,
        "intercept": classifier.intercept_.tolist() if classifier else None,
        "probabilities": probabilities.tolist() if probabilities is not None else None, "labels": cal_labels.astype(int).tolist(),
        "ref_max": cal["ref_max"].tolist(),
        "interpretation": "Oracle structural diagnostic: realized next-day profile is forbidden as a forecasting input; does not establish S forecast calibration."}
    report["top_5_lines"] = [r["id"] for r in sorted(report["lines"], key=lambda r:(-r["all_hourly"]["positive"],-r["max_loading"]))[:5]]
    report["excluded_voltage_buses"] = [r["id"] for r in report["buses"] if r["exclude_calibration_train_over_95pct"]]
    report["calibration_figures"] = figure_support(report, cfg["stage_b"]["calibration"].get("daily_calibration_prevalence_range"))
    for key in ("buses", "lines"):
        with (out/f"{key}.csv").open("w") as stream:
            columns = ["id", "all_positive", "all_hours", "all_fraction", "train_fraction", "cal_positive", "cal_hours",
                "cal_forecast_positive", "cal_forecast_hours", "cal_daily_positive", "cal_daily_negative", "daily_5plus_5plus"]
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for r in report[key]:
                c = r["calibration"]
                writer.writerow(dict(zip(columns, [r["id"],r["all_hourly"]["positive"],r["all_hourly"]["denominator"],r["all_hourly"]["rate"],
                    r["train_hourly"]["rate"],c["hourly_all"]["positive"],c["hourly_all"]["denominator"],
                    c["hourly_forecast"]["positive"],c["hourly_forecast"]["denominator"],c["daily_any"]["positive"],
                    c["daily_any"]["negative"],c["daily_5positive_5negative"]])))
    write_json(out/"events.json", report)
    scale = {"benchmark_id": manifest["benchmark_id"], "fit_partition": "train", "unique_hours": len(train_values["pd_mw"]), "training_shards_sha256": training_shards,
             "rule": "max(population standard deviation over unique training hours, physical floor); tile across 24 horizons",
             "coordinate_order": [], "scale_vector_one_hour": [], "features": {}}
    for key, values in train_values.items():
        values = np.array(values)
        floor = .01 if key in ("vm", "va_rad") else 1.
        scales = np.maximum(values.std(0), floor)
        ids = list(range(1,values.shape[1]+1))
        mask = np.ones(values.shape[1], dtype=bool)
        if key == "pd_mw": mask = buses[:,2] != 0
        elif key == "qd_mvar": mask[:] = False  # Deterministic power-factor copy; score physical marginals separately.
        elif key == "va_rad": mask[net["reference_bus"]] = False
        elif key == "pg_mw": mask = net["gen"][:,8] > net["gen"][:,9]
        scale["features"][key] = {"mean": values.mean(0).tolist(), "population_std": values.std(0).tolist(),
            "floor": floor, "scale": scales.tolist(), "joint_score_mask": mask.tolist()}
        for i in np.flatnonzero(mask):
            scale["coordinate_order"].append(f"{key}:{ids[i]}")
            scale["scale_vector_one_hour"].append(float(scales[i]))
    scale["dimension_one_hour"] = len(scale["coordinate_order"])
    scale["dimension_24_hours"] = 24*scale["dimension_one_hour"]
    write_json(out/"train_coordinate_scales.json", scale)
    print({"profile_auroc": report["profile_diagnostic"]["auroc"], "excluded_buses": report["excluded_voltage_buses"],
           "top_5_lines": report["top_5_lines"], "dimension": scale["dimension_one_hour"]}, flush=True)


if __name__ == "__main__":
    main()
