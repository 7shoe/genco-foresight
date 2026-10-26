"""Recompute post headline numbers from saved reports; standard-library-only."""
import argparse
import hashlib
import json
from pathlib import Path


def collect(root):
    evidence = {}
    def read(relative):
        path = root / relative
        evidence[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    b = {m: read(f"stage_b/{m}.json") for m in ("C", "P", "P0", "S", "A", "D")}
    c = read("stage_c/report.json")
    d = read("stage_b/diagnostics/report.json")
    h = read("stage_c/hybrid/report.json")
    p = read("stage_b/probe/report.json")
    rows = [r for r in c["events"] if r["family"] == "voltage" and r["resolution"] == "hourly"]
    numbers = {"test_origins": b["A"]["origins"], "forecast_hours": d["ref_counts"]["forecast_hours"],
        "modes": {}, "hybrid": h, "embedding_probe": {k: {f: r["normalized"] for f, r in v["families"].items()} for k, v in p["horizons"].items()},
        "parameter_hash_accounting": d["parameter_hash_accounting"]}
    for m, score in b.items():
        vm = [v for key, v in c["continuous"]["test"][m].items() if key.startswith("voltage_bus_")]
        numbers["modes"][m] = {"joint_energy_score": score["joint_energy_score"],
            "hourly_voltage_macro_brier_raw": sum(r["modes"][m]["raw"]["brier"] for r in rows) / len(rows),
            "hourly_voltage_macro_brier_mapped": sum(r["modes"][m]["mapped"]["brier"] for r in rows) / len(rows),
            "headline_vm": {key: sum(r[key] * r["count"] for r in vm) / sum(r["count"] for r in vm)
                for key in ("crps", "coverage_50", "coverage_90", "width_50", "width_90")},
            "physical_balance": d["scopes"]["all"]["modes"][m]["balance"],
            "A_zero_stress_test_buses": [r["event"] for r in rows if r["modes"]["A"]["raw"]["mean_probability"] == 0] if m == "A" else None}
    numbers["energy_reduction_A_vs_C_fraction"] = 1 - b["A"]["joint_energy_score"] / b["C"]["joint_energy_score"]
    numbers["hybrid_warm_to_cold_total_work_ratio"] = h["arms"]["A_forecast_mean"]["total_work_seconds_excluding_shared_setup"] / h["arms"]["cold"]["total_work_seconds_excluding_shared_setup"]
    # Include every numerical result/qualification used outside the headline tables.
    numbers["stress_events"] = c["events"]
    numbers["joint_coordinate_groups"] = d["groups"]
    numbers["joint_coordinates_per_hour"] = sum(v["coordinates_per_hour"] for v in d["groups"].values())
    numbers["reference_profile_counts"] = d["ref_counts"]
    numbers["same_input_ood_bins"] = d["ood_P"]
    numbers["all_and_high_profile_diagnostics"] = d["scopes"]
    if (root / "post_contract.json").exists():
        numbers["method_and_dataset_contract"] = read("post_contract.json")
    result = {"primary_record": "v3", "numbers": numbers, "input_sha256": evidence,
        "source_note": "Macro voltage Brier averages the seven frozen components equally; equal resolved counts in v3. Event maps do not alter continuous coverage. Hybrid total includes forecast preparation, excludes separately reported shared setup."}
    if (root / "v4/stage_b/report.json").exists():
        result["headlines"] = release_headlines(root, read, b, c, d, h, p, numbers)
    return result


def release_headlines(root, read, b, c, d, h, probe, numbers):
    """Named numerical claims only; all measurements come from frozen reports."""
    v4 = read("v4/stage_b/report.json")
    context = read("stage_d/final_release/context.json")
    records = {}
    def add(key, value, unit, source):
        records[key] = {"value": value, "unit": unit, "source": source}
    for key, value in context["claims"].items():
        add(key, value["value"], value["unit"], "stage_d/final_release/context.json#claims/" + key)
    for version in ("v3", "v4"):
        for mode in ("C", "P", "P0", "S", "A", "D"):
            source = "stage_b/diagnostics/report.json" if version == "v3" else "v4/stage_b/report.json"
            if version == "v3":
                row = d["scopes"]["all"]["modes"][mode]
                vm = numbers["modes"][mode]["headline_vm"]
                pg = row["headline"]["pg_free"]
                groups = row["groups"] if mode in ("C", "S", "A", "D") else {}
                bal = row["balance"]
                es = b[mode]["joint_energy_score"]
            else:
                row = v4["modes"][mode]
                vm, pg, groups, bal, es = row["headline_vm"], row["free_generator_pg"], row["family_energy"], row["balance"], row["joint_energy_score"]
            values = {
                "joint_es": (es, "train-scaled score"),
                "vm_crps": (vm["crps"], "p.u."),
                "vm_coverage90_pct": (100 * vm["coverage_90"], "%"),
                "vm_coverage50_pct": (100 * vm["coverage_50"], "%"),
                "vm_width90": (vm["width_90"], "p.u."),
                "pg_free_mae": (pg["mae"], "MW"),
                "pg_free_coverage90_pct": (100 * pg["coverage_90"], "%"),
                "p_imbalance": (bal["residual_p_mw"]["mean_abs"], "MW/bus"),
                "q_imbalance": (bal["residual_q_mvar"]["mean_abs"], "MVAr/bus"),
            }
            for key, (value, unit) in values.items():
                if version == "v4":
                    location = source + "#modes/" + mode
                elif key == "joint_es":
                    location = f"stage_b/{mode}.json#joint_energy_score"
                elif key.startswith("vm_"):
                    location = f"stage_c/report.json#continuous/test/{mode} (count-weighted headline buses)"
                else:
                    location = source + "#scopes/all/modes/" + mode
                add(f"{version}.{mode}.{key}", value, unit, location)
            for group, values in groups.items():
                location = f"#scopes/all/modes/{mode}/groups/{group}" if version == "v3" else f"#modes/{mode}/family_energy/{group}"
                add(f"{version}.{mode}.family_es.{group}", values["group_energy_score"], "separately train-scaled score", source + location)
            if version == "v3":
                for kind in ("raw", "mapped"):
                    add(f"v3.{mode}.voltage_brier_{kind}", numbers["modes"][mode][f"hourly_voltage_macro_brier_{kind}"], "Brier score", "stage_c/report.json#hourly-voltage-macro")
        prevalence = read(f"stage_a_{version}/stage_b_prevalence.json")["prevalence"]
        events = list(dict.fromkeys(r["event"] for r in prevalence))
        for event in events:
            for resolution in ("hourly", "daily_any"):
                rows = [next(r for r in prevalence if r["event"] == event and r["resolution"] == resolution and r["partition"] == part)
                        for part in ("train", "model_selection", "risk_calibration", "test")]
                add(f"{version}.prevalence.{event}.{resolution}_pct", [100 * r["prevalence"] for r in rows],
                    "%; train, selection, calibration, test", f"stage_a_{version}/stage_b_prevalence.json")
                add(f"{version}.prevalence.{event}.{resolution}_counts",
                    [[r["positive"], r["total"], r.get("unknown", 0)] for r in rows],
                    "[positive,resolved,unknown]; train, selection, calibration, test", f"stage_a_{version}/stage_b_prevalence.json")
    for label, row in d["ood_P"].items():
        for key, unit in (("vm", "p.u."), ("pg_mw", "MW")):
            add(f"v3.ood.{label}.{key}_mae", row["same_input_errors"][key]["mae"], unit,
                "stage_b/diagnostics/report.json#ood_P/" + label)
    add("v3.test_hours_high_profile", d["ref_counts"]["forecast_hours"]["hours_ge_0_8"], "hours", "stage_b/diagnostics/report.json#ref_counts")
    add("v4.A_vs_S.joint_es_reduction_pct", 100 * (1-v4["modes"]["A"]["joint_energy_score"]/v4["modes"]["S"]["joint_energy_score"]), "%",
        "v4/stage_b/report.json#1-A.joint_energy_score/S.joint_energy_score")
    for name, group in d["groups"].items():
        add("joint.coordinates." + name, group["coordinates_per_hour"], "coordinates/hour", "stage_b/diagnostics/report.json#groups")
    add("joint.coordinates.total", numbers["joint_coordinates_per_hour"], "coordinates/hour", "stage_b/diagnostics/report.json#groups")
    for event in c["events"]:
        if event["resolution"] == "hourly":
            add("v3.prevalence_ratio." + event["event"], event["prevalence_ratio"], "larger/smaller", "stage_c/report.json#events")
    add("v3.hybrid.work_ratio", numbers["hybrid_warm_to_cold_total_work_ratio"], "warm/cold", "stage_c/hybrid/report.json#arms")
    for name, arm in h["arms"].items():
        add("v3.hybrid." + name + ".total_work_seconds", arm["total_work_seconds_excluding_shared_setup"], "seconds", "stage_c/hybrid/report.json#arms/" + name)
        add("v3.hybrid." + name + ".valid", arm["valid"], "valid solves", "stage_c/hybrid/report.json#arms/" + name)
        add("v3.hybrid." + name + ".iterations", arm["iterations_all_returned"]["mean"], "mean iterations", "stage_c/hybrid/report.json#arms/" + name)
        add("v3.hybrid." + name + ".request_ms", 1000*arm["request_seconds"]["mean"], "mean milliseconds", "stage_c/hybrid/report.json#arms/" + name)
    for horizon, row in probe["horizons"].items():
        for family, value in row["families"].items():
            add(f"v3.probe.h{horizon}.{family}.normalized_mse", value["normalized"]["mse"], "train-scaled MSE", "stage_b/probe/report.json#horizons/" + horizon)
    return records


def markdown_list(records):
    lines = ["# Headline number keys", "", "Generated from frozen saved reports; source paths are relative to evidence/runs in the release package.",
        "Keys next to numbers in post.md are retained in HTML comments for audit without cluttering the rendered post.", "",
        "| Key | Value | Unit / definition | Evidence |", "|---|---|---|---|"]
    for key, record in sorted(records.items()):
        value = json.dumps(record["value"], ensure_ascii=False, allow_nan=False)
        lines.append(f"| {key} | {value} | {record['unit']} | {record['source']} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs"))
    p.add_argument("--list", action="store_true", help="Print the complete named numerical-claim list as Markdown")
    a = p.parse_args()
    result = collect(a.root)
    print(markdown_list(result["headlines"]) if a.list else json.dumps(result, indent=2, allow_nan=False))
