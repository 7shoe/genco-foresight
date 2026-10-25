"""Frozen event-family prevalence and calibration/test reporting roles."""
from pathlib import Path
import csv
from fractions import Fraction

import numpy as np

from common import ROOT, load_config, read_json, sha256, write_json
from data.loader.network import load_network
from solver.checks import physical_quantities


def prevalence_shift(calibration, test, threshold):
    a=Fraction(calibration["positive"],calibration["total"])
    b=Fraction(test["positive"],test["total"])
    if min(a,b)==0:
        return None, bool(max(a,b)>0)
    ratio=max(a,b)/min(a,b)
    return float(ratio),ratio>Fraction(str(threshold))


def reporting_policy(calibration, test, resolution, settings):
    ratio, shifted = prevalence_shift(calibration, test, settings["prevalence_shift_ratio"])
    low, high = settings.get("daily_calibration_prevalence_range", [0, 1])
    rate = Fraction(calibration["positive"], calibration["total"])
    eligible = resolution == "hourly" or Fraction(str(low)) <= rate <= Fraction(str(high))
    return {"ratio": ratio, "exceeds_1p5": shifted,
        "resolution_role": "primary" if resolution == "hourly" else "secondary",
        "panel_eligible_by_prevalence": eligible,
        "panel_exclusion_reason": None if eligible else "Daily calibration prevalence outside 20%-80%",
        "raw_frequency_role": "report_side_by_side",
        "posthoc_map_role": "report_side_by_side_with_shift_caption" if shifted else "report_side_by_side",
        "posthoc_fit_partition": "risk_calibration", "prevalence_table_required": True,
        "shift_caption_required": shifted, "training_blocker": False}


def main():
    cfg = load_config()
    if "grid" in cfg:
        from scripts.variant_events import run
        return run(cfg)
    root = Path(cfg["paths"]["temporal_root"])
    manifest = read_json(root/"manifest.json")
    hashes = {r["file"]: r["sha256"] for r in manifest["shards"]}
    net = load_network(config=load_config())
    ids = cfg["stage_b"]["calibration"]["headline_voltage_bus_ids"]
    selected = [net["bus_ids"].tolist().index(i) for i in ids]
    counts = {}
    for episode in manifest["episodes"]:
        name = f"episodes/{episode['episode_id']}.json"
        assert sha256(root/name) == hashes[name]
        records = read_json(root/name)["records"]
        assert len(records) == 504 and all(r["state_available"] for r in records)
        vm = np.array([r["target_state"]["vm"] for r in records])[:,selected]
        b = net["bus"][selected]
        voltage = (vm <= b[:,12]+.05*(b[:,11]-b[:,12])) | (vm >= b[:,11]-.05*(b[:,11]-b[:,12]))
        thermal = np.array([physical_quantities(net, r["pd_mw"], r["qd_mvar"], r["target_state"])["loading"][0] > .95 for r in records])[:,None]
        events = np.concatenate([voltage, thermal], axis=1)
        for j, event in enumerate([f"voltage_bus_{i}" for i in ids]+["thermal_line_1"]):
            for resolution, values in (("hourly_all",events[:,j]), ("hourly",events[72:,j]),
                                       ("daily_any", events[72:,j].reshape(18,24).any(1))):
                key = (event,resolution,episode["split"])
                record = counts.setdefault(key, {"event":event,"resolution":resolution,"partition":episode["split"],"positive":0,"total":0})
                record["positive"] += int(values.sum()); record["total"] += len(values)
    rows = list(counts.values())
    for row in rows: row["prevalence"] = row["positive"]/row["total"]
    roles = []
    threshold = cfg["stage_b"]["calibration"]["prevalence_shift_ratio"]
    for event in [f"voltage_bus_{i}" for i in ids]+["thermal_line_1"]:
        for resolution in ("hourly","daily_any"):
            a = counts[(event,resolution,"risk_calibration")]["prevalence"]
            b = counts[(event,resolution,"test")]["prevalence"]
            policy = reporting_policy(counts[(event,resolution,"risk_calibration")],
                                      counts[(event,resolution,"test")],resolution,cfg["stage_b"]["calibration"])
            roles.append({"event":event,"resolution":resolution,"calibration_prevalence":a,"test_prevalence":b,
                "ratio_infinite": min(a,b)==0 and max(a,b)>0, **policy})
    out = Path(cfg["paths"].get("evidence_root", ROOT/"reports"))
    out.mkdir(parents=True, exist_ok=True)
    report = {"benchmark_id":manifest["benchmark_id"],"dataset_manifest_sha256":sha256(root/"manifest.json"),
              "headline_bus_ids":ids,"definition":"hourly/daily_any use target hours72..503; hourly_all includes initial history",
              "prevalence":rows,"reporting_roles":roles,"selection":"User-frozen event family; no tuning on test prevalence",
              "thermal_oracle_evidence":str(out/"stage_a_followup/events.json")}
    write_json(out/"stage_b_prevalence.json",report)
    with (out/"stage_b_prevalence.csv").open("w") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=["# Frozen calibration event prevalence", "", "Rates use forecast-target hours and complete forecast days. Each cell is hourly / daily-any percent.", "",
           "| Event | Train | Model selection | Risk calibration | Test | >1.5× shift: hourly / daily |", "|---|---:|---:|---:|---:|---|"]
    for event in [f"voltage_bus_{i}" for i in ids]+["thermal_line_1"]:
        cells=[f"{100*counts[(event,'hourly',s)]['prevalence']:.2f} / {100*counts[(event,'daily_any',s)]['prevalence']:.2f}" for s in ("train","model_selection","risk_calibration","test")]
        shifts=["yes" if r["exceeds_1p5"] else "no" for r in roles if r["event"]==event]
        lines.append("| "+" | ".join([event,*cells," / ".join(shifts)])+" |")
    lines += ["", "Prevalence is a diagnostic, never a training blocker. Hourly voltage and line-1 events are primary. Display test raw and calibration-fitted mapped probabilities side by side, with the calibration/test prevalence table beside every reliability diagram. Flag ratios above 1.5× in the caption without suppressing panels. Daily-any is secondary and shown only when calibration prevalence is within [20%,80%], alongside the existing support checks. Full counts and all-hour rates are in the adjacent JSON/CSV.", "",
              "Consult [events.json](stage_a_followup/events.json) for this dataset's binding-corridor counts and realized-profile AUROC. The realized next-day profile is an oracle diagnostic, not a permitted future input."]
    (out/"stage_b_prevalence.md").write_text("\n".join(lines)+"\n")
    print({"rows":len(rows),"shifted_events":sum(r["exceeds_1p5"] for r in roles)},flush=True)


if __name__ == "__main__":
    main()
