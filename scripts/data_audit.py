"""Recheck released temporal labels, source separation, hashes, and event support."""
from pathlib import Path

import numpy as np

from common import load_config, read_json, sha256, write_json
from data.loader.dataset import daily_origins
from data.loader.network import load_network
from data.synthetic.generate import demands
from solver.checks import check_solution, physical_quantities


def audit(config):
    root = Path(config["paths"]["temporal_root"])
    manifest = read_json(root/"manifest.json")
    net = load_network()
    from data.synthetic.generate import prepare_benchmark
    _, descriptor, _, frame, scaled = prepare_benchmark(config)
    expected_hashes = {item["file"]: item["sha256"] for item in manifest["shards"]}
    pilot_report = read_json(root/"pilot_report.json")
    pilot_prefix = None
    maxima, summaries = {}, []
    pools = manifest["source_block_pools"]
    for split, values in pools.items():
        other = {i for key, items in pools.items() if key != split for i in items}
        if not set(values).isdisjoint(other):
            raise ValueError("Cross-partition source-profile leakage")
    totals = {"attempted": 0, "valid": 0, "unresolved": 0, "valid_duals": 0}
    for episode in manifest["episodes"]:
        name = f"episodes/{episode['episode_id']}.json"
        path = root/name
        if sha256(path) != expected_hashes[name]:
            raise ValueError(f"Shard checksum mismatch: {name}")
        shard = read_json(path)
        if len(shard["records"]) != 504 or shard["benchmark_id"] != descriptor["benchmark_id"]:
            raise ValueError("Incomplete episode or incorrect provenance")
        for original in pilot_report["shards"]:
            if original["file"] == name:
                prefix = dict(shard, records=shard["records"][:original["hours"]])
                retained = root/"pilot"/f"{episode['episode_id']}_{original['hours']}.json"
                write_json(retained, prefix)
                digest = sha256(retained)
                if digest != original["sha256"]:
                    raise ValueError("The verified 200-hour pilot prefix changed during expansion")
                pilot_prefix = {"file": str(retained.relative_to(root)), "sha256": digest,
                                "hours": original["hours"]}
        lo = episode["source_start_hour"]
        pd_expected, qd_expected = demands(scaled[lo:lo+504], net["bus"][:, 2], net["bus"][:, 3],
                                          manifest["generation"], episode["seed"])
        thermal, voltage, available = [], [], []
        for hour, record in enumerate(shard["records"]):
            if record["hour"] != hour or record["time_utc"] != frame.time_utc.iloc[lo+hour]:
                raise ValueError("Hour identity changed")
            np.testing.assert_array_equal(record["pd_mw"], pd_expected[hour])
            np.testing.assert_array_equal(record["qd_mvar"], qd_expected[hour])
            check = check_solution(net, record["pd_mw"], record["qd_mvar"], record["solver"])
            if check["valid"] != record["state_available"]:
                raise ValueError("Stored validity mask disagrees with independent checks")
            totals["attempted"] += 1
            totals["valid"] += int(check["valid"])
            totals["unresolved"] += int(not check["valid"])
            totals["valid_duals"] += int(record["duals_available"])
            available.append(check["valid"])
            if check["valid"]:
                if record["target_state"] != record["solver"]["state"]:
                    raise ValueError("Targets disagree with checked solver state")
                for key,value in check["errors"].items():
                    maxima[key] = max(maxima.get(key, 0), value)
                quantities = physical_quantities(net, record["pd_mw"], record["qd_mvar"], record["target_state"])
                thermal.append(quantities["loading"] > 0.95)
                vm = np.asarray(record["target_state"]["vm"])
                b = net["bus"]
                voltage.append((vm <= b[:,12]+0.05*(b[:,11]-b[:,12])) |
                               (vm >= b[:,11]-0.05*(b[:,11]-b[:,12])))
            else:
                if record["target_state"] is not None:
                    raise ValueError("Unresolved states must have missing targets")
                thermal.append(np.zeros(len(net["branch"]), dtype=bool))
                voltage.append(np.zeros(30, dtype=bool))
        thermal, voltage, available = np.asarray(thermal), np.asarray(voltage), np.asarray(available)
        daily = []
        for origin in daily_origins():
            valid = available[origin:origin+24]
            events = {}
            for family, values in (("thermal", thermal), ("voltage", voltage)):
                positive = bool(values[origin:origin+24][valid].any())
                events[family] = True if positive else (False if valid.all() else None)
            daily.append({"origin": origin, "events": events, "valid_hours": int(valid.sum())})
        summaries.append({"episode_id": episode["episode_id"], "split": episode["split"],
            "valid_hours": int(available.sum()), "thermal_component_event_hours": thermal.sum(0).tolist(),
            "voltage_component_event_hours": voltage.sum(0).tolist(), "forecast_days": daily})
    if pilot_prefix is None:
        raise ValueError("Missing retained pilot-prefix evidence")
    manifest["retained_pilot"] = pilot_prefix
    write_json(root/"manifest.json", manifest)
    report = {"passed": True, "benchmark_id": manifest["benchmark_id"], **totals,
              "max_independent_errors": maxima, "episodes": summaries,
              "source_pools_disjoint": True, "deterministic_demands_verified": True,
              "shard_checksums_verified": True, "pilot_prefix_verified": True,
              "retained_pilot": pilot_prefix,
              "reactive_only_buses": np.flatnonzero((net["bus"][:, 2] == 0) & (net["bus"][:, 3] != 0)).tolist(),
              "unique_source_blocks_per_split": {key:len(values) for key,values in pools.items()},
              "test_forecast_days": sum(len(e["forecast_days"]) for e in summaries if e["split"] == "test")}
    write_json(root/"quality_report.json", report)
    print({k:v for k,v in report.items() if k != "episodes"}, flush=True)
    return report


if __name__ == "__main__":
    audit(load_config())
