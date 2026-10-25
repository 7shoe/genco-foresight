"""Resumable, episode-separated demand generation and checked Ipopt labels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from common import ROOT, load_config, read_json, sha256, write_json
from data.download.prepare import eia_profile
from data.loader.network import load_network
from solver.adapter import IpoptSolver
from solver.checks import check_solution


SPLITS = (("train", 12, 10), ("model_selection", 2, 2), ("risk_calibration", 2, 2), ("test", 4, 3))


def episode_plan(seed=42, source_blocks_one_based=None, split_episodes=None):
    """Disjoint 21-day source blocks; reuse is restricted to the same partition."""
    blocks = np.random.default_rng(seed).permutation(17)
    counts = split_episodes or {split: count for split, count, _ in SPLITS}
    if set(counts) != {s for s, _, _ in SPLITS} or sum(counts.values()) != 20:
        raise ValueError("Episode partitions must assign all 20 episodes")
    if source_blocks_one_based is not None:
        expected = {split for split, _, _ in SPLITS}
        flattened = [i for values in source_blocks_one_based.values() for i in values]
        if (set(source_blocks_one_based) != expected or
                any(type(i) is not int for i in flattened) or sorted(flattened) != list(range(1, 18))):
            raise ValueError("Source pools must be disjoint and cover blocks 1 through 17 exactly once")
        if any(not values or len(values) > counts[split] for split, values in source_blocks_one_based.items()):
            raise ValueError("Each pool must be nonempty and covered by its assigned episodes")
    episodes, pools, offset = [], {}, 0
    for split, count, pool_count in SPLITS:
        pool = (sorted(i-1 for i in source_blocks_one_based[split]) if source_blocks_one_based is not None
                else sorted(int(i) for i in blocks[offset:offset+pool_count]))
        pools[split] = pool
        offset += pool_count
        for j in range(counts[split]):
            # Cover every source block before reusing one, with independent noise.
            block = pool[j % len(pool)]
            episodes.append({"episode_id": f"{split}_{j:02d}", "split": split,
                "source_block": block, "source_start_hour": block*504, "hours": 504,
                "seed": int(np.random.SeedSequence([seed, len(episodes)]).generate_state(1)[0])})
    return episodes, pools


def demands(profile, baseline_pd, baseline_qd, settings, seed):
    """Datakit's multiplicative uniform law plus stationary lognormal AR(1)."""
    rho, innovation, sigma = settings["rho"], settings["innovation_std"], settings["sigma"]
    if not abs(rho) < 1 or not 0 <= sigma < 1 or innovation < 0 or not (np.asarray(profile) > 0).all():
        raise ValueError("Invalid positive-load sampling law")
    rng = np.random.default_rng(seed)
    variance = innovation**2/(1-rho**2)
    u = rng.normal(0, np.sqrt(variance), len(baseline_pd))
    pd_values = []
    for ref in profile:
        pd_values.append(baseline_pd * ref * rng.uniform(1-sigma, 1+sigma, len(u)) * np.exp(u-variance/2))
        u = rho*u+rng.normal(0, innovation, len(u))
    pd_values = np.asarray(pd_values)
    qd_values = np.broadcast_to(baseline_qd, pd_values.shape).copy()
    active = baseline_pd != 0
    qd_values[:, active] = pd_values[:, active] * (baseline_qd[active]/baseline_pd[active])
    return pd_values, qd_values


def prepare_benchmark(config):
    from gridfm_datakit.perturbations.load_perturbation import LoadScenariosFromAggProfile
    network = load_network(config=config)
    profile_path = eia_profile(config)
    frame = pd.read_csv(profile_path)
    dataset = config.get("dataset", {})
    episodes, pools = episode_plan(config["seed"], dataset.get("source_blocks_one_based"), config["task"]["split_episodes"])
    settings = config["generation"]
    train_hours = np.concatenate([np.arange(block*504, (block+1)*504) for block in pools["train"]])
    training = frame.demand_mw.to_numpy()[train_hours]
    low, high = float(training.min()), float(training.max())
    # Invoke Datakit's scaling mechanism with training-defined endpoints only.
    # Adding endpoints lets its min/max scaler process held-out values without refitting.
    values = frame.demand_mw.to_numpy()
    scaled = settings["scale_low"] + (values-low)*(settings["scale_high"]-settings["scale_low"])/(high-low)
    train_scaled = LoadScenariosFromAggProfile.min_max_scale(training, settings["scale_low"], settings["scale_high"])
    np.testing.assert_allclose(scaled[train_hours], train_scaled, rtol=0, atol=1e-14)
    if not (scaled > 0).all():
        raise ValueError("Held-out profile extrapolation became nonpositive; benchmark settings must be revised explicitly")
    descriptor = {"schema": "genco-temporal-v1", "seed": config["seed"], "episodes": episodes,
        "source_block_pools": pools, "unused_source_days": list(range(357, 365)),
        "profile_sha256": sha256(profile_path), "case_sha256": network["case_sha256"],
        "generation": settings, "solver_budget": config["solver"].get("budget", {}),
        "train_profile_min": low, "train_profile_max": high, "timezone": "UTC",
        "geography": f"ERCOT aggregate hourly profile applied to {network.path.stem}",
        "independence": "independent stochastic replicates conditional on reused source blocks",
        "datakit_use": ["Network MATPOWER parsing", "makeYbus", "LoadScenariosFromAggProfile.min_max_scale"],
        "source_data_units": "EIA hourly MWh divided by one hour gives mean MW",
        "zero_pd_rule": "Pd remains zero; retain baseline reactive-only Qd",
        "implementation_sha256": {str(p): sha256(ROOT/p) for p in map(Path, [
            "data/synthetic/generate.py", "data/loader/network.py", "solver/adapter.py",
            "solver/checks.py", "solver/julia/worker.jl", "solver/julia/Project.toml", "solver/julia/Manifest.toml"])}}
    if dataset:
        descriptor["dataset"] = dataset
        descriptor["split_episodes"] = config["task"]["split_episodes"]
        descriptor["source_calendar"] = [
            {"block_one_based": i+1, "partition": split,
             "start_utc": str(frame.time_utc.iloc[i*504]),
             "end_utc_inclusive": str(frame.time_utc.iloc[(i+1)*504-1])}
            for split, pool in pools.items() for i in pool]
    identifier = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()
    descriptor["benchmark_id"] = identifier
    root = Path(config["paths"].get("temporal_root", str(Path(config["paths"]["data_root"])/config["task"]["canonical_case"])))
    if root.resolve().is_relative_to(Path("/home")) or root.resolve().is_relative_to(ROOT):
        raise ValueError("Temporal data must be outside /home and the repository; use the configured /lus/flare data root")
    root.mkdir(parents=True, exist_ok=True)
    path = root / "manifest.json"
    if path.exists():
        previous = read_json(path)
        if previous["benchmark_id"] != identifier:
            raise ValueError("Benchmark configuration changed; choose a new external data root")
    else:
        write_json(path, descriptor)
    return root, descriptor, network, frame, scaled


def label(config, maximum_hours=200, require_pilot=True):
    root, manifest, network, frame, scaled = prepare_benchmark(config)
    if maximum_hours > 200 and require_pilot:
        pilot_path = root / "pilot_report.json"
        if not pilot_path.exists() or not read_json(pilot_path).get("accepted", False):
            raise RuntimeError("A verified 200-hour pilot is required before expansion")
    completed = 0
    started = time.perf_counter()
    dual_proof_path = Path(config["paths"].get("stage_a_run", Path(config["paths"]["runs_root"])/"stage_a"))/"solver_smoke.json"
    dual_proof = read_json(dual_proof_path) if dual_proof_path.exists() else {}
    dual_sign = dual_proof.get("dual_check", {}).get("sign")
    dual_verified = dual_proof.get("dual_check", {}).get("valid", False)
    with IpoptSolver(config, root / "julia.log", network=network) as solver:
        for episode in manifest["episodes"]:
            path = root / "episodes" / f"{episode['episode_id']}.json"
            shard = read_json(path) if path.exists() else {"benchmark_id": manifest["benchmark_id"],
                "episode_id": episode["episode_id"], "versions": solver.versions, "records": []}
            if shard["benchmark_id"] != manifest["benchmark_id"]:
                raise ValueError("Shard benchmark ID mismatch")
            lo = episode["source_start_hour"]
            pd_values, qd_values = demands(scaled[lo:lo+504], network["bus"][:, 2], network["bus"][:, 3],
                                           manifest["generation"], episode["seed"])
            for old in shard["records"]:
                np.testing.assert_array_equal(old["pd_mw"], pd_values[old["hour"]])
                np.testing.assert_array_equal(old["qd_mvar"], qd_values[old["hour"]])
            for hour in range(len(shard["records"]), min(504, maximum_hours-completed)):
                pd_mw, qd_mvar = pd_values[hour], qd_values[hour]
                result = solver.solve(pd_mw, qd_mvar)
                if result["status"] == "WORKER_UNRESOLVED":
                    # Persist this failed attempt, then stop rather than issuing requests to a dead worker.
                    result["checks"] = {"valid": False, "reason": "worker_unresolved"}
                else:
                    result["checks"] = check_solution(network, pd_mw, qd_mvar, result)
                valid = result["checks"]["valid"]
                dual_valid = bool(valid and dual_verified and "dual_p_raw" in result and
                                  np.isfinite(result["dual_p_raw"]).all())
                if dual_valid:
                    result["dual_p_cost_per_mw"] = (np.asarray(result["dual_p_raw"])*dual_sign/network["baseMVA"]).tolist()
                    result["dual_convention"] = "one-hour benchmark objective; raw nodal dual times verified sign/baseMVA"
                shard["records"].append({"hour": hour, "time_utc": str(frame.time_utc.iloc[lo+hour]),
                    "pd_mw": pd_mw.tolist(), "qd_mvar": qd_mvar.tolist(), "state_available": valid,
                    "duals_available": dual_valid, "target_state": result.get("state") if valid else None,
                    "solver": result})
                if (hour+1) % 25 == 0 or hour == 503:
                    write_json(path, shard)
                    print(episode["episode_id"], hour+1, "valid", sum(r["state_available"] for r in shard["records"]), flush=True)
                if result["status"] == "WORKER_UNRESOLVED":
                    write_json(path, shard)
                    raise RuntimeError("Worker deadline/exit: failed attempt saved; resume with a fresh process")
            write_json(path, shard)
            completed += len(shard["records"])
            if completed >= maximum_hours:
                break
    report = audit(root)
    report["dual_sign_check_sha256"] = sha256(dual_proof_path) if dual_proof_path.exists() else None
    report["this_invocation_seconds"] = time.perf_counter()-started
    if maximum_hours == 200:
        report["accepted"] = report["attempted"] == 200 and report["valid"] >= 1 and report["hashes_verified"]
        write_json(root / "pilot_report.json", report)
    else:
        write_json(root / "generation_report.json", report)
    print(report, flush=True)
    return report


def audit(root):
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    previous = {r["file"]: r for r in manifest.get("shards", [])}
    records, shards = [], []
    for episode in manifest["episodes"]:
        path = root / "episodes" / f"{episode['episode_id']}.json"
        if not path.exists():
            continue
        shard = read_json(path)
        if shard["benchmark_id"] != manifest["benchmark_id"]:
            raise ValueError("Shard provenance mismatch")
        if [r["hour"] for r in shard["records"]] != list(range(len(shard["records"]))):
            raise ValueError("Noncontiguous/duplicate hour IDs")
        for r in shard["records"]:
            if r["state_available"] != (r["target_state"] is not None):
                raise ValueError("Missing labels must remain masked, not zero substitutes")
        records.extend(shard["records"])
        name = str(path.relative_to(root))
        digest = sha256(path)
        if name in previous and previous[name]["hours"] == len(shard["records"]) and previous[name]["sha256"] != digest:
            raise ValueError(f"Completed shard checksum changed: {path}")
        shards.append({"file": name, "hours": len(shard["records"]), "sha256": digest})
    valid = [r for r in records if r["state_available"]]
    report = {"benchmark_id": manifest["benchmark_id"], "attempted": len(records), "valid": len(valid),
        "unresolved": len(records)-len(valid), "valid_duals": sum(r["duals_available"] for r in records),
        "thermal_stress_hours": sum(r["solver"]["checks"]["thermal_stress"] for r in valid),
        "voltage_stress_hours": sum(r["solver"]["checks"]["voltage_stress"] for r in valid),
        "solver_request_seconds": sum(r["solver"]["request_seconds"] for r in records),
        "hashes_verified": True, "shards": shards}
    manifest["shards"] = shards
    write_json(root / "manifest.json", manifest)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v3.yaml")
    parser.add_argument("--hours", type=int, default=200)
    args = parser.parse_args()
    if not 1 <= args.hours <= 10080:
        parser.error("hours must be between 1 and 10080")
    label(load_config(args.config), args.hours)


if __name__ == "__main__":
    main()
