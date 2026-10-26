"""Resumable per-hour solver shards; original demand law and independent checks."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import numpy as np
import pandas as pd
from power_fm_foresight._config import PACKAGE_ROOT, read_json, sha256, write_json
from power_fm_foresight._data.assets import eia_profile
from power_fm_foresight._data.network import load_network, graph_for_demand
from power_fm_foresight._data.generate import episode_plan, demands
from power_fm_foresight._solver.adapter import IpoptSolver
from power_fm_foresight._solver.checks import check_solution

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
        "implementation_sha256": {str(p.relative_to(PACKAGE_ROOT)):sha256(p) for p in sorted(PACKAGE_ROOT.rglob("*.py"))}}
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
    root.mkdir(parents=True, exist_ok=True)
    path = root / "manifest.json"
    if path.exists():
        previous = read_json(path)
        if previous["benchmark_id"] != identifier:
            raise ValueError("Benchmark configuration changed; choose a new external data root")
    else:
        write_json(path, descriptor)
    return root, descriptor, network, frame, scaled


def solver_smoke(config):
    root = Path(config["paths"].get("stage_a_run", Path(config["paths"]["runs_root"])/"stage_a"))
    root.mkdir(parents=True, exist_ok=True)
    network = load_network(config=config)
    pd, qd = network["bus"][:, 2], network["bus"][:, 3]
    result = {"case_sha256": network["case_sha256"], "perturbation_mw": 0.01,
              "independent_tolerance": 1e-5, "objective_convention": "benchmark cost per hour; one-hour snapshots"}
    with IpoptSolver(config, root / "julia.log") as solver:
        result["versions"] = solver.versions
        cold = solver.solve(pd, qd)
        cold["checks"] = check_solution(network, pd, qd, cold)
        result["cold"] = cold
        write_json(root / "solver_smoke.json", result)
        if not cold["checks"]["valid"]:
            raise RuntimeError(f"Cold reference failed: {cold}")
        warm = solver.solve(pd, qd, initial_state=cold["state"])
        warm["checks"] = check_solution(network, pd, qd, warm)
        result["warm"] = warm
        graph = graph_for_demand(network, pd, qd)
        initial = cold["state"] if config["solver"].get("warm_start_base", False) else None
        round_trip = solver.solve(graph, initial_state=initial)
        graph_pd, graph_qd = graph["bus"].x[:, 0].numpy(), graph["bus"].x[:, 1].numpy()
        round_trip["checks"] = check_solution(network, graph_pd, graph_qd, round_trip)
        result["graph_round_trip"] = round_trip
        bus = int(np.flatnonzero(pd > 0)[0])
        perturbed_pd = pd.copy(); perturbed_pd[bus] += result["perturbation_mw"]
        perturbed = solver.solve(perturbed_pd, qd, initial_state=initial)
        perturbed["checks"] = check_solution(network, perturbed_pd, qd, perturbed)
        result["perturbed"] = perturbed
        if perturbed["checks"]["valid"] and "dual_p_raw" in cold:
            derivative = (perturbed["objective"]-cold["objective"])/result["perturbation_mw"]
            raw_per_mw = cold["dual_p_raw"][bus] / network["baseMVA"]
            sign = 1 if abs(derivative-raw_per_mw) <= abs(derivative+raw_per_mw) else -1
            error = abs(derivative-sign*raw_per_mw)/max(1, abs(derivative))
            result["dual_check"] = {"bus_index": bus, "hold_qd_fixed": True, "sign": sign,
                "finite_difference": derivative, "signed_dual_per_mw": sign*raw_per_mw,
                "relative_error": error, "valid": error < 0.02}
        else:
            result["dual_check"] = {"valid": False, "reason": "unavailable_dual_or_reference"}
    write_json(root / "solver_smoke.json", result)
    print({"cold_valid": cold["checks"]["valid"], "warm_valid": warm["checks"]["valid"],
           "dual_check": result["dual_check"]}, flush=True)
    if not warm["checks"]["valid"] or not round_trip["checks"]["valid"] or not result["dual_check"]["valid"]:
        raise RuntimeError("Warm-start or dual validation failed; see solver_smoke.json")

    return result


def hour_digest(payload):
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def generate_worker(cfg, worker, workers, smoke):
    root,manifest,net,frame,scaled=prepare_benchmark(cfg)
    with IpoptSolver(cfg,root/f'worker_{worker}.log',network=net,startup_seconds=600) as solver:
        for i,episode in enumerate(manifest['episodes']):
            lo=episode['source_start_hour'];pd,qd=demands(scaled[lo:lo+504],net.bus[:,2],net.bus[:,3],cfg['generation'],episode['seed'])
            for hour in range(504):
                if (i*504+hour)%workers!=worker:continue
                path=root/'hours'/episode['episode_id']/f'{hour:03}.json'
                if path.exists():
                    payload=read_json(path);digest=payload.pop('sha256')
                    if hour_digest(payload)!=digest or payload['benchmark_id']!=manifest['benchmark_id']:raise ValueError('Hour shard changed')
                    old=payload['record']
                    if payload['episode_id']!=episode['episode_id'] or old['hour']!=hour:raise ValueError('Hour identity changed')
                    np.testing.assert_array_equal(old['pd_mw'],pd[hour]);np.testing.assert_array_equal(old['qd_mvar'],qd[hour])
                    continue
                result=solver.solve(pd[hour],qd[hour],initial_state=smoke['cold']['state'] if cfg['solver'].get('warm_start_base') else None)
                unresolved=result['status']=='WORKER_UNRESOLVED'
                result['checks']=dict(valid=False,reason='worker_unresolved') if unresolved else check_solution(net,pd[hour],qd[hour],result)
                valid=result['checks']['valid'];dual=bool(valid and 'dual_p_raw' in result and np.isfinite(result['dual_p_raw']).all())
                if dual:
                    result['dual_p_cost_per_mw']=(np.asarray(result['dual_p_raw'])*smoke['dual_check']['sign']/net.baseMVA).tolist()
                    result['dual_convention']='one-hour benchmark objective; raw nodal dual times verified sign/baseMVA'
                row=dict(hour=hour,time_utc=str(frame.time_utc.iloc[lo+hour]),pd_mw=pd[hour].tolist(),qd_mvar=qd[hour].tolist(),
                    state_available=valid,duals_available=dual,target_state=result.get('state') if valid else None,solver=result)
                payload=dict(benchmark_id=manifest['benchmark_id'],episode_id=episode['episode_id'],record=row)
                write_json(path,dict(payload,sha256=hour_digest(payload)))
                if unresolved:raise RuntimeError('Unresolved solver worker; failure saved, not silently retried')


def generate(cfg,workers):
    root,manifest,_,_,_=prepare_benchmark(cfg)
    smoke_path=Path(cfg['paths']['stage_a_run'])/'solver_smoke.json'
    smoke=read_json(smoke_path) if smoke_path.exists() else solver_smoke(cfg)
    if not smoke['cold']['checks']['valid'] or not smoke['dual_check']['valid']:raise ValueError('Solver smoke/dual audit failed')
    if workers==1:generate_worker(cfg,0,1,smoke)
    else:
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures=[pool.submit(generate_worker,cfg,i,workers,smoke) for i in range(workers)]
            for f in futures:f.result()
    for e in manifest['episodes']:
        rows=[]
        for h in range(504):
            payload=read_json(root/'hours'/e['episode_id']/f'{h:03}.json');digest=payload.pop('sha256')
            if hour_digest(payload)!=digest:raise ValueError('Hour shard checksum changed')
            rows.append(payload['record'])
        write_json(root/'episodes'/f"{e['episode_id']}.json",dict(benchmark_id=manifest['benchmark_id'],episode_id=e['episode_id'],versions=smoke['versions'],records=rows))
    from .generation_summary import audit
    report=audit(root);write_json(root/'generation_report.json',report)
    return report
