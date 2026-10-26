"""Audited train/selection data and real production timing for the case118 gate."""
import json
import math
import os
from pathlib import Path
import signal
import statistics
import sys
import time

from power_fm_foresight._config import ROOT, load_config, read_json, sha256, write_json


class PreparationGuard(Exception):
    pass


def guard():
    if time.time()>float(os.environ["GENCO_V2_JOB_START"])+3000:
        raise PreparationGuard("Clean data preparation boundary")


def configuration(root):
    return read_json(root/"case118/configuration.json")


def setup(root):
    from power_fm_foresight.grids import grid_config
    from experiments.data_generation import label
    cfg=grid_config("case118_ieee",load_config(ROOT/"configs/v4.yaml"))
    folder=root/"case118"
    for key,suffix in (("temporal_root","data"),("stage_a_run","audit"),("training_run","adapter"),
                       ("runs_root","runs"),("evidence_root","audit"),("data_root","data")):
        cfg["paths"][key]=str(folder/suffix)
    cfg["stage_b"].update(read_json(root/"configuration.json")["stage_b"])
    cfg["stage_b"]["coordinate_scales"]=str(folder/"audit/stage_a_followup/train_coordinate_scales.json")
    cfg["v2"]={"stage":1,"run_identity":f"{root.name}/case118-unassigned","protocol_sha256":sha256(root/"protocol.json")}
    path=folder/"configuration.json"
    if path.exists() and read_json(path)!=cfg:raise ValueError("Case118 preparation configuration changed")
    write_json(path,cfg)
    if not (folder/"audit/solver_smoke.json").exists():
        from experiments.solver_smoke import main
        sys.argv=["solver_smoke","--config",str(path)]
        main()
    smoke=read_json(folder/"audit/solver_smoke.json")
    if not (smoke["cold"]["checks"]["valid"] and smoke["warm"]["checks"]["valid"] and smoke["dual_check"]["valid"]):
        raise ValueError("Case118 solver/dual prerequisite failed")
    guard()
    if not (folder/"data/pilot_report.json").exists():label(cfg,maximum_hours=200)
    if not read_json(folder/"data/pilot_report.json")["accepted"]:raise ValueError("Case118 pilot failed")
    write_json(folder/"setup.json",{"passed":True,"configuration_sha256":sha256(path)})


def label_episode(root,name):
    import numpy as np
    from power_fm_foresight._data.generate import demands
    from experiments.data_generation import prepare_benchmark
    from power_fm_foresight._solver.adapter import IpoptSolver
    from power_fm_foresight._solver.checks import check_solution
    cfg=configuration(root)
    data,manifest,network,frame,scaled=prepare_benchmark(cfg)
    ep=next(e for e in manifest["episodes"] if e["episode_id"]==name)
    if ep["split"] not in ("train","model_selection"):raise ValueError("Held-out labeling is outside this protocol")
    path=data/"episodes"/f"{name}.json"
    smoke=read_json(root/"case118/audit/solver_smoke.json")
    with IpoptSolver(cfg,root/f"case118/logs/{name}.julia.log",network=network) as solver:
        shard=read_json(path) if path.exists() else {"benchmark_id":manifest["benchmark_id"],
            "episode_id":name,"versions":solver.versions,"records":[]}
        if shard["benchmark_id"]!=manifest["benchmark_id"]:raise ValueError("Shard provenance changed")
        lo=ep["source_start_hour"]
        pd,qd=demands(scaled[lo:lo+504],network.bus[:,2],network.bus[:,3],cfg["generation"],ep["seed"])
        for old in shard["records"]:
            np.testing.assert_array_equal(old["pd_mw"],pd[old["hour"]])
            np.testing.assert_array_equal(old["qd_mvar"],qd[old["hour"]])
        try:
            for hour in range(len(shard["records"]),504):
                guard()
                result=solver.solve(pd[hour],qd[hour])
                result["checks"]=check_solution(network,pd[hour],qd[hour],result)
                valid=result["checks"]["valid"]
                dual_valid=bool(valid and "dual_p_raw" in result and np.isfinite(result["dual_p_raw"]).all())
                if dual_valid:
                    result["dual_p_cost_per_mw"]=(np.asarray(result["dual_p_raw"])*smoke["dual_check"]["sign"]/network.baseMVA).tolist()
                    result["dual_convention"]="one-hour benchmark objective; raw nodal dual times verified sign/baseMVA"
                shard["records"].append({"hour":hour,"time_utc":str(frame.time_utc.iloc[lo+hour]),
                    "pd_mw":pd[hour].tolist(),"qd_mvar":qd[hour].tolist(),"state_available":valid,
                    "duals_available":dual_valid,"target_state":result.get("state") if valid else None,"solver":result})
                if (hour+1)%25==0:write_json(path,shard)
                if result["status"]=="WORKER_UNRESOLVED":raise RuntimeError("Solver worker unresolved; no crash retry")
        finally:write_json(path,shard)


def audit_cache(root):
    import numpy as np
    import torch
    from power_fm_foresight._data.coordinates import fit_scales
    from experiments.data_generation import audit, prepare_benchmark
    from power_fm_foresight._data.generate import demands
    from power_fm_foresight._solver.checks import check_solution
    from power_fm_foresight._training.inputs import build_cache
    cfg=configuration(root)
    data,manifest,network,frame,scaled=prepare_benchmark(cfg)
    records=[];valid=0;count=0
    for ep in manifest["episodes"]:
        if ep["split"] not in ("train","model_selection"):continue
        guard()
        shard=read_json(data/"episodes"/f"{ep['episode_id']}.json")
        if len(shard["records"])!=504 or shard["benchmark_id"]!=manifest["benchmark_id"]:
            raise ValueError("Incomplete or incompatible case118 shard")
        lo=ep["source_start_hour"]
        pd,qd=demands(scaled[lo:lo+504],network.bus[:,2],network.bus[:,3],cfg["generation"],ep["seed"])
        for hour,row in enumerate(shard["records"]):
            if row["hour"]!=hour or row["time_utc"]!=str(frame.time_utc.iloc[lo+hour]):raise ValueError("Hour identity changed")
            np.testing.assert_array_equal(row["pd_mw"],pd[hour])
            np.testing.assert_array_equal(row["qd_mvar"],qd[hour])
            checked=check_solution(network,pd[hour],qd[hour],row["solver"])
            if checked["valid"]!=row["state_available"] or row["state_available"]!=(row["target_state"] is not None):
                raise ValueError("Independent validity/target mask mismatch")
            if row["state_available"] and row["target_state"]!=row["solver"]["state"]:raise ValueError("Target changed")
            count+=1;valid+=int(checked["valid"])
        if ep["split"]=="train":records.extend(shard["records"])
    if count!=6048 or valid/count<.99:raise ValueError("Case118 valid fraction below predeclared 0.99")
    pilot=read_json(data/"pilot_report.json")
    for row in pilot["shards"]:
        shard=read_json(data/row["file"])
        prefix=data/"pilot"/Path(row["file"]).name
        write_json(prefix,dict(shard,records=shard["records"][:row["hours"]]))
        if sha256(prefix)!=row["sha256"]:raise ValueError("Pilot prefix changed")
    report=audit(data)
    scales=fit_scales(network,records)
    scale=np.asarray(scales["scale_vector_one_hour"])
    if not np.isfinite(scale).all() or not (scale>0).all():raise ValueError("Invalid training scales")
    scales["benchmark_id"]=manifest["benchmark_id"]
    write_json(cfg["stage_b"]["coordinate_scales"],scales)
    write_json(root/"case118/audit/readiness.json",{"stage_a_audit_passed":True,"training_allowed":True,
        "scope":"Train and model-selection labels only; no held-out evaluation audit is claimed",
        "manifest_sha256":sha256(data/"manifest.json"),"attempted":count,"valid":valid,
        "pilot_prefix_verified":True,"independent_checks":True,"shards":report["shards"]})
    guard()
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(42)
    if torch.xpu.device_count()!=1:raise RuntimeError("Expected one isolated cache-preparation tile")
    torch.xpu.set_device(0)
    build_cache(cfg,"xpu")
    write_json(root/"case118/cache_ready.json",{"passed":True,"cache_manifest_sha256":sha256(root/"case118/adapter/cache/manifest.json")})


def timing(root,mode):
    import torch
    from power_fm_foresight._training.loop import run
    cfg=configuration(root)
    if os.environ.get("ZE_AFFINITY_MASK")!="0" or torch.xpu.device_count()!=1:raise RuntimeError("Timing tile convention changed")
    torch.xpu.set_device(0)
    cfg["stage_b"]["deterministic_algorithms"]=mode=="on"
    cfg["v2"]["run_identity"]=f"{root.name}/case118/{mode}"
    run(cfg,"xpu",root/f"case118/{mode}",stop_after_steps=70)


def worker(root,name):
    def terminated(signum,frame):raise PreparationGuard("Preparation terminated")
    signal.signal(signal.SIGTERM,terminated)
    try:
        if name=="case118_setup":setup(root)
        elif name=="case118_audit_cache":audit_cache(root)
        elif name.startswith("case118_label_"):label_episode(root,name.removeprefix("case118_label_"))
        elif name in ("case118_on","case118_off"):timing(root,name.removeprefix("case118_"))
        else:raise ValueError("Unknown case118 worker")
    except PreparationGuard:
        print({"clean_preparation_boundary":True,"worker":name},flush=True)


def job(root,deadline,record):
    from experiments.v2.stage1 import launch
    folder=root/"case118"
    def wait_group(names):
        tasks=[launch(root,name,0,deadline,1) for name in names]
        try:
            while tasks:
                for task in list(tasks):
                    code=task["process"].poll()
                    if code is None:continue
                    task["stream"].close();tasks.remove(task)
                    record["workers"].append({"name":task["name"],"returncode":code,
                        "wall_seconds":time.time()-task["began"],"log":task["log"]})
                    if code:raise RuntimeError(f"Case118 worker failed: {task['log']}")
                if time.time()>deadline:raise TimeoutError("Case118 family guard exceeded")
                time.sleep(1)
        finally:
            for task in tasks:
                if task["process"].poll() is None:os.killpg(task["process"].pid,signal.SIGTERM)
            for task in tasks:
                try:task["process"].wait(timeout=10)
                except Exception:
                    os.killpg(task["process"].pid,signal.SIGKILL);task["process"].wait()
                task["stream"].close()
                record["workers"].append({"name":task["name"],"returncode":task["process"].returncode,
                    "terminated_after_family_failure":True,"wall_seconds":time.time()-task["began"],"log":task["log"]})
    if not (folder/"setup.json").exists():wait_group(["case118_setup"])
    if not (folder/"setup.json").exists():return {"complete":False}
    manifest=read_json(folder/"data/manifest.json")
    names=[]
    for ep in manifest["episodes"]:
        if ep["split"] not in ("train","model_selection"):continue
        path=folder/"data/episodes"/f"{ep['episode_id']}.json"
        if not path.exists() or len(read_json(path)["records"])<504:names.append("case118_label_"+ep["episode_id"])
    if names:wait_group(names)
    if time.time()>deadline-300:return {"complete":False}
    if not (folder/"cache_ready.json").exists():wait_group(["case118_audit_cache"])
    if not (folder/"cache_ready.json").exists():return {"complete":False}
    timings={};curves={}
    for mode in ("off","on"):
        status_path=folder/mode/"status.json"
        status=read_json(status_path) if status_path.exists() else {}
        if status.get("steps_after",0)<70:
            if time.time()>deadline-180:return {"complete":False}
            wait_group([f"case118_{mode}"])
        status=read_json(folder/mode/"status.json")
        if status["steps_after"]<70 and status["stop_reason"]=="job_walltime_guard":
            return {"complete":False}
        rows=[json.loads(line) for line in (folder/mode/"metrics.jsonl").read_text().splitlines()]
        if len(rows)!=70 or status["steps_after"]!=70 or not status["base_unchanged"]:raise ValueError("Incomplete case118 timing")
        if not all(math.isfinite(row[key]) for row in rows for key in ("loss","energy","balance","gradient_norm")):
            raise ValueError("Nonfinite case118 timing metric")
        values=sorted(row["step_seconds"] for row in rows if 21<=row["steps"]<=70)
        timings[mode]={"median_step_seconds":statistics.median(values),"p90_step_seconds":values[math.ceil(.9*len(values))-1],
            "worker_wall_seconds":status["cumulative_training_wall_seconds"],"last_startup_seconds":status["startup_seconds"],
            "window_steps":[21,70],"metrics_sha256":sha256(folder/mode/"metrics.jsonl")}
        curves[mode]=[(row["origin"],row["noise_sha256"]) for row in rows]
    if curves["on"]!=curves["off"]:raise ValueError("Unmatched case118 origins/noise")
    result={"schema_version":1,"complete":True,"passed":True,"timings":timings,
            "speedup":timings["on"]["median_step_seconds"]/timings["off"]["median_step_seconds"]}
    write_json(folder/"timing_gate.json",result)
    return result
