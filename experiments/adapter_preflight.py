"""Allocated-node 192-graph adapter forward/backward gate, before training loops."""
import argparse
import gc
import hashlib
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

from power_fm_foresight._config import ROOT, load_config, read_json, sha256, write_json
from power_fm_foresight._data.assets import checkpoint_assets
from power_fm_foresight._data.coordinates import record_values
from power_fm_foresight._data.dataset import load_temporal_manifest
from power_fm_foresight._data.network import graph_for_demand, load_network
from power_fm_foresight.genco import Genco
from power_fm_foresight.temporal import TemporalAdapter
from power_fm_foresight.scoring import fair_energy_score
from power_fm_foresight._solver.checks import physical_quantities


def tensor_digest(state):
    digest = hashlib.sha256()
    for name,value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def fixture(cfg, root):
    """Only train_00; historical state is separate from target labels."""
    net = load_network(config=cfg)
    load_temporal_manifest(cfg["paths"]["temporal_root"], net)
    assets = checkpoint_assets(cfg)
    base = Genco(assets,cfg["checkpoint"],grid=net)
    path = Path(cfg["paths"]["temporal_root"])/"episodes/train_00.json"
    rows = read_json(path)["records"]
    observed,targets = rows[:72],rows[72:96]
    scales = read_json(ROOT/cfg["stage_b"]["coordinate_scales"])
    cache_path = root/"preflight_history.pt"
    cache_key = {"episode_sha256":sha256(path),"origin":72,"checkpoint":cfg["checkpoint"],"graphkit":cfg["upstream"]["graphkit"]}
    start = time.perf_counter()
    if cache_path.exists():
        cache = torch.load(cache_path,map_location="cpu",weights_only=True)
        if cache["key"] != cache_key: raise ValueError("Stale historical embedding cache")
        embeddings = cache["embeddings"]
        cache_seconds = cache["creation_seconds"]
    else:
        chunks = {"bus":[],"gen":[]}
        for lo in range(0,72,24):
            graphs = [graph_for_demand(net,r["pd_mw"],r["qd_mvar"],r["target_state"]) for r in observed[lo:lo+24]]
            with torch.no_grad():
                _,e = base.model(base.batch(graphs),return_embeddings=True)
            for kind,count in (("bus",net.n_bus),("gen",net.n_online_gen)):
                chunks[kind].append(e[kind].reshape(-1,count,e[kind].shape[-1]).detach().cpu())
        embeddings = {kind:torch.cat(parts) for kind,parts in chunks.items()}
        cache_seconds = time.perf_counter()-start
        torch.save({"key":cache_key,"embeddings":embeddings,"creation_seconds":cache_seconds},cache_path)
    history = {f"embedding_{kind}":value for kind,value in embeddings.items()}
    history["observed_bus"] = torch.tensor(np.stack([np.stack([r["pd_mw"],r["qd_mvar"],r["target_state"]["vm"],r["target_state"]["va_rad"]],axis=-1) for r in observed]),dtype=torch.float32)
    history["observed_gen"] = torch.tensor(np.stack([np.stack([r["target_state"]["pg_mw"],r["target_state"]["qg_mvar"]],axis=-1) for r in observed]),dtype=torch.float32)[:,net.online_gen]
    for kind in ("bus","gen"): history[f"available_{kind}"]=torch.isfinite(history[f"observed_{kind}"])
    timestamps = pd.to_datetime([r["time_utc"] for r in targets],utc=True)
    h,d = timestamps.hour.to_numpy(),timestamps.dayofweek.to_numpy()
    calendar = torch.tensor(np.stack([np.sin(2*np.pi*h/24),np.cos(2*np.pi*h/24),np.sin(2*np.pi*d/7),
        np.cos(2*np.pi*d/7),(d>=5).astype(float),np.arange(1,25)/24],axis=-1),dtype=torch.float32)
    graph = graph_for_demand(net,observed[-1]["pd_mw"],observed[-1]["qd_mvar"])
    count = cfg["stage_b"]["future_graph_batch_size"]
    future = base.batch([graph]*count)
    assert future.num_graphs == count and future["bus"].num_nodes == count*net.n_bus
    target_vectors = []
    for r in targets:
        physics = physical_quantities(net,r["pd_mw"],r["qd_mvar"],r["target_state"])
        values = record_values(net,r)
        target_vectors.append(np.concatenate([np.asarray(values[key])[spec["joint_score_mask"]]
            for key,spec in scales["features"].items() if any(spec["joint_score_mask"])]))
    target = torch.tensor(np.asarray(target_vectors).flatten(),dtype=torch.float32)
    return base,net,scales,history,calendar,future,target,cache_seconds


def run_device(cfg,device,root):
    torch.set_num_threads(16)
    torch.set_num_interop_threads(1)
    torch.manual_seed(cfg["seed"])
    report = {"device":device,"hostname":platform.node(),"python":sys.executable,"torch":torch.__version__,
        "cpu_threads":torch.get_num_threads(),"cpu_affinity_count":len(os.sched_getaffinity(0)),
        "dtype":"float32","graphs_per_batch":192,"forecasts_per_step":1,"samples_per_forecast":8,"hours_per_sample":24,
        "timing_scope":"cached-history encoder + sampling + single192-graph GENCO + physics + fair energy score + backward; optimizer timed separately",
        "training_steps_per_epoch":4908,"training_budget_seconds":21600,"status":"pending"}
    report_path = root/f"preflight_{device}.json"
    write_json(report_path,report)
    if device == "xpu" and not torch.xpu.is_available():
        report.update(status="unavailable");write_json(report_path,report);return report
    setup = time.perf_counter()
    base,net,scales,history,calendar,future,target,cache_seconds = fixture(cfg,root)
    model = TemporalAdapter(base,net,scales).to(device).train()
    history={key:value.to(device) for key,value in history.items()}
    calendar,future,target=calendar.to(device),future.to(device),target.to(device)
    coordinate_scale=model.coordinate_scale.repeat(24)
    torch.manual_seed(43)
    noise={key:value.to(device) for key,value in model.noise(8,24,device="cpu").items()}
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3,foreach=False)
    base_before=tensor_digest(model.backbone.state_dict())
    adapter_before=tensor_digest({key:value for key,value in model.state_dict().items() if not key.startswith("backbone.")})
    report["parameters_trainable"]=sum(p.numel() for p in model.parameters() if p.requires_grad)
    report["parameters_frozen"]=sum(p.numel() for p in model.backbone.parameters())
    assert report["parameters_trainable"] == 91461
    report["history_cache_creation_seconds_for_72_hours"]=cache_seconds
    report["setup_seconds"]=time.perf_counter()-setup
    if device == "xpu":
        torch.xpu.synchronize()
        report["accelerator_name"]=torch.xpu.get_device_name()
        torch.xpu.reset_peak_memory_stats()
    def synchronize():
        if device == "xpu": torch.xpu.synchronize()
    def one_step():
        optimizer.zero_grad(set_to_none=True)
        synchronize();start=time.perf_counter()
        result=model(history,calendar,future,noise)
        energy=fair_energy_score(result["joint"],target,coordinate_scale)
        loss=energy+.01*result["power_balance"]
        loss.backward()
        synchronize();backward_end=time.perf_counter()
        optimizer.step()
        synchronize();end=time.perf_counter()
        return {"forward_backward_seconds":backward_end-start,"optimizer_seconds":end-backward_end,
                "training_step_seconds":end-start,"loss":float(loss.detach()),"energy_score":float(energy.detach()),
                "balance_penalty":float(result["power_balance"].detach())},result
    print(f"{device}: warmup full192-graph step",flush=True)
    cold,result=one_step();del result;gc.collect()
    report["cold_step"]=cold;write_json(report_path,report)
    print(f"{device}: measured full192-graph step",flush=True)
    measured,result=one_step()
    groups={}
    for name,p in model.named_parameters():
        if p.requires_grad:
            if p.grad is None or not torch.isfinite(p.grad).all(): raise RuntimeError(f"Missing/nonfinite gradient: {name}")
            kind=name.split('.')[0]
            groups[kind]=groups.get(kind,0.)+float(p.grad.detach().abs().sum())
    if not all(v>0 for v in groups.values()): raise RuntimeError(f"Inactive adapter group: {groups}")
    torch.testing.assert_close(result["values"]["pd_mw"],result["demand_seen_by_physics_mw"],rtol=1e-6,atol=1e-5)
    if torch.equal(result["values"]["pd_mw"][0],result["values"]["pd_mw"][1]): raise RuntimeError("Demand samples collapsed")
    assert all(p.grad is None and not p.requires_grad for p in model.backbone.parameters())
    base_after=tensor_digest(model.backbone.state_dict())
    assert base_before == base_after
    adapter_after=tensor_digest({key:value for key,value in model.state_dict().items() if not key.startswith("backbone.")})
    assert adapter_before != adapter_after
    assert all(not module._forward_hooks for module in (model.backbone.input_proj_bus,model.backbone.input_proj_gen,
                                                       model.backbone.mlp_bus,model.backbone.mlp_gen))
    seconds=measured["training_step_seconds"]
    report.update(status="passed",measured_step=measured,adapter_gradient_l1_by_group=groups,
        base_sha256_before=base_before,base_sha256_after=base_after,base_unchanged=True,adapter_updated=True,
        sample_demand_reaches_physics=True,hooks_cleared=True,
        projected_seconds_per_epoch=4908*seconds,projected_epochs_in_six_hours=21600/(4908*seconds),
        projected_complete_epochs_in_six_hours=math.floor(21600/(4908*seconds)),
        projection_limits="Compute-only steady-step extrapolation from one training window; excludes full history cache, I/O, validation, checkpointing and solver work.",
        peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        peak_device_allocated_bytes=torch.xpu.max_memory_allocated() if device=="xpu" else None)
    write_json(report_path,report)
    print(report,flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",choices=["cpu","xpu"])
    args=parser.parse_args()
    cfg=load_config();root=Path(cfg["paths"]["runs_root"])/"stage_b";root.mkdir(parents=True,exist_ok=True)
    if args.device:
        try:
            run_device(cfg,args.device,root)
        except Exception as exc:
            path=root/f"preflight_{args.device}.json"
            report=read_json(path) if path.exists() else {"device":args.device}
            report.update(status="failed",error=f"{type(exc).__name__}: {exc}");write_json(path,report)
            raise
        return
    env=dict(os.environ,OMP_NUM_THREADS="16",MKL_NUM_THREADS="16",OPENBLAS_NUM_THREADS="1")
    for device in ("cpu","xpu"):
        with (root/f"preflight_{device}.log").open("w") as log:
            try:
                result=subprocess.run([sys.executable,"-m","scripts.adapter_preflight","--device",device],env=env,
                                      stdout=log,stderr=subprocess.STDOUT,timeout=1500)
                print(device,"exit",result.returncode,flush=True)
            except subprocess.TimeoutExpired:
                path=root/f"preflight_{device}.json";report=read_json(path) if path.exists() else {"device":device}
                report.update(status="timeout",budget_seconds=1500);write_json(path,report)
    reports={d:read_json(root/f"preflight_{d}.json") for d in ("cpu","xpu")}
    passed={d:r for d,r in reports.items() if r["status"]=="passed"}
    selected=min(passed,key=lambda d:passed[d]["measured_step"]["training_step_seconds"]) if passed else None
    summary={"devices":reports,"selected_training_device":selected,
        "selection_rule":"lowest measured steady training-step seconds among passing identical192-graph trials",
        "source_sha256":{p:sha256(ROOT/p) for p in ("models/temporal.py","scoring.py","scripts/adapter_preflight.py")},
        "settings_sha256":sha256(ROOT/"reports/stage_b_settings.json"),"status":"passed" if selected else "blocked"}
    write_json(root/"preflight.json",summary);write_json(ROOT/"reports/stage_b_preflight.json",summary)
    print({"selected_training_device":selected,"status":summary["status"]},flush=True)
    if not selected: raise RuntimeError("Neither device passed the fixed batch gate")


if __name__ == "__main__":
    main()
