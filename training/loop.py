"""Budgeted training with resumable mid-epoch order and selection-only stopping."""
import hashlib
import os
from pathlib import Path
import platform
import time

import torch

from common import ROOT,read_json,sha256,write_json
from data.download.prepare import checkpoint_assets
from data.loader.network import graph_for_demand,load_network
from models.genco import Genco
from models.temporal import TemporalAdapter
from scoring import fair_energy_score
from training.inputs import CachedWindows
from training.state import (checkpoint,epoch_order,next_progress,restore,seed_all,
                            tensor_digest,train_state)


class TimeGuard(Exception): pass


def make_manifest(cfg):
    dataset=Path(cfg["paths"]["temporal_root"])
    return {"mode":"A","dataset_id":cfg["dataset"]["id"],"dataset_manifest_sha256":sha256(dataset/"manifest.json"),
        "benchmark_id":read_json(dataset/"manifest.json")["benchmark_id"],"base_checkpoint":cfg["checkpoint"],
        "graphkit":cfg["upstream"]["graphkit"],"seed":cfg["seed"],"settings":cfg["stage_b"],
        "coordinate_scales_sha256":sha256(ROOT/cfg["stage_b"]["coordinate_scales"]),
        "training_origins":cfg["task"]["split_episodes"]["train"]*409,
        "selection_origins":cfg["task"]["split_episodes"]["model_selection"]*18,
        "deterministic_algorithms":True,
        "python_hash_seed":os.environ.get("PYTHONHASHSEED"),
        "source_sha256":{p:sha256(ROOT/p) for p in ("models/temporal.py","scoring.py","training/state.py","training/inputs.py","training/loop.py")},
        "budget_definition":"Cumulative worker-job wall time including startup, validation and checkpointing; excludes queue wait, separate cache preparation and resume tests",
        "rng_protocol":"Restore Python/NumPy/Torch CPU/XPU/CUDA states; deterministic epoch permutation; independent fixed selection-noise generator"}


def initialize(cfg,device):
    torch.use_deterministic_algorithms(True)
    seed_all(cfg["seed"])
    torch.set_num_threads(1 if device=="xpu" else 16)
    net=load_network();base=Genco(checkpoint_assets(cfg),cfg["checkpoint"],device)
    scales=read_json(ROOT/cfg["stage_b"]["coordinate_scales"])
    model=TemporalAdapter(base,net,scales).to(device).train()
    opt_cfg=cfg["stage_b"]["optimizer"]
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=opt_cfg["lr"],weight_decay=opt_cfg["weight_decay"],foreach=False)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda step:1.0)
    graph=graph_for_demand(net,net["bus"][:,2],net["bus"][:,3])
    template=base.batch([graph]*cfg["stage_b"]["future_graph_batch_size"])
    assert template.num_graphs==192
    cache=CachedWindows(cfg)
    return model,optimizer,scheduler,template,cache


def synchronize(device):
    if device=="xpu":torch.xpu.synchronize()
    elif device=="cuda":torch.cuda.synchronize()


def training_step(model,optimizer,scheduler,template,cache,identity,cfg,device):
    history,calendar,target=cache.window(identity,device)
    noise=model.noise(cfg["stage_b"]["training_samples"],24)
    # A chained digest audits the actual sequence and sampled innovations across restarts.
    noise_digest=tensor_digest(noise)
    optimizer.zero_grad(set_to_none=True)
    result=model(history,calendar,template,noise)
    energy=fair_energy_score(result["joint"],target,model.coordinate_scale.repeat(24))
    balance=result["power_balance"]
    loss=energy+cfg["stage_b"]["loss"]["power_balance_weight"]*balance
    if not torch.isfinite(loss):raise RuntimeError("Nonfinite training loss")
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
        cfg["stage_b"]["optimizer"]["gradient_clip_norm"],error_if_nonfinite=True)
    optimizer.step();scheduler.step()
    return {"energy":float(energy.detach()),"balance":float(balance.detach()),"loss":float(loss.detach()),
        "gradient_norm":float(norm),"noise_sha256":noise_digest}


def selection_score(model,template,cache,cfg,device,guard):
    scores=[];model.eval()
    try:
        with torch.no_grad():
            for index,identity in enumerate(cache.indices["model_selection"]):
                history,calendar,target=cache.window(identity,device)
                # Local CPU generator: common random numbers across epochs, no training RNG consumption.
                g=torch.Generator().manual_seed(cfg["stage_b"]["selection_noise_seed"]+index)
                predictions=[]
                for _ in range(cfg["stage_b"]["forecast_samples"]//cfg["stage_b"]["training_samples"]):
                    guard()
                    k=cfg["stage_b"]["training_samples"]
                    noise={"factor":torch.randn(k,8,generator=g),"bus":torch.randn(k,24,30,3,generator=g),
                           "gen":torch.randn(k,24,6,1,generator=g)}
                    result=model(history,calendar,template,{key:v.to(device) for key,v in noise.items()})
                    predictions.append(result["joint"])
                score=fair_energy_score(torch.cat(predictions),target,model.coordinate_scale.repeat(24))
                if not torch.isfinite(score):raise RuntimeError("Nonfinite selection energy score")
                scores.append(float(score))
        return sum(scores)/len(scores)
    finally:model.train()


def walltime_seconds(value):
    if value is None:return 3600
    if ":" not in str(value):return int(value)
    result=0
    for part in str(value).split(":"):result=60*result+int(part)
    return result


def run(cfg,device,run_root=None,stop_after_steps=None,crash_at_steps=None,test_mode=False):
    entry=time.time();job_start=float(os.environ.get("GENCO_JOB_START_EPOCH",entry))
    root=Path(run_root or cfg["paths"]["training_run"]);root.mkdir(parents=True,exist_ok=True)
    if (root/"failure.json").exists():raise RuntimeError("Prior worker failed; inspect failure.json before resuming")
    readiness=read_json(Path(cfg["paths"]["stage_a_run"])/"readiness.json")
    if not readiness["stage_a_audit_passed"] or not readiness["training_allowed"]:
        raise RuntimeError("Stage A audit has not passed")
    if readiness["manifest_sha256"] != sha256(Path(cfg["paths"]["temporal_root"])/"manifest.json"):
        raise RuntimeError("Dataset changed after the Stage A audit")
    if device=="xpu" and not torch.xpu.is_available():raise RuntimeError("XPU is unavailable on this node")
    manifest=make_manifest(cfg)
    if (root/"manifest.json").exists() and read_json(root/"manifest.json")!=manifest:
        raise ValueError("Training implementation or frozen configuration changed")
    write_json(root/"manifest.json",manifest)
    model,opt,sched,template,cache=initialize(cfg,device)
    base_digest=tensor_digest(model.backbone.state_dict())
    state_path=root/"latest.pt"
    progress=restore(state_path,model,opt,sched,manifest) if state_path.exists() else next_progress()
    if "initial_adapter_sha256" not in progress:
        progress["initial_adapter_sha256"]=tensor_digest(train_state(model))
    if progress["finished"] and not test_mode:
        job_name=os.environ.get("PBS_JOBID",f"local_{int(entry)}").split('.')[0]
        write_json(root/"jobs"/f"{job_name}.json",{"status":"skipped_after_stop","job_id":os.environ.get("PBS_JOBID"),
            "node":platform.node(),"steps_done":0,"epochs_reached":progress["epoch"],
            "wall_used_seconds":time.time()-job_start,"job_start_epoch":job_start,"job_end_epoch":time.time()})
        return progress
    used_before=progress["cumulative_training_wall_seconds"]
    if (root/"status.json").exists():
        previous=read_json(root/"status.json")
        if previous.get("checkpoint_sha256")==sha256(state_path):
            used_before=max(used_before,previous["cumulative_training_wall_seconds"])
    steps_before=progress["steps"];epochs_before=progress["epoch"]
    guard_seconds=min(cfg["stage_b"]["job_training_guard_seconds"],walltime_seconds(os.environ.get("PBS_WALLTIME"))-300)
    job_deadline=job_start+guard_seconds
    budget=cfg["stage_b"]["training_budget_seconds"]
    startup=time.time()-job_start
    startup_record={"job_id":os.environ.get("PBS_JOBID"),"node":platform.node(),
        "device":device,"job_start_epoch":job_start,"startup_seconds":startup,"steps_before":steps_before,
        "cache":"torch.load(mmap=True), per-hour history slices","target_seconds":120,
        "within_target":startup<=120}
    write_json(root/"startup.json",startup_record)
    print({"training_ready":True,"startup_seconds":startup,"resume_step":steps_before,"device":device},flush=True)
    last_save=time.time();last_metrics=None
    def used():return used_before+(time.time()-job_start)
    def save(path=state_path):
        progress["cumulative_training_wall_seconds"]=used()
        checkpoint(path,model,opt,sched,progress,manifest)
    def guard():
        if used() >= budget-15:
            progress.update(finished=True,stop_reason="six_hour_budget");raise TimeGuard
        if time.time()>=job_deadline-15:
            progress["stop_reason"]="job_walltime_guard";raise TimeGuard
    started_steps=time.perf_counter();status="continue"
    try:
        while not progress["finished"]:
            guard()
            epoch=progress["epoch"]
            order=epoch_order(cfg["seed"],epoch,len(cache.indices["train"]))
            while progress["step_in_epoch"]<len(order):
                guard()
                identity=cache.indices["train"][order[progress["step_in_epoch"]]]
                last_metrics=training_step(model,opt,sched,template,cache,identity,cfg,device)
                progress["step_sequence_sha256"]=hashlib.sha256((progress["step_sequence_sha256"]+
                    repr(identity)+last_metrics["noise_sha256"]).encode()).hexdigest()
                progress["steps"]+=1;progress["step_in_epoch"]+=1
                if progress["steps"]==steps_before+1:
                    startup_record["first_optimizer_step_completed_seconds"]=time.time()-job_start
                    write_json(root/"startup.json",startup_record)
                if progress["steps"]%100==0:print({"step":progress["steps"],"epoch":epoch,"loss":last_metrics["loss"]},flush=True)
                if crash_at_steps is not None and progress["steps"]==crash_at_steps:
                    save();os.kill(os.getpid(),9)
                if stop_after_steps is not None and progress["steps"]>=stop_after_steps:
                    progress["stop_reason"]="resume_test_step_limit";raise TimeGuard
                if time.time()-last_save>=cfg["stage_b"]["checkpoint_interval_seconds"]:
                    save();last_save=time.time()
            # A mid-validation interruption resumes this phase with the same fixed noise.
            score=selection_score(model,template,cache,cfg,device,guard)
            best=progress["best_selection"]
            improved=best is None or score<best
            if improved:progress.update(best_selection=score,best_epoch=epoch+1,bad_epochs=0)
            else:progress["bad_epochs"]+=1
            progress["epochs"].append({"epoch":epoch+1,"selection_energy_score":score,"steps":progress["steps"]})
            progress["epoch"]+=1;progress["step_in_epoch"]=0
            if progress["bad_epochs"]>=cfg["stage_b"]["early_stopping"]["patience"]:
                progress.update(finished=True,stop_reason="early_stopping")
            if progress["epoch"]>=cfg["stage_b"]["max_epochs"]:
                progress.update(finished=True,stop_reason="max_epochs")
            if tensor_digest(model.backbone.state_dict())!=base_digest:raise RuntimeError("Frozen base weights changed")
            save(root/f"epoch_{progress['epoch']:03d}.pt")
            if improved:save(root/"best.pt")
            save();last_save=time.time()
            write_json(root/"epochs.json",progress["epochs"])
            print({"epoch_completed":progress["epoch"],"selection_energy_score":score,"bad_epochs":progress["bad_epochs"]},flush=True)
    except TimeGuard:pass
    except BaseException as error:
        # Record the failure; do not automatically chain from a possibly invalid optimizer state.
        write_json(root/"failure.json",{"error":repr(error),"steps":progress["steps"],"job_id":os.environ.get("PBS_JOBID")})
        raise
    synchronize(device)
    save()
    if tensor_digest(model.backbone.state_dict())!=base_digest:raise RuntimeError("Frozen base weights changed")
    if progress["finished"]:status="finished"
    if test_mode:status="test_complete"
    result={"status":status,"job_id":os.environ.get("PBS_JOBID"),"node":platform.node(),"device":device,
        "job_start_epoch":job_start,"job_end_epoch":time.time(),"startup_seconds":startup,
        "first_optimizer_step_completed_seconds":startup_record.get("first_optimizer_step_completed_seconds"),
        "steps_before":steps_before,"steps_after":progress["steps"],"steps_done":progress["steps"]-steps_before,
        "epochs_before":epochs_before,"epochs_reached":progress["epoch"],"step_in_epoch":progress["step_in_epoch"],
        "wall_used_seconds":time.time()-job_start,"cumulative_training_wall_seconds":used(),
        "training_loop_seconds":time.perf_counter()-started_steps,"stop_reason":progress["stop_reason"],
        "base_unchanged":True,"base_sha256":base_digest,"adapter_sha256":tensor_digest(train_state(model)),
        "step_sequence_sha256":progress["step_sequence_sha256"],"best_selection":progress["best_selection"],
        "best_epoch":progress["best_epoch"],"last_training_metrics":last_metrics,
        "checkpoint_sha256":sha256(state_path)}
    job_name=os.environ.get("PBS_JOBID",f"local_{int(entry)}").split('.')[0]
    write_json(root/"jobs"/f"{job_name}.json",result);write_json(root/"status.json",result)
    print(result,flush=True)
    return result
