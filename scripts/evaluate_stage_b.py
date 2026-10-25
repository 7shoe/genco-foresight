"""Resumable P/S/A/D evaluation, one atomic artifact set per forecast origin."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from common import ROOT,load_config,read_json,sha256,write_json
from data.download.prepare import checkpoint_assets
from data.coordinates import bus_qg, record_values
from data.loader.network import graph_for_demand,load_network
from models.genco import Genco
from models.temporal import TemporalAdapter
from scoring import UNITS,evaluate,merge_origin_scores,paired_error,variogram_pairs
from solver.adapter import IpoptSolver
from solver.checks import admittance_vectors,check_solution
from training.forecaster import RidgeForecaster,atomic_npz,episode_arrays,prepare as prepare_ridge
from training.inputs import calendar_features,causal_window
from training.state import atomic_torch_save,tensor_digest


MODES=("P","S","A","D")
LEADS=(0,6,12,18,23)
SAMPLE_IDS={"P":(0,),"S":(0,32),"A":(0,32),"D":(0,32)}


def locations(cfg):
    root=Path(cfg["paths"]["runs_root"])/"stage_b"
    if not str(root.resolve()).startswith("/lus/flare/"):raise ValueError("Aurora artifacts must stay on Lustre")
    root.mkdir(parents=True,exist_ok=True)
    return root,Path(cfg["paths"]["temporal_root"]),Path(cfg["paths"]["training_run"])


def physics(values,network):
    """Vectorized independent both-end AC flows, rating fractions, and balance residuals."""
    v=values["vm"]*np.exp(1j*values["va_rad"])
    e=network["branch"];f,t=e[:,:2].astype(int).T
    yff,yft,ytf,ytt=admittance_vectors(network)
    sf=v[...,f]*np.conj(yff*v[...,f]+yft*v[...,t])*network["baseMVA"]
    st=v[...,t]*np.conj(ytf*v[...,f]+ytt*v[...,t])*network["baseMVA"]
    result=dict(values,pf_mw=sf.real,qf_mvar=sf.imag,pt_mw=st.real,qt_mvar=st.imag)
    rated=network.rated_branches;loading=np.full(sf.shape,np.nan)
    loading[...,rated]=np.maximum(abs(sf[...,rated]),abs(st[...,rated]))/e[rated,5]
    injection=np.zeros(v.shape,dtype=complex)
    for line,(a,b) in enumerate(zip(f,t)):
        injection[...,a]+=sf[...,line];injection[...,b]+=st[...,line]
    bus=network["bus"]
    injection+=(bus[:,4]-1j*bus[:,5])*values["vm"]**2
    generation=np.zeros_like(injection)
    for i in np.flatnonzero(network.online_gen):
        generation[...,network.gen_bus[i]] += values["pg_mw"][...,i]
    generation[...,network.qg_buses] += 1j*values["qg_mvar"]
    residual=generation-values["pd_mw"]-1j*values["qd_mvar"]-injection
    for key in ("pf_mw","qf_mvar","pt_mw","qt_mvar"):
        result[key] = np.where(network.online_branch, result[key], np.nan)
    return dict(result,loading=loading,residual_p_mw=residual.real,residual_q_mvar=residual.imag)


def qd_from_pd(pd,network):
    p,q=network["bus"][:,2:4].T
    ratio=np.divide(q,p,out=np.zeros_like(q),where=p!=0)
    return np.where(p!=0,pd*ratio,q)


def frozen(base,network,pd,templates,qd=None):
    pd=np.asarray(pd);qd=qd_from_pd(pd,network) if qd is None else np.asarray(qd);k,h=pd.shape[:2]
    rows=[]
    for start in range(0,k,8):
        count=min(8,k-start);n=count*h
        batch=templates[n].clone();x=batch["bus"].x.clone()
        x[:,0]=torch.as_tensor(pd[start:start+count].reshape(-1),device=base.device,dtype=x.dtype)/base.normalizer.baseMVA
        x[:,1]=torch.as_tensor(qd[start:start+count].reshape(-1),device=base.device,dtype=x.dtype)/base.normalizer.baseMVA
        batch["bus"].x=x
        with torch.no_grad():output=base.model(batch)
        bus=output["bus"].reshape(count,h,network.n_bus,4).cpu().numpy()
        rows.append({"vm":bus[...,0],"va_rad":bus[...,1],
            "pg_mw":output["gen"].reshape(count,h,network.n_online_gen).cpu().numpy()*float(base.normalizer.baseMVA),
            "qg_mvar":bus[:,:,network.qg_buses,3]*float(base.normalizer.baseMVA)})
    values={key:np.concatenate([r[key] for r in rows]) for key in rows[0]}
    if not network.online_gen.all():
        full = np.full((k,h,network.n_gen),np.nan)
        full[...,network.online_gen] = values["pg_mw"]
        values["pg_mw"] = full
    return physics(dict(values,pd_mw=pd,qd_mvar=qd),network)


def load_model(cfg,device):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    if device.startswith("xpu"):torch.xpu.set_device(torch.device(device))
    net=load_network(config=cfg);base=Genco(checkpoint_assets(cfg),cfg["checkpoint"],device)
    scales=read_json(cfg["stage_b"]["coordinate_scales"])
    model=TemporalAdapter(base,net,scales).to(device).eval()
    train=Path(cfg["paths"]["training_run"])
    best=torch.load(train/"best.pt",weights_only=True,map_location="cpu")
    original=train/"manifest.training.json"
    assert best["manifest"]==read_json(original if original.exists() else train/"manifest.json")
    missing,extra=model.load_state_dict(best["model"],strict=False)
    assert not extra and all(k.startswith("backbone.") for k in missing)
    assert best["progress"]["best_epoch"]==read_json(train/"status.json")["best_epoch"]
    graph=graph_for_demand(net,net["bus"][:,2],net["bus"][:,3])
    templates={n:base.batch([graph]*n) for n in (24,192)}
    return net,base,model,scales,templates


def cache_test(cfg,base,network):
    root,data,_=locations(cfg);manifest=read_json(data/"manifest.json");files={}
    for episode in manifest["episodes"]:
        if episode["split"]!="test":continue
        name=episode["episode_id"];path=data/f"episodes/{name}.json";dest=root/f"cache/{name}.pt"
        key={"episode_sha256":sha256(path),"checkpoint":cfg["checkpoint"],"graphkit":cfg["upstream"]["graphkit"]}
        if dest.exists():assert torch.load(dest,mmap=True,weights_only=True)["key"]==key
        else:
            rows=read_json(path)["records"];arrays=episode_arrays(rows,network);embedding={"bus":[],"gen":[]}
            for lo in range(0,len(rows),192):
                graphs=[graph_for_demand(network,r["pd_mw"],r["qd_mvar"]) for r in rows[lo:lo+192]]
                with torch.no_grad():_,emb=base.model(base.batch(graphs),return_embeddings=True)
                for kind,n in (("bus",network.n_bus),("gen",network.n_online_gen)):embedding[kind].append(emb[kind].reshape(-1,n,emb[kind].shape[-1]).cpu())
            values={f"embedding_{k}":torch.cat(v) for k,v in embedding.items()}
            values["observed_bus"]=torch.tensor(np.stack([arrays[k] for k in ("pd","qd","vm","va_rad")],axis=-1),dtype=torch.float32)
            values["observed_gen"]=torch.tensor(np.stack([arrays[k] for k in ("pg_mw","qg_mvar")],axis=-1),dtype=torch.float32)[:,network.online_gen]
            for kind in ("bus","gen"):values[f"available_{kind}"]=torch.isfinite(values[f"observed_{kind}"])
            values["calendar"]=torch.from_numpy(arrays["calendar"])
            # Targets are loaded separately by the evaluation worker.
            atomic_torch_save(dest,{"key":key,"data":values})
        files[name]={"path":str(dest),"sha256":sha256(dest)}
    write_json(root/"cache/manifest.json",files)


def prepare(cfg,device):
    root,data,train=locations(cfg)
    net,base,model,scales,_=load_model(cfg,device)
    # Keep the exact training contract available for validating saved checkpoints.
    original=train/"manifest.training.json"
    if not original.exists():shutil.copy2(train/"manifest.json",original)
    measured=sum(p.numel() for p in model.parameters() if p.requires_grad)
    backbone=tensor_digest(model.backbone.state_dict())
    assert backbone==read_json(train/"status.json")["base_sha256"]
    manifest=read_json(original)
    manifest.update(trainable_parameters=measured,backbone_sha256=backbone,best_epoch=18,
        selection_curve_path=str(train/"epochs.json"),training_manifest_sha256=sha256(original),
        backbone_hash_definition="SHA256 of sorted state-dict names and contiguous CPU tensor bytes; includes buffers",
        evaluation_provenance={"requested_trainable_parameters":92345,
          "requested_backbone_sha256":"9dbf5596b4eb63a3b0e23a7d9a84427c33f3829a07ad2d7dbd12e99e364545a4",
          "requested_values_match_measurement":False,"measured_trainable_parameters":measured,
          "best_checkpoint_sha256":sha256(train/"best.pt"),"training_checkpoint_manifest_preserved":str(original)})
    write_json(train/"manifest.json",manifest)
    prepare_ridge(cfg,net,root/"forecaster")
    cache_test(cfg,base,net)
    dataset=read_json(data/"manifest.json")
    origins=[{"episode_id":e["episode_id"],"origin":o,"id":f"{e['episode_id']}_{o:04d}"}
             for e in dataset["episodes"] if e["split"]=="test" for o in range(72,481,24)]
    assert len(origins)==72
    pairs=variogram_pairs(scales,net);atomic_npz(root/"variogram_pairs.npz",pairs=pairs)
    protocol={"dataset_id":cfg["dataset"]["id"],"dataset_manifest_sha256":sha256(data/"manifest.json"),
        "origins":origins,"sample_counts":{"P":1,"S":64,"A":64,"D":64},
        "best_epoch":18,"checkpoint_sha256":sha256(train/"best.pt"),"seed":cfg["seed"],
        "ridge_manifest_sha256":sha256(root/"forecaster/manifest.json"),
        "coordinate_scales_sha256":sha256(cfg["stage_b"]["coordinate_scales"]),
        "variogram_pair_rule":"adjacent hours per retained coordinate plus physical connected-bus pairs; equal pair weights; p=0.5",
        "variogram_pairs_sha256":sha256(root/"variogram_pairs.npz"),"variogram_pairs":len(pairs),
        "marginals":"All physically defined components; Pd/Qd additionally reported on active buses in Gate B",
        "crps":"fair IID U-statistic; absolute error for P's point mass",
        "pit":"randomized finite-ensemble rank: (count_less + U*(count_equal+1))/(M+1)",
        "same_input_audit":{"origins":"all 72","sample_ids":SAMPLE_IDS,"lead_indices_zero_based":LEADS,
          "planned_unique_solves":72*len(LEADS)*5,"A_D_identical_inputs":True,
          "raw_refined_comparison":"Only the same audit subset; never compare its smaller ensemble against full-ensemble headline scores"},
        "solver":cfg["solver"],"refined_is_forecast":True,"solver_nonconvergence_is_infeasibility":False,
        "source_sha256":{p:sha256(ROOT/p) for p in ("scripts/evaluate_stage_b.py","training/forecaster.py","scoring.py","models/temporal.py")}}
    if (root/"protocol.json").exists():assert read_json(root/"protocol.json")==json.loads(json.dumps(protocol)),"Evaluation protocol changed"
    write_json(root/"protocol.json",protocol)
    print({"prepared_origins":len(origins),"residual_blocks":3130,"trainable_parameters":measured},flush=True)


def target_values(rows,network):
    values = [record_values(network, r) for r in rows]
    return physics({key:np.array([v[key] for v in values]) for key in values[0]},network)


def audit_samples(folder,values,targets,network,solver):
    """Fixed subset, all origins. A and D reuse the same numerical solutions."""
    reference_path=folder/"solver_references.json"
    records=read_json(reference_path) if reference_path.exists() else {}
    references={};available={};audit={}
    for mode in ("P","S","A"):
        sample_ids=SAMPLE_IDS[mode];shape=(len(sample_ids),len(LEADS))
        state={k:np.full((*shape,n),np.nan) for k,n in (("vm",network.n_bus),("va_rad",network.n_bus),("pg_mw",network.n_gen),("qg_mvar",network.n_gen))}
        valid=np.zeros(shape,dtype=bool)
        demand={k:values[mode][k][np.array(sample_ids)[:,None],np.array(LEADS)[None]] for k in ("pd_mw","qd_mvar")}
        for i,s in enumerate(sample_ids):
            for j,h in enumerate(LEADS):
                key=f"{mode}_{s}_{h}"
                if key not in records:
                    solved=solver.solve(demand["pd_mw"][i,j],demand["qd_mvar"][i,j])
                    checked=check_solution(network,demand["pd_mw"][i,j],demand["qd_mvar"][i,j],solved)
                    records[key]={"mode":mode,"sample_id":s,"lead_index":h,"pd_mw":demand["pd_mw"][i,j].tolist(),
                        "qd_mvar":demand["qd_mvar"][i,j].tolist(),"solver":solved,"check":checked}
                    write_json(reference_path,records)
                r=records[key];valid[i,j]=r["check"]["valid"]
                if valid[i,j]:
                    for k in state:state[k][i,j]=r["solver"]["state"][k]
        state["qg_mvar"] = bus_qg(network,state["qg_mvar"])
        references[mode]=physics(dict(state,**demand),network);available[mode]=valid
    references["D"]=references["A"];available["D"]=available["A"]
    assert np.array_equal(values["D"]["pd_mw"],values["A"]["pd_mw"])
    assert np.array_equal(values["D"]["qd_mvar"],values["A"]["qd_mvar"])
    for mode in MODES:
        ids=np.array(SAMPLE_IDS[mode]);leads=np.array(LEADS)
        raw={k:v[ids[:,None],leads[None]] for k,v in values[mode].items()}
        audit[mode]={"same_input_approximation":paired_error(raw,references[mode],available[mode]),
            "resolved":int(available[mode].sum()),"attempts":int(available[mode].size),
            "sample_ids":ids.tolist(),"lead_indices":leads.tolist(),"raw_vs_refined_forecast_subset":{}}
        # Matched subset means: require all selected reference samples resolved per hour.
        valid=available[mode].all(0)
        truth={k:v[leads] for k,v in targets.items()}
        for label,prediction in (("raw",raw),("solver_refined",references[mode])):
            mean={k:v.mean(0) for k,v in prediction.items()}
            audit[mode]["raw_vs_refined_forecast_subset"][label]=paired_error(mean,truth,valid)
        atomic_npz(folder/f"{mode}_solver_subset.npz",**references[mode],resolved=available[mode])
    return audit


def worker(cfg,device,shard,shards):
    root,data,train=locations(cfg);protocol=read_json(root/"protocol.json")
    protocol_hash=sha256(root/"protocol.json")
    net,base,model,scales,templates=load_model(cfg,device)
    base_before=tensor_digest(base.model.state_dict());ridge=RidgeForecaster(root/"forecaster")
    with np.load(root/"variogram_pairs.npz") as f:pairs=f["pairs"]
    cache_manifest=read_json(root/"cache/manifest.json");cached={};episodes={}
    solver=None;started=time.time();completed=[]
    try:
        for index,origin in enumerate(protocol["origins"]):
            if index%shards!=shard:continue
            folder=root/"origins"/origin["id"];folder.mkdir(parents=True,exist_ok=True)
            done=folder/"done.json"
            if done.exists():
                prior=read_json(done);assert prior["protocol_sha256"]==protocol_hash
                assert all(sha256(folder/k)==v for k,v in prior["artifacts"].items())
                completed.append(origin["id"]);continue
            name=origin["episode_id"];o=origin["origin"];seed=cfg["seed"]+10000+index
            if name not in episodes:
                rows=read_json(data/f"episodes/{name}.json")["records"]
                episodes[name]=(rows,episode_arrays(rows,net))
                cached[name]=torch.load(cache_manifest[name]["path"],weights_only=True,mmap=True)["data"]
            rows,arrays=episodes[name];targets=target_values(rows[o:o+24],net)
            means,samples,baseline_info=ridge.forecast(arrays,o,seed)
            values={}
            for mode in MODES:
                path=folder/f"{mode}.npz"
                if path.exists():
                    with np.load(path) as f:values[mode]={k:f[k] for k in f.files}
                    continue
                if mode in ("P","S"):values[mode]=frozen(base,net,means[None] if mode=="P" else samples,templates)
                elif mode=="A":
                    history={k:v[o-72:o].to(device) for k,v in cached[name].items() if k.startswith(("embedding_","observed_","available_"))}
                    calendar=torch.cat([cached[name]["calendar"][o:o+24],torch.arange(1,25)[:,None]/24],dim=1).to(device)
                    generator=torch.Generator().manual_seed(seed);chunks=[]
                    with torch.no_grad():
                        for _ in range(8):
                            noise={"factor":torch.randn(8,8,generator=generator),"bus":torch.randn(8,24,net.n_bus,3,generator=generator),
                                   "gen":torch.randn(8,24,net.n_online_gen,1,generator=generator)}
                            result=model(history,calendar,templates[192],{k:v.to(device) for k,v in noise.items()})
                            assert torch.equal(result["values"]["pd_mw"],result["demand_seen_by_physics_mw"]) or torch.allclose(result["values"]["pd_mw"],result["demand_seen_by_physics_mw"],atol=1e-5,rtol=1e-6)
                            chunks.append({k:v.cpu().numpy() for k,v in result["values"].items()})
                    values[mode]=physics({k:np.concatenate([v[k] for v in chunks]) for k in chunks[0]},net)
                else:values[mode]=frozen(base,net,values["A"]["pd_mw"],templates,qd=values["A"]["qd_mvar"])
                atomic_npz(path,**values[mode])
            masks={k:np.isfinite(v) for k,v in targets.items() if k in UNITS}
            atomic_npz(folder/"targets.npz",**targets)
            for mode in MODES:
                report,arrays_score=evaluate(values[mode],targets,masks,{"scales":scales,"variogram_pairs":pairs,
                    "mode":mode,"origin_id":origin["id"],"seed":seed})
                report["physical_balance"]={"p_mean_abs_mw":float(np.abs(values[mode]["residual_p_mw"]).mean()),
                    "q_mean_abs_mvar":float(np.abs(values[mode]["residual_q_mvar"]).mean())}
                write_json(folder/f"{mode}_scores.json",report)
                atomic_npz(folder/f"{mode}_scores.npz",**arrays_score)
            if solver is None:solver=IpoptSolver(cfg,root/f"logs/solver_{shard:02d}.log")
            audit=audit_samples(folder,values,targets,net,solver)
            for mode in MODES:write_json(folder/f"{mode}_audit.json",audit[mode])
            write_json(folder/"baseline.json",baseline_info)
            assert tensor_digest(base.model.state_dict())==base_before
            artifacts={p.name:sha256(p) for p in folder.iterdir() if p.is_file() and not p.name.startswith(".") and p.name!="done.json"}
            write_json(done,{"protocol_sha256":protocol_hash,"origin":origin,"seed":seed,
                "artifacts":artifacts,"backbone_unchanged":True,"node":platform.node(),"device":device})
            completed.append(origin["id"])
            print({"shard":shard,"origin":origin["id"],"completed":len(completed)},flush=True)
    finally:
        if solver is not None:solver.close()
    write_json(root/f"workers/shard_{shard:02d}.json",{"completed":completed,"device":device,"node":platform.node(),
        "wall_seconds":time.time()-started,"backbone_unchanged":tensor_digest(base.model.state_dict())==base_before})


def weighted_errors(rows):
    result={}
    for key in rows[0]:
        n=sum(r[key]["count"] for r in rows)
        result[key]={"count":n,"unit":rows[0][key]["unit"],**{m:sum(r[key][m]*r[key]["count"] for r in rows if r[key]["count"])/n if n else None for m in ("mae","mse")}}
    return result


def merge(cfg):
    root,_,_=locations(cfg);protocol=read_json(root/"protocol.json");ids=[o["id"] for o in protocol["origins"]]
    assert len(ids)==len(set(ids))==72
    for identity in ids:
        done=read_json(root/f"origins/{identity}/done.json")
        assert done["protocol_sha256"]==sha256(root/"protocol.json")
    table=["# Gate B — temporal_case30_v3","","72 held-out daily origins; P is deterministic; S/A/D use 64 joint samples. Raw forecasts only in the main table.","",
        "| Mode | Joint ES ↓ | Variogram ↓ | Active Pd MAE (MW) ↓ | Vm MAE (p.u.) ↓ | Vm CRPS ↓ | Vm 50% coverage | Vm 90% coverage | Vm 90% width | Line-1 loading MAE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    results={};active=np.flatnonzero(load_network(config=cfg)["bus"][:,2]!=0)
    for mode in MODES:
        reports=[read_json(root/f"origins/{identity}/{mode}_scores.json") for identity in ids]
        result=merge_origin_scores(reports);audits=[read_json(root/f"origins/{identity}/{mode}_audit.json") for identity in ids]
        result["same_input_approximation"]={"errors":weighted_errors([a["same_input_approximation"] for a in audits]),
            "resolved":sum(a["resolved"] for a in audits),"attempts":sum(a["attempts"] for a in audits),
            "sample_ids":SAMPLE_IDS[mode],"lead_indices":LEADS}
        result["raw_vs_refined_forecast_subset"]={label:weighted_errors([a["raw_vs_refined_forecast_subset"][label] for a in audits]) for label in ("raw","solver_refined")}
        result["physical_balance"]={k:float(np.mean([r["physical_balance"][k] for r in reports])) for k in reports[0]["physical_balance"]}
        result["active_pd_mae_mw"]=float(np.mean(np.asarray(result["variables"]["pd_mw"]["by_component"]["mae"])[active]))
        buses=np.asarray(cfg["stage_b"]["calibration"]["headline_voltage_bus_ids"])-1
        result["headline_voltage"]={"bus_ids":(buses+1).tolist(),**{k:float(np.mean(np.asarray(result["variables"]["vm"]["by_component"][k])[buses]))
            for k in ("mae","mse","crps","coverage_50","coverage_90","width_50","width_90")}}
        result["protocol_sha256"]=sha256(root/"protocol.json")
        write_json(root/f"{mode}.json",result);results[mode]=result
        vm=result["headline_voltage"]
        table.append(f"| {mode} | {result['joint_energy_score']:.6f} | {result['joint_variogram_p0_5']['score']:.6f} | {result['active_pd_mae_mw']:.4f} | {vm['mae']:.6f} | {vm['crps']:.6f} | {vm['coverage_50']:.1%} | {vm['coverage_90']:.1%} | {vm['width_90']:.6f} | {result['variables']['loading']['by_component']['mae'][0]:.5f} |")
    table += ["","Vm columns use the seven frozen headline buses {1,9,11,13,15,21,22}. Per-variable MSE, CRPS, physical energy/variogram scores, both interval widths, PIT histograms, all-bus/per-component and per-origin metrics are in P.json/S.json/A.json/D.json. Physical angle units are radians; loading is a fraction of rating.","",
        "## Same-input GENCO approximation audit","","Fixed subset: all origins, leads 1/7/13/19/24; samples 0 and 32 for S/A/D, sample 0 for P. A and D use identical demand inputs and numerical references. Nonconvergence remains unresolved.","",
        "| Mode | Resolved / attempted | Vm paired MAE (p.u.) | Pg paired MAE (MW) | Loading paired MAE |","|---|---:|---:|---:|---:|"]
    for mode,r in results.items():
        a=r["same_input_approximation"];e=a["errors"]
        table.append(f"| {mode} | {a['resolved']} / {a['attempts']} | {e['vm']['mae']:.6f} | {e['pg_mw']['mae']:.4f} | {e['loading']['mae']:.5f} |")
    table += ["","Raw versus solver-refined forecast errors are saved separately on this same smaller subset; they are not full 64-sample scores. No post-hoc calibration or F training is included.","",
        "Selected A checkpoint: epoch 18. Measured trainable parameters: 91,461. Backbone checksum: df0c0596b72eed5951189c854ae8f07063c389b08bea6f05d3fe00f8fb9eba4a. Residual pool: 3,130 training-only blocks."]
    (root/"gate_b.md").write_text("\n".join(table)+"\n")
    write_json(root/"gate_b.json",{"status":"forecast_evaluation_complete_probe_pending","origins":72,"modes":MODES,
        "protocol_sha256":sha256(root/"protocol.json"),"results":{m:sha256(root/f"{m}.json") for m in MODES}})
    print("\n".join(table),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("action",choices=["prepare","worker","merge","run"])
    p.add_argument("--device",default="xpu:0");p.add_argument("--shard",type=int,default=0);p.add_argument("--shards",type=int,default=1)
    args=p.parse_args();cfg=load_config();root,_,_=locations(cfg)
    if args.action=="prepare":prepare(cfg,args.device)
    elif args.action=="worker":worker(cfg,args.device,args.shard,args.shards)
    elif args.action=="merge":merge(cfg)
    else:
        count=torch.xpu.device_count() if args.device.startswith("xpu") else min(12,os.cpu_count() or 1)
        if args.device.startswith("xpu") and not count:raise RuntimeError("No XPU on this compute node")
        prepare(cfg,args.device)
        processes=[];streams=[]
        try:
            for shard in range(count):
                path=root/f"logs/evaluation_{shard:02d}.log";path.parent.mkdir(exist_ok=True)
                stream=path.open("a");streams.append(stream)
                device=f"xpu:{shard}" if args.device.startswith("xpu") else "cpu"
                cmd=[sys.executable,"-u","-m","scripts.evaluate_stage_b","worker","--device",device,"--shard",str(shard),"--shards",str(count)]
                processes.append(subprocess.Popen(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT))
            codes=[process.wait() for process in processes]
            write_json(root/"evaluation_job.json",{"job_id":os.environ.get("PBS_JOBID"),"node":platform.node(),"workers":count,"exit_codes":codes})
            if any(codes):raise RuntimeError(f"Evaluation worker failure: {codes}; per-origin files retained")
        finally:
            for process in processes:
                if process.poll() is None:process.terminate()
            for stream in streams:stream.close()
        merge(cfg)


if __name__=="__main__":main()
