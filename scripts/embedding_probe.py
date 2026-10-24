"""Last Gate B experiment: linear probes of added frozen GENCO representations."""
import os
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from threadpoolctl import threadpool_limits
import torch

from common import ROOT,load_config,read_json,sha256,write_json
from scoring import marginal_scores,summarize_marginals
from training.forecaster import atomic_npz


STATE_KEYS=("vm","va_rad","pg_mw","qg_mvar")
ALPHAS=(0.1,1.,10.,100.)
FAMILIES=("raw","raw_plus_genco")


def arrays(cfg,horizon):
    train=Path(cfg["paths"]["training_run"]);root=Path(cfg["paths"]["runs_root"])/"stage_b"
    sources=read_json(train/"cache/manifest.json")["files"]
    sources=dict(sources,**{k:dict(v,partition="test") for k,v in read_json(root/"cache/manifest.json").items()})
    parts={p:[] for p in ("train","model_selection","test")}
    for name,row in sources.items():
        assert sha256(row["path"])==row["sha256"]
        d=torch.load(row["path"],mmap=True,weights_only=True)["data"]
        b=d["observed_bus"].numpy();g=d["observed_gen"].numpy();calendar=d["calendar"].numpy()
        n=len(b)-horizon;t=np.arange(n)
        # Both families see the exact same whole-grid observed state and known calendar.
        raw=np.concatenate([b[t].reshape(n,-1),g[t].reshape(n,-1),calendar[t],calendar[t+horizon]],axis=-1)
        assert np.isfinite(raw).all(),"Missing histories need explicit fit-only imputation; never zero substitute"
        parts[row["partition"]].append({"raw":raw,"bus":d["embedding_bus"][:n].numpy(),
            "gen":d["embedding_gen"][:n].numpy(),"target_bus":b[t+horizon,:,2:4],
            "target_gen":g[t+horizon],"episode":np.repeat(name,n),"hours":t})
    return {p:{k:np.concatenate([r[k] for r in rows]) for k in rows[0]} for p,rows in parts.items()}


def ridge_candidates(x,y,selection_x,selection_y,mask):
    """Train-only standardization and selection-only alpha choice for two state heads."""
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    mean=x.mean(0);scale=x.std(0);scale[scale<1e-10]=1.
    z=(x-mean)/scale;selection=(selection_x-mean)/scale
    intercept=y.mean(0);centered=y-intercept
    eigen,vectors=np.linalg.eigh(z.T@z);eigen=np.maximum(eigen,0.)
    projection=vectors.T@(z.T@centered)
    candidates=[]
    for alpha in ALPHAS:
        beta=vectors@(projection/(eigen[:,None]+alpha))
        predicted=selection@beta+intercept
        score=marginal_scores(predicted[None],selection_y,np.broadcast_to(mask,selection_y.shape))
        valid=score["valid"]
        candidates.append((float(score["mse"][valid].mean()) if valid.any() else 0.,alpha,beta))
    score,alpha,beta=min(candidates,key=lambda v:(v[0],v[1]))
    return {"mean":mean,"scale":scale,"beta":beta,"intercept":intercept},alpha,score


def forecast(fit,x):return ((x-fit["mean"])/fit["scale"])@fit["beta"]+fit["intercept"]


def point_summary(prediction,target,mask):
    arrays=marginal_scores(prediction[None],target,mask)
    report=summarize_marginals(arrays)
    return {k:report[k] for k in ("mae","mse","count")}


def main():
    cfg=load_config();root=Path(cfg["paths"]["runs_root"])/"stage_b";out=root/"probe"
    assert (root/"gate_b.md").exists(),"Complete P/S/A/D evaluation before the probe"
    out.mkdir(exist_ok=True);torch.set_num_threads(1);started=time.time()
    scales=read_json(cfg["stage_b"]["coordinate_scales"])["features"]
    protocol={"source_sha256":sha256(__file__),"cache_api":"GNS_heterogeneous.forward(return_embeddings=True), final bus/gen latents",
        "horizons":[1,24],"families":FAMILIES,"alphas":ALPHAS,"fit_partition":"train",
        "alpha_partition":"model_selection","test_episodes":4,"target_crosses_episode":False,
        "raw_features":"Current whole-grid Pd/Qd/Vm/Va/Pg/Qg (132 coordinates) plus five calendar features at t and t+h",
        "embedding_features":"Same 142 raw/calendar features plus the target node's 384 frozen final latents",
        "heads":"Separate linear Vm/Va heads per bus and Pg/Qg heads per generator; alpha selected per node from selection normalized MSE",
        "claim_limit":"Incremental predictive association and differing feature dimensions; not causal evidence or proof of generality",
        "dataset_manifest_sha256":sha256(Path(cfg["paths"]["temporal_root"])/"manifest.json")}
    write_json(out/"protocol.json",protocol)
    all_results={}
    with threadpool_limits(limits=16):
        for horizon in (1,24):
            data=arrays(cfg,horizon);node_results=[];predictions={f:{} for f in FAMILIES}
            for kind,keys in (("bus",("vm","va_rad")),("gen",("pg_mw","qg_mvar"))):
                nnode=data["train"][f"target_{kind}"].shape[1]
                for family in FAMILIES:
                    for key in keys:predictions[family][key]=np.empty((len(data["test"]["raw"]),nnode))
                for node in range(nnode):
                    yscale=np.array([scales[k]["scale"][node] for k in keys])
                    mask=np.array([scales[k]["joint_score_mask"][node] for k in keys],dtype=bool)
                    targets={p:d[f"target_{kind}"][:,node]/yscale for p,d in data.items()}
                    for family in FAMILIES:
                        x={p:d["raw"] if family=="raw" else np.concatenate([d["raw"],d[kind][:,node]],axis=-1) for p,d in data.items()}
                        fit,alpha,selection=ridge_candidates(x["train"],targets["train"],x["model_selection"],targets["model_selection"],mask)
                        predicted=forecast(fit,x["test"])*yscale
                        for j,key in enumerate(keys):predictions[family][key][:,node]=predicted[:,j]
                        atomic_npz(out/f"coefficients/{horizon}_{family}_{kind}_{node:02d}.npz",**fit,yscale=yscale,score_mask=mask)
                        node_results.append({"family":family,"node_type":kind,"node_index":node,"alpha":alpha,
                            "selection_normalized_mse":selection,"features":x["train"].shape[-1]})
                print({"probe_horizon":horizon,"node_type":kind,"nodes_complete":nnode},flush=True)
            targets={"vm":data["test"]["target_bus"][...,0],"va_rad":data["test"]["target_bus"][...,1],
                     "pg_mw":data["test"]["target_gen"][...,0],"qg_mvar":data["test"]["target_gen"][...,1]}
            result={"train_hours":len(data["train"]["raw"]),"selection_hours":len(data["model_selection"]["raw"]),
                "test_hours":len(data["test"]["raw"]),"horizon":horizon,"nodes":node_results,"families":{}}
            saved={f"target_{k}":v for k,v in targets.items()}
            saved.update(episode=data["test"]["episode"],hour=data["test"]["hours"])
            for family in FAMILIES:
                variables={};normalized=[];truth_normalized=[];masks=[]
                for key in STATE_KEYS:
                    predicted=predictions[family][key];target=targets[key]
                    mask=np.broadcast_to(scales[key]["joint_score_mask"],target.shape)
                    variables[key]=point_summary(predicted,target,mask)
                    normalized.append(predicted/np.asarray(scales[key]["scale"]))
                    truth_normalized.append(target/np.asarray(scales[key]["scale"]))
                    masks.append(mask);saved[f"{family}_{key}"]=predicted
                xn,yn,m=np.concatenate(normalized,axis=-1),np.concatenate(truth_normalized,axis=-1),np.concatenate(masks,axis=-1)
                result["families"][family]={"variables":variables,"normalized":point_summary(xn,yn,m),"per_episode":{}}
                for episode in sorted(set(data["test"]["episode"])):
                    keep=data["test"]["episode"]==episode
                    result["families"][family]["per_episode"][str(episode)]=point_summary(xn[keep],yn[keep],m[keep])
            atomic_npz(out/f"predictions_h{horizon}.npz",**saved)
            write_json(out/f"horizon_{horizon}.json",result);all_results[str(horizon)]=result
    fig,axes=plt.subplots(1,2,figsize=(9,4.8),layout="constrained")
    colors=("#586c82","#19766a")
    for ax,h in zip(axes,(1,24)):
        r=all_results[str(h)];values=[r["families"][f]["normalized"]["mse"] for f in FAMILIES]
        ax.bar([0,1],values,color=colors,width=.6)
        for i,family in enumerate(FAMILIES):
            points=[v["mse"] for v in r["families"][family]["per_episode"].values()]
            ax.scatter(np.arange(4)*.045-.0675+i,points,color="black",s=18,zorder=3)
            ax.text(i,values[i],f"{values[i]:.3f}",ha="center",va="bottom",fontsize=10)
        ax.set_xticks([0,1],["Raw + calendar","Raw + calendar\n+ GENCO latents"])
        ax.set_title(f"State at t + {h} hour{'s' if h>1 else ''}")
        ax.set_ylabel("Train-scaled MSE (lower is better)");ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
    fig.suptitle("Do frozen GENCO embeddings add future-state information?",fontsize=13)
    fig.text(.5,-.035,"142 versus 526 features per node; same observed inputs and targets. Alpha chosen on selection episodes.\nBlack points: four test episodes. Frozen embeddings use the upstream return_embeddings API.",ha="center",fontsize=9)
    fig.savefig(out/"embedding_probe.png",dpi=180,bbox_inches="tight");plt.close(fig)
    report={"protocol":protocol,"horizons":all_results,"figure":str(out/"embedding_probe.png"),
        "wall_seconds":time.time()-started,"job_id":os.environ.get("PBS_JOBID")}
    write_json(out/"report.json",report)
    gate=read_json(root/"gate_b.json");gate.update(status="complete",probe_report=str(out/"report.json"),
        probe_report_sha256=sha256(out/"report.json"),probe_figure=str(out/"embedding_probe.png"))
    write_json(root/"gate_b.json",gate)
    with (root/"gate_b.md").open("a") as stream:
        stream.write("\n## Frozen embedding probe\n\n")
        stream.write("Separate node-wise ridge heads; raw/current solved state and calendar in both families. Adding 384 final GENCO latents increases each node's feature width from 142 to 526. Regularization uses selection episodes only.\n\n")
        stream.write("| Horizon | Raw normalized MSE | Raw + frozen GENCO normalized MSE |\n|---|---:|---:|\n")
        for h,r in all_results.items():stream.write(f"| +{h} h | {r['families']['raw']['normalized']['mse']:.6f} | {r['families']['raw_plus_genco']['normalized']['mse']:.6f} |\n")
        stream.write("\nFigure: [embedding_probe.png](probe/embedding_probe.png). Feature dimensions differ; predictive association is not a causal/generalization claim.\n")
    print({"probe_complete":True,"wall_seconds":report["wall_seconds"],"figure":report["figure"]},flush=True)


if __name__=="__main__":main()
