"""One frozen embedding cache; memory-mapped per-hour tensors yield causal histories."""
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from common import ROOT,read_json,sha256,write_json
from data.download.prepare import checkpoint_assets
from data.coordinates import record_values
from data.loader.dataset import load_temporal_manifest
from data.loader.network import graph_for_demand,load_network
from models.genco import Genco
from solver.checks import physical_quantities
from training.state import atomic_torch_save


def calendar_features(times):
    ts=pd.to_datetime(times,utc=True);h,d=ts.hour.to_numpy(),ts.dayofweek.to_numpy()
    return torch.tensor(np.stack([np.sin(2*np.pi*h/24),np.cos(2*np.pi*h/24),
        np.sin(2*np.pi*d/7),np.cos(2*np.pi*d/7),(d>=5).astype(float)],axis=-1),dtype=torch.float32)


def build_cache(cfg, device, deadline_epoch=None):
    started=time.perf_counter()
    batch_size=int(cfg["stage_b"].get("cache_graph_batch_size",192))
    if batch_size < 1:raise ValueError("Cache graph batch size must be positive")
    root=Path(cfg["paths"]["temporal_root"]);out=Path(cfg["paths"]["training_run"])/"cache"
    out.mkdir(parents=True,exist_ok=True)
    net=load_network(config=cfg)
    manifest=load_temporal_manifest(root,net);scales=read_json(ROOT/cfg["stage_b"]["coordinate_scales"])
    assert scales["benchmark_id"]==manifest["benchmark_id"]
    base=Genco(checkpoint_assets(cfg),cfg["checkpoint"],device,grid=net)
    hashes={r["file"]:r["sha256"] for r in manifest["shards"]}
    files={}
    # Training setup never reads calibration/test solved data.
    for ep in manifest["episodes"]:
        if ep["split"] not in ("train","model_selection"):continue
        name=ep["episode_id"];path=root/f"episodes/{name}.json"
        assert sha256(path)==hashes[f"episodes/{name}.json"]
        key={"benchmark_id":manifest["benchmark_id"],"episode_sha256":sha256(path),
            "checkpoint":cfg["checkpoint"],"graphkit":cfg["upstream"]["graphkit"],
            "scales_sha256":sha256(ROOT/cfg["stage_b"]["coordinate_scales"]),"cache_version":1}
        if batch_size != 192:key["graph_batch_size"]=batch_size
        dest=out/f"{name}.pt"
        if dest.exists():
            value=torch.load(dest,mmap=True,map_location="cpu",weights_only=True)
            if value["key"]!=key:raise ValueError("Stale embedding/history cache")
        else:
            rows=read_json(path)["records"];chunks={"bus":[],"gen":[]}
            for lo in range(0,len(rows),batch_size):
                if deadline_epoch is not None and time.time() >= deadline_epoch-60:
                    return {"incomplete":True,"reason":"allocation_guard"}
                # Only demand/static topology reaches frozen embeddings; solved state is separate.
                graphs=[graph_for_demand(net,r["pd_mw"],r["qd_mvar"]) for r in rows[lo:lo+batch_size]]
                with torch.no_grad():_,emb=base.model(base.batch(graphs),return_embeddings=True)
                for kind,n in (("bus",net.n_bus),("gen",net.n_online_gen)):chunks[kind].append(emb[kind].reshape(-1,n,emb[kind].shape[-1]).cpu())
            data={f"embedding_{k}":torch.cat(v) for k,v in chunks.items()}
            observed_bus=[];observed_gen=[];targets=[]
            for r in rows:
                state=r["target_state"] or {"vm":[np.nan]*net.n_bus,"va_rad":[np.nan]*net.n_bus,"pg_mw":[np.nan]*net.n_gen,"qg_mvar":[np.nan]*net.n_gen}
                observed_bus.append(np.stack([r["pd_mw"],r["qd_mvar"],state["vm"],state["va_rad"]],axis=-1))
                observed_gen.append(np.stack([np.asarray(state["pg_mw"])[net.online_gen].astype(float),
                    np.asarray(state["qg_mvar"])[net.online_gen].astype(float)],axis=-1))
                physics=physical_quantities(net,r["pd_mw"],r["qd_mvar"],state)
                values=record_values(net,r)
                targets.append(np.concatenate([np.asarray(values[k])[s["joint_score_mask"]]
                    for k,s in scales["features"].items() if any(s["joint_score_mask"])]))
            for kind,values in (("bus",observed_bus),("gen",observed_gen)):
                data[f"observed_{kind}"]=torch.tensor(np.array(values),dtype=torch.float32)
                data[f"available_{kind}"]=torch.isfinite(data[f"observed_{kind}"])
            data["targets"]=torch.tensor(np.array(targets),dtype=torch.float32)
            data["calendar"]=calendar_features([r["time_utc"] for r in rows])
            atomic_torch_save(dest,{"key":key,"data":data})
        files[name]={"path":str(dest),"sha256":sha256(dest),"partition":ep["split"]}
        print("cached",name,flush=True)
    report={"benchmark_id":manifest["benchmark_id"],"files":files,"wall_seconds":time.perf_counter()-started,
        "history_storage":"per-hour tensors memory-mapped once; 72-hour views contain only hours before origin",
        "partitions_cached":["train","model_selection"],"device":device}
    write_json(out/"manifest.json",report)
    return report


def causal_window(data, origin):
    history={k:v[origin-72:origin] for k,v in data.items() if k.startswith(("embedding_","observed_","available_"))}
    calendar=torch.cat([data["calendar"][origin:origin+24],torch.arange(1,25,dtype=torch.float32)[:,None]/24],dim=1)
    return history,calendar,data["targets"][origin:origin+24].flatten()


class CachedWindows:
    def __init__(self,cfg,grid):
        root=Path(cfg["paths"]["training_run"])/"cache"
        self.manifest=read_json(root/"manifest.json")
        expected=load_temporal_manifest(cfg["paths"]["temporal_root"],grid)["benchmark_id"]
        if self.manifest["benchmark_id"]!=expected:raise ValueError("Wrong dataset cache")
        self.data={};self.indices={"train":[],"model_selection":[]}
        for name,row in self.manifest["files"].items():
            if sha256(row["path"])!=row["sha256"]:raise ValueError("Embedding cache checksum changed")
            value=torch.load(row["path"],mmap=True,map_location="cpu",weights_only=True)
            if value["key"]["checkpoint"]!=cfg["checkpoint"]:raise ValueError("Wrong checkpoint in embedding cache")
            self.data[name]=value["data"]
            stride=1 if row["partition"]=="train" else 24
            self.indices[row["partition"]].extend((name,o) for o in range(72,len(self.data[name]["calendar"])-24+1,stride))

    def window(self, identity, device):
        name,origin=identity
        history,calendar,target=causal_window(self.data[name],origin)
        return {k:v.to(device) for k,v in history.items()},calendar.to(device),target.to(device)
