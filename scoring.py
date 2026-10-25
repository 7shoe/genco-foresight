"""Shared forecast/approximation scores, with explicit masks and physical units."""
import numpy as np
import torch


def fair_energy_score(samples, target, scale, mask=None):
    """IID-ensemble U-statistic, all off-diagonal pairs, identical coordinate masks."""
    if samples.ndim != 2 or samples.shape[0] < 2:
        raise ValueError("Expected at least two complete flattened trajectory samples")
    observed = torch.isfinite(target) if mask is None else mask & torch.isfinite(target)
    if not observed.any():
        raise ValueError("No observed target coordinates")
    k = samples.shape[0]
    x = samples[:,observed]/scale[observed]
    y = target[observed]/scale[observed]
    distance_to_target = torch.linalg.vector_norm(x-y,dim=-1).mean()
    pair_distance = torch.linalg.vector_norm(x[:,None]-x[None,:],dim=-1)
    return (distance_to_target-pair_distance.sum()/(2*k*(k-1)))/observed.sum().sqrt()


UNITS = {"pd_mw":"MW","qd_mvar":"MVAr","vm":"p.u.","va_rad":"rad",
         "pg_mw":"MW","qg_mvar":"MVAr","pf_mw":"MW","qf_mvar":"MVAr",
         "pt_mw":"MW","qt_mvar":"MVAr","loading":"fraction_of_rating"}


def ensemble_energy(samples, target, scale, mask=None):
    samples=np.asarray(samples,dtype=np.float64).reshape(len(samples),-1)
    target=np.asarray(target,dtype=np.float64).reshape(-1)
    scale=np.broadcast_to(scale,np.asarray(mask).shape if mask is not None else target.shape).reshape(-1)
    observed=np.isfinite(target) if mask is None else np.asarray(mask).reshape(-1)&np.isfinite(target)
    if not observed.any():return None
    if not np.isfinite(samples[:,observed]).all():raise ValueError("Nonfinite forecast on scored coordinate")
    if len(samples)==1:
        return float(np.linalg.norm((samples[0,observed]-target[observed])/scale[observed])/np.sqrt(observed.sum()))
    return float(fair_energy_score(torch.from_numpy(samples),torch.from_numpy(target),
                                   torch.from_numpy(scale.copy()),torch.from_numpy(observed)))


def marginal_scores(samples, target, mask=None, seed=42):
    """Fair CRPS for IID ensembles; deterministic point forecast uses absolute error."""
    x=np.asarray(samples,dtype=float);y=np.asarray(target,dtype=float)
    valid=np.isfinite(y) if mask is None else np.asarray(mask,dtype=bool)&np.isfinite(y)
    if not np.isfinite(x[:,valid]).all():raise ValueError("Nonfinite ensemble values")
    m=len(x);mean=x.mean(0);ordered=np.sort(x,axis=0)
    absolute=np.abs(x-y).mean(0)
    correction=(np.arange(m)*2-m+1).reshape((m,)+(1,)*y.ndim)
    crps=absolute if m==1 else absolute-(ordered*correction).sum(0)/(m*(m-1))
    less=(x<y).sum(0);equal=(x==y).sum(0)
    pit=(less+np.random.default_rng(seed).random(y.shape)*(equal+1))/(m+1)
    arrays={"mae":np.abs(mean-y),"mse":(mean-y)**2,"crps":crps,"pit":pit}
    for level in (50,90):
        tail=(1-level/100)/2
        lo,hi=np.quantile(x,[tail,1-tail],axis=0,method="linear")
        arrays[f"coverage_{level}"]=((y>=lo)&(y<=hi)).astype(float)
        arrays[f"width_{level}"]=hi-lo
    arrays={k:np.where(valid,v,np.nan) for k,v in arrays.items()}
    arrays["valid"]=valid
    return arrays


def summarize_marginals(arrays):
    valid=arrays["valid"];n=int(valid.sum())
    result={"count":n}
    for k,v in arrays.items():
        if k not in ("valid","pit"):result[k]=float(v[valid].mean()) if n else None
    hist=np.histogram(arrays["pit"][valid],bins=np.linspace(0,1,11))[0]
    result["pit_histogram"]=hist.tolist();result["pit_edges"]=np.linspace(0,1,11).tolist()
    result["by_component"]={"count":valid.sum(0).tolist()}
    for k,v in arrays.items():
        if k in ("valid","pit"):continue
        counts=valid.sum(0)
        means=np.divide(np.where(valid,v,0.).sum(0),counts,out=np.full(counts.shape,np.nan),where=counts>0)
        result["by_component"][k]=[float(v) if np.isfinite(v) else None for v in means]
    return result


def joint_coordinates(values, scales):
    return np.concatenate([np.asarray(values[k])[...,s["joint_score_mask"]]
        for k,s in scales["features"].items() if any(s["joint_score_mask"])],axis=-1)


def variogram_pairs(scales, network, horizon=24):
    """All adjacent-hour coordinates and connected-bus pairs, frozen across modes."""
    coordinate=[]
    for key,s in scales["features"].items():
        coordinate.extend((key,i) for i,keep in enumerate(s["joint_score_mask"]) if keep)
    d=len(coordinate);lookup={v:i for i,v in enumerate(coordinate)};pairs=[]
    for t in range(horizon-1):pairs.extend((t*d+i,(t+1)*d+i) for i in range(d))
    for t in range(horizon):
        for key in ("pd_mw","qd_mvar","vm","va_rad"):
            for f,to in network["branch"][getattr(network, "online_branch", np.ones(len(network["branch"]), dtype=bool)),:2].astype(int):
                if (key,f) in lookup and (key,to) in lookup:
                    pairs.append((t*d+lookup[key,f],t*d+lookup[key,to]))
    return np.asarray(sorted(set(tuple(sorted(p)) for p in pairs)),dtype=np.int64)


def variogram_score(samples,target,pairs,scale,mask=None):
    x=np.asarray(samples,dtype=float).reshape(len(samples),-1)/np.asarray(scale).reshape(-1)
    y=np.asarray(target,dtype=float).reshape(-1)/np.asarray(scale).reshape(-1)
    valid=np.isfinite(y) if mask is None else np.asarray(mask).reshape(-1)&np.isfinite(y)
    i,j=np.asarray(pairs).T;keep=valid[i]&valid[j];i,j=i[keep],j[keep]
    if not len(i):return {"score":None,"count":0}
    forecast=np.sqrt(np.abs(x[:,i]-x[:,j])).mean(0)
    truth=np.sqrt(np.abs(y[i]-y[j]))
    return {"score":float(np.mean((forecast-truth)**2)),"count":len(i)}


def evaluate(predictions, targets, masks, protocol):
    """Score one origin; samples [sample,hour,component], targets [hour,component]."""
    scales=protocol["scales"];pairs=protocol["variogram_pairs"]
    result={"mode":protocol["mode"],"origin_id":protocol["origin_id"],"prediction_type":"raw",
            "reference":"realized_hourly_AC_OPF","samples":len(predictions["vm"]),
            "no_uncertainty":len(predictions["vm"])==1,"variables":{}}
    detailed={}
    for index,key in enumerate(UNITS):
        arrays=marginal_scores(predictions[key],targets[key],masks[key],protocol["seed"]+index)
        row=summarize_marginals(arrays);row["unit"]=UNITS[key]
        target=np.asarray(targets[key]);m=np.asarray(masks[key])
        row["energy_score_physical"]=ensemble_energy(predictions[key],target,np.ones(target.shape),m)
        n=target.shape[-1];h=target.shape[0]
        temporal=np.array([(t*n+i,(t+1)*n+i) for t in range(h-1) for i in range(n)],dtype=int)
        row["variogram_p0_5_physical"]=variogram_score(predictions[key],target,temporal,np.ones(target.size),m)
        result["variables"][key]=row
        detailed.update({f"{key}__{k}":v for k,v in arrays.items()})
    x=joint_coordinates(predictions,scales);y=joint_coordinates(targets,scales)
    mask=joint_coordinates(masks,scales).astype(bool)
    scale=np.tile(scales["scale_vector_one_hour"],y.shape[0])
    result["joint_energy_score"]=ensemble_energy(x,y,scale,mask.reshape(-1))
    result["joint_variogram_p0_5"]=variogram_score(x,y,pairs,scale,mask)
    result["joint_observed_coordinates"]=int(mask.sum())
    return result,detailed


def paired_error(predictions, references, valid):
    """Same-input neural approximation: paired sample/state errors, not forecast scores."""
    report={}
    for key in UNITS:
        if key in ("pd_mw","qd_mvar"):continue
        x,y=np.asarray(predictions[key]),np.asarray(references[key])
        mask=np.broadcast_to(np.asarray(valid)[...,None],y.shape)&np.isfinite(y)
        if not np.isfinite(x[mask]).all():raise ValueError("Nonfinite paired neural state")
        error=x-y;n=int(mask.sum())
        report[key]={"count":n,"unit":UNITS[key],
            "mae":float(np.abs(error[mask]).mean()) if n else None,
            "mse":float((error[mask]**2).mean()) if n else None}
    return report


def merge_origin_scores(reports):
    """Preserve per-origin scores and combine coordinate metrics using valid counts."""
    result={"origins":len(reports),"mode":reports[0]["mode"],"samples":reports[0]["samples"],
            "prediction_type":"raw","variables":{},"per_origin":reports}
    for field in ("joint_energy_score",):
        values=[r[field] for r in reports if r[field] is not None]
        result[field]=float(np.mean(values)) if values else None
    rows=[r["joint_variogram_p0_5"] for r in reports];n=sum(r["count"] for r in rows)
    result["joint_variogram_p0_5"]={"score":sum(r["score"]*r["count"] for r in rows if r["count"])/n if n else None,"count":n}
    for key in UNITS:
        rows=[r["variables"][key] for r in reports];n=sum(r["count"] for r in rows)
        row={"count":n,"unit":UNITS[key]}
        for name in ("mae","mse","crps","coverage_50","coverage_90","width_50","width_90"):
            row[name]=sum(r[name]*r["count"] for r in rows if r["count"])/n if n else None
        for name in ("energy_score_physical",):
            values=[r[name] for r in rows if r[name] is not None]
            row[name]=float(np.mean(values)) if values else None
        vp=[r["variogram_p0_5_physical"] for r in rows];nv=sum(r["count"] for r in vp)
        row["variogram_p0_5_physical"]={"score":sum(r["score"]*r["count"] for r in vp if r["count"])/nv if nv else None,"count":nv}
        row["pit_histogram"]=np.sum([r["pit_histogram"] for r in rows],axis=0).tolist()
        row["pit_edges"]=rows[0]["pit_edges"]
        counts=np.sum([r["by_component"]["count"] for r in rows],axis=0)
        row["by_component"]={"count":counts.tolist()}
        for name in ("mae","mse","crps","coverage_50","coverage_90","width_50","width_90"):
            numer=np.zeros(len(counts))
            for r in rows:
                v=np.array([np.nan if x is None else x for x in r["by_component"][name]])
                numer+=np.nan_to_num(v)*r["by_component"]["count"]
            means=np.divide(numer,counts,out=np.full(counts.shape,np.nan),where=counts>0)
            row["by_component"][name]=[float(v) if np.isfinite(v) else None for v in means]
        result["variables"][key]=row
    return result
