"""Per-active-bus direct ridge forecasts and strictly causal joint residual blocks."""
from pathlib import Path
import os
import tempfile

import numpy as np

from common import read_json, sha256, write_json
from training.inputs import calendar_features


LAGS = (1, 2, 24, 48)
COVARIATES = ("pd_mw", "qd_mvar", "vm", "va_rad", "bus_pg_mw", "bus_qg_mvar")


def atomic_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def episode_arrays(rows, network):
    pd = np.asarray([r["pd_mw"] for r in rows], dtype=float)
    qd = np.asarray([r["qd_mvar"] for r in rows], dtype=float)
    state = {k: np.asarray([(r["target_state"] or {}).get(k, [np.nan]*n)
                            for r in rows], dtype=float)
             for k, n in (("vm",30),("va_rad",30),("pg_mw",6),("qg_mvar",6))}
    pg, qg = np.zeros_like(pd), np.zeros_like(pd)
    pg[:, network["gen_bus"]] = state["pg_mw"]
    qg[:, network["gen_bus"]] = state["qg_mvar"]
    covariates = np.stack([pd, qd, state["vm"], state["va_rad"], pg, qg], axis=-1)
    return {"covariates": covariates, "pd": pd, "qd": qd, **state,
            "calendar": calendar_features([r["time_utc"] for r in rows]).numpy()}


def design(data, origins, active):
    """[origin,bus,horizon,feature]; solved covariates are strictly before origin."""
    origins = np.asarray(origins, dtype=int)
    past = data["covariates"][origins[:,None]-np.array(LAGS)[None]][:,:,active]
    past = past.transpose(0,2,1,3).reshape(len(origins), len(active), -1)
    available = np.isfinite(past)
    # NaNs are imputed by the fit's train-only means; explicit masks remain features.
    past = np.concatenate([past, available.astype(float)], axis=-1)
    calendar = data["calendar"][origins[:,None]+np.arange(24)]
    return np.concatenate([np.broadcast_to(past[:,:,None], (*past.shape[:2],24,past.shape[-1])),
        np.broadcast_to(calendar[:,None], (len(origins),len(active),24,5))], axis=-1)


def fit_ridge(x, y, alpha=1.):
    """Independent bus/horizon models; intercept unpenalized, feature scaling on fit rows."""
    # x [n,b,h,f], y [n,b,h]. Only demand targets are required and are always known.
    finite = np.isfinite(x)
    count = finite.sum(axis=0)
    mean = np.divide(np.where(finite,x,0.).sum(axis=0),count,
                     out=np.zeros(x.shape[1:]),where=count>0)
    centered = np.where(finite,x-mean,0.)
    scale = np.sqrt(np.mean(centered**2,axis=0));scale[scale<1e-10]=1.
    z = centered/scale
    ymean = y.mean(axis=0)
    gram = np.einsum("nbhf,nbhg->bhfg",z,z,optimize=True)
    gram += alpha*np.eye(x.shape[-1])
    rhs = np.einsum("nbhf,nbh->bhf",z,y-ymean,optimize=True)
    beta = np.linalg.solve(gram,rhs[...,None])[...,0]
    return {"mean":mean,"scale":scale,"beta":beta,"intercept":ymean}


def predict_ridge(fit, x):
    z = np.where(np.isfinite(x),x-fit["mean"],0.)/fit["scale"]
    return np.einsum("...bhf,bhf->...bh",z,fit["beta"])+fit["intercept"]


def prefix_fit_from_moments(count, sums, cross, targets, xy, alpha=1.):
    """Same standardized ridge as fit_ridge; avoids rebuilding each growing prefix."""
    mean = sums/count
    centered = cross-count*mean[..., :,None]*mean[...,None,:]
    variance = np.maximum(np.diagonal(centered,axis1=-2,axis2=-1)/count,0.)
    scale = np.sqrt(variance);scale[scale<1e-10]=1.
    gram = centered/(scale[..., :,None]*scale[...,None,:])
    gram += alpha*np.eye(mean.shape[-1])
    ymean = targets/count
    rhs = (xy-count*mean*ymean[...,None])/scale
    beta = np.linalg.solve(gram,rhs[...,None])[...,0]
    return {"mean":mean,"scale":scale,"beta":beta,"intercept":ymean}


def prepare(cfg, network, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    root=Path(cfg["paths"]["temporal_root"]);dataset=read_json(root/"manifest.json")
    episodes=[e for e in dataset["episodes"] if e["split"]=="train"]
    active=np.flatnonzero(network["bus"][:,2]!=0)
    provenance={"dataset_manifest_sha256":sha256(root/"manifest.json"),
        "episodes":[e["episode_id"] for e in episodes],"active_bus_ids":network["bus_ids"][active].tolist(),
        "lags_hours":list(LAGS),"covariates":list(COVARIATES),"direct_heads":24,"alpha":1.,
        "origins":list(range(72,481)),"residual_origins":list(range(168,481)),
        "residual_blocks":len(episodes)*313,"fit_rule":"Only full 24-hour target blocks ending strictly before each rolling origin, across train episode prefixes",
        "residuals":"Observed demand minus zero-clipped rolling prediction; 24-hour all-bus blocks, overlapping and not independent",
        "source_sha256":sha256(__file__),"seed":cfg["seed"],
        "feature_scaling":"Within each fit; missing past values imputed with fit means plus explicit availability masks"}
    if (output/"manifest.json").exists():
        prior=read_json(output/"manifest.json")
        assert prior["protocol"]==provenance,"Forecaster protocol changed"
        assert all(sha256(output/k)==v for k,v in prior["artifacts"].items())
        return prior
    xs=[];ys=[]
    for e in episodes:
        path=root/f"episodes/{e['episode_id']}.json"
        rows=read_json(path)["records"];data=episode_arrays(rows,network)
        origins=np.arange(72,481)
        xs.append(design(data,origins,active))
        ys.append(data["pd"][origins[:,None]+np.arange(24)][:,:,active].transpose(0,2,1))
    x=np.asarray(xs);y=np.asarray(ys)  # [episode,origin,bus,horizon,(feature)]
    # All v3 states are resolved. Refuse to silently use a noncausal imputation shortcut.
    if not np.isfinite(x).all():
        raise ValueError("Missing historical covariates require prefix-specific imputation; refusing to fit")
    final=fit_ridge(x.reshape(-1,*x.shape[2:]),y.reshape(-1,*y.shape[2:]))
    atomic_npz(output/"coefficients.npz",**final,active=active)
    residuals=[];identities=[];clipped=0;total=0
    shape=x.shape[2:]; sums=np.zeros(shape);cross=np.zeros((*shape,shape[-1]))
    targets=np.zeros(shape[:2]);xy=np.zeros(shape);count=0;last=-1
    # Prefix statistics add each fully matured training window exactly once.
    for origin in range(168,481):
        mature=origin-24-72
        additions=x[:,last+1:mature+1].reshape(-1,*shape)
        labels=y[:,last+1:mature+1].reshape(-1,*shape[:2])
        count+=len(additions);sums+=additions.sum(0);targets+=labels.sum(0)
        cross+=np.einsum("nbhf,nbhg->bhfg",additions,additions,optimize=True)
        xy+=np.einsum("nbhf,nbh->bhf",additions,labels,optimize=True)
        last=mature
        fitted=prefix_fit_from_moments(count,sums,cross,targets,xy)
        predicted=predict_ridge(fitted,x[:,origin-72])
        clipped+=int((predicted<0).sum());total+=predicted.size
        predicted=np.maximum(predicted,0.)
        diff=(y[:,origin-72]-predicted).transpose(0,2,1)
        block=np.zeros((len(episodes),24,30));block[:,:,active]=diff
        residuals.extend(block)
        identities.extend((i,origin) for i in range(len(episodes)))
        if origin%48==0:print({"ridge_residual_origin":origin,"blocks":len(residuals)},flush=True)
    assert len(residuals)==3130
    atomic_npz(output/"residual_blocks.npz",residuals=np.asarray(residuals,dtype=np.float32),
               identities=np.asarray(identities,dtype=np.int64))
    result={"protocol":provenance,"rolling_negative_demand_fraction":clipped/total,
            "artifacts":{k:sha256(output/k) for k in ("coefficients.npz","residual_blocks.npz")}}
    write_json(output/"manifest.json",result)
    return result


class RidgeForecaster:
    def __init__(self, root):
        root=Path(root);self.manifest=read_json(root/"manifest.json")
        with np.load(root/"coefficients.npz") as f:self.fit={k:f[k] for k in f.files}
        self.active=self.fit.pop("active")
        with np.load(root/"residual_blocks.npz") as f:self.blocks=f["residuals"]

    def forecast(self, data, origin, seed):
        raw=predict_ridge(self.fit,design(data,[origin],self.active))[0].T
        mean=np.zeros((24,30));mean[:,self.active]=np.maximum(raw,0.)
        ids=np.random.default_rng(seed).integers(0,len(self.blocks),size=64)
        raw_samples=mean[None]+self.blocks[ids]
        return mean,np.maximum(raw_samples,0.),{"residual_block_ids":ids.tolist(),
            "mean_negative_fraction":float((raw<0).mean()),
            "sample_negative_fraction_active":float((raw_samples[:,:,self.active]<0).mean())}
