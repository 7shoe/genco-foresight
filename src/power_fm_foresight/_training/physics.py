"""Frozen GENCO forecasts and independent physical quantities."""
import numpy as np
import torch
from power_fm_foresight._data.coordinates import record_values
from power_fm_foresight._solver.checks import admittance_vectors

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



def target_values(rows,network):
    values = [record_values(network, r) for r in rows]
    return physics({key:np.array([v[key] for v in values]) for key in values[0]},network)
