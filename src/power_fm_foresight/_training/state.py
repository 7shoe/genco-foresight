"""Portable, atomic training state, including every RNG and restart counter."""
import hashlib
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

from power_fm_foresight._config import write_json


def cpu_tree(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value, list): return [cpu_tree(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu_tree(v) for v in value)
    return value


def tensor_digest(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def tree_digest(value):
    """Hash typed state values independently of tensor device or pickle storage IDs."""
    digest=hashlib.sha256()
    def visit(item):
        if isinstance(item,torch.Tensor):
            digest.update(f"tensor:{item.dtype}:{tuple(item.shape)}:".encode())
            digest.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item,dict):
            digest.update(b"dict{")
            for key in sorted(item,key=repr):visit(key);visit(item[key])
            digest.update(b"}")
        elif isinstance(item,(tuple,list)):
            digest.update(type(item).__name__.encode()+b"[")
            for entry in item:visit(entry)
            digest.update(b"]")
        else:digest.update(f"{type(item).__name__}:{item!r};".encode())
    visit(value)
    return digest.hexdigest()


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def rng_state():
    name,keys,pos,gauss,cached=np.random.get_state()
    return {"python":random.getstate(),"numpy":{"name":name,"keys":keys.tolist(),"pos":pos,"gauss":gauss,"cached":cached},
        "torch":torch.get_rng_state(),
        "xpu":torch.xpu.get_rng_state_all() if torch.xpu.is_available() else [],
        "cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    n=state["numpy"];np.random.set_state((n["name"],np.array(n["keys"],dtype=np.uint32),n["pos"],n["gauss"],n["cached"]))
    torch.set_rng_state(state["torch"])
    if state["xpu"] and torch.xpu.is_available(): torch.xpu.set_rng_state_all(state["xpu"])
    if state["cuda"] and torch.cuda.is_available(): torch.cuda.set_rng_state_all(state["cuda"])


def atomic_torch_save(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix=f".{path.name}.")
    try:
        with os.fdopen(fd,"wb") as stream:
            torch.save(cpu_tree(value),stream);stream.flush();os.fsync(stream.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def train_state(model):
    # The same machinery stores backbone parameters for F if they are unfrozen.
    include_base=any(p.requires_grad for p in model.backbone.parameters())
    return {k:v for k,v in model.state_dict().items() if include_base or not k.startswith("backbone.")}


def checkpoint(path, model, optimizer, scheduler, progress, manifest):
    atomic_torch_save(path,{"model":train_state(model),"optimizer":optimizer.state_dict(),
        "scheduler":scheduler.state_dict(),"rng":rng_state(),"progress":progress,"manifest":manifest})


def restore(path, model, optimizer, scheduler, manifest, audit_path=None):
    value=torch.load(path,map_location="cpu",weights_only=True)
    if value["manifest"] != manifest: raise ValueError("Checkpoint run manifest mismatch")
    missing,unexpected=model.load_state_dict(value["model"],strict=False)
    if unexpected or any(not k.startswith("backbone.") for k in missing):
        raise ValueError("Incomplete adapter checkpoint")
    if missing and any(p.requires_grad for p in model.backbone.parameters()):
        raise ValueError("Unfrozen backbone missing from checkpoint")
    optimizer.load_state_dict(value["optimizer"]);scheduler.load_state_dict(value["scheduler"])
    restore_rng(value["rng"])
    if audit_path is not None:
        restored={"model":train_state(model),"optimizer":optimizer.state_dict(),
            "scheduler":scheduler.state_dict(),"rng":rng_state()}
        checks={key:{"saved_sha256":tree_digest(value[key]),"restored_sha256":tree_digest(state)}
            for key,state in restored.items()}
        passed=all(row["saved_sha256"]==row["restored_sha256"] for row in checks.values())
        write_json(audit_path,{"schema_version":1,"passed":passed,"state":checks,
            "manifest_exact":True,"manifest_sha256":tree_digest(manifest),
            "progress_sha256":tree_digest(value["progress"]),"progress":value["progress"]})
        if not passed:raise ValueError("Checkpoint state did not restore exactly")
    return value["progress"]


def epoch_order(seed, epoch, count):
    return torch.randperm(count,generator=torch.Generator().manual_seed(seed+epoch)).tolist()


def next_progress():
    return {"epoch":0,"step_in_epoch":0,"steps":0,"best_selection":None,"bad_epochs":0,
        "best_epoch":None,"epochs":[],"cumulative_training_wall_seconds":0.,"finished":False,
        "stop_reason":None,"step_sequence_sha256":hashlib.sha256(b"").hexdigest()}
