"""Kill/restart the real adapter loop and compare with uninterrupted training."""
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from power_fm_foresight._config import load_config,write_json
from power_fm_foresight._training.inputs import build_cache


def compare(a,b):
    if isinstance(a,torch.Tensor):
        equal=torch.equal(a,b)
        delta=float((a.to(torch.float64)-b.to(torch.float64)).abs().max()) if a.numel() else 0.
        return equal,delta
    if isinstance(a,dict):
        if a.keys()!=b.keys():return False,float('inf')
        pairs=[compare(a[k],b[k]) for k in a]
    elif isinstance(a,(list,tuple)):
        if len(a)!=len(b):return False,float('inf')
        pairs=[compare(x,y) for x,y in zip(a,b)]
    else:return a==b,0.
    return all(x[0] for x in pairs),max((x[1] for x in pairs),default=0.)


def trial(cfg,device,steps,root):
    def invoke(name,crash=None):
        destination=root/device/name
        if (destination/"status.json").exists():raise ValueError("Do not overwrite a completed resume trial")
        destination.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,"-m","scripts.train_adapter","--device",device,"--run-root",str(destination),
                 "--stop-after-steps",str(steps),"--resume-test"]
        if crash:command += ["--crash-at-steps",str(crash)]
        env=dict(os.environ,GENCO_JOB_START_EPOCH=str(time.time()),PYTHONHASHSEED=str(cfg["seed"]))
        with (destination/("interrupted.log" if crash else "run.log")).open("w") as stream:
            result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,env=env)
        if result.returncode != (-9 if crash else 0):
            raise RuntimeError(f"Resume trial failed: {destination}; code {result.returncode}")
        return destination
    uninterrupted=invoke("uninterrupted")
    resumed=invoke("resumed",steps//2)
    killed=torch.load(resumed/"latest.pt",map_location="cpu",weights_only=True)
    assert killed["progress"]["steps"]==steps//2
    invoke("resumed")
    a=torch.load(uninterrupted/"latest.pt",map_location="cpu",weights_only=True)
    b=torch.load(resumed/"latest.pt",map_location="cpu",weights_only=True)
    report={"device":device,"total_steps_each_path":steps,"killed_with_sigkill_after_step":steps//2,
        "resumed_steps":steps-steps//2,"same_step_sequence":a["progress"]["step_sequence_sha256"]==b["progress"]["step_sequence_sha256"],
        "same_initial_weights":a["progress"]["initial_adapter_sha256"]==b["progress"]["initial_adapter_sha256"],
        "same_counters":all(a["progress"][k]==b["progress"][k] for k in ("epoch","step_in_epoch","steps","best_selection","bad_epochs"))}
    for key in ("model","optimizer","scheduler","rng"):
        exact,delta=compare(a[key],b[key]);report[key]={"bit_exact":exact,"max_absolute_difference":delta}
    report["passed"]=report["same_initial_weights"] and report["same_step_sequence"] and report["same_counters"] and all(report[k]["bit_exact"] for k in ("model","optimizer","scheduler","rng"))
    write_json(root/f"{device}.json",report)
    print(report,flush=True)
    return report


def main():
    cfg=load_config();root=Path(cfg["paths"]["training_run"])/"resume_test"
    root.mkdir(parents=True,exist_ok=True)
    build_cache(cfg,"xpu")
    # CPU fallback uses the same 192-graph path; the full 200-step test is on selected XPU.
    cpu=trial(cfg,"cpu",4,root)
    xpu=trial(cfg,"xpu",200,root)
    report={"passed":cpu["passed"] and xpu["passed"],"cpu":cpu,"xpu":xpu}
    write_json(root/"report.json",report)
    if not report["passed"]:raise RuntimeError("Restart comparison failed; training chain not released")


if __name__=="__main__":main()
