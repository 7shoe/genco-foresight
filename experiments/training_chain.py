"""One-node debug job chain; durable log, dependency prefetch, six-hour total cap."""
import argparse
import fcntl
import os
from pathlib import Path
import re
import subprocess
import time

from power_fm_foresight._config import ROOT,config_path,load_config,read_json,write_json


def job_state(job):
    for attempt in range(3):
        try:
            result=subprocess.run(["qstat","-xf",job],capture_output=True,text=True,timeout=30)
            if result.returncode:raise RuntimeError(result.stderr.strip())
            break
        except (subprocess.TimeoutExpired,RuntimeError):
            if attempt==2:raise
            time.sleep(5)
    state=re.search(r"job_state\s*=\s*(\w+)",result.stdout)
    code=re.search(r"Exit_status\s*=\s*(-?\d+)",result.stdout)
    return state[1] if state else None,int(code[1]) if code else None


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--device",choices=["cpu","xpu"],default="xpu")
    p.add_argument("--once",action="store_true",help="Submit/check once; rerun this command to continue manually")
    args=p.parse_args();cfg=load_config();root=Path(cfg["paths"]["training_run"])
    root.mkdir(parents=True,exist_ok=True)
    lock=(root/"chain.lock").open("w");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if cfg["stage_b"]["pbs_queue"] not in ("debug","debug-scaling"):raise ValueError("Only debug queues authorized")
    if not read_json(Path(cfg["paths"]["stage_a_run"])/"readiness.json")["training_allowed"]:raise RuntimeError("Stage A not ready")
    if not read_json(root/"resume_test/report.json")["passed"]:raise RuntimeError("Resume verification not passed")
    path=root/"chain.json"
    chain=read_json(path) if path.exists() else {"mode":"A","device":args.device,"status":"running",
        "jobs":[],"dependency_chaining":cfg.get("variant",{}).get("dependency_prefetch",True),"budget_seconds":cfg["stage_b"]["training_budget_seconds"],
        "manual_command":f"GENCO_CONFIG={config_path().resolve()} python -m scripts.training_chain --device {args.device}","created_epoch":time.time()}
    if chain["device"]!=args.device:raise ValueError("Do not change device inside an existing chain")
    (root/"logs").mkdir(exist_ok=True)
    def save():
        done=[j for j in chain["jobs"] if "result" in j]
        chain["cumulative_training_wall_seconds"]=sum(j["result"]["wall_used_seconds"] for j in done)
        chain["total_queue_wait_seconds"]=sum(j.get("queue_wait_seconds",0.) for j in done)
        chain["total_dependency_wait_seconds"]=sum(j.get("dependency_wait_seconds",0.) for j in done)
        chain["calendar_elapsed_seconds"]=time.time()-chain["created_epoch"]
        write_json(path,chain)
        rows=["# Adapter training chain","",f"Status: {chain['status']}. Queue wait excludes time waiting for the preceding dependency.","",
              "| Job | State | Node | Steps | Epochs reached | Wall s | Queue wait s |","|---|---|---|---:|---:|---:|---:|"]
        for j in chain["jobs"]:
            r=j.get("result",{});rows.append(f"| {j['job_id']} | {j.get('state','Q')} | {r.get('node','')} | {r.get('steps_done','')} | {r.get('epochs_reached','')} | {r.get('wall_used_seconds',0):.1f} | {j.get('queue_wait_seconds',0):.1f} |")
        rows += ["",f"Cumulative training job wall: {chain['cumulative_training_wall_seconds']:.1f} s; queue wait: {chain['total_queue_wait_seconds']:.1f} s; calendar elapsed: {chain['calendar_elapsed_seconds']:.1f} s.","",
                 "Each worker stops before 55 minutes; checkpoint every epoch and every 900 seconds. The cumulative training limit is 21,600 seconds. Cache creation and resume tests are separate preparation costs."]
        (root/"chain.md").write_text("\n".join(rows)+"\n")
    def submit(dependency=None):
        number=len(chain["jobs"])+1
        cmd=["qsub","-q",cfg["stage_b"]["pbs_queue"],"-N",f"genco-A-{number}",
             "-o",str(root/f"logs/job_{number:02d}.log"),"-e",str(root/f"logs/job_{number:02d}.err"),
             "-v",f"GENCO_TRAIN_DEVICE={args.device},GENCO_CONFIG={config_path().resolve()},GENCO_TRAIN_ROOT={root}"]
        if dependency:cmd += ["-W",f"depend=afterany:{dependency}"]
        cmd += [str(ROOT/"scripts/train_adapter.pbs")]
        result=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=60)
        if result.returncode:
            if dependency:
                chain["dependency_chaining"]=False;chain["dependency_rejection"]=result.stderr.strip();save();return
            if "limit" in result.stderr.lower():
                chain["submission_wait_reason"]=result.stderr.strip();save();return
            raise RuntimeError(result.stderr.strip())
        jid=result.stdout.strip().splitlines()[-1]
        chain["jobs"].append({"job_id":jid,"submitted_epoch":time.time(),"dependency":dependency,"state":"Q"})
        save();print("submitted",jid,"dependency",dependency,flush=True)
    try:
        while chain["status"]=="running":
            pending=[]
            for j in chain["jobs"]:
                if "result" in j or j.get("state")=="cancelled":continue
                state,code=job_state(j["job_id"]);j["state"]=state
                if state=="F":
                    result_path=root/"jobs"/f"{j['job_id'].split('.')[0]}.json"
                    if code!=0 or not result_path.exists():
                        chain.update(status="failed",failure=f"Job {j['job_id']} exit {code}; inspect its log");break
                    j["result"]=read_json(result_path)
                    dependency_end=j["submitted_epoch"]
                    if j["dependency"]:
                        prior=next(x for x in chain["jobs"] if x["job_id"]==j["dependency"])
                        dependency_end=max(dependency_end,prior.get("result",{}).get("job_end_epoch",dependency_end))
                    j["dependency_wait_seconds"]=dependency_end-j["submitted_epoch"]
                    j["queue_wait_seconds"]=max(0.,j["result"]["job_start_epoch"]-dependency_end)
                    if j["result"]["status"] in ("finished","skipped_after_stop"):chain["status"]="finished"
                else:pending.append(j)
            save()
            if chain["cumulative_training_wall_seconds"]>=chain["budget_seconds"]-15:chain["status"]="budget_complete"
            if chain["status"]!="running":
                for j in pending:
                    result=subprocess.run(["qdel",j["job_id"]],capture_output=True,text=True,timeout=30)
                    j["state"]="cancelled";j["cancellation_returncode"]=result.returncode
                save();break
            if not pending:submit()
            elif len(pending)==1 and pending[0]["state"]=="R" and chain["dependency_chaining"]:
                # The v4 priority amendment reserves a successor throughout training.
                # A surplus successor is cancelled by the stop/budget branch above.
                if chain.get("reserve_queue_slot",False) or chain["budget_seconds"]-chain["cumulative_training_wall_seconds"]>2*cfg["stage_b"]["job_training_guard_seconds"]:
                    submit(pending[0]["job_id"])
            save()
            if args.once:break
            time.sleep(30)
    except BaseException as error:
        chain["controller_error"]=repr(error);save();raise


if __name__=="__main__":main()
