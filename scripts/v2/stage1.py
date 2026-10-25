"""Bounded production ON/OFF chains; every continuation retains the frozen protocol."""
from copy import deepcopy
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from common import ROOT, load_config, read_json, sha256, write_json
from scripts.v2.compare_runs import compare_runs, in_epoch_envelope, noninferior_epoch
from scripts.v2.verify_stage import V2_ROOT, copy_cache, freeze_preparation, invoke, verify_frozen


def prepare(protocol_path, root):
    prerequisite=read_json(ROOT/"reports/v2/stage0.json")
    if not prerequisite["passed"]:
        raise ValueError("Stage 0 has not passed")
    protocol=read_json(protocol_path)
    if protocol["comparison"].get("numerical_acceptance") in ("epoch_envelope","epoch_noninferiority"):
        diagnosis=protocol["prerequisite"]["gradient_diagnosis"]
        if sha256(ROOT/diagnosis["path"])!=diagnosis["sha256"]:
            raise ValueError("Gradient diagnosis gate changed")
        result=read_json(ROOT/diagnosis["path"])
        if not result["passed"] or not result["all_30_group_cosines_at_least_0_99"]:
            raise ValueError("Gradient diagnosis does not authorize epoch-level acceptance")
        for row in protocol["historical_epoch_evidence"].values():
            for item in row.values():
                if sha256(item["path"])!=item["sha256"]:
                    raise ValueError("Historical epoch evidence changed")
    subprocess.run(["git","merge-base","--is-ancestor",protocol["prerequisite"]["main_merge"],"main"],check=True,cwd=ROOT)
    if (root/"prepared.json").exists():raise ValueError("Run identity already prepared")
    root.mkdir(parents=True,exist_ok=True)
    write_json(root/"protocol.json",protocol)
    from omegaconf import OmegaConf
    overrides=OmegaConf.to_container(OmegaConf.load(ROOT/"configs/v2/stage1.yaml"),resolve=False)
    overrides["paths"]={"training_run":str(root/"shared")}
    cfg=load_config(ROOT/"configs/v4.yaml",overrides=overrides)
    cfg["v2"]["protocol_sha256"]=sha256(root/"protocol.json")
    write_json(root/"configuration.json",cfg)
    result=freeze_preparation(root,protocol_path,protocol)
    if "historical_epoch_evidence" in protocol:
        prepared=read_json(root/"prepared.json")
        for row in protocol["historical_epoch_evidence"].values():
            for item in row.values():prepared["protected_sha256"][item["path"]]=item["sha256"]
        write_json(root/"prepared.json",prepared)
    return result


def gate(root,passed,**details):
    record={"schema_version":1,"stage":1,"passed":passed,"protocol_sha256":sha256(root/"protocol.json"),
            "evidence_root":str(root),**details}
    write_json(root/"gate.json",record)
    return record


def submit(root):
    verify_frozen(root)
    protocol=read_json(root/"protocol.json")
    if (root/"gate.json").exists():raise ValueError("Stage 1 gate already closed")
    phase="case118" if (root/"case30_gate.json").exists() else "case30"
    if phase=="case118" and not read_json(root/"case30_gate.json")["passed"]:
        raise ValueError("Case30 acceptance failed")
    with (V2_ROOT/"compute.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        q=subprocess.run(["qstat","-u",os.environ["USER"]],capture_output=True,text=True,timeout=30)
        if q.returncode or any(s.strip() and s.lstrip()[0].isdigit() for s in q.stdout.splitlines()):
            raise RuntimeError("Cannot submit while a user job is present or queue inspection fails")
        ledger=read_json(root/"submission.json") if (root/"submission.json").exists() else {"jobs":[]}
        for old in ledger["jobs"]:
            if not old.get("job_id"):raise RuntimeError("Unresolved submission; inspect ledger")
            boundary=root/"jobs"/f"{old['job_id'].split('.')[0]}.json"
            if not boundary.exists():
                gate(root,False,error="Missing worker boundary; full allocation charged",charged_seconds=3600)
                raise RuntimeError("Missing job boundary")
        count=sum(j["phase"]==phase for j in ledger["jobs"])
        if count>=protocol["job"][f"{phase}_max_allocations"]:
            gate(root,False,error=f"{phase} allocation cap exhausted")
            raise RuntimeError("Allocation cap exhausted")
        number=len(ledger["jobs"])+1
        if number>protocol["job"]["max_allocations"]:raise RuntimeError("Cumulative allocation cap")
        (root/"logs").mkdir(exist_ok=True)
        record={"phase":phase,"number":number,"submitted_epoch":time.time(),"reserved_seconds":3600}
        ledger["jobs"].append(record)
        write_json(root/"submission.json",ledger)
        result=subprocess.run(["qsub","-q",protocol["job"]["queue"],"-A",protocol["job"]["account"],
            "-N",f"genco-v2-s1-{number}","-o",str(root/f"logs/job_{number}.pbs.log"),
            "-v",f"GENCO_V2_ROOT={root},GENCO_V2_STAGE=1,GENCO_V2_PHASE={phase}",
            str(ROOT/"scripts/v2/verify.pbs")],cwd=ROOT,capture_output=True,text=True,timeout=45)
        record.update(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
        if result.returncode==0:record["job_id"]=result.stdout.strip()
        write_json(root/"submission.json",ledger)
        if result.returncode:raise RuntimeError(result.stderr)
        return record


def worker(root,name):
    verify_frozen(root)
    if not os.environ.get("PBS_JOBID"):raise RuntimeError("Compute allocation required")
    if name.startswith("case118"):
        from scripts.v2.prepare_case118 import worker as case_worker
        return case_worker(root,name)
    import torch
    if os.environ.get("ZE_FLAT_DEVICE_HIERARCHY")!="FLAT" or torch.xpu.device_count()!=1:
        raise RuntimeError("Expected one externally isolated flat tile")
    torch.xpu.set_device(0)
    cfg=read_json(root/"configuration.json")
    if name=="mechanism":
        from scripts.resume_test import trial
        path=root/"mechanism_configuration.json"
        cfg["v2"]["run_identity"]=f"{root.name}/mechanism"
        write_json(path,cfg)
        os.environ["GENCO_CONFIG"]=str(path)
        result=trial(cfg,"xpu",200,root/"mechanism")
        if not result["passed"]:raise RuntimeError("Deterministic mechanism test failed")
        return
    protocol=read_json(root/"protocol.json")
    if name not in protocol["runs"]:raise ValueError("Unknown worker")
    tile=str(protocol["runs"].index(name))
    if os.environ.get("ZE_AFFINITY_MASK")!=tile:raise RuntimeError("Tile convention mismatch")
    cfg["stage_b"]["deterministic_algorithms"]=name=="on_reference"
    cfg["v2"]["run_identity"]=f"{root.name}/{name}"
    stop=4090 if name=="off_b" and not (root/name/"latest.pt").exists() else None
    from training.loop import run
    write_json(root/name/"affinity.json",{"flat_tile":int(tile),"visible_devices":1,
        "device":str(torch.xpu.get_device_properties(0)),"node":os.uname().nodename,
        "job_id":os.environ["PBS_JOBID"],"torch_version":torch.__version__})
    run(cfg,"xpu",root/name,stop_after_steps=stop)


def command(root,name):
    return [sys.executable,"-u","-m","scripts.v2.verify_stage","worker","--stage","1","--root",str(root),"--name",name]


def launch(root,name,tile,deadline,attempt):
    began=time.time()
    env=dict(os.environ,ZE_FLAT_DEVICE_HIERARCHY="FLAT",ZE_AFFINITY_MASK=str(tile),PYTHONHASHSEED="42",
        GENCO_JOB_START_EPOCH=str(began),PBS_WALLTIME=str(max(301,int(deadline-began+240))))
    log=root/"logs"/f"{os.environ['PBS_JOBID'].split('.')[0]}_{name}_{attempt}.log"
    stream=log.open("w")
    process=subprocess.Popen(command(root,name),cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    return {"process":process,"stream":stream,"name":name,"began":began,"log":str(log)}


def case30(root,deadline,record):
    if not (root/"shared/cache/manifest.json").exists():copy_cache(root)
    if not (root/"mechanism/xpu.json").exists():
        env=dict(os.environ,ZE_AFFINITY_MASK="0",PYTHONHASHSEED="42",PBS_WALLTIME="3600")
        invoke([sys.executable,"-m","pytest","-q","-p","no:cacheprovider",
            "tests/test_v2_harness.py","tests/test_v2_stage1.py","--basetemp",str(root/"tmp/tests")],
            env,root/"logs/tests.log",deadline)
        invoke(command(root,"mechanism"),env,root/"logs/mechanism.log",deadline)
    if not read_json(root/"mechanism/xpu.json")["passed"]:raise RuntimeError("Mechanism gate failed")
    protocol=read_json(root/"protocol.json")
    running={};parked=set();attempts={name:0 for name in protocol["runs"]}
    try:
        while time.time()<deadline:
            comparison=protocol["comparison"]
            if comparison.get("numerical_acceptance")=="epoch_envelope":
                for name in attempts:
                    path=root/name/"epochs.json"
                    if path.exists():
                        for row in read_json(path):
                            if not in_epoch_envelope(row,comparison):
                                write_json(root/"epoch_failure.json",{"run":name,"observation":row,
                                    "bounds":comparison["epoch_envelopes"][str(row["epoch"])],
                                    "rule":"predeclared historical per-epoch min/max envelope"})
                                raise RuntimeError(f"{name} epoch {row['epoch']} outside historical ES envelope")
            if comparison.get("numerical_acceptance")=="epoch_noninferiority":
                for name in comparison["gated_runs"]:
                    path=root/name/"epochs.json"
                    if path.exists():
                        for row in read_json(path):
                            if not noninferior_epoch(row,comparison):
                                write_json(root/"epoch_failure.json",{"run":name,"observation":row,
                                    "bound":comparison["epoch_upper_bounds"][str(row["epoch"])],
                                    "rule":"OFF ES <= historical maximum + 0.005"})
                                raise RuntimeError(f"{name} epoch {row['epoch']} fails one-sided non-inferiority")
            for tile,name in enumerate(attempts):
                if name in running or name in parked:continue
                path=root/name/"status.json"
                status=read_json(path) if path.exists() else {}
                if status.get("status")=="finished":parked.add(name);continue
                if time.time()>deadline-120:parked.add(name);continue
                attempts[name]+=1
                running[name]=launch(root,name,tile,deadline,attempts[name])
            if not running:break
            for name,item in list(running.items()):
                code=item["process"].poll()
                if code is None:continue
                item["stream"].close()
                record["workers"].append({"name":name,"returncode":code,"wall_seconds":time.time()-item["began"],"log":item["log"]})
                del running[name]
                if code:raise RuntimeError(f"{name} exited {code}; {item['log']}")
                status=read_json(root/name/"status.json")
                if status["stop_reason"]!="verification_step_limit":parked.add(name)
            time.sleep(1)
        if running:raise TimeoutError("Worker exceeded family walltime guard")
    finally:
        for item in running.values():
            if item["process"].poll() is None:
                os.killpg(item["process"].pid,signal.SIGKILL)
                item["process"].wait()
            item["stream"].close()
            record["workers"].append({"name":item["name"],"returncode":item["process"].returncode,
                "terminated_after_family_failure":True,"wall_seconds":time.time()-item["began"],"log":item["log"]})
    complete=all((root/name/"status.json").exists() and read_json(root/name/"status.json")["status"]=="finished" for name in attempts)
    if complete:return compare_case30(root)
    return {"complete":False,"reason":"clean allocation boundary"}


def compare_case30(root):
    import torch
    from scripts.resume_test import compare
    protocol=read_json(root/"protocol.json")
    comparisons={}
    for left,right in protocol["comparison"]["pairs"]:
        selected=deepcopy(protocol)
        if right=="on_reference":
            selected["comparison"]["manifest_differences"]={p:[False,True] for p in protocol["comparison"]["reference_allowed_differences"]}
        result=compare_runs(root/left,root/right,selected)
        comparisons[f"{left}__{right}"]=result
        write_json(root/f"compare_{left}__{right}.json",result)
    states=[torch.load(root/name/"latest.pt",map_location="cpu",weights_only=True) for name in protocol["runs"]]
    rng_exact=all(compare(states[0]["rng"],s["rng"])[0] for s in states[1:])
    audits=[read_json(path) for name in protocol["runs"] for path in (root/name/"restores").glob("*.json")]
    boundary=any(r["passed"] and r["progress"]["steps"]==4090 and r["progress"]["epoch"]==0
                 and r["progress"]["step_in_epoch"]==4090 for r in (read_json(p) for p in (root/"off_b/restores").glob("*.json")))
    result={"schema_version":1,"complete":True,"passed":all(r["passed"] for r in comparisons.values()) and rng_exact
            and boundary and all(r["passed"] for r in audits),"comparisons":{k:v["passed"] for k,v in comparisons.items()},
            "rng_exact":rng_exact,"boundary_restart_verified":boundary,"restores_exact":all(r["passed"] for r in audits),
            "restore_count":len(audits)}
    write_json(root/"case30_gate.json",result)
    if not result["passed"]:gate(root,False,case30=result,case118="not run after failed acceptance")
    return result


def job(root):
    verify_frozen(root)
    protocol=read_json(root/"protocol.json")
    start=float(os.environ["GENCO_V2_JOB_START"])
    deadline=start+protocol["job"]["family_guard_seconds"]
    phase=os.environ["GENCO_V2_PHASE"]
    record={"schema_version":1,"job_id":os.environ["PBS_JOBID"],"node":os.uname().nodename,
            "phase":phase,"start_epoch":start,"workers":[],"status":"running"}
    try:
        if phase=="case30":
            result=case30(root,deadline,record)
            if protocol.get("adoption_gate")=="case30_only":
                gate(root,result.get("complete",False) and result.get("passed",False),case30=result,
                    case118="Excluded by the user's final Stage 1 acceptance protocol",
                    allocation_cap_exhausted=not result.get("complete",False))
        elif phase=="case118":
            from scripts.v2.prepare_case118 import job as case_job
            result=case_job(root,deadline,record)
            if result.get("complete"):
                gate(root,result["passed"],case30=read_json(root/"case30_gate.json"),case118=result)
        else:raise ValueError("Unknown verification phase")
        record.update(status="complete",result=result)
        verify_frozen(root)
    except BaseException as error:
        record.update(status="failed",error=repr(error))
        gate(root,False,error=repr(error),phase=phase)
    finally:
        record.update(end_epoch=time.time(),allocation_used_seconds=time.time()-start,reserved_seconds=3600)
        write_json(root/"jobs"/f"{os.environ['PBS_JOBID'].split('.')[0]}.json",record)
    print(record,flush=True)
    if record["status"]=="failed" or ((root/"gate.json").exists() and not read_json(root/"gate.json")["passed"]):
        raise SystemExit(1)


def dispatch(args,root):
    if args.action=="prepare":print(prepare(args.protocol,root))
    elif args.action=="submit":print(submit(root))
    elif args.action=="worker":worker(root,args.name)
    else:job(root)
