"""Durable background driver for the isolated v4 audit, setup, chain and evaluation."""
import argparse
from datetime import datetime
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from common import ROOT, config_path, load_config, read_json, write_json
from scripts.training_chain import job_state
from scripts.data_v4 import verify_variant


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-job", help="Existing generation/audit job; omit to submit an audit resume")
    args = parser.parse_args()
    cfg = load_config()
    verify_variant(cfg)
    root = Path(cfg["paths"]["runs_root"])
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "pipeline.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = root / "pipeline.json"
    state = read_json(path) if path.exists() else {"label": cfg["variant"]["label"], "primary_record": "v3",
        "data_job": args.data_job, "created_epoch": time.time(), "stage": "data_audit",
        "jobs": {"data_audit": args.data_job} if args.data_job else {}}
    assert not args.data_job or state["data_job"] == args.data_job
    state.update(controller_pid=os.getpid(), controller_host=os.uname().nodename)
    cutoff = datetime.fromisoformat(cfg["variant"]["ship_cutoff_ct"]).timestamp()

    def save():
        state.update(updated_epoch=time.time(), ship_cutoff_ct=cfg["variant"]["ship_cutoff_ct"],
                     ship_v3_with_v4_in_progress=time.time() >= cutoff and state["stage"] != "complete")
        write_json(path, state)

    def wait_job(label, job):
        state["stage"] = label
        while True:
            s, code = job_state(job)
            state["current_job"] = {"id": job, "state": s, "exit_status": code}
            save()
            if s == "F":
                if code != 0:
                    raise RuntimeError(f"{label} failed, job {job}, exit {code}; inspect preserved job log")
                return
            time.sleep(30)

    def submit(label, module):
        if label in state["jobs"]:
            return state["jobs"][label]
        variables = f"GENCO_STAGE_A_MODULE={module},GENCO_CONFIG={config_path().resolve()}"
        command = ["qsub", "-N", f"genco-v4-{label}", "-o", str(root / f"{label}.log"),
                   "-v", variables, str(ROOT / "scripts/stage_a.pbs")]
        # Concurrent Stage C jobs can temporarily fill the user's queued-job limit.
        while True:
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                job = result.stdout.strip().splitlines()[-1]
                state["jobs"][label] = job
                save()
                return job
            if "limit" not in result.stderr.lower():
                raise RuntimeError(result.stderr)
            state["submission_wait_reason"] = result.stderr.strip()
            save()
            time.sleep(30)

    try:
        job = args.data_job or state.get("data_job") or submit("data_audit", "scripts.data_v4")
        state["data_job"] = job
        wait_job("data_audit", job)
        assert read_json(Path(cfg["paths"]["stage_a_run"]) / "readiness.json")["training_allowed"]
        train = Path(cfg["paths"]["training_run"])
        if not (train / "resume_test/report.json").exists():
            wait_job("setup", submit("setup", "scripts.resume_test"))
        assert read_json(train / "resume_test/report.json")["passed"]
        state["stage"] = "training"
        save()
        with (root / "training_controller.log").open("a") as stream:
            process = subprocess.Popen([sys.executable, "-u", "-m", "scripts.training_chain", "--device", "xpu"],
                                       cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            state["training_controller_pid"] = process.pid
            while process.poll() is None:
                if (train / "status.json").exists():
                    state["training_status"] = read_json(train / "status.json")
                save()
                time.sleep(30)
            if process.returncode:
                raise RuntimeError(f"Training controller exit {process.returncode}")
        assert read_json(train / "chain.json")["status"] in ("finished", "budget_complete")
        state["training_finished_epoch"] = time.time()
        save()
        wait_job("evaluation", submit("evaluation", "scripts.evaluate_variant"))
        assert read_json(root / "stage_b/completion.json")["complete"]
        state["stage"] = "complete"
        state["completed_epoch"] = time.time()
        save()
    except BaseException as error:
        state["failure"] = repr(error)
        state["failed_stage"] = state["stage"]
        state["stage"] = "failed"
        save()
        raise


if __name__ == "__main__":
    main()
