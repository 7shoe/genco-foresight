"""Finish the authorized v3 calibration, two-arm hybrid and static figures."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from common import ROOT, config_path, load_config, read_json, write_json
from scripts.training_chain import job_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-job", required=True)
    args = parser.parse_args()
    cfg = load_config()
    assert cfg["dataset"]["id"] == "temporal_case30_v3"
    out = Path(cfg["paths"]["runs_root"]) / "stage_c"
    lock = (out / "pipeline.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = out / "pipeline.json"
    state = read_json(path) if path.exists() else {"calibration_job": args.calibration_job, "created_epoch": time.time()}
    state.update(controller_pid=os.getpid())
    def save():
        state["updated_epoch"] = time.time()
        write_json(path, state)
    def wait(job, stage):
        state["stage"] = stage
        while True:
            s, c = job_state(job)
            state["job"] = {"id": job, "state": s, "exit_status": c}
            save()
            if s == "F":
                if c != 0: raise RuntimeError(f"{stage} job {job} failed: {c}")
                return
            time.sleep(30)
    try:
        wait(args.calibration_job, "calibration_forecasts")
        if not (out / "report.json").exists():
            state["stage"] = "calibration_analysis"
            save()
            subprocess.run([sys.executable, "-u", "-m", "scripts.calibrate_stage_c"], cwd=ROOT, check=True)
        if "hybrid_job" not in state:
            state["stage"] = "hybrid_submission"
            while True:
                r = subprocess.run(["qsub", "-N", "genco-hybrid-v3", "-o", str(out / "hybrid.pbs.log"),
                    "-v", f"GENCO_STAGE_A_MODULE=scripts.hybrid_stage_c,GENCO_CONFIG={config_path().resolve()}",
                    str(ROOT / "scripts/stage_a.pbs")], cwd=ROOT, capture_output=True, text=True, timeout=60)
                if not r.returncode:
                    state["hybrid_job"] = r.stdout.strip().splitlines()[-1]
                    save()
                    break
                if "limit" not in r.stderr.lower(): raise RuntimeError(r.stderr)
                state["submission_wait_reason"] = r.stderr.strip()
                save()
                time.sleep(30)
        wait(state["hybrid_job"], "hybrid")
        state["stage"] = "figures"
        save()
        subprocess.run([sys.executable, "-u", "-m", "scripts.figures_stage_d"], cwd=ROOT, check=True)
        state["stage"] = "complete"
        state["completed_epoch"] = time.time()
        save()
    except BaseException as error:
        state["failed_stage"] = state.get("stage")
        state["stage"] = "failed"
        state["error"] = repr(error)
        save()
        raise


if __name__ == "__main__":
    main()
