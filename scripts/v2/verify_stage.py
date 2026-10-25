"""Prepare, submit and evaluate protocol-bound V2 verification allocations."""
import argparse
from copy import deepcopy
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from common import ROOT, load_config, read_json, sha256, write_json
from scripts.v2.compare_runs import compare_runs


V2_ROOT = Path("/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2")


def checked_root(path):
    path = Path(path).resolve()
    if not path.is_relative_to(V2_ROOT) or path == V2_ROOT:
        raise ValueError("Verification outputs must be below GENCO/runs/v2")
    return path


def prepare(protocol_path, root):
    root = checked_root(root)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "prepared.json").exists():
        raise ValueError("Run identity already prepared")
    protocol = read_json(protocol_path)
    if protocol["stage"] != 0 or protocol["schema_version"] != 1:
        raise ValueError("Only Stage 0 is implemented; dependent stages require its gate")
    write_json(root / "protocol.json", protocol)
    from omegaconf import OmegaConf
    overrides = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/v2/stage0.yaml"), resolve=False)
    overrides["paths"] = {"training_run": str(root / "shared")}
    cfg = load_config(ROOT / "configs/v4.yaml", overrides=overrides)
    cfg["v2"]["protocol_sha256"] = sha256(root / "protocol.json")
    write_json(root / "configuration.json", cfg)
    files = set(subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines())
    for directory in ("scripts/v2", "training", "configs/v2", "tests"):
        files.update(str(p.relative_to(ROOT)) for p in (ROOT / directory).glob("*.*") if p.suffix in (".py", ".pbs", ".yaml"))
    sources = {name: sha256(ROOT / name) for name in sorted(files)
               if Path(name).suffix in (".py", ".pbs", ".yaml", ".json")}
    sources[str(protocol_path.resolve().relative_to(ROOT))] = sha256(protocol_path)
    protected = {}
    for version in ("v3", "v4"):
        for name in ("manifest.json", "status.json", "epochs.json"):
            path = V2_ROOT.parent / f"adapter_{version}" / name
            if path.exists():
                protected[str(path)] = sha256(path)
    write_json(root / "prepared.json", {"schema_version": 1, "stage": 0,
        "created_epoch": time.time(), "protocol_sha256": sha256(root / "protocol.json"),
        "configuration_sha256": sha256(root / "configuration.json"), "source_sha256": sources,
        "protected_sha256": protected, "tag_v0.1.0": subprocess.check_output(
            ["git", "rev-parse", "refs/tags/v0.1.0"], cwd=ROOT, text=True).strip()})
    return {"root": str(root), "protocol_sha256": sha256(root / "protocol.json"), "sources": len(sources)}


def verify_frozen(root):
    prepared = read_json(root / "prepared.json")
    for name in ("protocol", "configuration"):
        if sha256(root / f"{name}.json") != prepared[f"{name}_sha256"]:
            raise ValueError(f"Prepared {name} changed")
    for name, expected in prepared["source_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Prepared source changed: {name}")
    for name, expected in prepared["protected_sha256"].items():
        if sha256(name) != expected:
            raise ValueError(f"Historical artifact changed: {name}")
    tag = subprocess.check_output(["git", "rev-parse", "refs/tags/v0.1.0"], cwd=ROOT, text=True).strip()
    if tag != prepared["tag_v0.1.0"]:
        raise ValueError("Release tag changed")


def submit(root):
    verify_frozen(root)
    with (V2_ROOT / "compute.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "submission.json").exists():
            raise ValueError("Stage 0 allocation cap already reserved; inspect submission.json")
        queue = subprocess.run(["qstat", "-u", os.environ["USER"]], capture_output=True, text=True, timeout=30)
        if queue.returncode:
            raise RuntimeError("Cannot inspect user queue: " + queue.stderr)
        if any(line.strip() and line.lstrip()[0].isdigit() for line in queue.stdout.splitlines()):
            raise RuntimeError("Another user job is present; no concurrent family or cancellation")
        (root / "logs").mkdir(exist_ok=True)
        protocol = read_json(root / "protocol.json")
        record = {"reserved_epoch": time.time(), "allocation_cap_seconds": protocol["job"]["walltime_seconds"],
                  "protocol_sha256": sha256(root / "protocol.json"), "state": "submitting"}
        write_json(root / "submission.json", record)
        result = subprocess.run(["qsub", "-q", protocol["job"]["queue"], "-A", protocol["job"]["account"],
            "-o", str(root / "logs/pbs.log"), "-v", f"GENCO_V2_ROOT={root}",
            str(ROOT / "scripts/v2/verify.pbs")], cwd=ROOT, capture_output=True, text=True, timeout=45)
        record.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
                      state="submitted" if result.returncode == 0 else "rejected")
        if result.returncode == 0:
            record["job_id"] = result.stdout.strip()
        write_json(root / "submission.json", record)
        if result.returncode:
            raise RuntimeError(result.stderr)
        return record


def invoke(command, env, log, deadline):
    with log.open("w") as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=max(1., deadline - time.time()))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise TimeoutError(f"Verification family guard: {log}")
        if code:
            raise RuntimeError(f"Worker exited {code}; see {log}")


def copy_cache(root):
    original = V2_ROOT.parent / "adapter_v4/cache/manifest.json"
    cache = deepcopy(read_json(original))
    destination = root / "shared/cache"
    destination.mkdir(parents=True)
    for name, row in cache["files"].items():
        target = destination / f"{name}.pt"
        shutil.copyfile(row["path"], target)
        if sha256(target) != row["sha256"]:
            raise ValueError(f"Cache copy mismatch: {name}")
        row["path"] = str(target)
    cache["copied_from"] = {"path": str(original), "sha256": sha256(original)}
    write_json(destination / "manifest.json", cache)


def worker(root, name):
    verify_frozen(root)
    if not os.environ.get("PBS_JOBID"):
        raise RuntimeError("Training verification requires a compute allocation")
    import torch
    if os.environ.get("ZE_FLAT_DEVICE_HIERARCHY") != "FLAT" or os.environ.get("ZE_AFFINITY_MASK") != "0":
        raise RuntimeError("Expected isolated flat tile 0")
    if not torch.xpu.is_available() or torch.xpu.device_count() != 1:
        raise RuntimeError("Expected exactly one visible XPU tile")
    torch.xpu.set_device(0)
    protocol = read_json(root / "protocol.json")
    if name not in protocol["runs"]:
        raise ValueError("Unregistered run")
    cfg = read_json(root / "configuration.json")
    cfg["v2"]["run_identity"] = f"{root.name}/{name}"
    from training.loop import run
    write_json(root / name / "affinity.json", {"flat_tile": 0, "visible_devices": 1,
        "device": str(torch.xpu.get_device_properties(0)), "node": os.uname().nodename,
        "job_id": os.environ["PBS_JOBID"], "torch_version": torch.__version__})
    run(cfg, "xpu", run_root=root / name, stop_after_steps=protocol["comparison"]["steps"])


def job(root):
    if not os.environ.get("PBS_JOBID"):
        raise RuntimeError("Verification job requires PBS")
    verify_frozen(root)
    protocol = read_json(root / "protocol.json")
    start = float(os.environ["GENCO_V2_JOB_START"])
    deadline = start + protocol["job"]["worker_family_guard_seconds"]
    record = {"schema_version": 1, "stage": protocol["stage"], "passed": False,
              "protocol_sha256": sha256(root / "protocol.json"), "job_id": os.environ["PBS_JOBID"],
              "node": os.uname().nodename, "start_epoch": start, "workers": []}
    try:
        copy_cache(root)
        env = dict(os.environ, ZE_FLAT_DEVICE_HIERARCHY="FLAT", ZE_AFFINITY_MASK="0", PYTHONHASHSEED="42")
        invoke([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_v2_harness.py",
                "--basetemp", str(root / "tmp/pytest")], env, root / "logs/tests.log", deadline)
        for name in protocol["runs"]:
            began = time.time()
            env.update(GENCO_JOB_START_EPOCH=str(began), PBS_WALLTIME=str(max(301, int(deadline - began + 300))))
            invoke([sys.executable, "-u", "-m", "scripts.v2.verify_stage", "worker", "--root", str(root),
                    "--name", name], env, root / f"logs/{name}.log", deadline)
            record["workers"].append({"name": name, "wall_seconds": time.time() - began})
            write_json(root / "job.json", record)
        left, right = [root / name for name in protocol["runs"]]
        comparison = compare_runs(left, right, protocol)
        write_json(root / "comparison.json", comparison)
        from scripts.resume_test import compare
        import torch
        a, b = [torch.load(p / "latest.pt", map_location="cpu", weights_only=True) for p in (left, right)]
        exact = {key: compare(a[key], b[key])[0] for key in ("model", "optimizer", "scheduler", "rng")}
        exact["progress"] = ({k: v for k, v in a["progress"].items() if k != "cumulative_training_wall_seconds"}
                            == {k: v for k, v in b["progress"].items() if k != "cumulative_training_wall_seconds"})
        write_json(root / "state_comparison.json", exact)
        genuine = {str(p): sha256(p) for p in right.iterdir() if p.is_file()}
        fixture = root / "altered_fixture"
        shutil.copytree(right, fixture)
        import json
        rows = [json.loads(line) for line in (fixture / "metrics.jsonl").read_text().splitlines()]
        control = protocol["negative_control"]
        next(row for row in rows if row["steps"] == control["step"])[control["field"]] += control["delta"]
        (fixture / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        negative = compare_runs(left, fixture, protocol)
        write_json(root / "negative_control.json", negative)
        protected = all(sha256(path) == value for path, value in genuine.items())
        verify_frozen(root)
        expected_failure = f"step {control['step']}: {control['field']}"
        record.update(passed=comparison["passed"] and all(exact.values()) and protected
                      and negative["failures"] == [expected_failure],
                      comparison_passed=comparison["passed"], state_exact=exact,
                      negative_control_detected=negative["failures"] == [expected_failure],
                      genuine_artifacts_unchanged=protected)
    except BaseException as error:
        record["error"] = repr(error)
    finally:
        record.update(end_epoch=time.time(), allocation_used_seconds=time.time() - start,
                      allocation_reserved_seconds=protocol["job"]["walltime_seconds"],
                      cumulative_allocation_cap_seconds=protocol["job"]["cumulative_allocation_cap_seconds"])
        record["evidence_sha256"] = {str(p): sha256(p) for p in root.rglob("*") if p.is_file()
            and p.name not in ("job.json", "gate.json", "pbs.log") and "tmp" not in p.parts and "cache" not in p.parts}
        write_json(root / "job.json", record)
        write_json(root / "gate.json", record)
    print(record, flush=True)
    if not record["passed"]:
        raise SystemExit(1)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "submit", "job", "worker"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--name")
    args = parser.parse_args()
    root = checked_root(args.root)
    if args.action == "prepare":
        print(prepare(args.protocol, root))
    elif args.action == "submit":
        print(submit(root))
    elif args.action == "worker":
        worker(root, args.name)
    else:
        job(root)


if __name__ == "__main__":
    main()
