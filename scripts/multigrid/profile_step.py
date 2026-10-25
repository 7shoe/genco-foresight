"""Profile the unchanged training step on synthetic inputs, one process per tile."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import shlex
import signal
import statistics
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PYTHON = "/lus/flare/projects/FRAME-IDP/siebenschuh/envs/reason_venv_2026.1.0/bin/python"
OUTPUT = Path("/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/multigrid/profile")
CASES = ("case14_ieee", "case30_ieee", "case57_ieee", "case118_ieee", "case500_goc", "case2000_goc")
PHASES = ("forward", "loss", "backward", "optimizer", "total")


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def summaries(rows):
    result = {}
    for key in PHASES:
        values = sorted(r[key] for r in rows)
        position = .9 * (len(values) - 1)
        low = math.floor(position)
        high = math.ceil(position)
        result[key] = {"median": statistics.median(values),
                       "p90": values[low] + (values[high] - values[low]) * (position - low)}
    return result


def fixture(args, record):
    import numpy as np
    import torch
    from data.coordinates import fit_scales
    from data.grids import grid_config
    from data.loader.network import graph_for_demand, load_network
    from models.genco import Genco
    from models.temporal import TemporalAdapter

    cfg = grid_config(args.case)
    cfg["stage_b"]["training_samples"] = args.K
    cfg["stage_b"]["future_graph_batch_size"] = args.K * 24
    grid = load_network(config=cfg)
    base = Genco(Path(cfg["paths"]["assets_root"]) / "genco-opfdata", cfg["checkpoint"], args.device, grid)
    rng = np.random.default_rng(42)
    rows = []
    for _ in range(8):
        state = dict(vm=1 + .005 * rng.normal(size=grid.n_bus),
                     va_rad=.001 * rng.normal(size=grid.n_bus),
                     pg_mw=(grid.gen[:, 8] + grid.gen[:, 9]) / 2 + rng.normal(size=grid.n_gen),
                     qg_mvar=(grid.gen[:, 3] + grid.gen[:, 4]) / 2 + rng.normal(size=grid.n_gen))
        rows.append(dict(pd_mw=grid.bus[:, 2] * rng.uniform(.9, 1, grid.n_bus),
                         qd_mvar=grid.bus[:, 3] * rng.uniform(.9, 1, grid.n_bus), target_state=state))
    scales = fit_scales(grid, rows)
    model = TemporalAdapter(base, grid, scales).to(args.device).train()
    history = {}
    for kind, count in (("bus", grid.n_bus), ("gen", grid.n_online_gen)):
        width = getattr(model, f"{kind}_gru").input_size - 32
        mean = getattr(model, f"{kind}_mean").cpu()
        scale = getattr(model, f"{kind}_scale").cpu()
        history[f"embedding_{kind}"] = torch.randn(72, count, width) * .1
        history[f"observed_{kind}"] = mean + torch.randn(72, *mean.shape) * scale
        history[f"available_{kind}"] = torch.ones_like(history[f"observed_{kind}"], dtype=torch.bool)
    calendar = torch.randn(24, 6) * .1
    means = torch.tensor([scales["features"][key]["mean"][i]
                          for key in model.score_keys
                          for i, keep in enumerate(scales["features"][key]["joint_score_mask"]) if keep])
    target = (means + torch.randn(24, len(means)) * model.coordinate_scale.cpu()).flatten()

    class SyntheticCache:
        def window(self, identity, device):
            return ({key: value.to(device) for key, value in history.items()},
                    calendar.to(device), target.to(device))

    graph = graph_for_demand(grid, grid.bus[:, 2] * .9, grid.bus[:, 3] * .9)
    template = base.batch([graph] * (args.K * 24))
    assert template["bus"].num_nodes == args.K * 24 * grid.n_bus
    assert template["gen"].num_nodes == args.K * 24 * grid.n_online_gen
    assert all(torch.isfinite(x).all() for x in template.x_dict.values())
    opt = cfg["stage_b"]["optimizer"]
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=opt["lr"], weight_decay=opt["weight_decay"], foreach=False)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.)
    record.update(dims={key: value for key, value in grid.describe().items() if not isinstance(value, list)},
                  graphs_per_step=args.K * 24, trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  frozen_parameters=sum(p.numel() for p in model.backbone.parameters()),
                  input_shapes={k: list(v.shape) for k, v in history.items()}, data_format_passed=True,
                  checkpoint=cfg["checkpoint"], network_sha256=grid.case_sha256,
                  loss=cfg["stage_b"]["loss"], optimizer=opt,
                  synthetic_inputs="Random histories, targets and synthetic scales; no solved or cached datasets")
    return cfg, model, optimizer, scheduler, template, SyntheticCache()


class PhaseTimer:
    def __init__(self, synchronize):
        self.synchronize = synchronize
        self.row = {}

    def clock(self):
        self.synchronize()
        return time.perf_counter()

    def forward(self, original):
        def call(*args, **kwargs):
            start = self.clock()
            result = original(*args, **kwargs)
            self.forward_end = self.clock()
            self.row["forward"] = self.forward_end - start
            return result
        return call

    def backward(self, original):
        def call(tensor, *args, **kwargs):
            start = self.clock()
            self.row["loss"] = start - self.forward_end
            result = original(tensor, *args, **kwargs)
            self.backward_end = self.clock()
            self.row["backward"] = self.backward_end - start
            return result
        return call

    def scheduler(self, original):
        def call(*args, **kwargs):
            result = original(*args, **kwargs)
            self.row["optimizer"] = self.clock() - self.backward_end
            return result
        return call


def worker(args):
    started = time.time()
    path = args.output / f"rank_{args.rank}.json"
    record = dict(case=args.case, K=args.K, device=args.device, deterministic=args.deterministic,
                  concurrent=args.concurrent, rank=args.rank, tile_mask=os.environ.get("ZE_AFFINITY_MASK"),
                  node=platform.node(), job_id=os.environ.get("PBS_JOBID"), status="running",
                  validation_only=args.validation_only, timings=[], steps_requested=args.steps,
                  source_sha256={p: digest(ROOT / p) for p in (
                      "training/loop.py", "models/temporal.py", "models/genco.py", "scoring.py",
                      "scripts/multigrid/profile_step.py")})
    save(path, record)
    torch = None
    try:
        if args.K == 1 and not args.validation_only:
            record.update(status="UNSUPPORTED_K1", reason="Unchanged fair energy score requires at least two samples")
            return
        import torch
        from training.loop import training_step
        from training.state import seed_all, tensor_digest
        torch.set_num_threads(16 if args.device == "cpu" else 1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(args.deterministic == "on")
        seed_all(42)
        if args.device == "xpu":
            if not torch.xpu.is_available() or torch.xpu.device_count() != 1:
                raise RuntimeError(f"Expected one isolated XPU tile; saw {torch.xpu.device_count()}")
            torch.xpu.set_device(0)
            props = torch.xpu.get_device_properties(0)
            record.update(device_name=props.name, device_total_bytes=props.total_memory,
                          visible_devices=torch.xpu.device_count())
            torch.xpu.reset_peak_memory_stats()
        record.update(torch_version=torch.__version__, dtype="float32", cpu_threads=torch.get_num_threads(),
                      cpu_affinity_count=len(os.sched_getaffinity(0)))
        cfg, model, optimizer, scheduler, template, cache = fixture(args, record)
        synchronize = torch.xpu.synchronize if args.device == "xpu" else lambda: None
        base_before = tensor_digest(model.backbone.state_dict())
        record["setup_seconds"] = time.time() - started
        save(path, record)

        if args.validation_only:
            model.eval()
            history, calendar, _ = cache.window(0, args.device)
            with torch.no_grad():
                for _ in range(5):
                    result = model(history, calendar, template, model.noise(args.K, 24))
                    del result
                synchronize()
                record["validation_seconds"] = 0.
                for origin in range(36):
                    validation_start = time.perf_counter()
                    history, calendar, _ = cache.window(origin, args.device)
                    for _ in range(64 // args.K):
                        result = model(history, calendar, template, model.noise(args.K, 24))
                        if origin == 0 and not torch.isfinite(result["joint"]).all():
                            raise RuntimeError("Nonfinite validation predictions")
                        del result
                    synchronize()
                    record["validation_origins_completed"] = origin + 1
                    record["validation_seconds"] += time.perf_counter() - validation_start
                    save(path, record)
                    if time.time() >= args.deadline:
                        raise TimeoutError("Job time guard during validation")
            record.update(status="OK", validation_graphs=36 * 64 * 24)
        else:
            timer = PhaseTimer(synchronize)
            with patch.object(model, "forward", timer.forward(model.forward)), \
                 patch.object(torch.Tensor, "backward", timer.backward(torch.Tensor.backward)), \
                 patch.object(scheduler, "step", timer.scheduler(scheduler.step)):
                warmup_start = time.perf_counter()
                for _ in range(5):
                    training_step(model, optimizer, scheduler, template, cache, 0, cfg, args.device)
                record["warmup_seconds"] = time.perf_counter() - warmup_start
                save(args.output / f"ready_{args.rank}.json", {"ready": True})
                while not (args.output / "go.json").exists():
                    if time.time() >= args.deadline:
                        raise TimeoutError("Timed out waiting for other tiles")
                    time.sleep(.05)
                if args.device == "xpu":
                    torch.xpu.reset_peak_memory_stats()
                record["timed_start_epoch"] = time.time()
                for step in range(args.steps):
                    timer.row = {}
                    start = timer.clock()
                    metrics = training_step(model, optimizer, scheduler, template, cache, step, cfg, args.device)
                    timer.row["total"] = timer.clock() - start
                    record["timings"].append(dict(timer.row))
                    record["last_metrics"] = metrics
                    save(path, record)
                    if time.time() >= args.deadline:
                        raise TimeoutError("Job time guard during timed steps")
                record["timed_end_epoch"] = time.time()
            record.update(status="OK", phase_seconds=summaries(record["timings"]),
                          warmup_steps=5, timed_steps=len(record["timings"]))
        record["backbone_unchanged"] = tensor_digest(model.backbone.state_dict()) == base_before
        if not record["backbone_unchanged"]:
            raise RuntimeError("Frozen backbone changed")
    except Exception as error:
        oom = "out of memory" in str(error).lower() or type(error).__name__ == "OutOfMemoryError"
        record.update(status="OOM" if oom else "TIMEOUT" if isinstance(error, TimeoutError) else "ERROR",
                      reason=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    finally:
        record["elapsed_seconds"] = time.time() - started
        record["host_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        if torch is not None and args.device == "xpu" and torch.xpu.is_initialized():
            record["peak_device_bytes"] = torch.xpu.max_memory_allocated()
            record["peak_reserved_bytes"] = torch.xpu.max_memory_reserved()
        save(path, record)
        print(json.dumps({k: record.get(k) for k in ("case", "K", "rank", "status", "reason", "phase_seconds")}), flush=True)


def group(args, label, case, k, device="xpu", deterministic="off", concurrent=1, validation=False):
    out = args.output / label
    out.mkdir(parents=True, exist_ok=False)
    processes = []
    deadline = min(args.deadline, time.time() + (900 if validation else 420))
    for rank in range(concurrent):
        env = dict(os.environ, ZE_FLAT_DEVICE_HIERARCHY="FLAT", ZE_AFFINITY_MASK=str(rank),
                   OMP_NUM_THREADS="16" if device == "cpu" else "1", MKL_NUM_THREADS="16" if device == "cpu" else "1",
                   OPENBLAS_NUM_THREADS="16" if device == "cpu" else "1", PYTHONHASHSEED="42")
        command = [sys.executable, str(Path(__file__).resolve()), case, "--K", str(k), "--device", device,
                   "--deterministic", deterministic, "--steps", str(args.steps), "--concurrent", str(concurrent),
                   "--output", str(out), "--worker", "--rank", str(rank), "--deadline", str(deadline)]
        if validation:
            command.append("--validation-only")
        with (out / f"rank_{rank}.log").open("w") as log:
            processes.append(subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                               stderr=subprocess.STDOUT, start_new_session=True))
    released = validation
    while any(p.poll() is None for p in processes):
        if not released and all((out / f"ready_{rank}.json").exists() or p.poll() is not None
                                for rank, p in enumerate(processes)):
            save(out / "go.json", {"epoch": time.time()})
            released = True
        if time.time() > deadline + 5:
            for p in processes:
                if p.poll() is None:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
            break
        time.sleep(.2)
    records = []
    for rank, p in enumerate(processes):
        path = out / f"rank_{rank}.json"
        r = read(path) if path.exists() else dict(case=case, K=k, device=device, rank=rank, concurrent=concurrent)
        if r.get("status") in (None, "running"):
            r.update(status="TIMEOUT" if time.time() > deadline else "ERROR", reason=f"Worker exited {p.returncode}; see rank log")
            save(path, r)
        r["result_path"] = str(path)
        records.append(r)
    print(label, [r["status"] for r in records], flush=True)
    intervals = [(r["timed_start_epoch"], r["timed_end_epoch"]) for r in records if "timed_end_epoch" in r]
    overlap = max(0., min(b for a, b in intervals) - max(a for a, b in intervals)) if intervals else None
    return dict(label=label, case=case, K=k, device=device, deterministic=deterministic,
                concurrent=concurrent, validation=validation, measured_overlap_seconds=overlap, records=records)


def estimate(row, validation):
    train = 4090 * row["phase_seconds"]["total"]["median"]
    return train, 19 * (train + (validation or 0)) / 3600


def report(args, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = summary.get("groups", [])
    validation = {g["case"]: g["records"][0].get("validation_seconds") for g in groups
                  if g["validation"] and g["records"][0]["status"] == "OK"}
    solo = [g["records"][0] for g in groups if g["label"].startswith("solo_")]
    best = {r["case"]: r for r in solo if r["status"] == "OK"}
    def number(value): return "—" if value is None else f"{value:.3f}"
    def phase(row, key):
        p = row.get("phase_seconds", {}).get(key, {})
        return f"{number(p.get('median'))} / {number(p.get('p90'))}"
    lines = ["# Multigrid step and tile profile", "",
             f"Job `{summary.get('job_id', 'not submitted')}`; status **{summary.get('status', 'pending')}**; "
             f"[raw results]({args.output}/summary.json). Float32; unchanged `training.loop.training_step`; synthetic inputs/scales, no datasets.",
             "Times are median / p90 seconds with XPU synchronization at phase boundaries. Forward includes the history encoder, sampling, GENCO and physics; optimizer includes gradient clipping and scheduling; total also includes transfers, RNG and the existing digest. Memory is peak allocated device GB; reserved/host memory and tile details are in JSON.",
             "Epoch estimates use 4,090 steps; 19-epoch estimates add measured 36×64 forward-only validation once per epoch when available († otherwise training only). Setup, checkpointing, validation scoring and queue wait are excluded. K=1 cannot train with the unchanged fair energy score; smaller K is not run after a larger K fits.", "",
             "## Table A", "",
             "| Grid | K | Device / deterministic | Status | Bus / branch / gen | Graphs/step | Fwd s | Loss s | Bwd s | Opt s | Step s | Peak GB | Train s/epoch | Validation s | Hours/19 epochs | Graphs/s |",
             "|---|---:|---|---|---|---:|---|---|---|---|---|---:|---:|---:|---:|---:|"]
    for g in groups:
        if g["validation"] or g["label"].startswith("concurrent_"): continue
        for r in g["records"]:
            d = r.get("dims", {})
            dims = "/".join(str(d.get(key, "—")) for key in ("n_bus", "n_branch", "n_gen"))
            train, hours, speed = None, None, None
            if r["status"] == "OK":
                train, hours = estimate(r, validation.get(r["case"]))
                speed = 24 * r["K"] / r["phase_seconds"]["total"]["median"]
            suffix = "†" if r["case"] not in validation and hours is not None else ""
            lines.append(f"| {r['case']} | {r['K']} | {g['device']} / {g['deterministic']} | {r['status']} | {dims} | {24*r['K']} | "
                         + " | ".join(phase(r, key) for key in PHASES)
                         + f" | {number(r.get('peak_device_bytes', 0)/1e9) if r.get('peak_device_bytes') is not None else '—'} | {number(train)} | {number(validation.get(r['case']))} | {number(hours)}{suffix} | {number(speed)} |")
    lines.extend(["", "## Table B", "",
                  "| Grid | Fits one tile / K | Single-tile hours | Tiles: 1 h / 3 h / 6 h, ideal | Recommended mode |",
                  "|---|---|---:|---|---|"])
    under_hour, chained, unviable = [], [], []
    for case in CASES:
        r = best.get(case)
        if r is None:
            attempted = [x for x in solo if x["case"] == case]
            state = "not measured" if not attempted else "not established: " + attempted[-1]["status"]
            lines.append(f"| {case} | {state} | — | — | — |")
            if attempted: unviable.append(case)
            continue
        _, hours = estimate(r, validation.get(case))
        (under_hour if hours <= 1 else chained).append(case)
        mode = "many models per node" if hours <= 1 and r["K"] == 8 else "one tile per model" if hours <= 6 else "sample-parallel across tiles" if r["K"] < 8 else "DDP across tiles/nodes"
        tiles = " / ".join(str(math.ceil(hours / target)) for target in (1, 3, 6))
        lines.append(f"| {case} | Y / {r['K']} | {hours:.3f}{'†' if case not in validation else ''} | {tiles} | {mode} |")
    lines.extend(["", "The requested ‘upper bound under ideal scaling’ applies to speedup: ceil(hours/target) is an optimistic **lower bound on required tiles**, not measured scaling or a memory guarantee. Multi-tile recommendations are prospective; no model sharding or DDP was tested.",
                  "", "## Table C", "",
                  "| Concurrent chains | Process / tile | Step median / p90 s | Median as % of solo | Status |",
                  "|---:|---:|---|---:|---|"])
    cotenants = [g for g in groups if g["label"].startswith("concurrent_")]
    baseline = next((g["records"][0]["phase_seconds"]["total"]["median"] for g in cotenants
                     if g["concurrent"] == 1 and g["records"][0]["status"] == "OK"), None)
    for g in cotenants:
        reference = baseline or best.get(g["case"], {}).get("phase_seconds", {}).get("total", {}).get("median")
        for r in g["records"]:
            relative = 100 * r["phase_seconds"]["total"]["median"] / reference if reference and r["status"] == "OK" else None
            lines.append(f"| {g['concurrent']} | {r['rank']} | {phase(r, 'total')} | {number(relative)} | {r['status']} |")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for k in (8, 4, 2, 1):
        rows = sorted((r for r in best.values() if r["K"] == k), key=lambda r: r["dims"]["n_bus"])
        if rows:
            ax.plot([r["dims"]["n_bus"] for r in rows], [r["phase_seconds"]["total"]["median"] for r in rows], "o-", label=f"K={k}")
    ax.set(xscale="log", yscale="log", xlabel="Buses", ylabel="Median training step (s)", title="Largest fitting K; unmeasured combinations omitted")
    ax.grid(True, which="both", alpha=.25)
    if best: ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / "step_time.png", dpi=160)
    plt.close(fig)
    lines.extend(["", f"![Step time by grid and K]({args.output}/step_time.png)", "",
                  "Single-tile 1 h estimate: " + (", ".join(under_hour) or "none established") + ".  ",
                  "Requires chained 1 h jobs under this recipe: " + (", ".join(chained) or "none established") + ".  ",
                  "Single-tile viability unestablished: " + (", ".join(unviable) or "none among completed configurations") + "; synthetic timing does not establish convergence."])
    target = ROOT / "reports/multigrid/profile.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n")


def suite(args):
    summary = dict(job_id=os.environ.get("PBS_JOBID"), node=platform.node(), started_epoch=time.time(),
                   status="running", groups=[], script_sha256=digest(__file__))
    def run(label, case, k, **kwargs):
        if time.time() > args.deadline - 30:
            return None
        value = group(args, label, case, k, **kwargs)
        summary["groups"].append(value)
        save(args.output / "summary.json", summary)
        report(args, summary)
        return value
    chosen = {}
    for case in CASES:
        for k in (8, 4, 2, 1):
            g = run(f"solo_{case}_k{k}", case, k)
            if g is None: break
            status = g["records"][0]["status"]
            if status == "OK":
                chosen[case] = k
                break
            if status != "OOM": break
    for case in ("case30_ieee", "case118_ieee"):
        run(f"deterministic_{case}", case, 8, deterministic="on")
    run("cpu_case30_ieee", "case30_ieee", 8, device="cpu")
    for n in (1, 4, 8, 12):
        run(f"concurrent_{n}", "case30_ieee", 8, concurrent=n)
    for case, k in chosen.items():
        run(f"validation_{case}", case, k, validation=True)
    status = "complete_with_failures" if any(r["status"] != "OK" for g in summary["groups"] for r in g["records"]) else "complete"
    summary.update(status=status if time.time() < args.deadline - 30 else "walltime_guard_partial",
                   finished_epoch=time.time())
    save(args.output / "summary.json", summary)
    report(args, summary)


def submit(args):
    args.output.mkdir(parents=True, exist_ok=False)
    q = shlex.quote
    script = f"""#!/bin/bash
#PBS -N genco-profile
#PBS -A EpiCalib
#PBS -q debug
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -m n
#PBS -o {args.output}/pbs.log
set -euo pipefail
profile_job_start=$(date +%s)
cd {q(str(ROOT))}
set +u
if ! type module >/dev/null 2>&1; then source /etc/profile; fi
module --quiet reset
module load oneapi/release/2026.1.0 mpich/prd/5.0.0.aurora_test.87e2045 gcc/14.3.0
module load libfabric/2.3.1 cray-pals/1.8.0 cray-libpals/1.8.0 intel_gpu_umd_aicoe/2026.06.19
module load hdf5/1.14.6 pti-gpu/0.17.0 miniforge3/25.11.0-1 frameworks/2026.1.0
source {q(str(Path(PYTHON).parent / 'activate'))}
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=42
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export TMPDIR={q(str(args.output / 'tmp'))}
mkdir -p "$TMPDIR"
mpiexec --pmi=pmix --envall -n 1 --ppn 1 --cpu-bind none \\
 {q(PYTHON)} -u {q(str(Path(__file__).resolve()))} all --steps {args.steps} --output {q(str(args.output))} --deadline "$((profile_job_start+3300))"
"""
    (args.output / "job.pbs").write_text(script)
    job = subprocess.run(["qsub"], input=script, text=True, capture_output=True, check=True).stdout.strip()
    summary = dict(job_id=job, status="queued", submitted_epoch=time.time(), groups=[],
                   git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                   script_sha256=digest(__file__))
    save(args.output / "summary.json", summary)
    report(args, summary)
    print(json.dumps({"job_id": job, "output": str(args.output)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", help="14/30/57/118/500/2000, full case name, or all")
    parser.add_argument("--K", type=int, choices=(8, 4, 2, 1), default=8)
    parser.add_argument("--device", choices=("xpu", "cpu"), default="xpu")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--deterministic", choices=("on", "off"), default="off")
    parser.add_argument("--concurrent", type=int, choices=range(1, 13), default=1)
    parser.add_argument("--output", type=Path, default=OUTPUT / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--rank", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--validation-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--deadline", type=float, default=time.time() + 3300, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.case = next((c for c in CASES if c.removeprefix("case").split("_")[0] == args.case), args.case)
    if args.case not in (*CASES, "all") or args.steps < 1:
        parser.error("Choose a supported case and a positive step count")
    if args.output.resolve().is_relative_to(Path("/home")):
        parser.error("Profile artifacts must stay outside /home")
    if args.submit:
        if args.case != "all": parser.error("--submit schedules the complete one-node matrix; use case all")
        submit(args)
    elif args.report:
        report(args, read(args.output / "summary.json"))
    elif args.worker:
        worker(args)
    else:
        if not os.environ.get("PBS_JOBID"):
            parser.error("Run profiling inside a compute allocation, or use all --submit")
        args.output.mkdir(parents=True, exist_ok=True)
        if args.case == "all":
            suite(args)
        else:
            groups = [group(args, f"solo_{args.case}_k{args.K}", args.case, args.K,
                            device=args.device, deterministic=args.deterministic)]
            if args.concurrent > 1:
                groups.append(group(args, f"concurrent_{args.concurrent}", args.case, args.K,
                                    device=args.device, deterministic=args.deterministic, concurrent=args.concurrent))
            summary = dict(job_id=os.environ.get("PBS_JOBID"), status="complete", groups=groups)
            save(args.output / "summary.json", summary)
            report(args, summary)
            print(args.output / "summary.json")


if __name__ == "__main__":
    main()
