"""Paired cold versus A forecast-mean primal starts on realized next-day inputs."""
import argparse
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch

from power_fm_foresight._config import ROOT, load_config, read_json, sha256, write_json
from power_fm_foresight._data.network import graph_for_demand
from power_fm_foresight._training.forecast import components, adapter_samples
from experiments.diagnose_stage_b import npz
from power_fm_foresight._solver.adapter import IpoptSolver
from power_fm_foresight._solver.checks import check_solution, physical_quantities
from power_fm_foresight._training.state import tensor_digest

STATE = ("vm", "va_rad", "pg_mw", "qg_mvar")


def freeze(cfg):
    b = Path(cfg["paths"]["runs_root"]) / "stage_b"
    out = b.parent / "stage_c/hybrid"
    out.mkdir(parents=True, exist_ok=True)
    p = read_json(b / "protocol.json")
    protocol = {"arms": ["cold", "A_forecast_mean"], "origins": p["origins"],
        "budget": cfg["solver"]["budget"], "test_protocol_sha256": sha256(b / "protocol.json"),
        "best_checkpoint_sha256": p["checkpoint_sha256"], "seed": cfg["seed"],
        "initial_state": "Arithmetic mean of the stored 64 A samples for Vm/Va/Pg/Qg, no projection; no future solved states",
        "inputs": "Identical realized Pd/Qd for both arms; fixed topology, independent hourly AC-OPF",
        "order": "Alternate cold/warm first by global origin index + lead index parity",
        "advance_opf_solves": 0, "solver_warmup": "Two excluded JIT requests per worker on first train record, one per arm; setup timed separately",
        "cost": "Report online request latency and iterations; add timed replay of the 64-sample A forecast and 72 historical embedding computations to warm total. Shared process/model/cache setup reported separately.",
        "history_timing": "Time a batch of 72 past graphs; record deviations from the immutable 192-batch historical cache. Timing outputs do not replace cached forecast inputs. Forecast replay itself must match stored A within 1e-5 absolute/relative.",
        "implementation_revision": "Initial timing preflight stopped before paired solves because 72- vs 192-graph embeddings were not identical. Preserve that attempt; no forecasts, model, solver settings or outcome criteria change.",
        "source_sha256": {p: sha256(ROOT / p) for p in ("scripts/hybrid_stage_c.py", "scripts/forecast_partition.py",
            "solver/adapter.py", "solver/checks.py", "solver/julia/worker.jl", "models/temporal.py")}}
    if (out / "protocol.json").exists():
        assert read_json(out / "protocol.json") == protocol
    else:
        write_json(out / "protocol.json", protocol)
    return b, out, protocol


def worker(cfg, shard, workers, device):
    started = time.perf_counter()
    b, out, p = freeze(cfg)
    net, base, model, _, templates = components(cfg, device)
    digest = tensor_digest(base.model.state_dict())
    data = Path(cfg["paths"]["temporal_root"])
    episodes, cached = {}, {}
    solver = IpoptSolver(cfg, out / f"logs/solver_{shard:02d}.log")
    try:
        train = read_json(data / "episodes/train_00.json")["records"][0]
        warmup = []
        for initial in (None, train["target_state"]):
            result = solver.solve(train["pd_mw"], train["qd_mvar"], initial_state=initial)
            assert check_solution(net, train["pd_mw"], train["qd_mvar"], result)["valid"]
            warmup.append(result["request_seconds"])
        setup = time.perf_counter() - started
        for index, o in enumerate(p["origins"]):
            if index % workers != shard:
                continue
            folder = out / "origins" / o["id"]
            done = folder / "done.json"
            if done.exists():
                prior = read_json(done)
                assert prior["protocol_sha256"] == sha256(out / "protocol.json")
                assert all(sha256(folder / name) == h for name, h in prior["artifacts"].items())
                continue
            name, origin = o["episode_id"], o["origin"]
            if name not in episodes:
                episodes[name] = read_json(data / f"episodes/{name}.json")["records"]
                cached[name] = torch.load(b / f"cache/{name}.pt", weights_only=True, mmap=True)["data"]
            records = episodes[name]
            predpath = b / "origins" / o["id"] / "A.npz"
            assert sha256(predpath) == read_json(predpath.parent / "done.json")["artifacts"]["A.npz"]
            values = npz(predpath)
            means = {k: values[k].mean(0) for k in STATE}
            timing = folder / "preparation.json"
            if not timing.exists():
                if device.startswith("xpu"):
                    torch.xpu.synchronize()
                hist_start = time.perf_counter()
                graphs = [graph_for_demand(net, r["pd_mw"], r["qd_mvar"]) for r in records[origin - 72:origin]]
                with torch.no_grad():
                    _, embeddings = base.model(base.batch(graphs), return_embeddings=True)
                embedding_differences = {}
                for kind, n in (("bus", net.n_bus), ("gen", net.n_online_gen)):
                    timed = embeddings[kind].reshape(72, n, embeddings[kind].shape[-1]).cpu()
                    stored = cached[name][f"embedding_{kind}"][origin - 72:origin]
                    assert torch.isfinite(timed).all()
                    embedding_differences[kind] = {"max_abs": float((timed - stored).abs().max()),
                        "rms": float((timed - stored).square().mean().sqrt()),
                        "note": "Timing only; original cached embeddings are retained for the actual forecast replay"}
                history_seconds = time.perf_counter() - hist_start
                replay, inference_seconds = adapter_samples(base, model, net, cached[name], origin, templates,
                                                             p["seed"] + 10000 + index, device)
                max_abs = {}
                for k in STATE + ("pd_mw", "qd_mvar"):
                    np.testing.assert_allclose(replay[k], values[k], atol=1e-5, rtol=1e-5)
                    max_abs[k] = float(np.max(np.abs(replay[k] - values[k])))
                write_json(timing, {"history_embedding_seconds": history_seconds, "A_inference_seconds": inference_seconds,
                    "preparation_seconds": history_seconds + inference_seconds, "forecast_replay_max_abs": max_abs,
                    "timed_history_vs_original_cache": embedding_differences,
                    "device": device, "prediction_sha256": sha256(predpath), "no_advance_opf_solves": True})
            for h in range(24):
                path = folder / f"hour_{h:02d}.json"
                if path.exists():
                    continue
                record = records[origin + h]
                initial = {k: means[k][h].tolist() for k in STATE}
                raw = physical_quantities(net, record["pd_mw"], record["qd_mvar"], initial)
                order = ("cold", "A_forecast_mean") if (index + h) % 2 == 0 else ("A_forecast_mean", "cold")
                row = {"origin": o, "lead": h + 1, "order": list(order), "initial_state": initial,
                    "initial_max_abs_p_mw": float(abs(raw["residual_p_mw"]).max()),
                    "initial_max_abs_q_mvar": float(abs(raw["residual_q_mvar"]).max()), "arms": {}}
                for arm in order:
                    result = solver.solve(record["pd_mw"], record["qd_mvar"], initial_state=initial if arm != "cold" else None)
                    check = check_solution(net, record["pd_mw"], record["qd_mvar"], result)
                    row["arms"][arm] = {"solver": result, "check": check,
                        "objective_delta_from_label": result["objective"] - record["solver"]["objective"] if check["valid"] else None}
                write_json(path, row)
            assert tensor_digest(base.model.state_dict()) == digest
            write_json(done, {"protocol_sha256": sha256(out / "protocol.json"), "backbone_unchanged": True,
                "artifacts": {path.name: sha256(path) for path in folder.iterdir() if path.is_file() and path.name != "done.json"}})
            print({"origin": o["id"], "hybrid_complete": True}, flush=True)
        write_json(out / f"worker_{shard:02d}.json", {"node": platform.node(), "device": device, "setup_seconds": setup,
            "solver_startup_seconds": solver.startup_seconds, "solver_warmup_request_seconds": warmup,
            "wall_seconds": time.perf_counter() - started})
    finally:
        solver.close()


def summary(x):
    x = np.asarray(x, float)
    return {"count": len(x), "sum": float(x.sum()), "mean": float(x.mean()),
            "median": float(np.median(x)), "p90": float(np.quantile(x, .9)), "max": float(x.max())}


def merge(cfg):
    b, out, p = freeze(cfg)
    rows, preparations = [], []
    for o in p["origins"]:
        folder = out / "origins" / o["id"]
        done = read_json(folder / "done.json")
        assert all(sha256(folder / name) == h for name, h in done["artifacts"].items())
        rows.extend(read_json(folder / f"hour_{h:02d}.json") for h in range(24))
        preparations.append(read_json(folder / "preparation.json"))
    report = {"paired_hours": len(rows), "origins": len(preparations), "protocol_sha256": sha256(out / "protocol.json"),
        "preparation_seconds_per_day": summary([r["preparation_seconds"] for r in preparations]), "arms": {},
        "notes": ["Identical realized demand, fixed 5-second/2000-iteration budget, independently checked residuals and limits.",
            "Forecast mean may be infeasible. Only primal starting values are supplied; Ipopt may internally move starts to bounds.",
            "Latency is measured request wall time including Python/Julia IPC and model construction. No advance OPF solves.",
            "Total work sums serial request seconds plus forecast preparation, not parallel elapsed time. Shared setup is reported separately.",
            "Recomputed 72-hour history embeddings are included per day (conservative relative to a rolling cache). Online latency excludes this advance forecast preparation."]}
    report["history_timing_qualification"] = {"batch_size_timed": 72, "original_cache_batch_size": 192,
        "timing_outputs_used_for_forecast": False,
        "max_embedding_difference": {kind: max(r["timed_history_vs_original_cache"][kind]["max_abs"] for r in preparations) for kind in ("bus", "gen")},
        "forecast_replay_max_abs": {k: max(r["forecast_replay_max_abs"][k] for r in preparations) for k in STATE + ("pd_mw", "qd_mvar")}}
    for arm in p["arms"]:
        a = [r["arms"][arm] for r in rows]
        success = [r for r in a if r["check"]["valid"]]
        report["arms"][arm] = {"attempts": len(a), "valid": len(success), "success_rate": len(success) / len(a),
            "iterations_all_returned": summary([r["solver"]["iterations"] for r in a if r["solver"].get("iterations") is not None]),
            "request_seconds": summary([r["solver"]["request_seconds"] for r in a]),
            "solver_wall_seconds": summary([r["solver"]["wall_seconds"] for r in a if r["solver"].get("wall_seconds") is not None]),
            "max_abs_objective_delta_from_label": max(abs(r["objective_delta_from_label"]) for r in success) if success else None}
        report["arms"][arm]["total_work_seconds_excluding_shared_setup"] = report["arms"][arm]["request_seconds"]["sum"] + (report["preparation_seconds_per_day"]["sum"] if arm != "cold" else 0)
    good = [r for r in rows if all(r["arms"][a]["check"]["valid"] for a in p["arms"])]
    report["paired_valid"] = len(good)
    report["warm_minus_cold_iterations"] = summary([r["arms"]["A_forecast_mean"]["solver"]["iterations"] - r["arms"]["cold"]["solver"]["iterations"] for r in good])
    report["warm_minus_cold_request_seconds"] = summary([r["arms"]["A_forecast_mean"]["solver"]["request_seconds"] - r["arms"]["cold"]["solver"]["request_seconds"] for r in good])
    report["workers"] = [read_json(path) for path in sorted(out.glob("worker_*.json"))]
    write_json(out / "report.json", report)
    lines = ["# Two-arm hybrid: cold vs A forecast-mean primal start", "", f"{len(rows)} paired realized hours, 72 forecast days, no advance OPF solves.", "",
        "| Arm | Valid / attempted | Mean iterations | Mean online request (s) | Forecast preparation (s total) | Total work excluding shared setup (s) |", "|---|---:|---:|---:|---:|---:|"]
    for arm, r in report["arms"].items():
        prep = report["preparation_seconds_per_day"]["sum"] if arm != "cold" else 0
        lines.append(f"| {arm} | {r['valid']}/{r['attempts']} | {r['iterations_all_returned']['mean']:.3f} | {r['request_seconds']['mean']:.6f} | {prep:.3f} | {r['total_work_seconds_excluding_shared_setup']:.3f} |")
    lines += ["", *["- " + note for note in report["notes"]]]
    (out / "report.md").write_text("\n".join(lines) + "\n")


def run(cfg, workers=12):
    _, out, _ = freeze(cfg)
    started = time.time()
    processes, streams = [], []
    try:
        for shard in range(workers):
            stream = (out / f"worker_{shard:02d}.log").open("a")
            streams.append(stream)
            processes.append(subprocess.Popen([sys.executable, "-u", "-m", "scripts.hybrid_stage_c", "--worker",
                "--shard", str(shard), "--workers", str(workers), "--device", f"xpu:{shard % torch.xpu.device_count()}"],
                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT))
        codes = [proc.wait() for proc in processes]
        if any(codes):
            raise RuntimeError(f"Hybrid workers failed: {codes}")
    finally:
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
        for stream in streams:
            stream.close()
    merge(cfg)
    write_json(out / "job.json", {"complete": True, "wall_seconds": time.time() - started, "job_id": os.environ.get("PBS_JOBID")})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--device", default="xpu:0")
    a = p.parse_args()
    worker(load_config(), a.shard, a.workers, a.device) if a.worker else run(load_config(), a.workers)


if __name__ == "__main__":
    main()
