"""Frozen-checkpoint forecasts for calibration or a new explicit dataset variant.

Reuse the Stage B model, ridge, physics and scorer without modifying its source
or original artifacts. Each forecast origin is atomic and independently resumable.
"""
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
from power_fm_foresight._data.dataset import load_temporal_manifest
from power_fm_foresight._data.network import graph_for_demand, load_network
from power_fm_foresight.genco import Genco
from power_fm_foresight.temporal import TemporalAdapter
from power_fm_foresight.scoring import UNITS, evaluate, merge_origin_scores, variogram_pairs
from power_fm_foresight._training.physics import physics, frozen, target_values
from experiments.evaluate_stage_b import audit_samples
from power_fm_foresight._training.forecaster import RidgeForecaster, atomic_npz, episode_arrays, prepare as prepare_ridge
from power_fm_foresight._training.state import atomic_torch_save, tensor_digest


MODES = ("C", "P", "P0", "S", "A", "D")


def output_root(cfg, partition):
    root = Path(cfg["paths"]["runs_root"])
    out = root / ("stage_c/calibration" if partition == "risk_calibration" else "stage_b")
    assert str(out.resolve()).startswith("/lus/flare/")
    if partition == "test" and cfg["dataset"]["id"] == "temporal_case30_v3":
        raise ValueError("Primary v3 test predictions already exist; reuse them without replacement")
    out.mkdir(parents=True, exist_ok=True)
    return out


from power_fm_foresight._training.forecast import components


from power_fm_foresight._training.forecast import adapter_samples


from power_fm_foresight._training.forecast import cache_partition


def prepare(cfg, partition, device):
    out = output_root(cfg, partition)
    net, base, model, scales, _ = components(cfg, device)
    data = Path(cfg["paths"]["temporal_root"])
    manifest = read_json(data / "manifest.json")
    forecaster = Path(cfg["paths"]["runs_root"]) / "stage_b/forecaster"
    prepare_ridge(cfg, net, forecaster)
    cache_partition(cfg, out, partition, base, net)
    # Reuse v3's original climatology pool; v4 gets a separately hashed train pool.
    pool_path = Path(cfg["paths"]["runs_root"]) / "stage_b/controls/climatology_pool.npz"
    if not pool_path.exists():
        blocks, identities = [], []
        for e in manifest["episodes"]:
            if e["split"] != "train":
                continue
            values = target_values(read_json(data / f"episodes/{e['episode_id']}.json")["records"], net)
            for start in range(0, 481, 24):
                assert all(np.isfinite(values[k][start:start + 24]).all() for k in UNITS), "C requires resolved train days"
                blocks.append({k: v[start:start + 24] for k, v in values.items()})
                identities.append({"episode_id": e["episode_id"], "start": start, "split": "train"})
        assert len(blocks) == 210
        atomic_npz(pool_path, **{k: np.stack([b[k] for b in blocks]) for k in blocks[0]})
        write_json(pool_path.parent / "pool_manifest.json", {"blocks": identities, "sha256": sha256(pool_path),
                    "dataset_manifest_sha256": sha256(data / "manifest.json")})
    origins = [{"episode_id": e["episode_id"], "origin": o, "id": f"{e['episode_id']}_{o:04d}"}
               for e in manifest["episodes"] if e["split"] == partition for o in range(72, 481, 24)]
    assert len(origins) == 72
    pairs = variogram_pairs(scales, net)
    atomic_npz(out / "variogram_pairs.npz", pairs=pairs)
    train = Path(cfg["paths"]["training_run"])
    protocol = {"dataset_id": cfg["dataset"]["id"], "partition": partition, "origins": origins, "modes": list(MODES),
        "best_epoch": read_json(train / "status.json")["best_epoch"], "checkpoint_sha256": sha256(train / "best.pt"),
        "dataset_manifest_sha256": sha256(data / "manifest.json"), "forecaster_path": str(forecaster),
        "forecaster_manifest_sha256": sha256(forecaster / "manifest.json"), "pool_path": str(pool_path),
        "pool_sha256": sha256(pool_path), "coordinate_scales_sha256": sha256(cfg["stage_b"]["coordinate_scales"]),
        "scales_path": cfg["stage_b"]["coordinate_scales"], "scorer_sha256": sha256(ROOT / "scoring.py"),
        "source_sha256": {p: sha256(ROOT / p) for p in ("scripts/forecast_partition.py", "scripts/evaluate_stage_b.py",
            "training/forecaster.py", "models/temporal.py", "common.py")},
        "seed": cfg["seed"], "seed_offset": 30000 if partition == "risk_calibration" else 10000,
        "climatology_seed_addition": 10000, "sample_counts": {m: 1 if m == "P" else 64 for m in MODES},
        "causality": "Only hours [origin-72, origin) supply observed states/embeddings. Future calendar is known. Targets are separate.",
        "same_input_audit": partition == "test", "model_fitting": False,
        "variant_label": cfg.get("variant", {}).get("label", "v3 primary")}
    if (out / "protocol.json").exists():
        assert read_json(out / "protocol.json") == protocol
    write_json(out / "protocol.json", protocol)
    return out


def worker(cfg, partition, shard, workers, device):
    out = output_root(cfg, partition)
    p = read_json(out / "protocol.json")
    digest = sha256(out / "protocol.json")
    net, base, model, scales, templates = components(cfg, device)
    base_digest = tensor_digest(base.model.state_dict())
    with np.load(p["pool_path"]) as f:
        pool = {k: f[k] for k in f.files}
    with np.load(out / "variogram_pairs.npz") as f:
        pairs = f["pairs"]
    ridge = RidgeForecaster(p["forecaster_path"])
    data = Path(cfg["paths"]["temporal_root"])
    cache_manifest = read_json(out / "cache/manifest.json")
    cached, episodes = {}, {}
    solver = None
    try:
        for index, origin in enumerate(p["origins"]):
            if index % workers != shard:
                continue
            folder = out / "origins" / origin["id"]
            done = folder / "done.json"
            if done.exists():
                record = read_json(done)
                assert record["protocol_sha256"] == digest
                assert all(sha256(folder / k) == h for k, h in record["artifacts"].items())
                continue
            name, o = origin["episode_id"], origin["origin"]
            if name not in cached:
                path = cache_manifest[name]["path"]
                assert sha256(path) == cache_manifest[name]["sha256"]
                cached[name] = torch.load(path, weights_only=True, mmap=True)["data"]
                records = read_json(data / f"episodes/{name}.json")["records"]
                episodes[name] = (records, episode_arrays(records, net))
            records, arrays = episodes[name]
            targets = target_values(records[o:o + 24], net)
            seed = p["seed"] + p["seed_offset"] + index
            mean, samples, info = ridge.forecast(arrays, o, seed)
            values, timings = {}, {}
            for mode in ("P", "S", "A", "D"):
                start = time.perf_counter()
                if mode in ("P", "S"):
                    values[mode] = frozen(base, net, mean[None] if mode == "P" else samples, templates)
                elif mode == "A":
                    values[mode], _ = adapter_samples(base, model, net, cached[name], o, templates, seed, device)
                else:
                    values[mode] = frozen(base, net, values["A"]["pd_mw"], templates, qd=values["A"]["qd_mvar"])
                timings[mode] = time.perf_counter() - start
            assert np.array_equal(values["A"]["pd_mw"], values["D"]["pd_mw"])
            draws = np.random.default_rng(seed + p["climatology_seed_addition"]).integers(0, 210, size=64)
            values["C"] = {k: v[draws] for k, v in pool.items()}
            values["P0"] = {k: np.repeat(v, 64, 0) for k, v in values["P"].items()}
            masks = {k: np.isfinite(targets[k]) for k in UNITS}
            atomic_npz(folder / "targets.npz", **targets)
            for mode in MODES:
                atomic_npz(folder / f"{mode}.npz", **values[mode])
                score, score_arrays = evaluate(values[mode], targets, masks, {"scales": scales, "variogram_pairs": pairs,
                    "mode": mode, "origin_id": origin["id"], "seed": seed + (10000 if mode in ("C", "P0") else 0)})
                score["physical_balance"] = {"p_mean_abs_mw": float(np.abs(values[mode]["residual_p_mw"]).mean()),
                                             "q_mean_abs_mvar": float(np.abs(values[mode]["residual_q_mvar"]).mean())}
                write_json(folder / f"{mode}_scores.json", score)
                atomic_npz(folder / f"{mode}_scores.npz", **score_arrays)
            if p["same_input_audit"]:
                if solver is None:
                    from power_fm_foresight._solver.adapter import IpoptSolver
                    solver = IpoptSolver(cfg, out / f"logs/solver_{shard:02d}.log")
                audits = audit_samples(folder, values, targets, net, solver)
                for mode, a in audits.items():
                    write_json(folder / f"{mode}_audit.json", a)
            write_json(folder / "timings.json", {"inference_seconds": timings, "device": device,
                        "climatology_block_indices": draws.tolist(), "baseline": info})
            assert tensor_digest(base.model.state_dict()) == base_digest
            artifacts = {path.name: sha256(path) for path in folder.iterdir() if path.is_file() and path.name != "done.json"}
            write_json(done, {"protocol_sha256": digest, "origin": origin, "seed": seed, "artifacts": artifacts,
                              "backbone_unchanged": True, "node": platform.node(), "device": device})
            print({"partition": partition, "origin": origin["id"], "done": True}, flush=True)
    finally:
        if solver is not None:
            solver.close()


def merge(cfg, partition):
    out = output_root(cfg, partition)
    p = read_json(out / "protocol.json")
    for mode in MODES:
        rows = [read_json(out / "origins" / o["id"] / f"{mode}_scores.json") for o in p["origins"]]
        result = merge_origin_scores(rows)
        result["physical_balance"] = {k: float(np.mean([r["physical_balance"][k] for r in rows])) for k in rows[0]["physical_balance"]}
        result["protocol_sha256"] = sha256(out / "protocol.json")
        write_json(out / f"{mode}.json", result)


def run(cfg, partition="risk_calibration", device="xpu", workers=12):
    start = time.time()
    out = prepare(cfg, partition, f"{device}:0" if device == "xpu" else device)
    processes, streams = [], []
    try:
        for shard in range(workers):
            stream = (out / f"worker_{shard:02d}.log").open("a")
            streams.append(stream)
            selected = f"xpu:{shard % torch.xpu.device_count()}" if device == "xpu" else device
            processes.append(subprocess.Popen([sys.executable, "-u", "-m", "scripts.forecast_partition", "--worker",
                "--partition", partition, "--shard", str(shard), "--workers", str(workers), "--device", selected],
                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT))
        codes = [process.wait() for process in processes]
        if any(codes):
            raise RuntimeError(f"Forecast worker failures: {codes}")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for stream in streams:
            stream.close()
    merge(cfg, partition)
    write_json(out / "forecast_job.json", {"complete": True, "partition": partition, "job_id": os.environ.get("PBS_JOBID"),
        "wall_seconds": time.time() - start, "workers": workers, "exit_codes": codes})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--partition", choices=("risk_calibration", "test"), default="risk_calibration")
    p.add_argument("--device", default="xpu")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--shard", type=int, default=0)
    args = p.parse_args()
    if args.worker:
        worker(load_config(), args.partition, args.shard, args.workers, args.device)
    else:
        run(load_config(), args.partition, args.device, args.workers)


if __name__ == "__main__":
    main()
