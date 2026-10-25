"""Post-Stage-B controls: train-only daily climatology and replicated point mass."""
import argparse
import os
import platform
import subprocess
import sys
import time

import numpy as np
import torch

from common import ROOT, load_config, read_json, sha256, write_json
from data.loader.network import load_network
from scoring import UNITS, evaluate, merge_origin_scores
from scripts.evaluate_stage_b import locations, target_values
from training.forecaster import atomic_npz


METRICS = ("mae", "mse", "crps", "coverage_50", "coverage_90", "width_50", "width_90")


def prepare(cfg):
    root, data, _ = locations(cfg)
    assert read_json(root / "gate_b.json")["status"] == "complete"
    original = read_json(root / "protocol.json")
    for path, digest in original["source_sha256"].items():
        assert sha256(ROOT / path) == digest, f"Frozen scorer/evaluation source changed: {path}"
    net = load_network(config=load_config())
    manifest = read_json(data / "manifest.json")
    blocks, records, source_hashes = [], [], {}
    for episode in manifest["episodes"]:
        if episode["split"] != "train":
            continue
        name = episode["episode_id"]
        path = data / f"episodes/{name}.json"
        source_hashes[name] = sha256(path)
        rows = read_json(path)["records"]
        values = target_values(rows, net)
        assert len(rows) == 504
        for start in range(0, 481, 24):
            block = {k: v[start:start + 24] for k, v in values.items()}
            assert all(np.isfinite(block[k]).all() for k in UNITS), "Climatology requires valid train blocks"
            blocks.append(block)
            records.append({"episode_id": name, "split": "train", "start": start, "stop_exclusive": start + 24})
    assert len(blocks) == 210 and len(source_hashes) == 10
    atomic_npz(root / "controls/climatology_pool.npz", **{k: np.stack([b[k] for b in blocks]) for k in blocks[0]})
    protocol = {
        "original_protocol_sha256": sha256(root / "protocol.json"),
        "scorer_sha256": sha256(ROOT / "scoring.py"), "source_sha256": sha256(__file__),
        "dataset_manifest_sha256": sha256(data / "manifest.json"),
        "train_source_hashes": source_hashes, "climatology_blocks": records,
        "climatology_pool_sha256": sha256(root / "controls/climatology_pool.npz"),
        "climatology": "Unconditional empirical joint demand and solved-state daily trajectories; midnight aligned; training only; includes all 21 train days; no GENCO inference or test conditioning",
        "sample_counts": {"C": 64, "P0": 64}, "draws": "whole 24-hour/all-component blocks, with replacement",
        "seed_rule": "42 + 20000 + original origin index", "seed": cfg["seed"],
        "P0": "64 identical copies of the saved P trajectory; zero spread, no extra forward pass",
        "P0_comparison": "Marginal/variogram/coverage/width scores match to float64 precision. ES permits rtol=1e-7 because the frozen multi-sample path uses sqrt(integer count) in float32 and P uses numpy float64. Finite-M randomized PIT differs with ensemble size.",
        "origins": original["origins"], "scales": cfg["stage_b"]["coordinate_scales"],
        "coordinate_scales_sha256": original["coordinate_scales_sha256"],
        "variogram_pairs_sha256": original["variogram_pairs_sha256"],
        "created_after_stage_b_evaluation": True, "model_or_split_changes": False,
    }
    path = root / "controls/protocol.json"
    if path.exists():
        assert read_json(path) == protocol, "Control protocol changed"
    write_json(path, protocol)


def worker(cfg, shard, shards):
    torch.set_num_threads(1)
    root, _, _ = locations(cfg)
    protocol = read_json(root / "controls/protocol.json")
    digest = sha256(root / "controls/protocol.json")
    scales = read_json(protocol["scales"])
    with np.load(root / "variogram_pairs.npz") as f:
        pairs = f["pairs"]
    with np.load(root / "controls/climatology_pool.npz") as f:
        pool = {k: f[k] for k in f.files}
    for index, origin in enumerate(protocol["origins"]):
        if index % shards != shard:
            continue
        folder = root / "controls/origins" / origin["id"]
        folder.mkdir(parents=True, exist_ok=True)
        done = folder / "done.json"
        if done.exists():
            saved = read_json(done)
            assert saved["protocol_sha256"] == digest
            assert all(sha256(folder / k) == v for k, v in saved["artifacts"].items())
            continue
        source = root / "origins" / origin["id"]
        seed = cfg["seed"] + 20000 + index
        draw_ids = np.random.default_rng(seed).integers(0, 210, size=64)
        with np.load(source / "targets.npz") as f:
            targets = {k: f[k] for k in f.files}
        masks = {k: np.isfinite(targets[k]) for k in UNITS}
        with np.load(source / "P.npz") as f:
            point = {k: np.repeat(f[k], 64, axis=0) for k in f.files}
        for mode, values in (("C", {k: v[draw_ids] for k, v in pool.items()}), ("P0", point)):
            report, arrays = evaluate(values, targets, masks, {
                "scales": scales, "variogram_pairs": pairs, "mode": mode,
                "origin_id": origin["id"], "seed": seed,
            })
            report["no_uncertainty"] = mode == "P0"
            report["physical_balance"] = {
                "p_mean_abs_mw": float(np.abs(values["residual_p_mw"]).mean()),
                "q_mean_abs_mvar": float(np.abs(values["residual_q_mvar"]).mean()),
            }
            if mode == "C":
                report["climatology_block_indices"] = draw_ids.tolist()
            else:
                before = read_json(source / "P_scores.json")
                comparisons = [abs(report["joint_variogram_p0_5"]["score"] - before["joint_variogram_p0_5"]["score"])]
                energies = [(report["joint_energy_score"], before["joint_energy_score"])]
                for key in UNITS:
                    a, b = report["variables"][key], before["variables"][key]
                    for metric in METRICS:
                        np.testing.assert_allclose(a[metric], b[metric], atol=1e-10, rtol=1e-12)
                        comparisons.append(abs(a[metric] - b[metric]))
                    np.testing.assert_allclose(a["variogram_p0_5_physical"]["score"], b["variogram_p0_5_physical"]["score"], atol=1e-10, rtol=1e-12)
                    energies.append((a["energy_score_physical"], b["energy_score_physical"]))
                assert max(comparisons) < 1e-10
                for actual, expected in energies:
                    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-7)
                report["point_mass_metric_max_abs_difference"] = max(comparisons)
                report["point_mass_energy_max_abs_difference"] = max(abs(a-b) for a,b in energies)
                report["point_mass_energy_max_relative_difference"] = max(abs(a-b)/max(abs(b),1e-15) for a,b in energies)
            atomic_npz(folder / f"{mode}.npz", **values)
            atomic_npz(folder / f"{mode}_scores.npz", **arrays)
            write_json(folder / f"{mode}_scores.json", report)
        write_json(done, {"protocol_sha256": digest, "origin": origin, "seed": seed,
            "artifacts": {p.name: sha256(p) for p in folder.iterdir() if p.is_file() and p.name != "done.json"}})
        print({"control_origin": origin["id"], "shard": shard}, flush=True)


def merge(cfg):
    root, _, _ = locations(cfg)
    protocol = read_json(root / "controls/protocol.json")
    active = np.flatnonzero(load_network(config=load_config())["bus"][:, 2] != 0)
    buses = np.asarray(cfg["stage_b"]["calibration"]["headline_voltage_bus_ids"]) - 1
    for mode in ("C", "P0"):
        reports = [read_json(root / f"controls/origins/{o['id']}/{mode}_scores.json") for o in protocol["origins"]]
        result = merge_origin_scores(reports)
        result["active_pd_mae_mw"] = float(np.mean(np.asarray(result["variables"]["pd_mw"]["by_component"]["mae"])[active]))
        result["headline_voltage"] = {"bus_ids": (buses + 1).tolist(), **{k: float(np.mean(np.asarray(result["variables"]["vm"]["by_component"][k])[buses])) for k in METRICS}}
        result["physical_balance"] = {k: float(np.mean([r["physical_balance"][k] for r in reports])) for k in reports[0]["physical_balance"]}
        result["protocol_sha256"] = sha256(root / "controls/protocol.json")
        if mode == "P0":
            result["point_mass_metric_max_abs_difference"] = max(r["point_mass_metric_max_abs_difference"] for r in reports)
            result["point_mass_energy_max_abs_difference"] = max(r["point_mass_energy_max_abs_difference"] for r in reports)
            result["point_mass_energy_max_relative_difference"] = max(r["point_mass_energy_max_relative_difference"] for r in reports)
            result["no_uncertainty"] = True
        write_json(root / f"{mode}.json", result)
        print({"mode": mode, "energy_score": result["joint_energy_score"], "headline_vm": result["headline_voltage"]}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=12)
    args = parser.parse_args()
    cfg = load_config()
    if args.worker:
        worker(cfg, args.shard, args.shards)
        return
    started = time.time()
    prepare(cfg)
    root, _, _ = locations(cfg)
    streams, jobs = [], []
    try:
        for shard in range(args.shards):
            stream = (root / f"controls/worker_{shard:02d}.log").open("a")
            streams.append(stream)
            jobs.append(subprocess.Popen([sys.executable, "-u", "-m", "scripts.score_controls", "--worker",
                "--shard", str(shard), "--shards", str(args.shards)], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT))
        codes = [p.wait() for p in jobs]
        if any(codes):
            raise RuntimeError(f"Control worker failures: {codes}")
    finally:
        for p in jobs:
            if p.poll() is None:
                p.terminate()
        for stream in streams:
            stream.close()
    merge(cfg)
    write_json(root / "controls/job.json", {"job_id": os.environ.get("PBS_JOBID"), "node": platform.node(),
        "wall_seconds": time.time() - started, "workers": args.shards, "exit_codes": codes, "status": "complete"})


if __name__ == "__main__":
    main()
