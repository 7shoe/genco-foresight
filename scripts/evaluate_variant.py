"""Frozen-scoring C/P/S/A/D evaluation and diagnostics for the isolated v4 run."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from common import ROOT, load_config, read_json, sha256, write_json
from scripts.forecast_partition import run as forecasts, MODES
from scripts.diagnose_stage_b import (npz, reference_series, group_diagnostics,
    balance, average_balances, average_errors, subset_summary, audit, LEADS)


def diagnose(cfg, shard, workers):
    b = Path(cfg["paths"]["runs_root"]) / "stage_b"
    p = read_json(b / "protocol.json")
    scales = read_json(cfg["stage_b"]["coordinate_scales"])
    free = np.flatnonzero(scales["features"]["pg_mw"]["joint_score_mask"])
    manifest = read_json(Path(cfg["paths"]["temporal_root"]) / "manifest.json")
    ref = reference_series(manifest, pd.read_csv(Path(cfg["paths"]["assets_root"]) / "profiles/eia_erco_demand_2025.csv"))
    for i, o in enumerate(p["origins"]):
        if i % workers != shard:
            continue
        folder = b / "origins" / o["id"]
        dest = b / "diagnostics/origins" / (o["id"] + ".json")
        done = read_json(folder / "done.json")
        assert all(sha256(folder / n) == h for n, h in done["artifacts"].items())
        if dest.exists():
            assert read_json(dest)["input_done_sha256"] == sha256(folder / "done.json")
            continue
        target = npz(folder / "targets.npz")
        r = ref[o["episode_id"]][o["origin"]:o["origin"] + 24]
        detail = {"input_done_sha256": sha256(folder / "done.json"), "modes": {}, "ood_P": {}}
        for m in MODES:
            values = npz(folder / f"{m}.npz")
            g = group_diagnostics(values, target, scales, np.ones(24, bool))
            score = read_json(folder / f"{m}_scores.json")
            np.testing.assert_allclose(g["joint_energy_from_contributions"], score["joint_energy_score"], rtol=1e-10, atol=1e-11)
            detail["modes"][m] = {"groups": g, "balance": balance(values, np.ones(24, bool))}
            if m in ("P", "S", "A", "D"):
                references = npz(folder / f"{m}_solver_subset.npz")
                detail["modes"][m]["same_input"] = audit(values, references, target, m, np.ones(24, bool), free)
            if m == "P":
                for label, hours in (("lt_0_6", r < .6), ("0_6_to_0_8", (r >= .6) & (r < .8)), ("ge_0_8", r >= .8)):
                    detail["ood_P"][label] = {"hours": int(hours.sum()),
                        "same_input": audit(values, references, target, m, hours, free),
                        "balance": balance(values, hours),
                        "balance_audited": balance(values, hours & np.isin(np.arange(24), LEADS))}
        write_json(dest, detail)


def merge(cfg):
    b = Path(cfg["paths"]["runs_root"]) / "stage_b"
    p = read_json(b / "protocol.json")
    scales = read_json(cfg["stage_b"]["coordinate_scales"])
    free = np.flatnonzero(scales["features"]["pg_mw"]["joint_score_mask"])
    rows = [read_json(b / "diagnostics/origins" / (o["id"] + ".json")) for o in p["origins"]]
    report = {"label": "in-distribution + physics-weighted", "primary_record": "v3", "ablation": False,
        "diagnostics_source_sha256": {name: sha256(ROOT / name) for name in ("scripts/evaluate_variant.py", "scripts/diagnose_stage_b.py")},
        "best_epoch": p["best_epoch"], "scales_path": cfg["stage_b"]["coordinate_scales"], "scales_sha256": p["coordinate_scales_sha256"],
        "scorer_sha256": sha256(ROOT / "scoring.py"), "dataset_id": cfg["dataset"]["id"], "modes": {}, "ood_P": {},
        "limits": ["Two changes after v3 scores (nominal profile range and physics-loss weight); this is exploratory, not an ablation.",
            "Train-derived scales use the unchanged recipe on the new training data. Absolute ES values across v3/v4 are not a controlled comparison.",
            "Noise and held-out profile extrapolation remain unchanged, so actual bus multipliers may leave [0.8,1.2].",
            "Unresolved realized states remain masked; demand remains observed. Missing past states use the original encoder/ridge availability masks.",
            "Family ES values are separately normalized and are not additive. Squared-distance allocations sum to joint ES but are diagnostic, not independent proper scores."]}
    from scripts.forecast_partition import components
    from training.state import tensor_digest, train_state
    _, base, adapter, _, _ = components(cfg, "cpu")
    weights = Path(cfg["paths"]["assets_root"]) / "genco-opfdata" / cfg["checkpoint"]["directory"] / cfg["checkpoint"]["weights"]
    report["provenance"] = {"trainable_parameters": sum(v.numel() for v in adapter.parameters() if v.requires_grad),
        "adapter_state_dict_elements_including_buffers": sum(v.numel() for v in train_state(adapter).values()),
        "base_checkpoint_file_sha256": sha256(weights), "base_named_tensor_bytes_sha256": tensor_digest(base.model.state_dict()),
        "adapter_checkpoint_file_sha256": p["checkpoint_sha256"], "best_epoch": p["best_epoch"],
        "selection_curve_path": str(Path(cfg["paths"]["training_run"]) / "epochs.json"),
        "training_manifest_unchanged": True}
    assert report["provenance"]["trainable_parameters"] == 91461
    assert report["provenance"]["adapter_state_dict_elements_including_buffers"] == 92345
    write_json(Path(cfg["paths"]["training_run"]) / "evaluation_provenance.json", report["provenance"])
    for m in MODES:
        score = read_json(b / f"{m}.json")
        detail = [r["modes"][m] for r in rows]
        score.update(free_generator_pg=subset_summary(score, "pg_mw", free),
            headline_vm=subset_summary(score, "vm", np.array([1, 9, 11, 13, 15, 21, 22]) - 1),
            line1_loading=subset_summary(score, "loading", [0]),
            balance=average_balances([r["balance"] for r in detail]), family_energy={})
        for name in detail[0]["groups"]["groups"]:
            g = [r["groups"]["groups"][name] for r in detail]
            score["family_energy"][name] = {"coordinates_per_hour": g[0]["coordinates_per_hour"], **{
                key: float(np.mean([v[key] for v in g if v[key] is not None])) if any(v[key] is not None for v in g) else None
                for key in ("group_energy_score", "joint_energy_contribution")}}
        if m in ("P", "S", "A", "D"):
            a = [r["same_input"] for r in detail]
            score["same_input"] = {"attempts": sum(r["attempts"] for r in a), "resolved": sum(r["resolved"] for r in a),
                "errors": average_errors([r["errors"] for r in a]),
                "raw_vs_refined": {k: average_errors([r["raw_vs_refined"][k] for r in a]) for k in ("raw", "solver_refined")}}
        # Compact report; full per-origin scorer output stays in the original mode JSON.
        score.pop("per_origin", None)
        report["modes"][m] = score
    for label in rows[0]["ood_P"]:
        r = [v["ood_P"][label] for v in rows]
        report["ood_P"][label] = {"hours": sum(x["hours"] for x in r), "same_input": average_errors([x["same_input"]["errors"] for x in r]),
            "balance": average_balances([x["balance"] for x in r]), "balance_audited": average_balances([x["balance_audited"] for x in r])}
    write_json(b / "report.json", report)
    lines = ["# V4: in-distribution + physics-weighted", "", f"Exploratory two-change variant; v3 remains primary. Best epoch {p['best_epoch']}. 72 test origins. All rows use the unchanged Stage B scorer.", "",
        "| Mode | Joint ES | Vm CRPS (7 buses) | Vm coverage90 | Free Pg MAE MW | Free Pg coverage90 | Mean abs P MW/bus | Mean abs Q MVAr/bus |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for m, r in report["modes"].items():
        lines.append(f"| {m} | {r['joint_energy_score']:.6f} | {r['headline_vm']['crps']:.6f} | {r['headline_vm']['coverage_90']:.4f} | {r['free_generator_pg']['mae']:.4f} | {r['free_generator_pg']['coverage_90']:.4f} | {r['balance']['residual_p_mw']['mean_abs']:.4f} | {r['balance']['residual_q_mvar']['mean_abs']:.4f} |")
    lines += ["", "Physical-unit MAE/MSE/CRPS/ES/variogram/PIT/50–90% coverage+width, family ES and allocations, same-input errors, raw-vs-refined errors, and ref_t bins: `report.json`; original scorer outputs: `{C,P,P0,S,A,D}.json`.", "", *["- " + v for v in report["limits"]]]
    (b / "report.md").write_text("\n".join(lines) + "\n")
    write_json(b / "completion.json", {"complete": True, "report_sha256": sha256(b / "report.json"), "origins": 72,
        "modes": list(MODES), "checkpoint_sha256": p["checkpoint_sha256"], "best_epoch": p["best_epoch"]})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--diag-worker", action="store_true")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--workers", type=int, default=12)
    a, cfg = p.parse_args(), load_config()
    assert cfg["dataset"]["id"] == "temporal_case30_v4"
    if a.diag_worker:
        diagnose(cfg, a.shard, a.workers)
        return
    training_manifest = read_json(Path(cfg["paths"]["training_run"]) / "manifest.json")
    assert all(sha256(ROOT / name) == digest for name, digest in training_manifest["source_sha256"].items()), "Training implementation changed before v4 evaluation"
    forecasts(cfg, "test", workers=a.workers)
    b = Path(cfg["paths"]["runs_root"]) / "stage_b"
    children, streams = [], []
    try:
        for shard in range(a.workers):
            stream = (b / f"diagnostics_{shard:02d}.log").open("a")
            streams.append(stream)
            children.append(subprocess.Popen([sys.executable, "-u", "-m", "scripts.evaluate_variant", "--diag-worker",
                "--shard", str(shard), "--workers", str(a.workers)], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT))
        codes = [p.wait() for p in children]
        assert not any(codes), codes
    finally:
        for p in children:
            if p.poll() is None: p.terminate()
        for s in streams: s.close()
    merge(cfg)


if __name__ == "__main__":
    main()
