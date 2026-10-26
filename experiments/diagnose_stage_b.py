"""Post-Gate-B diagnostics from immutable saved ensembles; no fitting or inference.

The existing scoring module is imported unchanged. Conditioning on realized ref_t
is retrospective, and does not define a deployable forecast or a new test split.
"""
import argparse
import csv
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

from power_fm_foresight._config import ROOT, load_config, read_json, sha256, write_json
from power_fm_foresight.scoring import (UNITS, ensemble_energy, evaluate, joint_coordinates,
                     marginal_scores, merge_origin_scores, paired_error,
                     summarize_marginals)
from power_fm_foresight._training.forecaster import atomic_npz


MODES = ("C", "P", "P0", "S", "A", "D")
GROUPS = {"Pd": ("pd_mw",), "Qd": ("qd_mvar",), "Vm": ("vm",),
          "Va": ("va_rad",), "Pg": ("pg_mw",), "Qg": ("qg_mvar",),
          "other": ("pf_mw", "qf_mvar", "pt_mw", "qt_mvar")}
SCOPES = ("all", "ref_ge_0_8")
LEADS = np.array([0, 6, 12, 18, 23])
SAMPLE_IDS = {"P": [0], "S": [0, 32], "A": [0, 32], "D": [0, 32]}
METRICS = ("mae", "mse", "crps", "coverage_50", "coverage_90", "width_50", "width_90")


def locations(cfg):
    root = Path(cfg["paths"]["runs_root"]) / "stage_b"
    out = root / "diagnostics"
    if not str(out.resolve()).startswith("/lus/flare/"):
        raise ValueError("Diagnostic artifacts must stay on Lustre")
    out.mkdir(parents=True, exist_ok=True)
    return root, out


def npz(path):
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def reference_series(manifest, profile):
    g = manifest["generation"]
    ref = g["scale_low"] + (profile.demand_mw.to_numpy() - manifest["train_profile_min"]) * (
        g["scale_high"] - g["scale_low"]) / (manifest["train_profile_max"] - manifest["train_profile_min"])
    return {e["episode_id"]: ref[e["source_start_hour"]:e["source_start_hour"] + e["hours"]]
            for e in manifest["episodes"] if e["split"] == "test"}


def group_diagnostics(values, targets, scales, hours):
    """Marginal group ES plus an exact allocation of the original joint ES.

    Squared-distance share allocation: d_g^2 / d_total for each sample-target
    and sample-pair distance, with zero allocated when d_total is zero. These
    contributions sum to joint ES; separately normalized group scores do not.
    """
    m = len(values["vm"])
    pieces, result = {}, {}
    for name, keys in GROUPS.items():
        chosen = {k: scales["features"][k] for k in keys
                  if any(scales["features"][k]["joint_score_mask"])}
        width = sum(sum(s["joint_score_mask"]) for s in chosen.values())
        result[name] = {"coordinates_per_hour": width, "observed_coordinates": 0,
                        "group_energy_score": None, "joint_energy_contribution": None}
        if not width:
            continue
        small = {"features": chosen}
        x, y = joint_coordinates(values, small), joint_coordinates(targets, small)
        mask = np.isfinite(y) & np.asarray(hours)[:, None]
        result[name]["observed_coordinates"] = int(mask.sum())
        if not mask.any():
            continue
        scale = np.concatenate([np.asarray(s["scale"])[s["joint_score_mask"]] for s in chosen.values()])
        result[name]["group_energy_score"] = ensemble_energy(x, y, np.tile(scale, len(y)), mask.reshape(-1))
        xx = np.asarray(x, dtype=float)[:, mask] / np.broadcast_to(scale, y.shape)[mask]
        yy = np.asarray(y, dtype=float)[mask] / np.broadcast_to(scale, y.shape)[mask]
        target_sq = np.sum((xx - yy) ** 2, axis=1)
        pair_sq = np.empty((m, m), dtype=float)
        # Bounded memory; avoid cancellation from a squared Gram-matrix identity.
        for start in range(0, m, 8):
            delta = xx[start:start + 8, None, :] - xx[None, :, :]
            pair_sq[start:start + 8] = np.einsum("ijk,ijk->ij", delta, delta)
        pieces[name] = (target_sq, pair_sq)
    if not pieces:
        return {"groups": result, "joint_energy_from_contributions": None}
    to_target = np.sqrt(sum(t for t, _ in pieces.values()))
    between = np.sqrt(sum(p for _, p in pieces.values()))
    dimension = sum(r["observed_coordinates"] for r in result.values())
    # Preserve the frozen scorer's float32 integer-count square root for M >= 2.
    normalizer = float(torch.tensor(dimension).sqrt()) if m >= 2 else np.sqrt(dimension)
    for name, (t, p) in pieces.items():
        target_share = np.divide(t, to_target, out=np.zeros_like(t), where=to_target > 0).mean()
        pair_share = np.divide(p, between, out=np.zeros_like(p), where=between > 0).sum()
        result[name]["joint_energy_contribution"] = float(
            (target_share - (pair_share / (2 * m * (m - 1)) if m > 1 else 0)) / normalizer)
    return {"groups": result, "joint_energy_from_contributions": sum(
        r["joint_energy_contribution"] for r in result.values() if r["joint_energy_contribution"] is not None)}


def balance(values, hours):
    out = {}
    for key in ("residual_p_mw", "residual_q_mvar"):
        x = np.abs(values[key][:, hours, :])
        n = x.shape[0] * x.shape[1]
        out[key] = {"count": int(x.size), "samples_times_hours_per_bus": n,
                    "mean_abs": float(x.mean()) if x.size else None,
                    "per_bus_mean_abs": x.mean((0, 1)).tolist() if n else [None] * x.shape[-1]}
    return out


def average_errors(rows):
    out = {}
    for key in rows[0]:
        n = sum(r[key]["count"] for r in rows)
        out[key] = {"count": n, "unit": rows[0][key]["unit"], **{
            k: sum(r[key][k] * r[key]["count"] for r in rows if r[key]["count"]) / n if n else None
            for k in ("mae", "mse")}}
    return out


def average_balances(rows):
    out = {}
    for key in rows[0]:
        n = sum(r[key]["count"] for r in rows)
        nb = sum(r[key]["samples_times_hours_per_bus"] for r in rows)
        out[key] = {"count": n, "samples_times_hours_per_bus": nb,
                    "mean_abs": sum(r[key]["mean_abs"] * r[key]["count"] for r in rows if r[key]["count"]) / n if n else None,
                    "per_bus_mean_abs": (sum(np.array(r[key]["per_bus_mean_abs"]) * r[key]["samples_times_hours_per_bus"]
                         for r in rows if r[key]["samples_times_hours_per_bus"]) / nb).tolist() if nb else [None] * len(rows[0][key]["per_bus_mean_abs"])}
    return out


def subset_summary(result, key, indices):
    row = result["variables"][key]["by_component"]
    counts = np.asarray(row["count"])[indices]
    valid = counts > 0
    return {"count": int(counts.sum()), **{k: float(np.average(np.asarray(row[k], dtype=float)[indices][valid],
             weights=counts[valid])) if valid.any() else None for k in METRICS}}


def audit(raw, references, targets, mode, hours, free_pg):
    ids = np.array(SAMPLE_IDS[mode])
    selected = {k: v[ids[:, None], LEADS[None]] for k, v in raw.items()}
    valid = references["resolved"] & hours[LEADS][None]
    errors = paired_error(selected, references, valid)
    # paired_error intentionally expects all physical keys; select free Pg afterward.
    delta = selected["pg_mw"][..., free_pg] - references["pg_mw"][..., free_pg]
    keep = np.broadcast_to(valid[..., None], delta.shape) & np.isfinite(delta)
    errors["pg_free_mw"] = {"count": int(keep.sum()), "unit": "MW",
                            "mae": float(np.abs(delta[keep]).mean()) if keep.any() else None,
                            "mse": float((delta[keep] ** 2).mean()) if keep.any() else None}
    truth = {k: v[LEADS] for k, v in targets.items()}
    refined = {k: v for k, v in references.items() if k != "resolved"}
    matched = valid.all(0)
    return {"attempts": int(hours[LEADS].sum() * len(ids)), "resolved": int(valid.sum()), "errors": errors,
            "raw_vs_refined": {label: paired_error({k: v.mean(0) for k, v in x.items()}, truth, matched)
                                for label, x in (("raw", selected), ("solver_refined", refined))}}


def prepare(cfg):
    root, out = locations(cfg)
    original = read_json(root / "protocol.json")
    controls = read_json(root / "controls/protocol.json")
    data = Path(cfg["paths"]["temporal_root"])
    manifest = read_json(data / "manifest.json")
    profile = Path(cfg["paths"]["assets_root"]) / "profiles/eia_erco_demand_2025.csv"
    assert sha256(profile) == manifest["profile_sha256"]
    assert sha256(data / "manifest.json") == original["dataset_manifest_sha256"]
    for path, digest in original["source_sha256"].items():
        assert sha256(ROOT / path) == digest, path
    assert sha256(ROOT / "scoring.py") == controls["scorer_sha256"]
    scales_path = Path(cfg["stage_b"]["coordinate_scales"])
    assert sha256(scales_path) == original["coordinate_scales_sha256"]
    assert sha256(root / "variogram_pairs.npz") == original["variogram_pairs_sha256"]
    frame = pd.read_csv(profile)
    refs = reference_series(manifest, frame)
    hashes = {r["file"]: r["sha256"] for r in manifest["shards"]}
    # Check source-time alignment and realized targets without regenerating labels.
    for e in manifest["episodes"]:
        if e["split"] != "test":
            continue
        path = data / f"episodes/{e['episode_id']}.json"
        assert sha256(path) == hashes[f"episodes/{e['episode_id']}.json"]
        records = read_json(path)["records"]
        for h, r in enumerate(records):
            assert pd.Timestamp(r["time_utc"]) == pd.Timestamp(frame.time_utc.iloc[e["source_start_hour"] + h])
    immutable = [root / "protocol.json", root / "controls/protocol.json", scales_path, profile, data / "manifest.json",
                 root / "variogram_pairs.npz", Path(cfg["paths"]["training_run"]) / "best.pt",
                 Path(cfg["paths"]["training_run"]) / "manifest.json", root / "probe/protocol.json",
                 root / "probe/predictions_h1.npz", root / "probe/predictions_h24.npz"]
    immutable += [ROOT / p for p in original["source_sha256"]]
    immutable += [root / f"{m}.json" for m in MODES]
    protocol = {"diagnostic_source_sha256": sha256(__file__), "created_after_test_scores": True,
        "no_retraining_no_inference_no_new_solver_calls": True, "scorer_sha256": sha256(ROOT / "scoring.py"),
        "immutable_inputs": {str(p): sha256(p) for p in immutable}, "origins": original["origins"],
        "reference_series": {k: v.tolist() for k, v in refs.items()},
        "ref_t_definition": "Original train-derived affine profile transform; no clipping, refitting, or per-bus noise. Thresholds supplied by user: <0.6, [0.6,0.8), >=0.8.",
        "in_range_limit": "Realized ref_t >= 0.8 is a retrospective profile slice, not proof the forecast demand graphs are in the pretrained distribution.",
        "subset_scoring": "Keep the 24-hour layout and frozen variogram pairs; mask other hours. ES averaged equally over origins with >=1 retained hour, each normalized by sqrt(retained dimensions). Physical marginals weighted by valid coordinate count.",
        "group_scores": "Same fair ES and training coordinate scales, restricted to each group, divided by sqrt(its observed dimension); NOT additive.",
        "additive_allocation": "For each normalized Euclidean distance allocate d_g^2/d_total (0 if total=0), then apply original fair ES mean/pair subtraction and full joint dimension normalization. Contributions sum to joint ES; not standalone proper group scores or causal attributions.",
        "same_input_subset": {"leads_zero_based": LEADS.tolist(), "sample_ids": SAMPLE_IDS, "no_new_solves": True},
        "seed": cfg["seed"], "scales_path": str(scales_path), "training_run": cfg["paths"]["training_run"],
        "headline_bus_ids": cfg["stage_b"]["calibration"]["headline_voltage_bus_ids"]}
    if (out / "protocol.json").exists():
        assert read_json(out / "protocol.json") == protocol, "Diagnostic protocol changed; use a new output directory"
    write_json(out / "protocol.json", protocol)
    scales = read_json(scales_path)
    write_json(out / "coordinate_scales.json", scales)
    with (out / "coordinates.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["joint_index_zero_based", "group", "variable", "component_one_based", "scale", "unit", "train_population_std", "floor"])
        index = 0
        for key, s in scales["features"].items():
            for component, keep in enumerate(s["joint_score_mask"]):
                if keep:
                    group = next(g for g, keys in GROUPS.items() if key in keys)
                    writer.writerow([index, group, key, component + 1, s["scale"][component], UNITS[key], s["population_std"][component], s["floor"]])
                    index += 1
        assert index == 252


def worker(cfg, shard, shards):
    torch.set_num_threads(1)
    root, out = locations(cfg)
    protocol = read_json(out / "protocol.json")
    digest = sha256(out / "protocol.json")
    scales = read_json(protocol["scales_path"])
    free_pg = np.flatnonzero(scales["features"]["pg_mw"]["joint_score_mask"])
    pairs = npz(root / "variogram_pairs.npz")["pairs"]
    for index, origin in enumerate(protocol["origins"]):
        if index % shards != shard:
            continue
        dest = out / "origins" / origin["id"]
        done_path = dest / "done.json"
        if done_path.exists():
            done = read_json(done_path)
            assert done["protocol_sha256"] == digest
            assert all(sha256(dest / k) == v for k, v in done["artifacts"].items())
            assert all(sha256(p) == h for p, h in done["input_sha256"].items())
            continue
        start = time.monotonic()
        source = root / "origins" / origin["id"]
        control = root / "controls/origins" / origin["id"]
        inputs = {}
        for folder in (source, control):
            done = read_json(folder / "done.json")
            for name, h in done["artifacts"].items():
                path = folder / name
                assert sha256(path) == h, str(path)
                inputs[str(path)] = h
        targets = npz(source / "targets.npz")
        ref = np.asarray(protocol["reference_series"][origin["episode_id"]])[origin["origin"]:origin["origin"] + 24]
        hours_by_scope = {"all": np.ones(24, dtype=bool), "ref_ge_0_8": ref >= .8}
        details = {"origin": origin, "ref_t": ref.tolist(), "hours": {k: int(v.sum()) for k, v in hours_by_scope.items()},
                   "balance": {}, "groups": {}, "audit": {}, "ood_P": {}}
        for mode in MODES:
            folder = control if mode in ("C", "P0") else source
            values = npz(folder / f"{mode}.npz")
            references = npz(source / f"{mode}_solver_subset.npz") if mode in SAMPLE_IDS else None
            details["balance"][mode], details["groups"][mode], details["audit"][mode] = {}, {}, {}
            for scope, hours in hours_by_scope.items():
                details["balance"][mode][scope] = balance(values, hours)
                if scope == "all":
                    scores = read_json(folder / f"{mode}_scores.json")
                else:
                    masks = {k: np.isfinite(targets[k]) & hours[:, None] for k in UNITS}
                    scores, arrays = evaluate(values, targets, masks, {"scales": scales, "variogram_pairs": pairs,
                        "mode": mode, "origin_id": origin["id"], "seed": protocol["seed"] + (20000 if mode in ("C", "P0") else 10000) + index})
                    write_json(dest / f"{mode}_{scope}.json", scores)
                    atomic_npz(dest / f"{mode}_{scope}.npz", **arrays)
                if mode in ("C", "S", "A", "D"):
                    group = group_diagnostics(values, targets, scales, hours)
                    if hours.any():
                        np.testing.assert_allclose(group["joint_energy_from_contributions"], scores["joint_energy_score"], rtol=1e-10, atol=1e-11)
                    details["groups"][mode][scope] = group
                if references is not None:
                    details["audit"][mode][scope] = audit(values, references, targets, mode, hours, free_pg)
            if mode == "P":
                for label, hours in (("lt_0_6", ref < .6), ("0_6_to_0_8", (ref >= .6) & (ref < .8)), ("ge_0_8", ref >= .8)):
                    details["ood_P"][label] = {"forecast_hours": int(hours.sum()), "balance_all_forecast_hours": balance(values, hours),
                        "same_input": audit(values, references, targets, mode, hours, free_pg),
                        "balance_audited_hours": balance(values, hours & np.isin(np.arange(24), LEADS))}
        write_json(dest / "details.json", details)
        assert all(sha256(p) == h for p, h in inputs.items()), "Original evidence changed during diagnostics"
        files = {p.name: sha256(p) for p in dest.iterdir() if p.is_file() and p.name != "done.json"}
        write_json(done_path, {"protocol_sha256": digest, "input_sha256": inputs, "artifacts": files,
                              "wall_seconds": time.monotonic() - start, "node": platform.node()})
        print({"origin": origin["id"], "high_ref_hours": int((ref >= .8).sum()), "wall_seconds": time.monotonic() - start}, flush=True)


def provenance(cfg):
    from power_fm_foresight._data.network import load_network
    from power_fm_foresight.genco import Genco
    from power_fm_foresight.temporal import TemporalAdapter
    from power_fm_foresight._training.state import tensor_digest
    root, out = locations(cfg)
    train = Path(cfg["paths"]["training_run"])
    base_root = Path(cfg["paths"]["assets_root"]) / "genco-opfdata"
    base = Genco(base_root, cfg["checkpoint"], "cpu")
    model = TemporalAdapter(base, load_network(config=cfg), read_json(cfg["stage_b"]["coordinate_scales"]))
    checkpoint = torch.load(train / "best.pt", weights_only=True, map_location="cpu")
    state = checkpoint["model"]
    parameters = {k: p for k, p in model.named_parameters() if p.requires_grad}
    buffers = {k: b for k, b in model.named_buffers() if not k.startswith("backbone.")}
    assert set(state) == set(parameters) | set(buffers)
    assert all(state[k].shape == v.shape for k, v in {**parameters, **buffers}.items())
    counts = {"trainable_parameters": sum(p.numel() for p in parameters.values()),
              "adapter_buffer_elements": sum(b.numel() for b in buffers.values()),
              "adapter_state_dict_elements": sum(t.numel() for t in state.values())}
    assert counts == {"trainable_parameters": 91461, "adapter_buffer_elements": 884, "adapter_state_dict_elements": 92345}
    weight_path = base_root / cfg["checkpoint"]["directory"] / cfg["checkpoint"]["weights"]
    tensor_sha = tensor_digest(base.model.state_dict())
    assert tensor_sha == read_json(train / "status.json")["base_sha256"]
    result = {**counts, "best_epoch": checkpoint["progress"]["best_epoch"],
        "buffers": {k: {"elements": b.numel(), "dtype": str(b.dtype)} for k, b in buffers.items()},
        "base_checkpoint_path": str(weight_path), "base_checkpoint_file_sha256": sha256(weight_path),
        "base_state_tensor_bytes_sha256": tensor_sha, "adapter_checkpoint_file_sha256": sha256(train / "best.pt"),
        "adapter_state_tensor_bytes_sha256": tensor_digest(state),
        "tensor_hash_definition": "SHA256 concatenation of each sorted state-dict key UTF-8 bytes followed by its contiguous CPU NumPy tensor bytes; includes buffers. Does not hash the serialized file, shape metadata, or dtype metadata.",
        "correction": "92,345 counts adapter state-dict elements including 884 buffers, not trainable parameters. The supplied 9dbf... is the base checkpoint FILE SHA; df0c... hashes named tensor bytes. Earlier discrepancy wording conflated these definitions; it does not establish a changed checkpoint. Historical manifests/reports retained unchanged; this record corrects their interpretation."}
    assert result["base_checkpoint_file_sha256"] == "9dbf5596b4eb63a3b0e23a7d9a84427c33f3829a07ad2d7dbd12e99e364545a4"
    write_json(out / "parameter_hash_accounting.json", result)
    return result


def probe_diagnostic(root, protocol, scales):
    result = {}
    for horizon in (1, 24):
        x = npz(root / f"probe/predictions_h{horizon}.npz")
        ref = np.array([protocol["reference_series"][e][int(h) + horizon] for e, h in zip(x["episode"], x["hour"])])
        result[str(horizon)] = {}
        for scope, hours in (("all", np.ones(len(ref), dtype=bool)), ("ref_ge_0_8", ref >= .8)):
            row = {"hours": int(hours.sum()), "families": {}}
            for family in ("raw", "raw_plus_genco"):
                variables, normalized, truth, masks = {}, [], [], []
                for key in ("vm", "va_rad", "pg_mw", "qg_mvar"):
                    p, y = x[f"{family}_{key}"], x[f"target_{key}"]
                    s = scales["features"][key]
                    mask = np.broadcast_to(s["joint_score_mask"], y.shape) & hours[:, None]
                    variables[key] = summarize_marginals(marginal_scores(p[None], y, mask))
                    normalized.append(p / s["scale"])
                    truth.append(y / s["scale"])
                    masks.append(mask)
                score = summarize_marginals(marginal_scores(np.concatenate(normalized, -1)[None],
                    np.concatenate(truth, -1), np.concatenate(masks, -1)))
                row["families"][family] = {"normalized": score, "variables": variables}
            result[str(horizon)][scope] = row
    return result


def merge(cfg):
    root, out = locations(cfg)
    protocol = read_json(out / "protocol.json")
    scales = read_json(protocol["scales_path"])
    rows = [read_json(out / "origins" / o["id"] / "details.json") for o in protocol["origins"]]
    groups = {g: {"coordinates_per_hour": sum(sum(scales["features"][k]["joint_score_mask"]) for k in keys), "variables": list(keys)} for g, keys in GROUPS.items()}
    full_ref = np.concatenate(list(protocol["reference_series"].values()))
    forecast_ref = np.concatenate([r["ref_t"] for r in rows])
    result = {"protocol_sha256": sha256(out / "protocol.json"), "groups": groups,
        "ref_counts": {label: {"hours": len(r), "hours_ge_0_8": int((r >= .8).sum()), "fraction_ge_0_8": float((r >= .8).mean())}
                       for label, r in (("forecast_hours", forecast_ref), ("all_test_episode_hours_including_history", full_ref))},
        "scopes": {}, "ood_P": {}, "parameter_hash_accounting": provenance(cfg)}
    active = np.flatnonzero(scales["features"]["pd_mw"]["joint_score_mask"])
    free_pg = np.flatnonzero(scales["features"]["pg_mw"]["joint_score_mask"])
    buses = np.array(protocol["headline_bus_ids"]) - 1
    for scope in SCOPES:
        summary = {"forecast_hours": sum(r["hours"][scope] for r in rows),
                   "nonempty_origins": sum(r["hours"][scope] > 0 for r in rows), "modes": {}}
        for mode in MODES:
            if scope == "all":
                scores = read_json(root / f"{mode}.json")
            else:
                scores = merge_origin_scores([read_json(out / "origins" / o["id"] / f"{mode}_{scope}.json") for o in protocol["origins"]])
            scores.pop("per_origin", None)
            item = {"scores": scores, "balance": average_balances([r["balance"][mode][scope] for r in rows]),
                "headline": {"active_pd": subset_summary(scores, "pd_mw", active), "vm": subset_summary(scores, "vm", buses),
                             "line_1": subset_summary(scores, "loading", [0]), "pg_all": subset_summary(scores, "pg_mw", np.arange(len(scales["features"]["pg_mw"]["scale"]))),
                             "pg_free": subset_summary(scores, "pg_mw", free_pg)}}
            if mode in ("C", "S", "A", "D"):
                item["groups"] = {}
                for group in GROUPS:
                    rs = [r["groups"][mode][scope]["groups"][group] for r in rows]
                    item["groups"][group] = {"coordinates_per_hour": rs[0]["coordinates_per_hour"],
                        "observed_coordinates": sum(r["observed_coordinates"] for r in rs), **{
                        k: float(np.mean([r[k] for r in rs if r[k] is not None])) if any(r[k] is not None for r in rs) else None
                        for k in ("group_energy_score", "joint_energy_contribution")}}
                closure = sum(r["joint_energy_contribution"] for r in item["groups"].values() if r["joint_energy_contribution"] is not None)
                np.testing.assert_allclose(closure, scores["joint_energy_score"], rtol=1e-10, atol=1e-11)
                item["joint_contribution_closure_abs_error"] = abs(closure - scores["joint_energy_score"])
            if mode in SAMPLE_IDS:
                audits = [r["audit"][mode][scope] for r in rows]
                item["audit"] = {"resolved": sum(r["resolved"] for r in audits), "attempts": sum(r["attempts"] for r in audits),
                    "errors": average_errors([r["errors"] for r in audits]), "raw_vs_refined": {
                        k: average_errors([r["raw_vs_refined"][k] for r in audits]) for k in ("raw", "solver_refined")}}
            summary["modes"][mode] = item
        result["scopes"][scope] = summary
    for label in ("lt_0_6", "0_6_to_0_8", "ge_0_8"):
        rs = [r["ood_P"][label] for r in rows]
        result["ood_P"][label] = {"forecast_hours": sum(r["forecast_hours"] for r in rs),
            "same_input_resolved": sum(r["same_input"]["resolved"] for r in rs),
            "same_input_attempts": sum(r["same_input"]["attempts"] for r in rs),
            "same_input_errors": average_errors([r["same_input"]["errors"] for r in rs]),
            **{k: average_balances([r[k] for r in rs]) for k in ("balance_all_forecast_hours", "balance_audited_hours")}}
    result["probe"] = probe_diagnostic(root, protocol, scales)
    old_probe = read_json(root / "probe/report.json")
    for horizon in ("1", "24"):
        for family in ("raw", "raw_plus_genco"):
            np.testing.assert_allclose(result["probe"][horizon]["all"]["families"][family]["normalized"]["mse"],
                old_probe["horizons"][horizon]["families"][family]["normalized"]["mse"], atol=1e-12)
    assert all(sha256(p) == h for p, h in protocol["immutable_inputs"].items())
    checks = {"original_global_input_hashes_unchanged": True, "scorer_sha256": sha256(ROOT / "scoring.py"),
              "group_contribution_closure_checked_all_origins_and_modes": True, "probe_full_score_reproduced": True,
              "verified_per_origin_input_hashes": sum(len(read_json(out / "origins" / o["id"] / "done.json")["input_sha256"]) for o in protocol["origins"])}
    result["verification"] = checks
    write_json(out / "report.json", result)
    render(out, result, scales)
    print({"diagnostics_complete": True, "ref_counts": result["ref_counts"], "report": str(out / "report.md")}, flush=True)


def render(out, result, scales):
    def f(x):
        return "N/A" if x is None else f"{x:.6g}"
    def pct(x):
        return "N/A" if x is None else f"{100*x:.2f}%"
    def table(headers, rows):
        return ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows] + [""]
    lines = ["# Gate B follow-up diagnostics", "", "Saved predictions and numerical references only. No retraining, no new draws/forwards/solver calls, and no scorer changes. C is empirical train climatology; P is ridge→GENCO; P0 is 64 identical P copies; S is residual-block sampling→GENCO; A is the epoch-18 adapter; D uses A's exact demand through frozen GENCO.", "",
        "**Retrospective conditioning.** ref_t uses the original profile transform in the dataset manifest, not the noisy realized bus demands. The user-specified 0.8 threshold is only a proxy for pretrained support: the actual P/S/A/D forecast demand can differ. This diagnostic cannot prove that OOD causes the errors. The >=0.8 slice retains the original 24-hour coordinate layout and variogram pairs, masks other hours, and averages ES equally over nonempty origins. It is not a full-day forecast score or a new split.", ""]
    lines += table(["Denominator", "Hours", "ref_t >= 0.8", "Fraction"], [[label, r["hours"], r["hours_ge_0_8"], pct(r["fraction_ge_0_8"])] for label, r in result["ref_counts"].items()])
    lines += ["**P binned same-input approximation.** Only the pre-existing five leads per origin have numerical references. Imbalance is raw |generation - demand - AC network injection| averaged over buses, reported both on those audited hours and on every forecast hour. This is neither a forecast-error residual nor absolute net system imbalance.", ""]
    lines += table(["Realized ref_t", "Forecast hours", "Audited resolved/attempted", "Vm MAE p.u.", "Pg MAE MW (6 / free 2)", "Audit |P| MW/bus", "Audit |Q| MVAr/bus", "All-hour |P|", "All-hour |Q|"],
        [[label, r["forecast_hours"], f"{r['same_input_resolved']}/{r['same_input_attempts']}", f(r["same_input_errors"]["vm"]["mae"]),
          f(r["same_input_errors"]["pg_mw"]["mae"]) + " / " + f(r["same_input_errors"]["pg_free_mw"]["mae"]),
          *[f(r[k][v]["mean_abs"]) for k in ("balance_audited_hours", "balance_all_forecast_hours") for v in ("residual_p_mw", "residual_q_mvar")]] for label, r in result["ood_P"].items()])
    lines += ["The >=0.8 row is the restricted version of this binned table; the other two bins contain zero qualifying hours.", "",
        "**Joint coordinate accounting.** 252 per hour; 6,048 for a complete day. Qd is scored as a physical marginal but contributes no joint coordinates. Loading contributes none. Other means both-end P/Q flows, not an unlisted latent variable. Flows occupy 164/252 = 65.1% of coordinates; groups are not equally weighted. Vm uses all 30 buses, not only the seven headline buses. Pg excludes four fixed generators; Va excludes the slack angle. The 41-line flow blocks retain near-constant active-flow coordinates too.", "",
        "Scale = max(train population standard deviation, physical floor), fitted on 5,040 unique train hours, tiled across leads. Exact ordered coordinates and scales: [coordinates.csv](coordinates.csv); full original scales/masks/statistics: [coordinate_scales.json](coordinate_scales.json). These scales remain fixed in both scopes.", ""]
    lines += table(["Variable", "Joint coords/hour", "Physical unit", "Scale floor", "Retained scale min / max"],
        [[k, sum(s["joint_score_mask"]), UNITS[k], f(s["floor"]),
          (f(min(np.array(s["scale"])[s["joint_score_mask"]])) + " / " + f(max(np.array(s["scale"])[s["joint_score_mask"]]))) if any(s["joint_score_mask"]) else "N/A (excluded)"] for k, s in scales["features"].items()])
    for scope, summary in result["scopes"].items():
        modes = summary["modes"]
        lines += [f"## {'All forecast hours' if scope == 'all' else 'Secondary: realized ref_t >= 0.8'}", "",
                  f"{summary['forecast_hours']} retained hours; {summary['nonempty_origins']} nonempty origins out of 72. Same saved ensembles, masks, training scales and estimator.", ""]
        lines += table(["Mode", "Joint ES", "Variogram p=.5", "Active Pd MAE MW", "Headline Vm MAE p.u.", "Vm CRPS", "Vm 50/90% coverage", "Vm 90% width", "Line-1 loading MAE"],
            [[m, f(r["scores"]["joint_energy_score"]), f(r["scores"]["joint_variogram_p0_5"]["score"]), f(r["headline"]["active_pd"]["mae"]), f(r["headline"]["vm"]["mae"]),
              f(r["headline"]["vm"]["crps"]), pct(r["headline"]["vm"]["coverage_50"]) + " / " + pct(r["headline"]["vm"]["coverage_90"]), f(r["headline"]["vm"]["width_90"]), f(r["headline"]["line_1"]["mae"])] for m, r in modes.items()])
        lines += ["**Raw balance and Pg forecasts.** All-six Pg includes four fixed-zero coordinates; their exact zero errors and perfect interval coverage dilute the mean and inflate coverage. The two free generators are therefore also reported. Residual means cover all 30 buses and all saved samples; per-bus values and counts are in report.json.", ""]
        lines += table(["Mode", "|P| MW/bus", "|Q| MVAr/bus", "Pg MAE MW: all 6", "Pg coverage 50/90%: all 6", "Pg MAE MW: free 2", "Pg coverage 50/90%: free 2", "Free Pg 90% width MW"],
            [[m, f(r["balance"]["residual_p_mw"]["mean_abs"]), f(r["balance"]["residual_q_mvar"]["mean_abs"]), f(r["headline"]["pg_all"]["mae"]),
              pct(r["headline"]["pg_all"]["coverage_50"]) + " / " + pct(r["headline"]["pg_all"]["coverage_90"]), f(r["headline"]["pg_free"]["mae"]),
              pct(r["headline"]["pg_free"]["coverage_50"]) + " / " + pct(r["headline"]["pg_free"]["coverage_90"]), f(r["headline"]["pg_free"]["width_90"])] for m, r in modes.items()])
        lines += ["**Group-restricted ES.** Original estimator/scales applied within each group with that group's dimension normalizer. These values do not sum to joint ES; Qd is N/A, not a zero error.", ""]
        lines += table(["Mode", *GROUPS], [[m, *[f(modes[m]["groups"][g]["group_energy_score"]) for g in GROUPS]] for m in ("C", "S", "A", "D")])
        lines += ["**Additive allocation of joint ES.** Each Euclidean distance contributes d_g²/d_total to group g (zero if total distance is zero), followed by the unchanged fair pair subtraction and full joint-dimension normalization. Rows sum to joint ES. This algebraic diagnostic is not a proper score for a standalone group or a causal attribution; a contribution can be negative.", ""]
        lines += table(["Mode", *GROUPS, "Joint ES"], [[m, *[f(modes[m]["groups"][g]["joint_energy_contribution"]) for g in GROUPS], f(modes[m]["scores"]["joint_energy_score"])] for m in ("C", "S", "A", "D")])
        lines += ["**Same-input approximation (fixed audit subset).** A/D share solver references. Unresolved cases remain masked and are not called infeasible.", ""]
        lines += table(["Mode", "Resolved / attempted", "Vm MAE p.u.", "Pg MAE MW (6 / free 2)", "Loading MAE"],
            [[m, f"{r['audit']['resolved']}/{r['audit']['attempts']}", f(r["audit"]["errors"]["vm"]["mae"]),
              f(r["audit"]["errors"]["pg_mw"]["mae"]) + " / " + f(r["audit"]["errors"]["pg_free_mw"]["mae"]), f(r["audit"]["errors"]["loading"]["mae"])] for m, r in modes.items() if "audit" in r])
        lines += ["**Matched raw/refined forecast errors.** Only the fixed audit samples/leads, requiring all selected solves resolved at each hour. These are mean forecast errors against the realized target, not same-input errors and not full-64-sample refinement.", ""]
        lines += table(["Mode", "State", "Vm MAE p.u.", "Pg MAE MW all 6", "Loading MAE"],
            [[m, label, *[f(errors[k]["mae"]) for k in ("vm", "pg_mw", "loading")]] for m, r in modes.items() if "audit" in r for label, errors in r["audit"]["raw_vs_refined"].items()])
        lines += ["**Physical marginal metrics, all components.** Widths share each variable's physical units. The complete machine-readable report includes per-component counts/metrics and per-variable PIT histograms; per-origin restricted NPZs retain PIT arrays. Group masks apply to joint scores only.", ""]
        for key in UNITS:
            lines += [f"{key} ({UNITS[key]})", ""]
            lines += table(["Mode", "Count", "MAE", "MSE", "CRPS", "50% cov.", "90% cov.", "50% width", "90% width", "Physical ES", "Physical variogram"],
                [[m, r["scores"]["variables"][key]["count"], *[pct(r["scores"]["variables"][key][k]) if k.startswith("coverage") else f(r["scores"]["variables"][key][k]) for k in METRICS],
                  f(r["scores"]["variables"][key]["energy_score_physical"]), f(r["scores"]["variables"][key]["variogram_p0_5_physical"]["score"])] for m, r in modes.items()])
        lines += ["**Saved embedding probe, unchanged fitted heads.** Filter on target-hour ref_t, not input-hour ref_t. Probe origins follow its original all-hour protocol, so denominators differ from the midnight forecast table. Compare raw+calendar (142 features) vs raw+calendar+GENCO (526); this is incremental association, not an embedding-only or causal test.", ""]
        lines += table(["Horizon", "Retained target hours", "Raw normalized MSE", "+ GENCO normalized MSE"],
            [[h, r[scope]["hours"], *[f(r[scope]["families"][family]["normalized"]["mse"]) for family in ("raw", "raw_plus_genco")]] for h, r in result["probe"].items()])
    p = result["parameter_hash_accounting"]
    lines += ["## Provenance correction and verification", "", p["correction"], "",
        f"**91,461 trainable parameters + 884 buffer elements = 92,345 adapter state-dict elements.** This accounting is static and does not change on the restricted time slice. See [parameter_hash_accounting.json](parameter_hash_accounting.json) for every buffer and exact paths.", "",
        f"Base checkpoint file SHA256: `{p['base_checkpoint_file_sha256']}`. Base named-tensor-bytes SHA256: `{p['base_state_tensor_bytes_sha256']}`. Selected adapter checkpoint file SHA256: `{p['adapter_checkpoint_file_sha256']}`.", "", p["tensor_hash_definition"], "",
        "Frozen source/input hashes were checked; original artifact hashes were checked before and after each origin's diagnostics. Additive contributions were checked against the original joint ES for every scored origin/mode/scope, and full probe normalized MSE was reproduced. No original report, checkpoint, model, dataset, or scorer was modified by this script. Historical reports are retained; the above provenance correction supersedes their discrepancy interpretation.", "",
        "Reproduce on an allocated CPU node in the existing Aurora environment: `python -m scripts.diagnose_stage_b`. Completed diagnostic origins are reused only after protocol/input/output hash checks. Full machine-readable output: [report.json](report.json), [protocol.json](protocol.json), and `origins/<id>/`. Diagnostics are exploratory and retain the original four-test-episode limitations; no calibration or speedup claim follows from these scores.", ""]
    (out / "report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "prepare", "worker", "merge"), nargs="?", default="run")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--shard", type=int, default=0)
    args = parser.parse_args()
    cfg = load_config()
    root, out = locations(cfg)
    torch.set_num_threads(1)
    if args.action == "prepare":
        prepare(cfg)
    elif args.action == "worker":
        worker(cfg, args.shard, args.workers)
    elif args.action == "merge":
        merge(cfg)
    else:
        started = time.time()
        prepare(cfg)
        processes, streams = [], []
        try:
            for shard in range(args.workers):
                path = out / f"worker_{shard:02d}.log"
                stream = path.open("a")
                streams.append(stream)
                processes.append(subprocess.Popen([sys.executable, "-u", "-m", "scripts.diagnose_stage_b", "worker",
                    "--shard", str(shard), "--workers", str(args.workers)], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT))
            codes = [p.wait() for p in processes]
            write_json(out / "job.json", {"job_id": os.environ.get("PBS_JOBID"), "node": platform.node(),
                "workers": args.workers, "exit_codes": codes, "wall_seconds": time.time() - started})
            if any(codes):
                raise RuntimeError(f"Diagnostic workers failed: {codes}; completed origins retained")
        finally:
            for p in processes:
                if p.poll() is None:
                    p.terminate()
            for stream in streams:
                stream.close()
        merge(cfg)


if __name__ == "__main__":
    main()
