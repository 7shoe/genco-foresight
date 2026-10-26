"""Predeclared v4 variant: identical calendar/noise, narrower nominal profile."""
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from power_fm_foresight._config import ROOT, config_path, load_config, read_json, sha256, write_json


def verify_variant(cfg):
    v3 = load_config(ROOT / "configs/v3.yaml")
    assert cfg["dataset"]["id"] == "temporal_case30_v4"
    assert v3["dataset"]["id"] == "temporal_case30_v3"
    assert cfg["task"] == v3["task"] and cfg["solver"] == v3["solver"]
    assert cfg["checkpoint"] == v3["checkpoint"] and cfg["upstream"] == v3["upstream"]
    assert cfg["dataset"]["source_blocks_one_based"] == v3["dataset"]["source_blocks_one_based"]
    assert cfg["generation"] == dict(v3["generation"], scale_low=.8, scale_high=1.)
    b = dict(cfg["stage_b"])
    b["loss"] = dict(b["loss"], power_balance_weight=.01)
    b["coordinate_scales"] = v3["stage_b"]["coordinate_scales"]
    assert b == v3["stage_b"], "Only the requested loss weight and variant scale path may differ"
    assert cfg["stage_b"]["loss"]["power_balance_weight"] == 1.
    for key in ("temporal_root", "training_run", "stage_a_run", "runs_root"):
        assert cfg["paths"][key] != v3["paths"][key]
        assert str(Path(cfg["paths"][key]).resolve()).startswith("/lus/flare/")
    return v3


def main():
    started = time.time()
    cfg = load_config()
    v3 = verify_variant(cfg)
    run = Path(cfg["paths"]["stage_a_run"])
    run.mkdir(parents=True, exist_ok=True)
    control = {str(p): sha256(p) for p in (
        ROOT / "configs/v3.yaml", Path(v3["paths"]["temporal_root"]) / "manifest.json",
        Path(v3["paths"]["training_run"]) / "chain.json", Path(v3["paths"]["training_run"]) / "best.pt",
        Path(v3["paths"]["runs_root"]) / "stage_b/protocol.json")}
    frozen = run / "config_frozen.yaml"
    if frozen.exists():
        assert sha256(frozen) == sha256(config_path())
    else:
        shutil.copy2(config_path(), frozen)
    parent = Path(v3["paths"]["stage_a_run"])
    for name in ("parity", "xpu_report.json"):
        if not (run / name).exists():
            (run / name).symlink_to(parent / name, target_is_directory=(parent / name).is_dir())
    if not (run / "environment_before.txt").exists():
        shutil.copy2(parent / "environment_before.txt", run / "environment_before.txt")
    write_json(run / "variant.json", {"label": "in-distribution + physics-weighted", "ablation": False,
        "changes": {"nominal_profile_scale": {"v3": [.3, 1.], "v4": [.8, 1.]},
                    "power_balance_weight": {"v3": .01, "v4": 1.}},
        "control": "v3 remains primary; same seed/calendar/noise but a new dataset and new training run",
        "created_after_v3_forecasting_scores": True, "v3_control_sha256": control,
        "support_qualification": "Nominal profile endpoints are inside [0.8,1.2]. Original held-out affine extrapolation, iid noise and unbounded lognormal AR(1) remain unchanged; actual bus multipliers are not guaranteed in support. No clipping or retuning.",
        "deadline_ct": "2026-10-24T18:00:00-05:00", "deadline_policy": "Ship v3; describe v4 as in progress if unfinished"})
    subprocess.run([sys.executable, "-m", "scripts.solver_smoke", "--config", str(config_path().resolve())], check=True, cwd=ROOT)
    from experiments.data_generation import prepare_benchmark, label
    from power_fm_foresight._data.generate import demands
    root, descriptor, net, _, scaled = prepare_benchmark(cfg)
    parent_manifest = read_json(Path(v3["paths"]["temporal_root"]) / "manifest.json")
    assert descriptor["episodes"] == parent_manifest["episodes"]
    assert descriptor["source_calendar"] == parent_manifest["source_calendar"]
    if not (root / "pilot_report.json").exists():
        label(cfg, maximum_hours=200)
    assert read_json(root / "pilot_report.json")["accepted"]
    label(cfg, maximum_hours=10080)
    from experiments.data_audit import audit
    from experiments.variant_events import run as events
    audit(cfg)
    events(cfg)
    active = net["bus"][:, 2] != 0
    support = {}
    for split in cfg["task"]["split_episodes"]:
        profiles, ratios = [], []
        for e in descriptor["episodes"]:
            if e["split"] != split:
                continue
            ref = scaled[e["source_start_hour"]:e["source_start_hour"] + 504]
            pd, _ = demands(ref, net["bus"][:, 2], net["bus"][:, 3], cfg["generation"], e["seed"])
            profiles.extend(ref)
            ratios.append(pd[:, active] / net["bus"][active, 2])
        ref, multipliers = np.asarray(profiles), np.concatenate(ratios)
        support[split] = {"profile_hours": len(ref), "profile_min": float(ref.min()), "profile_max": float(ref.max()),
            "profile_fraction_outside_0p8_1p2": float(((ref < .8) | (ref > 1.2)).mean()),
            "active_bus_hours": multipliers.size, "actual_bus_multiplier_min": float(multipliers.min()),
            "actual_bus_multiplier_max": float(multipliers.max()),
            "actual_bus_fraction_below_0p8": float((multipliers < .8).mean()),
            "actual_bus_fraction_above_1p2": float((multipliers > 1.2).mean())}
    write_json(run / "support_diagnostic.json", support)
    subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_data.py", "tests/test_model.py",
        "tests/test_solver.py", "tests/test_adapter.py", "tests/test_evidence.py", "tests/test_variants.py",
        "--basetemp", str(run / "pytest_tmp"), "-o", f"cache_dir={run / 'pytest_cache'}",
        "--junitxml", str(run / "tests.xml")], check=True, cwd=ROOT)
    subprocess.run([sys.executable, "-m", "scripts.collect_stage_a", "--snapshot"], check=True, cwd=ROOT)
    assert all(sha256(p) == h for p, h in control.items())
    evidence = read_json(run / "stage_a.json")
    assert evidence["complete"]
    # Replace historical summary boilerplate with measured variant availability.
    evidence["limitations"] = [
        "Exploratory in-distribution + physics-weighted variant; two changes after v3 test scores, not an ablation.",
        "Same case30 topology, synthetic profile, source calendar and four test episodes as v3.",
        "Actual noisy bus multipliers can leave nominal support; see support_diagnostic.json.",
        "No probability-calibration or training claim follows from this data audit."]
    write_json(run / "stage_a.json", evidence)
    write_json(run / "readiness.json", {"dataset_id": cfg["dataset"]["id"], "stage_a_audit_passed": True,
        "training_allowed": True, "prevalence_is_training_blocker": False, "training_started": False,
        "manifest_sha256": sha256(root / "manifest.json"), "stage_a_sha256": sha256(run / "stage_a.json"),
        "v3_control_sha256": control, "v3_unchanged": True, "wall_seconds": time.time() - started})
    (root / "README.md").write_text(
        "# temporal_case30_v4\n\nExploratory **in-distribution + physics-weighted** variant; v3 remains primary.\n\n"
        "Same 10/2/4/4 calendar-disjoint episode assignment, profile, seed, iid ±3% noise, lognormal AR(1) rho=0.9/sigma=0.02 as v3. "
        "Nominal train-affine endpoints change from 0.30–1.00 to 0.80–1.00; no clipping. Held-out extrapolation and noise can leave [0.8,1.2]. "
        "A_v4 additionally changes the balance weight from 0.01 to 1.0, so this is not a controlled ablation. "
        "Both changes were chosen after v3 forecasting scores; this is disclosed exploratory follow-up, not independent confirmation.\n\n"
        f"Stage A and support/prevalence evidence: `{run}`. Numerical labels, failed-solve masks, dual checks and hashes use the same audit as v3.\n")
    print({"v4_training_allowed": True, "support": support, "readiness": str(run / "readiness.json")}, flush=True)


if __name__ == "__main__":
    main()
