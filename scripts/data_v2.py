"""Generate one predeclared v2 dataset and stop at the pre-training audit gate."""
from pathlib import Path
import shutil
import subprocess
import sys
import time

from common import ROOT, load_config, read_json, sha256, write_json


def prevalence_gate(report):
    """All frozen hourly/daily events must have a defined ratio at most 1.5."""
    from scripts.stage_b_prevalence import prevalence_shift
    rows = {(r["event"], r["resolution"], r["partition"]): r for r in report["prevalence"]}
    events = [f"voltage_bus_{i}" for i in (1, 9, 11, 13, 15, 21, 22)] + ["thermal_line_1"]
    checks = []
    for event in events:
        for resolution in ("hourly", "daily_any"):
            cal = rows[(event, resolution, "risk_calibration")]
            test = rows[(event, resolution, "test")]
            ratio, shifted = prevalence_shift(cal, test, 1.5)
            checks.append({"event": event, "resolution": resolution,
                "calibration_positive": cal["positive"], "calibration_total": cal["total"],
                "test_positive": test["positive"], "test_total": test["total"],
                "ratio": ratio, "passed": ratio is not None and not shifted})
    return {"passed": all(r["passed"] for r in checks), "threshold": 1.5,
            "definition": "Symmetric positive-prevalence ratio; hourly target hours and daily-any; exact rational comparison; zero/zero unsupported",
            "checks": checks, "failures": [r for r in checks if not r["passed"]]}


def prepare():
    """Freeze assignment before labels and preserve the original evidence files."""
    from data.synthetic.generate import prepare_benchmark
    cfg = load_config()
    run = Path(cfg["paths"]["stage_a_run"])
    if run.resolve().is_relative_to(Path("/home")) or run.resolve().is_relative_to(ROOT):
        raise ValueError("Runtime artifacts must be outside /home and the repository")
    run.mkdir(parents=True, exist_ok=True)
    prior = Path(cfg["paths"]["runs_root"])/"data_v2_provenance/config_v1.yaml"
    old = load_config(prior)
    assert cfg["generation"] == old["generation"], "Only source-pool assignment may change"
    assert cfg["solver"] == old["solver"] and cfg["task"] == old["task"]
    root, descriptor, _, _, _ = prepare_benchmark(cfg)
    frozen = run/"config_frozen.yaml"
    if frozen.exists() and sha256(frozen) != sha256(ROOT/"config.yaml"):
        raise ValueError("V2 configuration changed after the assignment was frozen")
    if not frozen.exists():
        shutil.copy2(ROOT/"config.yaml", frozen)
    reference = Path(cfg["paths"]["runs_root"])/"stage_a"
    inherited = {}
    for name in ("parity", "xpu_report.json"):
        target = run/name
        if not target.exists():
            target.symlink_to(reference/name, target_is_directory=(reference/name).is_dir())
        if not target.is_symlink() or target.resolve() != (reference/name).resolve():
            raise ValueError("Unexpected inherited evidence destination")
        inherited[name] = str(reference/name)
    if not (run/"environment_before.txt").exists():
        shutil.copy2(reference/"environment_before.txt", run/"environment_before.txt")
    record = {"dataset_id": cfg["dataset"]["id"], "benchmark_id": descriptor["benchmark_id"],
        "data_root": str(root), "configuration_sha256": sha256(frozen),
        "source_assignment": descriptor["dataset"], "source_calendar": descriptor["source_calendar"],
        "inherited_evidence": inherited,
        "inherited_evidence_reason": "Checkpoint, graph conversion, normalizer and device software unchanged; temporal data regenerated. Solver/dual smoke, full data audit and tests rerun.",
        "generation_law_unchanged_from_v1": True,
        "training_policy": "No training until all 16 calibration/test prevalence checks pass; no outcome-driven resampling."}
    if (run/"assignment_frozen.json").exists() and read_json(run/"assignment_frozen.json") != record:
        raise ValueError("Do not replace the predeclared assignment")
    write_json(run/"assignment_frozen.json", record)
    return cfg, run


def main():
    started = time.perf_counter()
    cfg, run = prepare()
    from scripts.solver_smoke import main as solver_smoke
    from data.synthetic.generate import label
    from scripts.data_audit import audit
    from scripts.stage_a_questions import main as events
    from scripts.stage_b_prevalence import main as prevalence
    solver_smoke()
    root = Path(cfg["paths"]["temporal_root"])
    if not (root/"pilot_report.json").exists():
        label(cfg, maximum_hours=200)
    if not read_json(root/"pilot_report.json")["accepted"]:
        raise RuntimeError("200-hour pilot failed; expansion blocked")
    label(cfg, maximum_hours=10080)
    audit(cfg)
    events()
    prevalence()
    subprocess.run([sys.executable, "-m", "pytest", "-q", "tests",
        "--basetemp", str(run/"pytest_tmp"), "-o", f"cache_dir={run/'pytest_cache'}",
        "--junitxml", str(run/"tests.xml")], check=True, cwd=ROOT)
    subprocess.run([sys.executable, "-m", "scripts.collect_stage_a", "--snapshot"], check=True, cwd=ROOT)
    prior = read_json(Path(cfg["paths"]["runs_root"])/"data_v2_provenance/v1_preserved.json")
    v1 = Path(prior["dataset_root"])
    assert sha256(v1/"manifest.json") == prior["manifest_sha256"]
    assert sha256(v1/"quality_report.json") == prior["quality_sha256"]
    for shard in read_json(v1/"manifest.json")["shards"]:
        assert sha256(v1/shard["file"]) == shard["sha256"]
    result = prevalence_gate(read_json(run/"stage_b_prevalence.json"))
    result.update({"dataset_id": cfg["dataset"]["id"],
        "benchmark_id": read_json(root/"manifest.json")["benchmark_id"],
        "manifest_sha256": sha256(root/"manifest.json"),
        "stage_a_report_sha256": sha256(run/"stage_a.json"),
        "events_sha256": sha256(run/"stage_a_followup/events.json"),
        "prevalence_sha256": sha256(run/"stage_b_prevalence.json"),
        "v1_manifest_audit_and_shards_unchanged": True,
        "training_allowed": result["passed"], "training_started": False,
        "wall_seconds": time.perf_counter()-started})
    write_json(run/"pretraining_gate.json", result)
    print({"data_v2_gate": "PASS" if result["passed"] else "FAIL",
           "failures": result["failures"], "report": str(run/"pretraining_gate.json")}, flush=True)


if __name__ == "__main__":
    main()
