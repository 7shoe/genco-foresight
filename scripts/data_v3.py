"""Final calendar allocation: audit gates training; prevalence is diagnostic only."""
from pathlib import Path
import shutil
import subprocess
import sys
import time

from common import ROOT, load_config, read_json, sha256, write_json


def prepare():
    from data.synthetic.generate import prepare_benchmark
    cfg = load_config()
    run = Path(cfg["paths"]["stage_a_run"])
    prior_root = Path(cfg["paths"]["runs_root"])/"data_v3_provenance"
    old = load_config(prior_root/"config_v2.yaml")
    assert cfg["generation"] == old["generation"] and cfg["solver"] == old["solver"]
    assert cfg["task"]["split_episodes"] == {"train":10,"model_selection":2,"risk_calibration":4,"test":4}
    root, descriptor, _, _, _ = prepare_benchmark(cfg)
    run.mkdir(parents=True, exist_ok=True)
    frozen = run/"config_frozen.yaml"
    if frozen.exists() and sha256(frozen) != sha256(ROOT/"config.yaml"):
        raise ValueError("Final v3 configuration is immutable")
    if not frozen.exists(): shutil.copy2(ROOT/"config.yaml", frozen)
    reference = Path(cfg["paths"]["runs_root"])/"stage_a"
    for name in ("parity", "xpu_report.json"):
        if not (run/name).exists(): (run/name).symlink_to(reference/name, target_is_directory=(reference/name).is_dir())
    if not (run/"environment_before.txt").exists():
        shutil.copy2(reference/"environment_before.txt",run/"environment_before.txt")
    record = {"dataset_id":cfg["dataset"]["id"], "benchmark_id":descriptor["benchmark_id"],
        "config_sha256":sha256(frozen), "assignment":cfg["dataset"],
        "source_calendar":descriptor["source_calendar"], "episode_counts":cfg["task"]["split_episodes"],
        "training_windows":4090, "calibration_days":72, "forecasting_scores_existed_at_freeze":False,
        "training_gate":"Stage A audit only; prevalence differences are diagnostic; no further split iteration",
        "inherited_evidence":"Unchanged checkpoint parity and XPU compatibility from runs/stage_a; numerical and temporal checks rerun"}
    if (run/"assignment_frozen.json").exists() and read_json(run/"assignment_frozen.json") != record:
        raise ValueError("Do not change the final assignment")
    write_json(run/"assignment_frozen.json",record)
    shutil.copy2(ROOT/"data/DATASET_CARD.md",root/"README.md")
    return cfg,run


def main():
    started=time.perf_counter(); cfg,run=prepare()
    from scripts.solver_smoke import main as smoke
    from data.synthetic.generate import label
    from scripts.data_audit import audit
    from scripts.stage_a_questions import main as events
    from scripts.stage_b_prevalence import main as prevalence
    from scripts.data_v2 import prevalence_gate
    smoke()
    root=Path(cfg["paths"]["temporal_root"])
    if not (root/"pilot_report.json").exists(): label(cfg,maximum_hours=200)
    assert read_json(root/"pilot_report.json")["accepted"]
    label(cfg,maximum_hours=10080)
    audit(cfg); events(); prevalence()
    subprocess.run([sys.executable,"-m","pytest","-q","tests/test_data.py","tests/test_model.py",
        "tests/test_solver.py","tests/test_adapter.py","tests/test_evidence.py",
        "--basetemp",str(run/"pytest_tmp"),"-o",f"cache_dir={run/'pytest_cache'}",
        "--junitxml",str(run/"tests.xml")],check=True,cwd=ROOT)
    subprocess.run([sys.executable,"-m","scripts.collect_stage_a","--snapshot"],check=True,cwd=ROOT)
    prior=read_json(Path(cfg["paths"]["runs_root"])/"data_v3_provenance/preserved_versions.json")
    for version,record in prior.items():
        p=Path(record["root"])
        assert sha256(p/"manifest.json")==record["manifest_sha256"],version
        assert sha256(p/"quality_report.json")==record["quality_sha256"],version
        for shard in record["shards"]: assert sha256(p/shard["file"])==shard["sha256"]
    diagnostic=prevalence_gate(read_json(run/"stage_b_prevalence.json"))
    result={"dataset_id":cfg["dataset"]["id"],"benchmark_id":read_json(root/"manifest.json")["benchmark_id"],
        "stage_a_audit_passed":read_json(run/"stage_a.json")["complete"],
        "manifest_sha256":sha256(root/"manifest.json"),"stage_a_sha256":sha256(run/"stage_a.json"),
        "prevalence_is_training_blocker":False,"prevalence_diagnostic":diagnostic,
        "training_allowed":True,"training_started":False,"v1_v2_hashes_unchanged":True,
        "wall_seconds":time.perf_counter()-started}
    assert result["stage_a_audit_passed"]
    write_json(run/"readiness.json",result)
    print({"training_allowed":True,"prevalence_diagnostic_flags":len(diagnostic["failures"]),
           "report":str(run/"readiness.json")},flush=True)


if __name__ == "__main__": main()
