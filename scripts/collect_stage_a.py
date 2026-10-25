"""Collect Stage A evidence without turning pending checks into passes."""
from pathlib import Path
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
import hashlib
import json
import zipfile
from importlib.metadata import distribution, distributions, version
from collections import Counter
import argparse

from common import ROOT, config_path, load_config, read_json, sha256, write_json


def snapshot_summary(evidence, config, root, temporal_root):
    quality = evidence["artifacts"]["quality"]
    manifest = read_json(temporal_root/"manifest.json")
    support = {}
    for split in ("train", "model_selection", "risk_calibration", "test"):
        episodes = [e for e in quality["episodes"] if e["split"] == split]
        support[split] = {"episodes": len(episodes), "forecast_days": sum(len(e["forecast_days"]) for e in episodes),
                          "unique_source_blocks": len(manifest["source_block_pools"][split]), "daily_events": {}}
        for family in ("thermal", "voltage"):
            counts = Counter(day["events"][family] for e in episodes for day in e["forecast_days"])
            support[split]["daily_events"][family] = {"positive": counts[True], "negative": counts[False], "unknown": counts[None]}
    generation = evidence["artifacts"]["generation"]
    summary = {"stage": "A", "complete": evidence["complete"], "gates": evidence["gates"],
        "source_pins": config["upstream"], "checkpoint": config["checkpoint"], "asset_hashes": config.get("asset_hashes", {}),
        "environment_packages": evidence["packages"], "base_environment_unchanged": evidence["gates"]["environment_unchanged"],
        "isolated_cli_packages": evidence["isolated_cli_packages"], "datakit_wheel_sha256": evidence["datakit_wheel_sha256"],
        "snapshot_parity": evidence["artifacts"]["parity"], "tests": evidence["artifacts"]["tests"],
        "xpu": evidence["artifacts"]["xpu"], "temporal": {
            "benchmark_id": manifest["benchmark_id"], "attempted": quality["attempted"], "valid": quality["valid"],
            "unresolved": quality["unresolved"], "valid_active_duals": quality["valid_duals"],
            "generation": manifest["generation"], "support_by_partition": support,
            "max_independent_errors": quality["max_independent_errors"],
            "thermal_stress_hours": generation["thermal_stress_hours"], "voltage_stress_hours": generation["voltage_stress_hours"],
            "pilot": {k:v for k,v in evidence["artifacts"]["pilot"].items() if k != "shards"},
            "pilot_prefix": quality["retained_pilot"], "generation_seconds": generation["this_invocation_seconds"]},
        "dual_check": evidence["artifacts"]["solver"]["dual_check"],
        "artifact_paths": {"report": str(root/"report.json"), "dataset": str(temporal_root),
                           "quality_report": str(temporal_root/"quality_report.json"), "parity": str(root/"parity")},
        "artifact_sha256": {"report": sha256(root/"report.json"), "dataset_manifest": sha256(temporal_root/"manifest.json"),
                            "quality": sha256(temporal_root/"quality_report.json")},
        "limitations": ["Case30, fixed topology/costs/commitment; synthetic ERCOT-to-IEEE profile mapping.",
            f"{support['test']['forecast_days']} test days from {support['test']['episodes']} episodes and {support['test']['unique_source_blocks']} source blocks; {support['risk_calibration']['forecast_days']} calibration days from {support['risk_calibration']['unique_source_blocks']} source blocks.",
            "Daily any-voltage-stress is always positive; it cannot establish discriminative daily calibration.",
            "No solver failures at the fixed budget: a numerical-difficulty classifier has no positive training examples.",
            "XPU forward/backward verified; no temporal adapter training or forecasting experiment has run.",
            "Plotly PNG export is unavailable in the isolated target; Matplotlib is the frozen PNG backend."]}
    write_json(Path(config["paths"].get("evidence_root", ROOT/"reports"))/"stage_a.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", action="store_true", help="Write compact repository evidence from the saved artifacts")
    args = parser.parse_args()
    config = load_config()
    root = Path(config["paths"].get("stage_a_run", Path(config["paths"]["runs_root"])/"stage_a"))
    root.mkdir(parents=True, exist_ok=True)
    after = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (root/"environment_after.txt").write_text(after)
    before_path = root/"environment_before.txt"
    before = before_path.read_text() if before_path.exists() else None
    packages = {name: version(name) for name in ("torch", "transformers", "torchvision", "torch-geometric",
        "torch-scatter", "numpy", "scipy", "pandas", "setuptools", "gridfm-graphkit", "gridfm-datakit")}
    datakit = distribution("gridfm-datakit")
    datakit_files = {name: sha256(datakit.locate_file(name)) for name in (
        "gridfm_datakit/network.py", "gridfm_datakit/perturbations/load_perturbation.py")}
    wheel = Path(config["paths"]["assets_root"])/"wheels/gridfm_datakit-1.1.0-py3-none-any.whl"
    wheel_matches = False
    if wheel.exists():
        with zipfile.ZipFile(wheel) as archive:
            wheel_matches = all(hashlib.sha256(archive.read(name)).hexdigest() == digest
                                for name,digest in datakit_files.items())
    graphkit_origin = json.loads(distribution("gridfm-graphkit").read_text("direct_url.json"))
    extra_path = Path(config["paths"]["assets_root"])/"cli_deps"
    extra = {d.metadata["Name"]: d.version for d in distributions(path=[str(extra_path)])} if extra_path.exists() else {}
    paths = {"parity": root/"parity/parity_report.json", "solver": root/"solver_smoke.json",
             "xpu": root/"xpu_report.json"}
    artifacts = {key: read_json(path) if path.exists() else None for key,path in paths.items()}
    test_path = root/"tests.xml"
    test_summary = None
    if test_path.exists():
        suite = ET.parse(test_path).getroot()
        if suite.tag == "testsuites":
            suite = suite.find("testsuite")
        test_summary = {key: int(suite.attrib.get(key, 0)) for key in ("tests", "failures", "errors", "skipped")}
    artifacts["tests"] = test_summary
    conversion_path = root/"parity/conversion_check.json"
    artifacts["conversion"] = read_json(conversion_path) if conversion_path.exists() else None
    temporal_root = Path(config["paths"].get("temporal_root", str(Path(config["paths"]["data_root"])/config["task"]["canonical_case"])))
    for key,name in (("pilot", "pilot_report.json"), ("generation", "generation_report.json")):
        path = temporal_root/name
        artifacts[key] = read_json(path) if path.exists() else None
    quality_path = temporal_root/"quality_report.json"
    artifacts["quality"] = read_json(quality_path) if quality_path.exists() else None
    manifest_path = temporal_root/"manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else None
    generation_source = Path(config.get('pilot', {}).get('reuse_labels', {}).get('source_snapshot', ROOT))
    data_version_matches = bool(manifest and manifest["generation"] == config["generation"] and
        manifest["solver_budget"] == config["solver"].get("budget", {}) and
        all(sha256(generation_source/name) == digest for name,digest in manifest["implementation_sha256"].items()))
    parity_protocol = root/"parity/protocol.json"
    parity_matches = bool(artifacts["parity"] and parity_protocol.exists() and
        artifacts["parity"]["protocol_sha256"] == sha256(parity_protocol))
    gates = {"environment_unchanged": before is not None and before == after,
        "cpu_parity": bool(artifacts["parity"] and artifacts["parity"].get("passed")),
        "parity_protocol_matches": parity_matches,
        "valid_numerical_reference": bool(artifacts["solver"] and artifacts["solver"].get("cold", {}).get("checks", {}).get("valid")),
        "graph_solver_round_trip": bool(artifacts["solver"] and artifacts["solver"].get("graph_round_trip", {}).get("checks", {}).get("valid")),
        "dual_units_sign": bool(artifacts["solver"] and artifacts["solver"].get("dual_check", {}).get("valid")),
        "pilot": bool(artifacts["pilot"] and artifacts["pilot"].get("accepted")),
        "full_10080_attempts": bool(artifacts["generation"] and artifacts["generation"].get("attempted") == 10080),
        "independent_conversion": bool(artifacts["conversion"] and artifacts["conversion"].get("passed")),
        "full_data_audit": bool(artifacts["quality"] and artifacts["quality"].get("passed") and
                                artifacts["quality"].get("test_forecast_days") == 72),
        "pilot_resume_prefix": bool(artifacts["quality"] and artifacts["quality"].get("pilot_prefix_verified")),
        "data_implementation_matches": data_version_matches,
        "datakit_source_verified": wheel_matches,
        "graphkit_pinned_commit": graphkit_origin.get("vcs_info", {}).get("commit_id") == config["upstream"]["graphkit"]["revision"],
        "tests": bool(test_summary and test_summary["tests"] > 0 and
                      test_summary["failures"] == test_summary["errors"] == test_summary["skipped"] == 0)}
    evidence = {"stage": "A", "complete": all(gates.values()), "gates": gates,
        "generation_source": str(generation_source),
        "audit_source_sha256": {name: sha256(ROOT/name) for name in ('solver/checks.py', 'scripts/data_audit.py', 'scripts/collect_stage_a.py')},
        "python": sys.version, "hostname": platform.node(), "packages": packages,
        "datakit_source_sha256": datakit_files, "isolated_cli_packages": extra,
        "datakit_wheel_sha256": sha256(wheel) if wheel.exists() else None,
        "graphkit_install_origin": graphkit_origin,
        "configuration_sha256": sha256(config_path()), "artifacts": artifacts,
        "paths": {"run": str(root), "data": str(temporal_root)},
        "config": config}
    write_json(root/"report.json", evidence)
    if args.snapshot:
        if not evidence["complete"]:
            raise RuntimeError("Stage A is incomplete; keep pending evidence in the external report")
        snapshot_summary(evidence, config, root, temporal_root)
    print({"complete": evidence["complete"], "gates": gates, "report": str(root/"report.json")}, flush=True)


if __name__ == "__main__":
    main()
