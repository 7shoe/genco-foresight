"""Assemble a local, offline evidence directory; never publish or install anything."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path
import shutil
import subprocess

import torch

from common import ROOT, load_config, read_json, sha256, write_json
from training.state import tensor_digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); cfg = load_config(ROOT / "config.yaml")
    dest = a.output.resolve()
    if not str(dest).startswith("/lus/flare/"):
        raise ValueError("Package bulk artifacts on /lus/flare, not in the checkout")
    runs = Path(cfg["paths"]["runs_root"])
    protocol = read_json(runs / "stage_b/protocol.json")
    for name, expected in protocol["source_sha256"].items():
        assert sha256(ROOT / name) == expected, f"Frozen evaluation source changed: {name}"
    post_before = sha256(ROOT / "submission/post.md")
    for version in ("v3", "v4"):
        assert read_json(runs / f"stage_a_{version}/readiness.json")["training_allowed"]
    dest.mkdir(parents=True, exist_ok=False)
    provenance = {}

    def copy(source, relative):
        source, target = Path(source), dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        provenance[str(relative)] = {"source": str(source), "source_sha256": sha256(source)}
        assert sha256(target) == provenance[str(relative)]["source_sha256"]

    def tree(source, relative, suffixes=None):
        source = Path(source)
        for path in sorted(source.rglob("*")):
            if path.is_file() and (suffixes is None or path.suffix in suffixes):
                copy(path, Path(relative) / path.relative_to(source))

    # Source snapshot includes local implementation files not yet committed by the user.
    names = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT).decode().split("\0")
    for name in sorted(set(names) - {""}):
        path = ROOT / name
        if path.is_file() and not path.is_symlink():
            copy(path, Path("code") / name)
    copy(ROOT / "reproducibility/README.md", "README.md")
    copy(ROOT / "data/DATASET_CARD.md", "datasets/DATASET_CARD.md")
    copy(ROOT / "LICENSE", "LICENSE")
    copy(ROOT / "THIRD_PARTY_NOTICES.md", "THIRD_PARTY_NOTICES.md")

    for version in ("v3", "v4"):
        data = Path(cfg["paths"]["data_root"]) / f"temporal_case30_{version}"
        for name in ("manifest.json", "generation_report.json", "quality_report.json", "pilot_report.json"):
            copy(data / name, f"datasets/temporal_case30_{version}/{name}")
        tree(data / "episodes", f"datasets/temporal_case30_{version}/episodes")
        audit = runs / f"stage_a_{version}"
        for name in ("stage_a.json", "readiness.json", "stage_b_prevalence.json", "stage_b_prevalence.md"):
            copy(audit / name, f"datasets/audit_{version}/{name}")
        if (audit / "support_diagnostic.json").exists():
            copy(audit / "support_diagnostic.json", f"datasets/audit_{version}/support_diagnostic.json")
        tree(audit / "stage_a_followup", f"datasets/audit_{version}/followup", {".json", ".csv"})
    for version in ("v1", "v2"):
        data = Path(cfg["paths"]["data_root"]) / f"temporal_case30_{version}"
        for name in ("manifest.json", "quality_report.json", "generation_report.json"):
            copy(data / name, f"datasets/history/{version}/{name}")
    for name in ("eia_erco_demand_2025.csv", "eia_erco_demand_2025.json"):
        copy(Path(cfg["paths"]["assets_root"]) / "profiles" / name, f"datasets/profile/{name}")
    tree(ROOT / "data/networks", "datasets/network")

    train = Path(cfg["paths"]["training_run"])
    original = torch.load(train / "best.pt", map_location="cpu", weights_only=True)
    state = original["model"]
    assert not any(k.startswith("backbone.") for k in state)
    assert sum(v.numel() for v in state.values()) == 92345
    model = dest / "model/adapter_v3"; model.mkdir(parents=True)
    torch.save(state, model / "best.pt")
    diagnostic = read_json(runs / "stage_b/diagnostics/report.json")["parameter_hash_accounting"]
    assert tensor_digest(state) == diagnostic["adapter_state_tensor_bytes_sha256"]
    assert sha256(train / "best.pt") == diagnostic["adapter_checkpoint_file_sha256"]
    base = Path(cfg["paths"]["assets_root"]) / "genco-opfdata" / cfg["checkpoint"]["directory"]
    assert sha256(base / cfg["checkpoint"]["weights"]) == diagnostic["base_checkpoint_file_sha256"]
    copy(base / cfg["checkpoint"]["normalizer"], "model/adapter_v3/normalizer_stats.pt")
    copy(base / cfg["checkpoint"]["configuration"], "model/adapter_v3/base_config.yaml")
    for name in ("manifest.json", "manifest.training.json", "chain.json", "chain.md", "epochs.json", "status.json"):
        copy(train / name, "model/adapter_v3/" + name)
    copy(train / "resume_test/report.json", "model/adapter_v3/resume_test.json")
    copy(ROOT / "reproducibility/MODEL_CARD.md", "model/adapter_v3/README.md")
    write_json(model / "base_reference.json", {**cfg["checkpoint"],
        "base_checkpoint_file_sha256": diagnostic["base_checkpoint_file_sha256"],
        "base_named_tensor_bytes_sha256": diagnostic["base_state_tensor_bytes_sha256"],
        "normalizer_sha256": sha256(model / "normalizer_stats.pt"),
        "normalizer_refit": False, "base_weights_included": False})
    write_json(model / "release_manifest.json", {"format": "adapter_only_plain_state_dict",
        "best_epoch": original["progress"]["best_epoch"], "trainable_parameters": 91461,
        "adapter_state_elements_including_buffers": 92345,
        "exported_adapter_file_sha256": sha256(model / "best.pt"),
        "original_training_checkpoint_file_sha256": sha256(train / "best.pt"),
        "adapter_named_tensor_bytes_sha256": tensor_digest(state),
        "tensor_hash_definition": diagnostic["tensor_hash_definition"]})

    evidence = Path("evidence/runs")
    for stage in ("stage_b", "stage_c", "stage_d"):
        for path in (runs / stage).iterdir():
            if path.is_file() and path.suffix in (".json", ".md"):
                copy(path, evidence / stage / path.name)
    for relative in ("stage_b/origins", "stage_b/controls", "stage_b/diagnostics", "stage_b/probe", "stage_b/forecaster",
                     "stage_c/calibration/origins", "stage_c/hybrid/origins", "stage_d/figures"):
        tree(runs / relative, evidence / relative, {".json", ".md", ".npz", ".png", ".csv"})
    for relative in ("stage_b/controls/protocol.json", "stage_c/hybrid/protocol.json", "stage_c/hybrid/report.json",
                     "stage_c/hybrid/report.md", "stage_c/calibration/protocol.json"):
        copy(runs / relative, evidence / relative)
    tree(runs / "stage_c/hybrid_attempt1_failed", evidence / "stage_c/hybrid_attempt1_failed",
         {".json", ".md", ".py", ".log"})
    copy(cfg["stage_b"]["coordinate_scales"], "small/coordinate_scales.json")
    copy(runs / "stage_b/variogram_pairs.npz", "small/variogram_pairs.npz")

    # All method constants and v4 qualifications in the frozen post have saved sources.
    contract = {"config_v3": cfg, "config_v4": load_config(ROOT / "configs/v4.yaml"),
        "data_v3": read_json(dest / "datasets/temporal_case30_v3/manifest.json"),
        "v3_label_audit": read_json(dest / "datasets/temporal_case30_v3/quality_report.json"),
        "v3_event_audit": read_json(runs / "stage_a_v3/stage_a_followup/events.json"),
        "data_v4": read_json(dest / "datasets/temporal_case30_v4/manifest.json"),
        "v4_label_audit": read_json(dest / "datasets/temporal_case30_v4/quality_report.json"),
        "v4_event_support": read_json(dest / "datasets/audit_v4/stage_b_prevalence.json"),
        "v4_support": read_json(dest / "datasets/audit_v4/support_diagnostic.json"),
        "ridge": read_json(runs / "stage_b/forecaster/manifest.json"),
        "calibration_protocol": read_json(runs / "stage_c/protocol.json"),
        "post_sha256": post_before,
        "sources": {name: sha256(ROOT / name) for name in ("config.yaml", "configs/v4.yaml", "submission/post.md")}}
    write_json(dest / evidence / "post_contract.json", contract)
    write_json(dest / "environment.json", {"packages": {d.metadata["Name"]: d.version for d in distributions() if d.metadata["Name"]},
        "note": "Recorded environment only. No packages were installed or changed by packaging."})
    assert sha256(ROOT / "submission/post.md") == post_before
    write_json(dest / "provenance.json", provenance)
    hashes = {str(path.relative_to(dest)): sha256(path) for path in sorted(dest.rglob("*")) if path.is_file()}
    chosen = [0, 54]  # First midnight forecast of test_00 and test_03; fixed, not selected by score.
    write_json(dest / "bundle.json", {"schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "label": "v0.1.0-rc local package, not a Git tag or published release",
        "scorer_sha256": sha256(ROOT / "scoring.py"), "post_sha256": post_before,
        "sha256": hashes, "origins": [{"id": protocol["origins"][i]["id"], "index": i} for i in chosen],
        "all_v3_test_and_calibration_forecasts_included": True,
        "non_aurora_reproduction": "not run", "published": False,
        "limitations": ["Small check rescores saved predictions; it does not train or run GENCO/Ipopt.",
            "Full pipeline retains site-specific paths and needs the pinned upstream base weights and Julia runtime.",
            "V1/v2 summaries are included; full v1/v2 episodes remain in their original Lustre directories.",
            "V4 data is audited; v4 training artifacts are deliberately excluded while training continues."]})
    hashes["bundle.json"] = sha256(dest / "bundle.json")
    (dest / "SHA256SUMS").write_text("".join(f"{h}  {name}\n" for name, h in sorted(hashes.items())))
    print({"directory": str(dest), "files": len(hashes), "bytes": sum(p.stat().st_size for p in dest.rglob("*") if p.is_file()),
           "bundle_sha256": hashes["bundle.json"], "post_unchanged": True, "published": False}, flush=True)


if __name__ == "__main__":
    main()
