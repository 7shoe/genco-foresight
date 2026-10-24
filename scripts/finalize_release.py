"""Add frozen v4 evidence to the local candidate and seal its payload hashes."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess

from common import ROOT, load_config, read_json, sha256, write_json


def check_frozen(runs):
    checks = {}
    for relative in ("stage_b/protocol.json", "adapter_v4/manifest.json", "v4/stage_b/protocol.json"):
        for name, digest in read_json(runs / relative)["source_sha256"].items():
            assert sha256(ROOT / name) == digest, name
            checks[name] = digest
    return checks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--add-v4", action="store_true")
    p.add_argument("--seal", action="store_true")
    a = p.parse_args()
    dest = a.bundle.resolve()
    assert str(dest).startswith("/lus/flare/"), "Bulk artifacts must stay on Lustre"
    cfg = load_config(ROOT / "config.yaml")
    runs = Path(cfg["paths"]["runs_root"])
    frozen = check_frozen(runs)
    previous = read_json(dest / "bundle.json")
    provenance = read_json(dest / "provenance.json")
    history = dest / "history/v3_only"
    if not history.exists():
        history.mkdir(parents=True)
        for name in ("bundle.json", "SHA256SUMS", "provenance.json"):
            shutil.copy2(dest / name, history / name)
        shutil.copy2(dest / "code/submission/post.md", history / "post.md")
        shutil.copytree(dest / "validation", history / "validation")

    def copy(source, relative):
        source, target = Path(source), dest / relative
        digest = sha256(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or sha256(target) != digest:
            shutil.copy2(source, target)
        assert sha256(target) == digest, relative
        provenance[str(relative)] = {"source": str(source), "source_sha256": digest}

    if a.add_v4:
        import torch
        from training.state import tensor_digest
        train = runs / "adapter_v4"
        completion = read_json(runs / "v4/stage_b/completion.json")
        assert completion["complete"]
        assert sha256(runs / "v4/stage_b/report.json") == completion["report_sha256"]
        assert sha256(train / "best.pt") == completion["checkpoint_sha256"]
        original = torch.load(train / "best.pt", map_location="cpu", weights_only=True)
        state = original["model"]
        assert original["progress"]["best_epoch"] == 14
        assert not any(name.startswith("backbone.") for name in state)
        assert sum(value.numel() for value in state.values()) == 92345
        model = dest / "model/adapter_v4"
        model.mkdir(parents=True, exist_ok=True)
        if not (model / "best.pt").exists():
            torch.save(state, model / "best.pt")
        exported = torch.load(model / "best.pt", map_location="cpu", weights_only=True)
        assert tensor_digest(exported) == tensor_digest(state)
        accounting = read_json(runs / "v4/stage_b/report.json")["provenance"]
        assert accounting["trainable_parameters"] == 91461
        for name in ("manifest.json", "chain.json", "chain.md", "epochs.json", "status.json", "evaluation_provenance.json"):
            copy(train / name, Path("model/adapter_v4") / name)
        copy(train / "resume_test/report.json", "model/adapter_v4/resume_test.json")
        for name in ("normalizer_stats.pt", "base_config.yaml", "base_reference.json"):
            copy(dest / "model/adapter_v3" / name, Path("model/adapter_v4") / name)
        write_json(model / "release_manifest.json", {
            "format": "adapter_only_plain_state_dict", "best_epoch": 14,
            "trainable_parameters": 91461, "buffer_elements": 884,
            "adapter_state_elements_including_buffers": 92345,
            "exported_adapter_file_sha256": sha256(model / "best.pt"),
            "original_training_checkpoint_file_sha256": sha256(train / "best.pt"),
            "adapter_named_tensor_bytes_sha256": tensor_digest(state),
            "base_checkpoint_file_sha256": accounting["base_checkpoint_file_sha256"],
            "base_named_tensor_bytes_sha256": accounting["base_named_tensor_bytes_sha256"],
            "tensor_hash_definition": "Sorted parameter/buffer name UTF-8 bytes followed by contiguous CPU NumPy tensor bytes.",
            "chain_log": "chain.md", "source_training_manifest": "manifest.json"})
        print("Adapter v4 exported and checked; copying frozen evaluation evidence.", flush=True)
        source = runs / "v4/stage_b"
        allowed = {"origins", "diagnostics", "forecaster", "controls"}
        count = 0
        for path in sorted(source.rglob("*")):
            rel = path.relative_to(source)
            if path.is_file() and path.suffix in (".json", ".md", ".npz") and (len(rel.parts) == 1 or rel.parts[0] in allowed):
                copy(path, Path("evidence/runs/v4/stage_b") / rel)
                count += 1
                if count % 500 == 0:
                    print(f"Copied {count} v4 evidence files.", flush=True)
        per_origin = {}
        for item in read_json(source / "protocol.json")["origins"]:
            directory = dest / "evidence/runs/v4/stage_b/origins" / item["id"]
            done = read_json(directory / "done.json")
            for name, digest in done["artifacts"].items():
                assert sha256(directory / name) == digest, (item["id"], name)
            per_origin[item["id"]] = {"done_sha256": sha256(directory / "done.json"), "artifacts": done["artifacts"]}
        write_json(dest / "model/adapter_v4/per_origin_hashes.json", per_origin)
        copy(load_config(ROOT / "configs/v4.yaml")["stage_b"]["coordinate_scales"], "small/v4/coordinate_scales.json")
        copy(source / "variogram_pairs.npz", "small/v4/variogram_pairs.npz")
        for version in ("v3", "v4"):
            copy(runs / f"stage_a_{version}/stage_b_prevalence.json", f"evidence/runs/stage_a_{version}/stage_b_prevalence.json")
        print(f"V4 evidence added: {len(per_origin)} origins, {count} files.", flush=True)

    if a.seal:
        names = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT).decode().split("\0")
        for name in sorted(set(names) - {""}):
            path = ROOT / name
            if path.is_file() and not path.is_symlink():
                copy(path, Path("code") / name)
        for source, target in (
            (ROOT / "reproducibility/README.md", "README.md"),
            (ROOT / "data/DATASET_CARD.md", "datasets/DATASET_CARD.md"),
            (ROOT / "reproducibility/MODEL_CARD.md", "model/README.md"),
            (ROOT / "reproducibility/MODEL_CARD.md", "model/adapter_v3/README.md"),
            (ROOT / "reproducibility/MODEL_CARD.md", "model/adapter_v4/README.md"),
        ):
            copy(source, target)
        for path in (runs / "stage_d/final_release").iterdir():
            if path.is_file() and path.suffix in (".json", ".md", ".png"):
                copy(path, Path("evidence/runs/stage_d/final_release") / path.name)
        # The checkout uses a Lustre symlink; the portable package needs real files.
        for path in (runs / "stage_d/final_release/figures").iterdir():
            if path.is_file():
                copy(path, Path("code/submission/figures") / path.name)
        write_json(dest / "provenance.json", provenance)
        excluded = {"bundle.json", "SHA256SUMS"}
        paths = [p for p in dest.rglob("*") if p.is_file()
                 and str(p.relative_to(dest)) not in excluded
                 and p.relative_to(dest).parts[0] != "validation"]
        hashes = {str(p.relative_to(dest)): sha256(p) for p in sorted(paths)}
        value = {**previous, "schema": 2,
            "label": "v0.1.0-rc local review package; intended source release v0.1.0",
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "post_sha256": sha256(ROOT / "submission/post.md"), "sha256": hashes,
            "frozen_source_sha256": frozen, "datasets": ["v3", "v4"],
            "all_v4_test_forecasts_included": True, "published": False,
            "limitations": ["CPU check rescores saved predictions; it does not train or run GENCO/Ipopt.",
                "Full pipeline retains site-specific paths and needs pinned upstream base weights and Julia.",
                "V1/v2 full episodes remain in original Lustre directories; summaries are included.",
                "No independent non-Aurora reproduction or NVIDIA execution is claimed.",
                "Validation outputs are outside the payload hash set to permit repeat checks."]}
        write_json(dest / "bundle.json", value)
        checksums = {**hashes, "bundle.json": sha256(dest / "bundle.json")}
        (dest / "SHA256SUMS").write_text("".join(f"{digest}  {name}\n" for name, digest in sorted(checksums.items())))
        print({"sealed_payload_files": len(hashes), "bundle_sha256": checksums["bundle.json"]}, flush=True)
    else:
        write_json(dest / "provenance.json", provenance)
        print("Artifact copy complete; run --seal after review documents are ready.", flush=True)


if __name__ == "__main__":
    main()
