"""Check review-document provenance and optionally every portable payload hash."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import re

from power_fm_foresight._config import ROOT, load_config, read_json, sha256, write_json
from experiments.finalize_release import check_frozen
from experiments.headline_numbers import collect, markdown_list


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path)
    p.add_argument("--payload", action="store_true")
    a = p.parse_args()
    runs = Path(load_config(ROOT / "configs/v3.yaml")["paths"]["runs_root"])
    final = runs / "stage_d/final_release"
    context = read_json(final / "context.json")
    for name, digest in context["protected_source_sha256"].items():
        assert sha256(ROOT / name) == digest, name
    frozen = check_frozen(runs)
    records = collect(runs)["headlines"]
    post = (ROOT / "submission/post.md").read_text()
    keys = set(re.findall(r"<!-- headline: ([\w.]+)(?:\[\d+\])? -->", post))
    claims = read_json(final / "post_claims.json")
    assert keys == set(claims["keys"]), keys.symmetric_difference(claims["keys"])
    assert not (keys - records.keys())
    assert sha256(ROOT / "submission/post.md") == claims["post_sha256"]
    assert (ROOT / "submission/headline_numbers.md").read_text() == markdown_list(records)
    assert not re.search(r"\b(genuinely|robust|novel)\b", post, flags=re.I)
    figures = read_json(final / "figures/manifest.json")
    for item in figures.values():
        assert sha256(final / "figures" / item["filename"]) == item["sha256"]
        assert item["sha256"][:12] in item["filename"]
    figure_refs = re.findall(r"\]\(figures/([^\)]+\.png)\)", post)
    assert set(figure_refs) == {x["filename"] for x in figures.values()}
    assert sha256(final / "figures/manifest.json") == claims["figure_manifest_sha256"]
    for target in re.findall(r"\]\(([^\)]+)\)", post):
        assert (ROOT / "submission" / target).exists(), target
    payload_count = 0
    if a.bundle:
        manifest = read_json(a.bundle / "bundle.json")
        assert manifest["post_sha256"] == claims["post_sha256"]
        assert sha256(a.bundle / "code/submission/post.md") == claims["post_sha256"]
        if a.payload:
            for name, digest in manifest["sha256"].items():
                assert sha256(a.bundle / name) == digest, name
                payload_count += 1
            for line in (a.bundle / "SHA256SUMS").read_text().splitlines():
                digest, name = line.split("  ", 1)
                expected = sha256(a.bundle / name) if name == "bundle.json" else manifest["sha256"][name]
                assert digest == expected, name
        for version in ("v3", "v4"):
            path = a.bundle / f"model/adapter_{version}"
            m = read_json(path / "release_manifest.json")
            assert sha256(path / "best.pt") == m["exported_adapter_file_sha256"]
            assert m["trainable_parameters"] == 91461
            assert m.get("buffer_elements", m["adapter_state_elements_including_buffers"] - m["trainable_parameters"]) == 884
        per_origin = read_json(a.bundle / "model/adapter_v4/per_origin_hashes.json")
        assert len(per_origin) == 72
    result = {"passed": True, "checked_utc": datetime.now(timezone.utc).isoformat(),
        "post_sha256": claims["post_sha256"], "headline_keys": len(records),
        "post_keys": len(keys), "figures": len(figures), "payload_files_checked": payload_count,
        "frozen_evaluation_files": len(frozen), "protected_sources": len(context["protected_source_sha256"]),
        "scope": "Artifact/document validation only; no experiment, training or publication."}
    output = a.bundle / "validation/final_release.json" if a.bundle else final / "review_check.json"
    write_json(output, result)
    print(result)


if __name__ == "__main__":
    main()
