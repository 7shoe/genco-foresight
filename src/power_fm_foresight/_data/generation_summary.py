"""Original label-completion summary."""
from pathlib import Path
import numpy as np
from power_fm_foresight._config import read_json, sha256, write_json

def audit(root):
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    previous = {r["file"]: r for r in manifest.get("shards", [])}
    records, shards = [], []
    for episode in manifest["episodes"]:
        path = root / "episodes" / f"{episode['episode_id']}.json"
        if not path.exists():
            continue
        shard = read_json(path)
        if shard["benchmark_id"] != manifest["benchmark_id"]:
            raise ValueError("Shard provenance mismatch")
        if [r["hour"] for r in shard["records"]] != list(range(len(shard["records"]))):
            raise ValueError("Noncontiguous/duplicate hour IDs")
        for r in shard["records"]:
            if r["state_available"] != (r["target_state"] is not None):
                raise ValueError("Missing labels must remain masked, not zero substitutes")
        records.extend(shard["records"])
        name = str(path.relative_to(root))
        digest = sha256(path)
        if name in previous and previous[name]["hours"] == len(shard["records"]) and previous[name]["sha256"] != digest:
            raise ValueError(f"Completed shard checksum changed: {path}")
        shards.append({"file": name, "hours": len(shard["records"]), "sha256": digest})
    valid = [r for r in records if r["state_available"]]
    report = {"benchmark_id": manifest["benchmark_id"], "attempted": len(records), "valid": len(valid),
        "unresolved": len(records)-len(valid), "valid_duals": sum(r["duals_available"] for r in records),
        "thermal_stress_hours": sum(r["solver"]["checks"]["thermal_stress"] for r in valid),
        "voltage_stress_hours": sum(r["solver"]["checks"]["voltage_stress"] for r in valid),
        "solver_request_seconds": sum(r["solver"]["request_seconds"] for r in records),
        "hashes_verified": True, "shards": shards}
    manifest["shards"] = shards
    write_json(root / "manifest.json", manifest)
    return report
