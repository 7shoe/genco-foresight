"""Download only selected pinned assets, preserving their content hashes."""
from __future__ import annotations

import argparse
import requests
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

from huggingface_hub import HfApi, hf_hub_download

# workhorse loader
from common import (load_config,
                    read_json,
                    sha256,
                    write_json)


DATASET_REVISION = "64bfe542a7fd195abe0b4b6e0a82520b43a73569"
DATASET_REPOSITORY = "gridfm/opfdata_case30_ieee"


def checkpoint_assets(config, download=False):
    """
    Check if checkpoints are in if not download
    """
    
    # check if OK on your system
    settings = config["checkpoint"]
    root = Path(config["paths"]["assets_root"]) / "genco-opfdata"
    previous = read_json(root / "manifest.json") if (root / "manifest.json").exists() else None
    
    # validity check
    if previous and (previous["repository"], previous["revision"]) != (settings["repository"], settings["revision"]):
        raise ValueError("Checkpoint asset directory belongs to another repository/revision")

    # paths
    names = [f"{settings['directory']}/{settings[key]}"
             for key in ("weights", "normalizer", "configuration")]
    names += [f"splits/{part}.pt" for part in ("train", "val", "test", "shuffled_indices")]
    manifest = {"repository": settings["repository"], "revision": settings["revision"], "files": {}}
    for name in names:
        path = root / name
        if not path.is_file():
            if not download:
                raise FileNotFoundError(f"Missing {path}; run python -m data.download.prepare --download")
            hf_hub_download(settings["repository"], name, revision=settings["revision"], local_dir=root)
        entry = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        if previous and name in previous["files"] and entry != previous["files"][name]:
            raise ValueError(f"Pinned checkpoint asset changed: {path}")
        manifest["files"][name] = entry
    
    # write log
    write_json(root / "manifest.json", manifest)
    
    return root


def converted_partitions(partitions, destination, download=False):
    """Prepare selected OPFData partitions w/o downloadig the full dataset"""
    
    # check path on your system
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    inventory_path = root / "hf_inventory.json"
    
    # cache the file listing to avoid repeating the API request
    if inventory_path.exists():
        inventory = read_json(inventory_path)
    else:
        info = HfApi().dataset_info(DATASET_REPOSITORY, revision=DATASET_REVISION, files_metadata=True)
        inventory = {"revision": info.sha,
                     "files": [{"name": f.rfilename, "bytes": f.size} for f in info.siblings]}
        write_json(inventory_path, inventory)
    selected = []
    prefixes = [f"{table}_data.parquet/scenario_partition={int(part)}/"
                for table in ("bus", "gen", "branch") for part in partitions]
    for record in inventory["files"]:
        name = record["name"]
        if not any(name.startswith(prefix) for prefix in prefixes):
            continue
        path = root / name
        if not path.is_file():
            if not download:
                raise FileNotFoundError(f"Missing reference parquet {path}; pass --download")
            hf_hub_download(DATASET_REPOSITORY, name, repo_type="dataset", revision=DATASET_REVISION,
                            local_dir=root)
        selected.append({"path": name, "sha256": sha256(path), "bytes": path.stat().st_size})
        
    # each partition needs one bus, generator & branch file
    if len(selected) != 3 * len(set(partitions)):
        raise ValueError("Expected one parquet per requested partition and table")
    
    # write manifsest
    write_json(root / "download_manifest.json", {"repository": DATASET_REPOSITORY,
               "revision": DATASET_REVISION, "files": selected})

    return root


def eia_profile(config, download=False):
    """Hourly ERCO demand for 2025 from the public EIA v2 API, in UTC."""
    
    # path for temporal data
    root = Path(config["paths"]["assets_root"]) / "profiles"
    root.mkdir(parents=True, exist_ok=True)
    
    # key CSV to simulate synthetic temporal evolution
    target = root / "eia_erco_demand_2025.csv"
    provenance = root / "eia_erco_demand_2025.json"
    expected_hash = config.get("asset_hashes", {}).get("ercot_2025_csv")
    if target.exists() and provenance.exists():
        actual_hash = sha256(target)
        if actual_hash != read_json(provenance)["sha256"] or (expected_hash and actual_hash != expected_hash):
            raise ValueError("Profile checksum changed")
        return target
    if not download:
        raise FileNotFoundError(f"Missing {target}; pass --download")
    url = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
    rows = []
    for offset in (0, 5000):
        params = {"api_key": "DEMO_KEY", "frequency": "hourly", "data[0]": "value",
                  "facets[respondent][]": "ERCO", "facets[type][]": "D",
                  "start": "2025-01-01T00", "end": "2025-12-31T23",
                  "sort[0][column]": "period", "sort[0][direction]": "asc",
                  "length": 5000, "offset": offset}
        response = requests.get(url, params=params, timeout=90)
        response.raise_for_status()
        payload = response.json()
        if "response" not in payload:
            raise RuntimeError(f"EIA response missing data: {payload}")
        write_json(root / f"eia_erco_2025_page_{offset}.json", payload)
        rows.extend(payload["response"]["data"])
    frame = pd.DataFrame(rows)
    frame["time_utc"] = pd.to_datetime(frame["period"], utc=True)
    frame["demand_mw"] = pd.to_numeric(frame["value"], errors="raise")
    frame = frame.sort_values("time_utc")[["time_utc", "demand_mw"]]
    expected = pd.date_range("2025-01-01", periods=8760, freq="h", tz="UTC")
    if not pd.DatetimeIndex(frame.time_utc).equals(expected) or not (frame.demand_mw > 0).all():
        raise ValueError("EIA profile must have all 8760 unique, positive hourly UTC observations")
    frame.to_csv(target, index=False)
    if expected_hash and sha256(target) != expected_hash:
        raise ValueError("EIA source changed since this benchmark was frozen; use the archived profile matching asset_hashes.ercot_2025_csv")
    write_json(provenance, {"source": url, "respondent": "ERCO", "type": "D",
               "frequency": "hourly", "timezone": "UTC", "year": 2025, "rows": len(frame),
               "retrieved_utc": datetime.now(timezone.utc).isoformat(), "sha256": sha256(target),
               "assumption": "ERCOT aggregate demand drives synthetic IEEE case30 bus loads"})
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v3.yaml", help="Currently, v3 is the default")
    parser.add_argument("--download", action="store_true", help="Explicitly allow missing-asset downloads")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--reference-partitions", nargs="*", type=int,
                        help="Download specified 200-scenario official OPFData partitions")
    args = parser.parse_args()
    config = load_config(args.config)
    print("Checkpoint assets:", checkpoint_assets(config, args.download), flush=True)
    
    # check validity
    if args.profile:
        print("Profile:", eia_profile(config, args.download), flush=True)
    if args.reference_partitions is not None:
        print("Reference data:", converted_partitions(args.reference_partitions,
              Path(config["paths"]["assets_root"])/"official_case30", args.download), flush=True)


if __name__ == "__main__":
    main()
