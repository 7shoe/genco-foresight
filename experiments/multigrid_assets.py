"""Fetch pinned model assets and retain the source of every network file."""
from pathlib import Path
import shutil

from huggingface_hub import hf_hub_download
import requests

from power_fm_foresight._config import ROOT, sha256, write_json
from power_fm_foresight.grids import grid_config, registry


def prepare(case):
    cfg = grid_config(case)
    target = Path(cfg["paths"]["network_file"])
    target.parent.mkdir(parents=True, exist_ok=True)
    local = ROOT / "data/networks" / target.name
    rambo = Path(cfg["solver"]["reference_source"]) / "cases" / target.name
    revision = "dc6be4b2f85ca0e776952ec22cbd4c22396ea5a3"
    source = local if local.exists() else rambo
    provenance = str(source)
    if not target.exists():
        if source.exists():
            shutil.copyfile(source, target)
        else:
            provenance = f"https://raw.githubusercontent.com/power-grid-lib/pglib-opf/{revision}/{target.name}"
            response = requests.get(provenance, timeout=60)
            response.raise_for_status()
            target.write_bytes(response.content)
    root = Path(cfg["paths"]["assets_root"]) / "genco-opfdata"
    settings = cfg["checkpoint"]
    files = {str(target): sha256(target)}
    for key in ("weights", "normalizer", "configuration"):
        name = f"{settings['directory']}/{settings[key]}"
        dest = root / name
        if not dest.exists():
            hf_hub_download(settings["repository"], name, revision=settings["revision"], local_dir=root)
        files[str(dest)] = sha256(dest)
    out = Path(cfg["paths"]["training_run"]).parent
    write_json(out / "assets.json", dict(checkpoint=settings, files=files, network_source=provenance,
                                         network_license="PGLib BSD-3-Clause; see data/networks/LICENSE"))
    print(case, "assets ready", flush=True)


if __name__ == "__main__":
    for case in registry()["grids"]:
        prepare(case)
