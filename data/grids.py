"""Resolve the six supported asset families without encoding network dimensions."""
from copy import deepcopy
from pathlib import Path

import yaml

from common import ROOT, load_config


def registry():
    return yaml.safe_load((ROOT / "configs/grids.yaml").read_text())


def check_case(expected, actual, source):
    """Reject missing or mismatched case identities before using an asset."""
    expected = expected.removeprefix("pglib_opf_")
    actual = actual.removeprefix("pglib_opf_") if isinstance(actual, str) else actual
    if actual != expected:
        raise ValueError(f"Case mismatch: {source} has {actual!r}; expected {expected!r}")


def grid_config(case, config=None):
    """Select a case and isolate all generated artifacts from the released runs."""
    catalog = registry()
    entry = catalog["grids"][case]
    cfg = deepcopy(config if config is not None else load_config(ROOT / "configs/v4.yaml"))
    cfg["grid"] = case
    cfg["task"]["canonical_case"] = Path(entry["pglib"]).stem
    assets = Path(cfg["paths"]["assets_root"])
    cfg["paths"]["network_file"] = str(assets / "networks" / entry["pglib"])
    cfg["paths"]["n1_cache"] = str(Path(cfg["paths"]["opfdata_root"]) / "processed/N1" / Path(entry["pglib"]).stem)
    cfg["paths"]["t1_cache"] = str(Path(cfg["paths"]["opfdata_root"]) / "processed/T1" / Path(entry["pglib"]).stem)
    cfg["paths"]["n1_splits"] = str(Path(cfg["paths"]["opfdata_root"]) / "processed/N1/splits" / Path(entry["pglib"]).stem)
    cfg["checkpoint"].update(repository=catalog["checkpoint_repository"], revision=catalog["checkpoint_revision"],
                             directory=f"{case}/seed42", configuration=entry["yaml"])
    cfg["paths"]["runs_root"] = str(assets.parent / "runs")
    run = Path(cfg["paths"]["runs_root"]) / "multigrid" / case
    cfg["paths"]["runs_root"] = str(run)
    cfg["paths"]["data_root"] = str(run / "datasets")
    for key, suffix in (("temporal_root", "data"), ("stage_a_run", "audit"), ("evidence_root", "audit"), ("training_run", "adapter")):
        cfg["paths"][key] = str(run / suffix)
    cfg["dataset"]["id"] = f"temporal_{case}_v4"
    cfg["stage_b"]["coordinate_scales"] = str(run / "audit/stage_a_followup/train_coordinate_scales.json")
    cfg["stage_b"]["calibration"]["selection"] = "calibration_support"
    for key in ("headline_voltage_bus_ids", "continuous_vm_bus_ids", "continuous_loading_line_ids", "secondary_thermal_daily_line_ids"):
        cfg["stage_b"]["calibration"].pop(key, None)
    return cfg


def network_path(config=None):
    cfg = config if config is not None else load_config()
    if "network_file" in cfg["paths"]:
        return Path(cfg["paths"]["network_file"])
    case = cfg["task"]["canonical_case"].removeprefix("pglib_opf_")
    name = registry()["grids"][case]["pglib"]
    local = ROOT / "data/networks" / name
    return local if local.exists() else Path(cfg["paths"]["assets_root"]) / "networks" / name
