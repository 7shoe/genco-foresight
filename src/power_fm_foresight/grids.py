"""Resolve the six supported asset families without encoding network dimensions."""
from __future__ import annotations

from dataclasses import dataclass
from gridfm_datakit.network import Network
import numpy as np
from copy import deepcopy
from pathlib import Path

import yaml

from power_fm_foresight._config import ROOT, CONFIG_ROOT, RESOURCE_ROOT, load_config


def registry():
    return yaml.safe_load((RESOURCE_ROOT / "grids.yaml").read_text())


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
    cfg = deepcopy(config if config is not None else load_config(CONFIG_ROOT / "v4.yaml"))
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
    local = RESOURCE_ROOT / "networks" / name
    return local if local.exists() else Path(cfg["paths"]["assets_root"]) / "networks" / name


@dataclass
class GridSpec:
    """One network, its original row identities, and its online equipment masks."""
    path: Path
    network: Network
    baseMVA: float
    bus: np.ndarray
    gen: np.ndarray
    branch: np.ndarray
    cost: np.ndarray
    gen_bus: np.ndarray
    bus_ids: np.ndarray
    reference_bus: int
    Ybus: object
    Yf: object
    Yt: object
    case_sha256: str
    solver_reference_bus: int | None = None

    def __getitem__(self, key):
        return getattr(self, key)

    @property
    def case(self): return self.path.stem.removeprefix("pglib_opf_")

    @property
    def n_bus(self): return len(self.bus)

    @property
    def n_gen(self): return len(self.gen)

    @property
    def n_branch(self): return len(self.branch)

    @property
    def online_gen(self): return self.gen[:, 7] == 1

    @property
    def online_branch(self): return self.branch[:, 10] == 1

    @property
    def n_online_gen(self): return int(self.online_gen.sum())

    @property
    def n_online_branch(self): return int(self.online_branch.sum())

    @property
    def active_demand(self): return self.bus[:, 2] != 0

    @property
    def rated_branches(self): return self.online_branch & (self.branch[:, 5] > 0)

    @property
    def qg_buses(self): return np.array(list(dict.fromkeys(self.gen_bus[self.online_gen])), dtype=int)

    def describe(self):
        return dict(n_bus=self.n_bus, n_gen=self.n_gen, n_branch=self.n_branch,
                    online_gen=self.n_online_gen, online_branch=self.n_online_branch,
                    active_demand=int(self.active_demand.sum()), rated_branches=int(self.rated_branches.sum()),
                    reference_bus=int(self.bus_ids[self.reference_bus]), baseMVA=self.baseMVA,
                    gen_bus=self.gen_bus.tolist(), qg_buses=self.qg_buses.tolist(),
                    offline_gen=(~self.online_gen).tolist(), offline_branch=(~self.online_branch).tolist())
