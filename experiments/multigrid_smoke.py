"""CPU loading and format gates; no solves, backward passes, or training."""
import argparse
from pathlib import Path
import platform
import time
import traceback

import numpy as np
import pandas as pd
import torch

from power_fm_foresight._config import ROOT, load_config, read_json, sha256, write_json
from power_fm_foresight.grids import grid_config, registry
from power_fm_foresight._data.dataset import SnapshotDataset, snapshot_graph
from power_fm_foresight._data.network import load_network, graph_for_demand
from power_fm_foresight._data.generate import demands
from power_fm_foresight.genco import Genco


def run(case):
    cfg = grid_config(case)
    out = Path(cfg["paths"]["training_run"]).parent
    report = dict(case=case, status="blocked", node=platform.node(), device="cpu",
                  scope="CPU loading and data-format support; solver/training performance unmeasured",
                  step_seconds=None, peak_device_memory_bytes=None, seconds_per_ipopt_solve=None)
    started = time.perf_counter()
    try:
        grid = load_network(config=cfg)
        report["dims"] = grid.describe()
        base = Genco(Path(cfg["paths"]["assets_root"])/"genco-opfdata", cfg["checkpoint"], "cpu", grid)
        report["strict_checkpoint_load"] = True
        released = load_config(ROOT/"configs/v4.yaml")
        manifest = read_json(Path(released["paths"]["temporal_root"])/"manifest.json")
        profile = pd.read_csv(Path(cfg["paths"]["assets_root"])/"profiles/eia_erco_demand_2025.csv")
        low, high = manifest["train_profile_min"], manifest["train_profile_max"]
        refs = cfg["generation"]["scale_low"]+(profile.demand_mw.to_numpy()-low)*(cfg["generation"]["scale_high"]-cfg["generation"]["scale_low"])/(high-low)
        ep = manifest["episodes"][0]; lo = ep["source_start_hour"]
        pd_values, qd_values = demands(refs[lo:lo+200], grid.bus[:,2], grid.bus[:,3], cfg["generation"], ep["seed"])
        assert pd_values.shape == qd_values.shape == (200,grid.n_bus)
        assert np.isfinite(pd_values).all() and np.isfinite(qd_values).all()
        active = grid.active_demand
        np.testing.assert_allclose(qd_values[:,active], pd_values[:,active]*(grid.bus[active,3]/grid.bus[active,2]))
        np.testing.assert_array_equal(pd_values[:,~active], np.zeros((len(pd_values),int((~active).sum()))))
        raw = graph_for_demand(grid, pd_values[0], qd_values[0])
        prepared = base.prepare(raw)
        assert prepared["bus"].num_nodes == grid.n_bus
        assert prepared["gen"].num_nodes == grid.n_online_gen
        assert prepared["bus","connects","bus"].num_edges == 2*grid.n_online_branch
        assert all(torch.isfinite(value).all() for value in prepared.x_dict.values())
        report["graph_shapes"] = {key:list(value.shape) for key,value in prepared.x_dict.items()}
        cache = SnapshotDataset(cfg["paths"]["n1_cache"],grid=grid)
        sample = snapshot_graph(cache[0])
        converted = base.prepare(sample)
        assert converted["bus"].num_nodes == grid.n_bus
        assert all(torch.isfinite(value).all() for value in converted.x_dict.values())
        report["opfdata"] = dict(root=cfg["paths"]["n1_cache"], samples=len(cache), checked_sample_id=0,
                                  bus_nodes=converted["bus"].num_nodes, generator_nodes=converted["gen"].num_nodes,
                                  manifest_sha256=sha256(cache.folder/"manifest.json"))
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out/"demand_format_smoke.npz", pd_mw=pd_values, qd_mvar=qd_values,
                 state_available=np.zeros(len(pd_values),bool))
        report.update(status="supported", input_hours=200, labels_generated=False,
                      demand_sha256=sha256(out/"demand_format_smoke.npz"), reference_bus_from_network=True,
                      bus_qg_coordinates=len(grid.qg_buses), negative_demand_buses=int((grid.bus[:,2]<0).sum()))
    except Exception as error:
        report.update(reason=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    report["cpu_check_seconds"] = time.perf_counter()-started
    write_json(out/"gate.json", report)
    print(case, report["status"], report.get("reason", ""), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=list(registry()["grids"]))
    args = parser.parse_args()
    for case in ([args.case] if args.case else registry()["grids"]): run(case)
