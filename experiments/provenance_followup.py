"""Allocated-node provenance audit; no package changes or temporal-model training."""
from pathlib import Path
import os
import platform
import sys

import torch
import torch_scatter

from power_fm_foresight._config import load_config, write_json
from power_fm_foresight._data.assets import checkpoint_assets
from power_fm_foresight._data.network import graph_for_demand, load_network
from power_fm_foresight.genco import Genco
from experiments.parity import evaluate


def libraries():
    return sorted(str(p) for p in torch.ops.loaded_libraries if "torch_scatter" in str(p))


def main():
    cfg = load_config()
    root = Path(cfg["paths"]["runs_root"])/"stage_a"
    report = {"hostname": platform.node(), "python": sys.executable,
              "loaded_after_import": libraries(), "xpu_available": torch.xpu.is_available()}
    if report["xpu_available"]:
        net = load_network()
        model = Genco(checkpoint_assets(cfg), cfg["checkpoint"], device="xpu")
        batch = model.batch([graph_for_demand(net, net["bus"][:, 2], net["bus"][:, 3])])
        model.model(batch)
        torch.xpu.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
            model.model(batch)
            torch.xpu.synchronize()
        names = sorted({event.key for event in trace.key_averages()})
        report["scatter_operator_names"] = [name for name in names if "scatter" in name]
        report["extension_operator_names"] = [name for name in names if name.startswith("torch_scatter::")]
        report["loaded_after_forward"] = libraries()
        report["mapped_shared_libraries"] = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                                                     if "torch_scatter" in line})
        report["native_scatter_add_seen"] = "aten::scatter_add_" in names
        report["xpu_forward_passed"] = True
    write_json(root/"scatter_provenance.json", report)
    print(report, flush=True)
    evaluate(cfg)


if __name__ == "__main__":
    main()
