"""Optional allocated-node XPU test; has no role in the CPU/data acceptance gate."""
import time

import torch
import torch_scatter
from torch_geometric.nn import TransformerConv

from power_fm_foresight._config import load_config, write_json
from power_fm_foresight._data.assets import checkpoint_assets
from power_fm_foresight._data.network import graph_for_demand, load_network
from power_fm_foresight.genco import Genco
from pathlib import Path


def main():
    config = load_config()
    report = {"device": "xpu", "available": torch.xpu.is_available(), "status": "pending"}
    path = Path(config["paths"]["runs_root"])/"stage_a/xpu_report.json"
    if not report["available"]:
        write_json(path, report)
        return
    started = time.perf_counter()
    try:
        x = torch.randn(10, 4, device="xpu", requires_grad=True)
        index = torch.tensor([0,1,1,2,2,2,3,3,3,3], device="xpu")
        torch_scatter.scatter_add(x, index, dim=0, dim_size=5).sum().backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x))
        report["scatter_forward_backward"] = True
        x = x.detach().requires_grad_(True)
        conv = TransformerConv(4, 8, heads=2).to("xpu")
        conv(x, torch.stack([index, index.flip(0)])).square().mean().backward()
        assert torch.isfinite(x.grad).all()
        report["transformer_forward_backward"] = True
        net = load_network()
        graph = graph_for_demand(net, net["bus"][:, 2], net["bus"][:, 3])
        model = Genco(checkpoint_assets(config), config["checkpoint"], device="xpu")
        batch = model.batch([graph])
        batch["bus"].x.requires_grad_(True)
        output = model.model(batch)
        output["bus"].square().mean().backward()
        assert torch.isfinite(batch["bus"].x.grad).all()
        report["genco_forward_backward"] = True
        report["status"] = "forward_backward_verified"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    report["elapsed_seconds"] = time.perf_counter()-started
    write_json(path, report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
