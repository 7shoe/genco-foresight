from pathlib import Path

import pytest
import torch

from common import load_config
from data.download.prepare import checkpoint_assets
from data.loader.network import graph_for_demand, load_network
from data.loader.dataset import OfficialSnapshotDataset, SnapshotDataset, snapshot_graph
from models.genco import Genco


@pytest.fixture(scope="module")
def base():
    config = load_config()
    try:
        assets = checkpoint_assets(config)
    except FileNotFoundError:
        pytest.skip("Pinned checkpoint assets not downloaded; run data.download.prepare --download")
    return Genco(assets, config["checkpoint"])


def test_own_sample_demand_reaches_model_and_physics(base):
    net = load_network()
    pd, qd = net["bus"][:, 2], net["bus"][:, 3]
    a = graph_for_demand(net, pd, qd)
    b = graph_for_demand(net, pd*1.05, qd*1.05)
    batch = base.batch([a, b])
    assert not torch.equal(batch["bus"].x[:30, :2], batch["bus"].x[30:, :2])
    output = base.predict([a, b])
    assert all(torch.isfinite(v).all() for v in output.values())
    assert not torch.equal(output["bus"][:30], output["bus"][30:])
    assert torch.isnan(a["bus"].y[:, 3]).all()  # Unknown Vm is not a zero target.


def test_frozen_base_supports_input_gradients_without_weight_updates(base):
    net = load_network()
    graph = graph_for_demand(net, net["bus"][:, 2], net["bus"][:, 3])
    before = {k: v.clone() for k, v in base.model.state_dict().items()}
    batch = base.batch([graph])
    batch["bus"].x.requires_grad_(True)
    output = base.model(batch)
    output["bus"].square().mean().backward()
    assert batch["bus"].x.grad is not None
    assert torch.isfinite(batch["bus"].x.grad).all()
    assert all(p.grad is None and not p.requires_grad for p in base.model.parameters())
    assert all(torch.equal(before[k], v) for k, v in base.model.state_dict().items())


def test_base_serialization_round_trip(base, tmp_path):
    net = load_network()
    graph = graph_for_demand(net, net["bus"][:, 2], net["bus"][:, 3])
    expected = base.predict([graph])
    target = tmp_path/"state.pt"
    torch.save(base.model.state_dict(), target)
    state = torch.load(target, map_location="cpu", weights_only=True)
    base.model.load_state_dict(state, strict=True)
    actual = base.predict([graph])
    assert all(torch.equal(expected[k], actual[k]) for k in expected)


def test_native_embedding_api_preserves_predictions(base):
    net = load_network()
    graph = graph_for_demand(net, net["bus"][:, 2], net["bus"][:, 3])
    ordinary = base.predict([graph])
    output, embeddings, _ = base.predict([graph], return_embeddings=True)
    assert all(torch.equal(ordinary[k], output[k]) for k in ordinary)
    assert embeddings["bus"].shape[0] == 30
    assert embeddings["gen"].shape[0] == 6
    assert all(torch.isfinite(e).all() for e in embeddings.values())


def test_downloaded_snapshot_route_matches_local_conversion(base):
    config = load_config()
    root = Path(config["paths"]["assets_root"])/"official_case30"
    if not (root/"bus_data.parquet/scenario_partition=0").exists():
        pytest.skip("Download official reference partition 0 to check the portable reader")
    reference = OfficialSnapshotDataset(root)[0]
    local = snapshot_graph(SnapshotDataset(config["paths"]["n1_cache"])[0])
    expected, actual = base.predict([reference]), base.predict([local])
    for key in expected:
        torch.testing.assert_close(expected[key], actual[key], rtol=2e-5, atol=1e-4)
