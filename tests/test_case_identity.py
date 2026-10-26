"""Reject mixed-grid assets before loading weights or running a model."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from power_fm_foresight._config import ROOT, load_config, write_json
from power_fm_foresight.grids import grid_config
from power_fm_foresight._data.dataset import OfficialSnapshotDataset, load_snapshot, load_temporal_manifest
from power_fm_foresight._data.network import graph_for_demand, load_network
from power_fm_foresight.genco import Genco
from power_fm_foresight.temporal import TemporalAdapter


@pytest.fixture(scope="module")
def grid():
    return load_network(config=grid_config("case30_ieee"))


def test_network_file_must_match_selected_case(tmp_path):
    cfg = grid_config("case30_ieee")
    with pytest.raises(ValueError, match="Case mismatch: network file"):
        load_network(tmp_path / "pglib_opf_case14_ieee.m", config=cfg)


def test_checkpoint_directory_must_match_network():
    cfg = grid_config("case30_ieee")
    cfg["checkpoint"]["directory"] = "case14_ieee/seed42"
    with pytest.raises(ValueError, match="Case mismatch: checkpoint directory"):
        load_network(config=cfg)


def test_renamed_network_file_does_not_hide_its_case(tmp_path, grid):
    path = tmp_path / "pglib_opf_case14_ieee.m"
    path.write_bytes(grid.path.read_bytes())
    with pytest.raises(ValueError, match="Case mismatch: parsed network"):
        load_network(path)


def test_wrong_checkpoint_yaml_fails_before_loading_weights(tmp_path, monkeypatch):
    cfg = grid_config("case30_ieee")
    settings = cfg["checkpoint"]
    source = Path(cfg["paths"]["assets_root"]) / "genco-opfdata" / settings["directory"] / settings["configuration"]
    config = yaml.safe_load(source.read_text())
    config["data"]["networks"] = ["case14_ieee"]
    path = tmp_path / settings["directory"] / settings["configuration"]
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr("power_fm_foresight.genco.torch.load", lambda *a, **k: pytest.fail("Weights loaded before validation"))
    with pytest.raises(ValueError, match="Case mismatch: checkpoint YAML"):
        Genco(tmp_path, settings)


def test_checkpoint_yaml_must_match_supplied_network(grid, monkeypatch):
    cfg = grid_config("case14_ieee")
    monkeypatch.setattr("power_fm_foresight.genco.torch.load", lambda *a, **k: pytest.fail("Weights loaded before validation"))
    with pytest.raises(ValueError, match="Case mismatch: checkpoint YAML"):
        Genco(Path(cfg["paths"]["assets_root"]) / "genco-opfdata", cfg["checkpoint"], grid=grid)


@pytest.mark.parametrize("case", ["case14_ieee", None])
def test_wrong_or_unidentified_graph_is_rejected(grid, case):
    base = object.__new__(Genco)
    base.case = "case30_ieee"
    graph = graph_for_demand(grid, grid.bus[:, 2], grid.bus[:, 3])
    if case is None:
        del graph.case_id
    else:
        graph.case_id = case
    with pytest.raises(ValueError, match="Case mismatch: input graph"):
        base.prepare(graph)


def test_adapter_cannot_wrap_another_grids_backbone(grid):
    with pytest.raises(ValueError, match="Case mismatch: adapter backbone"):
        TemporalAdapter(SimpleNamespace(case="case14_ieee"), grid, {})


def test_snapshot_entrypoint_rejects_wrong_dataset(tmp_path):
    cfg = grid_config("case30_ieee")
    cfg["paths"]["n1_cache"] = str(tmp_path)
    write_json(tmp_path / "manifest.json", dict(format="lumina-t1-data-v1", case="pglib_opf_case14_ieee",
                                               count=0, groups=[], group_size=1))
    with pytest.raises(ValueError, match="Case mismatch: snapshot dataset"):
        load_snapshot(0, cfg)


def test_official_snapshot_manifest_must_match_network(tmp_path, grid):
    write_json(tmp_path / "download_manifest.json", {"repository": "gridfm/opfdata_case14_ieee"})
    with pytest.raises(ValueError, match="Case mismatch: official snapshot dataset"):
        OfficialSnapshotDataset(tmp_path, grid=grid)


@pytest.mark.parametrize("manifest", [{}, {"case_sha256": "wrong"},
    {"case_sha256": "correct", "case": "case14_ieee"}])
def test_temporal_manifest_rejects_missing_or_wrong_identity(tmp_path, grid, manifest):
    manifest = dict(manifest)
    if manifest.get("case_sha256") == "correct":
        manifest["case_sha256"] = grid.case_sha256
    write_json(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="mismatch"):
        load_temporal_manifest(tmp_path, grid)


@pytest.mark.parametrize("version", ["v3", "v4"])
def test_released_temporal_manifests_remain_valid(version):
    cfg = load_config(ROOT / f"configs/{version}.yaml")
    grid = load_network(config=cfg)
    manifest = load_temporal_manifest(cfg["paths"]["temporal_root"], grid)
    assert manifest["case_sha256"] == grid.case_sha256


@pytest.mark.parametrize("entrypoint", ["training", "evaluation"])
def test_run_setup_rejects_wrong_temporal_data_before_model(tmp_path, monkeypatch, entrypoint):
    from power_fm_foresight._training import loop
    from power_fm_foresight._training import forecast as forecast_partition
    module, call = (loop, loop.initialize) if entrypoint == "training" else (forecast_partition, forecast_partition.components)
    cfg = grid_config("case30_ieee")
    cfg["paths"]["temporal_root"] = str(tmp_path)
    write_json(tmp_path / "manifest.json", {"case_sha256": "wrong"})
    monkeypatch.setattr(module, "Genco", lambda *a, **k: pytest.fail("Model loaded before data validation"))
    with pytest.raises(ValueError, match="Dataset/network mismatch"):
        call(cfg, "cpu")
