"""Official Graphkit and local CPU prediction parity, with recorded input identities."""
from copy import deepcopy
import os
from pathlib import Path
import platform
import subprocess
import sys
import numpy as np
import pandas as pd
import torch
import yaml
from power_fm_foresight._config import read_json, sha256, write_json
from power_fm_foresight._data.assets import checkpoint_assets, converted_partitions
from power_fm_foresight._data.dataset import SnapshotDataset, load_static_splits, snapshot_tables
from power_fm_foresight._data.network import load_network
from power_fm_foresight.genco import Genco


def prepare(config, download=False, spot_count=32, limit=None, batch_size=32):
    grid = load_network(config=config)
    cache = SnapshotDataset(config["paths"]["n1_cache"], grid=grid)
    assets = checkpoint_assets(config, download)
    root = Path(config["paths"]["runs_root"]) / "stage_a/parity"
    root.mkdir(parents=True, exist_ok=True)
    local_path = Path(config["paths"]["n1_splits"]) / "splits.npz"
    local = load_static_splits(local_path)
    official_path = assets / "splits/test.pt"
    official = np.asarray([int(i) for i in torch.load(official_path, weights_only=True)])
    ids = np.intersect1d(local["test"], official)
    if len(ids) == 0:
        raise ValueError("Empty official/local test intersection")
    if batch_size < 1 or (limit is not None and limit < spot_count):
        raise ValueError("Parity limit must cover every independent spot check")
    full_count = len(ids)
    if limit is not None:ids = ids[:limit]
    protocol = {"intersection_count": full_count, "limit": limit, "canonical_ids": ids.tolist(), "count": len(ids),
        "local_splits_sha256": sha256(local_path), "official_test_sha256": sha256(official_path),
        "input_rtol": 2e-6, "input_atol": 1e-4, "output_rtol": 2e-5, "output_atol": 1e-4,
        "spot_check_ids": ids[:spot_count].tolist(), "batch_size": batch_size,
        "reference": "unmodified pinned Graphkit evaluate CLI; derived contiguous IDs map to canonical IDs"}
    write_json(root / "protocol.json", protocol)  # Freeze before comparing results.
    selected_groups = {int(i)//cache.manifest["group_size"] for i in ids}
    for record in cache.manifest["groups"]:
        if limit is not None and record["group"] not in selected_groups:continue
        path = cache.folder / f"group_{record['group']}.pt"
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Published OPFData cache checksum mismatch: {path}")
    raw = root / "data" / config["checkpoint"]["directory"].split("/")[0] / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    tables = [[], [], []]
    selected_tables = {}
    for index, canonical in enumerate(ids):
        converted = snapshot_tables(cache[int(canonical)])
        if index < spot_count:
            selected_tables[int(canonical)] = tuple(t.copy() for t in converted)
        for destination, table in zip(tables, converted):
            table["scenario"] = index
            destination.append(table)
    for name, parts in zip(("bus", "gen", "branch"), tables):
        table = pd.concat(parts, ignore_index=True)
        if name == "bus":
            table = table.drop(columns=["min_q_mvar", "max_q_mvar"])
        table.to_parquet(raw / f"{name}_data.parquet", index=False)
    split_root = root / "splits"
    split_root.mkdir(exist_ok=True)
    for split in ("train", "val", "test"):
        values = torch.arange(len(ids), dtype=torch.long) if split == "test" else torch.empty(0, dtype=torch.long)
        torch.save(values, split_root / f"{split}.pt")
    original = assets / config["checkpoint"]["directory"] / config["checkpoint"]["configuration"]
    settings = yaml.safe_load(original.read_text())
    settings["data"].update(workers=0, scenarios=[len(ids)], split_from_existing_files=str(split_root))
    settings["training"].update(accelerator="cpu", devices=1, strategy="auto", batch_size=batch_size)
    settings["verbose"] = False
    (root / "evaluate.yaml").write_text(yaml.safe_dump(settings, sort_keys=False))
    # Independent official parquet spot check protects the local schema adapter.
    partitions = sorted({int(i)//200 for i in ids[:spot_count]})
    official_root = converted_partitions(partitions, Path(config["paths"]["assets_root"])/("official_"+config["checkpoint"]["directory"].split("/")[0]), download, config=config)
    differences = {}
    for name, table_index in (("bus", 0), ("gen", 1), ("branch", 2)):
        official_frames = []
        for partition in partitions:
            for path in (official_root / f"{name}_data.parquet" / f"scenario_partition={partition}").glob("*.parquet"):
                official_frames.append(pd.read_parquet(path))
        reference = pd.concat(official_frames, ignore_index=True)
        for canonical in ids[:spot_count]:
            ours = selected_tables[int(canonical)][table_index]
            theirs = reference[reference.scenario == canonical].reset_index(drop=True)
            if len(ours) != len(theirs):
                raise ValueError(f"Official export row count differs at {canonical}/{name}")
            for column in sorted(set(ours.columns) & set(theirs.columns)):
                a, b = ours[column].to_numpy(), theirs[column].to_numpy()
                np.testing.assert_allclose(a, b, rtol=protocol["input_rtol"], atol=protocol["input_atol"],
                                           err_msg=f"{canonical}/{name}/{column}")
                differences[f"{name}.{column}"] = max(differences.get(f"{name}.{column}", 0), float(abs(a-b).max()))
    write_json(root / "conversion_check.json", {"passed": True, "count": spot_count, "max_abs": differences,
                "official_dataset_revision": read_json(official_root/"download_manifest.json")["revision"]})
    print("Prepared parity:", len(ids), "snapshots; independent export checks:", spot_count, flush=True)
    return root



def evaluate(config):
    from gridfm_graphkit.datasets.graph_builder import build_hetero_data
    root = Path(config["paths"]["runs_root"]) / "stage_a/parity"
    protocol = read_json(root / "protocol.json")
    assets = checkpoint_assets(config)
    grid = load_network(config=config)
    cache = SnapshotDataset(config["paths"]["n1_cache"], grid=grid)
    model = Genco(assets, config["checkpoint"], device="cpu", grid=grid)
    folder = assets / config["checkpoint"]["directory"]
    command = [sys.executable, "-m", "gridfm_graphkit", "evaluate", "--config_conversion", "auto_inline",
        "--config", str(root/"evaluate.yaml"), "--data_path", str(root/"data"),
        "--model_path", str(folder/config["checkpoint"]["weights"]),
        "--normalizer_stats", str(folder/config["checkpoint"]["normalizer"]),
        "--num_workers", "0", "--batch_size", str(protocol["batch_size"]), "--save_output", "--output_path", str(root/"official"),
        "--log_dir", str(root/"mlruns"), "--exp_name", "genco-stage-a", "--run_name", "cpu-parity"]
    from .subset import serial_environment
    env = dict(serial_environment(os.environ), MLFLOW_ALLOW_FILE_STORE="true", MLFLOW_DISABLE_AGENT_HINT="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    extra = Path(config["paths"]["assets_root"]) / "cli_deps"
    if extra.exists():
        env["PYTHONPATH"] = str(extra) + os.pathsep + env.get("PYTHONPATH", "")
    write_json(root / "command.json", {"argv": command, "extra_dependency_path": str(extra)})
    with (root / "official_cli.log").open("w") as stream:
        subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=protocol.get("cap_seconds",1200))
    tables = [pd.read_parquet(root/"data"/config["checkpoint"]["directory"].split("/")[0]/"raw"/f"{name}_data.parquet") for name in ("bus", "gen", "branch")]
    # Match the upstream raw-table aggregation exactly.
    limits = tables[1].groupby(["scenario", "bus"])[["min_q_mvar", "max_q_mvar"]].sum().reset_index()
    tables[0] = tables[0].merge(limits, on=["scenario", "bus"], how="left").fillna(0)
    groups = [dict(tuple(table.groupby("scenario"))) for table in tables]
    predictions = []
    for lo in range(0, protocol["count"], protocol["batch_size"]):
        graphs = [build_hetero_data(*(group[i] for group in groups), scenario=i)
                  for i in range(lo, min(lo+protocol["batch_size"], protocol["count"]))]
        for graph in graphs:
            graph.case_id = cache.manifest["case"].removeprefix("pglib_opf_")
        output = model.predict(graphs)
        predictions.append(output["bus"].cpu().numpy())
    ours = np.concatenate(predictions)
    reference = pd.read_parquet(root / "official" / (config["checkpoint"]["directory"].split("/")[0]+"_predictions.parquet"))
    reference = reference.sort_values(["scenario", "bus"])
    columns = ["Vm_pred", "Va_pred", "Pg_pred", "Qg_pred"]
    theirs = reference[columns].to_numpy()
    np.testing.assert_allclose(ours, theirs, rtol=protocol["output_rtol"], atol=protocol["output_atol"])
    frame = reference[["scenario", "bus"]].copy()
    for i, name in enumerate(columns):
        frame[name] = ours[:, i]
    frame.to_parquet(root / "integration_predictions.parquet", index=False)
    result = {"passed": True, "snapshots": protocol["count"], "parameters": sum(p.numel() for p in model.model.parameters()),
        "normalizer_baseMVA": model.normalizer.baseMVA,
        "max_abs": {name: float(abs(ours[:,i]-theirs[:,i]).max()) for i,name in enumerate(columns)},
        "protocol_sha256": sha256(root/"protocol.json"), "source": "official Graphkit evaluate CLI"}
    result["provenance"] = {"hostname": platform.node(), "python": sys.executable,
        "python_binary_sha256": sha256(Path(sys.executable).resolve()),
        "torch_version": torch.__version__, "torch_core": torch._C.__file__,
        "torch_core_sha256": sha256(torch._C.__file__), "device": "cpu",
        "model_dtype": str(next(model.model.parameters()).dtype),
        "integration_output_dtype": str(ours.dtype), "official_output_dtype": str(theirs.dtype),
        "same_node": True, "same_python": command[0] == sys.executable,
        "execution": "official CLI is a local subprocess of the integration evaluator",
        "bit_identical": ours.dtype == theirs.dtype and ours.tobytes() == theirs.tobytes(),
        "scope": "bit-identical on the same node; no cross-machine/device claim"}
    write_json(root / "parity_report.json", result)
    print(result, flush=True)
