"""Read-only canonical OPFData cache and explicit Graphkit conversion.

Cache access adapted from lumina-sdk distillation/training/T1_single_topo/data.py
at e75ffcd74854c14e2960b9a9411ddd72377e5239 (Apache-2.0).
Copyright 2026 UChicago Argonne, LLC. See THIRD_PARTY_NOTICES.md.
Changes: lazy group loading, no SDK dependency, explicit schema/unit conversion.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.data.separate import separate

from common import read_json


def load_static_splits(path, count=300000):
    with np.load(path, allow_pickle=False) as source:
        parts = {name: source[name].copy() for name in ("train", "val", "test")}
    all_ids = np.concatenate(list(parts.values()))
    if len(all_ids) != count or not np.array_equal(np.sort(all_ids), np.arange(count)):
        raise ValueError("Static splits must be disjoint and cover every canonical ID exactly once")
    return parts


class SnapshotDataset:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.manifest = read_json(self.folder / "manifest.json")
        m = self.manifest
        if m["format"] != "lumina-t1-data-v1" or m["case"] != "pglib_opf_case30_ieee":
            raise ValueError("Unsupported cache schema/case")
        if m["count"] != len(m["groups"])*m["group_size"]:
            raise ValueError("Cache manifest count mismatch")
        self._groups = {}

    def __len__(self):
        return self.manifest["count"]

    def __getitem__(self, index):
        index = int(index)
        if not 0 <= index < len(self):
            raise IndexError(index)
        group, offset = divmod(index, self.manifest["group_size"])
        if group not in self._groups:
            record = self.manifest["groups"][group]
            path = self.folder / f"group_{group}.pt"
            stat = path.stat()
            if record["group"] != group or (stat.st_size, stat.st_mtime_ns) != (record["bytes"], record["mtime_ns"]):
                raise ValueError(f"Cache changed since publication: {path}")
            self._groups[group] = torch.load(path, mmap=True, map_location="cpu", weights_only=False)
        data, slices = self._groups[group]
        graph = separate(data.__class__, data, offset, slices, decrement=False)
        if graph.sample_id.item() != index:
            raise ValueError(f"Canonical ID mismatch: {index}")
        return graph


def snapshot_tables(graph):
    """Convert the documented local JSON schema to upstream physical tables."""
    base = float(graph.baseMVA.item())
    x = graph["bus"].x.numpy()
    n = len(x)
    scenario = int(graph.sample_id.item())
    pd_mw, qd, gs, bs = (np.zeros(n) for _ in range(4))
    load_bus = graph["load", "load_link", "bus"].edge_index[1].numpy()
    np.add.at(pd_mw, load_bus, graph["load"].x[:, 0].numpy()*base)
    np.add.at(qd, load_bus, graph["load"].x[:, 1].numpy()*base)
    if "shunt" in graph.node_types:
        shunt_bus = graph["shunt", "shunt_link", "bus"].edge_index[1].numpy()
        np.add.at(bs, shunt_bus, graph["shunt"].x[:, 0].numpy())
        np.add.at(gs, shunt_bus, graph["shunt"].x[:, 1].numpy())
    gen_bus = graph["generator", "generator_link", "bus"].edge_index[1].numpy()
    gx, gy = graph["generator"].x.numpy(), graph["generator"].y.numpy()
    qg, qmin, qmax = (np.zeros(n) for _ in range(3))
    np.add.at(qg, gen_bus, gy[:, 1]*base)
    np.add.at(qmin, gen_bus, gx[:, 5]*base)
    np.add.at(qmax, gen_bus, gx[:, 6]*base)
    by = graph["bus"].y.numpy()
    buses = pd.DataFrame({"scenario": scenario, "bus": np.arange(n), "Pd": pd_mw, "Qd": qd,
        "Qg": qg, "Vm": by[:, 1], "Va": np.rad2deg(by[:, 0]), "PQ": x[:, 3], "PV": x[:, 4],
        "REF": x[:, 5], "min_vm_pu": x[:, 1], "max_vm_pu": x[:, 2],
        "min_q_mvar": qmin, "max_q_mvar": qmax, "GS": gs, "BS": bs, "vn_kv": x[:, 0]})
    gens = pd.DataFrame({"scenario": scenario, "bus": gen_bus, "p_mw": gy[:, 0]*base,
        "min_p_mw": gx[:, 2]*base, "max_p_mw": gx[:, 3]*base,
        "min_q_mvar": gx[:, 5]*base, "max_q_mvar": gx[:, 6]*base,
        "cp0_eur": gx[:, 10], "cp1_eur_per_mw": gx[:, 9]/base,
        "cp2_eur_per_mw2": gx[:, 8]/base**2, "in_service": np.ones(len(gx))})
    rows = []
    for kind in ("ac_line", "transformer"):
        store = graph["bus", kind, "bus"]
        edge, values, labels = store.edge_index.numpy(), store.edge_attr.numpy(), store.edge_label.numpy()
        for j, a in enumerate(values):
            if kind == "ac_line":
                amin, amax, bfr, bto, r, reactance, rate = a[:7]
                tap, shift = 1.0, 0.0
            else:
                amin, amax, r, reactance, rate = a[:5]
                tap, shift, bfr, bto = a[7:11]
            y = 1 / complex(r, reactance)
            ratio = tap * np.exp(1j*shift)
            yff, yft = (y+1j*bfr)/abs(ratio)**2, -y/np.conj(ratio)
            ytf, ytt = -y/ratio, y+1j*bto
            rows.append({"scenario": scenario, "from_bus": edge[0, j], "to_bus": edge[1, j],
                "pf": labels[j, 2]*base, "qf": labels[j, 3]*base,
                "pt": labels[j, 0]*base, "qt": labels[j, 1]*base,
                "Yff_r": yff.real, "Yff_i": yff.imag, "Yft_r": yft.real, "Yft_i": yft.imag,
                "Ytf_r": ytf.real, "Ytf_i": ytf.imag, "Ytt_r": ytt.real, "Ytt_i": ytt.imag,
                "tap": tap, "ang_min": np.rad2deg(amin), "ang_max": np.rad2deg(amax),
                "rate_a": rate*base, "br_status": 1.0})
    return buses, gens, pd.DataFrame(rows)


def snapshot_graph(graph):
    from gridfm_graphkit.datasets.graph_builder import build_hetero_data
    return build_hetero_data(*snapshot_tables(graph), scenario=int(graph.sample_id.item()))


class OfficialSnapshotDataset:
    """Read pinned downloaded Graphkit parquet when the Lustre cache is absent."""
    def __init__(self, root):
        self.root = Path(root)
        self._partitions = {}

    def __len__(self):
        return 300000

    def __getitem__(self, sample_id):
        from gridfm_graphkit.datasets.graph_builder import build_hetero_data
        sample_id = int(sample_id)
        if not 0 <= sample_id < len(self):
            raise IndexError(sample_id)
        partition = sample_id//200
        if partition not in self._partitions:
            tables = []
            for name in ("bus", "gen", "branch"):
                folder = self.root/f"{name}_data.parquet"/f"scenario_partition={partition}"
                files = list(folder.glob("*.parquet"))
                if len(files) != 1:
                    raise FileNotFoundError(f"Missing pinned parquet {folder}; run python -m data.download.prepare "
                                            f"--download --reference-partitions {partition}")
                tables.append(pd.read_parquet(files[0]))
            limits = tables[1].groupby(["scenario", "bus"])[["min_q_mvar", "max_q_mvar"]].sum().reset_index()
            tables[0] = tables[0].merge(limits, on=["scenario", "bus"], how="left").fillna(0)
            self._partitions[partition] = tables
        selected = [t[t.scenario == sample_id].reset_index(drop=True) for t in self._partitions[partition]]
        if any(t.empty for t in selected):
            raise ValueError(f"Canonical ID {sample_id} is missing from its pinned partition")
        return build_hetero_data(*selected, scenario=sample_id)


def daily_origins(hours=504, history=72, horizon=24):
    return list(range(history, hours-horizon+1, 24))


class TemporalDataset:
    """Windows expose observed history and calendar separately from target labels."""
    def __init__(self, root, partition, history=72, horizon=24):
        self.root = Path(root)
        self.manifest = read_json(self.root / "manifest.json")
        self.history, self.horizon = history, horizon
        self.episodes = [e for e in self.manifest["episodes"] if e["split"] == partition]
        # Frozen Stage B protocol: all causal training windows; midnight evaluation.
        stride = 1 if partition == "train" else 24
        self.index = [(e["episode_id"], origin) for e in self.episodes
                      for origin in range(history, e["hours"]-horizon+1, stride)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        episode, origin = self.index[index]
        shard = read_json(self.root / "episodes" / f"{episode}.json")
        rows = shard["records"]
        if len(rows) < origin+self.horizon:
            raise ValueError("Episode is not fully labeled yet")
        observed = rows[origin-self.history:origin]
        target = rows[origin:origin+self.horizon]
        # Future demand/solved states never enter the input dictionary.
        return {"episode_id": episode, "origin": origin,
                "history": observed, "known_future": {"time_utc": [r["time_utc"] for r in target]},
                "targets": target}


def load_snapshot(sample_id, config=None):
    from common import load_config
    config = config or load_config()
    local = Path(config["paths"]["n1_cache"])
    if (local/"manifest.json").exists():
        return snapshot_graph(SnapshotDataset(local)[sample_id])
    return OfficialSnapshotDataset(Path(config["paths"]["assets_root"])/"official_case30")[sample_id]


def load_window(episode_id, origin, history=72, horizon=24, config=None):
    from common import load_config
    from data.loader.network import graph_for_demand, load_network
    config = config or load_config()
    root = Path(config["paths"].get("temporal_root", str(Path(config["paths"]["data_root"])/"temporal_case30")))
    manifest = read_json(root/"manifest.json")
    matches = [e for e in manifest["episodes"] if e["episode_id"] == episode_id]
    if len(matches) != 1:
        raise KeyError(episode_id)
    dataset = TemporalDataset(root, matches[0]["split"], history, horizon)
    try:
        index = dataset.index.index((episode_id, origin))
    except ValueError as error:
        raise ValueError("Origin is not a complete window on this partition's forecast schedule") from error
    window = dataset[index]
    network = load_network()
    window["history_graphs"] = [graph_for_demand(network, r["pd_mw"], r["qd_mvar"],
                                r["target_state"], scenario=r["hour"]) for r in window["history"]]
    return window
