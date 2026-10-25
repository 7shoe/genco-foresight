"""Fixed PGLib network parsed by Datakit, with explicit physical units."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matpowercaseframes import CaseFrames
from gridfm_datakit.network import Network, makeYbus
from scipy.sparse.csgraph import connected_components

from common import ROOT, sha256
from data.grids import check_case, network_path


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


def load_network(path=None, config=None):
    path = Path(path) if path is not None else network_path(config)
    if config is not None:
        check_case(config["task"]["canonical_case"], path.stem, "network file")
        check_case(path.stem, config["checkpoint"]["directory"].split("/")[0], "checkpoint directory")
    frames = CaseFrames(str(path))
    check_case(path.stem, frames.name, "parsed network")
    mpc = {name: (getattr(frames, name).values if isinstance(getattr(frames, name), pd.DataFrame)
                  else getattr(frames, name)) for name in frames._attributes}
    net = Network(mpc)
    b, g, e, cost = net.buses, net.gens, net.branches, net.gencosts
    gen_bus = g[:, 0].astype(int)
    if not np.all(cost[:, 0] == 2) or not np.all(cost[:, 3] == 3):
        raise ValueError("Expected quadratic benchmark generator costs")
    Y, Yf, Yt = makeYbus(net.baseMVA, b, e)
    if connected_components(abs(Y), directed=False)[0] != 1:
        raise ValueError("Islands are unsupported")
    refs = np.flatnonzero(b[:, 1] == 3)
    if len(refs) != 1:
        raise ValueError(f"Expected one reference bus, found {len(refs)}")
    return GridSpec(path, net, float(net.baseMVA), b, g, e, cost, gen_bus,
                    net.original_bus_indices.astype(int), int(refs[0]), Y, Yf, Yt, sha256(path))


def graph_tables(network, pd_mw, qd_mvar, state=None, scenario=0):
    """Raw physical Graphkit tables. Unknown labels are explicitly unavailable."""
    b, g, e, cost = (network[key] for key in ("bus", "gen", "branch", "cost"))
    base = network["baseMVA"]
    n, ng = len(b), len(g)
    pd_mw, qd_mvar = np.asarray(pd_mw), np.asarray(qd_mvar)
    if state is None:
        angles = np.full(n, np.nan)
        angles[network["reference_bus"]] = 0.0  # Known reference constraint, not a solved input.
        state = {"vm": np.full(n, np.nan), "va_rad": angles,
                 "pg_mw": np.full(ng, np.nan), "qg_mvar": np.full(ng, np.nan)}
    qg, qmin, qmax = np.zeros(n), np.zeros(n), np.zeros(n)
    online = network.online_gen
    for target, source in ((qg, state["qg_mvar"]), (qmin, g[:, 4]), (qmax, g[:, 3])):
        np.add.at(target, network.gen_bus[online], np.asarray(source)[online])
    buses = pd.DataFrame({"scenario": scenario, "bus": np.arange(n), "Pd": pd_mw, "Qd": qd_mvar,
        "Qg": qg, "Vm": state["vm"], "Va": np.rad2deg(state["va_rad"]),
        "PQ": (b[:, 1] == 1).astype(int), "PV": (b[:, 1] == 2).astype(int),
        "REF": (b[:, 1] == 3).astype(int), "min_vm_pu": b[:, 12], "max_vm_pu": b[:, 11],
        "min_q_mvar": qmin, "max_q_mvar": qmax, "GS": b[:, 4]/base, "BS": b[:, 5]/base,
        "vn_kv": b[:, 9]})
    generators = pd.DataFrame({"scenario": scenario, "bus": network["gen_bus"],
        "p_mw": state["pg_mw"], "min_p_mw": g[:, 9], "max_p_mw": g[:, 8],
        "min_q_mvar": g[:, 4], "max_q_mvar": g[:, 3], "cp0_eur": cost[:, 6],
        "cp1_eur_per_mw": cost[:, 5], "cp2_eur_per_mw2": cost[:, 4], "in_service": g[:, 7]})
    f, t = e[:, 0].astype(int), e[:, 1].astype(int)
    # Datakit's admittance matrices include tap ratios and phase shifts.
    yf, yt = network["Yf"], network["Yt"]
    yff = np.asarray(yf[np.arange(len(e)), f]).ravel()
    yft = np.asarray(yf[np.arange(len(e)), t]).ravel()
    ytf = np.asarray(yt[np.arange(len(e)), f]).ravel()
    ytt = np.asarray(yt[np.arange(len(e)), t]).ravel()
    branches = pd.DataFrame({"scenario": scenario, "from_bus": f, "to_bus": t,
        "pf": np.nan, "qf": np.nan, "pt": np.nan, "qt": np.nan,
        "Yff_r": yff.real, "Yff_i": yff.imag, "Yft_r": yft.real, "Yft_i": yft.imag,
        "Ytf_r": ytf.real, "Ytf_i": ytf.imag, "Ytt_r": ytt.real, "Ytt_i": ytt.imag,
        "tap": np.where(e[:, 8] == 0, 1, e[:, 8]), "ang_min": e[:, 11], "ang_max": e[:, 12],
        "rate_a": e[:, 5], "br_status": e[:, 10]})
    return buses, generators, branches


def graph_for_demand(network, pd_mw, qd_mvar, state=None, scenario=0):
    import torch
    from gridfm_graphkit.datasets.graph_builder import build_hetero_data
    tables = graph_tables(network, pd_mw, qd_mvar, state, scenario)
    graph = build_hetero_data(*tables, scenario=scenario)
    graph.case_id = network.case
    for key in ("bus", "gen"):
        graph[key].label_available = torch.isfinite(graph[key].y)
    graph.baseMVA = torch.tensor(network["baseMVA"], dtype=torch.float32)
    return graph
