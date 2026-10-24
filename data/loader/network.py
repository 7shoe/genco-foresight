"""Fixed PGLib network parsed by Datakit, with explicit physical units."""
from __future__ import annotations

import numpy as np
import pandas as pd
from matpowercaseframes import CaseFrames
from gridfm_datakit.network import Network, makeYbus
from scipy.sparse.csgraph import connected_components

from common import ROOT, sha256


def load_network(path=ROOT / "data/networks/pglib_opf_case30_ieee.m"):
    frames = CaseFrames(str(path))
    mpc = {name: (getattr(frames, name).values if isinstance(getattr(frames, name), pd.DataFrame)
                  else getattr(frames, name)) for name in frames._attributes}
    net = Network(mpc)
    b, g, e, cost = net.buses, net.gens, net.branches, net.gencosts
    if len(b) != 30 or not np.all(g[:, 7] == 1) or not np.all(e[:, 10] == 1):
        raise ValueError("Challenge supports connected case30 with all benchmark equipment online")
    gen_bus = g[:, 0].astype(int)
    if len(set(gen_bus)) != len(g):
        raise ValueError("Multiple generators per bus are unsupported")
    if not np.all(cost[:, 0] == 2) or not np.all(cost[:, 3] == 3):
        raise ValueError("Expected quadratic benchmark generator costs")
    Y, Yf, Yt = makeYbus(net.baseMVA, b, e)
    if connected_components(abs(Y), directed=False)[0] != 1:
        raise ValueError("Islands are unsupported")
    return {"baseMVA": float(net.baseMVA), "bus": b, "gen": g, "branch": e, "cost": cost,
            "gen_bus": gen_bus, "bus_ids": net.original_bus_indices.astype(int),
            "reference_bus": int(np.flatnonzero(b[:, 1] == 3).item()),
            "Ybus": Y, "Yf": Yf, "Yt": Yt, "case_sha256": sha256(path)}


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
    qg[network["gen_bus"]] = state["qg_mvar"]
    qmin[network["gen_bus"]], qmax[network["gen_bus"]] = g[:, 4], g[:, 3]
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
    for key in ("bus", "gen"):
        graph[key].label_available = torch.isfinite(graph[key].y)
    graph.baseMVA = torch.tensor(network["baseMVA"], dtype=torch.float32)
    return graph
