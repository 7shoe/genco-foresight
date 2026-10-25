"""Independent AC equations and physical-label checks in physical units.

These calculations do not use solver success flags or solver-reported flows.
"""
from __future__ import annotations

import numpy as np


def admittance_vectors(network):
    """Compute branch admittances with transformer taps and line chargng; some physics involved"""
    e = network["branch"]

    # series admittance  = inverse of resistance plus imaginary reactance
    y = np.divide(e[:, 10], e[:, 2] + 1j * e[:, 3], out=np.zeros(len(e), dtype=complex), where=e[:, 10] != 0)

    # MATPOWER uses 0 for an unspecified tap ratio, meaning unity
    tap = np.where(e[:, 8] == 0, 1.0, e[:, 8]) * np.exp(1j*np.deg2rad(e[:, 9]))

    # split line charging equally between the two branch ends
    charging = 0.5j * e[:, 4] * e[:, 10]
    return (
        (y+charging)/abs(tap)**2,
        -y/np.conj(tap),
        -y/tap,
        y+charging
    )


def physical_quantities(network, pd_mw, qd_mvar, state):
    """Recompute branch powers, bus balance residuals, and thermal loading"""
    b, g, e = (network[key] for key in ("bus", "gen", "branch"))
    base = network["baseMVA"]
    vm, va = np.asarray(state["vm"]), np.asarray(state["va_rad"])
    pg, qg = np.asarray(state["pg_mw"]), np.asarray(state["qg_mvar"])

    # build ℂ-valued, per-unit voltages
    v = vm * np.exp(1j*va)

    # branch endpoints must already be zero-based indices into the bus array
    f, t = e[:, 0].astype(int), e[:, 1].astype(int)
    yff, yft, ytf, ytt = admittance_vectors(network)

    # S = V * conj(I); each terminal power is positive into the branch
    sf = v[f] * np.conj(yff*v[f] + yft*v[t]) * base
    st = v[t] * np.conj(ytf*v[f] + ytt*v[t]) * base

    # sum all branch-terminal powers attached to each bus
    injection = np.zeros(len(b), dtype=complex)
    np.add.at(injection, f, sf)
    np.add.at(injection, t, st)

    # incl bus shunts ( whose power varies with squared voltage magnitude)
    injection += (b[:, 4] - 1j*b[:, 5]) * vm**2

    # accumulate generator outputs
    generation = np.zeros(len(b), dtype=complex)
    online = network.online_gen
    np.add.at(generation, network.gen_bus[online], (pg+1j*qg)[online])

    #  balanced bus has generation minus demand equal
    residual = generation - np.asarray(pd_mw) - 1j*np.asarray(qd_mvar) - injection

    # use the larger terminal apparent power
    rated = network.rated_branches
    loading = np.full(len(e), np.nan)
    loading[rated] = np.maximum(abs(sf[rated]), abs(st[rated])) / e[rated, 5]

    # output dict
    return {"pf_mw": sf.real,
            "qf_mvar": sf.imag,
            "pt_mw": st.real,
            "qt_mvar": st.imag,
            "residual_p_mw": residual.real,
            "residual_q_mvar": residual.imag,
            "loading": loading,
            "rated": rated
            }


def check_solution(network, pd_mw, qd_mvar, record, tolerance=1e-5):
    """Check state consistency, physical constraints, and reported solver values."""
    if "state" not in record:
        return {"valid": False, "reason": "no_primal_state"}

    state = {key: np.asarray(value, dtype=float) for key, value in record["state"].items()}
    # Require one voltage per bus and one active/reactive output per generator.
    expected = {"vm": network.n_bus,
                "va_rad": network.n_bus,
                "pg_mw": len(network["gen"]),
                "qg_mvar": len(network["gen"])}

    if any(state.get(k, np.array([])).shape != (size,) for k, size in expected.items()):
        return {"valid": False, "reason": "state_shape"}
    if not all(np.isfinite(value if key in ("vm", "va_rad") else value[network.online_gen]).all() for key, value in state.items()):
        return {"valid": False, "reason": "nonfinite_primal_state"}

    # matching dims alone not enough
    if list(record.get("bus_ids", [])) != network["bus_ids"].tolist():
        return {"valid": False, "reason": "bus_identity_mismatch"}

    #
    b, g, e, cost = (network[key] for key in ("bus", "gen", "branch", "cost"))

    # recompute physical quantities independently of the solver's reported flows
    q = physical_quantities(network, pd_mw, qd_mvar, state)

    # largest bound violation (0 is compliance)
    bound_error = lambda values, lower, upper: float(np.maximum(np.maximum(lower-values, values-upper), 0).max(initial=0))
    angle = state["va_rad"][e[:, 0].astype(int)] - state["va_rad"][e[:, 1].astype(int)]
    rated = q["rated"]

    # quadratic cost structuee (c2 * Pg² + c1 * Pg + c0) assumed
    objective = float(np.sum((cost[:, 4]*state["pg_mw"]**2 + cost[:, 5]*state["pg_mw"] + cost[:, 6])[network.online_gen]))

    # errors retain their own units
    errors = {
        "p_balance_mw": float(abs(q["residual_p_mw"]).max()),
        "q_balance_mvar": float(abs(q["residual_q_mvar"]).max()),
        "voltage_pu": bound_error(state["vm"], b[:, 12], b[:, 11]),
        "pg_mw": bound_error(state["pg_mw"][network.online_gen], g[network.online_gen, 9], g[network.online_gen, 8]),
        "qg_mvar": bound_error(state["qg_mvar"][network.online_gen], g[network.online_gen, 4], g[network.online_gen, 3]),
        "thermal_mva": float(np.maximum((q["loading"][rated]-1)*e[rated, 5], 0).max(initial=0)),
        "angle_rad": bound_error(angle[network.online_branch], np.deg2rad(e[network.online_branch, 11]), np.deg2rad(e[network.online_branch, 12])),
        "reference_rad": float(abs(state["va_rad"][network["reference_bus"]])),
        "objective_relative": abs(objective-record.get("objective", objective))/max(1, abs(objective)),
    }
    # compare reported terminal flows
    if "branch" in record:
        flow_error = []
        for name, physical in (("pf", "pf_mw"), ("qf", "qf_mvar"), ("pt", "pt_mw"), ("qt", "qt_mvar")):
            flow_error.extend(abs(np.asarray(record["branch"][name], dtype=float)-q[physical])[network.online_branch])
        errors["solver_flow_mismatch_mva"] = float(max(flow_error))

    # passing physical checks is necessary
    valid = bool(record.get("success", False) and max(errors.values()) <= tolerance)

    # output dict
    return {"valid": valid,
            "errors": errors,
            "tolerance": tolerance,
            "objective_recomputed": objective,
            "thermal_stress": bool((q["loading"][rated] > 0.95).any()),
            "voltage_stress": bool(((state["vm"] <= b[:, 12]+0.05*(b[:, 11]-b[:, 12])) |
                                    (state["vm"] >= b[:, 11]-0.05*(b[:, 11]-b[:, 12]))).any())
            }
