import numpy as np

from data.loader.network import load_network
from solver.checks import admittance_vectors, check_solution, physical_quantities


def test_independent_ac_equations_agree_with_datakit_admittance():
    net = load_network()
    rng = np.random.default_rng(2)
    state = {"vm": rng.uniform(0.96, 1.04, 30), "va_rad": rng.normal(0, 0.02, 30),
             "pg_mw": np.zeros(6), "qg_mvar": np.zeros(6)}
    q = physical_quantities(net, np.zeros(30), np.zeros(30), state)
    v = state["vm"]*np.exp(1j*state["va_rad"])
    expected = v*np.conj(net["Ybus"]@v)*net["baseMVA"]
    np.testing.assert_allclose(-q["residual_p_mw"], expected.real, atol=1e-10)
    np.testing.assert_allclose(-q["residual_q_mvar"], expected.imag, atol=1e-10)
    f, t = net["branch"][:, 0].astype(int), net["branch"][:, 1].astype(int)
    np.testing.assert_allclose(q["pf_mw"]+1j*q["qf_mvar"], v[f]*np.conj(net["Yf"]@v)*100, atol=1e-10)
    np.testing.assert_allclose(q["pt_mw"]+1j*q["qt_mvar"], v[t]*np.conj(net["Yt"]@v)*100, atol=1e-10)


def test_failure_and_nonfinite_labels_stay_unavailable():
    net = load_network()
    pd, qd = net["bus"][:, 2], net["bus"][:, 3]
    assert not check_solution(net, pd, qd, {"success": False, "status": "ITERATION_LIMIT"})["valid"]
    state = {"vm": [np.nan]*30, "va_rad": [0.]*30, "pg_mw": [0.]*6, "qg_mvar": [0.]*6}
    assert not check_solution(net, pd, qd, {"success": True, "state": state})["valid"]


def test_shunts_and_taps_affect_both_end_checks():
    net = load_network()
    yff, yft, ytf, ytt = admittance_vectors(net)
    transformer = np.flatnonzero((net["branch"][:, 8] != 0) & (net["branch"][:, 8] != 1))
    assert len(transformer) > 0
    assert np.any(abs(yff[transformer]-ytt[transformer]) > 1e-8)
    assert np.any(net["bus"][:, 5] != 0)


def test_each_sample_demand_reaches_balance_equations():
    net = load_network()
    state = {"vm": np.ones(30), "va_rad": np.zeros(30), "pg_mw": np.zeros(6), "qg_mvar": np.zeros(6)}
    pd, qd = net["bus"][:, 2].copy(), net["bus"][:, 3].copy()
    first = physical_quantities(net, pd, qd, state)
    pd[1] += 3
    second = physical_quantities(net, pd, qd, state)
    difference = np.zeros(30); difference[1] = -3
    np.testing.assert_allclose(second["residual_p_mw"]-first["residual_p_mw"], difference, atol=1e-12)
