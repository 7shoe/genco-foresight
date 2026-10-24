import numpy as np

from scripts.variant_events import counts
from scripts.evaluate_stage_b import target_values
from data.loader.network import load_network
from monitor import event_targets, daily_union


def test_unresolved_state_is_unknown_for_voltage_and_thermal_but_demand_remains():
    net = load_network()
    row = {"target_state": None, "pd_mw": net["bus"][:, 2].tolist(),
           "qd_mvar": net["bus"][:, 3].tolist(), "time_utc": "2025-01-01T00:00:00Z"}
    values = target_values([row] * 24, net)
    assert np.isfinite(values["pd_mw"]).all()
    assert np.isnan(values["vm"]).all() and np.isnan(values["loading"]).all()
    events = event_targets(values, net)
    for family in events.values():
        assert np.isnan(family["hourly"]).all()
        assert np.isnan(family["daily_any"]).all()
        c = counts(family["hourly"])
        assert c["denominator"] == 0 and c["rate"] is None and c["positive"] == 0


def test_partial_day_does_not_turn_unknown_into_negative():
    event = np.zeros((2, 24, 1), bool)
    known = np.ones_like(event)
    known[:, 2] = False
    event[0, 1] = True
    daily, valid = daily_union(event, known)
    np.testing.assert_array_equal(daily[:, 0], [True, False])
    np.testing.assert_array_equal(valid[:, 0], [True, False])
