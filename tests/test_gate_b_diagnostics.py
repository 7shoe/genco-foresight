"""Diagnostic accounting and retrospective masks; no model fitting."""
import numpy as np
import pandas as pd

from power_fm_foresight.scoring import ensemble_energy, joint_coordinates
from experiments.diagnose_stage_b import group_diagnostics, reference_series, balance, average_balances


def test_additive_energy_allocation_matches_frozen_joint_with_mask():
    rng = np.random.default_rng(4200)
    keys = ("pd_mw", "qd_mvar", "vm", "va_rad", "pg_mw", "qg_mvar",
            "pf_mw", "qf_mvar", "pt_mw", "qt_mvar")
    scales = {"features": {k: {"joint_score_mask": [k != "qd_mvar"], "scale": [1. + i]}
                           for i, k in enumerate(keys)}}
    values = {k: rng.normal(size=(7, 4, 1)) for k in keys}
    targets = {k: rng.normal(size=(4, 1)) for k in keys}
    hours = np.array([True, False, True, False])
    x, y = joint_coordinates(values, scales), joint_coordinates(targets, scales)
    scale = np.tile([s["scale"][0] for s in scales["features"].values() if s["joint_score_mask"][0]], 4)
    mask = np.broadcast_to(hours[:, None], y.shape)
    expected = ensemble_energy(x, y, scale, mask.reshape(-1))
    actual = group_diagnostics(values, targets, scales, hours)
    np.testing.assert_allclose(actual["joint_energy_from_contributions"], expected, rtol=1e-12, atol=1e-12)
    assert actual["groups"]["Qd"]["group_energy_score"] is None
    assert actual["groups"]["Qd"]["observed_coordinates"] == 0
    # Excluded-hour outliers cannot affect either group scores or allocation.
    for v in values.values():
        v[:, ~hours] += 1e9
    changed = group_diagnostics(values, targets, scales, hours)
    assert changed == actual
    empty = group_diagnostics(values, targets, scales, np.zeros(4, dtype=bool))
    assert empty["joint_energy_from_contributions"] is None


def test_reference_series_uses_training_transform_without_reclipping():
    manifest = {"generation": {"scale_low": .3, "scale_high": 1.}, "train_profile_min": 10., "train_profile_max": 20.,
                "episodes": [{"episode_id": "x", "split": "test", "source_start_hour": 1, "hours": 3}]}
    profile = pd.DataFrame({"demand_mw": [0., 10., 20., 30., 40.]})
    np.testing.assert_allclose(reference_series(manifest, profile)["x"], [.3, 1., 1.7])


def test_balance_aggregates_samples_and_hours_not_absolute_system_sum():
    v = np.array([[[2., -2.], [6., -6.], [10., -10.]]])
    x = {"residual_p_mw": v, "residual_q_mvar": v * 2}
    a = balance(x, np.array([True, False, False]))
    b = balance(x, np.array([False, True, True]))
    result = average_balances([a, b])
    assert result["residual_p_mw"]["mean_abs"] == 6.
    assert result["residual_q_mvar"]["mean_abs"] == 12.
    assert result["residual_p_mw"]["count"] == 6
    assert result["residual_p_mw"]["per_bus_mean_abs"] == [6., 6.]
