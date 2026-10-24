import numpy as np
import pytest

from monitor import daily_union, event_probabilities, fit_map, apply_map, reliability


def test_daily_union_preserves_joint_dependence_and_missing_label_logic():
    event = np.array([[[True], [False]], [[False], [True]]])
    daily, available = daily_union(event, np.ones_like(event))
    assert available.all() and daily.mean() == 1.
    assert 1 - np.prod(1 - event.mean(0)) == .75  # Independence would be wrong.
    missing = np.array([[False], [False]])
    value, valid = daily_union(missing, np.array([[True], [False]]))
    assert not value.item() and not valid.item()
    value, valid = daily_union(np.array([[True], [False]]), np.array([[True], [False]]))
    assert value.item() and valid.item()


def test_map_rejects_test_fit_and_checks_distinct_calibration_days():
    y = np.zeros((10, 24, 2))
    y[:5, :, 0] = 1
    y[:1, :, 1] = 1  # Many positive hours, but only one positive day.
    p = np.full(y.shape, .2)
    with pytest.raises(ValueError, match="risk_calibration"):
        fit_map(p, y, "hourly", "test")
    fitted = fit_map(p, y, "hourly", "risk_calibration")
    assert fitted["support"]["eligible"] == [True, False]
    output = apply_map(p, fitted)
    assert np.isfinite(output[..., 0]).all() and np.isnan(output[..., 1]).all()
    np.testing.assert_allclose(output[..., 0], .5, atol=.001)


def test_reliability_retains_endpoints_and_does_not_zero_impute_unknown():
    result = reliability([0, 1, .5, np.nan], [0, 1, np.nan, 1])
    assert result["count"] == 2 and result["brier"] == 0
    assert result["bins"][0]["count"] == result["bins"][-1]["count"] == 1
    assert not any(b["plot_supported"] for b in result["bins"])


def test_daily_support_requires_calibration_prevalence_between_20_and_80():
    y = np.ones((40, 1)); y[:5] = 0
    fitted = fit_map(np.full_like(y, .5), y, "daily_any", "risk_calibration")
    assert not fitted["supported"]
