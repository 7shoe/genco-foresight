import numpy as np

from power_fm_foresight.scoring import ensemble_energy, marginal_scores


def test_replicated_point_mass_preserves_scores_but_not_finite_m_rank_pit():
    # Non-square dimension exposes the frozen ES float32 count-square-root path.
    point = np.arange(1, 31, dtype=float).reshape(1, 6, 5)
    repeated = np.repeat(point, 64, axis=0)
    target = np.zeros((6, 5))
    original = marginal_scores(point, target, seed=11)
    control = marginal_scores(repeated, target, seed=11)
    for metric in ("mae", "mse", "crps", "coverage_50", "coverage_90", "width_50", "width_90"):
        np.testing.assert_allclose(control[metric], original[metric], atol=1e-12, rtol=1e-12)
    expected = np.sqrt(np.mean(point ** 2))
    np.testing.assert_allclose(ensemble_energy(point, target, np.ones(30)), expected, rtol=1e-12)
    np.testing.assert_allclose(ensemble_energy(repeated, target, np.ones(30)), expected, rtol=1e-7)
    assert np.all(control["pit"] < 1 / 65)
    assert np.all(original["pit"] < 1 / 2)
    assert not np.array_equal(original["pit"], control["pit"])
