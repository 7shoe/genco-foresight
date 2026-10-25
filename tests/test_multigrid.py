"""CPU checks for network identities, inactive equipment, and scoring coordinates."""
from pathlib import Path

import numpy as np
import pytest

from common import ROOT, load_config
from data.coordinates import bus_qg, fit_scales, record_values
from data.grids import grid_config, registry
from data.loader.network import graph_tables, load_network
from monitor import event_values
from solver.checks import admittance_vectors, physical_quantities
from training.forecaster import episode_arrays


@pytest.mark.parametrize("case", registry()["grids"])
def test_selector_interpolation_and_overrides(case):
    path = ROOT / f"configs/multigrid/{case}.yaml"
    original = load_config(path)
    assert original == grid_config(case)
    replacement = "/lus/flare/projects/example/user"
    cfg = load_config(path, overrides={"base": replacement, "stage_b": {"training_samples": 2}})
    assert "base" not in cfg
    assert cfg["stage_b"]["training_samples"] == 2
    assert cfg["paths"]["n1_cache"] == f"{replacement}/acopf_datasets/processed/N1/pglib_opf_{case}"
    assert cfg["paths"]["training_run"] == f"{replacement}/GENCO/runs/multigrid/{case}/adapter"
    assert cfg["solver"]["julia_executable"].startswith(replacement + "/")
    assert cfg["paths"]["reference_solver_checkout"] == original["paths"]["reference_solver_checkout"]
    custom = load_config(path, overrides={"paths": {"training_run": "${paths.runs_root}/custom"}})
    assert custom["paths"]["training_run"] == custom["paths"]["runs_root"] + "/custom"


@pytest.mark.parametrize("case", registry()["grids"])
def test_registry_and_dimensions(case):
    cfg = load_config(ROOT / f"configs/multigrid/{case}.yaml")
    grid = load_network(config=cfg)
    assert grid.n_bus == grid.Ybus.shape[0]
    assert grid.Yf.shape == (grid.n_branch, grid.n_bus)
    assert grid.gen_bus.shape == (grid.n_gen,)
    assert grid.bus[grid.reference_bus, 1] == 3
    assert len(grid.qg_buses) == len(set(grid.gen_bus[grid.online_gen]))
    assert Path(cfg["paths"]["training_run"]).is_relative_to(Path(cfg["paths"]["runs_root"]))
    assert f"/multigrid/{case}/" in cfg["paths"]["temporal_root"]
    assert f"/multigrid/{case}/" in cfg["stage_b"]["coordinate_scales"]
    assert grid.describe()["reference_bus"] == grid.bus_ids[grid.reference_bus]


@pytest.mark.parametrize("case", ["case500_goc", "case2000_goc"])
def test_colocated_generators_and_offline_masks(case):
    grid = load_network(config=grid_config(case))
    pg = np.arange(grid.n_gen, dtype=float)
    qg = pg+.5
    pg[~grid.online_gen] = np.nan; qg[~grid.online_gen] = np.nan
    state = dict(vm=np.ones(grid.n_bus), va_rad=np.zeros(grid.n_bus), pg_mw=pg, qg_mvar=qg)
    row = dict(hour=0, time_utc="2025-01-01T00:00:00Z", pd_mw=grid.bus[:,2], qd_mvar=grid.bus[:,3],
               target_state=state, state_available=True)
    values = record_values(grid, row)
    np.testing.assert_allclose(values["qg_mvar"].sum(), np.nansum(qg))
    assert len(values["qg_mvar"]) < grid.n_online_gen
    tables = graph_tables(grid, row["pd_mw"], row["qd_mvar"], state)
    np.testing.assert_allclose(tables[0].Qg.to_numpy()[grid.qg_buses], bus_qg(grid, qg))
    expected = np.zeros(grid.n_bus)
    np.add.at(expected, grid.gen_bus[grid.online_gen], grid.gen[grid.online_gen,3])
    np.testing.assert_allclose(tables[0].max_q_mvar, expected)
    array = episode_arrays([row], grid)
    np.testing.assert_allclose(array["covariates"][0,:,4].sum(), np.nansum(pg))
    physics = physical_quantities(grid, row["pd_mw"], row["qd_mvar"], state)
    assert np.isfinite(physics["residual_p_mw"]).all()
    assert np.isnan(values["pg_mw"][~grid.online_gen]).all()
    assert np.isnan(values["pf_mw"][~grid.online_branch]).all()
    for vector in admittance_vectors(grid):
        assert (vector[~grid.online_branch] == 0).all()
    scales = fit_scales(grid, [row, row])
    assert not np.asarray(scales["features"]["pg_mw"]["joint_score_mask"])[~grid.online_gen].any()
    assert not np.asarray(scales["features"]["pf_mw"]["joint_score_mask"])[~grid.online_branch].any()
    assert not scales["features"]["va_rad"]["joint_score_mask"][grid.reference_bus]
    assert len(scales["generator_history"]["qg_mvar"]["mean"]) == grid.n_gen
    assert len(scales["features"]["qg_mvar"]["scale"]) == len(grid.qg_buses)
    events = event_values(values, grid)
    assert events["thermal"][0].shape == (grid.rated_branches.sum(),)
