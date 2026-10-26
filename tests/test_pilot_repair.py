"""Reference identity, offline warm starts and resumable hour ownership."""
from types import SimpleNamespace

import numpy as np
import pytest

from power_fm_foresight._config import read_json, write_json
from experiments.pilots.hour_shards import assignments, read_hour, seal


def test_hour_workers_cover_pilot_and_full_data_once():
    episodes = [{'episode_id': str(i)} for i in range(20)]
    for maximum in (200, 10080):
        partitions = [assignments(episodes, maximum, w, 96) for w in range(96)]
        ids = [(ep['episode_id'], hour) for part in partitions for ep, hour in part]
        assert len(ids) == len(set(ids)) == maximum
        assert all(part for part in partitions)


def test_hour_resume_rejects_tampering_and_wrong_identity(tmp_path):
    payload = dict(benchmark_id='fixed', episode_id='train_00', record={'hour': 17, 'target_state': None})
    path = tmp_path / '017.json'
    write_json(path, dict(payload, sha256=seal(payload)))
    assert read_hour(path, 'fixed', 'train_00', 17) == payload['record']
    with pytest.raises(ValueError):read_hour(path, 'changed', 'train_00', 17)
    with pytest.raises(ValueError):read_hour(path, 'fixed', 'train_00', 18)
    changed = read_json(path); changed['record']['target_state'] = {'vm': [1.]}
    write_json(path, changed)
    with pytest.raises(ValueError):read_hour(path, 'fixed', 'train_00', 17)


def test_offline_warm_starts_do_not_mutate_saved_labels():
    from power_fm_foresight._solver.adapter import IpoptSolver
    solver = IpoptSolver.__new__(IpoptSolver)
    solver.grid = SimpleNamespace(n_bus=2, online_gen=np.array([True, False]))
    solver.casefile = 'fixture.m'; solver.settings = {'budget': {'seconds': 600, 'wall_seconds': 600, 'max_iter': 3000}}
    captured = []
    solver.request = lambda request, timeout: captured.append((request, timeout)) or {'success': True}
    state = dict(vm=[1., 1.], va_rad=[0., 0.], pg_mw=[42., None], qg_mvar=[3., None])
    solver.solve([1., 1.], [0., 0.], initial_state=state)
    request, timeout = captured[0]
    assert timeout == 600 and request['max_iter'] == 3000
    assert request['initial_state']['pg_mw'] == [42., 0.]
    assert request['initial_state']['qg_mvar'] == [3., 0.]
    assert state['pg_mw'][1] is None and state['qg_mvar'][1] is None


def test_audit_reference_is_independent_of_model_coordinate_reference():
    from power_fm_foresight._data.network import load_network
    from power_fm_foresight._solver.checks import check_solution
    grid = load_network()
    state = dict(vm=np.ones(grid.n_bus), va_rad=np.full(grid.n_bus, .1),
                 pg_mw=np.zeros(grid.n_gen), qg_mvar=np.zeros(grid.n_gen))
    grid.solver_reference_bus = (grid.reference_bus + 1) % grid.n_bus
    state['va_rad'][grid.solver_reference_bus] = 0
    row = dict(state=state, bus_ids=grid.bus_ids.tolist(), success=True)
    result = check_solution(grid, grid.bus[:, 2], grid.bus[:, 3], row)
    assert result['errors']['reference_rad'] == 0
    assert state['va_rad'][grid.reference_bus] == .1
    state['va_rad'] += .05
    assert check_solution(grid, grid.bus[:, 2], grid.bus[:, 3], row)['errors']['reference_rad'] == pytest.approx(.05)
