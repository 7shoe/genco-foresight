"""Pilot isolation, bounded continuation and interrupted shard ownership."""
from types import SimpleNamespace

import pytest

from power_fm_foresight._config import read_json, write_json
from experiments.pilots import run
from experiments.pilots.data import assigned_episodes


def test_two_pilots_are_separate_from_fleet():
    from experiments.fleet.run import chains as fleet_chains
    pilots = run.chains()
    assert [(c['case'], c['seed'], c['tile'], c['K'], c['label']) for c in pilots] == [
        ('case500_goc', 0, 0, 8, 'K=8 pilot'), ('case2000_goc', 0, 2, 2, 'K=2 pilot')]
    assert not {c['case'] for c in pilots} & {c['case'] for c in fleet_chains()}
    episodes = list(range(20))
    partitions = [assigned_episodes(episodes, w) for w in range(4)]
    assert sorted(sum(partitions, [])) == episodes
    assert all(len(p) == 5 and p == sorted(p) for p in partitions)


def test_data_gate_budget_and_pilot_stop(tmp_path):
    c, other = run.chains(); out = run.chain_root(tmp_path, c)
    assert run.task(tmp_path, c) == 'data'
    write_json(tmp_path / c['case'] / 'ready.json', {'passed': True})
    assert run.task(tmp_path, c) == 'train'
    assert run.task(tmp_path, other) == 'data'
    write_json(out / 'adapter/status.json', {'status': 'finished', 'stop_reason': 'max_epochs'})
    assert run.task(tmp_path, c) is None
    write_json(run.chain_root(tmp_path, other) / 'budget.json', {'data_seconds': run.DATA_BUDGET})
    assert run.task(tmp_path, other) is None
    assert (tmp_path / other['case'] / 'blocked.json').exists()


def test_next_allocation_submitted_without_consulting_fleet(tmp_path, monkeypatch):
    import experiments.training_chain as pbs
    monkeypatch.setattr(run, 'verify_pin', lambda root: None)
    monkeypatch.setattr(run, 'reconcile', lambda root, job: None)
    monkeypatch.setattr(pbs, 'job_state', lambda job: ('F', 0))
    write_json(tmp_path / 'protocol.json', {'job': {'max_allocations': 24}})
    write_json(tmp_path / 'controller.json', {'jobs': [{'job_id': '123.pbs', 'number': 1}]})
    captured = []
    def submit(argv, **kwargs):
        captured.append(argv)
        return SimpleNamespace(returncode=0, stdout='124.pbs\n', stderr='')
    monkeypatch.setattr(run.subprocess, 'run', submit)
    run.controller(tmp_path, once=True)
    state = read_json(tmp_path / 'controller.json')
    assert state['jobs'][0]['finished']
    assert state['jobs'][1]['job_id'] == '124.pbs'
    assert captured[0][:3] == ['qsub', '-q', 'debug']
    assert state['jobs'][1]['reserved_node_seconds'] == 3600
    assert all('debug-scaling' not in a for a in captured[0])


def test_missing_boundary_charged_and_parked(tmp_path):
    c = run.chains()[0]; out = run.chain_root(tmp_path, c)
    write_json(out / 'attempts/123_data.json', {'phase': 'data', 'budget_before': {'data_seconds': 10., 'training_seconds': 0.}})
    run.reconcile(tmp_path, {'job_id': '123.pbs'})
    assert read_json(out / 'budget.json')['data_seconds'] == 3610
    assert run.task(tmp_path, c) is None


def test_interrupted_shard_resumes_exact_prefix(tmp_path, monkeypatch):
    import numpy as np
    import pandas as pd
    from experiments.pilots import data
    import experiments.data_generation as generation
    from power_fm_foresight._data import generate as demand_law
    import power_fm_foresight._solver.adapter as adapter
    import power_fm_foresight._solver.checks as checks
    ep = {'episode_id': 'train_00', 'source_start_hour': 0, 'seed': 42}
    manifest = {'benchmark_id': 'fixture', 'episodes': [ep]}
    net = SimpleNamespace(bus=np.array([[0., 0., 1., 2.]]), baseMVA=100.)
    frame = pd.DataFrame({'time_utc': [str(i) for i in range(504)]})
    values = np.arange(504, dtype=float)[:, None]
    monkeypatch.setattr(generation, 'prepare_benchmark', lambda cfg: (tmp_path, manifest, net, frame, values))
    monkeypatch.setattr(demand_law, 'demands', lambda *args: (values, 2*values))
    monkeypatch.setattr(checks, 'check_solution', lambda *args: {'valid': True})
    calls = []
    class Solver:
        versions = {'fixture': True}
        def __init__(self, *args, **kwargs):pass
        def __enter__(self):return self
        def __exit__(self, *args):pass
        def solve(self, pd, qd):
            calls.append(float(pd[0]))
            return {'status': 'SOLVED', 'state': {'vm': [1.]}, 'dual_p_raw': [1.]}
    monkeypatch.setattr(adapter, 'IpoptSolver', Solver)
    def interrupted(seconds=60):
        if len(calls) == 3:raise data.PreparationGuard('fixture restart')
    monkeypatch.setattr(data, 'guard', interrupted)
    write_json(tmp_path / 'pilot_report.json', {'accepted': True})
    write_json(tmp_path / 'audit/solver_smoke.json', {'dual_check': {'valid': True, 'sign': 1}})
    cfg = {'paths': {'stage_a_run': str(tmp_path / 'audit')}, 'generation': {}}
    with pytest.raises(data.PreparationGuard):data.label_shards(cfg, 0)
    path = tmp_path / 'episodes/train_00.json'; prefix = read_json(path)['records']
    assert len(prefix) == 3
    monkeypatch.setattr(data, 'guard', lambda *args: None)
    data.label_shards(cfg, 0)
    shard = read_json(path)
    assert shard['records'][:3] == prefix
    assert calls == list(range(504))
    data.label_shards(cfg, 0)
    assert calls == list(range(504))
    shard['records'][0]['pd_mw'] = [999.]
    write_json(path, shard)
    with pytest.raises(AssertionError):data.label_shards(cfg, 0)
