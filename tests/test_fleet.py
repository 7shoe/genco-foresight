"""Fleet scheduling, immutable inputs and strict per-chain gates."""
from pathlib import Path
import pytest

from power_fm_foresight._config import write_json, sha256
from experiments.fleet.run import chains, task, chain_root
from experiments.fleet.data import headlines


def test_twelve_distinct_slots_and_real_seeds():
    work = chains()
    assert len(work) == 12
    for node in (0, 1):
        assigned = [c for c in work if c['node'] == node]
        assert len(assigned) == 6
        assert sorted(c['tile'] for c in assigned) == [0, 2, 4, 6, 8, 10]
    for case in {c['case'] for c in work}:
        assert sorted(c['seed'] for c in work if c['case'] == case) == [0, 1, 2]


def test_training_waits_for_its_own_audit(tmp_path):
    owner, peer = chains()[:2]
    assert task(tmp_path, owner) == 'data'
    assert task(tmp_path, peer) == 'wait'
    write_json(tmp_path / owner['case'] / 'ready.json', {'passed': True})
    assert task(tmp_path, owner) == task(tmp_path, peer) == 'train'
    write_json(tmp_path / owner['case'] / 'blocked.json', {'reason': 'audit failed'})
    assert task(tmp_path, owner) is None and task(tmp_path, peer) is None
    assert task(tmp_path, chains()[3]) == 'data'


def test_budget_stops_updates_and_evaluation_starts_without_other_seeds(tmp_path):
    c = chains()[0]; out = chain_root(tmp_path, c)
    write_json(tmp_path / c['case'] / 'ready.json', {'passed': True})
    write_json(out / 'budget.json', {'training_seconds': 21600, 'evaluation_seconds': 0})
    (out / 'adapter').mkdir()
    (out / 'adapter/best.pt').write_bytes(b'checkpoint')
    assert task(tmp_path, c) == 'evaluate'
    write_json(out / 'stage_b/completion.json', {'complete': True})
    assert task(tmp_path, c) is None
    for seed in (1, 2):
        write_json(tmp_path / c['case'] / f'seed{seed}/stage_b/completion.json', {'complete': True})
    assert task(tmp_path, c) == 'mixture'


def test_mechanical_headlines_do_not_fall_back_or_read_test_scores():
    events = {'buses': [{'id': 12, 'calibration_hourly_support': {'eligible': True}},
                        {'id': 1, 'calibration_hourly_support': {'eligible': False}}],
              'lines': [{'id': 1, 'rated': False, 'calibration_hourly_support': {'eligible': True}}]}
    assert headlines(events)['voltage_bus_ids'] == [12]
    assert headlines(events)['thermal_line_ids'] == []
    assert headlines({'buses': [], 'lines': []})['voltage_bus_ids'] == []


def test_pin_rejects_source_and_configuration_changes(tmp_path, monkeypatch):
    import experiments.fleet.run as fleet
    snapshot = tmp_path / 'source'; snapshot.mkdir()
    source = snapshot / 'worker.py'; source.write_text('frozen')
    config = tmp_path / 'config.yaml'; config.write_text('seed: 0')
    monkeypatch.setattr(fleet, 'ROOT', snapshot)
    write_json(tmp_path / 'source_pin.json', {'snapshot': str(snapshot), 'files': {'worker.py': sha256(source)},
        'configuration_sha256': {str(config): sha256(config)}, 'protected_sha256': {}})
    fleet.verify_pin(tmp_path)
    config.write_text('seed: 42')
    with pytest.raises(ValueError, match='configuration'):
        fleet.verify_pin(tmp_path)
    config.write_text('seed: 0'); source.write_text('changed')
    with pytest.raises(ValueError, match='source'):
        fleet.verify_pin(tmp_path)
