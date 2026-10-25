"""Offline labels and independent evaluation admission must preserve their inputs."""
from copy import deepcopy

import numpy as np
import pytest

from common import read_json, sha256, write_json
from data.loader.network import load_network
from solver.checks import check_solution, physical_quantities


@pytest.mark.parametrize('missing', [None, np.nan])
def test_offline_generators_are_masked_before_arithmetic(missing):
    net = load_network()
    net.gen[0, 7] = 0
    state = dict(vm=np.ones(net.n_bus), va_rad=np.zeros(net.n_bus),
                 pg_mw=[10.] * net.n_gen, qg_mvar=[2.] * net.n_gen)
    expected = physical_quantities(net, net.bus[:, 2], net.bus[:, 3], state)
    state['pg_mw'][0] = missing; state['qg_mvar'][0] = missing
    actual = physical_quantities(net, net.bus[:, 2], net.bus[:, 3], state)
    for key in actual:
        np.testing.assert_array_equal(actual[key], expected[key])
    checked = check_solution(net, net.bus[:, 2], net.bus[:, 3],
                             dict(state=state, bus_ids=net.bus_ids.tolist(), success=True))
    assert np.isfinite(checked['objective_recomputed'])
    assert all(np.isfinite(v) for v in checked['errors'].values())
    state['pg_mw'][1] = None
    assert check_solution(net, net.bus[:, 2], net.bus[:, 3],
                          dict(state=state, bus_ids=net.bus_ids.tolist(), success=True))['reason'] == 'nonfinite_primal_state'


def test_evaluation_admits_only_finished_private_copies(tmp_path, monkeypatch):
    from scripts.recovery import run
    original = tmp_path / 'original'
    source = original / 'case30_ieee/seed0/adapter'; source.mkdir(parents=True)
    out = tmp_path / 'recovery'; (out / 'case30_ieee/seed0').mkdir(parents=True)
    monkeypatch.setattr(run, 'FLEET', original)
    c = dict(case='case30_ieee', seed=0)
    write_json(source / 'status.json', dict(status='running'))
    assert not run.admit(out, c)
    write_json(source / 'status.json', dict(status='finished'))
    for name in ('best.pt', 'manifest.json', 'epochs.json'):
        (source / name).write_bytes(b'fixture')
    before = {p.name: (sha256(p), p.stat().st_mtime_ns) for p in source.iterdir()}
    assert run.admit(out, c)
    assert before == {p.name: (sha256(p), p.stat().st_mtime_ns) for p in source.iterdir()}
    run.input_check(out, c)
    assert (out / 'case30_ieee/seed0/adapter/best.pt').stat().st_ino != (source / 'best.pt').stat().st_ino
    (source / 'best.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Read-only input changed'):run.input_check(out, c)


def test_unindexed_xpu_uses_launcher_local_zero(monkeypatch):
    from scripts import forecast_partition as fp
    monkeypatch.setenv('ZE_FLAT_DEVICE_HIERARCHY', 'FLAT')
    monkeypatch.setenv('ZE_AFFINITY_MASK', '11')
    monkeypatch.setattr(fp.torch.xpu, 'device_count', lambda: 1)
    calls = []
    monkeypatch.setattr(fp.torch.xpu, 'set_device', calls.append)
    def stop(**kwargs):raise LookupError('device selection reached')
    monkeypatch.setattr(fp, 'load_network', stop)
    with pytest.raises(LookupError):fp.components({}, 'xpu')
    assert str(calls[0]) == 'xpu:0'
    monkeypatch.setattr(fp.torch.xpu, 'device_count', lambda: 12)
    with pytest.raises(ValueError, match='launcher-isolated'):fp.components({}, 'xpu')


def test_existing_label_recovery_cannot_generate():
    from data.synthetic.generate import label
    with pytest.raises(RuntimeError, match='regeneration is forbidden'):
        label(dict(pilot=dict(reuse_labels={'source_root': 'read-only'})))
