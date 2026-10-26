"""Grid-aware replication runner using the frozen forecasts, scorer and audits."""
from copy import deepcopy
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import time

from power_fm_foresight.training import mixture_samples
import numpy as np
import torch

from power_fm_foresight._config import ROOT, load_config, read_json, sha256, write_json
from power_fm_foresight.scoring import UNITS, evaluate
from power_fm_foresight._training.forecast import components, adapter_samples
from experiments.forecast_partition import prepare as prepare_forecasts
from power_fm_foresight._training.physics import frozen, target_values
from experiments.evaluate_stage_b import audit_samples
from experiments.diagnose_stage_b import npz, subset_summary
from experiments.replication_evaluate import completion_valid, finish_origin, differences
from power_fm_foresight._training.evaluation import merged
from power_fm_foresight._training.forecaster import RidgeForecaster, episode_arrays, atomic_npz
from power_fm_foresight._training.state import tensor_digest
from experiments.fleet.run import CASES, config_file, chain_root, text_file

MODES = ('C', 'P', 'P0', 'S', 'A', 'D')
HEADLINES = ('joint_es', 'vm_crps', 'vm_coverage90_pct', 'pg_free_mae', 'p_imbalance')


def remaining():
    return float(os.environ['GENCO_FLEET_DEADLINE'])-time.time()


from power_fm_foresight._training.evaluation import score


def headline(row, scales, indices):
    vm = subset_summary(row, 'vm', np.asarray(indices, dtype=int))
    pg = subset_summary(row, 'pg_mw', np.flatnonzero(scales['features']['pg_mw']['joint_score_mask']))
    return dict(joint_es=row['joint_energy_score'], vm_crps=vm['crps'],
        vm_coverage90_pct=None if vm['coverage_90'] is None else 100*vm['coverage_90'],
        pg_free_mae=pg['mae'], p_imbalance=row['physical_balance']['p_mean_abs_mw'])


def released_reference():
    cfg = load_config(ROOT / 'configs/v4.yaml')
    source = Path(cfg['paths']['runs_root']) / 'stage_b'
    return cfg, source, read_json(source / 'protocol.json')


def scorer_check(root):
    path = root / 'scorer_reproduction.json'
    if path.exists():
        assert read_json(path)['passed'] and read_json(path)['scorer_sha256'] == sha256(ROOT / 'scoring.py')
        return
    cfg, source, protocol = released_reference()
    origin = protocol['origins'][0]; folder = source / 'origins' / origin['id']
    scales = read_json(cfg['stage_b']['coordinate_scales']); pairs = npz(source / 'variogram_pairs.npz')['pairs']
    checks = {}; inputs = {}
    for mode in MODES:
        actual, arrays = score(npz(folder / f'{mode}.npz'), npz(folder / 'targets.npz'), scales, pairs, mode, origin['id'], 0)
        expected = read_json(folder / f'{mode}_scores.json')
        check = differences(actual, {k: expected[k] for k in actual}, atol=1e-12, rtol=1e-12)
        stored = npz(folder / f'{mode}_scores.npz')
        check['arrays_exact'] = arrays.keys() == stored.keys() and all(np.array_equal(arrays[k], stored[k], equal_nan=True) for k in arrays)
        check['passed'] &= check['arrays_exact']; checks[mode] = check
        for suffix in ('.npz', '_scores.json', '_scores.npz'):
            p = folder / f'{mode}{suffix}'; inputs[str(p)] = sha256(p)
    result = dict(passed=all(x['passed'] for x in checks.values()), checks=checks, input_sha256=inputs,
        scorer_sha256=sha256(ROOT / 'scoring.py'), scope='First released v4 origin, all six modes and detailed score arrays')
    write_json(path, result)
    assert result['passed'], 'Frozen scorer reproduction failed'


def evaluate_chain(root, c):
    torch.use_deterministic_algorithms(True); torch.set_num_threads(1)
    scorer_check(root)
    cfg = load_config(config_file(root, c['case'], c['seed']))
    cfg = deepcopy(cfg); cfg['seed'] = 42
    out = chain_root(root, c); b = out / 'stage_b'; b.mkdir(parents=True, exist_ok=True)
    protocol_path = b / 'protocol.json'
    reused = c['case'] == 'case30_ieee'
    source = None
    if reused:
        _, source, released = released_reference()
        p = dict(dataset_id=cfg['dataset']['id'], origins=released['origins'], seed=42, training_seed=c['seed'],
            modes=list(MODES), checkpoint_sha256=sha256(out / 'adapter/best.pt'),
            best_epoch=read_json(out / 'adapter/status.json')['best_epoch'],
            coordinate_scales_sha256=sha256(cfg['stage_b']['coordinate_scales']),
            dataset_manifest_sha256=sha256(Path(cfg['paths']['temporal_root']) / 'manifest.json'),
            scorer_sha256=sha256(ROOT / 'scoring.py'), baseline_source=str(source),
            released_protocol_sha256=sha256(source / 'protocol.json'),
            baseline_policy='C/P/P0/S copied read-only by hash; A/D use the new selected checkpoint')
        if protocol_path.exists():
            assert read_json(protocol_path) == p
        else:
            write_json(protocol_path, p)
        cache_manifest = read_json(source / 'cache/manifest.json')
        pairs = npz(source / 'variogram_pairs.npz')['pairs']
        pool = ridge = None
    else:
        if not protocol_path.exists():
            prepare_forecasts(cfg, 'test', 'xpu')
        p = read_json(protocol_path)
        assert p['seed'] == 42 and p['checkpoint_sha256'] == sha256(out / 'adapter/best.pt')
        cache_manifest = read_json(b / 'cache/manifest.json')
        pairs = npz(b / 'variogram_pairs.npz')['pairs']
        pool = npz(p['pool_path']); ridge = RidgeForecaster(p['forecaster_path'])
    if remaining() < 300:
        return
    net, base, model, scales, templates = components(cfg, 'xpu')
    digest = sha256(protocol_path); base_before = tensor_digest(base.model.state_dict())
    cached, episodes = {}, {}; solver = None
    try:
        for index, origin in enumerate(p['origins']):
            folder = b / 'origins' / origin['id']
            if completion_valid(folder, digest):
                continue
            if remaining() < 300:
                return
            folder.mkdir(parents=True, exist_ok=True)
            name, o = origin['episode_id'], origin['origin']
            if name not in cached:
                item = cache_manifest[name]; assert sha256(item['path']) == item['sha256']
                cached[name] = torch.load(item['path'], weights_only=True, mmap=True)['data']
                if not reused:
                    records = read_json(Path(cfg['paths']['temporal_root']) / f'episodes/{name}.json')['records']
                    episodes[name] = (records, episode_arrays(records, net))
            values, inputs = {}, {}
            if reused:
                baseline = source / 'origins' / origin['id']
                files = ['targets.npz'] + [f'{mode}{suffix}' for mode in ('C', 'P', 'P0', 'S') for suffix in ('.npz', '_scores.json', '_scores.npz')]
                for name in files:
                    src = baseline / name; dst = folder / name
                    if not dst.exists():
                        shutil.copyfile(src, dst)
                    assert sha256(src) == sha256(dst); inputs[str(src)] = sha256(src)
                target = npz(folder / 'targets.npz')
                values = {m: npz(folder / f'{m}.npz') for m in ('C', 'P', 'P0', 'S')}
                if not (folder / 'solver_references.json').exists():
                    write_json(folder / 'solver_references.json', {k: v for k, v in read_json(baseline / 'solver_references.json').items() if k.startswith(('P_', 'S_'))})
            else:
                records, arrays = episodes[name]
                target = target_values(records[o:o+24], net)
                mean, samples, _ = ridge.forecast(arrays, o, 10042+index)
                values['P'] = frozen(base, net, mean[None], templates)
                values['S'] = frozen(base, net, samples, templates)
                values['P0'] = {k: np.repeat(v, 64, 0) for k, v in values['P'].items()}
                draws = np.random.default_rng(20042+index).integers(0, 210, size=64)
                values['C'] = {k: v[draws] for k, v in pool.items()}
                atomic_npz(folder / 'targets.npz', **target)
            if not (folder / 'A.npz').exists():
                a, elapsed = adapter_samples(base, model, net, cached[origin['episode_id']], o, templates, 10042+index, 'xpu')
                atomic_npz(folder / 'A.npz', **a)
                write_json(folder / 'inference.json', dict(seconds_A=elapsed, node=os.uname().nodename))
            values['A'] = npz(folder / 'A.npz')
            if not (folder / 'D.npz').exists():
                atomic_npz(folder / 'D.npz', **frozen(base, net, values['A']['pd_mw'], templates, qd=values['A']['qd_mvar']))
            values['D'] = npz(folder / 'D.npz')
            for mode in MODES:
                if reused and mode in ('C', 'P', 'P0', 'S'):
                    continue
                if mode not in ('A', 'D'):
                    atomic_npz(folder / f'{mode}.npz', **values[mode])
                row, arrays = score(values[mode], target, scales, pairs, mode, origin['id'], index)
                write_json(folder / f'{mode}_scores.json', row); atomic_npz(folder / f'{mode}_scores.npz', **arrays)
            if solver is None:
                from power_fm_foresight._solver.adapter import IpoptSolver
                solver = IpoptSolver(cfg, b / 'solver.log')
            audits = audit_samples(folder, values, target, net, solver)
            for mode, audit in audits.items():
                write_json(folder / f'{mode}_audit.json', audit)
            assert tensor_digest(base.model.state_dict()) == base_before
            finish_origin(folder, digest, dict(backbone_unchanged=True, source_inputs_sha256=inputs))
            print({'chain': c['id'], 'origin': origin['id'], 'evaluated': True}, flush=True)
    finally:
        if solver is not None:
            solver.close()
    selected = read_json(root / c['case'] / 'stage_a/stage_a_followup/headline_components.json')
    indices = [net.bus_ids.tolist().index(bus) for bus in selected['voltage_bus_ids']]
    result = dict(training_seed=c['seed'], evaluation_seed=42, headline_components=selected, modes={}, headlines={},
        protocol_sha256=digest, training_manifest_sha256=sha256(out / 'adapter/manifest.json'))
    for mode in MODES:
        row = merged([read_json(b / 'origins' / o['id'] / f'{mode}_scores.json') for o in p['origins']])
        write_json(b / f'{mode}.json', row)
        result['headlines'][mode] = headline(row, scales, indices)
    write_json(b / 'report.json', result)
    write_json(b / 'completion.json', dict(complete=True, report_sha256=sha256(b / 'report.json'),
        completed_utc=datetime.now(timezone.utc).isoformat(), origins=len(p['origins']), modes=list(MODES),
        origin_done_sha256={o['id']: sha256(b / 'origins' / o['id'] / 'done.json') for o in p['origins']}))
    summary(root)


def summary(root):
    results = {}; lines = ['# Cross-grid OFF fleet', '', 'Mean ± sample SD across training seeds 0/1/2. Evaluation noise is fixed at seed 42. No incomplete seed set is labelled three-seed.', '',
        '| Grid | Mode | Joint ES | Vm CRPS | Vm coverage90 % | Free Pg MAE | P imbalance |', '|---|---|---:|---:|---:|---:|---:|']
    for case in CASES:
        paths = [root / case / f'seed{s}/stage_b' for s in (0, 1, 2)]
        if not all((p / 'completion.json').exists() for p in paths):
            lines.append(f'| {case} | pending | — | — | — | — | — |'); continue
        rows = [read_json(p / 'report.json')['headlines'] for p in paths]; stats = {}
        for mode in MODES:
            stats[mode] = {}
            for key in HEADLINES:
                values = [r[mode][key] for r in rows]
                stats[mode][key] = (dict(mean=float(np.mean(values)), sd=float(np.std(values, ddof=1))) if all(v is not None for v in values) else dict(mean=None, sd=None))
            cells = [f"{stats[mode][k]['mean']:.6g} ± {stats[mode][k]['sd']:.4g}" if stats[mode][k]['mean'] is not None else 'N/A (no eligible coordinates)' for k in HEADLINES]
            lines.append('| ' + case + ' | ' + mode + ' | ' + ' | '.join(cells) + ' |')
        results[case] = dict(seeds=[0, 1, 2], per_seed=rows, mean_sd=stats)
        path = root / case / 'mixture/report.json'
        if path.exists():
            mix = read_json(path); results[case]['mixture'] = mix
            for mode, row in mix['modes'].items():
                cells = [f'{row[k]:.6g}' if row[k] is not None else 'N/A' for k in HEADLINES]
                lines.append('| ' + case + ' | mixture ' + mode + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', 'The A/D mixture concatenates 64 trajectories from each of three fresh seeds (192 equal-weight draws), then rescored jointly with all cross-seed pairs. It is not the mean of three ES values. C/P/P0/S are fixed baselines shared across seeds.',
        'Headline buses come only from the mechanical risk-calibration support rule. Different grid scales/coordinates make cross-grid ES descriptive.']
    write_json(root / 'cross_grid.json', results); text_file(root / 'cross_grid.md', '\n'.join(lines)+'\n')


def mixture(root, c):
    cfg = load_config(config_file(root, c['case'], 0)); grid = root / c['case']; out = grid / 'mixture'
    from power_fm_foresight._data.network import load_network
    net = load_network(config=cfg); scales = read_json(cfg['stage_b']['coordinate_scales'])
    b = grid / 'seed0/stage_b'; protocol = read_json(b / 'protocol.json')
    _, released, _ = released_reference()
    pairs = npz((released if c['case'] == 'case30_ieee' else b) / 'variogram_pairs.npz')['pairs']
    p = dict(origins=protocol['origins'], seeds=[0, 1, 2], weights=[1/3]*3, samples=192,
        scorer_sha256=sha256(ROOT / 'scoring.py'), modes=['A', 'D'],
        input_completion_sha256={str(grid / f'seed{s}/stage_b/completion.json'): sha256(grid / f'seed{s}/stage_b/completion.json') for s in (0, 1, 2)})
    if (out / 'protocol.json').exists():
        assert read_json(out / 'protocol.json') == p
    else:
        write_json(out / 'protocol.json', p)
    digest = sha256(out / 'protocol.json')
    for index, origin in enumerate(p['origins']):
        folder = out / 'origins' / origin['id']
        if completion_valid(folder, digest):
            continue
        if remaining() < 180:
            return
        target = npz(b / 'origins' / origin['id'] / 'targets.npz'); hashes = {}
        for mode in ('A', 'D'):
            paths = [grid / f'seed{s}/stage_b/origins' / origin['id'] / f'{mode}.npz' for s in (0, 1, 2)]
            blocks = [npz(path) for path in paths]; values = mixture_samples(blocks)
            row, arrays = score(values, target, scales, pairs, mode, origin['id'], index)
            write_json(folder / f'{mode}_scores.json', row); atomic_npz(folder / f'{mode}_scores.npz', **arrays)
            hashes.update({str(path): sha256(path) for path in paths})
        finish_origin(folder, digest, dict(sample_sources_sha256=hashes))
    selected = read_json(grid / 'stage_a/stage_a_followup/headline_components.json')
    indices = [net.bus_ids.tolist().index(bus) for bus in selected['voltage_bus_ids']]
    result = dict(seeds=[0, 1, 2], samples=192, equal_weights=True, modes={})
    for mode in ('A', 'D'):
        row = merged([read_json(out / 'origins' / o['id'] / f'{mode}_scores.json') for o in p['origins']])
        write_json(out / f'{mode}.json', row); result['modes'][mode] = headline(row, scales, indices)
    write_json(out / 'report.json', result)
    write_json(out / 'completion.json', dict(complete=True, report_sha256=sha256(out / 'report.json')))
    summary(root)
