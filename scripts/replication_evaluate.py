"""Replication-only orchestration around the unchanged released forecasts/scorer.

All writes are confined to runs/replication. Seed zero and baseline artifacts are
read-only. The full saved-array rescore and a seed-zero forward check precede any
new-seed scoring. No fitting, model implementation or scoring formula lives here.
"""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from common import ROOT, load_config, read_json, sha256, write_json
from scoring import UNITS, evaluate, merge_origin_scores
from scripts.replication_chain import settings, paths, guard_release, atomic_text
from scripts.forecast_partition import components, adapter_samples
from scripts.evaluate_stage_b import frozen, audit_samples
from scripts.diagnose_stage_b import (npz, subset_summary, group_diagnostics,
                                     average_errors, audit)
from training.forecaster import atomic_npz
from training.state import tensor_digest

MODES = ('C', 'P', 'P0', 'S', 'A', 'D')
HEADLINES = ('joint_es', 'vm_crps', 'vm_coverage90_pct', 'pg_free_mae', 'p_imbalance')
VM = np.array([1, 9, 11, 13, 15, 21, 22]) - 1


def released(version):
    cfg = load_config(ROOT / ('configs/v3.yaml' if version == 'v3' else 'configs/v4.yaml'))
    b = Path(cfg['paths']['runs_root']) / 'stage_b'
    return cfg, b, read_json(b / 'protocol.json')


def source_folder(version, identity, mode='A'):
    _, b, _ = released(version)
    return b / ('controls/origins' if version == 'v3' and mode in ('C', 'P0') else 'origins') / identity


def score(values, target, scales, pairs, mode, identity, index):
    masks = {k: np.isfinite(target[k]) for k in UNITS}
    seed = settings()[0]['forecast_rng_seed'] + 10000 + index + (10000 if mode in ('C', 'P0') else 0)
    row, arrays = evaluate(values, target, masks, dict(scales=scales, variogram_pairs=pairs,
                          mode=mode, origin_id=identity, seed=seed))
    row['physical_balance'] = {'p_mean_abs_mw': float(np.abs(values['residual_p_mw']).mean()),
                              'q_mean_abs_mvar': float(np.abs(values['residual_q_mvar']).mean())}
    return row, arrays


def headline(row, scales):
    free = np.flatnonzero(scales['features']['pg_mw']['joint_score_mask'])
    vm = subset_summary(row, 'vm', VM)
    pg = subset_summary(row, 'pg_mw', free)
    return {'joint_es': row['joint_energy_score'], 'vm_crps': vm['crps'],
            'vm_coverage90_pct': 100 * vm['coverage_90'], 'pg_free_mae': pg['mae'],
            'p_imbalance': row['physical_balance']['p_mean_abs_mw']}


def merged(rows):
    result = merge_origin_scores(rows)
    result['physical_balance'] = {k: float(np.mean([r['physical_balance'][k] for r in rows]))
                                 for k in rows[0]['physical_balance']}
    return result


def differences(a, b, path='', atol=1e-12, rtol=1e-12):
    """Compare all common scientific fields; retain worst values, not just a boolean."""
    result = dict(passed=True, max_absolute_difference=0., max_relative_difference=0.,
                  numeric_values=0, failures=[])
    def visit(x, y, name):
        if isinstance(y, dict):
            for k, value in y.items():
                if k not in x:
                    result['failures'].append(name + '/' + k + ': missing'); continue
                visit(x[k], value, name + '/' + k)
        elif isinstance(y, list):
            if len(x) != len(y):
                result['failures'].append(name + ': length'); return
            for i, (xx, yy) in enumerate(zip(x, y)): visit(xx, yy, f'{name}/{i}')
        elif isinstance(y, (int, float)) and not isinstance(y, bool):
            delta = abs(float(x) - float(y)); relative = delta / max(abs(float(y)), 1e-12)
            result['numeric_values'] += 1
            result['max_absolute_difference'] = max(result['max_absolute_difference'], delta)
            result['max_relative_difference'] = max(result['max_relative_difference'], relative)
            if delta > atol + rtol * abs(float(y)): result['failures'].append(name)
        elif x != y: result['failures'].append(name)
    visit(a, b, path)
    result['passed'] = not result['failures']
    result['failures'] = result['failures'][:20]
    return result


def completion_valid(folder, protocol_hash):
    done = Path(folder) / 'done.json'
    if not done.exists(): return False
    d = read_json(done)
    assert d['protocol_sha256'] == protocol_hash
    assert all(sha256(Path(folder) / n) == h for n, h in d['artifacts'].items())
    return True


def finish_origin(folder, protocol_hash, extra=None):
    write_json(folder / 'done.json', dict(protocol_sha256=protocol_hash,
        artifacts={p.name: sha256(p) for p in folder.iterdir() if p.is_file() and p.name != 'done.json'},
        **(extra or {})))


def time_left():
    start = os.environ.get('GENCO_FAMILY_JOB_START')
    return 3300 - (time.time() - float(start)) if start else float('inf')


def reproduce_scores(shard=0, workers=1, limit=None):
    """Rescore every released origin/mode, including detailed arrays and headline keys."""
    torch.set_num_threads(1); guard_release()
    _, root = settings(); dest = root / 'case30_ieee/reproduction'
    runner = sha256(__file__)
    tasks = [(v, i, o) for v in ('v3', 'v4') for i, o in enumerate(released(v)[2]['origins'])]
    for n, (version, index, origin) in enumerate(tasks):
        if n % workers != shard or (limit is not None and n >= limit): continue
        if time_left() < 120: return
        cfg, b, _ = released(version); identity = origin['id']
        out = dest / version / 'origins' / (identity + '.json')
        if out.exists():
            prior = read_json(out)
            assert prior['runner_sha256'] == runner
            assert all(sha256(p) == h for p, h in prior['input_sha256'].items())
            assert prior['passed']; continue
        scales = read_json(cfg['stage_b']['coordinate_scales']); pairs = npz(b / 'variogram_pairs.npz')['pairs']
        target_path = source_folder(version, identity) / 'targets.npz'; target = npz(target_path)
        checks, reports, inputs = {}, {}, {str(target_path): sha256(target_path)}
        for mode in MODES:
            folder = source_folder(version, identity, mode)
            raw_path, expected_path = folder / f'{mode}.npz', folder / f'{mode}_scores.json'
            actual, arrays = score(npz(raw_path), target, scales, pairs, mode, identity, index)
            expected = read_json(expected_path)
            # The v3 control wrapper annotates P0 as a point mass; this is not a scorer output.
            if version == 'v3' and mode == 'P0': actual['no_uncertainty'] = True
            check = differences(actual, {k:expected[k] for k in actual})
            check['excluded_wrapper_annotations'] = sorted(set(expected) - set(actual))
            stored_arrays = npz(folder / f'{mode}_scores.npz')
            check['detailed_arrays_exact'] = (arrays.keys() == stored_arrays.keys() and all(
                np.array_equal(arrays[k], stored_arrays[k], equal_nan=True) for k in arrays))
            check['passed'] &= check['detailed_arrays_exact']
            checks[mode] = check; reports[mode] = actual
            for p in (raw_path, expected_path, folder / f'{mode}_scores.npz'): inputs[str(p)] = sha256(p)
        result = dict(passed=all(x['passed'] for x in checks.values()), checks=checks,
                      scores=reports, input_sha256=inputs, runner_sha256=runner)
        write_json(out, result)
        print({'reproduction': version, 'origin': identity, 'passed': result['passed']}, flush=True)
        assert result['passed'], f'Released score reproduction failed: {out}'


def reproduce_forward(version, device):
    """One fixed first origin per dataset through the new inference orchestration."""
    guard_release(); _, root = settings(); out = root / 'case30_ieee/reproduction' / version / 'forward.json'
    if out.exists():
        d = read_json(out); assert d['runner_sha256'] == sha256(__file__) and d['passed']; return
    cfg, b, protocol = released(version); origin = protocol['origins'][0]
    net, base, model, scales, templates = components(cfg, device)
    source = read_json(b / 'cache/manifest.json')[origin['episode_id']]
    assert sha256(source['path']) == source['sha256']
    cached = torch.load(source['path'], mmap=True, weights_only=True)['data']
    vals, _ = adapter_samples(base, model, net, cached, origin['origin'], templates, 10042, device)
    values = {'A': vals, 'D': frozen(base, net, vals['pd_mw'], templates, qd=vals['qd_mvar'])}
    target = npz(source_folder(version, origin['id']) / 'targets.npz')
    pairs = npz(b / 'variogram_pairs.npz')['pairs']; checks = {}
    for mode, value in values.items():
        original = npz(source_folder(version, origin['id']) / f'{mode}.npz')
        all_close = all(np.allclose(value[k], original[k], atol=1e-5, rtol=1e-5, equal_nan=True) for k in original)
        maxima = {k: float(np.nanmax(np.abs(value[k] - original[k]))) for k in original}
        now, _ = score(value, target, scales, pairs, mode, origin['id'], 0)
        old = read_json(source_folder(version, origin['id']) / f'{mode}_scores.json')
        h = differences(headline(now, scales), headline(old, scales), atol=1e-8, rtol=1e-5)
        checks[mode] = dict(arrays_within_tolerance=all_close, max_absolute_by_variable=maxima,
                            headline_check=h, passed=all_close and h['passed'])
    result = dict(passed=all(x['passed'] for x in checks.values()), checks=checks,
        origin=origin, device=device, node=os.uname().nodename, runner_sha256=sha256(__file__),
        scope='First of 72 origins per dataset, A/D full forward; all 72 are rescored separately.',
        tolerances={'array_atol':1e-5,'array_rtol':1e-5,'headline_atol':1e-8,'headline_rtol':1e-5},
        note='Tolerance comparison across nodes, not a claim of same-node bit-identical forward replay.')
    write_json(out, result); assert result['passed'], f'Forward reproduction failed: {out}'


def reproduction_merge():
    _, root = settings(); out = root / 'case30_ieee/reproduction'; report = out / 'report.json'
    if report.exists():
        prior = read_json(report)
        assert prior['passed'] and prior['runner_sha256'] == sha256(__file__)
        return True
    from scripts.headline_numbers import collect
    expected = collect(root.parent)['headlines']; checks = {}; maxrel = maxabs = 0.
    all_inputs = {}; numeric = 0
    for version in ('v3', 'v4'):
        cfg, _, p = released(version); scales = read_json(cfg['stage_b']['coordinate_scales'])
        files = [out / version / 'origins' / (o['id'] + '.json') for o in p['origins']]
        forward = out / version / 'forward.json'
        if not all(x.exists() for x in files) or not forward.exists(): return False
        rows = [read_json(x) for x in files]
        fwd = read_json(forward)
        assert fwd['passed'] and fwd['runner_sha256'] == sha256(__file__)
        for row in rows:
            assert row['passed'] and row['runner_sha256'] == sha256(__file__)
            for c in row['checks'].values():
                maxrel = max(maxrel, c['max_relative_difference']); maxabs = max(maxabs, c['max_absolute_difference'])
                numeric += c['numeric_values']
        for mode in MODES:
            actual = headline(merged([r['scores'][mode] for r in rows]), scales)
            desired = {k: expected[f'{version}.{mode}.{k}']['value'] for k in HEADLINES}
            check = differences(actual, desired); checks[f'{version}.{mode}'] = check
            maxrel = max(maxrel, check['max_relative_difference']); maxabs = max(maxabs, check['max_absolute_difference'])
        all_inputs.update({str(x): sha256(x) for x in [*files, forward]})
    result = dict(passed=all(c['passed'] for c in checks.values()), runner_sha256=sha256(__file__),
        max_relative_difference=maxrel, max_absolute_difference=maxabs, numeric_values_compared=numeric,
        relative_denominator='max(abs(released value), 1e-12)',
        rescored_origins_per_dataset=72, modes=list(MODES), datasets=['v3','v4'],
        forward_checks={v: str(out / v / 'forward.json') for v in ('v3','v4')},
        headline_checks=checks, input_sha256=all_inputs, completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(report, result); print(result, flush=True)
    assert result['passed'], 'Headline reproduction failed'
    return True


def require_gate():
    _, root = settings(); path = root / 'case30_ieee/reproduction/report.json'
    gate = read_json(path)
    assert gate['passed'] and gate['runner_sha256'] == sha256(__file__), 'Reproduction gate not passed for this runner'
    return path


def chain_protocol(chain):
    gate = require_gate(); out, train, config = paths(chain); cfg = load_config(config)
    _, source, old = released(chain['version']); state = read_json(train / 'status.json')
    assert state['status'] == 'finished' or read_json(out / 'budget.json')['used_seconds'] >= 21585
    p = dict(dataset_id=cfg['dataset']['id'], origins=old['origins'], training_seed=chain['seed'],
        evaluation_seed=42, sample_counts={m: 1 if m == 'P' else 64 for m in MODES},
        checkpoint_sha256=sha256(train / 'best.pt'), best_epoch=state['best_epoch'],
        coordinate_scales_sha256=sha256(cfg['stage_b']['coordinate_scales']),
        dataset_manifest_sha256=sha256(Path(cfg['paths']['temporal_root']) / 'manifest.json'),
        released_protocol_sha256=sha256(source / 'protocol.json'),
        reproduction_gate_sha256=sha256(gate), runner_sha256=sha256(__file__),
        scorer_sha256=sha256(ROOT / 'scoring.py'), baseline_source=str(source),
        baseline_policy='C/P/P0/S frozen seed-0 predictions and solver references reused read-only by hash.',
        same_input_audit='Unchanged fixed samples 0,32 and leads 0,6,12,18,23; A and D share numerical solutions.')
    b = out / 'stage_b'; path = b / 'protocol.json'
    if path.exists(): assert read_json(path) == p
    else: write_json(path, p)
    return cfg, b, p


def evaluate_chain(identity, device, shard=0, workers=1):
    f, _ = settings(); chain = next(c for c in f['chains'] if c['id'] == identity)
    guard_release(); cfg, b, p = chain_protocol(chain); version = chain['version']
    net, base, model, scales, templates = components(cfg, device)
    base_before = tensor_digest(base.model.state_dict())
    _, source, _ = released(version); pairs = npz(source / 'variogram_pairs.npz')['pairs']
    manifest = read_json(source / 'cache/manifest.json'); cached = {}; solver = None
    digest = sha256(b / 'protocol.json')
    try:
        for index, origin in enumerate(p['origins']):
            if index % workers != shard: continue
            if time_left() < 180: return
            folder = b / 'origins' / origin['id']
            if completion_valid(folder, digest): continue
            folder.mkdir(parents=True, exist_ok=True)
            baseline = source_folder(version, origin['id']); inputs = {}
            # Copy, never link, to prevent audit_samples from writing into released paths.
            files = [('targets.npz', baseline)]
            for mode in ('C','P','P0','S'):
                sf = source_folder(version, origin['id'], mode)
                files += [(f'{mode}{suffix}', sf) for suffix in ('.npz','_scores.json','_scores.npz')]
            for name, sf in files:
                src, dst = sf / name, folder / name
                if not dst.exists(): shutil.copy2(src, dst)
                assert sha256(dst) == sha256(src); inputs[str(src)] = sha256(src)
            target = npz(folder / 'targets.npz'); name = origin['episode_id']
            if name not in cached:
                c = manifest[name]; assert sha256(c['path']) == c['sha256']
                cached[name] = torch.load(c['path'], weights_only=True, mmap=True)['data']
            values = {m: npz(folder / f'{m}.npz') for m in ('P', 'S')}
            if not (folder / 'A.npz').exists():
                a, elapsed = adapter_samples(base, model, net, cached[name], origin['origin'], templates, 10042 + index, device)
                atomic_npz(folder / 'A.npz', **a)
                write_json(folder / 'inference.json', {'seconds_A':elapsed,'node':os.uname().nodename,'device':device})
            values['A'] = npz(folder / 'A.npz')
            if not (folder / 'D.npz').exists():
                d = frozen(base, net, values['A']['pd_mw'], templates, qd=values['A']['qd_mvar'])
                atomic_npz(folder / 'D.npz', **d)
            values['D'] = npz(folder / 'D.npz')
            for m in ('A','D'):
                row, arrays = score(values[m], target, scales, pairs, m, origin['id'], index)
                write_json(folder / f'{m}_scores.json', row); atomic_npz(folder / f'{m}_scores.npz', **arrays)
            reference_path = folder / 'solver_references.json'
            if not reference_path.exists():
                old = read_json(baseline / 'solver_references.json')
                write_json(reference_path, {k:v for k,v in old.items() if k.startswith(('P_','S_'))})
            if solver is None:
                from solver.adapter import IpoptSolver
                solver = IpoptSolver(cfg, b / f'logs/solver_{shard}.log')
            audits = audit_samples(folder, values, target, net, solver)
            free = np.flatnonzero(scales['features']['pg_mw']['joint_score_mask'])
            detail = {}
            for m in MODES:
                vals = values[m] if m in values else npz(folder / f'{m}.npz')
                groups = group_diagnostics(vals, target, scales, np.ones(24, bool))
                stored = read_json(folder / f'{m}_scores.json')
                np.testing.assert_allclose(groups['joint_energy_from_contributions'], stored['joint_energy_score'], atol=1e-11, rtol=1e-10)
                detail[m] = {'groups':groups}
                if m in audits:
                    write_json(folder / f'{m}_audit.json', audits[m])
                    detail[m]['same_input'] = audit(vals, npz(folder / f'{m}_solver_subset.npz'), target, m, np.ones(24,bool), free)
            write_json(folder / 'diagnostics.json', detail)
            assert tensor_digest(base.model.state_dict()) == base_before
            finish_origin(folder, digest, {'backbone_unchanged':True,'sample_dependent_demand_verified':True,
                                          'source_inputs_sha256':inputs})
            print({'chain':identity,'origin':origin['id'],'complete':True},flush=True)
    finally:
        if solver is not None: solver.close()


def merge_chain(identity):
    f, _ = settings(); chain = next(c for c in f['chains'] if c['id'] == identity)
    cfg, b, p = chain_protocol(chain); digest = sha256(b / 'protocol.json')
    folders = [b / 'origins' / o['id'] for o in p['origins']]
    if not all(completion_valid(x,digest) for x in folders): return False
    scales = read_json(cfg['stage_b']['coordinate_scales']); free = np.flatnonzero(scales['features']['pg_mw']['joint_score_mask'])
    report = dict(protocol=p, modes={}, headlines={}, verification={'stage_a':True,'released_case30_parity_reused':True,
        'seed0_reproduction':True,'origin_hashes_verified':True,'frozen_backbone':True,
        'sample_dependent_demand':True,'forecast_and_approximation_separate':True})
    diagnostics = [read_json(x/'diagnostics.json') for x in folders]
    for mode in MODES:
        result = merged([read_json(x / f'{mode}_scores.json') for x in folders])
        write_json(b / f'{mode}.json', result)
        report['headlines'][mode] = headline(result, scales)
        result.pop('per_origin'); result['headline_vm'] = subset_summary(result,'vm',VM)
        result['free_generator_pg'] = subset_summary(result,'pg_mw',free)
        result['family_energy'] = {g:{key:float(np.mean([d[mode]['groups']['groups'][g][key] for d in diagnostics
                                                       if d[mode]['groups']['groups'][g][key] is not None]))
                                if any(d[mode]['groups']['groups'][g][key] is not None for d in diagnostics) else None
                                for key in ('group_energy_score','joint_energy_contribution')}
                                for g in diagnostics[0][mode]['groups']['groups']}
        if mode in ('P','S','A','D'):
            a = [d[mode]['same_input'] for d in diagnostics]
            result['same_input'] = {'attempts':sum(x['attempts'] for x in a),'resolved':sum(x['resolved'] for x in a),
                'errors':average_errors([x['errors'] for x in a]),
                'raw_vs_refined':{k:average_errors([x['raw_vs_refined'][k] for x in a]) for k in ('raw','solver_refined')}}
        report['modes'][mode] = result
    write_json(b / 'report.json', report)
    write_json(b / 'completion.json', {'complete':True,'completed_utc':datetime.now(timezone.utc).isoformat(),
        'report_sha256':sha256(b / 'report.json'),'origins':72,
        'origin_done_sha256':{x.name:sha256(x/'done.json') for x in folders}})
    return True


def seed_folder(version, seed, origin, mode='A'):
    if seed == 0: return source_folder(version, origin, mode)
    _, root = settings(); return root / 'case30_ieee' / version / f'seed{seed}/stage_b/origins' / origin


def weights(rows, scales):
    free = np.flatnonzero(scales['features']['pg_mw']['joint_score_mask'])
    return np.array([[1, sum(np.array(r['variables']['vm']['by_component']['count'])[VM]),
        sum(np.array(r['variables']['vm']['by_component']['count'])[VM]),
        sum(np.array(r['variables']['pg_mw']['by_component']['count'])[free]), 1] for r in rows], dtype=float)


def summary(version):
    gate = require_gate(); f, root = settings(); cfg, _, p = released(version)
    dest = root / 'case30_ieee' / version; scales = read_json(cfg['stage_b']['coordinate_scales'])
    seeds = [0] + [i for i in (1,2) if (dest / f'seed{i}/stage_b/completion.json').exists()]
    if len(seeds) == 1: return
    rows = {seed:{m:[read_json(seed_folder(version,seed,o['id'],m)/f'{m}_scores.json')
                    for o in p['origins']] for m in MODES} for seed in seeds}
    seed_stats = {str(seed):{m:headline(merged(r),scales) for m,r in modes.items()} for seed,modes in rows.items()}
    rng = np.random.default_rng(f['bootstrap']['seed']); draws = rng.integers(0,72,size=(f['bootstrap']['draws'],72))
    pairs = {}; allweights = weights(rows[0]['A'],scales)
    for baseline in ('S','D'):
        delta = np.mean([np.array([[headline(a,scales)[k]-headline(b,scales)[k] for k in HEADLINES]
                        for a,b in zip(rows[s]['A'],rows[s][baseline])]) for s in seeds],axis=0)
        w = allweights[draws]; sampled = (delta[draws]*w).sum(1)/w.sum(1)
        estimate = (delta*allweights).sum(0)/allweights.sum(0)
        pairs['A-'+baseline] = {k:{'difference':float(estimate[j]),'ci95':np.quantile(sampled[:,j],[.025,.975]).tolist()}
                               for j,k in enumerate(HEADLINES)}
    mean_sd = {m:{k:{'mean':float(np.mean([seed_stats[str(s)][m][k] for s in seeds])),
                         'sd':float(np.std([seed_stats[str(s)][m][k] for s in seeds],ddof=1))}
                   for k in HEADLINES} for m in MODES}
    result = dict(completed_seeds=seeds, all_three_complete=len(seeds)==3, per_seed=seed_stats,
        mean_sd=mean_sd, paired_origin_bootstrap=pairs, bootstrap=f['bootstrap'],
        seed_mapping=f['seed_mapping'], reproduction_gate_sha256=sha256(gate),
        qualification='Training seed variability only; baselines and evaluation random draws are held fixed. CI pools seeds first and resamples paired origins; not a confidence interval over three training seeds.')
    write_json(dest / 'summary.json',result)
    lines=[f'## {version}: completed evaluation seeds {seeds}','',
        '| Mode | Joint ES mean ± SD | Vm CRPS mean ± SD | Vm coverage90 % mean ± SD | Free Pg MAE mean ± SD | P imbalance mean ± SD |',
        '|---|---:|---:|---:|---:|---:|']
    for m in MODES:
        lines.append('| '+m+' | '+' | '.join(f"{mean_sd[m][k]['mean']:.6g} ± {mean_sd[m][k]['sd']:.4g}" for k in HEADLINES)+' |')
    lines += ['', '| Paired difference | Metric | Estimate | Descriptive 95% origin CI |','|---|---|---:|---|']
    for label, metrics in pairs.items():
        for k,v in metrics.items(): lines.append(f"| {label} | {k} | {v['difference']:.6g} | {v['ci95']} |")
    lines += ['', f['bootstrap']['qualification'], '', 'Per-seed values and all modes: '+str(dest/'summary.json')]
    atomic_text(dest/'summary.md','\n'.join(lines)+'\n')


def mixture(version, shard=0, workers=1):
    gate = require_gate(); _, root = settings(); cfg,b,p = released(version)
    out = root/'case30_ieee'/version/'mixture'; scales=read_json(cfg['stage_b']['coordinate_scales'])
    assert all((root/'case30_ieee'/version/f'seed{i}/stage_b/completion.json').exists() for i in (1,2))
    protocol = dict(origins=p['origins'],seeds=[0,1,2],samples=192,weights='equal',
        runner_sha256=sha256(__file__),gate_sha256=sha256(gate),scorer_sha256=sha256(ROOT/'scoring.py'),
        qualification=settings()[0]['ensemble']['qualification'])
    if not (out/'protocol.json').exists(): write_json(out/'protocol.json',protocol)
    assert read_json(out/'protocol.json') == protocol
    digest=sha256(out/'protocol.json');pairs=npz(b/'variogram_pairs.npz')['pairs']
    for i,o in enumerate(p['origins']):
        if i%workers!=shard:continue
        if time_left()<180:return
        folder=out/'origins'/o['id']
        if completion_valid(folder,digest):continue
        target=npz(source_folder(version,o['id'])/'targets.npz'); source_hashes={}
        for m in ('A','D'):
            files=[seed_folder(version,s,o['id'],m)/f'{m}.npz' for s in (0,1,2)]
            blocks=[npz(x) for x in files]; values={k:np.concatenate([v[k] for v in blocks]) for k in blocks[0]}
            row,arrays=score(values,target,scales,pairs,m,o['id'],i)
            write_json(folder/f'{m}_scores.json',row);atomic_npz(folder/f'{m}_scores.npz',**arrays)
            source_hashes.update({str(x):sha256(x) for x in files})
        finish_origin(folder,digest,{'sample_sources_sha256':source_hashes})
    folders=[out/'origins'/o['id'] for o in p['origins']]
    if all(completion_valid(x,digest) for x in folders):
        result={'variant':'equal-weight three-seed 192-draw mixture','modes':{},'qualification':protocol['qualification']}
        for m in ('A','D'):
            r=merged([read_json(x/f'{m}_scores.json') for x in folders]);write_json(out/f'{m}.json',r)
            result['modes'][m]=headline(r,scales)
        write_json(out/'report.json',result)
        write_json(out/'completion.json',{'complete':True,'completed_utc':datetime.now(timezone.utc).isoformat(),
                   'report_sha256':sha256(out/'report.json')})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['reproduce-scores','reproduce-forward','reproduce-merge','evaluate','merge','summary','mixture'])
    p.add_argument('--version',choices=['v3','v4']);p.add_argument('--chain');p.add_argument('--device',default='xpu')
    p.add_argument('--shard',type=int,default=0);p.add_argument('--workers',type=int,default=1);p.add_argument('--limit',type=int)
    a=p.parse_args();torch.set_num_threads(1)
    if a.action=='reproduce-scores':reproduce_scores(a.shard,a.workers,a.limit)
    elif a.action=='reproduce-forward':reproduce_forward(a.version,a.device)
    elif a.action=='reproduce-merge':reproduction_merge()
    elif a.action=='evaluate':evaluate_chain(a.chain,a.device,a.shard,a.workers)
    elif a.action=='merge':merge_chain(a.chain)
    elif a.action=='summary':summary(a.version)
    else:mixture(a.version,a.shard,a.workers)


if __name__=='__main__':main()
