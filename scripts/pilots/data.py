"""Disjoint Ipopt episode shards followed by the full temporal Stage A audit."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from common import ROOT, load_config, read_json, sha256, write_json
from scripts.fleet.data import headlines
from scripts.pilots.run import config_file


class PreparationGuard(Exception):
    pass


def guard(seconds=60):
    if time.time() >= float(os.environ['GENCO_PILOTS_DEADLINE'])-seconds:
        raise PreparationGuard('Clean data preparation boundary; continue next allocation')


def command(module, cfg_path, log, *args):
    guard()
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, GENCO_CONFIG=str(cfg_path))
    for key in list(env):
        if key.startswith(('PMI_', 'PMIX_', 'PALS_', 'OMPI_', 'SLURM_')) or key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'NODE_RANK', 'MASTER_ADDR', 'MASTER_PORT'):
            env.pop(key)
    with log.open('a') as stream:
        process = subprocess.Popen([sys.executable, '-u', '-m', module, *args], cwd=ROOT, env=env,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=max(1, float(env['GENCO_PILOTS_DEADLINE'])-time.time()-30))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait()
            raise PreparationGuard(f'Allocation boundary during {module}; saved artifacts retained')
    if code == 75:
        raise PreparationGuard(f'{module} reached its allocation boundary')
    if code:
        raise RuntimeError(f'{module} failed ({code}); see {log}')


def assigned_episodes(episodes, worker, workers=4):
    if workers < 1 or not 0 <= worker < workers:
        raise ValueError('Invalid shard worker assignment')
    return episodes[worker::workers]


def validate_prefix(shard, manifest, episode, pd, qd, frame):
    import numpy as np
    if shard['benchmark_id'] != manifest['benchmark_id'] or shard['episode_id'] != episode['episode_id']:
        raise ValueError('Shard provenance changed')
    records = shard['records']
    if len(records) > 504 or [r['hour'] for r in records] != list(range(len(records))):
        raise ValueError('Noncontiguous or duplicate shard hours')
    for row in records:
        hour = row['hour']
        np.testing.assert_array_equal(row['pd_mw'], pd[hour])
        np.testing.assert_array_equal(row['qd_mvar'], qd[hour])
        if row['time_utc'] != str(frame.time_utc.iloc[episode['source_start_hour']+hour]):
            raise ValueError('Shard timestamp changed')


def label_shards(cfg, worker):
    import numpy as np
    from data.synthetic.generate import demands, prepare_benchmark
    from solver.adapter import IpoptSolver
    from solver.checks import check_solution
    data, manifest, network, frame, scaled = prepare_benchmark(cfg)
    smoke = read_json(Path(cfg['paths']['stage_a_run']) / 'solver_smoke.json')
    if not read_json(data / 'pilot_report.json')['accepted'] or not smoke['dual_check']['valid']:
        raise ValueError('Solver and pilot prerequisites must pass')
    with IpoptSolver(cfg, data / f'julia_worker{worker}.log', network=network) as solver:
        for episode in assigned_episodes(manifest['episodes'], worker):
            guard()
            name = episode['episode_id']; path = data / 'episodes' / f'{name}.json'
            shard = read_json(path) if path.exists() else dict(benchmark_id=manifest['benchmark_id'],
                episode_id=name, versions=solver.versions, records=[])
            lo = episode['source_start_hour']
            pd, qd = demands(scaled[lo:lo+504], network.bus[:, 2], network.bus[:, 3],
                cfg['generation'], episode['seed'])
            validate_prefix(shard, manifest, episode, pd, qd, frame)
            try:
                for hour in range(len(shard['records']), 504):
                    guard()
                    result = solver.solve(pd[hour], qd[hour])
                    unresolved = result['status'] == 'WORKER_UNRESOLVED'
                    result['checks'] = {'valid': False, 'reason': 'worker_unresolved'} if unresolved else check_solution(network, pd[hour], qd[hour], result)
                    valid = result['checks']['valid']
                    dual_valid = bool(valid and 'dual_p_raw' in result and np.isfinite(result['dual_p_raw']).all())
                    if dual_valid:
                        result['dual_p_cost_per_mw'] = (np.asarray(result['dual_p_raw'])*smoke['dual_check']['sign']/network.baseMVA).tolist()
                        result['dual_convention'] = 'one-hour benchmark objective; raw nodal dual times verified sign/baseMVA'
                    shard['records'].append(dict(hour=hour, time_utc=str(frame.time_utc.iloc[lo+hour]),
                        pd_mw=pd[hour].tolist(), qd_mvar=qd[hour].tolist(), state_available=valid,
                        duals_available=dual_valid, target_state=result.get('state') if valid else None, solver=result))
                    if (hour+1) % 25 == 0:
                        write_json(path, shard)
                        print(name, hour+1, flush=True)
                    if unresolved:
                        raise RuntimeError('Solver worker unresolved; park this grid without retry')
            finally:
                write_json(path, shard)


def support_report(cfg, audit):
    import numpy as np
    from data.synthetic.generate import prepare_benchmark, demands
    _, descriptor, net, _, scaled = prepare_benchmark(cfg)
    support = {}
    for split in cfg['task']['split_episodes']:
        profiles, multipliers = [], []
        for episode in descriptor['episodes']:
            if episode['split'] != split:
                continue
            ref = scaled[episode['source_start_hour']:episode['source_start_hour']+504]
            pd, _ = demands(ref, net.bus[:, 2], net.bus[:, 3], cfg['generation'], episode['seed'])
            profiles.extend(ref); multipliers.append(pd[:, net.active_demand]/net.bus[net.active_demand, 2])
        values = np.concatenate(multipliers); refs = np.asarray(profiles)
        support[split] = dict(profile_min=float(refs.min()), profile_max=float(refs.max()),
            actual_bus_multiplier_min=float(values.min()), actual_bus_multiplier_max=float(values.max()),
            fraction_below_0p8=float((values < .8).mean()), fraction_above_1p2=float((values > 1.2).mean()))
    write_json(audit / 'support_diagnostic.json', support)


def prepare_grid(root, case, cfg):
    import torch
    torch.use_deterministic_algorithms(True)
    grid = root / case; audit = Path(cfg['paths']['stage_a_run']); audit.mkdir(parents=True, exist_ok=True)
    cfg_path = config_file(root, case); data = Path(cfg['paths']['temporal_root'])
    if cfg['pilot'].get('reuse_labels'):
        required = (data / 'pilot_report.json', data / 'generation_report.json',
                    audit / 'solver_smoke.json', audit / 'parity/conversion_check.json', audit / 'parity/parity_report.json')
        if not all(path.is_file() for path in required):
            raise RuntimeError('Existing-label recovery requires complete generation and parity evidence; no regeneration')
    if not (audit / 'environment_before.txt').exists():
        (audit / 'environment_before.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    parity = audit / 'parity'
    if not (parity / 'conversion_check.json').exists():
        command('scripts.parity', cfg_path, audit / 'logs/parity_prepare.log', '--config', str(cfg_path),
            '--prepare', '--download', '--limit', '32', '--batch-size', '1')
    if not (parity / 'parity_report.json').exists():
        command('scripts.parity', cfg_path, audit / 'logs/parity_evaluate.log', '--config', str(cfg_path))
    assert read_json(parity / 'parity_report.json')['passed']
    if not (audit / 'solver_smoke.json').exists():
        command('scripts.solver_smoke', cfg_path, audit / 'logs/solver_smoke.log', '--config', str(cfg_path))
    smoke = read_json(audit / 'solver_smoke.json')
    assert smoke['cold']['checks']['valid'] and smoke['warm']['checks']['valid'] and smoke['dual_check']['valid']
    from data.synthetic.generate import label, prepare_benchmark
    guard()
    if not (data / 'pilot_report.json').exists():
        if cfg['pilot'].get('solver_processes'):
            from scripts.pilots.hour_shards import generate
            generate(cfg, cfg_path, 200)
            result = read_json(data / 'pilot_report.json')
        else:
            result = label(cfg, 200, deadline_epoch=float(os.environ['GENCO_PILOTS_DEADLINE'])-60)
        if result.get('incomplete'):
            raise PreparationGuard('200-hour data pilot continues next allocation')
    assert read_json(data / 'pilot_report.json')['accepted']
    if (data / 'formulation_review_required.json').exists():
        raise RuntimeError('Recorded formulation review must be resolved before expansion')
    projection = data / 'generation_projection.json'
    if projection.exists() and read_json(projection)['projected_solve_wall_seconds'] > 10800:
        raise RuntimeError('Recorded full-size projection requires a new pilot half-size identity')
    if cfg['pilot'].get('solver_processes') and not (data / 'generation_report.json').exists():
        from scripts.pilots.hour_shards import generate
        generate(cfg, cfg_path, 10080)
    if not (data / 'generation_report.json').exists():
        prepare_benchmark(cfg)
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(command, 'scripts.pilots.data', cfg_path, audit / f'logs/shard_worker{w}.log',
                'shards', '--worker', str(w)) for w in range(4)]
            errors = []
            for future in futures:
                try:
                    future.result()
                except Exception as error:
                    errors.append(error)
            failures = [e for e in errors if not isinstance(e, PreparationGuard)]
            if failures:raise failures[0]
            if errors:raise errors[0]
        command('scripts.pilots.data', cfg_path, audit / 'logs/merge.log', 'merge')
    if not (data / 'quality_report.json').exists():
        command('scripts.data_audit', cfg_path, audit / 'logs/data_audit.log')
    if not (audit / 'events_complete.json').exists():
        command('scripts.pilots.data', cfg_path, audit / 'logs/events.log', 'events')
    if not (audit / 'stage_a.json').exists():
        shutil.copyfile(root / 'tests.xml', audit / 'tests.xml')
        command('scripts.collect_stage_a', cfg_path, audit / 'logs/collect.log', '--snapshot')
    summary = read_json(audit / 'stage_a.json')
    assert summary['complete']
    summary['limitations'] = [f'{case}: fixed topology, costs and commitment; synthetic ERCOT profile mapping.',
        'CPU subset parity (200): seeded official/local test intersection, byte-identical; independent conversion checked separately on32 snapshots.',
        'All10080 temporal hours independently audited, including72 test origins; solver failures retain missing-label masks.',
        'Pilot only; no fleet evaluation, cross-grid result or Stage2 correctness claim.']
    write_json(audit / 'stage_a.json', summary)
    write_json(audit / 'readiness.json', dict(stage_a_audit_passed=True, training_allowed=True,
        dataset_id=cfg['dataset']['id'], manifest_sha256=sha256(data / 'manifest.json'),
        stage_a_sha256=sha256(audit / 'stage_a.json'), prevalence_is_training_blocker=False))
    guard()
    if not (grid / 'shared/cache/manifest.json').exists():
        from training.inputs import build_cache
        result = build_cache(cfg, 'xpu', deadline_epoch=float(os.environ['GENCO_PILOTS_DEADLINE'])-30)
        if result.get('incomplete'):
            raise PreparationGuard('Embedding cache resumes at the next completed episode')
    events = read_json(audit / 'stage_a_followup/events.json')
    write_json(audit / 'stage_a_followup/headline_components.json', headlines(events))
    out = grid / 'seed0/adapter/cache'; out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(grid / 'shared/cache/manifest.json', out / 'manifest.json')
    paths = [data / 'manifest.json', audit / 'readiness.json', audit / 'stage_a.json',
        audit / 'stage_a_followup/events.json', audit / 'stage_a_followup/headline_components.json',
        Path(cfg['stage_b']['coordinate_scales']), grid / 'shared/cache/manifest.json']
    write_json(grid / 'ready.json', dict(case=case, passed=True, pilot=cfg['pilot'], data_seed=42,
        training_seed=0, sha256={str(p): sha256(p) for p in paths}, ready_epoch=time.time()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['shards', 'merge', 'events'])
    parser.add_argument('--worker', type=int)
    args = parser.parse_args(); cfg = load_config()
    if not os.environ.get('PBS_JOBID'):
        raise RuntimeError('Data generation and audit require a compute allocation')
    def terminate(signum, frame):
        raise PreparationGuard('Allocation boundary signal')
    signal.signal(signal.SIGTERM, terminate)
    try:
        if args.action == 'shards':
            label_shards(cfg, args.worker)
        elif args.action == 'merge':
            from data.synthetic.generate import audit
            data = Path(cfg['paths']['temporal_root']); started = time.perf_counter()
            report = audit(data)
            if report['attempted'] != 10080:
                raise ValueError('Incomplete episode shards')
            report.update(this_invocation_seconds=time.perf_counter()-started,
                timing_scope='Merge only; generation wall cost is in the pilot worker budget',
                dual_sign_check_sha256=sha256(Path(cfg['paths']['stage_a_run']) / 'solver_smoke.json'),
                parallel_generator_sha256=sha256(Path(__file__)))
            write_json(data / 'generation_report.json', report)
        else:
            from scripts.variant_events import run
            run(cfg)
            audit = Path(cfg['paths']['stage_a_run'])
            support_report(cfg, audit)
            write_json(audit / 'events_complete.json', dict(passed=True))
    except PreparationGuard as error:
        print(str(error), flush=True); raise SystemExit(75)


if __name__ == '__main__':
    main()
