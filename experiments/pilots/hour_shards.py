"""Atomic hour shards with fixed base starts, independent of worker scheduling."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import numpy as np

from power_fm_foresight._config import load_config, read_json, sha256, write_json
from experiments.data_generation import audit, prepare_benchmark
from power_fm_foresight._data.generate import demands
from experiments.pilots.data import PreparationGuard, command, guard, validate_prefix
from power_fm_foresight._solver.adapter import IpoptSolver
from power_fm_foresight._solver.checks import check_solution


def assignments(episodes, maximum, worker, workers):
    if workers < 1 or not 0 <= worker < workers:
        raise ValueError('Invalid hour worker assignment')
    return [(episodes[i // 504], i % 504) for i in range(worker, maximum, workers)]


def seal(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def read_hour(path, benchmark, episode, hour):
    payload = read_json(path)
    digest = payload.pop('sha256')
    if seal(payload) != digest or payload['benchmark_id'] != benchmark or payload['episode_id'] != episode:
        raise ValueError(f'Hour shard checksum/provenance changed: {path}')
    if payload['record']['hour'] != hour:
        raise ValueError('Hour identity changed')
    return payload['record']


def worker(cfg, index, maximum):
    data, manifest, network, frame, scaled = prepare_benchmark(cfg)
    smoke = read_json(Path(cfg['paths']['stage_a_run']) / 'solver_smoke.json')
    if not smoke['cold']['checks']['valid'] or not smoke['dual_check']['valid']:
        raise ValueError('Checked base solution and dual convention required')
    workers = cfg['pilot']['solver_processes']
    tasks = assignments(manifest['episodes'], maximum, index, workers)
    pending = [(ep, h) for ep, h in tasks if not (data / 'hours' / ep['episode_id'] / f'{h:03}.json').exists()]
    if not pending:
        return
    reserve = cfg['solver']['budget']['wall_seconds'] + 90
    guard(reserve)
    base = smoke['cold']['state'] if cfg['solver'].get('warm_start_base') else None
    with IpoptSolver(cfg, data / f'julia_worker{index}.log', network=network,
                     startup_seconds=600) as solver:
        current = None
        for ep, hour in pending:
            guard(reserve)
            if current != ep['episode_id']:
                current = ep['episode_id']; lo = ep['source_start_hour']
                pd, qd = demands(scaled[lo:lo+504], network.bus[:, 2], network.bus[:, 3],
                                 cfg['generation'], ep['seed'])
            result = solver.solve(pd[hour], qd[hour], initial_state=base)
            unresolved = result['status'] == 'WORKER_UNRESOLVED'
            result['checks'] = ({'valid': False, 'reason': 'worker_unresolved'} if unresolved else
                                check_solution(network, pd[hour], qd[hour], result))
            valid = result['checks']['valid']
            dual_valid = bool(valid and 'dual_p_raw' in result and np.isfinite(result['dual_p_raw']).all())
            if dual_valid:
                result['dual_p_cost_per_mw'] = (np.asarray(result['dual_p_raw'])*smoke['dual_check']['sign']/network.baseMVA).tolist()
                result['dual_convention'] = 'one-hour benchmark objective; raw nodal dual times verified sign/baseMVA'
            row = dict(hour=hour, time_utc=str(frame.time_utc.iloc[lo+hour]),
                pd_mw=pd[hour].tolist(), qd_mvar=qd[hour].tolist(), state_available=valid,
                duals_available=dual_valid, target_state=result.get('state') if valid else None, solver=result)
            payload = dict(benchmark_id=manifest['benchmark_id'], episode_id=current, record=row)
            write_json(data / 'hours' / current / f'{hour:03}.json', dict(payload, sha256=seal(payload)))
            print(current, hour, result['status'], result.get('solve_seconds'), flush=True)
            if unresolved:
                raise RuntimeError('Solver worker unresolved; saved failure is not silently retried')


def merge(cfg, maximum):
    data, manifest, network, frame, scaled = prepare_benchmark(cfg)
    smoke_path = Path(cfg['paths']['stage_a_run']) / 'solver_smoke.json'
    smoke = read_json(smoke_path)
    times, requests, iterations, statuses = [], [], [], Counter()
    started = time.monotonic()
    for i, ep in enumerate(manifest['episodes']):
        guard(180)
        count = min(504, maximum-i*504)
        if count <= 0:
            break
        rows = [read_hour(data / 'hours' / ep['episode_id'] / f'{h:03}.json',
                         manifest['benchmark_id'], ep['episode_id'], h) for h in range(count)]
        shard = dict(benchmark_id=manifest['benchmark_id'], episode_id=ep['episode_id'],
                     versions=smoke['versions'], records=rows)
        lo = ep['source_start_hour']
        pd, qd = demands(scaled[lo:lo+504], network.bus[:, 2], network.bus[:, 3], cfg['generation'], ep['seed'])
        validate_prefix(shard, manifest, ep, pd, qd, frame)
        for row in rows:
            result = row['solver']; statuses[result['status']] += 1
            check = check_solution(network, row['pd_mw'], row['qd_mvar'], result)
            if check['valid'] != row['state_available']:
                raise ValueError('Independent hour audit disagrees with saved validity')
            if result.get('success'):
                times.append(result['solve_seconds']); requests.append(result['request_seconds'])
                iterations.append(result['iterations'])
        target = data / 'episodes' / f"{ep['episode_id']}.json"
        if target.exists():
            previous = read_json(target)
            if previous['records'] != rows[:len(previous['records'])]:
                raise ValueError('Previously merged prefix changed')
        write_json(target, shard)
    guard(180)
    report = audit(data)
    report.update(this_invocation_seconds=time.monotonic()-started,
        timing_scope='Merge/audit only; complete wall cost in seed0/budget.json',
        dual_sign_check_sha256=sha256(smoke_path), parallel_generator_sha256=sha256(Path(__file__)),
        solve_statistics=dict(completed=len(times), median_seconds=float(np.median(times)) if times else None,
            max_seconds=max(times, default=None), median_request_seconds=float(np.median(requests)) if requests else None,
            max_request_seconds=max(requests, default=None), median_iterations=float(np.median(iterations)) if iterations else None,
            max_iterations=max(iterations, default=None), status_counts=dict(statuses)))
    if maximum == 200:
        report['accepted'] = report['attempted'] == 200 and report['valid'] >= 1 and report['hashes_verified']
        write_json(data / 'pilot_report.json', report)
        median = report['solve_statistics']['median_seconds']
        if cfg['task']['canonical_case'] == 'case2000_goc' and median is not None and median > 120:
            write_json(data / 'formulation_review_required.json', dict(median_seconds=median,
                reason='Predeclared 120-second trigger: compare Datakit/PowerModels formulation before expansion'))
            raise RuntimeError('Case2000 median exceeds 120 seconds; formulation review required')
        if not requests:
            raise RuntimeError('No completed pilot solves; cannot authorize expansion')
        projected = float(np.mean(requests))*10080/cfg['pilot']['solver_processes']
        write_json(data / 'generation_projection.json', dict(projected_solve_wall_seconds=projected,
            full_size_cap_seconds=10800, episodes=20, includes_startup_and_audit=False))
        if projected > 10800:
            raise RuntimeError('Full-size projection exceeds three hours; prepare a new pilot half-size identity')
    else:
        write_json(data / 'generation_report.json', report)
    print({k:v for k,v in report.items() if k != 'shards'}, flush=True)


def generate(cfg, cfg_path, maximum):
    data, _, _, _, _ = prepare_benchmark(cfg)
    workers = cfg['pilot']['solver_processes']
    if len(os.sched_getaffinity(0)) < workers:
        raise RuntimeError('CPU affinity cannot support the declared solver process count')
    started = time.time()
    write_json(data / f'workers_{maximum}_{os.environ["PBS_JOBID"].split(".")[0]}.json',
        dict(processes=workers, available_cpus=len(os.sched_getaffinity(0)),
             warm_start='fixed checked base' if cfg['solver'].get('warm_start_base') else 'cold', start_epoch=started))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(command, 'scripts.pilots.hour_shards', cfg_path,
            data / f'logs/hour_worker{w}.log', '--worker', str(w), '--hours', str(maximum)) for w in range(workers)]
        errors = []
        for future in futures:
            try:
                future.result()
            except Exception as error:
                errors.append(error)
        failures = [e for e in errors if not isinstance(e, PreparationGuard)]
        if failures:
            raise failures[0]
        if errors:
            raise errors[0]
    guard()
    merge(cfg, maximum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=int, required=True)
    parser.add_argument('--hours', type=int, required=True, choices=(200, 10080))
    args = parser.parse_args()
    if not os.environ.get('PBS_JOBID'):
        raise RuntimeError('Numerical generation requires a compute allocation')
    def terminate(signum, frame):
        raise PreparationGuard('Allocation boundary signal')
    signal.signal(signal.SIGTERM, terminate)
    try:
        worker(load_config(), args.worker, args.hours)
    except PreparationGuard as error:
        print(error, flush=True); raise SystemExit(75)


if __name__ == '__main__':
    main()
