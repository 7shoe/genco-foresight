"""Twelve independent, pinned OFF chains with bounded PBS continuation."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from common import ROOT, load_config, read_json, sha256, write_json

PYTHON = '/lus/flare/projects/FRAME-IDP/siebenschuh/envs/reason_venv_2026.1.0/bin/python'
GENCO = Path('/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO')
CASES = ('case14_ieee', 'case30_ieee', 'case57_ieee', 'case118_ieee')


def read(path, default=None):
    return read_json(path) if Path(path).exists() else ({} if default is None else default)


def text_file(path, text):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(text); os.replace(temporary, path)


def chains():
    return [dict(id=f'{case}_seed{seed}', case=case, seed=seed,
                 node=i // 2, tile=2 * ((i % 2) * 3 + seed))
            for i, case in enumerate(CASES) for seed in (0, 1, 2)]


def config_file(root, case, seed=None):
    return root / 'configs' / (f'{case}_data.yaml' if seed is None else f'{case}_seed{seed}.yaml')


def chain_root(root, c):
    return root / c['case'] / f"seed{c['seed']}"


def verify_pin(root):
    pin = read_json(root / 'source_pin.json')
    if ROOT != Path(pin['snapshot']):
        raise ValueError('Fleet execution requires its pinned snapshot')
    for name, digest in pin['files'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f'Pinned source changed: {name}')
    for name, digest in pin['configuration_sha256'].items():
        if sha256(name) != digest:
            raise ValueError(f'Pinned configuration changed: {name}')
    for name, digest in pin['protected_sha256'].items():
        if sha256(name) != digest:
            raise ValueError(f'Released artifact changed: {name}')
    return pin


def prepare(root, protocol_path):
    from omegaconf import OmegaConf
    from data.grids import grid_config
    if root.exists():
        raise ValueError('A fleet needs a fresh identity')
    root.mkdir(parents=True)
    protocol = read_json(protocol_path)
    write_json(root / 'protocol.json', protocol)
    protocol_hash = sha256(root / 'protocol.json')
    original = load_config(ROOT / 'configs/v4.yaml')
    configs = []
    for case in CASES:
        grid = root / case
        assets = grid / 'assets'
        assets.mkdir(parents=True)
        source = Path(original['paths']['assets_root'])
        for name in ('profiles', 'networks', 'cli_deps', 'wheels'):
            if (source / name).exists():
                (assets / name).symlink_to(source / name, target_is_directory=True)
        cp = assets / 'genco-opfdata'; cp.mkdir()
        for name in (case, 'splits'):
            (cp / name).symlink_to(source / 'genco-opfdata' / name, target_is_directory=True)
        cfg = grid_config(case, OmegaConf.to_container(OmegaConf.load(ROOT / 'configs/v4.yaml'), resolve=False))
        cfg['seed'] = 42
        cfg['paths'].update(assets_root=str(assets), runs_root=str(grid), temporal_root=str(grid / 'data'),
            stage_a_run=str(grid / 'stage_a'), evidence_root=str(grid / 'stage_a'), training_run=str(grid / 'shared'))
        cfg['paths']['network_file'] = str(assets / 'networks' / Path(cfg['paths']['network_file']).name)
        cfg['stage_b']['coordinate_scales'] = str(grid / 'stage_a/stage_a_followup/train_coordinate_scales.json')
        cfg['stage_b'].update(deterministic_algorithms=False, pbs_queue='debug-scaling',
            metrics_interval_steps=100, training_budget_seconds=21600, checkpoint_interval_seconds=900)
        if case == 'case30_ieee':
            cfg['dataset'] = deepcopy(original['dataset'])
            cfg['paths']['temporal_root'] = original['paths']['temporal_root']
        for key in ('headline_voltage_bus_ids', 'continuous_voltage_bus_ids', 'headline_thermal_line_ids',
                    'continuous_loading_line_ids', 'secondary_thermal_daily_line_ids'):
            cfg['stage_b']['calibration'].pop(key, None)
        cfg['stage_b']['calibration']['selection'] = 'calibration_support'
        data_path = config_file(root, case)
        data_path.parent.mkdir(exist_ok=True)
        data_path.write_text(OmegaConf.to_yaml(OmegaConf.create(cfg), resolve=False)); configs.append(data_path)
        for seed in (0, 1, 2):
            training = deepcopy(cfg)
            out = grid / f'seed{seed}'
            training['seed'] = seed
            training['paths'].update(runs_root=str(out), training_run=str(out / 'adapter'))
            training['v2'] = dict(stage=1, run_identity=f'{root.name}/{case}/seed{seed}', protocol_sha256=protocol_hash)
            path = config_file(root, case, seed)
            path.write_text(OmegaConf.to_yaml(OmegaConf.create(training), resolve=False)); configs.append(path)
    snapshot = root / 'source'; snapshot.mkdir()
    tracked = subprocess.check_output(['git', 'ls-files'], cwd=ROOT, text=True).splitlines()
    names = set(tracked)
    names.update(str(p.relative_to(ROOT)) for p in (ROOT / 'scripts/fleet').glob('*') if p.is_file())
    names.update(['tests/test_fleet.py', 'reports/v2/fleet.protocol.json'])
    files = {}
    for name in sorted(names):
        src = ROOT / name
        if not src.is_file() or '.pytest_cache' in src.parts:
            continue
        dst = snapshot / name; dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst); files[name] = sha256(dst); dst.chmod(0o444)
    protected = {}
    for version in ('v3', 'v4'):
        for name in ('manifest.json', 'status.json', 'epochs.json', 'best.pt'):
            path = GENCO / f'runs/adapter_{version}' / name
            protected[str(path)] = sha256(path)
    for name in ('manifest.json', 'quality_report.json'):
        path = Path(original['paths']['temporal_root']) / name
        protected[str(path)] = sha256(path)
    pin = dict(snapshot=str(snapshot), source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        working_tree_changes=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True).splitlines(),
        files=files, configuration_sha256={str(p): sha256(p) for p in [*configs, root / 'protocol.json']},
        protected_sha256=protected, release_tag=subprocess.check_output(['git', 'rev-parse', 'refs/tags/v0.1.0'], cwd=ROOT, text=True).strip())
    write_json(root / 'source_pin.json', pin)
    alias = GENCO / 'runs/fleet'
    if alias.exists() or alias.is_symlink():
        raise ValueError('runs/fleet already exists; refusing to replace it')
    alias.symlink_to(root, target_is_directory=True)
    status(root)
    return dict(root=str(root), snapshot=str(snapshot), protocol_sha256=protocol_hash)


def status(root):
    state = read(root / 'controller.json')
    rows = ['# Small-grid OFF fleet', '', f'Updated {datetime.now(timezone.utc).isoformat()}.', '',
        '12 fresh training seeds 0/1/2; v4 law, K8/B1/fp32, AdamW, OFF. Each chain has a 21,600-second worker budget.', '',
        '| Chain | Data | Phase | Steps | Epochs | Best ES | Worker seconds | Evaluation |',
        '|---|---|---|---:|---:|---:|---:|---|']
    for c in chains():
        out = chain_root(root, c); s = read(out / 'adapter/status.json'); budget = read(out / 'budget.json')
        ready = (root / c['case'] / 'ready.json').exists()
        phase = 'blocked' if (out / 'blocked.json').exists() or (root / c['case'] / 'blocked.json').exists() else (
            'finished' if s.get('status') == 'finished' or budget.get('training_seconds', 0) >= 21585 else ('ready/running' if ready else 'preparing data'))
        done = (out / 'stage_b/completion.json').exists()
        rows.append(f"| {c['id']} | {'audited' if ready else 'pending'} | {phase} | {s.get('steps_after', 0)} | {s.get('epochs_reached', 0)} | {s.get('best_selection', '')} | {budget.get('training_seconds', 0):.1f} | {'complete' if done else 'pending'} |")
    rows += ['', '| Job | State | Reserved node-hours |', '|---|---|---:|']
    for j in state.get('jobs', []):
        rows.append(f"| {j.get('job_id', 'submission pending')} | {j.get('state', '')} | 2 |")
    rows += ['', f"Controller: {state.get('status', 'prepared')}; pinned source: {root / 'source'}.",
        f"Protocol: {root / 'protocol.json'}. Cross-grid results: {root / 'cross_grid.md'}.",
        'Data preparation and evaluation are separately accounted; running costs are finalized at job boundaries. Missing worker boundaries are charged conservatively and parked.',
        'Grid comparisons have different train-derived scales and mechanical headline coordinates; cross-grid ES is descriptive.']
    text_file(root / 'STATUS.md', '\n'.join(rows) + '\n')


def task(root, c):
    out = chain_root(root, c)
    if (out / 'blocked.json').exists() or (root / c['case'] / 'blocked.json').exists():
        return None
    if not (root / c['case'] / 'ready.json').exists():
        owner_budget = read(root / c['case'] / 'seed0/budget.json')
        if owner_budget.get('data_seconds', 0) >= 10620:
            write_json(root / c['case'] / 'blocked.json', dict(reason='Three-hour data preparation cap reached'))
            return None
        return 'data' if c['seed'] == 0 else 'wait'
    s = read(out / 'adapter/status.json'); budget = read(out / 'budget.json')
    if s.get('status') != 'finished' and budget.get('training_seconds', 0) < 21585:
        return 'train'
    if not (out / 'adapter/best.pt').exists():
        write_json(out / 'blocked.json', dict(reason='No selected checkpoint at training stop'))
        return None
    if not (out / 'stage_b/completion.json').exists():
        if budget.get('evaluation_seconds', 0) >= 21420:
            write_json(out / 'blocked.json', dict(reason='Six-hour evaluation cap reached'))
            return None
        return 'evaluate'
    if c['seed'] == 0 and all((root / c['case'] / f'seed{s}/stage_b/completion.json').exists() for s in (0, 1, 2)):
        if not (root / c['case'] / 'mixture/completion.json').exists():
            return 'mixture'
    return None


def invoke(root, c, phase, deadline):
    out = chain_root(root, c); out.mkdir(parents=True, exist_ok=True)
    job = os.environ['PBS_JOBID'].split('.')[0]
    name = f'{job}_{phase}'
    began = time.time()
    budget = read(out / 'budget.json', {'training_seconds': 0., 'evaluation_seconds': 0., 'data_seconds': 0.})
    limit_key, limit = ('training_seconds', 21600) if phase == 'train' else (('data_seconds', 10800) if phase == 'data' else ('evaluation_seconds', 21600))
    deadline = min(deadline, began + limit - budget[limit_key] - 60)
    if phase == 'train':
        prior = read(out / 'adapter/status.json')
        if prior and (out / 'adapter/latest.pt').exists():
            prior['cumulative_training_wall_seconds'] = max(prior['cumulative_training_wall_seconds'], budget['training_seconds'])
            write_json(out / 'adapter/status.json', prior)
    write_json(out / 'attempts' / f'{name}.json', dict(phase=phase, job=job, start_epoch=began, budget_before=budget))
    env = dict(os.environ, ZE_FLAT_DEVICE_HIERARCHY='FLAT', ZE_AFFINITY_MASK=str(c['tile']),
        PYTHONHASHSEED=str(c['seed']), GENCO_JOB_START_EPOCH=str(began),
        PBS_WALLTIME=str(max(301, int(deadline-began+300))), GENCO_FLEET_DEADLINE=str(deadline),
        GENCO_CONFIG=str(config_file(root, c['case'], None if phase == 'data' else c['seed'])))
    command = [PYTHON, '-u', '-m', 'scripts.fleet.run', 'worker', '--root', str(root), '--chain', c['id'], '--phase', phase]
    log = out / 'logs' / f'{name}.log'; log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a') as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=max(1, min(deadline-began+60, 21600-budget['training_seconds'] if phase == 'train' else deadline-began+60)))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL); process.wait(); code = -9
    elapsed = time.time()-began
    key = {'train': 'training_seconds', 'data': 'data_seconds'}.get(phase, 'evaluation_seconds')
    budget[key] += elapsed
    write_json(out / 'budget.json', budget)
    write_json(out / 'boundaries' / f'{name}.json', dict(phase=phase, job=job, returncode=code, wall_seconds=elapsed,
        node=os.uname().nodename, tile=c['tile'], budget=budget, log=str(log)))
    if code:
        destination = root / c['case'] if phase == 'data' else out
        write_json(destination / 'blocked.json', dict(phase=phase, error=f'Worker exit {code}', log=str(log), job=job))
    return code


def slot(root, c, deadline):
    while time.time() < deadline-180:
        phase = task(root, c)
        if phase is None:
            return
        if phase == 'wait':
            time.sleep(10); continue
        if invoke(root, c, phase, deadline):
            return
        if phase in ('train', 'evaluate', 'mixture') and task(root, c) == phase:
            return


def node(root):
    verify_pin(root)
    raw = next((os.environ[k] for k in ('PMI_RANK', 'PMIX_RANK', 'PALS_RANKID') if k in os.environ), None)
    if raw is None or int(raw) not in (0, 1):
        raise RuntimeError('Expected two MPI launcher ranks, one per node')
    rank = int(raw); job = os.environ['PBS_JOBID'].split('.')[0]
    deadline = float(os.environ['GENCO_FLEET_START']) + 3300
    write_json(root / 'jobs' / f'{job}_node{rank}_start.json', dict(rank=rank, node=os.uname().nodename, start_epoch=time.time()))
    if rank == 0 and not (root / 'tests.json').exists():
        log = root / 'logs/tests.log'
        with log.open('a') as stream:
            result = subprocess.run([PYTHON, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                'tests/test_fleet.py', 'tests/test_case_identity.py', 'tests/test_multigrid.py',
                'tests/test_v2_harness.py', 'tests/test_v2_stage1.py', '--basetemp', str(root / 'tmp/tests'), '--junitxml', str(root / 'tests.xml')],
                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=max(1, deadline-time.time()))
        write_json(root / 'tests.json', dict(passed=result.returncode == 0, log=str(log)))
    while not (root / 'tests.json').exists() and time.time() < deadline:
        time.sleep(2)
    if not read(root / 'tests.json').get('passed'):
        raise RuntimeError('Fleet preflight tests failed')
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(slot, root, c, deadline) for c in chains() if c['node'] == rank]
        for f in futures:
            f.result()
    write_json(root / 'jobs' / f'{job}_node{rank}.json', dict(rank=rank, node=os.uname().nodename, end_epoch=time.time()))
    status(root)


def worker(root, identity, phase):
    verify_pin(root)
    if not os.environ.get('PBS_JOBID'):
        raise RuntimeError('Fleet numerical work requires a compute allocation')
    c = next(x for x in chains() if x['id'] == identity)
    import torch
    if os.environ.get('ZE_FLAT_DEVICE_HIERARCHY') != 'FLAT' or os.environ.get('ZE_AFFINITY_MASK') != str(c['tile']) or torch.xpu.device_count() != 1:
        raise RuntimeError('Each chain must see exactly its assigned flat tile')
    torch.xpu.set_device(0); torch.set_num_threads(1)
    cfg = load_config()
    if phase == 'data':
        from scripts.fleet.data import prepare_grid
        prepare_grid(root, c['case'], cfg)
    elif phase == 'train':
        from training.loop import run
        ready = read_json(root / c['case'] / 'ready.json')
        for path, digest in ready['sha256'].items():
            if sha256(path) != digest:
                raise ValueError(f'Audited grid artifact changed: {path}')
        run(cfg, 'xpu')
    else:
        from scripts.fleet.evaluate import evaluate_chain, mixture
        (evaluate_chain if phase == 'evaluate' else mixture)(root, c)


def reconcile(root, job):
    jid = job['job_id'].split('.')[0]
    for c in chains():
        out = chain_root(root, c)
        for attempt in (out / 'attempts').glob(f'{jid}_*.json'):
            if (out / 'boundaries' / attempt.name).exists():
                continue
            a = read_json(attempt); budget = read(out / 'budget.json', a['budget_before'])
            key = {'train': 'training_seconds', 'data': 'data_seconds'}.get(a['phase'], 'evaluation_seconds')
            budget[key] = max(budget[key], a['budget_before'][key]+3600)
            write_json(out / 'budget.json', budget)
            target = root / c['case'] if a['phase'] == 'data' else out
            write_json(target / 'blocked.json', dict(reason='Missing worker boundary; full hour charged', attempt=str(attempt)))
    status(root)


def controller(root, once=False):
    from scripts.training_chain import job_state
    verify_pin(root)
    root.mkdir(exist_ok=True)
    with (root / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read(root / 'controller.json', {'jobs': [], 'created_epoch': time.time()})
        while True:
            try:
                state.update(pid=os.getpid(), host=os.uname().nodename, status='running', updated_epoch=time.time())
                active = False
                for job in state['jobs']:
                    if job.get('finished'):
                        continue
                    if not job.get('job_id'):
                        raise RuntimeError('Ambiguous submission: inspect qsub record before continuing')
                    st, code = job_state(job['job_id']); job.update(state=st, exit_status=code)
                    if st == 'F':
                        reconcile(root, job); job['finished'] = True
                        if code and not all((root / 'jobs' / f"{job['job_id'].split('.')[0]}_node{r}.json").exists() for r in (0, 1)):
                            state['status'] = 'launcher_failed'; write_json(root / 'controller.json', state); status(root); return
                    else:
                        active = True
                work = [task(root, c) for c in chains()]
                if not active and not any(work):
                    state['status'] = 'complete_or_blocked'
                elif not active:
                    if len(state['jobs']) >= read_json(root / 'protocol.json')['job']['max_allocations']:
                        state['status'] = 'allocation_cap_reached'
                    else:
                        verify_pin(root)
                        number = len(state['jobs'])+1
                        record = dict(number=number, submitted_epoch=time.time(), state='submitting', reserved_node_seconds=7200)
                        state['jobs'].append(record); write_json(root / 'controller.json', state)
                        log = root / 'logs' / f'job_{number:02d}.log'; log.parent.mkdir(exist_ok=True)
                        result = subprocess.run(['qsub', '-q', 'debug-scaling', '-A', 'EpiCalib', '-N', f'genco-fleet-{number}',
                            '-o', str(log), '-v', f'GENCO_FLEET_ROOT={root}', str(ROOT / 'scripts/fleet/job.pbs')],
                            cwd=ROOT, capture_output=True, text=True, timeout=45)
                        record.update(stdout=result.stdout, stderr=result.stderr, returncode=result.returncode)
                        if result.returncode:
                            state['jobs'].pop(); state['submission_wait_reason'] = result.stderr
                        else:
                            record.update(job_id=result.stdout.strip(), state='Q')
                            print('Submitted', record['job_id'], flush=True)
                write_json(root / 'controller.json', state); status(root)
                if once or state['status'] != 'running':
                    return
            except Exception as error:
                write_json(root / 'controller_error.json', dict(error=repr(error), time_epoch=time.time()))
                if once:
                    raise
            time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'controller', 'node', 'worker', 'status'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--chain'); parser.add_argument('--phase'); parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to(GENCO / 'runs/v2'):
        raise ValueError('Fleet artifacts must be in GENCO/runs/v2')
    if args.action == 'prepare':
        print(prepare(root, args.protocol), flush=True)
    elif args.action == 'controller':
        controller(root, args.once)
    elif args.action == 'node':
        node(root)
    elif args.action == 'worker':
        worker(root, args.chain, args.phase)
    else:
        status(root)


if __name__ == '__main__':
    main()
