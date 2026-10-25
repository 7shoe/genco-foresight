"""Two independent large-grid pilots with bounded PBS continuation."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import json
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
CASES = ('case500_goc', 'case2000_goc')
DATA_BUDGET = 21600
TRAIN_BUDGET = 43200


def read(path, default=None):
    return read_json(path) if Path(path).exists() else ({} if default is None else default)


def text_file(path, text):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(text); os.replace(temporary, path)


def chains():
    return [dict(id=f'{case}_seed0', case=case, seed=0, node=0, tile=2*i,
                 K=8 if i == 0 else 2, label='K=8 pilot' if i == 0 else 'K=2 pilot')
            for i, case in enumerate(CASES)]


def config_file(root, case, seed=None):
    return root / 'configs' / (f'{case}_data.yaml' if seed is None else f'{case}_seed{seed}.yaml')


def chain_root(root, c):
    return root / c['case'] / f"seed{c['seed']}"


def validate_recipe(cfg, c, phase):
    settings = cfg['stage_b']
    expected = dict(training_samples=c['K'], future_graph_batch_size=c['K']*24,
        max_epochs=6, deterministic_algorithms=False, forecast_batch_size=1,
        forecast_samples=64, training_budget_seconds=TRAIN_BUDGET)
    if any(settings.get(key) != value for key, value in expected.items()):
        raise ValueError('Pilot recipe changed')
    if (cfg['seed'] != (42 if phase == 'data' else 0) or
        settings['adapter']['covariance_rank'] != 8 or
        settings['early_stopping']['patience'] != 3 or
        settings['loss']['power_balance_weight'] != 1.0 or
        cfg['pilot']['label'] != c['label'] or cfg['pilot']['cross_grid_table']):
        raise ValueError('Pilot seed, objective, label or stopping policy changed')


def verify_pin(root):
    pin = read_json(root / 'source_pin.json')
    if ROOT != Path(pin['snapshot']):
        raise ValueError('Pilot execution requires its pinned snapshot')
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
        raise ValueError('A pilot family needs a fresh identity')
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
        cfg['pilot'] = dict(label='K=8 pilot' if case == 'case500_goc' else 'K=2 pilot',
            cross_grid_table=False, epoch_cap=6, training_seed=0, data_seed=42)
        repair = protocol.get('repair')
        if repair:
            cfg['pilot']['solver_processes'] = repair['solver_processes'][case]
            cfg['solver']['budget'] = repair['solver_budget'][case]
            cfg['solver']['warm_start_base'] = case == 'case2000_goc'
            resolved = OmegaConf.to_container(OmegaConf.create(cfg), resolve=True)
            solver = resolved['solver']
            network_file = source / 'networks' / Path(cfg['paths']['network_file']).name
            env = dict(os.environ, LD_LIBRARY_PATH='', JULIA_DEPOT_PATH=solver['julia_depot'],
                JULIA_PKG_OFFLINE='true', JULIA_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            output = subprocess.check_output([solver['julia_executable'], '--startup-file=no',
                f'--project={ROOT / "solver/julia"}', str(ROOT / 'scripts/pilots/reference.jl'), str(network_file)],
                env=env, text=True, timeout=120)
            metadata = json.loads(next(line.removeprefix('REFERENCE_JSON ') for line in output.splitlines()
                if line.startswith('REFERENCE_JSON ')))
            metadata['case_sha256'] = sha256(network_file)
            cfg['solver']['reference_network'] = metadata
            write_json(grid / 'stage_a/reference_network.json', metadata)
            previous = Path(repair['previous_root']) / case
            write_json(grid / 'seed0/budget.json', read_json(previous / 'seed0/budget.json'))
            parity_root = Path(repair['parity_root']) / case
            gate = read_json(parity_root / 'gate.json')
            if not gate['passed'] or not gate['bit_identical'] or gate['report_sha256'] != sha256(parity_root / 'stage_a/parity/parity_report.json'):
                raise ValueError('Accepted subset parity evidence changed')
            (grid / 'stage_a/parity').symlink_to(parity_root / 'stage_a/parity', target_is_directory=True)
            write_json(grid / 'stage_a/parity_reuse.json', dict(gate=str(parity_root / 'gate.json'),
                gate_sha256=sha256(parity_root / 'gate.json'), reason='Unchanged checkpoint and official snapshot inputs; new temporal graphs use corrected network reference and are checked in the solver round-trip'))
            text_file(grid / 'data/DATASET_CARD.md', f'# {case} temporal pilot\n\nsubset parity (200): seed42, CPU, bit-identical; evidence {parity_root}.\n\nFull20 episodes, v4 demand law. GridSpec, temporal graph reference flag, angle mask and solver audit use the PowerModels parsed reference. The scorer code and physical equations are unchanged.\n')
        cfg['paths'].update(assets_root=str(assets), runs_root=str(grid), temporal_root=str(grid / 'data'), data_root=str(grid / 'data'),
            stage_a_run=str(grid / 'stage_a'), evidence_root=str(grid / 'stage_a'), training_run=str(grid / 'shared'))
        cfg['paths']['network_file'] = str(assets / 'networks' / Path(cfg['paths']['network_file']).name)
        cfg['stage_b']['coordinate_scales'] = str(grid / 'stage_a/stage_a_followup/train_coordinate_scales.json')
        cfg['stage_b'].update(deterministic_algorithms=False, pbs_queue='debug',
            metrics_interval_steps=100, training_budget_seconds=TRAIN_BUDGET, checkpoint_interval_seconds=900,
            max_epochs=6, training_samples=8 if case == 'case500_goc' else 2,
            future_graph_batch_size=192 if case == 'case500_goc' else 48, cache_graph_batch_size=24)
        for key in ('headline_voltage_bus_ids', 'continuous_voltage_bus_ids', 'headline_thermal_line_ids',
                    'continuous_loading_line_ids', 'secondary_thermal_daily_line_ids'):
            cfg['stage_b']['calibration'].pop(key, None)
        cfg['stage_b']['calibration']['selection'] = 'calibration_support'
        data_path = config_file(root, case)
        data_path.parent.mkdir(exist_ok=True)
        data_path.write_text(OmegaConf.to_yaml(OmegaConf.create(cfg), resolve=False)); configs.append(data_path)
        for seed in (0,):
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
    names.update(str(p.relative_to(ROOT)) for p in (ROOT / 'scripts/pilots').glob('*') if p.is_file())
    names.update(['tests/test_pilots.py', 'reports/v2/large_grid_pilots.protocol.json'])
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
    alias = GENCO / protocol.get('status_alias', 'runs/pilots')
    if alias.exists() or alias.is_symlink():
        raise ValueError(f'{alias} already exists; refusing to replace it')
    alias.symlink_to(root, target_is_directory=True)
    status(root)
    return dict(root=str(root), snapshot=str(snapshot), protocol_sha256=protocol_hash)


def status(root):
    state = read(root / 'controller.json')
    rows = ['# Large-grid OFF pilots', '', f'Updated {datetime.now(timezone.utc).isoformat()}.', '',
        'Additional debug family; excluded from the cross-grid fleet table. Seed 0, B1, fp32, epoch cap 6, patience 3.', '',
        '| Pilot | Data | State | Steps | Epochs | Best selection ES | Data seconds / cap | Training seconds / cap |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for c in chains():
        out = chain_root(root, c); training = read(out / 'adapter/status.json'); budget = read(out / 'budget.json')
        blocked = read(out / 'blocked.json') or read(root / c['case'] / 'blocked.json')
        ready = (root / c['case'] / 'ready.json').exists()
        phase = 'blocked' if blocked else training.get('status', 'ready' if ready else 'preparing')
        rows.append(f"| {c['case']} {c['label']} | {'audited' if ready else 'pending'} | {phase} | {training.get('steps_after', 0)} | {training.get('epochs_reached', 0)} | {training.get('best_selection', '')} | {budget.get('data_seconds', 0):.1f} / {DATA_BUDGET} | {budget.get('training_seconds', 0):.1f} / {TRAIN_BUDGET} |")
        if blocked:
            rows.append(f"\n{c['case']} stopped: {blocked}\n")
    rows += ['', '| Job | State | Reserved node-hours |', '|---|---|---:|']
    for j in state.get('jobs', []):
        rows.append(f"| {j.get('job_id', 'submission pending')} | {j.get('state', '')} | 1 |")
    rows += ['', f"Controller: {state.get('status', 'prepared')}; source: {root / 'source'}.",
        f"Protocol: {root / 'protocol.json'}. Maximum {read(root / 'protocol.json').get('job', {}).get('max_allocations', 24)} one-hour allocations.",
        'Worker wall costs include startup and checkpoint/audit overhead, exclude queue wait, and finalize at job boundaries.',
        'Solver process counts are recorded in each configuration; data gates must pass before that grid trains. No full fleet evaluation or cross-grid table entry.']
    text_file(root / 'STATUS.md', '\n'.join(rows) + '\n')


def task(root, c):
    out = chain_root(root, c)
    if (out / 'blocked.json').exists() or (root / c['case'] / 'blocked.json').exists():
        return None
    budget = read(out / 'budget.json')
    if not (root / c['case'] / 'ready.json').exists():
        if budget.get('data_seconds', 0) >= DATA_BUDGET-180:
            write_json(root / c['case'] / 'blocked.json', dict(reason='Data preparation budget reached'))
            return None
        return 'data'
    training = read(out / 'adapter/status.json')
    if training.get('status') == 'finished':
        return None
    if budget.get('training_seconds', 0) >= TRAIN_BUDGET-180:
        write_json(out / 'blocked.json', dict(reason='Pilot training worker budget reached'))
        return None
    return 'train'


def invoke(root, c, phase, deadline):
    out = chain_root(root, c); out.mkdir(parents=True, exist_ok=True)
    job = os.environ['PBS_JOBID'].split('.')[0]
    name = f'{job}_{phase}'
    began = time.time()
    budget = read(out / 'budget.json', {'training_seconds': 0., 'data_seconds': 0.})
    limit_key, limit = ('training_seconds', TRAIN_BUDGET) if phase == 'train' else ('data_seconds', DATA_BUDGET)
    deadline = min(deadline, began + limit - budget[limit_key] - 60)
    if phase == 'train':
        prior = read(out / 'adapter/status.json')
        if prior and (out / 'adapter/latest.pt').exists():
            prior['cumulative_training_wall_seconds'] = max(prior['cumulative_training_wall_seconds'], budget['training_seconds'])
            write_json(out / 'adapter/status.json', prior)
    write_json(out / 'attempts' / f'{name}.json', dict(phase=phase, job=job, start_epoch=began, budget_before=budget))
    env = dict(os.environ, ZE_FLAT_DEVICE_HIERARCHY='FLAT', ZE_AFFINITY_MASK=str(c['tile']),
        PYTHONHASHSEED=str(c['seed']), GENCO_JOB_START_EPOCH=str(began),
        PBS_WALLTIME=str(max(301, int(deadline-began+300))), GENCO_PILOTS_DEADLINE=str(deadline),
        GENCO_CONFIG=str(config_file(root, c['case'], None if phase == 'data' else c['seed'])))
    command = [PYTHON, '-u', '-m', 'scripts.pilots.run', 'worker', '--root', str(root), '--chain', c['id'], '--phase', phase]
    log = out / 'logs' / f'{name}.log'; log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a') as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=max(1, min(deadline-began+60, TRAIN_BUDGET-budget['training_seconds'] if phase == 'train' else deadline-began+60)))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL); process.wait(); code = -9
    elapsed = time.time()-began
    key = {'train': 'training_seconds', 'data': 'data_seconds'}.get(phase, 'evaluation_seconds')
    budget[key] += elapsed
    write_json(out / 'budget.json', budget)
    write_json(out / 'boundaries' / f'{name}.json', dict(phase=phase, job=job, returncode=code, wall_seconds=elapsed,
        node=os.uname().nodename, tile=c['tile'], budget=budget, log=str(log)))
    if code not in (0, 75):
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
        if task(root, c) == phase:
            return


def node(root):
    verify_pin(root)
    raw = next((os.environ[k] for k in ('PMI_RANK', 'PMIX_RANK', 'PALS_RANKID') if k in os.environ), None)
    if raw is None or int(raw) != 0:
        raise RuntimeError('Expected one MPI launcher rank on one node')
    rank = int(raw); job = os.environ['PBS_JOBID'].split('.')[0]
    deadline = float(os.environ['GENCO_PILOTS_START']) + 3300
    write_json(root / 'jobs' / f'{job}_node{rank}_start.json', dict(rank=rank, node=os.uname().nodename, start_epoch=time.time()))
    if rank == 0 and not (root / 'tests.json').exists():
        log = root / 'logs/tests.log'
        with log.open('a') as stream:
            result = subprocess.run([PYTHON, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                'tests/test_pilots.py', 'tests/test_pilot_repair.py', 'tests/test_fleet.py', 'tests/test_case_identity.py', 'tests/test_multigrid.py',
                'tests/test_v2_harness.py', 'tests/test_v2_stage1.py', '--basetemp', str(root / 'tmp/tests'), '--junitxml', str(root / 'tests.xml')],
                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=max(1, deadline-time.time()))
        write_json(root / 'tests.json', dict(passed=result.returncode == 0, log=str(log)))
    while not (root / 'tests.json').exists() and time.time() < deadline:
        time.sleep(2)
    if not read(root / 'tests.json').get('passed'):
        raise RuntimeError('Pilot preflight tests failed')
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(slot, root, c, deadline) for c in chains() if c['node'] == rank]
        for f in futures:
            f.result()
    write_json(root / 'jobs' / f'{job}_node{rank}.json', dict(rank=rank, node=os.uname().nodename, end_epoch=time.time()))
    status(root)


def worker(root, identity, phase):
    verify_pin(root)
    if not os.environ.get('PBS_JOBID'):
        raise RuntimeError('Pilot numerical work requires a compute allocation')
    c = next(x for x in chains() if x['id'] == identity)
    import torch
    if os.environ.get('ZE_FLAT_DEVICE_HIERARCHY') != 'FLAT' or os.environ.get('ZE_AFFINITY_MASK') != str(c['tile']) or torch.xpu.device_count() != 1:
        raise RuntimeError('Each chain must see exactly its assigned flat tile')
    torch.xpu.set_device(0); torch.set_num_threads(1)
    cfg = load_config()
    validate_recipe(cfg, c, phase)
    if phase == 'data':
        from scripts.pilots.data import prepare_grid
        from scripts.pilots.data import PreparationGuard
        try:
            prepare_grid(root, c['case'], cfg)
        except PreparationGuard as error:
            print(str(error), flush=True); raise SystemExit(75)
    elif phase == 'train':
        from training.loop import run
        ready = read_json(root / c['case'] / 'ready.json')
        for path, digest in ready['sha256'].items():
            if sha256(path) != digest:
                raise ValueError(f'Audited grid artifact changed: {path}')
        run(cfg, 'xpu')


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
                        if code and not all((root / 'jobs' / f"{job['job_id'].split('.')[0]}_node{r}.json").exists() for r in (0,)):
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
                        record = dict(number=number, submitted_epoch=time.time(), state='submitting', reserved_node_seconds=3600)
                        state['jobs'].append(record); write_json(root / 'controller.json', state)
                        log = root / 'logs' / f'job_{number:02d}.log'; log.parent.mkdir(exist_ok=True)
                        result = subprocess.run(['qsub', '-q', 'debug', '-A', 'EpiCalib', '-N', f'genco-pilots-{number}',
                            '-o', str(log), '-v', f'GENCO_PILOTS_ROOT={root}', str(ROOT / 'scripts/pilots/job.pbs')],
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
        raise ValueError('Pilot artifacts must be in GENCO/runs/v2')
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
