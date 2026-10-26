"""One final continuation allocation, followed by one evaluation-only allocation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


PYTHON = '/lus/flare/projects/FRAME-IDP/siebenschuh/envs/reason_venv_2026.1.0/bin/python'
CASES = ('case14_ieee', 'case30_ieee', 'case57_ieee', 'case118_ieee')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'{path.name}.{os.getpid()}.{time.time_ns()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def verify(hashes):
    for path, digest in hashes.items():
        if sha(path) != digest:
            raise ValueError(f'Pinned input changed: {path}')


def validate_config(parent, child):
    original, changed = deepcopy(parent), deepcopy(child)
    if child['stage_b']['training_budget_seconds'] != parent['stage_b']['training_budget_seconds'] + 3600:
        raise ValueError('Only the authorized one-hour budget extension is allowed')
    if child['v2']['run_identity'] == parent['v2']['run_identity']:
        raise ValueError('Continuation requires a new run identity')
    if child['paths']['training_run'] == parent['paths']['training_run']:
        raise ValueError('Parent training directory must remain read-only')
    for cfg in (original, changed):
        for key in ('training_run', 'runs_root'):
            cfg['paths'].pop(key, None)
        cfg['stage_b'].pop('training_budget_seconds')
        for key in ('run_identity', 'protocol_sha256', 'continuation'):
            cfg['v2'].pop(key, None)
    if original != changed:
        raise ValueError('Scientific configuration changed during continuation')


def check_pin(root):
    verify(read(root / 'operations/pin.json')['sha256'])


def train_worker(root, identity):
    check_pin(root)
    spec = read(root / 'chains' / f'{identity}.json')
    verify(spec['input_sha256'])
    pin = read(spec['source_pin'])
    verify({str(Path(pin['snapshot']) / p): h for p, h in pin['files'].items()})
    verify(read(spec['ready'])['sha256'])
    sys.path.insert(0, spec['source'])
    import torch
    from common import load_config
    from training.loop import make_manifest, run
    from training.state import atomic_torch_save, tree_digest
    if not os.environ.get('PBS_JOBID'):
        raise RuntimeError('Checkpoint transfer and training require a compute node')
    assert os.environ['ZE_FLAT_DEVICE_HIERARCHY'] == 'FLAT' and torch.xpu.device_count() == 1
    torch.xpu.set_device(0)
    torch.set_num_threads(1)
    parent_cfg, cfg = load_config(spec['parent_config']), load_config(spec['config'])
    validate_config(parent_cfg, cfg)
    parent = Path(spec['parent_adapter'])
    out = Path(cfg['paths']['training_run'])
    expected_parent, child_manifest = make_manifest(parent_cfg), make_manifest(cfg)
    if read(parent / 'manifest.json') != expected_parent:
        raise ValueError('Parent manifest does not match its pinned source and configuration')
    if (out / 'latest.pt').exists():
        raise ValueError('Final continuation is single-attempt; refuse to overwrite a child checkpoint')
    audit = {'passed': True, 'parent': str(parent), 'child': str(out), 'state': {}}
    for name in ('latest.pt', 'best.pt'):
        saved = torch.load(parent / name, map_location='cpu', weights_only=True)
        if saved['manifest'] != expected_parent:
            raise ValueError(f'Parent checkpoint manifest mismatch: {name}')
        if name == 'latest.pt' and saved['progress']['finished']:
            raise ValueError('Do not restart an already stopped chain')
        before = {k: tree_digest(v) for k, v in saved.items() if k != 'manifest'}
        saved['manifest'] = child_manifest
        atomic_torch_save(out / name, saved)
        restored = torch.load(out / name, map_location='cpu', weights_only=True)
        after = {k: tree_digest(v) for k, v in restored.items() if k != 'manifest'}
        if before != after or restored['manifest'] != child_manifest:
            raise ValueError('Continuation changed optimizer, model, RNG or progress state')
        audit['state'][name] = dict(parent_sha256=sha(parent / name), child_sha256=sha(out / name), unchanged=before)
    write(out / 'manifest.json', child_manifest)
    write(out / 'continuation_audit.json', audit)
    run(cfg, 'xpu')
    restores = [read(p) for p in (out / 'restores').glob('*.json')]
    if not restores or not all(r['passed'] for r in restores):
        raise ValueError('Exact optimizer/RNG restoration failed')
    verify(spec['input_sha256'])


def invoke(root, identity, command, env, deadline):
    began = time.time()
    record = dict(status='running', began=began, job=os.environ['PBS_JOBID'], node=os.uname().nodename)
    path = root / 'states' / f'{identity}.json'
    write(path, record)
    with (root / 'logs' / f'{identity}.log').open('a') as log:
        process = subprocess.Popen(command, env=dict(os.environ, **env), stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=max(1, deadline-time.time()+60))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            code = -9
    record.update(status='complete' if code == 0 else 'incomplete' if code == 75 else 'failed',
                  code=code, ended=time.time(), worker_seconds=time.time()-began)
    write(path, record)
    return code


def rank():
    value = int(next(os.environ[k] for k in ('PMI_RANK', 'PMIX_RANK', 'PALS_RANKID') if k in os.environ))
    if value not in (0, 1):
        raise ValueError('Exactly one launcher rank per node, two nodes, is required')
    return value


def train_node(root):
    check_pin(root)
    node = rank()
    deadline = float(os.environ['FINAL_JOB_START']) + 3300
    specs = [read(p) for p in sorted((root / 'chains').glob('*.json'))]
    def launch(spec):
        began = time.time()
        env = dict(ZE_FLAT_DEVICE_HIERARCHY='FLAT', ZE_AFFINITY_MASK=str(spec['tile']),
                   PYTHONHASHSEED=str(spec['seed']), GENCO_JOB_START_EPOCH=str(began),
                   PBS_WALLTIME=str(max(301, int(deadline-began+300))))
        return invoke(root, spec['identity'], [PYTHON, '-u', str(Path(__file__).resolve()),
                      'train-worker', '--root', str(root), '--identity', spec['identity']], env, deadline)
    with ThreadPoolExecutor(max_workers=6) as pool:
        codes = list(pool.map(launch, [s for s in specs if s['node'] == node]))
    write(root / f'train_node{node}.json', dict(complete=all(c == 0 for c in codes), codes=codes, ended=time.time()))
    if any(c != 0 for c in codes):
        raise RuntimeError('A continuation worker failed; no retry is permitted')


def evaluate_spec(root, spec_path, deadline, mixture=False):
    spec = read(spec_path)
    out = Path(spec['out'])
    def launch(shard):
        if time.time() > deadline-150:
            return 75
        env = dict(ZE_FLAT_DEVICE_HIERARCHY='FLAT', ZE_AFFINITY_MASK=str(shard),
                   GENCO_FLEET_DEADLINE=str(deadline), PYTHONPATH=spec['source'])
        identity = f"eval_{spec['case']}_{spec.get('training_seed', 'mix'+str(len(spec.get('members', []))))}_{shard}"
        return invoke(root, identity, [PYTHON, '-u', str(root / 'operations/evaluate.py'),
                      '--spec', str(spec_path), '--shard', str(shard)], env, deadline)
    if mixture:
        codes = [launch(s) for s in range(4)]
    else:
        with ThreadPoolExecutor(max_workers=4) as pool:
            codes = list(pool.map(launch, range(4)))
    return all(c == 0 for c in codes) and (out / 'completion.json').exists()


def seed_spec(root, case, seed, model, config):
    from omegaconf import OmegaConf
    plan = read(root / 'plan.json')
    window = Path(plan['parent_window'])
    out = root / 'evaluation' / case / f'seed{seed}'
    train = out / 'adapter'
    train.mkdir(parents=True, exist_ok=True)
    inputs = {}
    for name in ('best.pt', 'manifest.json', 'status.json', 'epochs.json'):
        path = model / name
        digest = sha(path)
        shutil.copyfile(path, train / name)
        if sha(path) != digest or sha(train / name) != digest:
            raise ValueError('Training checkpoint changed while admitting evaluation')
        inputs[str(path)] = digest
    cfg = OmegaConf.load(config)
    cfg.paths.training_run = str(train)
    cfg.paths.runs_root = str(out)
    cfg_path = out / 'evaluation.yaml'
    cfg_path.write_text(OmegaConf.to_yaml(cfg, resolve=False))
    baseline = window.parent / 'recovery_20261025_v1/evaluation' / case / 'seed0/stage_b'
    cached = baseline
    if case == 'case30_ieee':
        cached = Path('/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v4/stage_b')
    spec = dict(kind='checkpoint', case=case, training_seed=seed, source=plan['evaluation_source'],
                config=str(cfg_path), out=str(out / 'stage_b'), baseline=str(baseline),
                cache_manifest=str(cached / 'cache/manifest.json'), pairs=str(cached / 'variogram_pairs.npz'),
                input_sha256=inputs, headlines=str(window.parent / 'fleet_20261025_off_v1' / case /
                'stage_a/stage_a_followup/headline_components.json'), audit=str(window / 'evaluation' / case / 'evaluation.json'),
                label='Final post-freeze selected checkpoint; continuation may be budget-capped')
    write(out / 'spec.json', spec)
    return out / 'spec.json'


def eval_node(root):
    check_pin(root)
    node = rank()
    plan = read(root / 'plan.json')
    verify(plan['evaluation_source_sha256'])
    window = Path(plan['parent_window'])
    deadline = float(os.environ['FINAL_JOB_START']) + 3300
    cases = CASES[:3] if node == 0 else CASES[3:]
    for case in cases:
        members = []
        for seed in range(8):
            original = window.parent / 'recovery_20261025_v1/evaluation' / case / f'seed{seed}/stage_b'
            if seed >= 3:
                original = window / 'evaluation' / case / f'seed{seed}/stage_b'
            identity = f'{case}_seed{seed}'
            child = root / 'chains' / f'{identity}.json'
            if child.exists():
                spec = read(child)
                state_path = root / 'states' / f'{identity}.json'
                model = Path(spec['child_adapter'])
                if not state_path.exists() or read(state_path).get('status') != 'complete':
                    members.append(None)
                    continue
                audit = model / 'continuation_audit.json'
                restores = [read(f) for f in (model / 'restores').glob('*.json')]
                if not read(audit)['passed'] or not restores or not all(r['passed'] for r in restores):
                    raise ValueError('Continuation correctness gate failed')
                path = seed_spec(root, case, seed, model, spec['config'])
                ok = evaluate_spec(root, path, deadline)
                members.append(Path(read(path)['out']) if ok else None)
            elif (original / 'completion.json').exists():
                members.append(original)
            else:
                model = window / 'ensemble' / case / f'seed{seed}/adapter'
                if read(model / 'status.json').get('status') != 'finished':
                    members.append(None)
                    continue
                path = seed_spec(root, case, seed, model, window / 'configs' / f'{case}_seed{seed}.yaml')
                ok = evaluate_spec(root, path, deadline)
                members.append(Path(read(path)['out']) if ok else None)
        for size in (8, 5):
            selected = members[:size]
            if any(m is None for m in selected) or time.time() > deadline-150:
                write(root / 'mixtures' / case / f'size{size}/unavailable.json',
                      dict(reason='Missing verified member or allocation time exhausted', members=[str(m) for m in selected]))
                continue
            baseline = window.parent / 'recovery_20261025_v1/evaluation' / case / 'seed0/stage_b'
            cached = Path('/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v4/stage_b') if case == 'case30_ieee' else baseline
            out = root / 'mixtures' / case / f'size{size}'
            spec = dict(kind='mixture', case=case, source=plan['evaluation_source'],
                        config=str(window.parent / 'recovery_20261025_v1/evaluation/configs' / f'{case}_seed0.yaml'),
                        out=str(out), baseline=str(baseline), members=[str(m) for m in selected], seeds=list(range(size)),
                        pairs=str(cached / 'variogram_pairs.npz'),
                        input_sha256={str(m / 'completion.json'):sha(m / 'completion.json') for m in selected},
                        headlines=str(window.parent / 'fleet_20261025_off_v1' / case / 'stage_a/stage_a_followup/headline_components.json'),
                        audit=str(window / 'evaluation' / case / 'evaluation.json'), label='Post-freeze, includes final budget-capped chains')
            write(out / 'spec.json', spec)
            evaluate_spec(root, out / 'spec.json', deadline, mixture=True)
    write(root / f'eval_node{node}.json', dict(complete=True, ended=time.time()))


def job_state(identity):
    result = subprocess.run(['qstat', '-xf', '-F', 'json', identity], capture_output=True, text=True, timeout=40)
    if result.returncode:
        raise RuntimeError(result.stderr)
    value = next(iter(json.loads(result.stdout)['Jobs'].values()))
    return dict(id=identity, state=value['job_state'], exit_status=value.get('Exit_status'))


def report(root):
    plan = read(root / 'plan.json')
    training = {}
    for f in sorted((root / 'chains').glob('*.json')):
        spec = read(f)
        p = Path(spec['child_adapter']) / 'status.json'
        state = root / 'states' / f.name
        training[f.stem] = dict(worker=read(state) if state.exists() else {}, training=read(p) if p.exists() else {})
    rows = []
    frozen_rows = read(Path(plan['parent_window']) / 'mixture_sizes.json')['rows']
    for case in CASES:
        for size in (1, 2, 3, 5, 8):
            p = root / 'mixtures' / case / f'size{size}' / 'report.json'
            if size < 5:
                frozen = next(r for r in frozen_rows if r['case'] == case and r['size'] == size)
                rows.append(dict(case=case, size=size, complete=True, metrics=frozen['metrics'],
                                 evidence=str(Path(plan['parent_window']) / 'mixture_sizes.json'),
                                 sha256=sha(Path(plan['parent_window']) / 'mixture_sizes.json')))
                continue
            value = read(p) if p.exists() else {}
            rows.append(dict(case=case, size=size, complete=bool(value), metrics=value.get('headlines', {}).get('A', {}),
                             evidence=str(p), sha256=sha(p) if p.exists() else None))
    write(root / 'results.json', dict(category='post-freeze supplement', training=training, mixtures=rows,
                                    protocol_sha256=sha(root / 'protocol.json')))
    lines = ['# Final post-freeze continuation', '', 'One final training allocation and one evaluation-only allocation; no resubmission.',
             'Original 19:00 freeze results remain separate. Mixtures include final budget-capped chains with equal weight.', '',
             '| Chain | Steps added | Epochs / best | Best selection ES | Stop |', '|---|---:|---|---:|---|']
    for name, value in training.items():
        s = value['training']
        lines.append(f"| {name} | {s.get('steps_done', '')} | {s.get('epochs_reached', '')} / {s.get('best_epoch', '')} | {s.get('best_selection', '')} | {s.get('stop_reason', value['worker'].get('status', 'not started'))} |")
    lines += ['', '| Grid | Mixture size | ES | Vm CRPS | Coverage90 % | P imbalance MW |', '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        m = row['metrics']
        lines.append(f"| {row['case']} | {row['size']} | " + ' | '.join(str(m.get(k, 'unavailable')) for k in ('joint_es', 'vm_crps', 'vm_coverage90_pct', 'p_imbalance')) + ' |')
    lines += ['', 'Missing cells remain unmeasured. Fixed evaluation noise is shared across checkpoints; these are descriptive empirical mixtures.',
              'Headline voltage coordinates retain the mechanical support rule; undefined coordinates are not replaced.']
    (root / 'RESULTS.md').write_text('\n'.join(lines) + '\n')


def watch(root):
    check_pin(root)
    lock = (root / 'watch.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        try:
            state = job_state(read(root / 'training_submission.json')['job_id'])
            write(root / 'training_job.json', state)
            if state['state'] == 'F':
                break
        except Exception as error:
            write(root / 'poll_error.json', dict(error=repr(error), at=time.time()))
        write(root / 'watch_status.json', dict(pid=os.getpid(), status='waiting for final training', updated=time.time()))
        time.sleep(30)
    submission = root / 'evaluation_submission.json'
    if not submission.exists():
        write(submission, dict(status='submitting', at=time.time()))
        result = subprocess.run(['qsub', '-q', 'debug-scaling', '-A', 'EpiCalib', '-N', 'genco-final-eval',
                                 '-o', str(root / 'logs/evaluation.pbs.log'), '-v', f'FINAL_ROOT={root},FINAL_ACTION=eval-node',
                                 str(root / 'operations/job.pbs')], capture_output=True, text=True, timeout=45)
        write(submission, dict(status='submitted' if result.returncode == 0 else 'rejected',
                               job_id=result.stdout.strip(), stderr=result.stderr, code=result.returncode, at=time.time()))
        if result.returncode:
            raise RuntimeError('Evaluation submission rejected; no automatic retry')
    identity = read(submission).get('job_id')
    if not identity:
        raise RuntimeError('Ambiguous evaluation submission; manual inspection required')
    while True:
        try:
            state = job_state(identity)
            write(root / 'evaluation_job.json', state)
            if state['state'] == 'F':
                break
        except Exception as error:
            write(root / 'poll_error.json', dict(error=repr(error), at=time.time()))
        write(root / 'watch_status.json', dict(pid=os.getpid(), status='waiting for evaluation-only job', updated=time.time()))
        time.sleep(30)
    report(root)
    write(root / 'watch_status.json', dict(pid=os.getpid(), status='jobs complete; no further submissions', updated=time.time()))
    failures = [p.stem for p in (root / 'chains').glob('*.json')
                if not (root / 'states' / p.name).exists() or read(root / 'states' / p.name).get('status') != 'complete']
    write(root / 'gate.json', dict(passed=not failures, failed_chains=failures,
                                  protocol_sha256=sha(root / 'protocol.json'), results_sha256=sha(root / 'results.json')))
    if failures:
        raise RuntimeError('Final continuation correctness failed; release finalization parked')
    plan = read(root / 'plan.json')
    spec = importlib.util.spec_from_file_location('frozen_finalizer', plan['finalizer'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    window = Path(plan['parent_window'])
    with (window / 'finalizer.lock').open('a') as final_lock:
        fcntl.flock(final_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not module.jobs_finished(window):
            raise RuntimeError('Earlier jobs still active; final merge deferred')
        subprocess.run([PYTHON, str(Path(plan['finalizer']).with_name('report.py')), '--root', str(window)], check=True)
        module.consolidate(window, read(root / 'finalization.json'))
    write(root / 'FINAL.json', dict(status='complete', original_final=str(window / 'FINAL.json'), results=str(root / 'RESULTS.md')))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('train-node', 'train-worker', 'eval-node', 'watch', 'report'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--identity')
    args = parser.parse_args()
    if args.action == 'train-worker':
        train_worker(args.root, args.identity)
    else:
        globals()[args.action.replace('-', '_')](args.root)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if '--root' in sys.argv:
            root = Path(sys.argv[sys.argv.index('--root')+1])
            if sys.argv[1] == 'watch':
                write(root / 'WATCH_ERROR.json', dict(error=repr(error), at=time.time()))
        raise
