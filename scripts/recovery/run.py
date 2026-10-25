"""Independent evaluation and pilot recovery without writes to their source runs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
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
from scripts.pilots.run import PYTHON, GENCO, read, text_file, verify_pin

FLEET = GENCO / 'runs/v2/fleet_20261025_off_v1'
OLD_PILOTS = GENCO / 'runs/v2/large_grid_pilots_20261025_repair_v2'
CASES = ('case14_ieee', 'case30_ieee', 'case57_ieee', 'case118_ieee')


def protected_writes(root):
    def check(event, args):
        paths = []
        if event == 'open' and not isinstance(args[0], int):
            mode, flags = args[1:3]
            if (mode and any(c in mode for c in 'wax+')) or (flags and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)):
                paths = [args[0]]
        elif event in ('os.remove', 'os.rmdir', 'os.mkdir', 'os.chmod', 'os.utime'):
            paths = args[:1]
        elif event in ('os.rename', 'os.link', 'os.symlink'):
            paths = args[:2] if event == 'os.rename' else args[1:2]
        for name in paths:
            if isinstance(name, (str, bytes, os.PathLike)):
                path = Path(os.fsdecode(name)).resolve()
                if path.is_relative_to(GENCO) and not path.is_relative_to(root):
                    raise PermissionError(f'Recovery cannot write outside its new run: {path}')
    sys.addaudithook(check)


def snapshot(root, configuration):
    dest = root / 'source'; dest.mkdir()
    files = {}
    for name in subprocess.check_output(['git', 'ls-files'], cwd=ROOT, text=True).splitlines():
        src = ROOT / name
        if not src.is_file():continue
        target = dest / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target); target.chmod(0o444); files[name] = sha256(target)
    pin = dict(snapshot=str(dest), source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        files=files, configuration_sha256={str(p):sha256(p) for p in configuration}, protected_sha256={})
    for family in ('evaluation', 'pilots'):
        folder = root / family
        (folder / 'source').symlink_to(dest, target_is_directory=True)
        write_json(folder / 'source_pin.json', pin)
    write_json(root / 'source_pin.json', pin)


def prepare(root, protocol_path):
    from copy import deepcopy
    from omegaconf import OmegaConf
    if root.exists():raise ValueError('Recovery requires a new identity')
    root.mkdir(parents=True)
    protocol = read_json(protocol_path); write_json(root / 'protocol.json', protocol)
    configuration = [root / 'protocol.json']
    for family in ('evaluation', 'pilots'):
        out = root / family; (out / 'configs').mkdir(parents=True)
        (out / 'logs').mkdir(); (out / 'tmp').mkdir()
        write_json(out / 'protocol.json', dict(protocol, job=protocol['jobs'][family]))
        configuration.append(out / 'protocol.json')
    evaluation = root / 'evaluation'
    for case in CASES:
        grid = evaluation / case; grid.mkdir()
        source = FLEET / case / 'stage_a/stage_a_followup/headline_components.json'
        target = grid / 'stage_a/stage_a_followup/headline_components.json'
        target.parent.mkdir(parents=True); shutil.copyfile(source, target)
        for seed in range(3):
            cfg = OmegaConf.to_container(OmegaConf.load(FLEET / 'configs' / f'{case}_seed{seed}.yaml'), resolve=False)
            out = grid / f'seed{seed}'; out.mkdir()
            cfg['paths'].update(runs_root=str(out), training_run=str(out / 'adapter'))
            cfg['recovery'] = dict(source_training=str(FLEET / case / f'seed{seed}/adapter'), purpose='Read-only evaluation')
            path = evaluation / 'configs' / f'{case}_seed{seed}.yaml'
            path.write_text(OmegaConf.to_yaml(OmegaConf.create(cfg), resolve=False)); configuration.append(path)
    pilots = root / 'pilots'
    for case in ('case500_goc', 'case2000_goc'):
        grid = pilots / case; data = grid / 'data'; audit = grid / 'stage_a'
        data.mkdir(parents=True); audit.mkdir()
        source = OLD_PILOTS / case
        for name in ('manifest.json', 'pilot_report.json', 'generation_report.json'):
            shutil.copyfile(source / 'data' / name, data / name)
        (data / 'episodes').symlink_to(source / 'data/episodes', target_is_directory=True)
        for name in ('solver_smoke.json', 'reference_network.json', 'parity_reuse.json'):
            shutil.copyfile(source / 'stage_a' / name, audit / name)
        (audit / 'parity').symlink_to((source / 'stage_a/parity').resolve(), target_is_directory=True)
        shutil.copyfile(source / 'data/DATASET_CARD.md', data / 'DATASET_CARD.md')
        for seed in (None, 0):
            name = f'{case}_data.yaml' if seed is None else f'{case}_seed0.yaml'
            cfg = OmegaConf.to_container(OmegaConf.load(OLD_PILOTS / 'configs' / name), resolve=False)
            cfg['paths'].update(runs_root=str(grid if seed is None else grid / 'seed0'), temporal_root=str(data), data_root=str(data),
                stage_a_run=str(audit), evidence_root=str(audit), training_run=str(grid / ('shared' if seed is None else 'seed0/adapter')))
            cfg['stage_b']['coordinate_scales'] = str(audit / 'stage_a_followup/train_coordinate_scales.json')
            cfg['pilot']['reuse_labels'] = dict(source_root=str(source / 'data'), source_snapshot=str(OLD_PILOTS / 'source'),
                manifest_sha256=sha256(source / 'data/manifest.json'))
            cfg['pilot'].pop('solver_processes', None)
            if seed is not None:
                cfg['v2'] = dict(stage=1, run_identity=f'{root.name}/pilots/{case}/seed0', protocol_sha256=sha256(root / 'protocol.json'))
            path = pilots / 'configs' / name
            path.write_text(OmegaConf.to_yaml(OmegaConf.create(cfg), resolve=False)); configuration.append(path)
        write_json(grid / 'seed0/budget.json', read_json(source / 'seed0/budget.json'))
    snapshot(root, configuration)
    status(root)
    return str(root)


def eval_chains():
    return [dict(id=f'{case}_seed{seed}', case=case, seed=seed, tile=3*i+seed)
            for i, case in enumerate(CASES) for seed in range(3)]


def admit(root, chain):
    out = root / chain['case'] / f"seed{chain['seed']}"
    if (out / 'input_pin.json').exists():return True
    source = FLEET / chain['case'] / f"seed{chain['seed']}/adapter"
    finished = read(source / 'status.json').get('status') == 'finished'
    controller = read(FLEET / 'controller.json')
    exhausted = controller.get('status') in ('allocation_cap_reached', 'complete_or_blocked') and all(
        j.get('finished') for j in controller.get('jobs', []))
    if not finished and not exhausted:return False
    if not (source / 'best.pt').exists():return False
    target = out / 'adapter'; target.mkdir(exist_ok=True)
    hashes = {}
    for name in ('best.pt', 'manifest.json', 'status.json', 'epochs.json', 'manifest.training.json'):
        src = source / name
        if not src.exists():continue
        before = sha256(src); shutil.copyfile(src, target / name)
        if sha256(src) != before or sha256(target / name) != before:raise ValueError('Finished checkpoint changed while copying')
        (target / name).chmod(0o444); hashes[str(src)] = before
    write_json(out / 'input_pin.json', dict(source=str(source), source_sha256=hashes, admitted_epoch=time.time(),
        training_stop='finished' if finished else 'original controller stopped at its recorded limit'))
    return True


def input_check(root, c):
    pin = read_json(root / c['case'] / f"seed{c['seed']}/input_pin.json")
    for path, digest in pin['source_sha256'].items():
        if sha256(path) != digest:raise ValueError(f'Read-only input changed: {path}')


def eval_task(root, c):
    out = root / c['case'] / f"seed{c['seed']}"
    if (out / 'blocked.json').exists():return None
    if not (out / 'stage_b/completion.json').exists():
        return 'evaluate' if admit(root, c) else 'waiting'
    if c['seed'] == 0 and not (root / c['case'] / 'mixture/completion.json').exists():
        if any((root / c['case'] / f'seed{s}/blocked.json').exists() for s in range(3)):
            return None
        if all((root / c['case'] / f'seed{s}/stage_b/completion.json').exists() for s in range(3)):
            return 'mixture'
        return 'waiting'
    return None


def eval_worker(root, identity, phase):
    verify_pin(root); protected_writes(root.parent)
    if not os.environ.get('PBS_JOBID'):raise RuntimeError('Evaluation requires a compute allocation')
    c = next(c for c in eval_chains() if c['id'] == identity)
    import torch
    if os.environ.get('ZE_FLAT_DEVICE_HIERARCHY') != 'FLAT' or os.environ.get('ZE_AFFINITY_MASK') != str(c['tile']) or torch.xpu.device_count() != 1:
        raise ValueError('Expected one isolated physical tile, addressed as local xpu:0')
    torch.xpu.set_device(0)
    input_check(root, c)
    from scripts.fleet.evaluate import evaluate_chain, mixture
    (evaluate_chain if phase == 'evaluate' else mixture)(root, c)
    input_check(root, c)


def invoke(root, c, phase, deadline):
    out = root / c['case'] / f"seed{c['seed']}"; (out / 'logs').mkdir(exist_ok=True)
    job = os.environ['PBS_JOBID'].split('.')[0]; name = f'{job}_{phase}'
    began = time.time(); budget = read(out / 'budget.json', {'worker_seconds':0.})
    cap = read_json(root / 'protocol.json')['jobs']['evaluation']['worker_seconds_per_chain']
    deadline = min(deadline, began+cap-budget['worker_seconds']-60)
    if deadline-began < 300:
        write_json(out / 'blocked.json', dict(reason='Evaluation worker budget reached')); return
    write_json(out / 'attempts' / f'{name}.json', dict(start_epoch=began, phase=phase, budget_before=budget))
    env = dict(os.environ, ZE_FLAT_DEVICE_HIERARCHY='FLAT', ZE_AFFINITY_MASK=str(c['tile']),
        GENCO_FLEET_DEADLINE=str(deadline), PYTHONDONTWRITEBYTECODE='1')
    for key in list(env):
        if key.startswith(('PMI_', 'PMIX_', 'PALS_', 'OMPI_', 'SLURM_')):env.pop(key)
    with (out / 'logs' / f'{name}.log').open('a') as stream:
        child = subprocess.Popen([PYTHON, '-u', '-m', 'scripts.recovery.run', 'eval-worker', '--root', str(root),
            '--chain', c['id'], '--phase', phase], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:code = child.wait(timeout=max(1, deadline-time.time()+30))
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid, signal.SIGKILL); child.wait()
            code = 75
    elapsed = time.time()-began; budget['worker_seconds'] += elapsed; write_json(out / 'budget.json', budget)
    write_json(out / 'boundaries' / f'{name}.json', dict(returncode=code, wall_seconds=elapsed, budget=budget))
    if code not in (0,75):write_json(out / 'blocked.json', dict(phase=phase, exit_code=code, log=str(out / 'logs' / f'{name}.log')))


def eval_node(root):
    verify_pin(root)
    deadline = float(os.environ['GENCO_RECOVERY_START'])+3300
    if not (root / 'tests.json').exists():
        with (root / 'logs/tests.log').open('a') as stream:
            code = subprocess.run([PYTHON, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests/test_recovery.py',
                'tests/test_solver.py', 'tests/test_evaluation.py', 'tests/test_evaluation_controls.py',
                '--basetemp', str(root / 'tmp/tests'), '--junitxml', str(root / 'tests.xml')],
                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=300).returncode
        write_json(root / 'tests.json', dict(passed=code == 0))
    if not read(root / 'tests.json').get('passed'):raise RuntimeError('Recovery tests failed')
    from scripts.fleet.evaluate import scorer_check, summary
    scorer_check(root)
    def slot(c):
        while time.time() < deadline-300:
            phase = eval_task(root, c)
            if phase is None:return
            if phase == 'waiting':time.sleep(15);continue
            invoke(root, c, phase, deadline)
            if eval_task(root, c) == phase:return
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(slot, c) for c in eval_chains()]
        for future in futures:future.result()
    summary(root)
    write_json(root / 'jobs' / f"{os.environ['PBS_JOBID'].split('.')[0]}_boundary.json", dict(end_epoch=time.time()))
    status(root.parent)


def eval_controller(root, once=False):
    from scripts.training_chain import job_state
    verify_pin(root)
    with (root / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read(root / 'controller.json', {'jobs':[]})
        while True:
            try:
                state.update(pid=os.getpid(), status='running', updated_epoch=time.time())
                active = False
                for job in state['jobs']:
                    if job.get('finished'):continue
                    if not job.get('job_id'):raise RuntimeError('Ambiguous qsub; inspect saved submission')
                    st, code = job_state(job['job_id']);job.update(state=st, exit_status=code)
                    if st == 'F':
                        job['finished'] = True
                        if code:
                            state['status']='launcher_failed';write_json(root/'controller.json',state);status(root.parent);return
                    else:active=True
                tasks = [eval_task(root,c) for c in eval_chains()]
                if not active and not any(tasks):state['status']='complete_or_blocked'
                elif not active and any(t not in (None,'waiting') for t in tasks):
                    if len(state['jobs']) >= read_json(root/'protocol.json')['jobs']['evaluation']['max_allocations']:
                        state['status']='allocation_cap_reached'
                    else:
                        number=len(state['jobs'])+1;record=dict(number=number,state='submitting',submitted_epoch=time.time())
                        state['jobs'].append(record);write_json(root/'controller.json',state)
                        response=subprocess.run(['qsub','-q','debug','-A','EpiCalib','-N',f'genco-eval-{number}',
                            '-o',str(root/'logs'/f'job_{number:02d}.log'),'-v',f'GENCO_RECOVERY_ROOT={root}',
                            str(ROOT/'scripts/recovery/evaluate.pbs')],cwd=ROOT,capture_output=True,text=True,timeout=45)
                        record.update(stdout=response.stdout,stderr=response.stderr,returncode=response.returncode)
                        if response.returncode:
                            state['jobs'].pop();state['submission_wait_reason']=response.stderr
                        else:record.update(job_id=response.stdout.strip(),state='Q');print('Submitted',record['job_id'],flush=True)
                write_json(root/'controller.json',state);status(root.parent)
                if once or state['status']!='running':return
            except Exception as error:
                write_json(root/'controller_error.json',dict(error=repr(error),epoch=time.time()))
                if once:raise
            time.sleep(30)


def status(root):
    lines=['# Recovery status','',f'Updated {datetime.now(timezone.utc).isoformat()}.','',
        'Original fleet job/controller/snapshot/run directories are read-only inputs.','',
        '| Item | Status |','|---|---|']
    evaluation=root/'evaluation'
    for case in CASES:
        rows=[]
        for seed in range(3):
            out=evaluation/case/f'seed{seed}'
            state='complete' if (out/'stage_b/completion.json').exists() else ('blocked' if (out/'blocked.json').exists() else 'not started')
            if state=='not started' and list((out/'attempts').glob('*.json')):state='running / resumable'
            rows.append(f'{seed}: {state}')
        lines.append(f"| Evaluation {case} | {'; '.join(rows)}; mixture {'complete' if (evaluation/case/'mixture/completion.json').exists() else 'not started'} |")
    for case in ('case500_goc','case2000_goc'):
        p=root/'pilots'/case;training=read(p/'seed0/adapter/status.json')
        state=training.get('status','not started')
        if (p/'blocked.json').exists() or (p/'seed0/blocked.json').exists():state='blocked'
        lines.append(f"| {case} pilot | audit {'passed' if (p/'data/quality_report.json').exists() else 'pending'}; training {state} |")
    for family in ('evaluation','pilots'):
        c=read(root/family/'controller.json');jobs=c.get('jobs',[])
        lines.append(f"| {family} controller | {c.get('status','not started')}; latest job {jobs[-1].get('job_id','submission pending') if jobs else 'not submitted'} |")
    lines.append('| Stage 1 verdict | '+('complete' if (root/'stage1_verdict.json').exists() and read(root/'stage1_verdict.json').get('complete') else 'pending test evaluations')+' |')
    text_file(root/'STATUS.md','\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','eval-controller','eval-node','eval-worker','status'))
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--protocol',type=Path)
    parser.add_argument('--chain');parser.add_argument('--phase');parser.add_argument('--once',action='store_true')
    args=parser.parse_args();root=args.root.resolve()
    if not root.is_relative_to(GENCO/'runs/v2'):raise ValueError('Recovery outputs must be in new Lustre runs/v2')
    if args.action=='prepare':print(prepare(root,args.protocol))
    elif args.action=='eval-controller':eval_controller(root,args.once)
    elif args.action=='eval-node':eval_node(root)
    elif args.action=='eval-worker':eval_worker(root,args.chain,args.phase)
    else:status(root)


if __name__=='__main__':main()
