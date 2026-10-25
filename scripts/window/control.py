"""Pinned tasks sharing one PBS allocation, with an absolute submission cutoff."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else ({} if default is None else default)


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): h.update(block)
    return h.hexdigest()


def ready(root, task):
    state = read(root / 'states' / f"{task['id']}.json")
    if state.get('status') in ('complete', 'blocked', 'budget_capped'): return False
    if time.time() < task.get('not_before', 0): return False
    return all(Path(p).exists() for p in task.get('requires', []))


def status(root):
    state = read(root / 'controller.json')
    rows = ['# Release window status', '', f'Updated {datetime.now(timezone.utc).isoformat()}.',
            '', 'Window: 2026-10-25 14:20–19:00 America/Chicago. No new submissions after 19:00.',
            '', '| Task | Status | Worker seconds | Job |', '|---|---|---:|---|']
    for path in sorted((root / 'tasks').glob('*.json')):
        task = read(path); s = read(root / 'states' / path.name)
        rows.append(f"| {task['id']} | {s.get('status', 'not started')} | {s.get('worker_seconds', 0):.1f} | {s.get('job', '')} |")
    rows += ['', '| Allocation | State |', '|---|---|']
    rows.extend(f"| {j.get('id', '')} | {j.get('state', '')} |" for j in state.get('jobs', []))
    rows += ['', 'Phase B/C/D task records are added only after their source snapshot and protocol are pinned.',
             f"Controller: {state.get('status', 'not started')}; protocol: {root / 'protocol.json'}."]
    temporary = root / f'STATUS.{os.getpid()}.tmp'; temporary.write_text('\n'.join(rows)+'\n')
    os.replace(temporary, root / 'STATUS.md')


def invoke(root, task, deadline):
    path = root / 'states' / f"{task['id']}.json"
    state = read(path, {'worker_seconds': 0., 'attempts': []})
    job = os.environ['PBS_JOBID'].split('.')[0]
    if any(a['job'] == job for a in state['attempts']): return
    for filename, digest in task['sha256'].items():
        if sha(filename) != digest: raise ValueError(f'Pinned task changed: {filename}')
    began = time.time(); budget = task['budget_seconds'] - state['worker_seconds']
    if budget < 120:
        state['status'] = 'budget_capped'; write(path, state); return
    end = min(deadline, began + budget - 30)
    env = dict(os.environ, **task.get('env', {}), GENCO_JOB_START_EPOCH=str(began),
               PBS_WALLTIME=str(max(301, int(end-began+300))), GENCO_FLEET_DEADLINE=str(end),
               WINDOW_TASK=str(root / 'tasks' / f"{task['id']}.json"))
    log = root / 'logs' / f"{task['id']}_{job}.log"
    attempt = dict(job=job, began=began, deadline=end, log=str(log), node=os.uname().nodename)
    state.update(status='running', job=job); state['attempts'].append(attempt); write(path, state)
    with log.open('a') as stream:
        p = subprocess.Popen(task['command'], cwd=task['cwd'], env=env, stdout=stream,
                             stderr=subprocess.STDOUT, start_new_session=True)
        try: code = p.wait(timeout=max(1, end-time.time()+90))
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGTERM)
            try: p.wait(timeout=20)
            except subprocess.TimeoutExpired: os.killpg(p.pid, signal.SIGKILL); p.wait()
            code = -9
    elapsed = time.time()-began; state['worker_seconds'] += elapsed
    attempt.update(code=code, ended=time.time(), wall_seconds=elapsed)
    complete = task.get('once', False)
    if task.get('completion'):
        s = read(task['completion'])
        complete = s.get('status') == 'finished' or s.get('complete') is True or s.get('passed') is True
    state['status'] = ('blocked' if code not in (0, 75) else 'complete' if complete else
                       'budget_capped' if state['worker_seconds'] >= task['budget_seconds']-120 else 'resumable')
    write(path, state); status(root)


def node(root):
    for name, digest in read(root/'operations_pin.json')['sha256'].items():
        if sha(name) != digest: raise ValueError(f'Allocation launcher changed: {name}')
    rank = int(next(os.environ[k] for k in ('PMI_RANK', 'PMIX_RANK', 'PALS_RANKID') if k in os.environ))
    protocol = read(root / 'protocol.json'); deadline = float(os.environ['WINDOW_START']) + 3300
    futures = {}; used = set()
    with ThreadPoolExecutor(max_workers=12) as pool:
        while time.time() < min(deadline-180, protocol['freeze_epoch']):
            for identity, (future, slots) in list(futures.items()):
                if future.done(): future.result(); used.difference_update(slots); del futures[identity]
            for path in sorted((root / 'tasks').glob('*.json')):
                task = read(path); slots = set(task.get('slots', [0]))
                if task['node'] != rank or task['id'] in futures or slots & used or not ready(root, task): continue
                s = read(root / 'states' / path.name)
                if any(a['job'] == os.environ['PBS_JOBID'].split('.')[0] for a in s.get('attempts', [])): continue
                used.update(slots); futures[task['id']] = (pool.submit(invoke, root, task, deadline), slots)
            time.sleep(5)
    write(root / 'boundaries' / f"{os.environ['PBS_JOBID'].split('.')[0]}_node{rank}.json", dict(ended=time.time(), rank=rank))


def freeze(root):
    protocol = read(root / 'protocol.json'); stopped = []
    for item in protocol.get('stop_submitters_at_freeze', []):
        try:
            cmd = Path(f"/proc/{item['pid']}/cmdline").read_bytes().replace(b'\0', b' ').decode()
            if item['match'] in cmd:
                os.kill(item['pid'], signal.SIGTERM); stopped.append(item)
        except (FileNotFoundError, ProcessLookupError): pass
    record = dict(freeze_epoch=protocol['freeze_epoch'], recorded_epoch=time.time(), stopped_submitters=stopped,
                  tasks={p.stem: dict(task=read(p), state=read(root/'states'/p.name)) for p in (root/'tasks').glob('*.json')})
    write(root / 'freeze.json', record)
    rows = ['# 19:00 CT freeze', '', 'No new jobs may be submitted after 2026-10-26 00:00:00 UTC.', '',
            '| Run | Category | Status at freeze | Snapshot |', '|---|---|---|---|']
    for item in record['tasks'].values():
        t, s = item['task'], item['state']
        rows.append(f"| {t.get('run_directory', t['id'])} | {t.get('category', 'verification')} | {s.get('status', 'not started')} | {t.get('snapshot', t['cwd'])} |")
    (root / 'FREEZE.md').write_text('\n'.join(rows)+'\n')


def controller(root):
    protocol = read(root / 'protocol.json')
    with (root / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read(root / 'controller.json', {'jobs': []})
        while time.time() < protocol['freeze_epoch']:
            try:
                state.update(status='running', pid=os.getpid(), updated=time.time())
                active = False
                for j in state['jobs']:
                    if j.get('state') == 'F': continue
                    r = subprocess.run(['qstat', '-xf', '-F', 'json', j['id']], capture_output=True, text=True, timeout=30)
                    if r.returncode: active = True; continue
                    v = next(iter(json.loads(r.stdout)['Jobs'].values())); j.update(state=v['job_state'], exit_status=v.get('Exit_status'))
                    if j['state'] == 'F':
                        for path in (root/'states').glob('*.json'):
                            s = read(path)
                            if s.get('status') == 'running' and s.get('job') == j['id'].split('.')[0]:
                                s.update(status='blocked', reason='Missing worker boundary; charge full allocation')
                                s['worker_seconds'] += 3600; write(path, s)
                    active |= j['state'] != 'F'
                if not active and time.time() < protocol['freeze_epoch']-90 and len(state['jobs']) < protocol['max_allocations']:
                    number = len(state['jobs'])+1
                    r = subprocess.run(['qsub', '-q', 'debug-scaling', '-A', 'EpiCalib', '-N', f'genco-window-{number}',
                        '-o', str(root/'logs'/f'job_{number:02d}.log'), '-v', f'WINDOW_ROOT={root}',
                        str(root/'operations/job.pbs')], capture_output=True, text=True, timeout=45)
                    if r.returncode == 0:
                        state['jobs'].append(dict(id=r.stdout.strip(), state='Q', submitted=time.time()))
                        print('Submitted', r.stdout.strip(), flush=True)
                    else: state['submission_wait_reason'] = r.stderr
                write(root/'controller.json', state); status(root)
            except Exception as e: write(root/'controller_error.json', dict(error=repr(e), at=time.time()))
            time.sleep(min(15, max(0, protocol['freeze_epoch']-time.time())))
        freeze(root); state['status']='frozen'; write(root/'controller.json', state); status(root)


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('action', choices=['controller','node','status']); p.add_argument('--root',type=Path,required=True)
    a=p.parse_args(); globals()[a.action](a.root)
