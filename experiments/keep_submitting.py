"""Detached PBS continuation for pinned families, independent of a Codex session."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from power_fm_foresight._config import read_json, write_json


def poll(control):
    configuration = read_json(control / 'control.json')
    result = {}
    for family in ('fleet', 'pilots'):
        settings = configuration[family]; root = Path(settings['root'])
        state = read_json(root / 'controller.json')
        if state.get('status') in ('launcher_failed', 'allocation_cap_reached'):
            result[family] = dict(status=state['status']); continue
        submitted = len(state['jobs'])-settings['jobs_at_handoff']
        active = any(not job.get('finished') for job in state['jobs'])
        if submitted >= configuration['additional_allocations_per_family'] and not active:
            result[family] = dict(status='continuation_cap_reached', submitted=submitted)
            continue
        # The original controller retains all per-chain budgets and scientific stops.
        if submitted >= configuration['additional_allocations_per_family']:
            from experiments.training_chain import job_state
            for job in state['jobs']:
                if job.get('finished'):continue
                st, code = job_state(job['job_id'])
                job.update(state=st, exit_status=code)
                if st == 'F':job['finished']=True
            write_json(root / 'controller.json', state)
            result[family] = dict(status='waiting_for_last_allocation', submitted=submitted)
            continue
        with (control / f'{family}_continuation.log').open('a') as stream:
            completed = subprocess.run([sys.executable, '-u', '-m', f'scripts.{family}.run',
                'controller', '--root', str(root), '--once'], cwd=root / 'source',
                env=dict(os.environ, PYTHONPATH=str(root / 'source'), PYTHONDONTWRITEBYTECODE='1'),
                stdout=stream, stderr=subprocess.STDOUT, timeout=240)
        state = read_json(root / 'controller.json')
        result[family] = dict(status=state['status'], returncode=completed.returncode,
            submitted=len(state['jobs'])-settings['jobs_at_handoff'], last_job=state['jobs'][-1]['job_id'])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control',required=True,type=Path)
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args()
    with (args.control / 'continuation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            state=dict(pid=os.getpid(),host=os.uname().nodename,updated_epoch=time.time())
            try:state['families']=poll(args.control)
            except Exception as error:state['error']=repr(error)
            write_json(args.control / 'continuation.json',state)
            if args.once:return
            time.sleep(30)


if __name__=='__main__':main()
