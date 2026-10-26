"""Run bounded subset parity and hand accepted evidence to pinned workers."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from power_fm_foresight._config import ROOT, read_json, sha256, write_json
from experiments.subset_parity import serial_environment


def run_case(control, family, case):
    family_root = Path(read_json(control / 'control.json')[family]['root'])
    grid = family_root / case
    output = control / case
    protocol = output / 'protocol.json'
    if not protocol.exists():raise ValueError('Record subset IDs before verification')
    output.mkdir(exist_ok=True)
    began = time.time()
    env = serial_environment(os.environ)
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(ROOT))
    with (output / 'worker.log').open('a') as stream:
        process = subprocess.Popen([sys.executable, '-u', '-m', 'scripts.subset_parity',
            '--config', str(family_root / f'configs/{case}_data.yaml'), '--output', str(output)],
            cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=1200)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait()
            code = 124
    elapsed = time.time()-began
    budget_path = grid / 'seed0/budget.json'
    budget = read_json(budget_path)
    budget['data_seconds'] += elapsed
    write_json(budget_path, budget)
    write_json(output / 'attempt.json', dict(returncode=code, wall_seconds=elapsed,
        job_id=os.environ['PBS_JOBID'], budget=budget, protocol_sha256=sha256(protocol)))
    if code:
        write_json(output / 'gate.json', dict(passed=False, returncode=code,
            reason='20-minute cap' if code == 124 else 'See worker.log', wall_seconds=elapsed))
        return False
    gate = read_json(output / 'gate.json')
    if not gate['passed'] or not gate['bit_identical']:return False
    old = grid / 'stage_a/parity'; archive = grid / 'stage_a/parity_before_subset200'
    if old.is_symlink() and old.resolve() == (output / 'stage_a/parity').resolve():
        return True
    if archive.exists():raise ValueError('Previous parity archive already exists')
    old.rename(archive)
    old.symlink_to(output / 'stage_a/parity', target_is_directory=True)
    write_json(output / 'promotion.json', dict(original_parity=str(archive), active_parity=str(old),
        original_protocol_sha256=sha256(archive / 'protocol.json'), gate_sha256=sha256(output / 'gate.json')))
    card = grid / 'data/DATASET_CARD.md'; card.parent.mkdir(parents=True, exist_ok=True)
    card.write_text(f'# {case} temporal v4-law dataset\n\n'
        'Validation scope: **subset parity (200)**. A fixed random sample without replacement from the official/local test-ID intersection uses NumPy default_rng(42); IDs are sorted and recorded before execution.\n\n'
        f'CPU fp32 outputs are bit-identical to the pinned official Graphkit CLI. Elapsed {elapsed:.1f} seconds; cap1200 seconds. Protocol: {protocol}. Gate: {output / "gate.json"}.\n\n'
        'The independent32-snapshot official-parquet conversion evidence is retained separately. This is not full-intersection parity and does not bypass any solver or full temporal Stage A audit.\n')
    print(case, 'subset parity (200) PASSED', elapsed, flush=True)
    return True


def release_case118(control):
    root = Path(read_json(control / 'control.json')['fleet']['root'])
    case = 'case118_ieee'
    if not run_case(control, 'fleet', case):return
    blocked = root / case / 'blocked.json'
    if blocked.exists():
        previous = read_json(blocked)
        if previous.get('job') != '8963092' or previous.get('phase') != 'data':
            raise ValueError('Only the recorded case118 parity timeout may be released')
        shutil.copy2(blocked, control / case / 'superseded_parity_timeout.json')
        blocked.unlink()
    deadline = float(os.environ['GENCO_FLEET_START'])+3300
    write_json(control / case / 'released.json', dict(job_id=os.environ['PBS_JOBID'],
        deadline=deadline, time_epoch=time.time(), gate_sha256=sha256(control / case / 'gate.json')))
    code = """from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from experiments.fleet.run import chains, slot, verify_pin
import os
root=Path(os.environ['GENCO_FLEET_ROOT']);verify_pin(root)
deadline=float(os.environ['GENCO_FLEET_START'])+3300
with ThreadPoolExecutor(max_workers=3) as pool:
    futures=[pool.submit(slot,root,c,deadline) for c in chains() if c['case']=='case118_ieee']
    for future in futures:future.result()
"""
    env = serial_environment(os.environ)
    env['PYTHONPATH'] = str(root / 'source')
    with (control / case / 'released_workers.log').open('a') as stream:
        subprocess.run([sys.executable, '-u', '-c', code], cwd=root / 'source', env=env,
            stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=max(1,deadline-time.time()+90))
    write_json(control / case / 'release_boundary.json', dict(job_id=os.environ['PBS_JOBID'], time_epoch=time.time()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('family', choices=['fleet', 'pilots'])
    parser.add_argument('--control', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('PBS_JOBID'):raise RuntimeError('Compute allocation required')
    if args.family == 'fleet':
        release_case118(args.control)
    else:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run_case, args.control, 'pilots', case) for case in ('case500_goc','case2000_goc')]
            results = [f.result() for f in futures]
        root = Path(read_json(args.control / 'control.json')['pilots']['root'])
        job = os.environ['PBS_JOBID'].split('.')[0]
        write_json(root / 'jobs' / f'{job}_node0.json', dict(time_epoch=time.time(),
            subset_results=results, training_blocked='Separate cold-solver failures retained'))
        if not all(results):raise SystemExit(1)


if __name__ == '__main__':main()
