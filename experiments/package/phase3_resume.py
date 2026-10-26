"""One authorized Phase 3 restart, with an explicit source-transition receipt."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def check_source(root):
    pin = read(root / 'source_pin.json')
    for name, digest in pin['files'].items():
        if sha(root / 'source' / name) != digest:
            raise ValueError(f'Pinned source changed: {name}')
    protocol = read(root / 'protocol.json')
    if sha(root / 'protocol.json') != pin['protocol_sha256']:
        raise ValueError('Restart protocol changed')
    if sha(__file__) != pin.get('driver_sha256', protocol['driver_sha256']):
        raise ValueError('Restart driver changed')
    for name, digest in protocol['protocol_inputs_sha256'].items():
        if sha(root / name) != digest:
            raise ValueError(f'Protocol input changed: {name}')
    return protocol


def dispatch(root, argv):
    protocol = check_source(root)
    from power_fm_foresight import campaign
    from power_fm_foresight.cli import main
    if argv and argv[0] == 'train':
        expected = ['train', '--resume', protocol['resume_run'], '--max-epochs', '2']
        if argv != expected:
            raise ValueError('This restart driver admits only the declared resume command')
        actual = campaign.source_identity()
        transition = protocol['source_transition']
        if actual != transition['execution_source']:
            raise ValueError('Execution package differs from the authorized source')

        def authorized_resume(options, explicit):
            if set(explicit) != {'command', 'resume', 'max_epochs'} or explicit['max_epochs'] != 2:
                raise ValueError('Unexpected restart override')
            value = campaign.load_campaign(explicit['resume'])
            if value['source'] != transition['checkpoint_campaign_source']:
                raise ValueError('Campaign source differs from the authorized predecessor')
            if sha(Path(value['root']) / 'campaign.json') != protocol['campaign_sha256']:
                raise ValueError('Frozen campaign changed')
            if value['options']['max_epochs'] != 2 or value['options']['parallel'] != 1:
                raise ValueError('Frozen epoch cap or device layout differs')
            changed = {k for k in actual['files'] if actual['files'][k] != value['source']['files'].get(k)}
            if set(actual['files']) != set(value['source']['files']) or changed != set(transition['changed_package_files']):
                raise ValueError('Source transition exceeds its declared file set')
            receipt = dict(command='pff train --resume', protocol_sha256=sha(root / 'protocol.json'),
                           creation_source=value['source'], execution_source=actual,
                           configuration=value['options'], passed=True,
                           policy='One authorized controller-only transition; original manifests retained')
            write(root / 'source_transition.json', receipt)
            write(Path(value['root']) / 'invocations' / f'authorized_restart_{os.environ["PBS_JOBID"].split(".")[0]}.json', receipt)
            return value

        # The package's ordinary source-equality rule remains unchanged.
        campaign.resume_campaign = authorized_resume
    return main(argv)


def worker(root):
    if not os.environ.get('PBS_JOBID'):
        raise RuntimeError('This verification requires a compute allocation')
    protocol = check_source(root)
    source = root / 'source'
    for path, digest in protocol['initial_input_sha256'].items():
        if sha(path) != digest:
            raise ValueError(f'Restart input changed: {path}')
    import torch
    checkpoint = torch.load(protocol['checkpoint'], map_location='cpu', weights_only=True)
    progress = checkpoint['progress']
    if (progress['steps'], progress['epoch'], progress['step_in_epoch']) != (4090, 0, 4090):
        raise ValueError('The restart must begin at the saved pre-validation boundary')
    if checkpoint['manifest'] != read(Path(protocol['checkpoint']).parent / 'manifest.json'):
        raise ValueError('Checkpoint manifest differs from its original run')
    write(root / 'checkpoint_preflight.json', dict(passed=True, progress=progress,
          checkpoint_sha256=sha(protocol['checkpoint']), full_state_keys=sorted(checkpoint)))
    del checkpoint
    pff = str(root / 'bin/pff')
    run = protocol['resume_run']
    py = sys.executable
    commands = [
        ('resume', [pff, 'train', '--resume', run, '--max-epochs', '2'], 2200),
        ('evaluate', [pff, 'evaluate', '--config', str(root / 'config.json'), '--runs', run,
                      '--device', 'xpu:0', '--out', str(root / 'evaluation')], 900),
        ('report', [pff, 'report', '--runs', str(root / 'evaluation'), '--kind', 'summary',
                    '--format', 'md', 'json', 'csv', '--out', str(root / 'summary')], 120),
        ('regression', [pff, 'verify', '--check', 'regression', '--protocol', str(root / 'regression.json'),
                        '--out', str(root / 'verify_regression')], 1200),
        ('reproducibility', [pff, 'verify', '--check', 'reproducibility', '--protocol', str(root / 'reproducibility.json'),
                             '--runs', run, '--out', str(root / 'verify_reproducibility')], 120),
        ('suite', [py, '-m', 'pytest', '-q', '-ra', '-p', 'no:cacheprovider', 'tests',
                   '--basetemp', str(root / 'tmp/suite'), '--junitxml', str(root / 'suite.xml')], 180),
    ]
    began = float(os.environ['GENCO_V2_JOB_START'])
    deadline = began + protocol['job']['guard_seconds']
    results = []
    incomplete = False
    for name, command, cap in commands:
        remaining = deadline - time.time()
        if remaining < 90:
            incomplete = name != 'suite'
            break
        started = time.time()
        local = dict(os.environ, GENCO_JOB_START_EPOCH=str(started), PBS_WALLTIME=str(int(remaining + 300)))
        if name == 'resume':
            local['PBS_WALLTIME'] = str(int(min(remaining, cap) + 300))
        write(root / 'logs' / f'{name}.command.json', dict(argv=command, cap_seconds=min(remaining, cap), started=started))
        code = 124
        with (root / 'logs' / f'{name}.log').open('w') as stream:
            process = subprocess.Popen(command, cwd=source, env=local, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=min(remaining, cap) + 10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        results.append(dict(check=name, returncode=code, seconds=time.time() - started))
        write(root / 'progress.json', dict(results=results, job_id=os.environ['PBS_JOBID']))
        if code and name not in ('regression', 'reproducibility', 'suite'):
            break
        if name == 'resume':
            status = read(Path(run) / 'adapter/status.json')
            if status['epochs_reached'] != 2 or status['status'] != 'finished':
                incomplete = True
                break
    required = [r for r in results if r['check'] != 'suite']
    passed = len(required) == len(commands) - 1 and not incomplete and all(r['returncode'] == 0 for r in required)
    if passed:
        passed = all(read(root / name / 'gate.json')['passed'] for name in ('verify_regression', 'verify_reproducibility'))
    write(root / 'gate.json', dict(passed=passed, incomplete=incomplete, results=results,
          job_id=os.environ['PBS_JOBID'], source_commit=read(root / 'source_pin.json')['source_commit'],
          suite_policy='Diagnostic only; run last; excluded from gate verdict',
          protocol_sha256=sha(root / 'protocol.json'), worker_seconds=time.time() - began,
          reserved_node_seconds=3600, further_submissions_allowed=False,
          evidence_sha256={str(p.relative_to(root)):sha(p) for p in (root / 'logs').iterdir() if p.is_file()}))
    return 0 if passed else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--dispatch', action='store_true')
    args, argv = parser.parse_known_args()
    if argv[:1] == ['--']:
        argv = argv[1:]
    try:
        result = dispatch(args.root, argv) if args.dispatch else worker(args.root)
    except Exception as error:
        if not args.dispatch:
            write(args.root / 'gate.json', dict(passed=False, failure=repr(error), further_submissions_allowed=False))
        raise
    raise SystemExit(result)
