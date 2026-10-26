"""Run the recorded CLI gate in bounded allocations from an immutable snapshot."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def boundary(root):
    from power_fm_foresight.cli import main
    campaign=root/'campaign'
    chain=campaign/'case30_ieee/seed0/conditioned_lambda1_K8'
    if (chain/'boundary_checkpoint.json').exists():
        return
    if (campaign/'campaign.json').exists():
        argv=['train','--resume',str(campaign)]
    else:
        argv=['train','--config',str(root/'config.json'),'--case','30','--dataset',str(root/'datasets'),
              '--seed','0','--recipe','v4-off','--max-epochs','2','--device','xpu','--device-index','0','--out',str(campaign)]
    if main(argv,stop_after_steps=4090):
        raise RuntimeError('Boundary training command failed')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--phase',choices=('prepare','train','results','boundary'),required=True)
    args=parser.parse_args();root=args.root;source=Path(os.environ.get('GENCO_V2_SOURCE',root/'source'))
    assert os.environ.get('PBS_JOBID'), 'Verification requires a compute allocation'
    pin_path=Path(os.environ.get('GENCO_V2_PIN',root/'source_pin.json'))
    pin=json.loads(pin_path.read_text())
    assert all(sha(source/p)==h for p,h in pin['files'].items())
    assert all(sha(root/p)==h for p,h in pin['inputs'].items())
    if args.phase=='boundary':
        boundary(root)
        return
    protocol=json.loads((root/'protocol.json').read_text())
    for path in (source/'src/power_fm_foresight').rglob('*'):
        if not path.is_file() or '__pycache__' in path.parts:continue
        assert '/lus/flare' not in path.read_text(), path
        if path.suffix=='.py':
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node,ast.ImportFrom):assert not (node.module or '').startswith(('experiments','scripts'))
                if isinstance(node,ast.Import):assert not any(n.name.startswith(('experiments','scripts')) for n in node.names)
    py=str(root/'venvs/with_stack/bin/python')
    pff=str(root/'venvs/with_stack/bin/pff')
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PFF_ROOT=str(root.parents[2]))
    env.pop('GENCO_CONFIG',None)
    env['PYTHONPATH']=os.pathsep.join((str(source/'src'),str(source)))
    began=float(os.environ['GENCO_V2_JOB_START']);results=[]
    job=os.environ['PBS_JOBID'].split('.')[0]
    logs=root/'logs'/job;logs.mkdir(parents=True,exist_ok=True)
    chain=root/'campaign/case30_ieee/seed0/conditioned_lambda1_K8'
    old=root.parents[1]/'v2/final_consolidation_20261026_v1/regression'
    commands={
        'prepare':[
            ('installation',[sys.executable,str(source/'experiments/package/install_gate.py'),'--root',str(root),'--source',str(source)],300),
            ('pff_entrypoint',[pff,'--help'],30),
            ('long_entrypoint',[str(root/'venvs/with_stack/bin/power-fm-foresight'),'--help'],30),
            ('suite',[py,'-m','pytest','-q','-ra','-p','no:cacheprovider','tests','--basetemp',str(root/'tmp/suite'),'--junitxml',str(root/'suite.xml')],900),
            ('prepare',[pff,'prepare','--config',str(root/'config.json'),'--case','30','--dataset',str(root/'datasets'),
                        '--reuse',str(root.parents[2]/'data/temporal_case30_v4'),'--device','cpu','--parity-samples','200'],1800)],
        'train':[
            ('suite',[py,'-m','pytest','-q','-ra','-p','no:cacheprovider','tests','--basetemp',str(root/'tmp/final_suite'),'--junitxml',str(root/'final_suite.xml')],900),
            ('boundary',[py,str(source/'experiments/package/phase3_worker.py'),'--root',str(root),'--phase','boundary'],3300),
            ('resume',[pff,'train','--resume',str(root/'campaign')],3300),
            ('reproducibility',[pff,'verify','--check','reproducibility','--protocol',str(root/'reproducibility.json'),
                                '--runs',str(root/'campaign'),'--out',str(root/'verify_reproducibility')],120)],
        'results':[
            ('evaluate',[pff,'evaluate','--config',str(root/'config.json'),'--runs',str(root/'campaign'),
                         '--device','xpu:0','--out',str(root/'evaluation')],900),
            ('report',[pff,'report','--runs',str(root/'evaluation'),'--kind','summary',
                       '--format','md','json','csv','png','--out',str(root/'summary')],120),
            ('cli_regression',[pff,'verify','--check','regression','--protocol',str(root/'regression.json'),
                               '--out',str(root/'verify_regression')],1200),
            ('regression',[py,str(source/'experiments/window/regression.py'),'--source',str(source),
                           '--out',str(root/'regression'),'--reference',str(old)],1200),
            ('scorer_loader',[py,str(source/'experiments/package/scorer_gate.py'),'--root',str(root/'phase2'),
                             '--source',str(source)],900)],
    }[args.phase]
    incomplete=False
    for name,cmd,cap in commands:
        if args.phase=='train' and name=='resume' and not (chain/'boundary_checkpoint.json').exists():
            incomplete=True;break
        if args.phase=='train' and name=='reproducibility':
            if not json.loads((chain/'chain.status.json').read_text())['finished']:
                incomplete=True;break
        remaining=protocol['job']['guard_seconds']-(time.time()-began)
        if remaining<90:incomplete=True;break
        started=time.time();code=124
        local_env=dict(env,GENCO_JOB_START_EPOCH=str(started),PBS_WALLTIME=str(int(remaining+300)))
        write(logs/f'{name}.command.json',dict(argv=cmd,started=started,cap_seconds=min(cap,remaining)))
        with (logs/f'{name}.log').open('w') as log:
            worker=subprocess.Popen(cmd,cwd=source,env=local_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=worker.wait(timeout=min(cap,remaining))
            except subprocess.TimeoutExpired:
                os.killpg(worker.pid,signal.SIGTERM)
                try:worker.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(worker.pid,signal.SIGKILL);worker.wait()
        results.append(dict(check=name,returncode=code,seconds=time.time()-started))
        write(root/'progress.json',dict(phase=args.phase,job_id=job,results=results))
        if code:break
    passed=not incomplete and len(results)==len(commands) and all(r['returncode']==0 for r in results)
    if passed and args.phase=='prepare':
        suites=list(ET.parse(root/'suite.xml').getroot().iter('testsuite'))
        assert sum(int(s.attrib['tests']) for s in suites)>=158
        assert all(int(s.attrib.get(k,0))==0 for s in suites for k in ('failures','errors','skipped'))
        assert json.loads((root/'datasets/case30_ieee/dataset.json').read_text())['passed']
    if passed and args.phase=='train':
        assert json.loads((root/'verify_reproducibility/gate.json').read_text())['passed']
    if passed and args.phase=='results':
        for path in ('verify_regression/gate.json','regression/gate.json','phase2/scorer_loader/gate.json'):
            assert json.loads((root/path).read_text())['passed']
    write(root/'jobs'/f'{job}.json',dict(phase=args.phase,passed=passed,incomplete=incomplete,results=results,
        job_id=job,source_commit=pin['source_commit'],source_pin_sha256=sha(pin_path),
        protocol_sha256=sha(root/'protocol.json'),worker_seconds=time.time()-began,reserved_node_seconds=3600,
        evidence_sha256={str(p.relative_to(root)):sha(p) for p in logs.iterdir() if p.is_file()}))
    sys.exit(0 if passed or incomplete else 1)


if __name__=='__main__':
    main()
