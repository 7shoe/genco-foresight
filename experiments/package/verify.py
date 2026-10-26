"""Execute the fixed Phase 2 protocol in one bounded compute allocation."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root; source = root/'source'
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def write(name, value):
        p=root/name; p.parent.mkdir(parents=True,exist_ok=True)
        tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(p)
    assert os.environ.get('PBS_JOBID')
    pin=json.loads((root/'source_pin.json').read_text())
    protocol=json.loads((root/'protocol.json').read_text())
    assert all(sha(source/p)==h for p,h in pin['files'].items())
    assert sha(root/'protocol.json')==pin['protocol_sha256']
    started=time.time(); results=[]
    for p in (source/'src').rglob('*'):
        if not p.is_file():continue
        assert '/lus/flare' not in p.read_text(), p
        if p.suffix == '.py':
            for n in ast.walk(ast.parse(p.read_text())):
                if isinstance(n, ast.ImportFrom):assert not (n.module or '').startswith(('experiments', 'scripts'))
                if isinstance(n, ast.Import):assert not any(x.name.startswith(('experiments', 'scripts')) for x in n.names)
    env=dict(os.environ,PYTHONPATH=os.pathsep.join((str(source/'src'),str(source))),PYTHONDONTWRITEBYTECODE='1')
    env.pop('GENCO_CONFIG',None)
    py=sys.executable
    reference=root.parents[1]/'v2/final_consolidation_20261026_v1/regression'
    commands=[
        ('installation',[py,str(source/'experiments/package/install_gate.py'),'--root',str(root),'--source',str(source)],300),
        ('suite',[py,'-m','pytest','-q','-ra','-p','no:cacheprovider','tests','--basetemp',str(root/'tmp/suite'),'--junitxml',str(root/'suite.xml')],900),
        ('regression',[py,str(source/'experiments/window/regression.py'),'--source',str(source),'--out',str(root/'regression'),'--reference',str(reference)],1200),
        ('scorer_loader',[py,str(source/'experiments/package/scorer_gate.py'),'--root',str(root),'--source',str(source)],900),
    ]
    for name,cmd,cap in commands:
        began=time.time(); remaining=protocol['job']['guard_seconds']-(began-started)
        code=124
        if remaining>0:
            with (root/'logs'/f'{name}.log').open('w') as log:
                worker=subprocess.Popen(cmd,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:code=worker.wait(timeout=min(cap,remaining))
                except subprocess.TimeoutExpired:
                    os.killpg(worker.pid,signal.SIGTERM)
                    try:worker.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(worker.pid,signal.SIGKILL);worker.wait()
        results.append(dict(check=name,returncode=code,seconds=time.time()-began))
        write('progress.json',dict(results=results,job_id=os.environ['PBS_JOBID']))
        if code:break
    passed=len(results)==len(commands) and all(x['returncode']==0 for x in results)
    if passed:
        suites=ET.parse(root/'suite.xml').getroot().iter('testsuite')
        counts=[s.attrib for s in suites]
        assert sum(int(s['tests']) for s in counts)>=143
        assert all(int(s.get(k,0))==0 for s in counts for k in ('failures','errors','skipped'))
        for name in ('installation.json','regression/gate.json','scorer_loader/gate.json'):
            assert json.loads((root/name).read_text())['passed']
    evidence=[p for p in root.rglob('*') if p.is_file() and (p.parent==root/'logs' or p.name in ('suite.xml','installation.json') or p.name=='gate.json' and p.parent!=root)]
    write('gate.json',dict(passed=passed,job_id=os.environ['PBS_JOBID'],source_commit=pin['source_commit'],
        source_pin_sha256=sha(root/'source_pin.json'),protocol_sha256=sha(root/'protocol.json'),results=results,
        worker_seconds=time.time()-started,reserved_node_seconds=3600,evidence_sha256={str(p.relative_to(root)):sha(p) for p in evidence}))
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
