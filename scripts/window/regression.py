"""Read-only case30 label and released-forward regression for each merge."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser(); p.add_argument('--source',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
p.add_argument('--reference',type=Path); p.add_argument('--audit-only',action='store_true'); a=p.parse_args()
if not os.environ.get('PBS_JOBID'): raise RuntimeError('Regression requires a compute allocation')
sys.path.insert(0,str(a.source)); a.out.mkdir(parents=True,exist_ok=True)
from common import load_config,read_json,sha256,write_json
import numpy as np
from data.loader.network import load_network
from solver.checks import check_solution
cfg=load_config(a.source/'configs/v4.yaml'); net=load_network(config=cfg)
shard=Path(cfg['paths']['temporal_root'])/'episodes/train_00.json'
rows=read_json(shard)['records'][:50]
checks=[check_solution(net,r['pd_mw'],r['qd_mvar'],r['solver']) for r in rows]
audit=dict(passed=all(c['valid'] for c in checks),hours=list(range(50)),checks=checks,input_sha256=sha256(shard))
if a.reference:
    reference=read_json(a.reference/'audit.json')
    assert reference['input_sha256']==audit['input_sha256']
    for old,new in zip(reference['checks'],checks):
        assert old['valid']==new['valid']
        for k,v in old['errors'].items(): np.testing.assert_allclose(new['errors'][k],v,atol=1e-12,rtol=1e-10)
else:
    for row,check in zip(rows,checks):
        old=row['solver']['checks']
        assert check['valid']==old['valid']
        for key,value in old['errors'].items(): np.testing.assert_allclose(check['errors'][key],value,atol=1e-12,rtol=1e-10)
write_json(a.out/'audit.json',audit); assert audit['passed']
if a.audit_only: sys.exit(0)
tests=[f'tests/test_{name}.py' for name in ('solver','model','adapter','training','case_identity','multigrid','v2_harness','v2_stage1','fleet','pilots','pilot_repair','subset_parity','recovery') if (a.source/f'tests/test_{name}.py').exists()]
env=dict(os.environ,GENCO_CONFIG=str(a.source/'configs/v4.yaml'),PYTHONPATH=str(a.source),PYTHONDONTWRITEBYTECODE='1')
with (a.out/'tests.log').open('w') as log:
    r=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider',*tests,'--basetemp',str(a.out/'tmp'),'--junitxml',str(a.out/'tests.xml')],cwd=a.source,env=env,stdout=log,stderr=subprocess.STDOUT)
assert r.returncode==0, f'Regression tests failed: {a.out}/tests.log'
import torch
from scripts.forecast_partition import components,adapter_samples
from scripts.evaluate_stage_b import frozen
from scripts.replication_evaluate import differences
torch.xpu.set_device(0)
net,base,model,scales,templates=components(cfg,'xpu:0')
b=Path(cfg['paths']['runs_root'])/'stage_b'; protocol=read_json(b/'protocol.json'); cache=read_json(b/'cache/manifest.json')
results=[]
for index,origin in enumerate(protocol['origins'][:2]):
    item=cache[origin['episode_id']]; assert sha256(item['path'])==item['sha256']
    cached=torch.load(item['path'],mmap=True,weights_only=True)['data']
    values,_=adapter_samples(base,model,net,cached,origin['origin'],templates,10042+index,'xpu:0')
    modes={'A':values,'D':frozen(base,net,values['pd_mw'],templates,qd=values['qd_mvar'])}
    record=dict(origin=origin['id'],modes={})
    for mode,values in modes.items():
        path=b/'origins'/origin['id']/f'{mode}.npz'
        with np.load(path) as old:
            errors={k:float(np.nanmax(np.abs(values[k]-old[k]))) for k in old.files}
            passed=all(np.allclose(values[k],old[k],atol=1e-5,rtol=1e-5,equal_nan=True) for k in old.files)
        record['modes'][mode]=dict(passed=passed,max_absolute=errors,reference_sha256=sha256(path))
        assert passed, record
    results.append(record)
    write_json(a.out/'forward.json',dict(origins=results))
write_json(a.out/'gate.json',dict(passed=True,audit_hours=50,forward_origins=2,tests=tests,
    source=str(a.source),protocol='phase_b.protocol.json',evidence_sha256={p.name:sha256(p) for p in a.out.iterdir() if p.is_file()}))
print('MERGE REGRESSION PASSED',a.out,flush=True)
