"""Verify A0 path identity, detached backbone gradients and production updates."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
from power_fm_foresight._config import ROOT,load_config,read_json,sha256,write_json
if not os.environ.get('PBS_JOBID'):raise RuntimeError('A0 verification requires compute')
out=a.root/'phase_c/gate';out.mkdir(parents=True,exist_ok=True)
with (out/'tests.log').open('w') as log:
    result=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider','tests/test_a0.py','tests/test_adapter.py',
        '--basetemp',str(out/'tmp'),'--junitxml',str(out/'tests.xml')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
        env=dict(os.environ,GENCO_CONFIG=str(ROOT/'configs/v4.yaml')))
assert result.returncode==0,'A0 tests failed'
import torch
from power_fm_foresight._training.loop import initialize,training_step
from power_fm_foresight._training.state import tensor_digest
torch.xpu.set_device(0);rows=[]
for case in ('case30_ieee','case57_ieee','case118_ieee'):
    cfg=load_config(a.root/'configs'/f'a0_{case}_seed0.yaml')
    model,opt,sched,template,cache=initialize(cfg,'xpu')
    torch.use_deterministic_algorithms(True)
    with torch.no_grad():
        expected=model.backbone(template);actual=model.unconditioned(template)
        assert all(torch.equal(actual[k],expected[k]) for k in expected)
        zero=model.demand_only_output(template,torch.zeros_like(expected['bus'][:,:2]),torch.zeros_like(expected['gen']))
        assert all(torch.equal(zero[k],expected[k]) for k in expected)
    torch.use_deterministic_algorithms(False)
    before=tensor_digest(model.backbone.state_dict())
    metric=training_step(model,opt,sched,template,cache,cache.indices['train'][0],cfg,'xpu')
    assert tensor_digest(model.backbone.state_dict())==before
    assert all(p.grad is None for p in model.backbone.parameters())
    assert model.bus_head.weight.grad[0].abs().sum()>0
    rows.append(dict(case=case,unconditioned_exact=True,zero_innovation_exact=True,metrics=metric,backbone_unchanged=True))
    del model,opt,sched,template,cache;torch.xpu.empty_cache()
write_json(out/'gate.json',dict(passed=True,cases=rows,source=str(ROOT),tests_sha256=sha256(out/'tests.xml'),
    protocol_sha256=sha256(a.root/'phase_c.protocol.json')))
print('A0 GATE PASSED',flush=True)
