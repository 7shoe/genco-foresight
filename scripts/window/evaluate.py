"""Disjoint-origin evaluation and equal-weight mixtures using the frozen scorer."""
import argparse
from pathlib import Path
import os
import sys
import time
import fcntl

p=argparse.ArgumentParser(); p.add_argument('--spec',type=Path,required=True); p.add_argument('--shard',type=int,default=0)
p.add_argument('--shards',type=int,default=4); a=p.parse_args()
import json
spec=json.loads(a.spec.read_text()); sys.path.insert(0,spec['source'])
if not os.environ.get('PBS_JOBID'): raise RuntimeError('Evaluation requires a compute allocation')
import numpy as np
import torch
from common import load_config,read_json,sha256,write_json
from scripts.fleet.evaluate import score,headline
from scripts.replication_evaluate import completion_valid,finish_origin,merged
from scripts.diagnose_stage_b import npz
from training.forecaster import atomic_npz
from data.loader.network import load_network

torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
cfg=load_config(spec['config']); net=load_network(config=cfg)
scales=read_json(cfg['stage_b']['coordinate_scales']); baseline=Path(spec['baseline'])
protocol=read_json(baseline/'protocol.json'); origins=protocol['origins']
pairs=npz(spec['pairs'])['pairs']; out=Path(spec['out']); out.mkdir(parents=True,exist_ok=True)
digest=sha256(a.spec)
for path,h in spec['input_sha256'].items():
    if sha256(path)!=h: raise ValueError(f'Evaluation input changed: {path}')
if spec['kind']=='checkpoint':
    assert torch.xpu.device_count()==1; torch.xpu.set_device(0)
    from scripts.forecast_partition import components,adapter_samples
    from scripts.evaluate_stage_b import frozen
    net,base,model,scales,templates=components(cfg,'xpu:0')
    cache=read_json(spec['cache_manifest']); cached={}
for index,origin in enumerate(origins):
    if index%a.shards!=a.shard: continue
    folder=out/'origins'/origin['id']
    if completion_valid(folder,digest): continue
    if time.time()>float(os.environ['GENCO_FLEET_DEADLINE'])-120: sys.exit(75)
    folder.mkdir(parents=True,exist_ok=True); original=baseline/'origins'/origin['id']
    target=npz(original/'targets.npz'); inputs={}
    if spec['kind']=='checkpoint':
        name=origin['episode_id'];item=cache[name]
        if name not in cached:
            assert sha256(item['path'])==item['sha256']
            cached[name]=torch.load(item['path'],mmap=True,weights_only=True)['data']
        if not (folder/'A.npz').exists():
            values,_=adapter_samples(base,model,net,cached[name],origin['origin'],templates,10042+index,'xpu:0')
            atomic_npz(folder/'A.npz',**values)
        values=npz(folder/'A.npz')
        if not (folder/'D.npz').exists(): atomic_npz(folder/'D.npz',**frozen(base,net,values['pd_mw'],templates,qd=values['qd_mvar']))
    for mode in ('A','D'):
        if spec['kind']=='mixture':
            paths=[Path(b)/'origins'/origin['id']/f'{mode}.npz' for b in spec['members']]
            blocks=[npz(path) for path in paths]; values={k:np.concatenate([b[k] for b in blocks]) for k in blocks[0]}
            inputs.update({str(path):sha256(path) for path in paths})
        else: values=npz(folder/f'{mode}.npz')
        row,arrays=score(values,target,scales,pairs,mode,origin['id'],index)
        write_json(folder/f'{mode}_scores.json',row); atomic_npz(folder/f'{mode}_scores.npz',**arrays)
    finish_origin(folder,digest,dict(input_sha256=inputs))
    print(dict(kind=spec['kind'],origin=origin['id'],shard=a.shard),flush=True)
write_json(out/f'shard_{a.shard}.json',dict(complete=True,protocol_sha256=digest,shard=a.shard,shards=a.shards))
with (out/'merge.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    if all((out/f'shard_{i}.json').exists() for i in range(a.shards)):
        selected=read_json(spec['headlines']);indices=[net.bus_ids.tolist().index(b) for b in selected['voltage_bus_ids']]
        result=dict(kind=spec['kind'],modes={},headlines={},protocol_sha256=digest,
                    same_input_P_Ipopt=read_json(spec['audit'])['same_input_P_Ipopt'],
                    seeds=spec.get('seeds',[spec.get('training_seed')]),samples=64*len(spec.get('members',[0])))
        for mode in ('C','P','P0','S','A','D'):
            if mode in ('A','D'):
                assert all(completion_valid(out/'origins'/o['id'],digest) for o in origins)
                row=merged([read_json(out/'origins'/o['id']/f'{mode}_scores.json') for o in origins])
            else:row=read_json(baseline/f'{mode}.json')
            result['modes'][mode]=row;result['headlines'][mode]=headline(row,scales,indices)
        write_json(out/'report.json',result)
        write_json(out/'completion.json',dict(complete=True,origins=len(origins),report_sha256=sha256(out/'report.json'),
                   completed_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()))
