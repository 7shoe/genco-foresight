"""Snapshot-only A/D pilot evaluation, with one test episode per tile."""
import argparse
import fcntl
import os
from pathlib import Path
import sys
import time

p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--shard',type=int,required=True);a=p.parse_args()
import json
spec=json.loads(a.spec.read_text());sys.path.insert(0,spec['source'])
if not os.environ.get('PBS_JOBID'):raise RuntimeError('Pilot evaluation requires compute')
import numpy as np
import torch
from power_fm_foresight._config import load_config,read_json,sha256,write_json
from power_fm_foresight._data.network import load_network,graph_for_demand
from power_fm_foresight.genco import Genco
from power_fm_foresight.temporal import TemporalAdapter
from power_fm_foresight._training.forecaster import episode_arrays,atomic_npz
from power_fm_foresight._training.state import atomic_torch_save,tensor_digest
from power_fm_foresight._training.physics import physics, frozen, target_values
from power_fm_foresight._training.evaluation import score
from experiments.fleet.evaluate import headline
from experiments.replication_evaluate import completion_valid, finish_origin
from power_fm_foresight._training.evaluation import merged
from power_fm_foresight.scoring import variogram_pairs
from experiments.diagnose_stage_b import npz

torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.xpu.set_device(0)
cfg=load_config(spec['config']);net=load_network(config=cfg);data=Path(cfg['paths']['temporal_root']);out=Path(spec['out'])
out.mkdir(parents=True,exist_ok=True);digest=sha256(a.spec)
for path,h in spec['input_sha256'].items():assert sha256(path)==h
base=Genco(Path(cfg['paths']['assets_root'])/'genco-opfdata',cfg['checkpoint'],'xpu:0',grid=net)
scales=read_json(cfg['stage_b']['coordinate_scales']);model=TemporalAdapter(base,net,scales).to('xpu:0').eval()
best=torch.load(Path(cfg['paths']['training_run'])/'best.pt',map_location='cpu',weights_only=True)
assert best['manifest']==read_json(Path(cfg['paths']['training_run'])/'manifest.json')
missing,extra=model.load_state_dict(best['model'],strict=False);assert not extra and all(k.startswith('backbone.') for k in missing)
k=spec['local_K'];graph=graph_for_demand(net,net.bus[:,2],net.bus[:,3]);template=base.batch([graph]*(24*k))
episodes=[e for e in read_json(data/'manifest.json')['episodes'] if e['split']=='test'];assert len(episodes)==4
episode=episodes[a.shard];rows=read_json(data/f"episodes/{episode['episode_id']}.json")['records']
cache=out/'cache'/f"{episode['episode_id']}.pt";arrays=episode_arrays(rows,net)
if not cache.exists():
    embedding={'bus':[],'gen':[]}
    for lo in range(0,len(rows),24):
        if time.time()>float(os.environ['GENCO_FLEET_DEADLINE'])-120:sys.exit(75)
        graphs=[graph_for_demand(net,r['pd_mw'],r['qd_mvar']) for r in rows[lo:lo+24]]
        with torch.no_grad():_,emb=base.model(base.batch(graphs),return_embeddings=True)
        for kind,n in [('bus',net.n_bus),('gen',net.n_online_gen)]:embedding[kind].append(emb[kind].reshape(-1,n,emb[kind].shape[-1]).cpu())
    cached={f'embedding_{kind}':torch.cat(v) for kind,v in embedding.items()}
    cached['observed_bus']=torch.tensor(np.stack([arrays[q] for q in ['pd','qd','vm','va_rad']],-1),dtype=torch.float32)
    cached['observed_gen']=torch.tensor(np.stack([arrays[q][:,net.online_gen] for q in ['pg_mw','qg_mvar']],-1),dtype=torch.float32)
    for kind in ['bus','gen']:cached[f'available_{kind}']=torch.isfinite(cached[f'observed_{kind}'])
    cached['calendar']=torch.from_numpy(arrays['calendar'])
    atomic_torch_save(cache,cached)
cached=torch.load(cache,mmap=True,weights_only=True);pairs=variogram_pairs(scales,net)
for j,origin in enumerate(range(72,481,24)):
    identity=f"{episode['episode_id']}_{origin:04d}";folder=out/'origins'/identity
    if completion_valid(folder,digest):continue
    if time.time()>float(os.environ['GENCO_FLEET_DEADLINE'])-150:sys.exit(75)
    history={key:v[origin-72:origin].to('xpu:0') for key,v in cached.items() if key.startswith(('embedding_','observed_','available_'))}
    calendar=torch.cat([cached['calendar'][origin:origin+24],torch.arange(1,25)[:,None]/24],1).to('xpu:0')
    generator=torch.Generator().manual_seed(10042+a.shard*18+j);blocks=[];controls=[]
    with torch.no_grad():
        for chunk in range(64//k):
            noise={'factor':torch.randn(k,8,generator=generator),'bus':torch.randn(k,24,net.n_bus,3,generator=generator),'gen':torch.randn(k,24,net.n_online_gen,1,generator=generator)}
            result=model(history,calendar,template,{key:v.to('xpu:0') for key,v in noise.items()})
            block={key:v.cpu().numpy() for key,v in result['values'].items()};blocks.append(block)
            controls.append(frozen(base,net,block['pd_mw'],{24*k:template},qd=block['qd_mvar']))
    target=target_values(rows[origin:origin+24],net)
    for mode,parts in [('A',blocks),('D',controls)]:
        values=physics({key:np.concatenate([b[key] for b in parts]) for key in parts[0]},net)
        atomic_npz(folder/f'{mode}.npz',**values)
        row,detail=score(values,target,scales,pairs,mode,identity,a.shard*18+j)
        write_json(folder/f'{mode}_scores.json',row);atomic_npz(folder/f'{mode}_scores.npz',**detail)
    finish_origin(folder,digest);print(dict(pilot=net.case,origin=identity),flush=True)
write_json(out/f'shard_{a.shard}.json',dict(complete=True,episode=episode['episode_id']))
with (out/'merge.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    if all((out/f'shard_{i}.json').exists() for i in range(4)):
        selected=read_json(spec['headlines']);ids=[net.bus_ids.tolist().index(b) for b in selected['voltage_bus_ids']]
        result=dict(label=spec['label'],snapshot=read_json(spec['snapshot_record']),headlines={},modes={})
        for mode in ['A','D']:
            paths=sorted((out/'origins').glob(f'*/{mode}_scores.json'));assert len(paths)==72
            row=merged([read_json(p) for p in paths]);result['modes'][mode]=row;result['headlines'][mode]=headline(row,scales,ids)
        write_json(out/'report.json',result);write_json(out/'completion.json',dict(complete=True,origins=72,report_sha256=sha256(out/'report.json')))
