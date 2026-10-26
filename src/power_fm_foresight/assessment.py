"""Resumable local eval & joint mixtures using the frozen scorer"""

from copy import deepcopy
import hashlib
import json
import fcntl
from pathlib import Path
import numpy as np
import torch

from ._config import read_json, sha256, write_json
from .campaign import default_output, record_manifest, resolve_runs, run_config, select_device
from .data import graph_for_demand, load_network
from .scoring import UNITS, variogram_pairs, score, merged
from .training import components, adapter_samples, cache_partition, frozen, target_values, mixture_samples
from ._training.headlines import subset_summary
from ._training.audits import same_input_p
from ._training.forecaster import RidgeForecaster, atomic_npz, episode_arrays, prepare as prepare_ridge
from ._artifacts import completion_valid, finish_origin


def npz(path):
    with np.load(path) as f:return {k:f[k] for k in f.files}


def headline(row,scales,indices):
    vm=subset_summary(row,'vm',np.asarray(indices,dtype=int))
    pg=subset_summary(row,'pg_mw',np.flatnonzero(scales['features']['pg_mw']['joint_score_mask']))
    return dict(joint_es=row['joint_energy_score'],vm_crps=vm['crps'],
        vm_coverage90_pct=None if vm['coverage_90'] is None else 100*vm['coverage_90'],
        pg_free_mae=pg['mae'],p_imbalance=row['physical_balance']['p_mean_abs_mw'])


def headline_indices(cfg,net):
    path=Path(cfg['paths']['stage_a_run'])/'stage_a_followup/headline_components.json'
    if not path.exists():return None,f'Headline assembly skipped: missing preparation metadata {path}'
    return [net.bus_ids.tolist().index(i) for i in read_json(path)['voltage_bus_ids']],None


def run_identity(path,cfg):
    chain=read_json(path/'chain.json') if (path/'chain.json').exists() else {}
    adapter=Path(cfg['paths']['training_run']);status=read_json(adapter/'status.json')
    reason=status.get('stop_reason')
    return dict(case=cfg['task']['canonical_case'].removeprefix('pglib_opf_'),seed=cfg['seed'],
        variant=cfg['stage_b']['adapter'].get('model_policy','conditioned'),balance_weight=cfg['stage_b']['loss']['power_balance_weight'],
        samples=cfg['stage_b']['training_samples'],lr=cfg['stage_b']['optimizer']['lr'],label=chain.get('label',''),
        pilot=chain.get('pilot',cfg['stage_b']['training_samples']!=8),budget_capped=reason in ('training_budget','six_hour_budget'),
        finished=status.get('status')=='finished',stop_reason=reason,best_epoch=status.get('best_epoch'),
        epochs=status.get('epochs_reached'),run=str(path))


def compatible_key(record):
    fields=('case','variant','balance_weight','samples','lr','pilot','budget_capped')
    meta={k:record['identity'][k] for k in fields}
    for k in ('dataset_sha256','scales_sha256','checkpoint_base','precision','deterministic_algorithms','partition','eval_seed','forecast_samples','origins'):
        meta[k]=record['comparison'][k]
    return hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()


def baseline_inputs(cfg,net,out,origins):
    settings=cfg.get('evaluation',{})
    old=Path(settings['baseline_root']) if settings.get('baseline_root') else None
    if old:
        protocol=read_json(old/'protocol.json')
        if protocol['dataset_manifest_sha256']!=sha256(Path(cfg['paths']['temporal_root'])/'manifest.json'):
            raise ValueError('Baseline dataset differs')
        if protocol['coordinate_scales_sha256']!=sha256(cfg['stage_b']['coordinate_scales']):raise ValueError('Baseline coordinate scales differ')
        forecaster=Path(protocol.get('forecaster_path',old/'forecaster'));pool=Path(protocol.get('pool_path',old/'controls/climatology_pool.npz'))
        if protocol.get('forecaster_manifest_sha256') and sha256(forecaster/'manifest.json')!=protocol['forecaster_manifest_sha256']:
            raise ValueError('Frozen forecaster manifest changed')
        if protocol.get('pool_sha256') and sha256(pool)!=protocol['pool_sha256']:raise ValueError('Frozen climatology pool changed')
        return RidgeForecaster(forecaster),npz(pool),old
    forecaster=out/'forecaster';prepare_ridge(cfg,net,forecaster)
    pool=out/'climatology_pool.npz'
    if not pool.exists():
        manifest=read_json(Path(cfg['paths']['temporal_root'])/'manifest.json');blocks=[]
        for ep in manifest['episodes']:
            if ep['split']!='train':continue
            values=target_values(read_json(Path(cfg['paths']['temporal_root'])/f"episodes/{ep['episode_id']}.json")['records'],net)
            for start in range(0,481,24):
                if not all(np.isfinite(values[k][start:start+24]).all() for k in UNITS):raise ValueError('Climatology requires resolved train days')
                blocks.append({k:v[start:start+24] for k,v in values.items()})
        atomic_npz(pool,**{k:np.stack([b[k] for b in blocks]) for k in blocks[0]})
    return RidgeForecaster(forecaster),npz(pool),None


def evaluate_one(path,out,options):
    cfg=run_config(path,options['configuration']);identity=run_identity(path,cfg)
    if (identity['budget_capped'] or not identity['finished']) and not options['allow_budget_capped']:
        raise ValueError(f'Run is budget-capped or unfinished: {path}; use --allow-budget-capped')
    if not identity['finished']:identity['budget_capped']=True
    partition='model_selection' if options['partition']=='selection' else 'test'
    data=Path(cfg['paths']['temporal_root']);dataset=read_json(data/'manifest.json')
    origins=[dict(episode_id=e['episode_id'],origin=o,id=f"{e['episode_id']}_{o:04d}") for e in dataset['episodes'] if e['split']==partition for o in range(72,481,24)]
    origin_indices={origin['id']:i for i,origin in enumerate(origins)}
    selected=options['configuration'].get('evaluation',{}).get('origins')
    if selected:
        unknown=set(selected)-{o['id'] for o in origins}
        if unknown:raise ValueError(f'Unknown evaluation origins: {sorted(unknown)}')
        origins=[o for o in origins if o['id'] in selected]
    execution=deepcopy(cfg);execution['evaluation']=deepcopy(options['configuration'].get('evaluation',cfg.get('evaluation',{})))
    train=Path(cfg['paths']['training_run']);checkpoint=train/f"{options['checkpoint']}.pt"
    comparison=dict(dataset_sha256=sha256(data/'manifest.json'),scales_sha256=sha256(cfg['stage_b']['coordinate_scales']),
        checkpoint_base=cfg['checkpoint'],precision='fp32',deterministic_algorithms=cfg['stage_b'].get('deterministic_algorithms',True),
        partition=partition,eval_seed=options['eval_seed'],forecast_samples=options['samples'],origins=[o['id'] for o in origins],
        origin_indices={o['id']:origin_indices[o['id']] for o in origins})
    protocol=dict(identity=identity,comparison=comparison,checkpoint_sha256=sha256(checkpoint),checkpoint=options['checkpoint'],
                  origins=origins,modes=options['modes'],same_input_audit='P/Ipopt identical inputs, leads 0/6/12/18/23')
    baseline=execution['evaluation'].get('baseline_root')
    if baseline:
        old=Path(baseline);saved=read_json(old/'protocol.json')
        forecaster=Path(saved.get('forecaster_path',old/'forecaster'))
        inputs=[old/'protocol.json',Path(saved.get('pool_path',old/'controls/climatology_pool.npz')),
                *(forecaster/n for n in ('manifest.json','coefficients.npz','residual_blocks.npz'))]
        protocol['baseline_input_sha256']={str(p):sha256(p) for p in inputs}
    out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists() and read_json(out/'protocol.json')!=protocol:raise ValueError('Evaluation protocol changed')
    write_json(out/'protocol.json',protocol)
    record_manifest(out,'evaluate',protocol,device=options['device'],seeds=[identity['seed'],options['eval_seed']],paths=dict(run=path,output=out))
    digest=sha256(out/'protocol.json')
    device=options['device'];kind=select_device(device)
    selected_device=str(torch.device(device)) if ':' in device else (f'{kind}:0' if kind!='cpu' else 'cpu')
    net,base,model,scales,templates=components(cfg,selected_device,checkpoint=options['checkpoint'])
    count=options['samples']%8
    if count and count*24 not in templates:
        graph=graph_for_demand(net,net.bus[:,2],net.bus[:,3]);templates[count*24]=base.batch([graph]*(count*24))
    old=Path(execution['evaluation']['baseline_root']) if execution['evaluation'].get('baseline_root') else None
    if old and partition=='test':
        cache=read_json(old/'cache/manifest.json')
    else:
        with (out/'cache.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not (out/'cache/manifest.json').exists():cache_partition(cfg,out,partition,base,net)
            cache=read_json(out/'cache/manifest.json')
    ridge=pool=None;baseline_unavailable=None
    try:
        with (out/'baseline.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            ridge,pool,_=baseline_inputs(execution,net,out,origins)
    except ValueError as error:
        if 'requires resolved' not in str(error) and 'Missing historical covariates' not in str(error):raise
        baseline_unavailable=str(error)
    solver=None;cached={};episodes={}
    try:
        for position,origin in enumerate(origins):
            if position%options['shards']!=options['shard']:continue
            index=origin_indices[origin['id']]
            folder=out/'origins'/origin['id']
            if completion_valid(folder,digest):continue
            name,o=origin['episode_id'],origin['origin'];seed=10000+options['eval_seed']+index
            if name not in cached:
                item=cache[name]
                if sha256(item['path'])!=item['sha256']:raise ValueError('Evaluation cache changed')
                cached[name]=torch.load(item['path'],mmap=True,weights_only=True)['data']
                records=read_json(data/f'episodes/{name}.json')['records'];episodes[name]=(records,episode_arrays(records,net))
            records,arrays=episodes[name];target=target_values(records[o:o+24],net);values={};unavailable={}
            if ridge:
                mean,samples,_=ridge.forecast(arrays,o,seed,samples=options['samples'])
                values['P']=frozen(base,net,mean[None],templates)
                if 'S' in options['modes']:values['S']=frozen(base,net,samples,templates)
                if 'P0' in options['modes']:values['P0']={k:np.repeat(v,options['samples'],0) for k,v in values['P'].items()}
                if 'C' in options['modes']:
                    draws=np.random.default_rng(seed+10000).integers(0,len(next(iter(pool.values()))),size=options['samples'])
                    values['C']={k:v[draws] for k,v in pool.items()}
            else:
                unavailable.update({m:baseline_unavailable for m in options['modes'] if m in ('C','P','P0','S')})
            if any(m in options['modes'] for m in ('A','D')):
                values['A'],_=adapter_samples(base,model,net,cached[name],o,templates,seed,selected_device,samples=options['samples'])
                if 'D' in options['modes']:values['D']=frozen(base,net,values['A']['pd_mw'],templates,qd=values['A']['qd_mvar'])
            atomic_npz(folder/'targets.npz',**target)
            for mode in options['modes']:
                if mode not in values:continue
                atomic_npz(folder/f'{mode}.npz',**values[mode])
                row,detail=score(values[mode],target,scales,variogram_pairs(scales,net),mode,origin['id'],index,eval_seed=options['eval_seed'])
                write_json(folder/f'{mode}_scores.json',row);atomic_npz(folder/f'{mode}_scores.npz',**detail)
            if 'P' in values:
                if solver is None:
                    from .solver import IpoptSolver
                    solver=IpoptSolver(cfg,out/f'solver_{options["shard"]}.log',network=net)
                audit=same_input_p(folder,values['P'],net,solver)
            else:
                audit=dict(same_inputs=None,unavailable=baseline_unavailable);write_json(folder/'P_audit.json',audit)
            write_json(folder/'unavailable.json',unavailable)
            finish_origin(folder,digest,dict(same_input_audit=audit))
    finally:
        if solver:solver.close()
    if not all(completion_valid(out/'origins'/o['id'],digest) for o in origins):return dict(complete=False,output=str(out))
    indices,note=headline_indices(cfg,net)
    report=dict(identity=identity,comparison=comparison,output=str(out),complete=True,headlines={},modes={},unavailable={},
        same_input_audit={o['id']:read_json(out/'origins'/o['id']/'P_audit.json') for o in origins},config=cfg,scales=cfg['stage_b']['coordinate_scales'])
    if note:report['notes']=[note]
    for mode in options['modes']:
        paths=[out/'origins'/o['id']/f'{mode}_scores.json' for o in origins]
        if not all(p.exists() for p in paths):
            report['headlines'][mode]=None;report['unavailable'][mode]='Unavailable baseline coordinates or unresolved training history';continue
        row=merged([read_json(p) for p in paths]);write_json(out/f'{mode}.json',row)
        report['headlines'][mode]=headline(row,scales,indices) if indices is not None else None
        report['modes'][mode]=str(out/f'{mode}.json')
    if sha256(checkpoint)!=protocol['checkpoint_sha256']:raise ValueError('Checkpoint changed during evaluation; use an immutable checkpoint and a new output')
    if not all(sha256(p)==h for p,h in protocol.get('baseline_input_sha256',{}).items()):raise ValueError('Frozen baseline changed during evaluation')
    write_json(out/'report.json',report)
    return report


def ensemble_groups(reports):
    groups={}
    for report in reports:
        if report.get('complete'):groups.setdefault(compatible_key(report),[]).append(report)
    for key,members in groups.items():
        members.sort(key=lambda r:r['identity']['seed'])
        if len({r['identity']['seed'] for r in members})!=len(members):raise ValueError('Mixture has duplicate training seeds within a compatible group')
    return groups


def evaluate_mixtures(reports,sizes,out,modes,eval_seed):
    result=[]
    for key,members in ensemble_groups(reports).items():
        for size in sizes:
            if size>len(members):
                result.append(dict(group=key,size=size,available=False,reason='Insufficient finished compatible seeds'));continue
            selected=members[:size];dest=out/'mixtures'/key[:12]/f'k{size}';cfg=selected[0]['config']
            net=load_network(config=cfg);scales=read_json(selected[0]['scales']);pairs=variogram_pairs(scales,net)
            indices,note=headline_indices(cfg,net)
            record=dict(group=key,size=size,available=True,seeds=[r['identity']['seed'] for r in selected],weights=[1/size]*size,headlines={},identity=selected[0]['identity'],comparison=selected[0]['comparison'])
            if note:record['notes']=[note]
            record_manifest(dest,'evaluate mixture',record,device='cpu',seeds=record['seeds'],paths=dict(members=[r['output'] for r in selected]))
            for mode in modes:
                if mode not in ('A','D'):
                    record['headlines'][mode]=selected[0]['headlines'].get(mode)
                    record.setdefault('shared_baselines',[]).append(mode)
                    continue
                rows=[];available=True
                for origin in selected[0]['comparison']['origins']:
                    index=selected[0]['comparison']['origin_indices'][origin]
                    paths=[Path(r['output'])/'origins'/origin/f'{mode}.npz' for r in selected]
                    if not all(p.exists() for p in paths):available=False;break
                    blocks=[npz(p) for p in paths]
                    if len({len(next(iter(b.values()))) for b in blocks})!=1:raise ValueError('Equal-weight mixtures require equal member draw counts')
                    values=mixture_samples(blocks);targets=npz(Path(selected[0]['output'])/'origins'/origin/'targets.npz')
                    row,_=score(values,targets,scales,pairs,mode,origin,index,eval_seed=eval_seed);rows.append(row)
                    write_json(dest/'origins'/origin/f'{mode}_scores.json',row)
                record['headlines'][mode]=headline(merged(rows),scales,indices) if available and indices is not None else None
            write_json(dest/'report.json',record);result.append(record)
    return result


def evaluate_runs(options):
    if options['samples']<2:raise ValueError('Evaluation ensembles require at least two samples')
    if not 0<=options['shard']<options['shards']:raise ValueError('shard must be smaller than shards')
    runs=resolve_runs(options.get('runs'));out=Path(options.get('out') or default_output('evaluation'))
    manifest_options={k:v for k,v in options.items() if k not in ('shard','shards')}
    record_manifest(out,'evaluate',manifest_options,device=options['device'],seeds=[options['eval_seed']],paths=dict(runs=runs,output=out))
    reports=[]
    for path in runs:
        name=hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]
        reports.append(evaluate_one(path,out/'runs'/name,options))
    mixtures=evaluate_mixtures(reports,options['ensemble_sizes'],out,options['modes'],options['eval_seed']) if all(r.get('complete') for r in reports) else []
    result=dict(complete=all(r.get('complete') for r in reports),runs=reports,mixtures=mixtures)
    with (out/'summary.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (out/'evaluation.json').exists() and read_json(out/'evaluation.json').get('complete') and not result['complete']:
            return read_json(out/'evaluation.json')
        write_json(out/'evaluation.json',result)
    for run in runs:
        if (run/'chain.json').exists():write_json(run/'evaluations.json',dict(latest=str(out.resolve())))
    return result
