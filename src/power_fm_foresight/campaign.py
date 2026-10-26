"""Resolve command configuration and run independent local training chains."""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import shutil
import time

from omegaconf import OmegaConf

from . import __version__
from ._config import PACKAGE_ROOT, RESOURCE_ROOT, read_json, sha256, write_json
from .grids import registry, grid_config

DEFAULTS = {
    'prepare': dict(recipe='v4', workers=1, parity_samples=200, device='cpu', audit_only=False),
    'train': dict(recipe='v4-off', variant='conditioned', balance_weight=[1.], samples=8,
                  lr=1e-4, patience=3, chain_budget='6h', device='cpu', device_index=0,
                  parallel=1, dry_run=False, label='', max_epochs=1000),
    'evaluate': dict(checkpoint='best', partition='test', modes=['C','P','P0','S','A','D'],
                     samples=64, eval_seed=42, ensemble_sizes=[], shards=1, shard=0,
                     allow_budget_capped=False, device='cpu'),
    'report': dict(kind='summary', format=['md']),
    'verify': {},
}


def plain(value):
    if isinstance(value, Path):return str(value.resolve())
    if isinstance(value, dict):return {k: plain(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)):return [plain(v) for v in value]
    return value


def source_identity():
    files = {str(p.relative_to(PACKAGE_ROOT)): sha256(p) for p in sorted(PACKAGE_ROOT.rglob('*'))
             if p.is_file() and p.suffix in ('.py','.yaml','.json','.jl','.toml') and '__pycache__' not in p.parts}
    return dict(sha256=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(), files=files)


def duration(value):
    if isinstance(value, (int,float)):
        result=float(value)
    else:
        match=re.fullmatch(r'([0-9]+(?:\.[0-9]+)?)(s|m|h)?',str(value))
        if not match:raise ValueError('chain-budget must be seconds or a duration such as 30m or 6h')
        result=float(match[1])*{'s':1,'m':60,'h':3600,None:1}[match[2]]
    if not math.isfinite(result) or result<=0:raise ValueError('chain-budget must be positive')
    return result


def command_options(args):
    command=args['command']; supplied={}
    if args.get('config'):
        supplied=OmegaConf.to_container(OmegaConf.load(args['config']),resolve=False)
    recipe_name=args.get('recipe',supplied.get('cli',{}).get(command,{}).get('recipe',DEFAULTS[command].get('recipe','v4')))
    recipe=OmegaConf.load(RESOURCE_ROOT/'v4.yaml')
    # Non-solver commands do not require a Julia installation at config-resolution time.
    recipe.solver.julia_executable=os.environ.get('PFF_JULIA','')
    recipe.solver.julia_depot=os.environ.get('PFF_JULIA_DEPOT','')
    recipe.base=os.environ.get('PFF_ROOT',str(Path.cwd()))
    cfg=OmegaConf.to_container(OmegaConf.merge(recipe,supplied),resolve=True,throw_on_missing=True)
    cfg.pop('base',None)
    options=deepcopy(DEFAULTS[command])
    if command=='train':
        b=cfg['stage_b']
        options.update(device=cfg.get('device','cpu'),seed=[cfg['seed']],samples=b['training_samples'],
            balance_weight=[b['loss']['power_balance_weight']],lr=b['optimizer']['lr'],
            max_epochs=b['max_epochs'],patience=b['early_stopping']['patience'],chain_budget=b['training_budget_seconds'],
            variant=b['adapter'].get('model_policy','conditioned'))
        if recipe_name in ('v4-on','v4-off') and not ('stage_b' in supplied and 'deterministic_algorithms' in supplied['stage_b']):
            cfg['stage_b']['deterministic_algorithms']=recipe_name=='v4-on'
    options.update(cfg.get('cli',{}).get(command,{}))
    options.update({k:v for k,v in args.items() if k!='config'})
    options['recipe']=recipe_name
    if 'recipe' in args and command=='train':cfg['stage_b']['deterministic_algorithms']=recipe_name=='v4-on'
    if command=='train' and 'seed' not in args and 'seed' not in supplied and 'seed' not in supplied.get('cli',{}).get('train',{}):
        options.pop('seed',None)
    options['configuration']=cfg
    if args.get('config'):options['config']=str(Path(args['config']).resolve())
    return plain(options)


def record_manifest(out, command, configuration, *, device, seeds, paths, identity=None):
    out=Path(out)
    if not (out/'command.manifest.json').exists() and any((out/name).exists() for name in ('manifest.json','best.pt','latest.pt','episodes')):
        raise ValueError(f'Output is an existing data or training artifact: {out}; choose a new directory')
    out.mkdir(parents=True,exist_ok=True)
    value=dict(schema='pff-command.v1',command=command,configuration=plain(configuration),
               package_version=__version__,source=identity or source_identity(),device=device,
               seeds=plain(seeds),paths=plain(paths))
    path=out/'command.manifest.json'
    if path.exists() and read_json(path)!=value:
        raise ValueError(f'Output manifest is incompatible: {path}; choose a new output directory')
    write_json(path,value)
    return value


def case_name(case):
    try:return registry()['case_ids'][int(case)]
    except (KeyError,ValueError,TypeError):raise ValueError(f'Unsupported case: {case}') from None


def default_output(kind):
    if not os.environ.get('PFF_ROOT'):raise ValueError('--out or PFF_ROOT is required')
    return Path(os.environ['PFF_ROOT'])/'runs/cli'/f'{kind}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}'


def dataset_card(root, case):
    root=Path(root)
    path=root/'dataset.json' if (root/'dataset.json').exists() else root/case/'dataset.json'
    value=read_json(path)
    if value['case']!=case:raise ValueError('Prepared dataset case mismatch')
    if not value['passed']:raise ValueError('Dataset preparation has not passed')
    for p,h in value['input_sha256'].items():
        if sha256(p)!=h:raise ValueError(f'Prepared dataset input changed: {p}')
    return value


def expand_plan(options):
    cases=options.get('case');seeds=options.get('seed')
    if not cases or not seeds or not options.get('dataset'):
        raise ValueError('train requires --case, --dataset and --seed, or --resume')
    if len(set(cases))!=len(cases) or len(set(seeds))!=len(seeds):raise ValueError('Duplicate cases or seeds')
    weights=options['balance_weight']
    if len(set(weights))!=len(weights) or any(not math.isfinite(v) or v<0 for v in weights):raise ValueError('balance-weight values must be unique, finite and nonnegative')
    if options['samples']<2 or 64%options['samples']:raise ValueError('Training samples must be at least two and divide the fixed 64-sample selection ensemble')
    if any(type(s) is not int or s<0 for s in seeds):raise ValueError('Seeds must be nonnegative integers')
    if options['max_epochs']<1 or options['patience']<1:raise ValueError('max-epochs and patience must be positive')
    if not math.isfinite(options['lr']) or options['lr']<=0:raise ValueError('lr must be finite and positive')
    if options['parallel']>1 and options['device']=='cpu':raise ValueError('--parallel uses accelerator tiles; CPU chains run sequentially')
    duration(options['chain_budget'])
    return [dict(case=case_name(c),seed=s,balance_weight=w,
                 id=f'{case_name(c)}/seed{s}/{options["variant"]}_lambda{w:g}_K{options["samples"]}')
            for c,s,w in itertools.product(cases,seeds,weights)]


def prepare_campaign(options):
    plan=expand_plan(options);out=Path(options.get('out') or default_output('train')).resolve()
    if (out/'campaign.json').exists():raise ValueError('Campaign exists; use --resume')
    options=deepcopy(options);options['out']=str(out)
    identity=source_identity()
    record_manifest(out,'train',options,device=options['device'],seeds=options['seed'],paths=dict(output=out,dataset=options['dataset']),identity=identity)
    campaign=dict(schema='pff-campaign.v1',root=str(out),options=options,plan=plan,source=identity)
    write_json(out/'campaign.json',campaign)
    return campaign


def load_campaign(path):
    p=Path(path).resolve()
    if (p/'chain.json').exists():
        chain=read_json(p/'chain.json');campaign=read_json(Path(chain['campaign'])/'campaign.json')
        campaign=deepcopy(campaign);campaign['plan']=[x for x in campaign['plan'] if x['id']==chain['id']]
        return campaign
    if not (p/'campaign.json').exists():raise ValueError('No pff campaign or chain manifest; historical runs require their compatible pinned source')
    return read_json(p/'campaign.json')


def resume_campaign(options, explicit):
    campaign=load_campaign(explicit['resume']);frozen=campaign['options']
    if campaign['source']!=source_identity():raise ValueError('Resume rejected: package source changed')
    keys=set(explicit)-{'command','resume','dry_run','parallel','config'}
    if explicit.get('config'):
        cfg=options['configuration']
        if cfg!=frozen['configuration']:raise ValueError('Resume rejected: configuration changed')
    for key in keys:
        incoming=plain(explicit[key]);old=frozen.get(key)
        if key=='chain_budget':incoming,old=duration(incoming),duration(old)
        if incoming!=old:raise ValueError(f'Resume rejected: incompatible {key} override')
    requested=int(explicit.get('parallel',frozen['parallel']))
    if requested!=frozen['parallel']:raise ValueError('Resume rejected: tile mapping/parallel changed')
    campaign['parallel']=requested
    invocation=dict(command='train --resume',package_version=__version__,source=campaign['source'],
        configuration=frozen,device=frozen['device'],seeds=frozen['seed'],paths=dict(resume=str(explicit['resume'])))
    write_json(Path(campaign['root'])/'invocations'/f'{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.json',invocation)
    return campaign


def select_device(device, index=0):
    import torch
    parsed=torch.device(device)
    if parsed.index is not None:index=parsed.index
    if parsed.type=='cpu':return 'cpu'
    if parsed.type not in ('xpu','cuda'):raise ValueError('device must be cpu, xpu or cuda')
    backend=getattr(torch,parsed.type)
    if not backend.is_available() or not 0<=index<backend.device_count():raise ValueError(f'{parsed.type} device index {index} is unavailable')
    if parsed.type=='xpu':
        if os.environ.get('ZE_FLAT_DEVICE_HIERARCHY')!='FLAT':raise ValueError('XPU launchers must set ZE_FLAT_DEVICE_HIERARCHY=FLAT')
        if os.environ.get('ZE_AFFINITY_MASK') and backend.device_count()==1 and index!=0:
            raise ValueError('A launcher-isolated tile has local index zero')
    backend.set_device(index)
    return parsed.type


def allocation_deadline(configuration):
    if 'PBS_WALLTIME' not in os.environ:return None
    from .training import walltime_seconds
    start=float(os.environ.get('GENCO_JOB_START_EPOCH',time.time()))
    return start+min(configuration['stage_b']['job_training_guard_seconds'],walltime_seconds(os.environ['PBS_WALLTIME'])-300)


def execute_chain(campaign, chain, device_index, stop_after_steps, deadline):
    previous={k:os.environ.get(k) for k in ('GENCO_JOB_START_EPOCH','PBS_WALLTIME')}
    started=time.time()
    if deadline is not None and deadline-started<60:return None
    try:
        # Charge each chain for its own active worker time, keeping the allocation deadline.
        os.environ['GENCO_JOB_START_EPOCH']=str(started)
        if deadline is not None:os.environ['PBS_WALLTIME']=str(max(300,int(deadline-started)+300))
        return train_chain(campaign,chain,device_index,stop_after_steps)
    finally:
        for key,value in previous.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value


def train_chain(campaign, chain, device_index, stop_after_steps=None):
    from .training import run
    from .data import build_cache
    options=campaign['options'];root=Path(campaign['root'])/chain['id'];root.mkdir(parents=True,exist_ok=True)
    card=dataset_card(options['dataset'],chain['case'])
    cfg=grid_config(chain['case'],options['configuration'])
    cfg['paths'].update(card['config']['paths'])
    cfg['checkpoint']=deepcopy(card['config']['checkpoint'])
    cfg['task']=deepcopy(card['config']['task'])
    cfg['dataset']=deepcopy(card['config']['dataset'])
    cfg['solver']=deepcopy(card['config']['solver'])
    if cfg['generation']!=card['config']['generation']:raise ValueError('Training recipe differs from the prepared data law')
    cfg['seed']=chain['seed'];cfg['device']=options['device']
    cfg['paths']['runs_root']=str(root);cfg['paths']['training_run']=str(root/'adapter')
    cfg['stage_b']['coordinate_scales']=card['config']['stage_b']['coordinate_scales']
    b=cfg['stage_b'];b['training_samples']=options['samples'];b['future_graph_batch_size']=24*options['samples']
    b['optimizer']['lr']=options['lr'];b['loss']['power_balance_weight']=chain['balance_weight']
    b['adapter']['model_policy']=options['variant'];b['max_epochs']=options['max_epochs']
    b['early_stopping']['patience']=options['patience'];b['training_budget_seconds']=duration(options['chain_budget'])
    b.setdefault('metrics_interval_steps',100)
    cfg['v2']=dict(run_identity=str(root),label=options['label'])
    saved=dict(id=chain['id'],campaign=campaign['root'],config=cfg,label=options['label'],
               pilot=options['samples']!=8 or 'pilot' in options['label'].lower(),device_index=device_index)
    path=root/'chain.json'
    if path.exists() and read_json(path)!=saved:raise ValueError('Chain configuration or device mapping changed')
    write_json(path,saved);write_json(root/'config.json',cfg)
    record_manifest(root,'train',cfg,device=dict(type=options['device'],index=device_index),seeds=[chain['seed']],
                    paths=dict(dataset=cfg['paths']['temporal_root'],output=root),identity=campaign['source'])
    device=select_device(options['device'],device_index)
    cache=root/'adapter/cache/manifest.json'
    if not cache.exists():
        cache.parent.mkdir(parents=True,exist_ok=True)
        if card.get('training_cache'):
            import torch
            old=Path(card['training_cache'])
            for item in read_json(old)['files'].values():
                cached_state=torch.load(item['path'],mmap=True,map_location='cpu',weights_only=True)
                if cached_state['key']['scales_sha256']!=sha256(cfg['stage_b']['coordinate_scales']):raise ValueError('Reused training cache coordinate scales differ')
            shutil.copyfile(old,cache)
        else:
            import torch
            torch.use_deterministic_algorithms(True)
            prepared=build_cache(cfg,device,deadline_epoch=allocation_deadline(cfg))
            if prepared.get('incomplete'):
                progress=dict(steps=0,epoch=0,finished=False,stop_reason='cache_allocation_guard',best_epoch=None)
                write_json(root/'chain.status.json',progress)
                return progress
    boundary=root/'boundary_checkpoint.json'
    receipt=read_json(boundary) if boundary.exists() else None
    run(cfg,device,stop_after_steps=stop_after_steps)
    import torch
    progress=torch.load(root/'adapter/latest.pt',map_location='cpu',weights_only=True)['progress']
    journal=root/'adapter/metrics.jsonl'
    if stop_after_steps is not None and progress['steps']==stop_after_steps and not receipt:
        payload=journal.read_bytes()
        write_json(boundary,dict(steps=progress['steps'],epoch=progress['epoch'],step_in_epoch=progress['step_in_epoch'],
            checkpoint_sha256=sha256(root/'adapter/latest.pt'),journal_bytes=len(payload),journal_sha256=hashlib.sha256(payload).hexdigest()))
    if receipt is not None and progress['steps']>receipt['steps']:
        payload=journal.read_bytes()[:receipt['journal_bytes']]
        exact=hashlib.sha256(payload).hexdigest()==receipt['journal_sha256']
        write_json(root/'boundary_resume.json',dict(passed=exact,receipt=receipt,journal_prefix_exact=exact))
        if not exact:raise ValueError('Committed metrics changed across the boundary restart')
    write_json(root/'chain.status.json',dict(steps=progress['steps'],epochs=progress['epoch'],
        finished=progress['finished'],reason=progress['stop_reason'],best_epoch=progress['best_epoch'],
        label=options['label'],pilot=saved['pilot']))
    return progress


def run_campaign(campaign, parallel=1, stop_after_steps=None):
    options=campaign['options'];parallel=int(campaign.get('parallel',parallel))
    if parallel<1:raise ValueError('parallel must be positive')
    if parallel>1:
        import torch
        device=options['device']
        if device=='cpu':raise ValueError('Parallel chains require accelerator tiles')
        if options['device_index']+parallel>getattr(torch,device).device_count():raise ValueError('Not enough visible tiles for --parallel')
        if device=='xpu' and os.environ.get('ZE_FLAT_DEVICE_HIERARCHY')!='FLAT':raise ValueError('Set ZE_FLAT_DEVICE_HIERARCHY=FLAT')
    results=[];deadline=allocation_deadline(options['configuration'])
    def assigned_tile(chain, offset):
        saved=Path(campaign['root'])/chain['id']/'chain.json'
        return read_json(saved)['device_index'] if saved.exists() else options['device_index']+offset
    # Each wave owns disjoint tiles; workers never share a chain or optimizer.
    for lo in range(0,len(campaign['plan']),parallel):
        if deadline is not None and deadline-time.time()<60:break
        wave=campaign['plan'][lo:lo+parallel]
        if parallel==1:
            results.append(execute_chain(campaign,wave[0],assigned_tile(wave[0],0),stop_after_steps,deadline))
        else:
            with ProcessPoolExecutor(max_workers=parallel,mp_context=multiprocessing.get_context('spawn')) as pool:
                futures=[pool.submit(execute_chain,campaign,c,assigned_tile(c,i),stop_after_steps,deadline) for i,c in enumerate(wave)]
                results.extend(f.result() for f in futures)
    return results


def resolve_runs(paths):
    result=[]
    for p in map(Path,paths or []):
        if (p/'campaign.json').exists():
            c=read_json(p/'campaign.json');result.extend(Path(c['root'])/row['id'] for row in c['plan'])
        else:result.append(p.resolve())
    if not result:raise ValueError('--runs must identify at least one run or campaign')
    if len(set(result))!=len(result):raise ValueError('Duplicate runs')
    return result


def run_config(path, supplied=None):
    path=Path(path)
    if (path/'config.json').exists():return read_json(path/'config.json')
    manifest=read_json(path/'manifest.json')
    if manifest.get('configuration'):return manifest['configuration']
    if supplied is not None:
        cfg=deepcopy(supplied);cfg['paths']['training_run']=str(path)
        return cfg
    raise ValueError(f'No recorded configuration at {path}; supply --config')
