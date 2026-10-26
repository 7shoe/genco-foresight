"""Prepare or audit temporal datasets using the existing law, solver and checks."""
from copy import deepcopy
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import subprocess
import shutil
import time

import numpy as np
import pandas as pd

from ._config import PACKAGE_ROOT, RESOURCE_ROOT, read_json, sha256, write_json
from .campaign import case_name, record_manifest
from .grids import grid_config, network_path
from .data import load_network, load_temporal_manifest, demands


def existing_inputs(cfg):
    root=Path(cfg['paths']['temporal_root']);manifest=read_json(root/'manifest.json')
    net=load_network(config=cfg);load_temporal_manifest(root,net)
    if manifest['seed']!=cfg['seed']:raise ValueError('Reused data seed differs from the preparation configuration')
    if manifest['generation']!=cfg['generation'] or manifest.get('split_episodes',cfg['task']['split_episodes'])!=cfg['task']['split_episodes']:
        raise ValueError('Dataset law or split differs from the requested recipe')
    profile=Path(cfg['paths']['assets_root'])/'profiles/eia_erco_demand_2025.csv'
    if sha256(profile)!=manifest['profile_sha256']:raise ValueError('EIA profile hash changed')
    frame=pd.read_csv(profile);values=frame.demand_mw.to_numpy()
    train=np.concatenate([np.arange(i*504,(i+1)*504) for i in manifest['source_block_pools']['train']])
    low,high=float(values[train].min()),float(values[train].max())
    if (low,high)!=(manifest['train_profile_min'],manifest['train_profile_max']):raise ValueError('Train-only profile scaling changed')
    g=cfg['generation'];scaled=g['scale_low']+(values-low)*(g['scale_high']-g['scale_low'])/(high-low)
    return root,manifest,net,frame,scaled


def audit_labels(cfg):
    from .solver import check_solution
    root,manifest,net,frame,scaled=existing_inputs(cfg)
    expected={r['file']:r['sha256'] for r in manifest['shards']}
    pools=manifest['source_block_pools']
    for split,ids in pools.items():
        if set(ids)&{i for s,values in pools.items() if s!=split for i in values}:raise ValueError('Cross-partition source leakage')
    inputs={str(root/'manifest.json'):sha256(root/'manifest.json')};totals=dict(hours=0,valid=0,unresolved=0);maxima={}
    for episode in manifest['episodes']:
        p=root/f"episodes/{episode['episode_id']}.json";h=sha256(p)
        if h!=expected[str(p.relative_to(root))]:raise ValueError(f'Shard checksum changed: {p}')
        inputs[str(p)]=h;shard=read_json(p);rows=shard['records']
        if len(rows)!=504 or shard['benchmark_id']!=manifest['benchmark_id']:raise ValueError('Incomplete or incompatible episode')
        lo=episode['source_start_hour'];pd,qd=demands(scaled[lo:lo+504],net.bus[:,2],net.bus[:,3],manifest['generation'],episode['seed'])
        for hour,r in enumerate(rows):
            if r['hour']!=hour or r['time_utc']!=frame.time_utc.iloc[lo+hour]:raise ValueError('Hour identity changed')
            np.testing.assert_array_equal(r['pd_mw'],pd[hour]);np.testing.assert_array_equal(r['qd_mvar'],qd[hour])
            checked=check_solution(net,r['pd_mw'],r['qd_mvar'],r['solver'])
            if checked['valid']!=r['state_available']:raise ValueError('Stored validity disagrees with independent physics')
            if r['target_state']!=(r['solver']['state'] if checked['valid'] else None):raise ValueError('Target availability changed')
            totals['hours']+=1;totals['valid']+=int(checked['valid']);totals['unresolved']+=int(not checked['valid'])
            for k,v in checked.get('errors',{}).items():maxima[k]=max(maxima.get(k,0),v)
    result=dict(passed=True,**totals,max_independent_errors=maxima,source_pools_disjoint=True,
                deterministic_demands_verified=True,shard_checksums_verified=True,input_sha256=inputs)
    write_json(Path(cfg['paths']['stage_a_run'])/'stage_a.json',result)
    return result


def isolate_assets(source,dest,case):
    source,dest=Path(source),Path(dest);dest.mkdir(parents=True,exist_ok=True)
    for name in ('profiles','cli_deps',f'official_{case}'):
        if (source/name).exists() and not (dest/name).exists():(dest/name).symlink_to((source/name).resolve(),target_is_directory=True)
    old=source/'genco-opfdata';new=dest/'genco-opfdata';new.mkdir(exist_ok=True)
    for name in (f'{case}/seed42','splits'):
        target=new/name;target.parent.mkdir(parents=True,exist_ok=True)
        if (old/name).exists() and not target.exists():target.symlink_to((old/name).resolve(),target_is_directory=True)
    if (old/'manifest.json').exists() and not (new/'manifest.json').exists():shutil.copyfile(old/'manifest.json',new/'manifest.json')
    if (source/'networks').exists() and not (dest/'networks').exists():(dest/'networks').symlink_to((source/'networks').resolve(),target_is_directory=True)


def reuse_scales(source, dataset, destination):
    """Keep a recorded coordinate order after verifying its training-shard provenance."""
    value=read_json(source);manifest=read_json(Path(dataset)/'manifest.json')
    training={f"episodes/{e['episode_id']}.json" for e in manifest['episodes'] if e['split']=='train'}
    hashes={r['file']:r['sha256'] for r in manifest['shards'] if r['file'] in training}
    if (value['benchmark_id']!=manifest['benchmark_id'] or value['fit_partition']!='train' or
            value['training_shards_sha256']!=hashes or len(hashes)!=len(training)):
        raise ValueError('Recorded coordinate scales do not belong to these training shards')
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists() and sha256(destination)!=sha256(source):raise ValueError('Recorded coordinate scales changed')
    shutil.copyfile(source,destination)
    return dict(source=str(source),sha256=sha256(source),policy='Reuse exact recorded train-derived scales and coordinate order')


def subset_parity(cfg, out, count, conversion_source):
    import torch
    from ._data import parity, subset
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    protocol=subset.record(cfg,out/'protocol.json',count=count)
    if (out/'gate.json').exists():
        gate=read_json(out/'gate.json')
        if (gate['protocol_sha256']!=sha256(out/'protocol.json') or not gate['passed'] or
                gate['report_sha256']!=sha256(out/'stage_a/parity/parity_report.json')):
            raise ValueError('Stored subset parity failed or changed')
        return gate
    started=time.monotonic();selected=deepcopy(cfg);selected['paths']['runs_root']=str(out)
    conversion_source=Path(conversion_source)
    if not (conversion_source/'conversion_check.json').exists():
        conversion=deepcopy(cfg);conversion['paths']['runs_root']=str(out/'conversion')
        conversion_source=parity.prepare(conversion,download=True,spot_count=32,limit=32,batch_size=1)
    subset.prepare(selected,protocol,conversion_source)
    parity.evaluate(selected)
    report=read_json(out/'stage_a/parity/parity_report.json')
    gate=dict(passed=bool(report['passed'] and report['provenance']['bit_identical'] and time.monotonic()-started<=1200),
              label=f'subset parity ({count})',protocol_sha256=sha256(out/'protocol.json'),
              report_sha256=sha256(out/'stage_a/parity/parity_report.json'),seconds=time.monotonic()-started,
              count=count,bit_identical=report['provenance']['bit_identical'])
    write_json(out/'gate.json',gate)
    if not gate['passed']:raise RuntimeError('Subset parity failed')
    return gate


def prepare_datasets(options):
    if not options.get('case') or not options.get('dataset'):raise ValueError('prepare requires --case and --dataset')
    output=Path(options['dataset']).resolve()
    if options.get('reuse'):
        reuse=Path(options['reuse']).resolve()
        if output==reuse or reuse in output.parents:raise ValueError('Preparation output must be outside the reused dataset')
    source_cfg=options['configuration']
    invocation=output/'invocations'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') if (output/'command.manifest.json').exists() else output
    record_manifest(invocation,'prepare',options,device='cpu',seeds=[source_cfg['seed']],paths=dict(output=output,reuse=options.get('reuse')))
    for number in options['case']:
        case=case_name(number);out=output/case;out.mkdir(parents=True,exist_ok=True)
        cfg=grid_config(case,source_cfg)
        original_assets=Path(source_cfg['paths']['assets_root'])
        isolate_assets(original_assets,out/'assets',case)
        cfg['paths']['assets_root']=str(out/'assets')
        candidate=Path(source_cfg['paths'].get('network_file',original_assets/'networks'/f'pglib_opf_{case}.m'))
        if candidate.stem!=f'pglib_opf_{case}':candidate=original_assets/'networks'/f'pglib_opf_{case}.m'
        if not candidate.exists() and case=='case30_ieee':candidate=RESOURCE_ROOT/'networks/pglib_opf_case30_ieee.m'
        cfg['paths']['network_file']=str(candidate)
        if number in (500,2000) and not cfg['solver'].get('reference_network'):
            settings=cfg['solver']
            env=dict(os.environ,LD_LIBRARY_PATH='',JULIA_DEPOT_PATH=settings['julia_depot'],JULIA_PKG_OFFLINE='true',JULIA_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            output_text=subprocess.check_output([settings['julia_executable'],'--startup-file=no',f'--project={RESOURCE_ROOT / "julia"}',str(RESOURCE_ROOT/'julia/reference.jl'),str(candidate)],env=env,text=True,timeout=120)
            metadata=json.loads(next(line.removeprefix('REFERENCE_JSON ') for line in output_text.splitlines() if line.startswith('REFERENCE_JSON ')))
            metadata['case_sha256']=sha256(candidate);cfg['solver']['reference_network']=metadata
            write_json(out/'reference_network.json',metadata)
        if source_cfg['task']['canonical_case'].removeprefix('pglib_opf_')==case:
            for key in ('n1_cache','t1_cache','n1_splits'):cfg['paths'][key]=source_cfg['paths'][key]
        cfg['paths'].update(runs_root=str(out),stage_a_run=str(out/'audit'),evidence_root=str(out/'audit'),training_run=str(out/'shared'))
        cfg['stage_b']['coordinate_scales']=str(out/'audit/stage_a_followup/train_coordinate_scales.json')
        data=out/'data'
        if options.get('reuse'):
            reuse=Path(options['reuse']).resolve()
            if not (reuse/'manifest.json').exists():
                reuse=reuse/case/'data'
            if not (reuse/'manifest.json').exists():raise FileNotFoundError(f'No temporal dataset manifest at {reuse}')
            if data.exists() and data.resolve()!=reuse.resolve():raise ValueError('Reuse source differs from existing dataset')
            if not data.exists():data.symlink_to(reuse,target_is_directory=True)
        cfg['paths']['temporal_root']=str(data)
        scale_provenance=None
        recorded=Path(source_cfg['stage_b']['coordinate_scales'])
        if options.get('reuse') and recorded.exists() and source_cfg['task']['canonical_case']==cfg['task']['canonical_case']:
            cfg['stage_b']['coordinate_scales']=str(out/'audit/reused_coordinate_scales.json')
            scale_provenance=reuse_scales(recorded,data,cfg['stage_b']['coordinate_scales'])
        record_manifest(out,'prepare',cfg,device='cpu',seeds=[cfg['seed']],paths=dict(output=out,data=data))
        if not options.get('reuse') and not options['audit_only']:
            from ._data.labeling import generate
            generate(cfg,options['workers'])
        if not (data/'manifest.json').exists():raise FileNotFoundError('audit-only requires existing labels')
        audited=audit_labels(cfg)
        from ._data.events import run as event_audit
        event_audit(cfg)
        from ._data.network import load_network
        load_network(config=cfg)
        proof=subset_parity(cfg,out/'parity',options['parity_samples'],Path(source_cfg['paths']['stage_a_run'])/'parity')
        readiness=dict(stage_a_audit_passed=True,training_allowed=True,dataset_id=cfg['dataset']['id'],
            manifest_sha256=sha256(data/'manifest.json'),stage_a_sha256=sha256(out/'audit/stage_a.json'),prevalence_is_training_blocker=False)
        write_json(out/'audit/readiness.json',readiness)
        cache=Path(source_cfg['paths']['training_run'])/'cache/manifest.json'
        reusable_cache=None
        if options.get('reuse') and cache.exists():
            cached=read_json(cache)
            if cached['benchmark_id']==read_json(data/'manifest.json')['benchmark_id']:
                reusable_cache=str(cache)
        paths=audited['input_sha256'];paths[str(out/'audit/readiness.json')]=sha256(out/'audit/readiness.json')
        paths[cfg['stage_b']['coordinate_scales']]=sha256(cfg['stage_b']['coordinate_scales'])
        if reusable_cache:paths[reusable_cache]=sha256(reusable_cache)
        if scale_provenance:paths[scale_provenance['source']]=scale_provenance['sha256']
        card=dict(schema='pff-dataset.v1',passed=True,case=case,recipe='v4',config=cfg,
            parity=proof,label=proof['label'],input_sha256=paths,training_cache=reusable_cache,
            coordinate_scale_provenance=scale_provenance,
            audit=str(out/'audit/stage_a.json'),headlines=str(out/'audit/stage_a_followup/headline_components.json'))
        write_json(out/'dataset.json',card)
        (out/'DATASET_CARD.md').write_text(f'# {case} v4\n\nStage A audit passed; {audited["hours"]} hours, {audited["unresolved"]} unresolved labels remain masked.\n\n{proof["label"]}: bit-identical CPU predictions; IDs and hashes in parity/protocol.json.\n\nHeadline components use risk-calibration support only.\n')
        print(dict(prepared=case,card=str(out/'dataset.json')),flush=True)
