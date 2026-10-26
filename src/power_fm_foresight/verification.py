"""The two fixed verification protocols: recorded regression and non-inferiority."""
import math
from pathlib import Path
import json

from omegaconf import OmegaConf

from ._config import read_json, sha256, write_json, load_config
from ._artifacts import differences
from .campaign import command_options, default_output, record_manifest, resolve_runs, run_config


def protocol(path,check):
    value=OmegaConf.to_container(OmegaConf.load(path),resolve=True,throw_on_missing=True)
    if value.get('schema')!='pff-verification.v1' or value.get('check')!=check:raise ValueError('Protocol check/schema mismatch')
    if check=='regression':
        if value.get('absolute_tolerance')!=1e-7 or value.get('relative_tolerance')!=0:raise ValueError('Regression protocol must retain the fixed 1e-7 absolute gate')
    else:
        if value.get('slack')!=.005 or value.get('epochs')!=[1,2] or value.get('restart_after_steps')!=4090:
            raise ValueError('Reproducibility protocol must retain the fixed two-epoch, boundary-restore and +0.005 gate')
        if value['training']!={**value['training'],'case':[30],'seed':[0],'recipe':'v4-off','max_epochs':2}:
            raise ValueError('Reproducibility training configuration changed')
        if len(value['reference_runs'])!=3:raise ValueError('Three recorded fleet seed references are required')
    return value


def finite(value):
    if isinstance(value,dict):return all(finite(v) for v in value.values())
    if isinstance(value,list):return all(finite(v) for v in value)
    return not isinstance(value,(int,float)) or isinstance(value,bool) or math.isfinite(value)


def metrics(path):
    return [json.loads(line) for line in (Path(path)/'metrics.jsonl').read_text().splitlines()]


def check_reproducibility(candidate,p,references):
    candidate=Path(candidate);adapter=candidate/'adapter' if (candidate/'chain.json').exists() else candidate
    cfg=run_config(candidate)
    b=cfg['stage_b']
    configuration_ok=(cfg['seed']==0 and cfg['task']['canonical_case']=='pglib_opf_case30_ieee' and
        b.get('deterministic_algorithms') is False and b['training_samples']==8 and
        b['optimizer']['lr']==1e-4 and b['loss']['power_balance_weight']==1. and b['max_epochs']==2 and
        b['adapter'].get('model_policy','conditioned')=='conditioned')
    epochs={r['epoch']:r for r in read_json(adapter/'epochs.json')};curve=metrics(adapter)
    if set(p['epochs'])-set(epochs):return dict(passed=False,reason='Two complete epochs are required')
    checks=[]
    for epoch in p['epochs']:
        upper=max(next(r['selection_energy_score'] for r in e if r['epoch']==epoch) for e,_ in references)+p['slack']
        actual=epochs[epoch]['selection_energy_score'];checks.append(dict(epoch=epoch,actual=actual,upper=upper,passed=math.isfinite(actual) and actual<=upper))
    all_finite=finite(curve) and finite(list(epochs.values()))
    reference_curves=[{r['steps']:r for r in rows} for _,rows in references]
    loss=[]
    for row in curve:
        step=row['steps']
        if all(step in c for c in reference_curves):
            values=[c[step]['loss'] for c in reference_curves]
            loss.append(dict(step=step,candidate=row['loss'],reference_min=min(values),reference_max=max(values),
                delta_from_max=row['loss']-max(values),relative_to_max=(row['loss']-max(values))/max(abs(max(values)),1e-12)))
    restores=[read_json(x) for x in (adapter/'restores').glob('*.json')]
    boundary=[r for r in restores if r['progress']['steps']==p['restart_after_steps'] and r['progress']['step_in_epoch']==4090 and r['progress']['epoch']==0]
    exact=bool(boundary) and all(r['passed'] and r['manifest_exact'] and all(s['saved_sha256']==s['restored_sha256'] for s in r['state'].values()) for r in boundary)
    intervals=[r['steps'] for r in curve]
    committed_order=intervals==list(range(p['loss_curve']['cadence_steps'],8181,p['loss_curve']['cadence_steps']))
    journal=read_json(candidate/'boundary_resume.json') if (candidate/'boundary_resume.json').exists() else {}
    exact=exact and journal.get('passed',False) and journal.get('journal_prefix_exact',False)
    return dict(passed=configuration_ok and all(c['passed'] for c in checks) and all_finite and exact and committed_order,
        configuration_exact=configuration_ok,
        selection=checks,finite_metrics=all_finite,exact_boundary_restore=exact,journal_order=committed_order,
        loss_curve_diagnostics=loss,stopping_epoch='Not tested',restores=boundary)


def verify(options):
    p=protocol(options['protocol'],options['check']);out=Path(options.get('out') or default_output('verify'))
    record_manifest(out,'verify',p,device=p.get('device',p.get('training',{}).get('device')),seeds=[p.get('eval_seed',0)],paths=dict(protocol=options['protocol'],output=out))
    hashes={str(Path(options['protocol']).resolve()):sha256(options['protocol'])}
    if p['check']=='regression':
        from .assessment import evaluate_runs
        from .campaign import run_config
        cfg=load_config(p['configuration']);cfg['evaluation']=dict(baseline_root=p['saved_evaluation'],origins=p['origins'])
        args=command_options(dict(command='evaluate',runs=options.get('runs') or [p['run']],device=p['device'],out=str(out/'evaluation'),
            modes=p['modes'],samples=p['samples'],eval_seed=p['eval_seed'],allow_budget_capped=True))
        args['configuration']=cfg
        runs=resolve_runs(args['runs'])
        for run in runs:
            checkpoint=Path(run_config(run,cfg)['paths']['training_run'])/'best.pt';hashes[str(checkpoint)]=sha256(checkpoint)
        for origin in p['origins']:
            target=Path(p['saved_evaluation'])/'origins'/origin/'targets.npz';hashes[str(target)]=sha256(target)
            for mode in p['modes']:
                path=Path(p['saved_evaluation'])/'origins'/origin/f'{mode}_scores.json';hashes[str(path)]=sha256(path)
        write_json(out/'inputs.json',hashes)
        result=evaluate_runs(args);checks=[]
        for run in result['runs']:
            for origin in p['origins']:
                for mode in p['modes']:
                    actual=read_json(Path(run['output'])/'origins'/origin/f'{mode}_scores.json')
                    old=read_json(Path(p['saved_evaluation'])/'origins'/origin/f'{mode}_scores.json')
                    checks.append(dict(origin=origin,mode=mode,**differences(actual,old,atol=1e-7,rtol=0)))
        gate=dict(passed=result['complete'] and all(c['passed'] for c in checks),checks=checks)
    else:
        references=[]
        for path in map(Path,p['reference_runs']):
            for name in ('epochs.json','metrics.jsonl','manifest.json'):
                hashes[str(path/name)]=sha256(path/name)
            references.append((read_json(path/'epochs.json'),metrics(path)))
        manifests=[read_json(Path(path)/'manifest.json') for path in p['reference_runs']]
        if sorted(m['seed'] for m in manifests)!=[0,1,2]:raise ValueError('Reference seeds must be 0, 1 and 2')
        for key in ('dataset_manifest_sha256','coordinate_scales_sha256','base_checkpoint','global_K','origins_per_step','precision_policy'):
            if any(m[key]!=manifests[0][key] for m in manifests):raise ValueError(f'Incompatible fleet references: {key}')
        write_json(out/'inputs.json',hashes)
        if options.get('runs'):
            runs=resolve_runs(options['runs'])
        else:
            from .cli import main
            campaign=out/'campaign'
            argv=['train','--config',p['configuration'],'--case','30','--dataset',p['dataset'],'--seed','0','--recipe','v4-off','--max-epochs','2',
                '--device',p['training']['device'],'--device-index',str(p['training']['device_index']),'--out',str(campaign)]
            if not (campaign/'campaign.json').exists():
                if main(argv,stop_after_steps=p['restart_after_steps']):raise RuntimeError('Training command failed')
            runs=resolve_runs([campaign])
            if not all((run/'boundary_checkpoint.json').exists() for run in runs):
                if main(['train','--resume',str(campaign)],stop_after_steps=p['restart_after_steps']):raise RuntimeError('Boundary continuation failed')
            if not all((run/'boundary_checkpoint.json').exists() for run in runs):
                return dict(passed=False,incomplete=True,reason='Allocation ended before the controlled boundary; resume with the same protocol')
            if main(['train','--resume',str(campaign)]):raise RuntimeError('Resume command failed')
        for run in runs:
            adapter=run/'adapter' if (run/'chain.json').exists() else run
            manifest=read_json(adapter/'manifest.json')
            for key in ('dataset_manifest_sha256','coordinate_scales_sha256','base_checkpoint','global_K','origins_per_step','precision_policy'):
                if manifest[key]!=manifests[0][key]:raise ValueError(f'Candidate differs from fleet reference: {key}')
        checks=[check_reproducibility(run,p,references) for run in runs]
        gate=dict(passed=all(c['passed'] for c in checks),runs=[str(r) for r in runs],checks=checks)
    if not all(sha256(path)==h for path,h in hashes.items()):raise ValueError('Read-only verification input changed')
    gate.update(protocol_sha256=sha256(options['protocol']),input_sha256=hashes)
    write_json(out/'gate.json',gate)
    return gate
