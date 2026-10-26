"""CLI grammar, frozen configuration and read-only reporting contracts."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from power_fm_foresight.cli import parser
from power_fm_foresight.campaign import (command_options, expand_plan, prepare_campaign,
    record_manifest, resume_campaign, duration)
from power_fm_foresight._config import read_json, write_json


def options(tmp_path, *extra):
    return command_options(vars(parser().parse_args(['train','--case','14','30','57','118','--seed','0','1','2',
        '--dataset',str(tmp_path/'data'),'--out',str(tmp_path/'campaign'),*extra])))


def test_exact_verbs_and_flags():
    p=parser()
    action=next(a for a in p._actions if hasattr(a,'choices') and isinstance(a.choices,dict))
    assert set(action.choices)=={'prepare','train','evaluate','report','verify'}
    assert parser().parse_args(['prepare','--case','30','--dataset','d']).case==[30]
    assert parser().parse_args(['train','--resume','r']).resume==Path('r')
    with pytest.raises(SystemExit):p.parse_args(['train','--pbs'])
    with pytest.raises(SystemExit):p.parse_args(['train','--case','200'])
    with pytest.raises(SystemExit):p.parse_args(['train','--parallel','0'])
    with pytest.raises(SystemExit):p.parse_args(['evaluate','--modes','F'])


def test_plan_cartesian_product(tmp_path):
    opts=options(tmp_path)
    plan=expand_plan(opts)
    assert len(plan)==12 and len({c['id'] for c in plan})==12
    assert [c['seed'] for c in plan[:3]]==[0,1,2]
    assert len(expand_plan(options(tmp_path,'--balance-weight','0.1','1','10')))==36
    assert len(expand_plan(options(tmp_path,'--variant','a0')))==12
    with pytest.raises(ValueError):expand_plan(options(tmp_path,'--samples','1'))
    with pytest.raises(ValueError):expand_plan(options(tmp_path,'--lr','nan'))


def test_flag_yaml_recipe_precedence(tmp_path):
    config=tmp_path/'config.yaml'
    config.write_text(yaml.safe_dump(dict(seed=7,stage_b=dict(optimizer=dict(lr=.002),training_samples=4,deterministic_algorithms=True),
        cli=dict(train=dict(balance_weight=[.1,10.],max_epochs=6)))))
    opts=options(tmp_path,'--config',str(config))
    assert opts['lr']==.002 and opts['samples']==4 and opts['max_epochs']==6
    assert opts['configuration']['stage_b']['deterministic_algorithms'] is True
    assert opts['balance_weight']==[.1,10.]
    opts=options(tmp_path,'--config',str(config),'--recipe','v4-off','--lr','.0003','--samples','8','--max-epochs','2')
    assert opts['lr']==.0003 and opts['samples']==8 and opts['max_epochs']==2
    assert opts['configuration']['stage_b']['deterministic_algorithms'] is False
    assert opts['seed']==[0,1,2]


def test_manifest_and_dry_plan_do_not_load_model(tmp_path, monkeypatch):
    from power_fm_foresight import campaign
    monkeypatch.setattr(campaign,'train_chain',lambda *a:pytest.fail('dry plan ran training'))
    opts=options(tmp_path,'--dry-run');c=prepare_campaign(opts)
    manifest=read_json(Path(c['root'])/'command.manifest.json')
    assert manifest['configuration']['dry_run']
    assert manifest['package_version'] and len(manifest['source']['sha256'])==64
    assert manifest['device'] and manifest['seeds']==[0,1,2]
    assert manifest['paths']['dataset']==str(tmp_path/'data')
    assert not (tmp_path/'campaign'/c['plan'][0]['id']).exists()
    with pytest.raises(ValueError,match='use --resume'):prepare_campaign(opts)


def test_resume_rejects_incompatible_override_and_source(tmp_path,monkeypatch):
    from power_fm_foresight import campaign
    c=prepare_campaign(options(tmp_path))
    explicit=dict(command='train',resume=Path(c['root']),lr=.01)
    with pytest.raises(ValueError,match='lr'):resume_campaign(command_options(explicit),explicit)
    explicit=dict(command='train',resume=Path(c['root']))
    assert resume_campaign(command_options(explicit),explicit)['plan']==c['plan']
    monkeypatch.setattr(campaign,'source_identity',lambda:dict(sha256='changed',files={}))
    with pytest.raises(ValueError,match='source'):resume_campaign(command_options(explicit),explicit)


def test_manifest_will_not_rewrite_incompatible_record(tmp_path):
    first=record_manifest(tmp_path,'report',{'kind':'status'},device='none',seeds=[],paths={})
    before=(tmp_path/'command.manifest.json').read_bytes()
    with pytest.raises(ValueError):record_manifest(tmp_path,'report',{'kind':'summary'},device='none',seeds=[],paths={})
    assert before==(tmp_path/'command.manifest.json').read_bytes()
    assert first['configuration']=={'kind':'status'}


def test_existing_artifacts_are_not_output_directories(tmp_path):
    write_json(tmp_path/'manifest.json',dict(historical=True))
    with pytest.raises(ValueError,match='existing data or training artifact'):
        record_manifest(tmp_path,'train',{},device='cpu',seeds=[0],paths={})
    assert not (tmp_path/'command.manifest.json').exists()


def test_numerical_comparison_rejects_nonfinite_values():
    from power_fm_foresight._artifacts import differences
    for value in (float('nan'),float('inf')):
        assert not differences(dict(score=value),dict(score=1.))['passed']


def test_individual_chain_resume_keeps_tile(tmp_path,monkeypatch):
    from power_fm_foresight import campaign
    c=prepare_campaign(options(tmp_path))
    c['plan']=c['plan'][1:2]
    write_json(Path(c['root'])/c['plan'][0]['id']/'chain.json',dict(device_index=3))
    seen=[]
    monkeypatch.setattr(campaign,'train_chain',lambda c,p,index,stop:seen.append(index))
    campaign.run_campaign(c)
    assert seen==[3]


def test_train_wrapper_reads_authoritative_checkpoint_progress(tmp_path,monkeypatch):
    import torch
    from power_fm_foresight import campaign, data, training
    c=prepare_campaign(options(tmp_path))
    chain=c['plan'][0];cfg=deepcopy(c['options']['configuration'])
    card=dict(config=cfg)
    monkeypatch.setattr(campaign,'dataset_card',lambda *a:card)
    monkeypatch.setattr(campaign,'select_device',lambda *a:'cpu')
    monkeypatch.setattr(data,'build_cache',lambda *a:None)
    progress=dict(steps=4090,epoch=0,step_in_epoch=4090,finished=False,
                  stop_reason='verification_step_limit',best_epoch=None)
    def run(cfg,device,stop_after_steps):
        root=Path(cfg['paths']['training_run'])
        root.mkdir(parents=True,exist_ok=True)
        torch.save(dict(progress=progress),root/'latest.pt')
        (root/'metrics.jsonl').write_text('{"steps":4000,"loss":1.0}\n')
        return dict(status='continue',steps_after=4090,epochs_reached=0)
    monkeypatch.setattr(training,'run',run)
    assert campaign.train_chain(c,chain,0,4090)==progress
    root=Path(c['root'])/chain['id']
    assert read_json(root/'boundary_checkpoint.json')['steps']==4090
    assert read_json(root/'chain.status.json')['steps']==4090


def test_duration_and_invalid_combinations(tmp_path):
    assert duration('6h')==21600 and duration('30m')==1800 and duration(60)==60
    for value in ('-1h','1d',0,float('nan')):
        with pytest.raises(ValueError):duration(value)
    with pytest.raises(ValueError):expand_plan(options(tmp_path,'--balance-weight','1','1'))


def test_report_reads_only_persisted_results(tmp_path,monkeypatch):
    from power_fm_foresight import training, reporting
    monkeypatch.setattr(training,'run',lambda *a,**kw:pytest.fail('report trained'))
    monkeypatch.setattr(training,'components',lambda *a,**kw:pytest.fail('report loaded a model'))
    data=tmp_path/'results'
    row=dict(complete=True,identity=dict(case='case30_ieee',seed=0,variant='conditioned',balance_weight=1.,samples=8,lr=1e-4,pilot=False,budget_capped=True),
        headlines=dict(A=dict(joint_es=.3,vm_crps=None,vm_coverage90_pct=None,pg_free_mae=2.,p_imbalance=4.)))
    write_json(data/'evaluation.json',dict(runs=[row],mixtures=[]))
    opts=command_options(dict(command='report',runs=[data],out=tmp_path/'report',format=['md','json','csv']))
    rows=reporting.report_runs(opts)
    assert rows[0]['budget_capped'] and rows[0]['vm_crps'] is None
    assert rows[1]['row_type']=='mean_sd' and rows[1]['joint_es_sd'] is None
    assert all((tmp_path/'report'/f'report.{suffix}').exists() for suffix in ('md','json','csv'))


def test_ensemble_group_order_and_compatibility():
    from power_fm_foresight.assessment import ensemble_groups
    base=dict(complete=True,identity=dict(case='case30_ieee',variant='conditioned',balance_weight=1,samples=8,lr=1e-4,pilot=False,budget_capped=False),
        comparison={k:None for k in ('dataset_sha256','scales_sha256','checkpoint_base','precision','deterministic_algorithms','partition','eval_seed','forecast_samples','origins')})
    rows=[]
    for seed in (2,0,1):
        row=deepcopy(base);row['identity']['seed']=seed;rows.append(row)
    groups=ensemble_groups(rows)
    assert [r['identity']['seed'] for r in next(iter(groups.values()))]==[0,1,2]
    rows[0]['identity']['samples']=2
    assert len(ensemble_groups(rows))==2
    with pytest.raises(ValueError,match='duplicate'):ensemble_groups([rows[1],rows[1]])


def test_campaign_report_counts_each_seed_once(tmp_path):
    from power_fm_foresight.reporting import persisted, summary_rows
    c=prepare_campaign(options(tmp_path));runs=[]
    for chain in c['plan'][:3]:
        path=Path(c['root'])/chain['id']
        write_json(path/'evaluations.json',dict(latest=str(tmp_path/'evaluation')))
        runs.append(dict(complete=True,identity=dict(run=str(path),case='case14_ieee',seed=chain['seed']),
                         headlines=dict(A=dict(joint_es=.2,vm_crps=.1,vm_coverage90_pct=90.,pg_free_mae=1.,p_imbalance=2.))))
    write_json(tmp_path/'evaluation/evaluation.json',dict(runs=runs,mixtures=[]))
    individual=[persisted(Path(c['root'])/chain['id']) for chain in c['plan'][:3]]
    assert [len(r['runs']) for r in individual]==[1,1,1]
    rows=summary_rows(individual+individual,'summary')
    assert len(rows)==4 and rows[-1]['n']==3


def test_fixed_verification_policy(tmp_path):
    from power_fm_foresight.verification import protocol
    p=tmp_path/'protocol.json'
    write_json(p,dict(schema='pff-verification.v1',check='regression',absolute_tolerance=1e-6,relative_tolerance=0))
    with pytest.raises(ValueError,match='1e-7'):protocol(p,'regression')
    write_json(p,dict(schema='pff-verification.v1',check='regression',absolute_tolerance=1e-7,relative_tolerance=0))
    assert protocol(p,'regression')['absolute_tolerance']==1e-7


def test_train_requires_explicit_seed_or_config_seed(tmp_path):
    args=dict(command='train',case=[30],dataset=str(tmp_path))
    with pytest.raises(ValueError,match='seed'):expand_plan(command_options(args))
    with pytest.raises(ValueError,match='divide'):expand_plan(options(tmp_path,'--samples','3'))


def test_noninferiority_accepts_improvement_and_rejects_gap(tmp_path):
    from power_fm_foresight.verification import check_reproducibility
    run=tmp_path/'run';adapter=run/'adapter'
    cfg=dict(seed=0,task=dict(canonical_case='pglib_opf_case30_ieee'),stage_b=dict(deterministic_algorithms=False,
        training_samples=8,optimizer=dict(lr=1e-4),loss=dict(power_balance_weight=1.),max_epochs=2,adapter={}))
    write_json(run/'chain.json',{});write_json(run/'config.json',cfg)
    write_json(adapter/'epochs.json',[dict(epoch=1,selection_energy_score=.1),dict(epoch=2,selection_energy_score=.1)])
    (adapter/'metrics.jsonl').write_text(''.join(json.dumps(dict(steps=s,loss=.2))+'\n' for s in range(100,8181,100)))
    write_json(adapter/'restores/a.json',dict(passed=True,manifest_exact=True,progress=dict(steps=4090,step_in_epoch=4090,epoch=0),state=dict(rng=dict(saved_sha256='x',restored_sha256='x'))))
    write_json(run/'boundary_resume.json',dict(passed=True,journal_prefix_exact=True))
    p=dict(epochs=[1,2],slack=.005,restart_after_steps=4090,loss_curve=dict(cadence_steps=100))
    references=[([dict(epoch=1,selection_energy_score=.5),dict(epoch=2,selection_energy_score=.4)],[]) for _ in range(3)]
    assert check_reproducibility(run,p,references)['passed']
    write_json(adapter/'epochs.json',[dict(epoch=1,selection_energy_score=.506),dict(epoch=2,selection_energy_score=.1)])
    assert not check_reproducibility(run,p,references)['passed']
