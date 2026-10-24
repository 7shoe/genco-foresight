"""Replication-only multi-chain launcher; released scientific files stay immutable."""
import argparse
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import re

import yaml
from common import ROOT, read_json, write_json, sha256, load_config
from scripts.training_chain import job_state

FAMILY = ROOT / 'configs/replication/family.json'


def settings():
    f = read_json(FAMILY)
    return f, Path(f['root'])


def atomic_text(path, text):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent)
    with os.fdopen(fd, 'w') as stream: stream.write(text)
    os.replace(tmp, path)


def guard_release():
    f, _ = settings()
    tag = subprocess.check_output(['git','rev-parse','refs/tags/v0.1.0'],cwd=ROOT).decode().strip()
    assert tag == f['release_tag_object'], 'Release tag changed'
    manifest = read_json(Path(f['release_bundle'])/'bundle.json')
    checked = 0
    for name, digest in manifest['sha256'].items():
        if name.startswith('code/'):
            p = ROOT / name[5:]
            assert p.is_file() and sha256(p) == digest, f'Released file changed: {p}'
            checked += 1
    return checked


def paths(chain):
    _, root = settings()
    out = root/'case30_ieee'/chain['version']/f"seed{chain['seed']}"
    return out, out/'adapter', ROOT/'configs/replication'/f"case30_{chain['id']}.yaml"


def decision(message, rule, chain=None):
    _, root = settings(); folder=root/'decisions'; folder.mkdir(parents=True,exist_ok=True)
    write_json(folder/f'{time.time_ns()}_{os.getpid()}.json', {'time_utc':datetime.now(timezone.utc).isoformat(),
        'chain':chain,'decision':message,'rule':rule})


def prepare():
    f, root=settings(); count=guard_release(); root.mkdir(parents=True,exist_ok=True)
    assets=root/'case30_ieee/assets'; assets.mkdir(parents=True,exist_ok=True)
    old=Path(load_config(ROOT/'config.yaml')['paths']['assets_root'])
    cp=assets/'genco-opfdata'; cp.mkdir(exist_ok=True)
    for name in ('case30_ieee','splits'):
        if not (cp/name).exists(): (cp/name).symlink_to(old/'genco-opfdata'/name,target_is_directory=True)
    if not (cp/'manifest.json').exists(): shutil.copy2(old/'genco-opfdata/manifest.json',cp/'manifest.json')
    if not (assets/'profiles').exists(): (assets/'profiles').symlink_to(old/'profiles',target_is_directory=True)
    for c in f['chains']:
        out, train, config=paths(c); train.mkdir(parents=True,exist_ok=True)
        original=load_config(ROOT/('config.yaml' if c['version']=='v3' else 'configs/v4.yaml'))
        cfg=load_config(ROOT/('config.yaml' if c['version']=='v3' else 'configs/v4.yaml'))
        cfg['seed']=c['seed']
        cfg['paths'].update(runs_root=str(out),training_run=str(train),assets_root=str(assets),evidence_root=str(out/'evidence'))
        text=yaml.safe_dump(cfg,sort_keys=False)
        if config.exists(): assert config.read_text()==text
        else: config.write_text(text)
        cache=train/'cache'; cache.mkdir(exist_ok=True)
        source=Path(original['paths']['training_run'])/'cache/manifest.json'
        if not (cache/'manifest.json').exists(): shutil.copy2(source,cache/'manifest.json')
        assert sha256(cache/'manifest.json')==sha256(source)
        readiness=read_json(Path(cfg['paths']['stage_a_run'])/'readiness.json')
        assert readiness['stage_a_audit_passed'] and readiness['training_allowed']
        assert readiness['manifest_sha256']==sha256(Path(cfg['paths']['temporal_root'])/'manifest.json')
        write_json(out/'provenance.json',{'training_seed':c['seed'],'logical_seed':c['seed'],'seed0_rng':42,
            'frozen_data':cfg['paths']['temporal_root'],'data_manifest_sha256':readiness['manifest_sha256'],
            'stage_a_audit':cfg['paths']['stage_a_run'],'cache_manifest_sha256':sha256(source),
            'cache_reused_read_only':True,'config_sha256':sha256(config),'original_config_sha256':sha256(ROOT/('config.yaml' if c['version']=='v3' else 'configs/v4.yaml')),
            'non_path_changes':{'seed':[42,c['seed']]},'queue_override':'debug-scaling; no optimization-setting changes'})
    for grid in f['blocked_grids']:
        write_json(root/grid/'status.json',{'status':'blocked','parity':'not run; no numerical parity failure claimed',
            'data':'not generated','training':'not started','reason':f['blockers'],
            'rule':'If a grid needs a code change to run, stop and report; do not patch.'})
    if not (root/'protocol.json').exists():
        write_json(root/'protocol.json',{**f,'released_files_checked':count,'frozen_before_new_training':True,
            'created_utc':datetime.now(timezone.utc).isoformat()})
        decision('Park priorities 2–4: released loader, solver, parity and tensor dimensions are case30-specific.',
                 'Grid needs a code change → stop and report; released files immutable.')
    reports()


def chain_state(c):
    out, train, _=paths(c)
    state=read_json(train/'status.json') if (train/'status.json').exists() else {}
    ledger=read_json(out/'budget.json') if (out/'budget.json').exists() else {'used_seconds':0.,'crashes':0}
    return out,train,state,ledger


def reports():
    f,root=settings(); now=datetime.now(timezone.utc).isoformat()
    control=read_json(root/'controller.json') if (root/'controller.json').exists() else {}
    rows=['# Replication status','',f'Updated {now}. Release v0.1.0 unchanged. No publication.','',
        '| Chain | Job | State | Steps | Epoch | Best epoch | Selection ES | Budget seconds | Evidence |',
        '|---|---|---|---:|---:|---:|---:|---:|---|']
    for c in f['chains']:
        out,train,s,l=chain_state(c)
        phase='parked' if l.get('parked') else ('finished' if s.get('status')=='finished' or l['used_seconds']>=21585 else 'pending/running')
        curve=read_json(train/'epochs.json') if (train/'epochs.json').exists() else []
        best=min(curve,key=lambda x:x['selection_energy_score']) if curve else {}
        steps=s.get('steps_after',0)
        logs=sorted((out/'logs').glob('*_training.log')) if (out/'logs').exists() else []
        if logs:
            with logs[-1].open('rb') as stream:
                stream.seek(max(0,logs[-1].stat().st_size-8192)); tail=stream.read().decode(errors='replace')
            found=re.findall(r"'step': (\d+)",tail)
            if found:steps=max(steps,int(found[-1]))
        live=next((j['job_id'] for j in control.get('jobs',[]) if j.get('state')=='R' and not j.get('finished')),None)
        shown_job=live if phase=='pending/running' and live else s.get('job_id',control.get('current_job',''))
        rows.append(f"| {c['id']} | {shown_job} | {phase} | {steps} | {max(s.get('epochs_reached',0),len(curve))} | {best.get('epoch',s.get('best_epoch',''))} | {best.get('selection_energy_score',s.get('best_selection',''))} | {l['used_seconds']:.1f} | {train} |")
    rows += ['', '| Grid | Data / audit | Parity | Evaluation |','|---|---|---|---|',
        '| case30_ieee | Existing frozen v3/v4 passed; reused by checksum | Existing same-node bit-identical result reused | Per-seed pending until verification |']
    rows += [f'| {g} | blocked: fixed case30 code | not run | blocked |' for g in f['blocked_grids']]
    rows += ['', 'Budget column is completed production-job wall time, including startup/selection/checkpointing. Running job time is added at its boundary; resume tests are separately recorded preparation.',
             f"Controller phase: {control.get('phase','training')}; calendar elapsed: {time.time()-control.get('created_epoch',time.time()):.1f} s; completed queue wait: {sum(j.get('queue_wait_seconds',0) for j in control.get('jobs',[])):.1f} s."]
    verification=['| Chain | Resume 200 vs 100 + kill + 100 | Startup s | First optimizer step s | Evidence |',
                  '|---|---|---:|---:|---|']
    for c in f['chains']:
        out,train,s,l=chain_state(c)
        resume=read_json(train/'resume_test/report.json') if (train/'resume_test/report.json').exists() else {}
        startup=read_json(train/'startup.json') if (train/'startup.json').exists() else {}
        if (train/'latest.pt').exists() and not (out/'first_checkpoint.json').exists():
            write_json(out/'first_checkpoint.json',{'first_observed_utc':now,'file_mtime_utc':datetime.fromtimestamp((train/'latest.pt').stat().st_mtime,timezone.utc).isoformat(),
                       'path':str(train/'latest.pt'),'note':'First production latest.pt observed by controller; resume-test files excluded.'})
        verification.append(f"| {c['id']} | {'bit-identical, max tensor difference 0' if resume.get('passed') else 'pending / failed'} | {startup.get('startup_seconds','')} | {startup.get('first_optimizer_step_completed_seconds','')} | {train}/resume_test/report.json; {train}/startup.json |")
        job_rows=['# Replication chain '+c['id'],'','| Job | Node | Steps | Epochs | Production wall s | Queue wait s |','|---|---|---:|---:|---:|---:|']
        results=[]
        for j in control.get('jobs',[]):
            evidence=out/'boundaries'/f"{j['job_id'].split('.')[0]}.json"
            if not evidence.exists():continue
            boundary=read_json(evidence);r=boundary.get('training_result',{})
            if not r:continue
            results.append({'job_id':j['job_id'],'queue_wait_seconds':j.get('queue_wait_seconds',0),**boundary})
            job_rows.append(f"| {j['job_id']} | {boundary.get('node','')} | {r.get('steps_done','')} | {r.get('epochs_reached','')} | {boundary.get('production_wall_seconds',0):.2f} | {j.get('queue_wait_seconds',0):.2f} |")
        job_rows += ['',f"Cumulative production wall: {l['used_seconds']:.2f} s / 21600 s. Crashes: {l['crashes']}. Preparation includes the separately retained resume test.",
                     f"Selection curve: {train}/epochs.json. Resume evidence: {train}/resume_test/report.json."]
        atomic_text(out/'chain.md','\n'.join(job_rows)+'\n')
        write_json(out/'chain.json',{'chain':c,'jobs':results,'budget':l,'selection_curve':str(train/'epochs.json'),
                   'startup':startup,'resume_test_passed':resume.get('passed'),'calendar_elapsed_seconds':time.time()-control.get('created_epoch',time.time())})
    rows += ['',*verification]
    atomic_text(root/'STATUS.md','\n'.join(rows)+'\n')
    morning=['# Replication morning report','',f'Updated {now}; initial report exists before 07:30 CT and is refreshed at job boundaries and hourly.','',
        '## Priority 1 — case30 seeds','',*rows[4:10],
        '', 'Selection curves: each chain’s `adapter/epochs.json`; checkpoints and startup records remain alongside it.',
        '',*verification,
        '', '## Priority 2 — frozen multi-grid OOD audit','',
        '| Grid | Status | Evidence |','|---|---|---|',
        f'| case30 | released v3 OOD complete | {root.parent}/stage_b/diagnostics/report.json |',
        *[f'| {g} | blocked before execution | {root/g}/status.json |' for g in f['blocked_grids']],
        '', 'The five-grid figure is not available; no unsupported grid is labelled a parity failure.',
        '', '## Priority 3 — v4 recipe on 14/57/118','',
        '| Work | Status | Evidence |','|---|---|---|',
        *[f'| {g} data / preflight / three seeds | blocked: released dimensions | {root/g}/status.json |' for g in f['blocked_grids'][:3]],
        '', '## Priority 4 — case500','',
        '| Work | Status | Evidence |','|---|---|---|',
        f'| Data / K=4 or 8 / 12-hour preflight | blocked; no time/memory measurement invented | {root/"case500_goc/status.json"} |',
        '', '## Seed-0 reproduction and new-seed results','']
    check=root/'case30_ieee/reproduction/report.json'
    if check.exists():
        v=read_json(check);morning += [f"Passed: {v.get('passed')}; max relative difference: {v.get('max_relative_difference')}; evidence: {check}"]
    else:morning += ['Pending. New-seed scoring is gated on reproduction of released seed-0 values.']
    if (check.parent/'blocked.json').exists():morning += [f'Reproduction blocked: {check.parent/"blocked.json"}; new-seed scoring remains parked. Training state is shown above.']
    for version in ('v3','v4'):
        table=root/'case30_ieee'/version/'summary.md'
        morning += ['',table.read_text() if table.exists() else f'{version}: no new seed evaluation completed; no seed mean/SD or CI claimed.']
        mix=root/'case30_ieee'/version/'mixture/report.json'
        if mix.exists():morning += ['',f'Equal-weight 192-draw mixture completed: {mix}. '+f['ensemble']['qualification']]
    morning += ['', '## Decisions under the overnight rules','']
    for p in sorted((root/'decisions').glob('*.json')) if (root/'decisions').exists() else []:
        d=read_json(p);morning.append(f"- {d['time_utc']}: {d['decision']} Rule: {d['rule']} Evidence: {p}")
    morning += ['', '## Questions for 08:00 CT','',
        '1. Authorize a later branch to generalize the released case30 loader/solver/dimensions? Default: keep priorities 2–4 parked on this branch; do not change the release.',
        '2. Treat paired-origin bootstrap CIs as descriptive despite four correlated source episodes? Default: label that limitation prominently and retain origin-level pairs; make no independent-day claim.']
    if (check.parent/'blocked.json').exists():
        morning += ['3. Reproduction failed or needs runner investigation. Default: retain the failure record and withhold all new-seed scores until the cause is resolved without changing released science.']
    atomic_text(root/'MORNING_REPORT.md','\n'.join(morning)+'\n')
    when=datetime.fromisoformat(f['decision_table_ct']).timestamp()
    if time.time()>=when:
        lines=['# Sunday inclusion decision','',f'Updated {now}. Cutoff {f["inclusion_cutoff_ct"]}.','',
               '| Entry | Complete and audited before cutoff | Evidence |','|---|---|---|']
        cutoff=datetime.fromisoformat(f['inclusion_cutoff_ct']).timestamp()
        for c in f['chains']:
            out,_,_=paths(c);completion=out/'stage_b/completion.json'
            ready=completion.exists() and datetime.fromisoformat(read_json(completion)['completed_utc']).timestamp()<cutoff
            lines.append(f"| {c['id']} | {ready} | {completion} |")
        lines += [f'| {g} | no — blocked | {root/g}/status.json |' for g in f['blocked_grids']]
        lines += ['', 'No automatic post edit. Pending work remains Next steps.']
        atomic_text(root/'SUNDAY_DECISION.md','\n'.join(lines)+'\n')


def invoke(command,env,log,timeout):
    with Path(log).open('a') as stream:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:return process.wait(timeout=max(1,timeout))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);process.wait();return -9


def worker(identity):
    f,root=settings();c=next(x for x in f['chains'] if x['id']==identity)
    out,train,s,ledger=chain_state(c);_,_,config=paths(c)
    os.environ.update(GENCO_CONFIG=str(config),PYTHONHASHSEED=str(c['seed']),ZE_FLAT_DEVICE_HIERARCHY='FLAT',ZE_AFFINITY_MASK=str(c['tile']))
    start=float(os.environ['GENCO_FAMILY_JOB_START']);job=os.environ['PBS_JOBID'].split('.')[0]
    (out/'logs').mkdir(exist_ok=True)
    def boundary(phase):
        write_json(out/'boundaries'/f'{job}.json',{'job_id':job,'phase':phase,'budget':ledger})
    if ledger.get('parked') or s.get('status')=='finished' or ledger['used_seconds']>=21585:
        boundary('idle');return
    import torch
    assert torch.xpu.device_count()==1, f'Expected isolated tile; got {torch.xpu.device_count()}'
    torch.xpu.set_device(0)
    write_json(out/'affinity.json',{'tile_flat_index':c['tile'],'visible_devices':1,'device':str(torch.xpu.get_device_properties(0)),
        'job_id':job,'node':os.uname().nodename,'source':f['affinity_reference']})
    cfg=load_config(config)
    if not (train/'resume_test/report.json').exists():
        from scripts.resume_test import trial
        try:
            r=trial(cfg,'xpu',200,train/'resume_test')
            write_json(train/'resume_test/report.json',r)
            if not r['passed']:raise RuntimeError('Bit-identical resume comparison failed')
        except Exception as e:
            ledger.update(parked=True,reason=repr(e),last_accounted_job=job);write_json(out/'budget.json',ledger);boundary('resume_failed')
            decision(f'{identity}: parked after resume-test failure: {e}','Resume test fails → launcher config only; otherwise skip chain.',identity);return
    assert read_json(train/'resume_test/report.json')['passed']
    remaining=3600-(time.time()-start)
    if remaining<600:
        boundary('preparation_complete_time_guard');return
    if (train/'failure.json').exists():
        history=out/'failures';history.mkdir(exist_ok=True)
        (train/'failure.json').rename(history/f'{job}_prior_failure.json')
    latest=train/'latest.pt'
    if latest.exists():
        recovery=dict(s,cumulative_training_wall_seconds=max(ledger['used_seconds'],s.get('cumulative_training_wall_seconds',0)),checkpoint_sha256=sha256(latest))
        write_json(train/'status.json',recovery)
    began=time.time()
    write_json(out/'attempt_start.json',{'job':job,'start_epoch':began,'budget_before':ledger['used_seconds']})
    env=dict(os.environ,GENCO_JOB_START_EPOCH=str(began),PBS_WALLTIME=str(int(remaining)))
    code=invoke([f['python'],'-u','-m','scripts.train_adapter','--device','xpu'],env,out/'logs'/f'{job}_training.log',min(remaining-40,21600-ledger['used_seconds']-2))
    elapsed=time.time()-began;ledger['used_seconds']+=elapsed;ledger['last_accounted_job']=job
    result=read_json(train/'status.json') if (train/'status.json').exists() else {}
    if code:
        ledger['crashes']+=1
        ledger['parked']=ledger['crashes']>=2 or not latest.exists()
        decision(f'{identity}: exit {code}; '+('parked' if ledger['parked'] else 'one restart from latest.pt scheduled'),
                 'Any chain crashes → resume latest.pt once; second crash parks chain.',identity)
    write_json(out/'budget.json',ledger)
    write_json(out/'boundaries'/f'{job}.json',{'job_id':job,'node':os.uname().nodename,'returncode':code,'production_wall_seconds':elapsed,'budget':ledger,'training_result':result})


def phase_plan():
    f,root=settings()
    active=[]
    for c in f['chains']:
        _,_,s,l=chain_state(c)
        if not l.get('parked') and s.get('status')!='finished' and l['used_seconds']<21585:active.append(c)
    if active:return 'training',active
    gate=root/'case30_ieee/reproduction/report.json'
    if not gate.exists():
        if (gate.parent/'blocked.json').exists():return 'blocked',[]
        return 'reproduction',[]
    if not read_json(gate).get('passed'):return 'blocked',[]
    ready=[]
    for c in f['chains']:
        out,train,s,l=chain_state(c)
        evaluation=read_json(out/'evaluation_state.json') if (out/'evaluation_state.json').exists() else {}
        if (not l.get('parked') and not evaluation.get('parked') and (train/'best.pt').exists()
            and not (out/'stage_b/completion.json').exists()):ready.append(c)
    if ready:return 'evaluation',ready
    variants=[]
    for v in ('v3','v4'):
        b=root/'case30_ieee'/v
        if all((b/f'seed{s}/stage_b/completion.json').exists() for s in (1,2)) and not (b/'mixture/completion.json').exists():
            if not (b/'mixture/blocked.json').exists():variants.append(v)
    return ('mixture',variants) if variants else ('idle',[])


def job():
    f,root=settings();guard_release();reports()
    phase,work=phase_plan();jid=os.environ['PBS_JOBID'].split('.')[0]
    job_record={'job_id':os.environ['PBS_JOBID'],'node':os.uname().nodename,'phase':phase,
                'start_epoch':float(os.environ['GENCO_FAMILY_JOB_START']),
                'launcher_sha256':sha256(__file__),'runner_sha256':sha256(ROOT/'scripts/replication_evaluate.py'),
                'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()}
    write_json(root/'case30_ieee/jobs'/f'{jid}_start.json',job_record)
    tasks=[];streams=[]
    def launch(name,module,args,tile,log):
        Path(log).parent.mkdir(parents=True,exist_ok=True);stream=Path(log).open('a');streams.append(stream)
        env=dict(os.environ,ZE_FLAT_DEVICE_HIERARCHY='FLAT',ZE_AFFINITY_MASK=str(tile))
        proc=subprocess.Popen([f['python'],'-u','-m',module,*args],cwd=ROOT,env=env,
                              stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        tasks.append({'name':name,'process':proc,'log':str(log)})
    if phase=='training':
        for c in f['chains']:
            out,_,_=paths(c)
            launch(c['id'],'scripts.replication_chain',['worker','--chain',c['id']],c['tile'],out/'logs'/f'{jid}_launcher.log')
    gate=root/'case30_ieee/reproduction'
    if not (gate/'report.json').exists() and not (gate/'blocked.json').exists():
        for shard in range(4):
            launch(f'reproduce_cpu_{shard}','scripts.replication_evaluate',['reproduce-scores','--shard',str(shard),'--workers','4'],
                   4,gate/'logs'/f'{jid}_cpu_{shard}.log')
        for tile,v in enumerate(('v3','v4'),4):
            launch(f'reproduce_forward_{v}','scripts.replication_evaluate',['reproduce-forward','--version',v,'--device','xpu:0'],
                   tile,gate/'logs'/f'{jid}_forward_{v}.log')
    if phase=='evaluation':
        for c in work:
            out,_,_=paths(c)
            launch(c['id'],'scripts.replication_evaluate',['evaluate','--chain',c['id'],'--device','xpu:0'],
                   c['tile'],out/'logs'/f'{jid}_evaluation.log')
    if phase=='mixture':
        for v in work:
            for shard in range(4):
                launch(f'{v}_mixture_{shard}','scripts.replication_evaluate',['mixture','--version',v,'--shard',str(shard),'--workers','4'],
                       4,root/'case30_ieee'/v/'mixture/logs'/f'{jid}_{shard}.log')
    try:
        for task in tasks:
            task['returncode']=task['process'].wait()
        from scripts.replication_evaluate import reproduction_merge,merge_chain,summary
        if not (gate/'report.json').exists() and not (gate/'blocked.json').exists():
            failures=[{k:v for k,v in t.items() if k!='process'} for t in tasks if t['name'].startswith('reproduce') and t['returncode']]
            if failures:
                write_json(gate/'blocked.json',{'failures':failures,'job':jid})
                decision('Seed-0 reproduction stopped; new-seed scoring parked. See reproduction/blocked.json.',
                         'New runner reproduction must pass before any new seed is scored; questions beyond rules are parked.')
            else:reproduction_merge()
        if phase=='evaluation':
            for c in work:
                task=next(t for t in tasks if t['name']==c['id']);out,_,_=paths(c)
                if task['returncode']:
                    state_path=out/'evaluation_state.json'
                    state=read_json(state_path) if state_path.exists() else {'crashes':0}
                    state['crashes']+=1;state['parked']=state['crashes']>=2;state['last_log']=task['log']
                    write_json(state_path,state)
                    decision(f"{c['id']} evaluation failed; "+('parked' if state['parked'] else 'one retry of incomplete origins'),
                             'One crash retry; retain atomic completed origins and park on a second crash.',c['id'])
                else:merge_chain(c['id'])
            for v in ('v3','v4'):summary(v)
        if phase=='mixture':
            for v in work:
                failed=[t['log'] for t in tasks if t['name'].startswith(v) and t['returncode']]
                if failed:
                    write_json(root/'case30_ieee'/v/'mixture/blocked.json',{'logs':failed,'job':jid})
                    decision(f'{v} mixture scoring parked after runner failure; source ensembles preserved.',
                             'Judgment beyond the overnight rules → park and report.')
    finally:
        for task in tasks:
            if task['process'].poll() is None:
                os.killpg(task['process'].pid,signal.SIGTERM)
        for stream in streams:stream.close()
        job_record.update(end_epoch=time.time(),tasks=[{k:v for k,v in t.items() if k!='process'} for t in tasks])
        write_json(root/'case30_ieee/jobs'/f'{jid}.json',job_record);reports()


def controller(once=False):
    f,root=settings();root.mkdir(parents=True,exist_ok=True)
    lock=(root/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=read_json(root/'controller.json') if (root/'controller.json').exists() else {'jobs':[],'created_epoch':time.time()}
    cutoff=datetime.fromisoformat(f['inclusion_cutoff_ct']).timestamp()
    while time.time()<cutoff:
        state.update(pid=os.getpid(),host=os.uname().nodename,updated_epoch=time.time())
        pending=None
        for j in state['jobs']:
            if j.get('finished'):continue
            st,code=job_state(j['job_id']);j.update(state=st,exit_status=code)
            if st=='F':
                j['finished']=True
                jid=j['job_id'].split('.')[0]
                record=root/'case30_ieee/jobs'/f'{jid}.json'
                started=root/'case30_ieee/jobs'/f'{jid}_start.json'
                if started.exists():j['phase']=read_json(started)['phase']
                if record.exists():
                    r=read_json(record);j['result']=r;j['phase']=r.get('phase','training')
                    dependency_end=j['submitted_epoch']
                    if j.get('dependency'):
                        prior=next(x for x in state['jobs'] if x['job_id']==j['dependency'])
                        dependency_end=max(dependency_end,prior.get('result',{}).get('end_epoch',dependency_end))
                    j['dependency_wait_seconds']=dependency_end-j['submitted_epoch']
                    j['queue_wait_seconds']=max(0,r['start_epoch']-dependency_end)
                for c in f['chains'] if j.get('phase','training')=='training' else []:
                    out,train,s,l=chain_state(c)
                    if (out/'boundaries'/f'{jid}.json').exists() or l.get('parked') or s.get('status')=='finished':continue
                    if l.get('last_accounted_job')==jid:continue
                    attempt=read_json(out/'attempt_start.json') if (out/'attempt_start.json').exists() else {}
                    if attempt.get('job')==jid:
                        # Conservative end at detection, so lost post-checkpoint wall time is charged.
                        l['used_seconds']=max(l['used_seconds'],attempt['budget_before']+time.time()-attempt['start_epoch'])
                    l['crashes']+=1;l['last_accounted_job']=jid
                    l['parked']=l['crashes']>=2 or not (train/'latest.pt').exists()
                    write_json(out/'budget.json',l)
                    decision(f"{c['id']}: missing worker boundary; " + ('parked' if l['parked'] else 'one checkpoint retry'),
                        'Any chain crashes → resume latest.pt once; second crash parks chain.',c['id'])
                if j.get('phase')=='evaluation' and not record.exists():
                    for c in f['chains']:
                        out,train,s,l=chain_state(c)
                        if l.get('parked') or (out/'stage_b/completion.json').exists():continue
                        state_path=out/'evaluation_state.json'
                        evaluation=read_json(state_path) if state_path.exists() else {'crashes':0}
                        if evaluation.get('parked') or evaluation.get('last_accounted_job')==jid:continue
                        evaluation['crashes']+=1;evaluation['parked']=evaluation['crashes']>=2
                        evaluation['last_accounted_job']=jid;write_json(state_path,evaluation)
                        decision(f"{c['id']}: evaluation job ended without its completion record; "+('parked' if evaluation['parked'] else 'one retry of saved partial origins'),
                                 'One crash retry; second crash parks chain.',c['id'])
                if code:decision(f"Job {j['job_id']} ended with exit {code}; worker artifacts retained.",'Crash recovery requires latest checkpoint; second crash parks chain.')
            else:pending=j
        phase,active=phase_plan();state['phase']=phase
        unfinished=[j for j in state['jobs'] if not j.get('finished')]
        prefetch=(phase=='training' and len(unfinished)==1 and unfinished[0].get('state')=='R'
                  and not state.get('dependency_prefetch_rejected'))
        if (not pending and phase not in ('idle','blocked')) or prefetch:
            guard_release()
            # A single family may occupy the queue; unrelated jobs are never cancelled.
            q=subprocess.run(['qstat','-u',os.environ.get('USER','siebenschuh')],capture_output=True,text=True,timeout=30)
            if q.returncode:raise RuntimeError('Cannot inspect queue safely: '+q.stderr)
            existing=[line.split()[0].split('.')[0] for line in q.stdout.splitlines() if line.strip() and line.lstrip()[0].isdigit()]
            expected=unfinished[0]['job_id'].split('.')[0] if prefetch else None
            occupied=any(j!=expected for j in existing)
            if not occupied:
                log=root/'case30_ieee/logs';log.mkdir(parents=True,exist_ok=True)
                number=len(state['jobs'])+1
                command=['qsub','-q',f['queue'],'-N',f'genco-rep-{number}','-o',str(log/f'job_{number:02d}.log'),str(ROOT/'scripts/replication_chain.pbs')]
                dependency=unfinished[0]['job_id'] if prefetch else None
                if dependency:command[-1:-1]=['-W',f'depend=afterany:{dependency}']
                run=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=45)
                if run.returncode:
                    state['queue_wait_reason']=run.stderr.strip()
                    if prefetch:
                        state['dependency_prefetch_rejected']=run.stderr.strip()
                        decision('Dependency prefetch rejected; controller will submit after each job boundary: '+run.stderr.strip(),
                                 'Dependency chaining if queue allows; otherwise boundary resubmission.')
                else:
                    jid=run.stdout.strip().splitlines()[-1];state['jobs'].append({'job_id':jid,'submitted_epoch':time.time(),'state':'Q','phase':phase,'dependency':dependency})
                    state['current_job']=jid;print('Submitted',jid,flush=True)
            else:state['queue_wait_reason']='Existing user job occupies slot; no unrelated cancellation.'
        write_json(root/'controller.json',state);reports()
        if once:return
        time.sleep(30)
    reports()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','controller','job','worker','status']);p.add_argument('--chain');p.add_argument('--once',action='store_true');a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='controller':
        while True:
            try:
                controller(a.once);break
            except BlockingIOError:raise
            except Exception as exc:
                _,root=settings()
                write_json(root/'controller_error.json',{'error':repr(exc),'time_utc':datetime.now(timezone.utc).isoformat(),
                           'policy':'No submission until the next successful state/release/queue check.'})
                print('Controller check failed; retrying safely:',repr(exc),flush=True)
                if a.once:raise
                time.sleep(30)
    elif a.action=='job':job()
    elif a.action=='worker':worker(a.chain)
    else:reports()


if __name__=='__main__':main()
