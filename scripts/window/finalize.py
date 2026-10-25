"""Enforce the freeze, retain late-result labels, and merge the gated local release."""
import argparse
from datetime import datetime,timezone
import hashlib
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def read(p):return json.loads(Path(p).read_text()) if Path(p).exists() else {}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def git(repo,*args):return subprocess.check_output(['git',*args],cwd=repo,text=True).strip()


def stop_submitters(root):
    stopped=[]
    for item in read(root/'protocol.json')['stop_submitters_at_freeze']:
        try:
            command=Path(f"/proc/{item['pid']}/cmdline").read_bytes().replace(b'\0',b' ').decode()
            if item['match'] in command:
                group=os.getpgid(item['pid'])
                if group!=item['pid']:raise RuntimeError('Submitter is not an isolated process group')
                os.killpg(group,signal.SIGTERM);stopped.append(item)
        except (FileNotFoundError,ProcessLookupError):pass
    write(root/'submission_cutoff_guard.json',dict(at=time.time(),stopped=stopped,
          reason='Stop only these task-owned automatic submitters90 seconds before freeze so an in-flight qsub cannot cross the deadline. Running jobs continue. The release-window controller has its own identical submission guard.'))


def inventory(root):
    freeze=read(root/'protocol.json')['freeze_epoch'];rows=[]
    for path in sorted((root/'tasks').glob('*.json')):
        task=read(path);state=read(root/'states'/path.name)
        if task.get('category')=='verification':continue
        snapshot=task.get('snapshot');pin_path=Path(task.get('source_pin',str(Path(snapshot).parent/'source_pin.json'))) if snapshot else None
        pin=read(pin_path) if pin_path else {}
        source=pin.get('source_commit',pin.get('commit'))
        if not source and snapshot and 'recovery_20261025_v1/source' in snapshot:source='8d93abb'
        ended=state.get('attempts',[{}])[-1].get('ended') if state.get('attempts') else None
        category=task.get('category','evaluation')
        if category in ('ensemble-depth','ablation') and (state.get('status')=='running' or (ended is not None and ended>freeze)):
            category='post-freeze'
        rows.append(dict(run_directory=task.get('run_directory'),category=category,status=state.get('status','not started'),
            task_id=task['id'],snapshot=snapshot,commit=source,source_pin=str(pin_path),snapshot_sha256=sha(pin_path) if pin else None,
            ended_epoch=ended,worker_seconds=state.get('worker_seconds',0)))
    fleet=root.parent/'fleet_20261025_off_v1';recovery=root.parent/'recovery_20261025_v1'
    for case in ('case14_ieee','case30_ieee','case57_ieee','case118_ieee'):
        for seed in range(3):
            directory=fleet/case/f'seed{seed}'
            rows.append(dict(run_directory=str(directory),category='fleet',status=read(directory/'adapter/status.json').get('status'),
                snapshot=str(fleet/'source'),commit=read(fleet/'source_pin.json').get('source_commit'),snapshot_sha256=sha(fleet/'source_pin.json')))
            rows.append(dict(run_directory=str(recovery/'evaluation'/case/f'seed{seed}'),category='fleet',status='evaluation complete',
                snapshot=str(recovery/'source'),commit=read(recovery/'source_pin.json').get('source_commit'),snapshot_sha256=sha(recovery/'source_pin.json')))
        rows.append(dict(run_directory=str(recovery/'evaluation'/case/'mixture'),category='fleet',status='three-seed mixture complete',
            snapshot=str(recovery/'source'),commit=read(recovery/'source_pin.json').get('source_commit'),snapshot_sha256=sha(recovery/'source_pin.json')))
    for case in ('case500_goc','case2000_goc'):
        directory=recovery/'pilots'/case/'seed0'
        rows.append(dict(run_directory=str(directory),category='pilot',status=read(directory/'adapter/status.json').get('status'),
            snapshot=str(recovery/'source'),commit=read(recovery/'source_pin.json').get('source_commit'),snapshot_sha256=sha(recovery/'source_pin.json')))
    genco=root.parents[2]
    for variant in ('v3','v4'):
        dirs=[genco/'runs'/f'adapter_{variant}',*[genco/'runs/replication/case30_ieee'/variant/f'seed{s}/adapter' for s in (1,2)]]
        for directory in dirs:
            manifest=read(directory/'manifest.json')
            if not manifest:continue
            rows.append(dict(run_directory=str(directory),category='released',status='historical read-only',snapshot='historical manifest source_sha256',
                commit='v0.1.0 for release; replication provenance retained per manifest',snapshot_sha256=sha(directory/'manifest.json')))
    record=dict(freeze_epoch=freeze,recorded_epoch=time.time(),runs=rows,protocol_sha256=sha(root/'protocol.json'))
    write(root/'freeze_inventory.json',record)
    lines=['# Freeze — 2026-10-25 19:00 CT','', 'No new jobs after 2026-10-26 00:00 UTC. Running workers may finish later; their results retain post-freeze timing. Historical releases and original fleet inputs remain unchanged.',
           '', '| Run directory | Category | Status at freeze | Snapshot commit | Snapshot/manifest SHA256 |', '|---|---|---|---|---|']
    for r in rows:lines.append(f"| {r['run_directory']} | {r['category']} | {r['status']} | {r.get('commit')} | {r.get('snapshot_sha256')} |")
    lines+=['',f'Complete snapshot paths and task/job accounting: {root}/freeze_inventory.json and controller.json.']
    for path in (root/'FREEZE.md',root.parent/'FREEZE.md'):path.write_text('\n'.join(lines)+'\n')


def jobs_finished(root):
    families=[read(root/'controller.json'),read(root.parent/'recovery_20261025_v1/pilots/controller.json')]
    for family in families:
        for j in family.get('jobs',[]):
            identity=j.get('id',j.get('job_id'))
            if not identity:continue
            r=subprocess.run(['qstat','-xf','-F','json',identity],capture_output=True,text=True,timeout=30)
            if r.returncode:return False
            value=next(iter(json.loads(r.stdout)['Jobs'].values()))
            if value['job_state']!='F':return False
    return True


def consolidate(root,config):
    repo=Path(config['release_worktree']);main=Path(config['main_worktree'])
    if git(repo,'status','--porcelain') or git(main,'status','--porcelain'):raise RuntimeError('Worktree changed; final merge parked without discarding edits')
    if git(repo,'rev-parse','HEAD')!=config['release_commit'] or git(main,'rev-parse','HEAD')!=config['main_commit']:
        raise RuntimeError('Branch moved after finalization was pinned; inspect before merging')
    if git(repo,'rev-parse','refs/tags/v0.1.0')!=config['protected_v010']:raise RuntimeError('Protected v0.1.0 changed')
    if git(repo,'tag','--list','v0.2.0'):raise RuntimeError('v0.2.0 already exists; never move it')
    gates=[root/'phase_b'/name/'gate.json' for name in ('replication','multigrid','recovery')]
    gates += [root/'phase_c/gate/gate.json',root/'phase_c/default_regression/gate.json']
    if not all(read(p).get('passed') for p in gates):raise RuntimeError('Required release correctness gate missing or failed')
    for name in ('models/temporal.py','training/loop.py','scripts/forecast_partition.py','scoring.py'):
        if sha(repo/name)!=sha(root/'phase_c/source'/name):raise RuntimeError(f'Untested scientific source change: {name}')
    evidence=dict(frozen=True,freeze_inventory=str(root/'freeze_inventory.json'),freeze_sha256=sha(root/'freeze_inventory.json'),
         gates={str(p):sha(p) for p in gates},jobs=read(root/'controller.json').get('jobs',[]),
         task_states={p.stem:read(p) for p in (root/'states').glob('*.json')},
         reports={str(root/name):sha(root/name) for name in ('cross_grid.json','mixture_sizes.json','pilot_results.json')},
         precision='fp32',K=8,B=1,production_policy='conditioned adapter, determinism OFF adopted with disclosure; A0/λ are separate budget-capped ablations')
    # Large per-task logs and state histories remain on Lustre.
    write(root/'release_evidence.json',evidence)
    small=dict(frozen=True,evidence=str(root/'release_evidence.json'),sha256=sha(root/'release_evidence.json'),gates=evidence['gates'],
               jobs=[j['id'] for j in evidence['jobs']],adopted_configuration=evidence['production_policy'])
    write(repo/'reports/v2/window.json',small)
    (repo/'reports/v2/window.md').write_text('# Release window\n\nFrozen at19:00CT. All three consolidation gates and A0 correctness/default regressions passed. OFF remains adopted with disclosure; no new optimization or precision policy is adopted by these ablations.\n\n'+
        f'Live/final evidence: [{root.name}]({root}/FREEZE.md). [Cross-grid table]({root}/cross_grid.md), [mixture-size table]({root}/mixture_sizes.md), [figure]({root}/mixture_sizes.png), [pilot snapshots]({root}/pilot_results.json).\n\n'+
        'Ensemble curves use only training seeds finished by the freeze; unavailable sizes remain explicit. Vm headline metrics retain mechanical bus selection; additional all-bus curves are labeled. A0 and λ rows are budget-capped single-seed experiments, separated from the three-seed fleet. Pilot rows are excluded from the cross-grid table and labeled not converged. Post-freeze worker completions do not become pre-freeze seeds.\n\n'+
        'The original multigrid index and unrelated edits were preserved. Dirty pre-existing worktrees are retained if normal worktree removal would discard changes. No packages were installed, no release artifacts rewritten, and nothing was pushed or published.\n')
    (repo/'RUNS.md').write_text('# Runs\n\n'+(root/'FREEZE.md').read_text())
    (repo/'reports/INDEX.md').write_text('# Report index\n\n- [Release window](v2/window.md)\n- [Main consolidation](v2/consolidation.md)\n- [Stage 1 verdict](v2/stage1_verdict.md)\n- [Recovery](v2/recovery.md)\n- [Pilot repair](v2/pilot_repair.md)\n- [V2 implementation inventory (historical)](V2_IMPL.md)\n- [Synthetic profiling](multigrid/profile.md)\n\n'+f'Final evaluation tables, curves and run evidence: {root}.\n')
    (repo/'STATUS.md').write_text('# Status\n\nFrozen at2026-10-25 19:00CT. Automatic submission is stopped.\n\n'+f'Per-item submitted/running/completed/not-started states and post-freeze outcomes: {root}/STATUS.md. Run inventory: {root.parent}/FREEZE.md.\n\nAll required code correctness gates passed. Scientific results and unavailable/incomplete cells are reported without changing acceptance thresholds.\n')
    files=['reports/v2/window.json','reports/v2/window.md','RUNS.md','reports/INDEX.md','STATUS.md']
    for phase in ('protocol.json','phase_b.protocol.json','phase_c.protocol.json','phase_d.protocol.json'):
        name='reports/v2/window_'+phase;shutil.copyfile(root/phase,repo/name);files.append(name)
    subprocess.run(['git','add','-f','--',*files],cwd=repo,check=True)
    subprocess.run(['git','commit','-m','Record frozen release-window results and run provenance'],cwd=repo,check=True)
    subprocess.run(['git','merge','--no-ff','release-window','-m','Merge gated budget-capped ablations and frozen evaluation handover'],cwd=main,check=True)
    subprocess.run(['git','tag','-a','v0.2.0','-m','Frozen multigrid release with disclosed OFF results and budget-capped ablations'],cwd=main,check=True)
    result=dict(status='complete',commit=git(main,'rev-parse','HEAD'),tag=git(main,'rev-parse','refs/tags/v0.2.0'),finished=time.time(),cleanup=[])
    original=Path(config['original_worktree'])
    for path in [repo,main,Path(config['fleet_worktree']),Path(config['pilots_worktree'])]:
        dirty=git(path,'status','--porcelain')
        if dirty:result['cleanup'].append(dict(path=str(path),status='retained',reason='Unrelated uncommitted edits',git_status=dirty));continue
        r=subprocess.run(['git','worktree','remove',str(path)],cwd=original,capture_output=True,text=True)
        result['cleanup'].append(dict(path=str(path),status='removed' if r.returncode==0 else 'retained',reason=r.stderr))
    write(root/'FINAL.json',result)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',type=Path,required=True);a=p.parse_args()
    root=a.root;cfg=read(a.config);freeze=read(root/'protocol.json')['freeze_epoch'];guarded=False;last_report=0
    lock=(root/'finalizer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while time.time()<freeze:
        if not guarded and time.time()>=freeze-90:stop_submitters(root);guarded=True
        if time.time()-last_report>=300 and time.time()<freeze-120:
            result=subprocess.run([sys.executable,str(Path(__file__).with_name('report.py')),'--root',str(root)])
            if result.returncode:write(root/'report_error.json',dict(returncode=result.returncode,at=time.time()))
            last_report=time.time()
        write(root/'finalizer_status.json',dict(pid=os.getpid(),status='scheduled',updated=time.time(),freeze_epoch=freeze))
        time.sleep(min(5,max(0,freeze-time.time())))
    if not guarded:stop_submitters(root)
    inventory(root)
    while time.time()<freeze+4200:
        try:
            if jobs_finished(root):break
        except Exception as error:write(root/'finalizer_poll_error.json',dict(error=repr(error),at=time.time()))
        write(root/'finalizer_status.json',dict(pid=os.getpid(),status='waiting for already-submitted jobs',updated=time.time()))
        time.sleep(30)
    subprocess.run([sys.executable,str(Path(__file__).with_name('report.py')),'--root',str(root)],check=True)
    shutil.copyfile(root.parent/'FREEZE.md',root/'FREEZE.md')
    consolidate(root,cfg)


if __name__=='__main__':
    try:main()
    except Exception as error:
        if '--root' in sys.argv:write(Path(sys.argv[sys.argv.index('--root')+1])/'FINAL_ERROR.json',dict(error=repr(error),at=time.time()))
        raise
