"""Capture scheduled best checkpoints without touching their training chains."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

from omegaconf import OmegaConf

PYTHON='/lus/flare/projects/FRAME-IDP/siebenschuh/envs/reason_venv_2026.1.0/bin/python'
CASES=('case14_ieee','case30_ieee','case57_ieee','case118_ieee')


def read(p):return json.loads(Path(p).read_text()) if Path(p).exists() else {}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)


def capture(root,identity,config,category,source):
    out=root/'snapshots'/identity
    if (out/'snapshot.json').exists():return out
    cfg=OmegaConf.load(config);training=Path(cfg.paths.training_run)
    if not (training/'best.pt').exists():return None
    destination=out/'adapter';destination.mkdir(parents=True,exist_ok=True)
    inputs={}
    for name in ('best.pt','manifest.json','status.json','epochs.json'):
        p=training/name
        if not p.exists():continue
        digest=sha(p);shutil.copyfile(p,destination/name)
        if digest!=sha(p) or digest!=sha(destination/name):return None
        inputs[str(p)]=digest
    import torch
    best=torch.load(destination/'best.pt',map_location='cpu',weights_only=True)
    assert best['manifest']==read(destination/'manifest.json')
    original=read(destination/'status.json');write(out/'original_status.json',original)
    epochs=read(destination/'epochs.json');progress=best['progress']
    original.update(best_epoch=progress['best_epoch'],best_selection=progress['best_selection'],snapshot_projection=True)
    write(destination/'status.json',original)
    record=dict(category=category,source_training=str(training),source_code=str(source),captured_utc=datetime.now(timezone.utc).isoformat(),
                source_sha256=inputs,best_epoch=progress['best_epoch'],selection_ES=progress['best_selection'],
                epochs_observed=len(epochs),epochs=epochs,training_status=read(training/'status.json').get('status'),
                note='Private evaluation status projects best_epoch from the copied checkpoint; original training files are unchanged.')
    cfg.paths.training_run=str(destination);cfg.paths.runs_root=str(out);cfg.seed=42
    path=out/'evaluation.yaml';path.write_text(OmegaConf.to_yaml(cfg,resolve=False))
    for p in destination.iterdir():p.chmod(0o444)
    write(out/'snapshot.json',record);return out


def install(root,identity,out,case,source,pilot=False):
    spec_path=out/'spec.json'
    if spec_path.exists():return
    cfg=OmegaConf.load(out/'evaluation.yaml');sample_k=int(cfg.stage_b.training_samples)
    baseline=root.parent/'recovery_20261025_v1/evaluation'/case/'seed0/stage_b'
    cache_base=Path('/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v4/stage_b') if case=='case30_ieee' else baseline
    spec=dict(kind='checkpoint',source=str(source),config=str(out/'evaluation.yaml'),out=str(out/'stage_b'),baseline=str(baseline),
              cache_manifest=str(cache_base/'cache/manifest.json'),pairs=str(cache_base/'variogram_pairs.npz'),training_seed=0,
              input_sha256={str(p):sha(p) for p in (out/'adapter').iterdir() if p.is_file()},
              snapshot_record=str(out/'snapshot.json'),local_K=sample_k,
              label=f"pilot, {read(out/'snapshot.json')['epochs_observed']} epochs, not converged"+('; K=2 pilot' if sample_k==2 else '') if pilot else 'budget-capped ablation',
              headlines=str(root.parent/('recovery_20261025_v1/pilots' if pilot else 'fleet_20261025_off_v1')/case/'stage_a/stage_a_followup/headline_components.json'),
              audit=str(root/'evaluation'/case/'evaluation.json'))
    write(spec_path,spec)
    runner=root/'scheduled_operations'/('pilot_evaluate.py' if pilot else 'evaluate.py')
    for shard in range(4):
        tile=2*shard+(1 if case=='case2000_goc' else 0) if pilot else (shard+4*(0 if identity.startswith('a0') else 1 if identity.startswith('lambda01') else 2))%12
        task=dict(id=f'snapshot_{identity}_shard{shard}',category='pilot' if pilot else 'ablation',node=2 if pilot else 3+CASES.index(case),
                  slots=[tile],cwd=str(source),snapshot=str(source),run_directory=str(out),budget_seconds=7200,
                  completion=str(out/'stage_b'/f'shard_{shard}.json'),command=[PYTHON,'-u',str(runner),'--spec',str(spec_path),'--shard',str(shard)],
                  env=dict(ZE_FLAT_DEVICE_HIERARCHY='FLAT',ZE_AFFINITY_MASK=str(tile),PYTHONPATH=str(source)),
                  sha256={str(p):sha(p) for p in [runner,spec_path]})
        write(root/'tasks'/f"{task['id']}.json",task)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    schedule=read(root/'phase_d.protocol.json'); source=Path(schedule['source'])
    while time.time()<read(root/'protocol.json')['freeze_epoch']:
        try:
            if time.time()>=schedule['pilot_snapshot_epoch']:
                for case in ('case500_goc','case2000_goc'):
                    identity=f'pilot_{case}_1745';config=root.parent/'recovery_20261025_v1/pilots/configs'/f'{case}_seed0.yaml'
                    out=capture(root,identity,config,'pilot',source)
                    if out:install(root,identity,out,case,source,pilot=True)
            if time.time()>=schedule['phase_c_snapshot_epoch']:
                for path in sorted((root/'configs').glob('*_seed0.yaml')):
                    identity=path.stem;case=read(root/'tasks'/f'{identity}.json')
                    if case.get('category')!='ablation':continue
                    cfg=OmegaConf.load(path);grid=cfg.grid
                    out=capture(root,identity+'_1815',path,'ablation',root/'phase_c/source')
                    if out:install(root,identity+'_1815',out,grid,root/'phase_c/source')
            write(root/'scheduled_status.json',dict(status='watching',pid=os.getpid(),updated=time.time(),
                pilot_snapshot_epoch=schedule['pilot_snapshot_epoch'],phase_c_snapshot_epoch=schedule['phase_c_snapshot_epoch']))
        except Exception as error:write(root/'scheduled_error.json',dict(error=repr(error),at=time.time()))
        time.sleep(5)


if __name__=='__main__':main()
