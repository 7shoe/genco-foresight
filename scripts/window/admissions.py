"""Admit immutable completed checkpoints and shard evaluation across spare tiles."""
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


def read(p): return json.loads(Path(p).read_text()) if Path(p).exists() else {}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+'.tmp');temp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');os.replace(temp,p)


def admit(root,case,seed,source):
    run=root/'ensemble'/case/f'seed{seed}'
    state=read(root/'states'/f'ensemble_{case}_seed{seed}.json')
    if state.get('status') not in ('complete','budget_capped'): return
    out=root/'evaluation'/case/f'seed{seed}';spec_path=out/'spec.json'
    if spec_path.exists(): return
    if not (run/'adapter/best.pt').exists(): return
    train=out/'adapter';train.mkdir(parents=True,exist_ok=True)
    inputs={}
    for name in ('best.pt','manifest.json','status.json','epochs.json','manifest.training.json'):
        path=run/'adapter'/name
        if not path.exists():continue
        h=sha(path);shutil.copyfile(path,train/name);assert sha(path)==sha(train/name)==h
        (train/name).chmod(0o444);inputs[str(path)]=h
    cfg=OmegaConf.load(root/'configs'/f'{case}_seed{seed}.yaml')
    cfg.paths.training_run=str(train);cfg.paths.runs_root=str(out)
    config=out/'evaluation.yaml';config.write_text(OmegaConf.to_yaml(cfg,resolve=False))
    baseline=root.parent/'recovery_20261025_v1/evaluation'/case/'seed0/stage_b'
    cache_base=Path('/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v4/stage_b') if case=='case30_ieee' else baseline
    spec=dict(kind='checkpoint',source=str(source),config=str(config),out=str(out/'stage_b'),baseline=str(baseline),
              cache_manifest=str(cache_base/'cache/manifest.json'),pairs=str(cache_base/'variogram_pairs.npz'),input_sha256=inputs,
              training_seed=seed,admitted_utc=datetime.now(timezone.utc).isoformat(),stop=state['status'])
    spec.update(headlines=str(root.parent/'fleet_20261025_off_v1'/case/'stage_a/stage_a_followup/headline_components.json'),
                audit=str(root/'evaluation'/case/'evaluation.json'))
    write(spec_path,spec); submit_shards(root,spec_path,case,seed)


def submit_shards(root,spec_path,case,seed):
    spec=read(spec_path); runner=root/'evaluation_operations/evaluate.py'
    for shard in range(4):
        identity=f"eval_{case}_seed{seed}_shard{shard}" if spec['kind']=='checkpoint' else f"mixture_{case}_size{len(spec['members'])}_shard{shard}"
        tile=(4*(int(seed)%3)+shard)%12
        if spec['kind']=='mixture':slots=list(range(12))
        else:slots=[tile]
        task=dict(id=identity,category='evaluation',node=3+CASES.index(case),slots=slots,cwd=spec['source'],snapshot=spec['source'],
                  run_directory=spec['out'],budget_seconds=7200,completion=str(Path(spec['out'])/f'shard_{shard}.json'),
                  command=[PYTHON,'-u',str(runner),'--spec',str(spec_path),'--shard',str(shard)],
                  env=dict(ZE_FLAT_DEVICE_HIERARCHY='FLAT',ZE_AFFINITY_MASK=str(tile),PYTHONPATH=spec['source']),
                  sha256={str(p):sha(p) for p in (runner,spec_path)})
        write(root/'tasks'/f'{identity}.json',task)


def mixtures(root,case,source):
    baseline=root.parent/'recovery_20261025_v1/evaluation'/case
    for size in (2,5,8):
        members=[baseline/f'seed{s}/stage_b' if s<3 else root/'evaluation'/case/f'seed{s}/stage_b' for s in range(size)]
        if not all((m/'completion.json').exists() for m in members):continue
        out=root/'mixtures'/case/f'size{size}'; spec_path=out/'spec.json'
        if spec_path.exists():continue
        cache_base=Path('/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v4/stage_b') if case=='case30_ieee' else baseline/'seed0/stage_b'
        spec=dict(kind='mixture',source=str(source),config=str(root.parent/'recovery_20261025_v1/evaluation/configs'/f'{case}_seed0.yaml'),
                  baseline=str(baseline/'seed0/stage_b'),out=str(out),members=[str(m) for m in members],seeds=list(range(size)),
                  pairs=str(cache_base/'variogram_pairs.npz'),input_sha256={str(m/'completion.json'):sha(m/'completion.json') for m in members},
                  headlines=str(root.parent/'fleet_20261025_off_v1'/case/'stage_a/stage_a_followup/headline_components.json'),
                  audit=str(root/'evaluation'/case/'evaluation.json'))
        write(spec_path,spec);submit_shards(root,spec_path,case,0)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True);a=p.parse_args()
    freeze=read(a.root/'protocol.json')['freeze_epoch']
    while time.time()<freeze:
        try:
            for case in CASES:
                for seed in range(3,8):admit(a.root,case,seed,a.source)
                mixtures(a.root,case,a.source)
            write(a.root/'admissions_status.json',dict(pid=os.getpid(),updated=time.time(),status='watching'))
        except Exception as e:write(a.root/'admissions_error.json',dict(error=repr(e),at=time.time()))
        time.sleep(15)


if __name__=='__main__':main()
