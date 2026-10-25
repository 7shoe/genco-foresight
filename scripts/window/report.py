"""Describe completed fleet, ensemble-depth, ablation and pilot evidence."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import statistics


def read(p):return json.loads(Path(p).read_text()) if Path(p).exists() else {}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fmt(v):return '—' if v is None else f'{v:.7g}'


def report(root):
    cases=('case14_ieee','case30_ieee','case57_ieee','case118_ieee');metrics=('joint_es','vm_crps','vm_coverage90_pct','p_imbalance')
    recovery=root.parent/'recovery_20261025_v1/evaluation';freeze=read(root/'protocol.json')['freeze_epoch']
    rows=[];hashes={}
    for case in cases:
        eligible=[0,1,2]
        for seed in range(3,8):
            state=read(root/'states'/f'ensemble_{case}_seed{seed}.json')
            if state.get('status') in ('complete','budget_capped') and state['attempts'][-1].get('ended',float('inf'))<=freeze:eligible.append(seed)
        for size in (1,2,3,5,8):
            path=(recovery/case/'seed0/stage_b/report.json' if size==1 else recovery/case/'mixture/report.json' if size==3 else root/'mixtures'/case/f'size{size}/report.json')
            completion=path.with_name('completion.json');value=read(path);members=list(range(size))
            ready=completion.exists() and all(s in eligible for s in members)
            h=value.get('headlines',{}).get('A',value.get('modes',{}).get('A',{})) if ready else {}
            full=value.get('modes',{}).get('A',{}) if ready else {}
            if ready and 'variables' not in full:full=read(path.with_name('A.json'))
            voltage=full.get('variables',{}).get('vm',{})
            rows.append(dict(case=case,size=size,seeds=members,finished_seeds=eligible,samples=64*size,
                             status='complete' if ready else 'not available at cutoff' if len(eligible)<size else 'evaluation pending',
                             metrics={m:h.get(m) for m in metrics},path=str(path),
                             all_bus_vm_crps=voltage.get('crps'),
                             all_bus_vm_coverage90_pct=100*voltage['coverage_90'] if voltage.get('coverage_90') is not None else None))
            if ready:hashes[str(path)]=sha(path)
    lines=['# Mixture-size curves','',f'Updated {datetime.now(timezone.utc).isoformat()}. Training eligibility cutoff: 2026-10-25 19:00 CT.',
           '', 'Nested seed prefixes 0..m−1, 64 draws per seed, equal weights, jointly rescored with every cross-seed pair. Samples share the existing fixed-noise convention across checkpoints; this is a descriptive empirical mixture curve. Missing sizes are not interpolated. ES scales/masks differ across grids.',
           '', '| Grid | Size | Finished seeds | ES | Vm CRPS | Coverage90 % | P imbalance MW | Status |','|---|---:|---|---:|---:|---:|---:|---|']
    for r in rows:lines.append(f"| {r['case']} | {r['size']} | {','.join(map(str,r['finished_seeds']))} | "+' | '.join(fmt(r['metrics'][m]) for m in metrics)+f" | {r['status']} |")
    lines+=['','Vm columns above retain the mechanical headline-bus rule; case14 has no eligible headline buses, so those values are undefined. The figure uses all-bus Vm CRPS/coverage for every grid, consistently, as an additional diagnostic. This does not change headline selection, training masks or scales.','',
            '| Grid | Size | All-bus Vm CRPS | All-bus coverage90 % |','|---|---:|---:|---:|']
    lines += [f"| {r['case']} | {r['size']} | {fmt(r['all_bus_vm_crps'])} | {fmt(r['all_bus_vm_coverage90_pct'])} |" for r in rows]
    (root/'mixture_sizes.md').write_text('\n'.join(lines)+'\n')
    (root/'mixture_sizes.json').write_text(json.dumps(dict(rows=rows,input_sha256=hashes,freeze_epoch=freeze),indent=2)+'\n')
    text=(recovery/'cross_grid.md').read_text()
    text+='\n## Budget-capped A0 and λ ablations\n\nThese rows use one seed and a 3.3-hour worker cap; the fleet has a different training budget. They are separate descriptive experiments, not three-seed fleet estimates or an adoption gate. λ=1 uses the existing case30 seed0 fleet checkpoint.\n\n| Run | Mode | Epochs observed / best | ES | Vm CRPS | Coverage90 % | P imbalance MW |\n|---|---|---|---:|---:|---:|---:|\n'
    ablations=[];pilots=[]
    for folder in sorted((root/'snapshots').glob('*')):
        s=read(folder/'snapshot.json');r=read(folder/'stage_b/report.json')
        if not s:continue
        row=dict(run=folder.name,snapshot=s,complete=(folder/'stage_b/completion.json').exists(),report=r)
        (pilots if s['category']=='pilot' else ablations).append(row)
        if s['category']!='ablation' or not row['complete']:continue
        for mode in ['A','D']:
            h=r['headlines'][mode]
            text+=f"| {folder.name} | {mode} | {s['epochs_observed']} / {s['best_epoch']} | "+' | '.join(fmt(h.get(m)) for m in metrics)+' |\n'
    base=read(recovery/'case30_ieee/seed0/stage_b/report.json')['headlines']['A']
    text+='| case30 λ=1 fleet seed0 reference | A | 25 / 22 | '+' | '.join(fmt(base.get(m)) for m in metrics)+' |\n'
    text+='\n[Mixture-size table](mixture_sizes.md) · [Mixture-size figure](mixture_sizes.png). Pilots are excluded from this cross-grid table.\n'
    (root/'cross_grid.md').write_text(text)
    (root/'cross_grid.json').write_text(json.dumps(dict(fleet=read(recovery/'cross_grid.json'),ablations=ablations,mixtures=rows),indent=2)+'\n')
    (root/'pilot_results.json').write_text(json.dumps(pilots,indent=2)+'\n')
    os.environ['MPLCONFIGDIR']=str(root/'tmp/matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,4,figsize=(15,11),squeeze=False)
    for i,case in enumerate(cases):
        selected=[r for r in rows if r['case']==case and r['status']=='complete']
        for j,metric in enumerate(metrics):
            y=[r['all_bus_vm_crps'] if j==1 else r['all_bus_vm_coverage90_pct'] if j==2 else r['metrics'][metric] for r in selected]
            ax=axes[i,j];ax.plot([r['size'] for r in selected],y,'o-')
            ax.set_xticks([1,2,3,5,8]);ax.set_xlim(.7,8.3);ax.grid(alpha=.25)
            if i==0:ax.set_title(('all-bus ' if j in (1,2) else '')+metric)
            if j==0:ax.set_ylabel(case)
            if i==3:ax.set_xlabel('Mixture size')
    fig.suptitle('Equal-weight mixtures: finished seeds only; missing sizes unmeasured')
    fig.tight_layout();fig.savefig(root/'mixture_sizes.png',dpi=160);fig.savefig(root/'mixture_sizes.pdf');plt.close(fig)
    if pilots:
        fig,ax=plt.subplots(figsize=(7,4))
        for p in pilots:
            epochs=p['snapshot']['epochs'];ax.plot([e['epoch'] for e in epochs],[e['selection_energy_score'] for e in epochs],'o-',label=p['run'])
        ax.set(xlabel='Completed epoch',ylabel='Selection ES',title='Pilots: best-so-far snapshot, not converged');ax.legend();ax.grid(alpha=.25)
        fig.tight_layout();fig.savefig(root/'pilot_selection_curves.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();report(a.root)
