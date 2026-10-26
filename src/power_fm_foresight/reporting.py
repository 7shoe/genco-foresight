"""Format persisted status and scores without loading a model or invoking a solver."""
import csv
import json
from pathlib import Path
import statistics

from ._config import read_json, write_json
from .campaign import default_output, record_manifest, resolve_runs

METRICS=('joint_es','vm_crps','vm_coverage90_pct','pg_free_mae','p_imbalance')


def persisted(path):
    path=Path(path)
    if (path/'evaluation.json').exists():return read_json(path/'evaluation.json')
    if (path/'report.json').exists():
        row=read_json(path/'report.json');return dict(runs=[row],mixtures=[])
    for child in (path/'evaluation',path/'stage_b'):
        if (child/'report.json').exists():return dict(runs=[read_json(child/'report.json')],mixtures=[])
    if (path/'evaluations.json').exists():
        result=persisted(Path(read_json(path/'evaluations.json')['latest']))
        selected=[r for r in result['runs'] if Path(r['identity']['run']).resolve()==path.resolve()]
        return dict(runs=selected,mixtures=[])
    if (path/'campaign.json').exists():
        groups=[persisted(p) for p in resolve_runs([path])]
        return dict(runs=[x for g in groups for x in g['runs']],mixtures=[x for g in groups for x in g.get('mixtures',[])])
    raise FileNotFoundError(f'No persisted evaluation at {path}; pass its --out directory')


def status_rows(paths):
    rows=[]
    for path in resolve_runs(paths):
        adapter=path/'adapter' if (path/'chain.json').exists() else path
        status=read_json(adapter/'status.json') if (adapter/'status.json').exists() else {}
        chain=read_json(path/'chain.json') if (path/'chain.json').exists() else {}
        rows.append(dict(run=str(path),status=status.get('status','not started'),epochs=status.get('epochs_reached'),
            steps=status.get('steps'),best_epoch=status.get('best_epoch'),selection_es=status.get('best_selection'),
            stop_reason=status.get('stop_reason'),pilot=chain.get('pilot',False),label=chain.get('label',''),
            budget_capped=status.get('stop_reason') in ('training_budget','six_hour_budget')))
    return rows


def summary_rows(results,kind):
    rows=[];groups={};seen=set();seen_mixtures=set()
    for result in results:
        for run in result['runs']:
            fingerprint=json.dumps(run,sort_keys=True)
            if fingerprint in seen:continue
            seen.add(fingerprint)
            identity=run.get('identity',{})
            if not run.get('complete',True):continue
            for mode,metrics in run.get('headlines',{}).items():
                row=dict(row_type='seed',**identity,mode=mode,**(metrics or {k:None for k in METRICS}))
                if kind!='ensemble':rows.append(row)
                key=tuple(identity.get(k) for k in ('case','variant','balance_weight','samples','lr','pilot','budget_capped'))+(mode,json.dumps(run.get('comparison',{}),sort_keys=True))
                groups.setdefault(key,[]).append(row)
        for mixture in result.get('mixtures',[]):
            fingerprint=json.dumps(mixture,sort_keys=True)
            if fingerprint in seen_mixtures:continue
            seen_mixtures.add(fingerprint)
            if not mixture.get('available',False):
                rows.append(dict(row_type='mixture',size=mixture['size'],available=False,reason=mixture['reason']));continue
            for mode,metrics in mixture['headlines'].items():
                identity={k:v for k,v in mixture['identity'].items() if k not in ('seed','run','best_epoch','epochs')}
                rows.append(dict(row_type='mixture',**identity,size=mixture['size'],seeds=mixture['seeds'],mode=mode,
                    **(metrics or {k:None for k in METRICS})))
    if kind!='ensemble':
        for values in groups.values():
            row={k:v for k,v in values[0].items() if k not in METRICS and k not in ('seed','run','best_epoch','epochs')}
            row.update(row_type='mean_sd',n=len(values),seeds=[v.get('seed') for v in values])
            for metric in METRICS:
                xs=[v[metric] for v in values]
                row[metric]=statistics.mean(xs) if all(x is not None for x in xs) else None
                row[metric+'_sd']=statistics.stdev(xs) if len(xs)>1 and all(x is not None for x in xs) else None
            rows.append(row)
    return rows


def render(rows,formats,out,title):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    if 'json' in formats:write_json(out/'report.json',dict(title=title,rows=rows))
    if 'csv' in formats:
        with (out/'report.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    if 'md' in formats:
        def cell(value):
            if value is None:return '—'
            if isinstance(value,float):return f'{value:.6g}'
            return str(value).replace('|','\\|').replace('\n',' ')
        lines=[f'# {title}','','Seed, mean ± sample SD, and jointly rescored mixture rows are separate. Unavailable cells remain unavailable. Pilots and budget-capped runs are labelled.','',
               '| '+' | '.join(keys)+' |','| '+' | '.join('---' for _ in keys)+' |']
        for row in rows:
            cells=[]
            for key in keys:
                value=cell(row.get(key))
                if row.get('row_type')=='mean_sd' and key in METRICS and row.get(key) is not None:
                    value+=' ± '+cell(row.get(key+'_sd'))
                cells.append(value)
            lines.append('| '+' | '.join(cells)+' |')
        (out/'report.md').write_text('\n'.join(lines)+'\n')
    if 'png' in formats:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        valid=[r for r in rows if r.get('joint_es') is not None]
        fig,ax=plt.subplots(figsize=(max(7,len(valid)*.3),4))
        if valid:
            ax.bar(range(len(valid)),[r['joint_es'] for r in valid])
            ax.set_xticks(range(len(valid)),[f"{r.get('case','')} {r.get('mode','')} {r.get('row_type','')} {r.get('seed',r.get('size',''))}" for r in valid],rotation=90)
        else:ax.text(.5,.5,'No available persisted ES',ha='center',transform=ax.transAxes)
        ax.set_ylabel('Joint ES');ax.set_title(title);fig.tight_layout();fig.savefig(out/'report.png',dpi=180);plt.close(fig)


def report_runs(options):
    paths=options.get('runs')
    if not paths:raise ValueError('report requires --runs')
    out=Path(options.get('out') or default_output('report'))
    record_manifest(out,'report',options,device='none (persisted results only)',seeds=[],paths=dict(inputs=paths,output=out))
    kind=options['kind']
    if kind=='status':rows=status_rows(paths)
    else:rows=summary_rows([persisted(p) for p in paths],kind)
    if kind=='comparison':
        if not options.get('reference'):raise ValueError('comparison requires --reference')
        reference=summary_rows([persisted(options['reference'])],'summary')
        for row in rows:row['role']='candidate'
        for row in reference:row['role']='reference'
        rows+=reference
    render(rows,options['format'],out,f'pff {kind}')
    return rows
