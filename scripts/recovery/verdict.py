"""Read-only comparison of the completed case30 OFF fleet and historical ON runs."""
import argparse
from pathlib import Path
import statistics

from common import read_json, sha256, write_json
from scripts.pilots.run import GENCO, text_file
from scripts.recovery.run import FLEET


def build(root):
    from scripts.fleet.evaluate import headline, HEADLINES, MODES
    from data.loader.network import load_network
    evidence = {}
    def read(path):
        evidence[str(path)] = sha256(path)
        return read_json(path)
    scales = read(GENCO / 'runs/stage_a_v4/stage_a_followup/train_coordinate_scales.json')
    buses = read(FLEET / 'case30_ieee/stage_a/stage_a_followup/headline_components.json')['voltage_bus_ids']
    net = load_network(); indices = [net.bus_ids.tolist().index(b) for b in buses]
    runs = []
    for mode in ('ON', 'OFF'):
        for seed in range(3):
            if mode == 'OFF':
                train = FLEET / f'case30_ieee/seed{seed}/adapter'
                report = root / f'evaluation/case30_ieee/seed{seed}/stage_b/report.json'
            else:
                train = GENCO / ('runs/adapter_v4' if seed == 0 else f'runs/replication/case30_ieee/v4/seed{seed}/adapter')
                report = GENCO / ('runs/v4/stage_b/report.json' if seed == 0 else f'runs/replication/case30_ieee/v4/seed{seed}/stage_b/report.json')
            status, epochs = read(train/'status.json'), read(train/'epochs.json')
            row = dict(mode=mode, logical_seed=seed, rng_seed=42 if mode == 'ON' and seed == 0 else seed,
                       selected_selection_es=status['best_selection'], best_epoch=status['best_epoch'],
                       last_epoch=epochs[-1]['epoch'], last_epoch_selection_es=epochs[-1]['selection_energy_score'],
                       stop_reason=status.get('stop_reason'), cumulative_training_wall_seconds=status.get('cumulative_training_wall_seconds'),
                       test=None)
            if report.exists():
                result = read(report)
                modes = result['modes'] or {m: read(report.parent / f'{m}.json') for m in MODES}
                row['test'] = {m: headline(modes[m], scales, indices) for m in MODES}
            runs.append(row)
    upper = max(r['selected_selection_es'] for r in runs if r['mode']=='ON')
    lower = min(r['selected_selection_es'] for r in runs if r['mode']=='ON')
    for row in runs:
        row['gap_above_ON_max'] = max(0., row['selected_selection_es']-upper)
        row['within_or_below_ON_range'] = row['selected_selection_es'] <= upper
    complete = all(r['test'] is not None for r in runs)
    passed = all(r['within_or_below_ON_range'] for r in runs if r['mode']=='OFF')
    confirmation = GENCO / 'runs/v2/stage1_confirmation_20261025_v3'
    conf = dict(gate=read(confirmation/'gate.json'), epochs={c:read(confirmation/c/'epochs.json') for c in ('off_a','off_b')})
    result = dict(schema_version=1, complete=complete, adopted_under_requested_rule=passed,
                  verdict='OFF within or below ON range' if passed else 'OFF exceeds historical ON range; gap recorded',
                  ON_selected_selection_range=[lower,upper], runs=runs, confirmation=conf,
                  comparison_voltage_bus_ids=buses, source_sha256=evidence)
    if complete:
        reference = [r['test']['A']['joint_es'] for r in runs if r['mode']=='ON']
        result['ON_test_A_range'] = [min(reference), max(reference)]
        result['OFF_test_A_gaps_above_ON_max'] = {str(r['logical_seed']): max(0., r['test']['A']['joint_es']-max(reference))
                                                for r in runs if r['mode']=='OFF'}
    write_json(root/'stage1_verdict.json', result)
    lines=['# Stage 1 fleet verdict','',result['verdict']+'.', '',
           'The requested rule has no added slack: every OFF selected-checkpoint selection ES must be at or below the maximum historical ON value. Test results provide context; they do not change that rule.', '',
           f'Historical ON selected-checkpoint range: [{lower:.9f}, {upper:.9f}]. All six training runs are complete. '+('Test evaluation is complete.' if complete else 'OFF test evaluation is still pending; this report is provisional only for the test table.'), '',
           '| Mode | Logical seed (RNG seed) | Selected ES | Best epoch | Last epoch | Last-epoch ES | Gap above ON maximum |',
           '|---|---|---:|---:|---:|---:|---:|']
    for r in runs:
        lines.append(f"| {r['mode']} | {r['logical_seed']} ({r['rng_seed']}) | {r['selected_selection_es']:.9f} | {r['best_epoch']} | {r['last_epoch']} | {r['last_epoch_selection_es']:.9f} | {r['gap_above_ON_max']:.9f} |")
    groups = {mode: [r['selected_selection_es'] for r in runs if r['mode']==mode] for mode in ('ON', 'OFF')}
    lines += ['', 'Selected-checkpoint means ± sample SD: '+ '; '.join(
        f'{mode} {statistics.mean(values):.9f} ± {statistics.stdev(values):.9f}' for mode, values in groups.items())+
        '. The OFF mean is below the ON maximum, but two individual OFF seeds exceed it; averaging does not hide those gaps.']
    lines += ['', '“Selected ES” is selection ES at best.pt, the checkpoint used for test evaluation. Last-epoch ES is shown separately. Historical logical seed0 used RNG42; the new fleet used literal RNG0. This is a historical three-seed comparison, not a paired experiment isolating determinism.', '',
              'Historical ON seeds0/2 early-stopped at epoch17; ON seed1 reached its six-hour budget at epoch19. OFF seeds0/1/2 early-stopped at epochs25/19/21. Thus the common epoch19 endpoint for seed1 has different stopping reasons. The earlier stopping-epoch ±1 proposal is not part of this verdict.', '',
              f'All test headline voltage metrics below are recomputed from the stored frozen-scorer component summaries using the same mechanically selected bus IDs {buses}; historical headline bus lists are not mixed with the new list. Joint ES and free-generator masks/scales are unchanged. C/P/P0/S use the released reference forecasts.', '',
              '| Mode | Seed | C ES | P ES | P0 ES | S ES | A ES | D ES |', '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in runs:
        values = [f"{r['test'][m]['joint_es']:.9f}" if r['test'] else 'pending' for m in MODES]
        lines.append(f"| {r['mode']} | {r['logical_seed']} | "+' | '.join(values)+' |')
    if complete:
        lines += ['', f"Historical test A ES range: {result['ON_test_A_range']}. OFF gaps above its maximum by seed: " +
                  ', '.join(f'{seed}: {gap:.9f}' for seed, gap in result['OFF_test_A_gaps_above_ON_max'].items())+'.']
        lines += ['', '| Forecast | Metric | ON mean ± sample SD | OFF mean ± sample SD |', '|---|---|---:|---:|']
        for m in MODES:
            for metric in HEADLINES:
                cells=[]
                for mode in ('ON','OFF'):
                    v=[r['test'][m][metric] for r in runs if r['mode']==mode]
                    cells.append(f'{statistics.mean(v):.9g} ± {statistics.stdev(v):.9g}' if all(x is not None for x in v) else 'undefined')
                lines.append(f'| {m} | {metric} | '+' | '.join(cells)+' |')
    observed=sum(len(v) for v in conf['epochs'].values())
    lines += ['', f'Confirmation job8962843 is contextual only. Six epoch comparisons were planned; {observed} are persisted. OFF B epoch2 recorded0.484601428 versus the fixed upper threshold0.463022624, a gap0.021578804. The controller stopped at that failure, so epoch3 was not observed. The one-of-six wording refers to the planned comparisons and does not establish five observed passes.', '',
              'No historical manifest, checkpoint or result was rewritten. No new training or acceptance threshold was introduced. This report does not reverse prior operational authorization to run OFF; it records that the latest requested historical-range criterion is not met.', '',
              f'Bulk evidence and all input SHA256 hashes: [{root.name}/stage1_verdict.json]({root}/stage1_verdict.json).',
              f'Protocol: [recovery.protocol.json](recovery.protocol.json). Report data SHA256: `{sha256(root / "stage1_verdict.json")}`.']
    text_file(root/'stage1_verdict.md','\n'.join(lines)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve()
    if not root.is_relative_to(GENCO/'runs/v2') or root==FLEET:raise ValueError('Verdict needs a new recovery output root')
    print(build(root)['verdict'])
