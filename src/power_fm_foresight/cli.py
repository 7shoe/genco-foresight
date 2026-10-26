"""Five local commands over the package workflows; scheduler-free by design."""
import argparse
from pathlib import Path

CASES = (14, 30, 57, 118, 500, 2000)
MODES = ('C', 'P', 'P0', 'S', 'A', 'D')


def positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return result


def nonnegative(value):
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError('must be a nonnegative integer')
    return result


def parser():
    p = argparse.ArgumentParser(prog='pff', description=__doc__)
    p.add_argument('--config', type=Path, default=argparse.SUPPRESS)
    verbs = p.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'train', 'evaluate', 'report', 'verify'):
        sub = verbs.add_parser(name, argument_default=argparse.SUPPRESS)
        sub.add_argument('--config', type=Path)
        if name in ('prepare', 'train'):
            sub.add_argument('--case', type=int, choices=CASES, nargs='+')
            sub.add_argument('--dataset', type=Path)
        if name == 'prepare':
            sub.add_argument('--recipe', choices=('v4',))
            sub.add_argument('--reuse', type=Path)
            sub.add_argument('--audit-only', action='store_true')
            sub.add_argument('--workers', type=positive)
            sub.add_argument('--parity-samples', type=positive)
            sub.add_argument('--device', choices=('cpu',))
        elif name == 'train':
            sub.add_argument('--seed', type=nonnegative, nargs='+')
            sub.add_argument('--recipe', choices=('v4-off', 'v4-on'))
            sub.add_argument('--variant', choices=('conditioned', 'a0'))
            sub.add_argument('--balance-weight', type=float, nargs='+')
            sub.add_argument('--samples', type=positive)
            sub.add_argument('--lr', type=float)
            sub.add_argument('--max-epochs', type=positive)
            sub.add_argument('--patience', type=positive)
            sub.add_argument('--chain-budget')
            sub.add_argument('--label')
            sub.add_argument('--device', choices=('cpu', 'xpu', 'cuda'))
            sub.add_argument('--device-index', type=nonnegative)
            sub.add_argument('--out', type=Path)
            sub.add_argument('--resume', type=Path)
            sub.add_argument('--parallel', type=positive)
            sub.add_argument('--dry-run', action='store_true')
        elif name == 'evaluate':
            sub.add_argument('--runs', type=Path, nargs='+')
            sub.add_argument('--checkpoint', choices=('best', 'latest'))
            sub.add_argument('--partition', choices=('test', 'selection'))
            sub.add_argument('--modes', choices=MODES, nargs='+')
            sub.add_argument('--samples', type=positive)
            sub.add_argument('--eval-seed', type=nonnegative)
            sub.add_argument('--ensemble-sizes', type=positive, nargs='+')
            sub.add_argument('--shards', type=positive)
            sub.add_argument('--shard', type=nonnegative)
            sub.add_argument('--allow-budget-capped', action='store_true')
            sub.add_argument('--device')
            sub.add_argument('--out', type=Path)
        elif name == 'report':
            sub.add_argument('--runs', type=Path, nargs='+')
            sub.add_argument('--kind', choices=('status', 'summary', 'ensemble', 'comparison'))
            sub.add_argument('--reference', type=Path)
            sub.add_argument('--format', choices=('md', 'json', 'csv', 'png'), nargs='+')
            sub.add_argument('--out', type=Path)
        else:
            sub.add_argument('--check', choices=('reproducibility', 'regression'), required=True)
            sub.add_argument('--protocol', type=Path, required=True)
            sub.add_argument('--runs', type=Path, nargs='+')
            sub.add_argument('--out', type=Path)
    return p


def main(argv=None, *, stop_after_steps=None):
    """Dispatch parsed arguments; protocols may request an existing checkpoint boundary."""
    p = parser()
    args = vars(p.parse_args(argv))
    try:
        from .campaign import command_options, prepare_campaign, resume_campaign, run_campaign
        options = command_options(args)
        command = args['command']
        if command == 'train':
            campaign = resume_campaign(options, args) if 'resume' in args else prepare_campaign(options)
            import json
            print(json.dumps(campaign['plan'], indent=2), flush=True)
            if not options.get('dry_run', False):
                run_campaign(campaign, parallel=options['parallel'], stop_after_steps=stop_after_steps)
        elif command == 'prepare':
            from .preparation import prepare_datasets
            prepare_datasets(options)
        elif command == 'evaluate':
            from .assessment import evaluate_runs
            evaluate_runs(options)
        elif command == 'report':
            from .reporting import report_runs
            report_runs(options)
        else:
            from .verification import verify
            if not verify(options)['passed']:
                return 1
    except (ValueError, FileNotFoundError) as error:
        p.error(str(error))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
