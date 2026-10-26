"""Read-only checkpoint, loader and scorer equivalence on three recorded origins."""
import argparse
from pathlib import Path
import gc
import os

import numpy as np
import torch

from power_fm_foresight.data import load_config
from power_fm_foresight._config import read_json, sha256, write_json
from power_fm_foresight.training import components, adapter_samples, frozen, score
from experiments.replication_evaluate import differences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('PBS_JOBID'), 'Model verification requires a compute allocation'
    protocol = read_json(args.root / 'protocol.json')['gates']['scorer_loader']
    cfg = load_config(args.source / 'configs/v4.yaml')
    bulk = Path(cfg['paths']['assets_root']).parent
    reference = bulk / 'runs/v4/stage_b'
    origins = read_json(reference / 'protocol.json')['origins'][:3]
    assert [o['id'] for o in origins] == protocol['origins']
    cache = read_json(reference / 'cache/manifest.json')
    with np.load(reference / 'variogram_pairs.npz') as f:
        pairs = f['pairs']
    configs = [cfg, load_config(bulk / 'runs/v2/fleet_20261025_off_v1/configs/case30_ieee_seed0.yaml')]
    results = []
    torch.xpu.set_device(0)
    for cfg, checkpoint, saved in zip(configs, protocol['checkpoints'], protocol['saved_evaluations']):
        assert Path(cfg['paths']['training_run']) / 'best.pt' == bulk / checkpoint
        hashes = {str(bulk / checkpoint): sha256(bulk / checkpoint)}
        net, base, model, scales, templates = components(cfg, 'xpu:0')
        for index, origin in enumerate(origins):
            item = cache[origin['episode_id']]
            assert sha256(item['path']) == item['sha256']
            hashes[item['path']] = item['sha256']
            cached = torch.load(item['path'], mmap=True, weights_only=True)['data']
            values, seconds = adapter_samples(base, model, net, cached, origin['origin'], templates,
                                               protocol['forecast_seed'] + index, 'xpu:0')
            modes = {'A': values, 'D': frozen(base, net, values['pd_mw'], templates, qd=values['qd_mvar'])}
            folder = bulk / saved / 'origins' / origin['id']
            with np.load(folder / 'targets.npz') as f:
                target = {k: f[k] for k in f.files}
            hashes[str(folder / 'targets.npz')] = sha256(folder / 'targets.npz')
            for mode, prediction in modes.items():
                actual, arrays = score(prediction, target, scales, pairs, mode, origin['id'], index)
                saved_path = folder / f'{mode}_scores.json'
                expected = read_json(saved_path)
                hashes[str(saved_path)] = sha256(saved_path)
                check = differences(actual, expected, atol=protocol['absolute_tolerance'], rtol=0)
                result = dict(checkpoint=checkpoint, origin=origin['id'], mode=mode,
                              forecast_seconds=seconds, **check)
                results.append(result)
                write_json(args.root / 'scorer_loader/progress.json', dict(results=results, input_sha256=hashes))
                write_json(args.root / 'scorer_loader' / Path(checkpoint).parts[-3] / origin['id'] / f'{mode}.json', actual)
                assert check['passed'], result
        assert all(sha256(p) == h for p, h in hashes.items()), 'Read-only input changed'
        del model, base, templates, cached, values, modes
        gc.collect()
        torch.xpu.empty_cache()
    write_json(args.root / 'scorer_loader/gate.json', dict(passed=True, comparisons=results))


if __name__ == '__main__':
    main()
