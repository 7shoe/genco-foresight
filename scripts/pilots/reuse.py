"""Audit an immutable generated dataset under a new audit implementation."""
from pathlib import Path

import numpy as np
import pandas as pd

from common import read_json, sha256
from data.loader.network import load_network


def benchmark_inputs(cfg):
    from data.synthetic.generate import episode_plan
    provenance = cfg['pilot']['reuse_labels']
    original = Path(provenance['source_root'])
    if sha256(original / 'manifest.json') != provenance['manifest_sha256']:
        raise ValueError('Original label manifest changed')
    source = read_json(original / 'manifest.json')
    manifest = read_json(Path(cfg['paths']['temporal_root']) / 'manifest.json')
    for key, value in source.items():
        if manifest.get(key) != value:
            raise ValueError(f'Copied generation provenance changed: {key}')
    for name, digest in source['implementation_sha256'].items():
        if sha256(Path(provenance['source_snapshot']) / name) != digest:
            raise ValueError(f'Original generation source changed: {name}')
    net = load_network(config=cfg)
    episodes, pools = episode_plan(cfg['seed'], cfg['dataset'].get('source_blocks_one_based'), cfg['task']['split_episodes'])
    if (source['case_sha256'] != net.case_sha256 or source['generation'] != cfg['generation'] or
            source['solver_budget'] != cfg['solver']['budget'] or episodes != source['episodes'] or pools != source['source_block_pools']):
        raise ValueError('Existing labels do not match the requested case, law, split or solver policy')
    profile = Path(cfg['paths']['assets_root']) / 'profiles/eia_erco_demand_2025.csv'
    if sha256(profile) != source['profile_sha256']:
        raise ValueError('Source profile changed')
    frame = pd.read_csv(profile)
    hours = np.concatenate([np.arange(b*504, (b+1)*504) for b in pools['train']])
    values = frame.demand_mw.to_numpy(); low, high = values[hours].min(), values[hours].max()
    if low != source['train_profile_min'] or high != source['train_profile_max']:
        raise ValueError('Training-only scaling endpoints changed')
    settings = cfg['generation']
    scaled = settings['scale_low'] + (values-low)*(settings['scale_high']-settings['scale_low'])/(high-low)
    return Path(cfg['paths']['temporal_root']), manifest, net, frame, scaled
