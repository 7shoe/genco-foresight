"""Seeded subset parity with immutable IDs and exact CPU output equality."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import yaml

from power_fm_foresight._config import ROOT, load_config, read_json, sha256, write_json
from power_fm_foresight._data.assets import checkpoint_assets
from power_fm_foresight._data.dataset import SnapshotDataset, load_static_splits, snapshot_tables
from power_fm_foresight._data.network import load_network
from experiments.parity import evaluate


def serial_environment(environment):
    return {k: v for k, v in environment.items()
        if not k.startswith(('PMI_', 'PMIX_', 'PALS_', 'OMPI_', 'SLURM_'))
        and k not in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'NODE_RANK', 'MASTER_ADDR', 'MASTER_PORT')}


def selected_ids(intersection, seed=42):
    if len(intersection) < 200:raise ValueError('Fewer than200 official/local test-intersection IDs')
    return np.sort(np.random.default_rng(seed).choice(intersection, 200, replace=False))


def record(cfg, destination):
    assets = Path(cfg['paths']['assets_root']) / 'genco-opfdata'
    local_path = Path(cfg['paths']['n1_splits']) / 'splits.npz'
    official_path = assets / 'splits/test.pt'
    local = load_static_splits(local_path)
    official = np.asarray([int(i) for i in torch.load(official_path, weights_only=True)])
    intersection = np.intersect1d(local['test'], official)
    protocol = dict(schema_version=1, label='subset parity (200)', seed=42,
        sampling='NumPy default_rng(42).choice(sorted intersection,200,replace=False); sorted for execution',
        canonical_ids=selected_ids(intersection).tolist(), count=200, intersection_count=len(intersection),
        local_splits_sha256=sha256(local_path), official_test_sha256=sha256(official_path),
        input_rtol=2e-6,input_atol=1e-4,output_rtol=0.,output_atol=0.,batch_size=1,
        precision='fp32',device='cpu',cap_seconds=1200,pass_rule='finite predictions, identical dtype and bytes',
        conversion='Retain the existing32-snapshot independent official-parquet conversion check with its hashes and original IDs.',
        reference='Unmodified pinned Graphkit evaluate CLI, CPU, no inherited MPI rank context',
        case=cfg['checkpoint']['directory'].split('/')[0],recorded_before_execution=True)
    if destination.exists() and read_json(destination) != protocol:raise ValueError('Subset protocol changed')
    write_json(destination,protocol)
    return protocol


def prepare(config, protocol, conversion_source):
    grid = load_network(config=config)
    cache = SnapshotDataset(config["paths"]["n1_cache"], grid=grid)
    assets = checkpoint_assets(config)
    root = Path(config["paths"]["runs_root"]) / "stage_a/parity"
    root.mkdir(parents=True, exist_ok=True)
    ids = np.asarray(protocol['canonical_ids'], dtype=np.int64)
    spot_count = 0
    batch_size = 1
    write_json(root / 'protocol.json', protocol)
    selected_groups = {int(i)//cache.manifest["group_size"] for i in ids}
    for record in cache.manifest["groups"]:
        if record["group"] not in selected_groups:continue
        path = cache.folder / f"group_{record['group']}.pt"
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Published OPFData cache checksum mismatch: {path}")
    raw = root / "data" / config["checkpoint"]["directory"].split("/")[0] / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    tables = [[], [], []]
    selected_tables = {}
    for index, canonical in enumerate(ids):
        converted = snapshot_tables(cache[int(canonical)])
        if index < spot_count:
            selected_tables[int(canonical)] = tuple(t.copy() for t in converted)
        for destination, table in zip(tables, converted):
            table["scenario"] = index
            destination.append(table)
    for name, parts in zip(("bus", "gen", "branch"), tables):
        table = pd.concat(parts, ignore_index=True)
        if name == "bus":
            table = table.drop(columns=["min_q_mvar", "max_q_mvar"])
        table.to_parquet(raw / f"{name}_data.parquet", index=False)
    split_root = root / "splits"
    split_root.mkdir(exist_ok=True)
    for split in ("train", "val", "test"):
        values = torch.arange(len(ids), dtype=torch.long) if split == "test" else torch.empty(0, dtype=torch.long)
        torch.save(values, split_root / f"{split}.pt")
    original = assets / config["checkpoint"]["directory"] / config["checkpoint"]["configuration"]
    settings = yaml.safe_load(original.read_text())
    settings["data"].update(workers=0, scenarios=[len(ids)], split_from_existing_files=str(split_root))
    settings["training"].update(accelerator="cpu", devices=1, strategy="auto", batch_size=batch_size)
    settings["verbose"] = False
    (root / "evaluate.yaml").write_text(yaml.safe_dump(settings, sort_keys=False))
    conversion = read_json(conversion_source / 'conversion_check.json')
    if not conversion['passed']:raise ValueError('Independent conversion did not pass')
    conversion['reused_evidence'] = {'path': str(conversion_source),
        'report_sha256': sha256(conversion_source / 'conversion_check.json'),
        'protocol_sha256': sha256(conversion_source / 'protocol.json')}
    write_json(root / 'conversion_check.json', conversion)
    return root


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--record-only',action='store_true')
    args=parser.parse_args();cfg=load_config(args.config)
    protocol_path=args.output/'protocol.json'
    if args.record_only:
        print(record(cfg,protocol_path));return
    if not os.environ.get('PBS_JOBID'):raise RuntimeError('CPU parity requires a compute allocation')
    protocol=read_json(protocol_path)
    if protocol != record(cfg,protocol_path):raise ValueError('Changed IDs')
    for key in set(os.environ)-set(serial_environment(os.environ)):os.environ.pop(key)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    old=Path(cfg['paths']['stage_a_run'])/'parity'
    cfg=deepcopy(cfg);cfg['paths']['runs_root']=str(args.output)
    began=time.time()
    prepare(cfg,protocol,old)
    evaluate(cfg)
    report=read_json(args.output/'stage_a/parity/parity_report.json')
    passed=bool(report['passed'] and report['provenance']['bit_identical'] and time.time()-began <=1200)
    gate=dict(passed=passed,label='subset parity (200)',protocol_sha256=sha256(protocol_path),
        report_sha256=sha256(args.output/'stage_a/parity/parity_report.json'),
        wall_seconds=time.time()-began,bit_identical=report['provenance']['bit_identical'],
        job_id=os.environ['PBS_JOBID'],host=os.uname().nodename)
    write_json(args.output/'gate.json',gate)
    if not passed:raise RuntimeError('Subset parity gate failed')
    print(gate,flush=True)


if __name__=='__main__':main()
