"""Generate each grid's v4-law data and publish readiness only after its audits."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from power_fm_foresight._config import ROOT, load_config, read_json, sha256, write_json
from experiments.fleet.run import config_file


def command(module, cfg_path, log, *args):
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, GENCO_CONFIG=str(cfg_path))
    with log.open('a') as stream:
        subprocess.run([sys.executable, '-u', '-m', module, *args], cwd=ROOT, env=env,
                       stdout=stream, stderr=subprocess.STDOUT, check=True,
                       timeout=max(1, float(os.environ['GENCO_FLEET_DEADLINE'])-time.time()))


def headlines(events):
    return dict(voltage_bus_ids=[r['id'] for r in events['buses'] if r['calibration_hourly_support']['eligible']],
        thermal_line_ids=[r['id'] for r in events['lines'] if r['rated'] and r['calibration_hourly_support']['eligible']],
        selection='Risk-calibration hourly support only: >=5 positive days, >=5 negative days, prevalence <0.95; retain all eligible components; no test-based selection or fallback.')


def prepare_grid(root, case, cfg):
    import torch
    torch.use_deterministic_algorithms(True)
    grid = root / case; audit = Path(cfg['paths']['stage_a_run']); audit.mkdir(parents=True, exist_ok=True)
    cfg_path = config_file(root, case)
    deadline = float(os.environ['GENCO_FLEET_DEADLINE'])
    original = load_config(ROOT / 'configs/v4.yaml')
    if case == 'case30_ieee':
        source = Path(original['paths']['stage_a_run'])
        readiness = read_json(source / 'readiness.json')
        assert readiness['stage_a_audit_passed'] and readiness['training_allowed']
        assert readiness['manifest_sha256'] == sha256(Path(cfg['paths']['temporal_root']) / 'manifest.json')
        for name in ('readiness.json', 'stage_a.json', 'stage_b_prevalence.json', 'stage_b_prevalence.md',
                     'support_diagnostic.json', 'stage_a_followup/events.json', 'stage_a_followup/train_coordinate_scales.json'):
            src = source / name
            if src.exists():
                dst = audit / name; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(src, dst)
        write_json(audit / 'reuse.json', dict(read_only_source=str(source),
            source_sha256={str(p): sha256(p) for p in source.rglob('*.json') if p.is_file()}))
        cache = grid / 'shared/cache'; cache.mkdir(parents=True, exist_ok=True)
        source_cache = Path(original['paths']['training_run']) / 'cache/manifest.json'
        shutil.copyfile(source_cache, cache / 'manifest.json')
    else:
        if not (audit / 'environment_before.txt').exists():
            (audit / 'environment_before.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
        parity = audit / 'parity'
        if not (parity / 'conversion_check.json').exists():
            command('scripts.parity', cfg_path, audit / 'logs/parity_prepare.log', '--config', str(cfg_path), '--prepare', '--download')
        if time.time() > deadline-180:
            return
        if not (parity / 'parity_report.json').exists():
            command('scripts.parity', cfg_path, audit / 'logs/parity_evaluate.log', '--config', str(cfg_path))
        assert read_json(parity / 'parity_report.json')['passed']
        if not (audit / 'solver_smoke.json').exists():
            command('scripts.solver_smoke', cfg_path, audit / 'logs/solver_smoke.log', '--config', str(cfg_path))
        smoke = read_json(audit / 'solver_smoke.json')
        assert smoke['cold']['checks']['valid'] and smoke['warm']['checks']['valid'] and smoke['dual_check']['valid']
        from experiments.data_generation import label
        data = Path(cfg['paths']['temporal_root'])
        if not (data / 'pilot_report.json').exists():
            result = label(cfg, 200, deadline_epoch=deadline-120)
            if result.get('incomplete'):
                return
            assert result['accepted']
        if not (data / 'quality_report.json').exists():
            result = label(cfg, 10080, deadline_epoch=deadline-120)
            if result.get('incomplete'):
                return
            assert result['attempted'] == 10080
            from experiments.data_audit import audit as audit_data
            audit_data(cfg)
        if time.time() > deadline-180:
            return
        if not (audit / 'stage_a_followup/events.json').exists():
            from experiments.variant_events import run as event_audit
            event_audit(cfg)
            from experiments.data_generation import prepare_benchmark
            from power_fm_foresight._data.generate import demands
            import numpy as np
            _, descriptor, net, _, scaled = prepare_benchmark(cfg)
            support = {}
            for split in cfg['task']['split_episodes']:
                profiles, multipliers = [], []
                for e in descriptor['episodes']:
                    if e['split'] != split:
                        continue
                    ref = scaled[e['source_start_hour']:e['source_start_hour']+504]
                    pd, _ = demands(ref, net.bus[:, 2], net.bus[:, 3], cfg['generation'], e['seed'])
                    profiles.extend(ref); multipliers.append(pd[:, net.active_demand]/net.bus[net.active_demand, 2])
                values = np.concatenate(multipliers); refs = np.asarray(profiles)
                support[split] = dict(profile_min=float(refs.min()), profile_max=float(refs.max()),
                    actual_bus_multiplier_min=float(values.min()), actual_bus_multiplier_max=float(values.max()),
                    fraction_below_0p8=float((values < .8).mean()), fraction_above_1p2=float((values > 1.2).mean()))
            write_json(audit / 'support_diagnostic.json', support)
        if not (audit / 'stage_a.json').exists():
            shutil.copyfile(root / 'tests.xml', audit / 'tests.xml')
            command('scripts.collect_stage_a', cfg_path, audit / 'logs/collect.log', '--snapshot')
        assert read_json(audit / 'stage_a.json')['complete']
        write_json(audit / 'readiness.json', dict(stage_a_audit_passed=True, training_allowed=True,
            dataset_id=cfg['dataset']['id'], manifest_sha256=sha256(data / 'manifest.json'),
            stage_a_sha256=sha256(audit / 'stage_a.json'), prevalence_is_training_blocker=False))
        if time.time() > deadline-180:
            return
        if not (grid / 'shared/cache/manifest.json').exists():
            from power_fm_foresight._training.inputs import build_cache
            build_cache(cfg, 'xpu')
    events = read_json(audit / 'stage_a_followup/events.json')
    write_json(audit / 'stage_a_followup/headline_components.json', headlines(events))
    from power_fm_foresight._data.network import load_network
    from power_fm_foresight._data.dataset import load_temporal_manifest
    from power_fm_foresight.genco import Genco
    from power_fm_foresight._data.assets import checkpoint_assets
    net = load_network(config=cfg)
    load_temporal_manifest(cfg['paths']['temporal_root'], net)
    base = Genco(checkpoint_assets(cfg), cfg['checkpoint'], 'xpu', grid=net)
    assert base.case == case
    for seed in (0, 1, 2):
        out = grid / f'seed{seed}/adapter/cache'; out.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(grid / 'shared/cache/manifest.json', out / 'manifest.json')
    paths = [Path(cfg['paths']['temporal_root']) / 'manifest.json', audit / 'readiness.json', audit / 'stage_a.json',
        audit / 'stage_a_followup/events.json', audit / 'stage_a_followup/headline_components.json',
        Path(cfg['stage_b']['coordinate_scales']), grid / 'shared/cache/manifest.json']
    write_json(grid / 'ready.json', dict(case=case, passed=True, data_seed=42, training_seeds=[0, 1, 2],
        case_identity_verified=True, sha256={str(p): sha256(p) for p in paths}, ready_epoch=time.time()))
