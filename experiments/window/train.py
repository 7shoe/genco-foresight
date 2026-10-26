"""Run an unchanged pinned trainer with an isolated tile and audited inputs."""
import json
import os
from pathlib import Path
import sys

task = json.loads(Path(os.environ['WINDOW_TASK']).read_text())
sys.path.insert(0, task['snapshot'])
from power_fm_foresight._config import load_config, read_json, sha256
import torch

if not os.environ.get('PBS_JOBID'): raise RuntimeError('Training requires a compute allocation')
assert os.environ['ZE_FLAT_DEVICE_HIERARCHY'] == 'FLAT' and torch.xpu.device_count() == 1
torch.xpu.set_device(0); torch.set_num_threads(1)
pin = read_json(task['source_pin'])
for name, digest in pin['files'].items():
    if sha256(Path(pin['snapshot']) / name) != digest: raise ValueError(f'Source changed: {name}')
for name, digest in read_json(task['ready'])['sha256'].items():
    if sha256(name) != digest: raise ValueError(f'Audited data changed: {name}')
cfg = load_config()
from power_fm_foresight._training.loop import run
run(cfg, 'xpu')
