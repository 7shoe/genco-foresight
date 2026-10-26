import copy
import random

import numpy as np
import torch

from power_fm_foresight._training.inputs import causal_window
from power_fm_foresight._training.state import atomic_torch_save,restore_rng,rng_state,seed_all
from power_fm_foresight._training.loop import walltime_seconds


def test_memory_mapped_history_has_no_future_solved_input(tmp_path):
    data={"embedding_bus":torch.randn(100,30,384),"observed_bus":torch.randn(100,30,4),
          "available_bus":torch.ones(100,30,4,dtype=torch.bool),"calendar":torch.randn(100,5),
          "targets":torch.randn(100,252)}
    path=tmp_path/"cache.pt";atomic_torch_save(path,data)
    mapped=torch.load(path,mmap=True,weights_only=True)
    before=causal_window(mapped,72)
    for key in ("embedding_bus","observed_bus","available_bus","targets"):
        mapped[key][72:]=0
    after=causal_window(mapped,72)
    assert all(torch.equal(v,after[0][k]) for k,v in before[0].items())
    assert torch.equal(before[1],after[1])


def test_portable_checkpoint_restores_all_cpu_rngs(tmp_path):
    seed_all(42);state=rng_state()
    expected=(random.random(),np.random.rand(),torch.randn(8))
    path=tmp_path/"rng.pt";atomic_torch_save(path,state)
    restore_rng(torch.load(path,weights_only=True,map_location="cpu"))
    assert random.random()==expected[0] and np.random.rand()==expected[1]
    assert torch.equal(torch.randn(8),expected[2])
    assert list(tmp_path.iterdir())==[path]


def test_pbs_time_formats():
    assert walltime_seconds("01:00:00")==walltime_seconds("3600")==3600
