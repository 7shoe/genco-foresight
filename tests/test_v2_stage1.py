"""Determinism defaults, audited restoration, and the actual stopping controller."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from power_fm_foresight._config import read_json, sha256, write_json
from power_fm_foresight._training import loop
from power_fm_foresight._training.state import checkpoint, restore, seed_all, tensor_digest


def test_historical_default_and_explicit_flag():
    assert loop.deterministic_setting({"stage_b": {}}) is True
    for flag in (True, False):
        assert loop.deterministic_setting({"stage_b": {"deterministic_algorithms": flag}}) is flag
    for invalid in ("false", 0, None):
        with pytest.raises(ValueError):
            loop.deterministic_setting({"stage_b": {"deterministic_algorithms": invalid}})


def toy(cfg, device):
    seed_all(42)
    torch.use_deterministic_algorithms(loop.deterministic_setting(cfg))
    model = torch.nn.Module()
    model.backbone = torch.nn.Linear(2, 2).requires_grad_(False)
    model.adapter = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.adapter.parameters(), lr=1e-4, foreach=False)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.)
    cache = SimpleNamespace(indices={"train": [("train_00", 72), ("train_00", 73)]})
    return model, optimizer, scheduler, None, cache


def toy_step(model, optimizer, scheduler, template, cache, identity, cfg, device):
    noise = torch.randn(2)
    loss = (model.adapter(noise) - 1).square().mean()
    optimizer.zero_grad()
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.adapter.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    scheduler.step()
    return {"loss": float(loss.detach()), "energy": float(loss.detach()), "balance": 0.,
            "gradient_norm": float(norm), "noise_sha256": tensor_digest({"noise": noise})}


def setup(tmp_path, monkeypatch):
    dataset, audit = tmp_path / "data", tmp_path / "audit"
    write_json(dataset / "manifest.json", {"fixture": True})
    write_json(audit / "readiness.json", {"training_allowed": True, "stage_a_audit_passed": True,
               "manifest_sha256": sha256(dataset / "manifest.json")})
    cfg = {"seed": 42, "paths": {"temporal_root": str(dataset), "stage_a_run": str(audit)},
        "stage_b": {"deterministic_algorithms": False, "metrics_interval_steps": 1,
                    "training_budget_seconds": 1000, "job_training_guard_seconds": 1000,
                    "checkpoint_interval_seconds": 60, "early_stopping": {"patience": 3}, "max_epochs": 10}}
    monkeypatch.delenv("GENCO_JOB_START_EPOCH", raising=False)
    monkeypatch.setattr(loop, "make_manifest", lambda value: deepcopy(value))
    monkeypatch.setattr(loop, "initialize", toy)
    monkeypatch.setattr(loop, "training_step", toy_step)
    monkeypatch.setattr(loop, "selection_score", lambda *args: 1.)
    return cfg


def test_patience_three_and_boundary_resume_use_real_controller(tmp_path, monkeypatch):
    cfg = setup(tmp_path, monkeypatch)
    a, b = tmp_path / "uninterrupted", tmp_path / "resumed"
    uninterrupted = loop.run(cfg, "cpu", a)
    before = loop.run(cfg, "cpu", b, stop_after_steps=2)
    assert before["epochs_reached"] == 0 and before["step_in_epoch"] == 2
    resumed = loop.run(cfg, "cpu", b)
    assert uninterrupted["stop_reason"] == resumed["stop_reason"] == "early_stopping"
    assert uninterrupted["epochs_reached"] == resumed["epochs_reached"] == 4
    assert uninterrupted["best_epoch"] == resumed["best_epoch"] == 1
    from experiments.resume_test import compare
    states = [torch.load(path / "latest.pt", map_location="cpu", weights_only=True) for path in (a, b)]
    for key in ("model", "optimizer", "scheduler", "rng"):
        assert compare(states[0][key], states[1][key])[0]
    records = list((b / "restores").glob("*.json"))
    assert len(records) == 1 and read_json(records[0])["passed"]


def test_mechanism_mode_forces_on_without_mutating_config(tmp_path, monkeypatch):
    cfg = setup(tmp_path, monkeypatch)
    root = tmp_path / "mechanism"
    loop.run(cfg, "cpu", root, stop_after_steps=1, test_mode=True)
    assert read_json(root / "manifest.json")["stage_b"]["deterministic_algorithms"] is True
    assert cfg["stage_b"]["deterministic_algorithms"] is False
    assert torch.are_deterministic_algorithms_enabled()


def test_restore_rejects_changed_configuration(tmp_path):
    model, opt, sched, _, _ = toy({"stage_b": {}}, "cpu")
    path = tmp_path / "state.pt"
    checkpoint(path, model, opt, sched, {"steps": 0}, {"mode": True})
    with pytest.raises(ValueError, match="manifest mismatch"):
        restore(path, model, opt, sched, {"mode": False})


def test_case118_timing_continues_partial_worker(tmp_path, monkeypatch):
    import io
    import json
    import time
    from experiments.v2 import prepare_case118, stage1
    folder=tmp_path/"case118"
    for name in ("setup.json","cache_ready.json"):write_json(folder/name,{"passed":True})
    write_json(folder/"data/manifest.json",{"episodes":[]})
    def evidence(mode,steps):
        write_json(folder/mode/"status.json",{"steps_after":steps,"base_unchanged":True,
            "stop_reason":"job_walltime_guard" if steps<70 else "verification_step_limit",
            "cumulative_training_wall_seconds":100.,"startup_seconds":2.})
        rows=[{"steps":i,"origin":["train_00",i],"noise_sha256":str(i),"loss":1.,"energy":.9,
               "balance":.1,"gradient_norm":2.,"step_seconds":.5 if mode=="on" else .25} for i in range(1,steps+1)]
        (folder/mode/"metrics.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
    evidence("off",30);evidence("on",70)
    launched=[]
    def launch(root,name,tile,deadline,attempt):
        launched.append(name);evidence("off",70)
        return {"process":SimpleNamespace(poll=lambda:0),"stream":io.StringIO(),"name":name,
                "began":time.time(),"log":"fixture.log"}
    monkeypatch.setattr(stage1,"launch",launch)
    monkeypatch.setattr(prepare_case118.time,"sleep",lambda _:None)
    result=prepare_case118.job(tmp_path,time.time()+500,{"workers":[]})
    assert launched==["case118_off"] and result["passed"] and result["speedup"]==2.
