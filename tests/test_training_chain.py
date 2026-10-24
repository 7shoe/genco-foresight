"""Queue-priority behavior without contacting PBS or running model training."""
import subprocess

import pytest

from common import read_json, write_json
from scripts import training_chain


def setup_chain(tmp_path, monkeypatch, reserve):
    root = tmp_path / "train"
    audit = tmp_path / "audit"
    write_json(audit / "readiness.json", {"training_allowed": True})
    write_json(root / "resume_test/report.json", {"passed": True})
    chain = {"mode": "A", "device": "xpu", "status": "running",
             "created_epoch": 0, "budget_seconds": 21600,
             "dependency_chaining": True, "reserve_queue_slot": reserve,
             "jobs": [{"job_id": "prior", "state": "F",
                       "result": {"wall_used_seconds": 20000}},
                      {"job_id": "current", "state": "R", "dependency": None,
                       "submitted_epoch": 1}]}
    write_json(root / "chain.json", chain)
    cfg = {"paths": {"training_run": str(root), "stage_a_run": str(audit)},
           "stage_b": {"pbs_queue": "debug", "job_training_guard_seconds": 3300}}
    monkeypatch.setattr(training_chain, "load_config", lambda: cfg)
    monkeypatch.setattr("sys.argv", ["training_chain", "--device", "xpu", "--once"])
    return root, chain


@pytest.mark.parametrize("reserve", [False, True])
def test_priority_prefetch_near_budget_end(tmp_path, monkeypatch, reserve):
    root, _ = setup_chain(tmp_path, monkeypatch, reserve)
    monkeypatch.setattr(training_chain, "job_state", lambda job: ("R", None))
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "successor\n", "")

    monkeypatch.setattr(training_chain.subprocess, "run", fake_run)
    training_chain.main()
    result = read_json(root / "chain.json")
    assert len(commands) == int(reserve)
    if reserve:
        assert commands[0][0] == "qsub"
        assert "depend=afterany:current" in commands[0]
        assert result["jobs"][-1]["dependency"] == "current"


def test_finished_training_cancels_reserved_successor(tmp_path, monkeypatch):
    root, chain = setup_chain(tmp_path, monkeypatch, True)
    chain["jobs"].append({"job_id": "successor", "state": "H",
                          "dependency": "current", "submitted_epoch": 2})
    write_json(root / "chain.json", chain)
    write_json(root / "jobs/current.json", {"status": "finished", "wall_used_seconds": 100,
                                           "job_start_epoch": 3, "job_end_epoch": 103})
    monkeypatch.setattr(training_chain, "job_state",
                        lambda job: ("F", 0) if job == "current" else ("H", None))
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(training_chain.subprocess, "run", fake_run)
    training_chain.main()
    result = read_json(root / "chain.json")
    assert commands == [["qdel", "successor"]]
    assert result["status"] == "finished"
    assert result["jobs"][-1]["state"] == "cancelled"
