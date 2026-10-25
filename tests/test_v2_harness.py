"""Fail-closed comparison and checkpoint/log recovery without running a model."""
import json

import pytest

from scripts.v2.compare_runs import compare_runs, digest, close
from training.metrics import TrainingMetrics


def put(path, value):
    path.write_text(json.dumps(value) + "\n")


def fixture(tmp_path):
    from pathlib import Path
    protocol = json.loads((Path(__file__).resolve().parents[1] / "reports/v2/stage0.protocol.json").read_text())
    protocol["comparison"].update(steps=2, origins=2)
    roots = [tmp_path / name for name in ("a", "b")]
    for root in roots:
        root.mkdir()
        manifest = dict(protocol["comparison"]["expected_manifest"], run_identity=root.name,
                        configuration={"v2": {"run_identity": root.name}})
        put(root / "manifest.json", manifest)
        (root / "latest.pt").write_bytes(b"checkpoint-fixture")
        put(root / "epochs.json", [])
        put(root / "status.json", {"base_unchanged": True, "steps_after": 2, "epochs_reached": 0,
            "stop_reason": "verification_step_limit",
            "origins_consumed": 2, "step_in_epoch": 2, "base_sha256": "base", "adapter_sha256": "adapter",
            "step_sequence_sha256": "sequence2", "checkpoint_sha256": digest(root / "latest.pt")})
        rows = [dict(schema_version=1, steps=i, origins_consumed=i, epoch=0, step_in_epoch=i,
            origin=["train_00", 72+i], step_sequence_sha256=f"sequence{i}", noise_sha256=f"noise{i}",
            loss=1., energy=.8, balance=.2, gradient_norm=3., step_seconds=.1, worker_seconds=i*.1) for i in (1, 2)]
        (root / "metrics.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
    return roots, protocol


def change_row(root, update):
    path = root / "metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0].update(update)
    path.write_text("".join(json.dumps(row)+"\n" for row in rows))


def test_exact_curves_ignore_only_clocks(tmp_path):
    roots, protocol = fixture(tmp_path)
    change_row(roots[1], {"step_seconds": 9., "worker_seconds": 123.})
    assert compare_runs(*roots, protocol)["passed"]


@pytest.mark.parametrize("change", [{"loss": 1.01}, {"noise_sha256": "wrong"},
    {"origins_consumed": 9}, {"gradient_norm": float("nan")}, {"energy": float("inf")}])
def test_changed_or_nonfinite_record_fails(tmp_path, change):
    roots, protocol = fixture(tmp_path)
    change_row(roots[1], change)
    assert not compare_runs(*roots, protocol)["passed"]


def test_missing_or_incomplete_curve_fails(tmp_path):
    roots, protocol = fixture(tmp_path)
    path = roots[1] / "metrics.jsonl"
    path.write_text(path.read_text().splitlines()[0]+"\n")
    assert not compare_runs(*roots, protocol)["passed"]
    path.unlink()
    assert not compare_runs(*roots, protocol)["passed"]


def test_wrong_configuration_and_checkpoint_fail(tmp_path):
    roots, protocol = fixture(tmp_path)
    path = roots[1] / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["global_K"] = 4
    put(path, manifest)
    assert not compare_runs(*roots, protocol)["passed"]
    manifest["global_K"] = 8
    put(path, manifest)
    (roots[1] / "latest.pt").write_bytes(b"corrupted")
    assert not compare_runs(*roots, protocol)["passed"]


def test_epoch_coverage_and_selection_curve(tmp_path):
    roots, protocol = fixture(tmp_path)
    protocol["comparison"]["epochs"] = 1
    for root in roots:
        put(root / "epochs.json", [{"epoch": 1, "steps": 2, "selection_energy_score": .2}])
        status = json.loads((root / "status.json").read_text())
        status["epochs_reached"] = 1
        put(root / "status.json", status)
    assert compare_runs(*roots, protocol)["passed"]
    put(roots[1] / "epochs.json", [{"epoch": 1, "steps": 2, "selection_energy_score": .3}])
    assert not compare_runs(*roots, protocol)["passed"]


def test_epoch_gate_preserves_noise_finiteness_and_declared_envelope(tmp_path):
    roots, protocol = fixture(tmp_path)
    protocol["comparison"].update(numerical_acceptance="epoch_envelope", epochs=1,
        epoch_envelopes={"1": {"minimum": .2, "maximum": .4}})
    protocol["comparison"].pop("thresholds")
    for root, score in zip(roots, (.2, .4)):
        put(root / "epochs.json", [{"epoch": 1, "steps": 2, "selection_energy_score": score}])
        status = json.loads((root / "status.json").read_text())
        status["epochs_reached"] = 1
        put(root / "status.json", status)
    change_row(roots[1], {"loss": 100., "gradient_norm": 1e6})
    assert compare_runs(*roots, protocol)["passed"]
    change_row(roots[1], {"noise_sha256": "different"})
    assert not compare_runs(*roots, protocol)["passed"]
    change_row(roots[1], {"noise_sha256": "noise1", "loss": float("nan")})
    assert not compare_runs(*roots, protocol)["passed"]
    change_row(roots[1], {"loss": 100.})
    put(roots[1] / "epochs.json", [{"epoch": 1, "steps": 2, "selection_energy_score": .400000001}])
    assert not compare_runs(*roots, protocol)["passed"]


def test_absolute_floor_handles_near_zero():
    assert close(0., 1e-9, {"atol": 2e-9, "rtol": 1e-3})
    assert not close(0., 1e-9, {"atol": 0., "rtol": 1e-3})
    assert not close(float("nan"), 0., {"atol": 1., "rtol": 1.})


def test_recovery_archives_only_uncommitted_tail(tmp_path):
    progress = dict(steps=0, origins_consumed=0, epoch=0, step_in_epoch=0, step_sequence_sha256="sequence")
    logger = TrainingMetrics(tmp_path, progress, 1)
    for step in (1, 2):
        progress.update(steps=step, origins_consumed=step, step_in_epoch=step)
        logger.record(progress, ("train_00", 72+step), dict(loss=1., energy=.9, balance=.1,
            gradient_norm=2., noise_sha256="noise"), .1, .2)
    logger.sync()
    logger.close()
    original = (tmp_path / "metrics.jsonl").read_bytes()
    with (tmp_path / "metrics.jsonl").open("ab") as stream:
        stream.write(b'{"torn":')
    progress.update(steps=1, origins_consumed=1, step_in_epoch=1)
    logger = TrainingMetrics(tmp_path, progress, 1)
    logger.close()
    assert (tmp_path / "metrics.jsonl").read_bytes() == original.splitlines(keepends=True)[0]
    assert next(tmp_path.glob("metrics.abandoned.*")).read_bytes() == original.splitlines(keepends=True)[1]+b'{"torn":'


def test_missing_committed_rows_and_invalid_cadence_rejected(tmp_path):
    for interval in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            TrainingMetrics(tmp_path, {"steps": 0}, interval)
    with pytest.raises(ValueError, match="Missing committed"):
        TrainingMetrics(tmp_path, {"steps": 1}, 1)
