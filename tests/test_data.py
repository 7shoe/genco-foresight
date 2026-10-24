import copy

import numpy as np

from common import write_json
from data.loader.dataset import TemporalDataset, daily_origins, load_static_splits
from data.synthetic.generate import demands, episode_plan


def test_episode_and_source_partitions_do_not_overlap():
    episodes, pools = episode_plan()
    assert len(episodes) == 20
    assert len({e["seed"] for e in episodes}) == 20
    for name, blocks in pools.items():
        other = {i for key, values in pools.items() if key != name for i in values}
        assert set(blocks).isdisjoint(other)
    assert sum(e["split"] == "test" for e in episodes)*len(daily_origins()) == 72
    assert all(o-72 >= 0 and o+24 <= 504 for o in daily_origins())


def test_static_split_leakage_is_rejected(tmp_path):
    import pytest
    path = tmp_path/"splits.npz"
    np.savez(path, train=np.array([0, 1]), val=np.array([2]), test=np.array([3]))
    assert len(load_static_splits(path, count=4)["train"]) == 2
    np.savez(path, train=np.array([0, 1]), val=np.array([1]), test=np.array([3]))
    with pytest.raises(ValueError, match="disjoint"):
        load_static_splits(path, count=4)


def test_demand_law_is_reproducible_and_handles_reactive_only_load():
    pd = np.array([10., 0., 0., 5.])
    qd = np.array([3., 0., -2., -1.])
    settings = {"rho": 0.9, "innovation_std": 0.02, "sigma": 0.03}
    a, b = demands(np.ones(100), pd, qd, settings, 7)
    again = demands(np.ones(100), pd, qd, settings, 7)
    np.testing.assert_array_equal(a, again[0])
    np.testing.assert_array_equal(b, again[1])
    np.testing.assert_array_equal(a[:, 1:3], 0)
    np.testing.assert_array_equal(b[:, 2], -2)
    np.testing.assert_allclose(b[:, [0, 3]], a[:, [0, 3]]*np.array([0.3, -0.2]))


def test_future_targets_do_not_enter_window_inputs(tmp_path):
    write_json(tmp_path/"manifest.json", {"episodes": [{"episode_id": "test_00", "split": "test", "hours": 504}]})
    records = [{"hour": i, "time_utc": f"hour_{i}", "pd_mw": [float(i)],
                "target_state": {"vm": [float(i)]}, "state_available": True} for i in range(504)]
    path = tmp_path/"episodes/test_00.json"
    write_json(path, {"records": records})
    before = TemporalDataset(tmp_path, "test")[0]
    changed = copy.deepcopy(records)
    for row in changed[72:]:
        row["pd_mw"] = [999999.]
        row["target_state"] = {"vm": [-999999.]}
    write_json(path, {"records": changed})
    after = TemporalDataset(tmp_path, "test")[0]
    assert before["history"] == after["history"]
    assert before["known_future"] == after["known_future"]
    assert before["targets"] != after["targets"]
    assert max(r["hour"] for r in before["history"]) == 71


def test_training_uses_all_complete_hourly_windows_and_evaluation_stays_midnight(tmp_path):
    episodes, _ = episode_plan()
    write_json(tmp_path/"manifest.json", {"episodes": episodes})
    train = TemporalDataset(tmp_path, "train")
    assert len(train) == 4908
    assert train.index[:409] == [("train_00", o) for o in range(72, 481)]
    for split, expected in (("model_selection", 36), ("risk_calibration", 36), ("test", 72)):
        dataset = TemporalDataset(tmp_path, split)
        assert len(dataset) == expected
        assert all(origin % 24 == 0 for _, origin in dataset.index)


def test_calendar_pools_are_disjoint_and_reuse_only_within_partition():
    import pytest
    from common import load_config
    cfg = load_config()
    declared = cfg["dataset"]["source_blocks_one_based"]
    episodes, pools = episode_plan(cfg["seed"], declared, cfg["task"]["split_episodes"])
    assert sorted(i for values in pools.values() for i in values) == list(range(17))
    for split, pool in pools.items():
        actual = [e for e in episodes if e["split"] == split]
        assert set(e["source_block"] for e in actual) == set(pool)
        assert all(e["source_block"]+1 in declared[split] for e in actual)
        # Every partition has source dates in both halves of the year.
        assert min(pool) <= 7 and max(pool) >= 8
    assert len([e for e in episodes if e["split"] == "train"]) == 10
    assert len(pools["risk_calibration"]) == len(pools["test"]) == 4
    assert len([e for e in episodes if e["split"] == "risk_calibration"])*18 == 72
    bad = copy.deepcopy(declared)
    bad["test"][0] = bad["train"][0]
    with pytest.raises(ValueError, match="disjoint"):
        episode_plan(cfg["seed"], bad, cfg["task"]["split_episodes"])


def test_v3_windows_and_reporting_policy(tmp_path):
    from common import load_config
    from scripts.stage_b_prevalence import reporting_policy
    cfg = load_config()
    episodes, _ = episode_plan(cfg["seed"], cfg["dataset"]["source_blocks_one_based"], cfg["task"]["split_episodes"])
    write_json(tmp_path/"manifest.json", {"episodes": episodes})
    assert len(TemporalDataset(tmp_path, "train")) == 4090
    assert len(TemporalDataset(tmp_path, "risk_calibration")) == 72
    settings = cfg["stage_b"]["calibration"]
    for resolution in ("hourly", "daily_any"):
        policy = reporting_policy({"positive": 60, "total": 100}, {"positive": 20, "total": 100}, resolution, settings)
        assert policy["panel_eligible_by_prevalence"] and policy["shift_caption_required"]
        assert not policy["training_blocker"] and policy["posthoc_fit_partition"] == "risk_calibration"
    for positive, eligible in ((19,False),(20,True),(80,True),(81,False)):
        policy = reporting_policy({"positive": positive, "total": 100}, {"positive": 10, "total": 100}, "daily_any", settings)
        assert policy["panel_eligible_by_prevalence"] == eligible


def test_v2_gate_checks_each_resolution_and_rejects_undefined_ratios():
    from scripts.data_v2 import prevalence_gate
    events = [f"voltage_bus_{i}" for i in (1, 9, 11, 13, 15, 21, 22)] + ["thermal_line_1"]
    rows = [{"event": e, "resolution": r, "partition": p,
             "positive": 4 if p == "risk_calibration" else 12,
             "total": 10 if p == "risk_calibration" else 20}
            for e in events for r in ("hourly", "daily_any") for p in ("risk_calibration", "test")]
    assert prevalence_gate({"prevalence": rows})["passed"]  # Exactly 1.5 is accepted.
    rows[-1]["positive"] = 13
    failed = prevalence_gate({"prevalence": rows})
    assert not failed["passed"] and len(failed["failures"]) == 1
    assert failed["failures"][0]["event"] == "thermal_line_1"
    rows[-1]["positive"] = rows[-2]["positive"] = 0
    assert not prevalence_gate({"prevalence": rows})["passed"]
