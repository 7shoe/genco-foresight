"""Offline rescoring of fixed saved v3/v4 origins and release evidence figures."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

from power_fm_foresight.scoring import evaluate, UNITS
from experiments.headline_numbers import collect

ROOT = Path(__file__).resolve().parents[1]
MODES = ("C", "P", "P0", "S", "A", "D")


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arrays(path):
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def check_numbers(actual, expected, path="root"):
    """Cross-version float tolerances, with exact counts and PIT bin membership."""
    if isinstance(actual, dict):
        for key, value in actual.items():
            if key in ("mode", "origin_id", "samples", "variables", "joint_energy_score", "joint_variogram_p0_5", "joint_observed_coordinates") or path != "root":
                assert key in expected, (path, key)
                check_numbers(value, expected[key], path + "." + key)
    elif isinstance(actual, list):
        assert len(actual) == len(expected), path
        for i, (a, b) in enumerate(zip(actual, expected)):
            check_numbers(a, b, f"{path}[{i}]")
    elif isinstance(actual, int):
        assert actual == expected, (path, actual, expected)
    elif isinstance(actual, float):
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-8, err_msg=path)
    else:
        assert actual == expected, (path, actual, expected)


def figures(runs, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    diag = read(runs / "stage_b/diagnostics/report.json")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    bins = ("lt_0_6", "0_6_to_0_8", "ge_0_8")
    for ax, key, label, multiplier in zip(axes, ("vm", "pg_mw", "imbalance"),
            ("Vm MAE (milli-p.u.)", "Pg MAE, all 6 generators (MW)", "Mean |P imbalance| / bus (MW)"), (1000, 1, 1)):
        values = [diag["ood_P"][b]["same_input_errors"][key]["mae"] * multiplier if key != "imbalance"
                  else diag["ood_P"][b]["balance_audited_hours"]["residual_p_mw"]["mean_abs"] for b in bins]
        ax.plot(range(3), values, "o-", color="#be6151")
        ax.set_xticks(range(3), ["<0.6", "0.6–0.8", "≥0.8"])
        ax.set_xlabel("Realized reference profile (retrospective)")
        ax.set_ylabel(label); ax.set_ylim(bottom=0); ax.grid(alpha=.2)
    fig.suptitle("Frozen GENCO: same-input error by realized-profile bin")
    fig.tight_layout(); fig.savefig(output / "figure_2_ood.png", dpi=150); plt.close(fig)
    report = read(runs / "stage_c/report.json")
    events = [r["event"] for r in report["events"] if r["resolution"] == "hourly"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for level, ax in zip((50, 90), axes):
        matrix = np.array([[report["continuous"]["test"][m][e][f"coverage_{level}"] for m in MODES] for e in events])
        ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        ax.set_xticks(range(6), MODES); ax.set_yticks(range(len(events)), events)
        ax.set_title(f"Nominal {level}% coverage")
        for i in range(len(events)):
            for j in range(6):
                ax.text(j, i, f"{matrix[i,j]:.1%}", ha="center", va="center", color="white" if matrix[i,j] > .6 else "black")
    fig.tight_layout(); fig.savefig(output / "figure_3c_coverage.png", dpi=150); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True, help="Local self-contained package directory")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--start-epoch", type=float, default=None)
    a = p.parse_args(); start = a.start_epoch or time.time()
    a.output.mkdir(parents=True, exist_ok=True)
    result = {"started_utc": datetime.fromtimestamp(start, timezone.utc).isoformat(),
              "host": platform.node(), "platform": platform.platform(), "python": sys.version,
              "source_directory": str(ROOT),
              "torch": torch.__version__, "numpy": np.__version__, "device": "cpu",
              "scope": "Saved-forecast rescoring and evidence figures; not model inference, retraining or OPF regeneration."}
    try:
        torch.set_num_threads(1)
        data = a.bundle.resolve()
        manifest = read(data / "bundle.json")
        for path, sha in manifest["sha256"].items():
            assert digest(data / path) == sha, path
        assert digest(ROOT / "scoring.py") == manifest["scorer_sha256"], "Frozen scorer changed"
        runs = data / "evidence/runs"
        checked = []
        versions = manifest.get("datasets", ["v3"])
        for version in versions:
            base = runs / ("stage_b" if version == "v3" else "v4/stage_b")
            small = data / ("small" if version == "v3" else "small/v4")
            scales = read(small / "coordinate_scales.json")
            pairs = arrays(small / "variogram_pairs.npz")["pairs"]
            if version == "v3":
                identities = manifest["origins"]
            else:
                protocol = read(base / "protocol.json")
                # Includes test_02 hour 240, whose solved state is missing.
                identities = [{"id": protocol["origins"][i]["id"], "index": i} for i in (0, 43)]
            for identity in identities:
                folder = base / "origins" / identity["id"]
                target = arrays(folder / "targets.npz")
                masks = {k: np.isfinite(target[k]) for k in UNITS}
                for mode in MODES:
                    seed = 42 + 10000 + identity["index"] + (10000 if mode in ("C", "P0") else 0)
                    mode_folder = base / "controls/origins" / identity["id"] if version == "v3" and mode in ("C", "P0") else folder
                    actual, _ = evaluate(arrays(mode_folder / f"{mode}.npz"), target, masks,
                        {"scales": scales, "variogram_pairs": pairs, "mode": mode, "origin_id": identity["id"], "seed": seed})
                    expected = read(mode_folder / f"{mode}_scores.json")
                    check_numbers(actual, expected)
                    checked.append({"dataset": version, "origin": identity["id"], "mode": mode, "energy_score": actual["joint_energy_score"],
                                    "vm_observed_coordinates": actual["variables"]["vm"]["count"]})
                print(f"Verified {version} {identity['id']}: six modes.", flush=True)
        figure_dir = a.output / "figures"; figure_dir.mkdir(exist_ok=True)
        figures(runs, figure_dir)
        if "v4" in versions:
            from experiments.figure_release_comparison import render
            render(runs, figure_dir)
        headline = collect(runs)
        (a.output / "headline_numbers.json").write_text(json.dumps(headline, indent=2, allow_nan=False) + "\n")
        result.update(passed=True, checked=checked, origins=len(checked)//len(MODES), datasets=versions,
            bundle_manifest_sha256=digest(data / "bundle.json"),
            publication="local only; no upload or download",
            scorer_sha256=digest(ROOT / "scoring.py"),
            figures={f.name:digest(f) for f in figure_dir.glob("*.png")})
    except BaseException as e:
        result.update(passed=False, failure=repr(e))
        raise
    finally:
        result["elapsed_seconds_including_setup"] = time.time() - start
        result["under_30_minutes"] = result["elapsed_seconds_including_setup"] < 1800
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (a.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps(result, indent=2), flush=True)
    assert result["under_30_minutes"], "Reproduction exceeded the 30-minute target"


if __name__ == "__main__":
    main()
