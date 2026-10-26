"""Calibration-only maps and transparent held-out event/interval diagnostics."""
from pathlib import Path
import time

import numpy as np

from power_fm_foresight._config import ROOT, load_config, read_json, sha256, write_json
from power_fm_foresight._data.network import load_network
from monitor import (RELIABILITY_EDGES, LOGIT_EPS, event_probabilities,
                     event_targets, fit_map, apply_map, reliability)
from power_fm_foresight.scoring import summarize_marginals
from experiments.diagnose_stage_b import npz
from power_fm_foresight._training.forecaster import atomic_npz

MODES = ("C", "P", "P0", "S", "A", "D")
RESOLUTIONS = ("hourly", "daily_any")


def event_selection(cfg, net):
    settings = cfg["stage_b"]["calibration"]
    if "headline_voltage_bus_ids" in settings:
        return dict(bus_ids=settings["headline_voltage_bus_ids"], line_ids=settings["continuous_loading_line_ids"])
    selected = read_json(Path(cfg["paths"]["stage_a_run"])/"stage_a_followup/headline_components.json")
    return dict(bus_ids=selected["voltage_bus_ids"], line_ids=selected["thermal_line_ids"])


def freeze(cfg, net=None):
    net = net if net is not None else load_network(config=cfg)
    selected = event_selection(cfg, net)
    out = Path(cfg["paths"]["runs_root"]) / "stage_c"
    out.mkdir(parents=True, exist_ok=True)
    b = out.parent / "stage_b"
    protocol = {"created_by": "scripts.calibrate_stage_c", "dataset": cfg["dataset"]["id"],
        "modes": list(MODES), "voltage_bus_ids": list(selected["bus_ids"]), "thermal_line_ids": list(selected["line_ids"]),
        "primary_resolution": "hourly", "secondary_resolution": "daily_any",
        "thresholds": "Vm <= Vmin + 0.05*(Vmax-Vmin) or >= Vmax-0.05*(Vmax-Vmin); loading > 0.95",
        "posthoc": "One sigmoid of clipped logit(p) per mode/family/resolution, pooled across supported components",
        "map_fit_partition": "risk_calibration", "logit_clip": LOGIT_EPS, "logistic_C": 1.,
        "support": "At least five distinct positive and negative calibration days; daily-any additionally 20-80% prevalence",
        "bins": RELIABILITY_EDGES.tolist(), "minimum_plotted_bin_count": 20,
        "prevalence_shift": "max(cal,test)/min(cal,test)>1.5; retain both panels; raw primary, mapped sensitivity",
        "uncertainty": "Descriptive reliability, no IID confidence bands; 72 days from four source episodes per held-out partition",
        "weights": "Uniform complete joint trajectories; daily-any evaluated within each trajectory",
        "continuous": "Existing unchanged scorer PIT and 50/90% intervals; event maps do not recalibrate continuous ensembles",
        "test_protocol_sha256": sha256(b / "protocol.json"),
        "controls_protocol_sha256": sha256(b / "controls/protocol.json"),
        "source_sha256": {p: sha256(ROOT / p) for p in ("monitor.py", "scripts/calibrate_stage_c.py", "scoring.py")}}
    if (out / "protocol.json").exists():
        assert read_json(out / "protocol.json") == protocol
    else:
        write_json(out / "protocol.json", protocol)
    return out


def collect(cfg, partition, out, net=None):
    root = out / "calibration" if partition == "risk_calibration" else out.parent / "stage_b"
    p = read_json(root / "protocol.json")
    if partition == "risk_calibration":
        assert p["partition"] == partition
    net = net if net is not None else load_network(config=cfg)
    selected = event_selection(cfg, net)
    families = {"voltage": [f"voltage_bus_{b}" for b in selected["bus_ids"]],
                "thermal": [f"thermal_line_{i}" for i in selected["line_ids"]]}
    targets = {f: {r: [] for r in RESOLUTIONS} for f in families}
    probabilities = {m: {f: {r: [] for r in RESOLUTIONS} for f in families} for m in MODES}
    continuous = {m: {} for m in MODES}
    evidence = {}
    components = [(f"voltage_bus_{bus}", "vm", net.bus_ids.tolist().index(bus)) for bus in selected["bus_ids"]] + [(f"thermal_line_{i}", "loading", i-1) for i in selected["line_ids"]]
    for o in p["origins"]:
        directory = root / "origins" / o["id"]
        target = event_targets(npz(directory / "targets.npz"), net, **event_selection(cfg, net))
        evidence[str(directory / "targets.npz")] = sha256(directory / "targets.npz")
        for f in families:
            for r in RESOLUTIONS:
                targets[f][r].append(target[f][r])
        for m in MODES:
            folder = root / "controls/origins" / o["id"] if partition == "test" and m in ("C", "P0") else directory
            done = read_json(folder / "done.json")
            # Old controls use the same artifact hash contract as the main run.
            artifacts = done.get("artifacts", {})
            for filename in (f"{m}.npz", f"{m}_scores.npz"):
                path = folder / filename
                digest = sha256(path)
                assert filename in artifacts and digest == artifacts[filename], path
                evidence[str(path)] = digest
            pred = event_probabilities(npz(folder / f"{m}.npz"), net, **event_selection(cfg, net))
            for f in families:
                for r in RESOLUTIONS:
                    probabilities[m][f][r].append(pred[f][r])
            scores = npz(folder / f"{m}_scores.npz")
            for event, key, index in components:
                dest = continuous[m].setdefault(event, {})
                for name, v in scores.items():
                    if name.startswith(key + "__"):
                        dest.setdefault(name.split("__")[1], []).append(v[:, index:index + 1])
    targets = {f: {r: np.stack(v) for r, v in d.items()} for f, d in targets.items()}
    probabilities = {m: {f: {r: np.stack(v) for r, v in d.items()} for f, d in fam.items()} for m, fam in probabilities.items()}
    continuous = {m: {event: summarize_marginals({k: np.concatenate(v) for k, v in arrays.items()})
                     for event, arrays in events.items()} for m, events in continuous.items()}
    write_json(out / f"inputs_{partition}.json", {"protocol_sha256": sha256(root / "protocol.json"),
        "origins": p["origins"], "artifacts": evidence})
    atomic_npz(out / f"events_{partition}.npz", **{
        **{f"target__{f}__{r}": v for f, d in targets.items() for r, v in d.items()},
        **{f"{m}__{f}__{r}": v for m, fam in probabilities.items() for f, d in fam.items() for r, v in d.items()}})
    return targets, probabilities, continuous, p["origins"]


def main():
    cfg = load_config()
    net = load_network(config=cfg)
    selected = event_selection(cfg, net)
    families = {"voltage": [f"voltage_bus_{b}" for b in selected["bus_ids"]],
                "thermal": [f"thermal_line_{i}" for i in selected["line_ids"]]}
    out = freeze(cfg, net)
    cy, cp, cc, co = collect(cfg, "risk_calibration", out, net)
    maps = {m: {f: {r: fit_map(cp[m][f][r], cy[f][r], r, "risk_calibration")
                    for r in RESOLUTIONS} for f in families} for m in MODES}
    # Persist the map before opening test event arrays; no model or threshold selection on test.
    write_json(out / "maps.json", {"maps": maps, "fit_inputs_sha256": sha256(out / "inputs_risk_calibration.json"),
        "protocol_sha256": sha256(out / "protocol.json"), "fit_completed_epoch": time.time()})
    ty, tp, tc, to = collect(cfg, "test", out, net)
    rates = read_json(Path(cfg["paths"]["stage_a_run"]) / "stage_b_prevalence.json")["prevalence"]
    train = {(r["event"], r["resolution"]): r["prevalence"] for r in rates if r["partition"] == "train"}
    report = {"protocol_sha256": sha256(out / "protocol.json"), "maps_sha256": sha256(out / "maps.json"),
        "continuous": {"risk_calibration": cc, "test": tc}, "events": [],
        "limitations": ["Only four independent source episodes (72 days) in each held-out partition; descriptive estimates, not independent-hour confidence claims.",
            "P/P0 have identical event probabilities; their finite-ensemble randomized PIT differs. Neither has nonzero interval spread.",
            "Maps change event probabilities only, not states, balances, PIT or continuous interval coverage.",
            "Numerical-difficulty classifier unavailable: zero failed labels in 5,040 training hours at the fixed budget; no discrimination or two-class calibration is estimable.",
            "Dual-price quantiles and scenario-selected third hybrid arm deferred; current deliverables prioritize voltage calibration and the requested two-arm experiment."]}
    for f, names in families.items():
        for r in RESOLUTIONS:
            for j, event in enumerate(names):
                c, t = cy[f][r][..., j], ty[f][r][..., j]
                cr, tr = float(np.nanmean(c)), float(np.nanmean(t))
                ratio = max(cr, tr) / min(cr, tr) if min(cr, tr) > 0 else (1. if cr == tr else None)
                shifted = ratio is None or ratio > 1.5
                row = {"event": event, "family": f, "resolution": r, "calibration_positive": int(np.nansum(c)),
                    "calibration_count": int(np.isfinite(c).sum()), "calibration_prevalence": cr,
                    "test_positive": int(np.nansum(t)), "test_count": int(np.isfinite(t).sum()), "test_prevalence": tr,
                    "prevalence_ratio": ratio, "shift_exceeds_1_5": shifted,
                    "map_role": "sensitivity; raw primary" if shifted else "raw and mapped side by side",
                    "support": {k: v[j] for k, v in maps["A"][f][r]["support"].items()},
                    "train_prevalence": train[(event, r)],
                    "train_constant_brier": float(np.nanmean((t - train[(event, r)]) ** 2)), "modes": {}}
                for m in MODES:
                    raw = tp[m][f][r][..., j]
                    mapped = apply_map(tp[m][f][r], maps[m][f][r])[..., j]
                    row["modes"][m] = {"raw": reliability(raw, t), "mapped": reliability(mapped, t),
                        "by_episode": {name: {"raw": reliability(raw[mask], t[mask]),
                            "mapped": reliability(mapped[mask], t[mask])} for name in sorted({o["episode_id"] for o in to})
                            for mask in [np.array([o["episode_id"] == name for o in to])]}}
                report["events"].append(row)
    write_json(out / "report.json", report)
    lines = ["# Stage C: v3 calibration", "", "Primary: hourly per-bus voltage and hourly line-1 loading. All forecasts are frozen. Sigmoid event maps fit only on calibration episodes; continuous diagnostics use the unchanged Stage B scorer.", "",
        "| Event | Resolution | Calibration +/n | Test +/n | Ratio | Eligible | C raw/map | P raw/map | P0 raw/map | S raw/map | A raw/map | D raw/map |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    fmt = lambda v: "NA" if v is None else f"{v:.5f}"
    for row in report["events"]:
        briers = [" / ".join(fmt(row["modes"][m][k]["brier"]) for k in ("raw", "mapped")) for m in MODES]
        lines.append(f"| {row['event']} | {row['resolution']} | {row['calibration_positive']}/{row['calibration_count']} | {row['test_positive']}/{row['test_count']} | {fmt(row['prevalence_ratio'])}{'*' if row['shift_exceeds_1_5'] else ''} | {row['support']['eligible']} | " + " | ".join(briers) + " |")
    lines += ["", "Cells are Brier scores (lower is better). * Prevalence ratio >1.5 or one rate zero: mapped sensitivity only. Unsupported daily events remain in this audit table but are excluded from daily reliability figures.", "", "Full reliability-bin counts, per-episode scores, PIT and 50/90% coverage/width for C/P/P0/S/A/D: `report.json`. Map parameters: `maps.json`. Input hashes: `inputs_*.json`.", ""]
    lines += ["- " + s for s in report["limitations"]]
    (out / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
