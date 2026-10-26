"""Verify saved Stage B evidence and produce the complete six-mode report."""
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np

from power_fm_foresight._config import ROOT, load_config, read_json, sha256, write_json
from power_fm_foresight.scoring import UNITS
from experiments.evaluate_stage_b import locations


MODES = ("C", "P", "P0", "S", "A", "D")
METRICS = ("mae", "mse", "crps", "coverage_50", "coverage_90", "width_50", "width_90")


def comparison_figure(root, results, buses):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    origins = read_json(root / "protocol.json")["origins"]
    colors = {"C": "#666666", "P": "#a0a0a0", "P0": "#bdbdbd", "S": "#3468a4", "A": "#cb6c2a", "D": "#8060a8"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for m in MODES:
        axes[0, 0].bar(m, results[m]["joint_energy_score"], color=colors[m])
        axes[0, 1].bar(m, results[m]["headline_voltage"]["coverage_90"] * 100, color=colors[m])
    axes[0, 0].set(title="Joint energy score (lower is better)", ylabel="Train-scaled ES")
    axes[0, 1].axhline(90, color="black", linestyle="--", label="Nominal 90%")
    axes[0, 1].set(title="Seven headline buses: 90% Vm coverage", ylabel="Coverage (%)", ylim=(0, 105))
    axes[0, 1].legend(frameon=False)
    pit = {}
    for column, (key, indices, title) in enumerate((("vm", buses, "Headline-bus Vm: randomized rank PIT"), ("loading", [0], "Line-1 loading: randomized rank PIT"))):
        pit[key] = {}
        for mode in ("C", "S", "A", "D"):
            counts = np.zeros(10, dtype=int)
            relative = "controls/origins" if mode == "C" else "origins"
            for origin in origins:
                with np.load(root / relative / origin["id"] / f"{mode}_scores.npz") as arrays:
                    v = arrays[f"{key}__pit"][:, indices]
                    counts += np.histogram(v[np.isfinite(v)], bins=np.linspace(0, 1, 11))[0]
            pit[key][mode] = {"counts": counts.tolist(), "total": int(counts.sum())}
            axes[1, column].stairs(counts / counts.sum() * 10, np.linspace(0, 1, 11), label=mode, color=colors[mode], linewidth=2)
        axes[1, column].axhline(1, color="black", linestyle="--", linewidth=1)
        axes[1, column].set(title=title, xlabel="PIT", ylabel="Density", xlim=(0, 1))
        axes[1, column].legend(frameon=False)
    for axis in axes.flat:
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
    fig.suptitle("Stage B: 72 test days, frozen scorer; C/S/A/D use 64 joint samples", fontsize=14)
    fig.savefig(root / "comparison.png", dpi=170)
    plt.close(fig)
    write_json(root / "headline_pit.json", {"bus_ids": (buses + 1).tolist(), "line_ids": [1], "bin_edges": np.linspace(0, 1, 11).tolist(), "histograms": pit})


def subset(result, key, indices):
    values = result["variables"][key]["by_component"]
    weights = np.asarray(values["count"])[indices]
    return {m: float(np.average(np.asarray(values[m])[indices], weights=weights)) for m in METRICS}


def verify(root):
    original = read_json(root / "protocol.json")
    controls = read_json(root / "controls/protocol.json")
    assert read_json(root / "controls/job.json")["status"] == "complete"
    assert controls["original_protocol_sha256"] == sha256(root / "protocol.json")
    for path, digest in original["source_sha256"].items():
        assert sha256(ROOT / path) == digest
    assert controls["source_sha256"] == sha256(ROOT / "scripts/score_controls.py")
    assert controls["scorer_sha256"] == sha256(ROOT / "scoring.py")
    assert sha256(root / "controls/climatology_pool.npz") == controls["climatology_pool_sha256"]
    assert len(controls["climatology_blocks"]) == 210
    assert {b["split"] for b in controls["climatology_blocks"]} == {"train"}
    assert len({b["episode_id"] for b in controls["climatology_blocks"]}) == 10
    train_names = {b["episode_id"] for b in controls["climatology_blocks"]}
    assert not train_names.intersection(o["episode_id"] for o in original["origins"])
    artifacts = 0
    with np.load(root / "controls/climatology_pool.npz") as pool:
        for origin in original["origins"]:
            identity = origin["id"]
            for relative, protocol in (("origins", root / "protocol.json"), ("controls/origins", root / "controls/protocol.json")):
                folder = root / relative / identity
                done = read_json(folder / "done.json")
                assert done["protocol_sha256"] == sha256(protocol)
                for name, digest in done["artifacts"].items():
                    assert sha256(folder / name) == digest, str(folder / name)
                    artifacts += 1
            folder = root / "origins" / identity
            control = root / "controls/origins" / identity
            with np.load(folder / "A.npz") as a, np.load(folder / "D.npz") as d:
                for key in ("pd_mw", "qd_mvar"):
                    np.testing.assert_array_equal(a[key], d[key])
            with np.load(folder / "P.npz") as p, np.load(control / "P0.npz") as p0:
                for key in UNITS:
                    np.testing.assert_array_equal(np.repeat(p[key], 64, axis=0), p0[key])
            draws = read_json(control / "C_scores.json")["climatology_block_indices"]
            with np.load(control / "C.npz") as c:
                for key in UNITS:
                    np.testing.assert_array_equal(c[key], pool[key][draws])
            for mode in MODES:
                where = control if mode in ("C", "P0") else folder
                report = read_json(where / f"{mode}_scores.json")
                assert report["origin_id"] == identity
                assert report["samples"] == (1 if mode == "P" else 64)
                for key in UNITS:
                    with np.load(where / f"{mode}_scores.npz") as score:
                        assert int(score[f"{key}__valid"].sum()) == report["variables"][key]["count"]
                    assert sum(report["variables"][key]["pit_histogram"]) == report["variables"][key]["count"]
    suites = ET.parse(root / "tests.xml").getroot()
    tests = sum(int(e.attrib.get("tests", 0)) for e in suites.iter("testsuite"))
    failures = sum(int(e.attrib.get("failures", 0)) + int(e.attrib.get("errors", 0)) for e in suites.iter("testsuite"))
    assert tests == 6 and failures == 0
    control_suites = ET.parse(root / "controls_tests.xml").getroot()
    control_tests = sum(int(e.attrib.get("tests", 0)) for e in control_suites.iter("testsuite"))
    control_failures = sum(int(e.attrib.get("failures", 0)) + int(e.attrib.get("errors", 0)) for e in control_suites.iter("testsuite"))
    assert control_tests == 1 and control_failures == 0
    evidence = {"status": "passed", "origins": 72, "modes": list(MODES), "per_origin_hashes_verified": artifacts,
        "frozen_source_hashes_verified": True, "A_D_demand_exact_all_origins": True,
        "P0_exact_copies_all_origins": True, "C_exact_train_block_draws_all_origins": True,
        "C_pool_train_episodes": 10, "C_pool_daily_blocks": 210, "held_out_climatology_pool_overlap": 0,
        "P0_point_metric_max_abs_difference": read_json(root / "P0.json")["point_mass_metric_max_abs_difference"],
        "P0_energy_max_abs_difference": read_json(root / "P0.json")["point_mass_energy_max_abs_difference"],
        "P0_energy_max_relative_difference": read_json(root / "P0.json")["point_mass_energy_max_relative_difference"],
        "scorer_and_forecaster_tests": {"passed": tests + control_tests, "failures": failures + control_failures,
            "junit": [str(root / "tests.xml"), str(root / "controls_tests.xml")]}}
    write_json(root / "verification.json", evidence)
    return evidence


def main():
    cfg = load_config()
    root, _, train = locations(cfg)
    subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_evaluation_controls.py",
        "--junitxml", str(root / "controls_tests.xml"), "--basetemp", str(root / "control_test_tmp"),
        "-o", f"cache_dir={root / 'pytest_cache'}"], cwd=ROOT, check=True)
    verification = verify(root)
    results = {m: read_json(root / f"{m}.json") for m in MODES}
    ridge = read_json(root / "forecaster/manifest.json")["protocol"]
    active = np.asarray(ridge["active_bus_ids"]) - 1
    buses = np.asarray(cfg["stage_b"]["calibration"]["headline_voltage_bus_ids"]) - 1
    comparison_figure(root, results, buses)
    probes = read_json(root / "probe/report.json")
    metadata = read_json(train / "manifest.json")
    coverage = {label: {m: subset(r, key, indices) for m, r in results.items()} for label, key, indices in (
        ("Active Pd (MW)", "pd_mw", active), ("Headline Vm (p.u.)", "vm", buses), ("Line-1 loading (rating fraction)", "loading", [0]))}
    header = ["| Mode | Joint ES ↓ | Variogram ↓ | Active Pd MAE (MW) | Headline Vm MAE (p.u.) | Headline Vm CRPS | Vm 50% coverage | Vm 90% coverage | Vm 90% width | Line-1 loading MAE |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    table = list(header)
    for m, r in results.items():
        v = r["headline_voltage"]
        table.append(f"| {m} | {r['joint_energy_score']:.6f} | {r['joint_variogram_p0_5']['score']:.6f} | {r['active_pd_mae_mw']:.4f} | {v['mae']:.6f} | {v['crps']:.6f} | {v['coverage_50']:.1%} | {v['coverage_90']:.1%} | {v['width_90']:.6f} | {r['variables']['loading']['by_component']['mae'][0]:.5f} |")
    lines = ["# Stage B — complete evaluation and controls", "",
        "The frozen forecasting evaluation, embedding probe, and requested climatology/zero-spread controls are complete. Completion describes the evidence gate; it does not establish successful calibration. All results are raw predictions unless explicitly labeled solver-refined. No F run, post-hoc probability map, or hybrid warm-start experiment was performed in this stage.", "",
        "**Protocol and methods.** temporal_case30_v3 has 10 train / 2 selection / 4 calibration / 4 test episodes. Each episode has 504 hours. All methods use the same 72 midnight test origins (1,728 forecast hours), 72-hour history where applicable, and 24-hour horizon. These are correlated days from four held-out source blocks, not 72 independent seasonal replicates. The calibration partition was not used for these forecasting fits or scores.", "",
        "| Mode | Definition | Samples per origin |", "|---|---|---:|",
        "| C | Training-only empirical joint demand + solved-state climatology; 210 whole midnight-aligned daily blocks from all 21 days of each of 10 training episodes; no history conditioning or GENCO inference | 64 |",
        "| P | Per-active-bus ridge mean demand → frozen GENCO | 1 |",
        "| P0 | 64 exact copies of P; a zero-spread scorer control | 64 identical |",
        "| S | Ridge mean plus training-only whole 24-hour/all-bus residual blocks → frozen GENCO | 64 |",
        "| A | Selected epoch-18 temporal adapter around frozen GENCO, with each sampled demand passed through its own conditioned graph | 64 |",
        "| D | A's exact Pd and Qd samples → unconditioned frozen GENCO | 64 |", "",
        "Ridge has 21 separate bus models with 24 direct horizon heads, lags 1/2/24/48, past Pd/Qd/Vm/Va/bus Pg/bus Qg, availability masks and calendar features. Final fitting uses 4,090 hourly train windows. The 3,130 overlapping rolling residual blocks use 313 origins per training episode; each rolling fit includes only target windows ending before its origin. This is the explicitly approved v3 count, replacing the obsolete 3,756 count from 12 train episodes. Each C draw selects one training block for every component and hour together; the 64 block indices are sampled with replacement. C and P0 were added after the original evaluation on request; no data split, model fit, checkpoint choice or scorer was changed.", "",
        "The energy score uses the fair all-off-diagonal-pair U-statistic, train-derived coordinate scales, and division by the square root of the number of observed coordinates. Fair marginal CRPS is in physical units; P uses absolute error. Variogram p=0.5 uses fixed adjacent-hour and connected-bus pairs for the scaled joint score; physical per-variable scores use adjacent-hour pairs. Interval endpoints use linear empirical quantiles. Missing reference values remain masked. The energy score measures a joint trajectory; it is not an event calibration score.", "",
        "**Gate B table.** Vm uses buses {1,9,11,13,15,21,22}; Pd uses the 21 active demand buses. Loading is a fraction of rating. Lower scores/errors are better; coverage should approach its nominal level.", "", *table, "",
        "![Score, coverage and headline PIT comparison](comparison.png)", "",
        "**Coverage and interval widths for every mode.** Width has the variable's physical unit; P/P0 retain zero-width intervals. The tables deliberately avoid dilution by zero-demand buses."]
    for label, group in coverage.items():
        lines += ["", f"**{label}**", "", "| Mode | 50% coverage | 50% width | 90% coverage | 90% width |", "|---|---:|---:|---:|---:|"]
        for m, v in group.items():
            lines.append(f"| {m} | {v['coverage_50']:.2%} | {v['width_50']:.6g} | {v['coverage_90']:.2%} | {v['width_90']:.6g} |")
    c, a, s, d = [results[m] for m in ("C", "A", "S", "D")]
    lines += ["", "**Interpretation and limits.**",
        f"A's joint ES is {a['joint_energy_score']:.6f}, versus {s['joint_energy_score']:.6f} for S, {d['joint_energy_score']:.6f} for D, and {c['joint_energy_score']:.6f} for unconditional C. A and D have identical demand samples, so their difference reflects conditioning/state innovations and their downstream physics rather than improved demand forecasts. A's demand MAE ({a['active_pd_mae_mw']:.4f} MW) exceeds ridge P's ({results['P']['active_pd_mae_mw']:.4f} MW).",
        f"A's nominal 90% headline-Vm interval covers only {a['headline_voltage']['coverage_90']:.1%}; S covers {s['headline_voltage']['coverage_90']:.1%}, D {d['headline_voltage']['coverage_90']:.1%}, and C {c['headline_voltage']['coverage_90']:.1%}. These results do not support a claim that A is calibrated. Unconditional C has better headline Vm CRPS ({c['headline_voltage']['crps']:.6f}) than A ({a['headline_voltage']['crps']:.6f}), and better line-1 loading MAE ({c['variables']['loading']['by_component']['mae'][0]:.5f} versus {a['variables']['loading']['by_component']['mae'][0]:.5f}). Thus A does not dominate the climatology baseline even though its joint ES and active-demand MAE improve. Per-variable physical metrics below expose errors hidden by a single scaled joint score. No architecture/scales were tuned after observing these results.",
        "The empirical sample pools contain overlapping or serially correlated days. Finite-ensemble fair-score corrections address the sampling estimator, not the uncertainty caused by reusing a small empirical training pool. Event reliability/Brier scores, prevalence-shift panels, calibration maps, and solver warm-start timing belong to the next stage. Fixed case30 topology, synthetic demand, benchmark costs, fixed commitment and no ramps limit generality.", "",
        "**All-variable forecast scores in physical units.** These tables include all defined components (30 buses, 6 generators, 41 rated branches), including fixed coordinates. Accordingly, all-bus demand coverage is inflated by inactive buses; use the active-Pd table above for the headline. MSE uses squared units. Physical ES uses the variable unit; with p=0.5 the physical variogram score has that unit as well."]
    for key, unit in UNITS.items():
        lines += ["", f"**{key} — {unit}**", "", "| Mode | Count | MAE | MSE | CRPS | Physical ES | Physical variogram | 50% cov. | 50% width | 90% cov. | 90% width |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for m, r in results.items():
            v = r["variables"][key]
            lines.append(f"| {m} | {v['count']} | {v['mae']:.6g} | {v['mse']:.6g} | {v['crps']:.6g} | {v['energy_score_physical']:.6g} | {v['variogram_p0_5_physical']['score']:.6g} | {v['coverage_50']:.2%} | {v['width_50']:.6g} | {v['coverage_90']:.2%} | {v['width_90']:.6g} |")
    lines += ["", "**PIT.** Ten bins [0,.1,...,1]; counts below cover all physically defined components. Individual masked PIT values are saved in each origin's `*_scores.npz`. The scorer uses randomized finite-ensemble ranks `(less + U*(equal+1))/(M+1)`. P has M=1 and therefore a coarsely randomized rank diagnostic; its PIT is not directly comparable to a continuous forecast PIT. P0 uses M=64 and exposes the zero-spread extremes. P and P0 must match scores/intervals, but their finite-M PIT is not expected to match. Ties at fixed/bounded coordinates can also obscure aggregate PIT; marginal intervals above use the frozen headline subsets.", "",
        "| Variable | Mode | Ten bin counts |", "|---|---|---|"]
    for key in UNITS:
        for m, r in results.items():
            lines.append(f"| {key} | {m} | {', '.join(map(str, r['variables'][key]['pit_histogram']))} |")
    lines += ["", "**Same-input GENCO approximation, separate from forecasting.** For all 72 origins, solve leads 1/7/13/19/24 and samples 0/32 (P: sample 0 only). There are 1,800 unique Ipopt requests; 1,797 resolve. The three unresolved S requests are masked, not labeled infeasible. D reuses A's identical inputs and references. P/S have different inputs, so only A/D form an exact input-matched comparison. The following are all-bus/all-generator/all-line paired errors, not errors against realized future demand.", "",
        "| Mode | Resolved / attempted | Vm MAE (p.u.) | Pg MAE (MW) | Loading MAE |", "|---|---:|---:|---:|---:|"]
    for m in ("P", "S", "A", "D"):
        v = results[m]["same_input_approximation"]; e = v["errors"]
        lines.append(f"| {m} | {v['resolved']} / {v['attempts']} | {e['vm']['mae']:.6f} | {e['pg_mw']['mae']:.4f} | {e['loading']['mae']:.6f} |")
    lines += ["", "C is empirical solved-state climatology and has no GENCO approximation metric; P0 inherits P's identical states. A improves the loading approximation relative to D but has worse paired Vm/Pg errors. A lower joint forecast score does not establish a uniformly better snapshot solver.", "",
        "**Raw versus solver-refined forecast errors on the matched subset only.** Predictions are means over the subset's 1 (P) or 2 (S/A/D) samples at five leads. Hours with any unresolved selected reference are masked. These smaller-ensemble numbers must not be compared as if they were the full 64-sample scores. A and D refined results are identical by construction.", "",
        "| Mode | Version | Vm MAE (p.u.) | Pg MAE (MW) | Loading MAE |", "|---|---|---:|---:|---:|"]
    for m in ("P", "S", "A", "D"):
        for version, e in results[m]["raw_vs_refined_forecast_subset"].items():
            lines.append(f"| {m} | {version} | {e['vm']['mae']:.6f} | {e['pg_mw']['mae']:.4f} | {e['loading']['mae']:.6f} |")
    lines += ["", "All paired MAE/MSE and coordinate counts, including angles/Qg/both-end branch flows, remain in the mode JSON files. Mean absolute bus power-balance residuals across every raw sample/hour/bus:", "",
        "| Mode | P residual (MW) | Q residual (MVAr) |", "|---|---:|---:|"]
    for m, r in results.items():
        b = r["physical_balance"]
        lines.append(f"| {m} | {b['p_mean_abs_mw']:.6g} | {b['q_mean_abs_mvar']:.6g} |")
    lines += ["", "A still has a mean absolute active-power imbalance of 5.46 MW per bus across raw samples; its outputs are not certified feasible AC-OPF solutions. C retains numerical feasibility for its own sampled training demands, not for the realized test demands. Solver refinement is a separate computation.", "", "**Frozen-embedding probe (run after the main evaluation).** Linear heads predict Vm/Va/Pg/Qg at t+1 and t+24 without crossing episodes. Raw features are current whole-grid Pd/Qd/Vm/Va/Pg/Qg plus current/future calendar (142 features). The comparison adds the target node's 384 final frozen GENCO latents (526 features), obtained with the official `return_embeddings=True` API. Train-only standardization; alpha from {.1,1,10,100}, selected on selection episodes. Different feature dimensions remain a limitation. This measures incremental predictive association, not causality.", "",
        "| Horizon | Train / selection / test hours | Raw normalized MSE | Raw + GENCO normalized MSE | Relative change |", "|---|---|---:|---:|---:|"]
    for h, p in probes["horizons"].items():
        raw = p["families"]["raw"]["normalized"]["mse"]; emb = p["families"]["raw_plus_genco"]["normalized"]["mse"]
        lines.append(f"| {h} h | {p['train_hours']} / {p['selection_hours']} / {p['test_hours']} | {raw:.6f} | {emb:.6f} | {(emb / raw - 1):+.2%} |")
    lines += ["", "Adding the embeddings worsens test MSE at both horizons; retain this negative result. Physical per-variable probe errors and per-test-episode scores are in `probe/report.json`.", "", "![Embedding probe](probe/embedding_probe.png)", "",
        "**Training, provenance and verification.** A completed seven one-hour `debug` allocations with walltime guards: 78,135 updates, 19 full epochs plus 425 updates; best selection epoch 18, ES 0.3996551865. Cumulative training-job wall was 21,585.3 s (5h59m45s), queue wait 1,268.7 s, calendar elapsed 22,994.0 s. The six-hour budget ended training; these are not evidence of early stopping. Historical-cache generation and resume tests are separate preparation costs.", "",
        f"Measured trainable parameters: **{metadata['trainable_parameters']:,}**. Backbone state-dict SHA256: `{metadata['backbone_sha256']}`. These disagree with the supplied 92,345 / `9dbf...` values; the manifest records the discrepancy and retains the original training manifest. The frozen backbone checksum matches the training chain and is checked before/after every evaluation worker. Selected checkpoint SHA256: `{metadata['evaluation_provenance']['best_checkpoint_sha256']}`.", "",
        f"Verification passed: {verification['scorer_and_forecaster_tests']['passed']} scorer/forecaster tests; {verification['per_origin_hashes_verified']} saved per-origin artifact hashes; frozen source hashes; exact A/D demand identity at every origin; exact P0 replication; C draws reconstructed from training-only blocks. Maximum P/P0 marginal/variogram/interval difference across origins/variables: {verification['P0_point_metric_max_abs_difference']:.3g}. Maximum energy-score difference: {verification['P0_energy_max_abs_difference']:.3g} absolute / {verification['P0_energy_max_relative_difference']:.3g} relative. This is consistent with the frozen multi-sample ES count-normalization square root using float32, while P uses float64; the scorer was not modified. The initial overly strict control assertion and partial outputs are preserved under `controls_attempt01_float_precision/`. All 20 dataset shard hashes match. The frozen CPU output-conversion check matches official normalizer inversion bit-for-bit on the same machine. XPU evaluation used 12 tile workers and retained each origin atomically. Climatology/control scoring used 12 CPU processes on an allocated node. No environment packages were installed or changed in this evaluation work.", "",
        "**Evidence and reproduction.** All bulk artifacts remain on `/lus/flare`; the source checkout contains code and small documentation only.", "",
        "| Evidence | Path relative to this report |", "|---|---|",
        "| Frozen evaluation contract | [protocol.json](protocol.json) |",
        "| Full per-variable/component/origin scores | [C.json](C.json), [P.json](P.json), [P0.json](P0.json), [S.json](S.json), [A.json](A.json), [D.json](D.json) |",
        "| Ridge coefficients and block pool provenance | [forecaster/manifest.json](forecaster/manifest.json) |",
        "| Additional control pool/draw protocol | [controls/protocol.json](controls/protocol.json) |",
        "| Verification and test evidence | [verification.json](verification.json), [tests.xml](tests.xml), [dataset_integrity.json](dataset_integrity.json), [frozen_inverse_check.json](frozen_inverse_check.json) |",
        "| Evaluation job | [evaluation_job.json](evaluation_job.json) |",
        "| Control job | [controls/job.json](controls/job.json) |",
        "| Probe report / coefficients / predictions | [probe/report.json](probe/report.json), `probe/coefficients/`, `probe/predictions_h*.npz` |",
        "| Headline-only PIT histograms and comparison figure | [headline_pit.json](headline_pit.json), [comparison.png](comparison.png) |",
        "| Per-origin samples, targets, scored arrays, solver audits, hashes | `origins/<id>/` and `controls/origins/<id>/` |",
        f"| Adapter manifest / selection curve / chain | [{train / 'manifest.json'}]({train / 'manifest.json'}), [{train / 'epochs.json'}]({train / 'epochs.json'}), [{train / 'chain.md'}]({train / 'chain.md'}) |", "",
        "On an allocated node in the existing Aurora environment, run the following in order. Completed origins resume only after matching protocol/artifact hashes. The probe uses saved histories; the last command recomputes this report from result files and validates their hashes.", "",
        "```bash", "python -m scripts.evaluate_stage_b run --device xpu", "python -m scripts.embedding_probe", "python -m scripts.score_controls", "python -m scripts.report_stage_b", "```", "",
        "PBS entry points are `scripts/evaluate_stage_b.pbs` and `scripts/stage_a.pbs` with `GENCO_STAGE_A_MODULE=scripts.embedding_probe`, `scripts.score_controls`, or `scripts.report_stage_b`. The full run is scientific reproduction, not a <30-minute clean-checkout demo; final packaging remains a later deliverable."]
    (root / "full_report.md").write_text("\n".join(lines) + "\n")
    write_json(root / "full_report.json", {"status": "complete", "mode_order": list(MODES),
        "coverage": coverage, "verification": verification, "result_sha256": {m: sha256(root / f"{m}.json") for m in MODES},
        "probe_report_sha256": sha256(root / "probe/report.json"), "report_generator_sha256": sha256(__file__),
        "markdown_sha256": sha256(root / "full_report.md"), "scientific_verdict": "Evaluation complete; learned intervals substantially undercover; no calibrated-risk claim supported"})
    gate = read_json(root / "gate_b.json")
    gate["additional_controls"] = {m: sha256(root / f"{m}.json") for m in ("C", "P0")}
    gate["full_report"] = str(root / "full_report.md")
    write_json(root / "gate_b.json", gate)
    prior = (root / "gate_b.md").read_text().split("<!-- controls -->")[0]
    (root / "gate_b.md").write_text(prior + "\n<!-- controls -->\n\n## Requested additional controls\n\n" + "\n".join(table) + "\n\nSee [the full six-mode report](full_report.md) for all coverage/widths, physical-variable scores, PIT, approximation/refinement, probe and verification results.\n")
    print("\n".join(table), flush=True)
    print({"report": str(root / "full_report.md"), "verification": verification}, flush=True)


if __name__ == "__main__":
    main()
