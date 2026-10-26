"""Static Matplotlib figures from saved v3 evidence; no model fitting or rescoring."""
from pathlib import Path
import os
import argparse

os.environ.setdefault("MPLCONFIGDIR", "/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/stage_d/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import networkx as nx
import numpy as np

from power_fm_foresight._config import ROOT, load_config, read_json, sha256, write_json
from power_fm_foresight._data.network import load_network
from monitor import event_probabilities
from experiments.diagnose_stage_b import npz

MODES = ("C", "P", "P0", "S", "A", "D")
COLORS = {"C": "#777777", "P": "#be6151", "P0": "#be6151", "S": "#3988b7", "A": "#dd8b26", "D": "#3b9b78"}
STYLE = {"C": "--", "P": "-", "P0": ":", "S": "-", "A": "-", "D": "-."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-pending-hybrid", action="store_true")
    args = parser.parse_args()
    cfg = load_config()
    root = Path(cfg["paths"]["runs_root"])
    b, c, out = root / "stage_b", root / "stage_c", root / "stage_d"
    dest = out / "figures"
    dest.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(out / "matplotlib"))
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                        "figure.facecolor": "white", "savefig.facecolor": "white"})
    report, diag = read_json(c / "report.json"), read_json(b / "diagnostics/report.json")
    hybrid = read_json(c / "hybrid/report.json") if (c / "hybrid/report.json").exists() else None
    if hybrid is None and not args.allow_pending_hybrid:
        raise FileNotFoundError("Hybrid result is pending; explicitly request partial rendering to omit Figure 5")
    evidence = [c / "report.json", c / "maps.json", b / "diagnostics/report.json", b / "probe/report.json"]
    if hybrid is not None: evidence.append(c / "hybrid/report.json")
    figures = {}
    def save(fig, name, caption, inputs):
        path = dest / (name + ".png")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        figures[name] = {"path": str(path), "sha256": sha256(path), "caption": caption, "inputs": inputs}

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 4.5); ax.axis("off")
    boxes = [(0.2, 2.5, "72 observed hours\nDemand + past solved states", "#dcebf4"),
             (4.2, 2.5, "GENCO + temporal adapter\n64 joint 24-hour trajectories", "#f7e4c6"),
             (8.2, 2.5, "Hourly operating stress\nRaw and calibration-mapped", "#dcebdd"),
             (0.2, .35, "Controls\nClimatology · ridge P/S", "#eeeeee"),
             (4.2, .35, "Calibration episodes only\nPooled sigmoid event maps", "#dcebdd"),
             (8.2, .35, "Realized demand arrives\nCold vs A mean → Ipopt", "#e6e3f0")]
    for x, y, label, color in boxes:
        ax.add_patch(FancyBboxPatch((x, y), 3.5, 1.1, boxstyle="round,pad=0.12", facecolor=color, edgecolor="#707070"))
        ax.text(x + 1.75, y + .55, label, ha="center", va="center", fontsize=11)
    for start, end in [((3.82, 3.05), (4.08, 3.05)), ((7.82, 3.05), (8.08, 3.05)),
                       ((5.95, 1.57), (9.3, 2.35))]:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": "#424242", "lw": 1.8})
    ax.set_title("GENCO Foresight: forecast first, measure calibration, verify with a solver", fontsize=15, pad=12)
    save(fig, "figure_1_workflow", "Past solved states are observed inputs. Future solved states are labels only. Fixed case30 topology, hourly independent AC-OPF, no ramps or commitment decisions.", ["configs/v3.yaml", "models/temporal.py", "monitor.py"])

    bins = ["lt_0_6", "0_6_to_0_8", "ge_0_8"]
    labels = ["<0.6", "0.6–0.8", "≥0.8"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, key, ylabel, multiplier in zip(axes, ("vm", "pg_mw", "imbalance"), ("Vm MAE (milli-p.u.)", "Pg MAE, all 6 generators (MW)", "Mean |P imbalance| / bus (MW)"), (1000, 1, 1)):
        vals = [diag["ood_P"][s]["same_input_errors"][key]["mae"] * multiplier if key != "imbalance" else diag["ood_P"][s]["balance_audited_hours"]["residual_p_mw"]["mean_abs"] for s in bins]
        ax.plot(range(3), vals, "o-", color=COLORS["P"], lw=2)
        ax.set_xticks(range(3), labels); ax.set_xlabel("Realized reference profile")
        ax.set_ylabel(ylabel); ax.set_ylim(bottom=0); ax.grid(alpha=.2)
        for i, value in enumerate(vals): ax.annotate(f"{value:.3g}", (i, value), xytext=(0, 8), textcoords="offset points", ha="center")
        ax.margins(y=.2)
    fig.suptitle("Frozen GENCO errors are larger in low-profile test hours", fontsize=15)
    fig.text(.5, -.02, "Retrospective profile bins: 1,057 / 382 / 289 forecast hours; 205 / 74 / 81 audited inputs. Only 16.72% of forecast hours have ref ≥0.8.", ha="center", fontsize=9)
    fig.tight_layout()
    save(fig, "figure_2_ood", "P only: neural and Ipopt solutions use identical predicted demand. Binning uses realized ref_t, an oracle diagnostic, not a forecast input or proof of complete input-support overlap. Imbalance uses the audited hours. Four Pg coordinates are fixed; free-Pg results remain in the diagnostics report.", [str(b / "diagnostics/report.json")])

    hourly = [r for r in report["events"] if r["resolution"] == "hourly"]
    fig, axes = plt.subplots(8, 3, figsize=(14, 23), gridspec_kw={"width_ratios": [1, 1, .65]})
    for row, panels in zip(hourly, axes):
        for ax, kind in zip(panels[:2], ("raw", "mapped")):
            ax.plot([0, 1], [0, 1], color="#bbbbbb", lw=1)
            for m in MODES:
                supported = [bin for bin in row["modes"][m][kind]["bins"] if bin["plot_supported"]]
                ax.plot([v["mean_probability"] for v in supported], [v["observed_frequency"] for v in supported],
                    linestyle=STYLE[m], marker="o", ms=3, color=COLORS[m], label=m)
            ax.set_xlim(-.03, 1.03); ax.set_ylim(-.03, 1.03); ax.grid(alpha=.15)
            ax.set_ylabel("Observed frequency"); ax.set_xlabel("Predicted probability")
            ax.set_title(f"{row['event']} · {kind}", fontsize=11)
            ax.text(.02, .96, f"Cal {row['calibration_prevalence']:.1%} · Test {row['test_prevalence']:.1%}",
                    transform=ax.transAxes, ha="left", va="top", fontsize=8, bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none"})
        panels[2].axis("off")
        panels[2].text(0, .8, f"Prevalence\n\nCalibration  {row['calibration_positive']}/{row['calibration_count']}\n                    {row['calibration_prevalence']:.1%}\nTest            {row['test_positive']}/{row['test_count']}\n                    {row['test_prevalence']:.1%}", va="top", fontsize=10)
        panels[2].text(0, .2, "SHIFT >1.5×\nMapped sensitivity only" if row["shift_exceeds_1_5"] else "Prevalence ratio ≤1.5×", fontsize=9)
    axes[0, 0].legend(ncol=6, fontsize=8, loc="lower right")
    fig.suptitle("Hourly event reliability: raw vs calibration-only sigmoid maps", fontsize=16, y=1.003)
    fig.tight_layout()
    save(fig, "figure_3_reliability", "72 calibration and 72 test days from four source episodes each. Each row prints calibration/test prevalence. Shifted panels are retained; raw results are primary and mapped results sensitivity analyses. Fixed bins [0,.1,.3,.5,.7,.9,1]; points require ≥20 samples. P/P0 event curves overlap. No independent-hour confidence claim.", [str(c / "report.json"), str(c / "maps.json")])

    daily = [r for r in report["events"] if r["resolution"] == "daily_any" and r["support"]["eligible"]]
    fig, axes = plt.subplots(len(daily), 3, figsize=(14, 2.9 * len(daily)), squeeze=False, gridspec_kw={"width_ratios": [1, 1, .65]})
    for row, panels in zip(daily, axes):
        for ax, kind in zip(panels[:2], ("raw", "mapped")):
            ax.plot([0, 1], [0, 1], color="#bbbbbb", lw=1)
            for m in MODES:
                bins_supported = [v for v in row["modes"][m][kind]["bins"] if v["plot_supported"]]
                ax.plot([v["mean_probability"] for v in bins_supported], [v["observed_frequency"] for v in bins_supported],
                        color=COLORS[m], linestyle=STYLE[m], marker="o", ms=3, label=m)
            ax.set_xlim(-.03, 1.03); ax.set_ylim(-.03, 1.03)
            ax.set_title(f"{row['event']} · {kind}", fontsize=11)
            ax.set_xlabel("Daily-any probability"); ax.set_ylabel("Observed daily frequency")
            ax.text(.02, .96, f"Cal {row['calibration_prevalence']:.1%} · Test {row['test_prevalence']:.1%}", transform=ax.transAxes, va="top", fontsize=8)
        panels[2].axis("off")
        panels[2].text(0, .8, f"Prevalence\n\nCalibration {row['calibration_positive']}/{row['calibration_count']} ({row['calibration_prevalence']:.1%})\nTest {row['test_positive']}/{row['test_count']} ({row['test_prevalence']:.1%})\n\n" + ("SHIFT >1.5×\nMapped sensitivity only" if row["shift_exceeds_1_5"] else "Prevalence ratio ≤1.5×"), va="top", fontsize=10)
    axes[0, 0].legend(ncol=6, fontsize=8, loc="lower right")
    fig.suptitle("Secondary daily-any reliability: calibration-supported events only", fontsize=15, y=1.003)
    fig.tight_layout()
    save(fig, "supplement_daily_reliability", "Daily unions are taken within joint trajectories. Only calibration prevalence 20–80% and ≥5 distinct days per class qualify; bus 1 is excluded by this predeclared rule. Fixed bins and ≥20-day plotting support can leave few points; all counts remain in report.json. Only four source episodes, no independence claim.", [str(c / "report.json"), str(c / "maps.json")])

    events = [r["event"] for r in hourly]
    fig, axes = plt.subplots(4, 2, figsize=(12, 12))
    for event, ax in zip(events, axes.flat):
        for m in MODES:
            r = report["continuous"]["test"][m][event]
            ax.plot(np.arange(10) / 10 + .05, np.array(r["pit_histogram"]) / r["count"], label=m, color=COLORS[m], linestyle=STYLE[m])
        ax.axhline(.1, color="#bbbbbb", lw=1); ax.set_ylim(bottom=0)
        ax.set_title(event); ax.set_xlabel("Randomized finite-ensemble PIT"); ax.set_ylabel("Fraction"); ax.grid(alpha=.15)
    axes[0, 0].legend(ncol=6, fontsize=8)
    fig.suptitle("Continuous calibration remains unmodified by event maps", fontsize=15)
    fig.tight_layout()
    save(fig, "figure_3b_pit", "All six modes, same frozen scorer. Continuous predictions are raw; mapping stress-event probabilities does not repair interval coverage. P has one sample, P0 has 64 repeated samples, so their randomized finite-ensemble ranks differ.", [str(c / "report.json")])
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for level, ax in zip((50, 90), axes):
        matrix = np.array([[report["continuous"]["test"][m][e][f"coverage_{level}"] for m in MODES] for e in events])
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        ax.set_xticks(range(6), MODES); ax.set_yticks(range(8), events); ax.set_title(f"Nominal {level}% coverage")
        for i in range(8):
            for j in range(6): ax.text(j, i, f"{matrix[i,j]:.1%}", ha="center", va="center", fontsize=9, color="white" if matrix[i,j] > .6 else "black")
    fig.tight_layout()
    save(fig, "figure_3c_coverage", "Test 50/90% coverage, all six modes, seven Vm buses and line-1 loading. Exact interval widths in coverage_table.md and report.json; no event-probability map changes these intervals.", [str(c / "report.json")])
    table = ["# Test continuous intervals", "", "| Coordinate | Mode | Coverage50 | Width50 | Coverage90 | Width90 |", "|---|---|---:|---:|---:|---:|"]
    for e in events:
        for m in MODES:
            r = report["continuous"]["test"][m][e]
            table.append(f"| {e} | {m} | {r['coverage_50']:.5f} | {r['width_50']:.6g} | {r['coverage_90']:.5f} | {r['width_90']:.6g} |")
    table += ["", "Vm widths are p.u.; line loading widths are fractions of rating."]
    (out / "coverage_table.md").write_text("\n".join(table) + "\n")

    net = load_network(config=load_config())
    origin = read_json(b / "protocol.json")["origins"][0]
    folder = b / "origins" / origin["id"]
    a = npz(folder / "A.npz")
    peak = int(a["pd_mw"].mean(0).sum(1).argmax())
    prob = event_probabilities(a, net, bus_ids=tuple(net.bus_ids), line_ids=tuple(np.flatnonzero(net.rated_branches)+1))
    graph = nx.Graph(); graph.add_nodes_from(range(net.n_bus)); graph.add_edges_from(net["branch"][:, :2].astype(int))
    pos = nx.spring_layout(graph, seed=42)
    edge_list = [tuple(v) for v in net["branch"][:, :2].astype(int)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    nx.draw_networkx_edges(graph, pos, ax=axes[0], edge_color="#bbbbbb")
    dots = nx.draw_networkx_nodes(graph, pos, ax=axes[0], node_color=prob["voltage"]["hourly"][peak], cmap="YlOrRd", vmin=0, vmax=1, node_size=260)
    texts = nx.draw_networkx_labels(graph, pos, labels={i: int(net.bus_ids[i]) for i in range(net.n_bus)}, ax=axes[0], font_size=8)
    for i, text in texts.items():
        text.set_color("white" if prob["voltage"]["hourly"][peak, i] > .6 else "black")
    edges = nx.draw_networkx_edges(graph, pos, ax=axes[1], edgelist=edge_list, edge_color=prob["thermal"]["hourly"][peak], edge_cmap=plt.cm.YlOrRd, edge_vmin=0, edge_vmax=1, width=3)
    nx.draw_networkx_nodes(graph, pos, ax=axes[1], node_color="#eeeeee", node_size=200)
    nx.draw_networkx_labels(graph, pos, labels={i: int(net.bus_ids[i]) for i in range(net.n_bus)}, ax=axes[1], font_size=8)
    for ax, title in zip(axes, ("Raw voltage-stress probability", "Raw thermal-stress probability")):
        ax.set_title(title); ax.axis("off")
    fig.colorbar(dots, ax=axes, shrink=.6, label="A probability")
    fig.suptitle(f"Case30 schematic · {origin['id']} · forecast peak lead {peak + 1}")
    save(fig, "figure_4_network", "Schematic, not geographic coordinates. First test origin fixed prospectively for illustration; peak chosen from A mean aggregate forecast demand. All buses/lines shown with raw probabilities; maps fit only to the frozen headline families are not extrapolated to other components.", [str(folder / "A.npz"), str(net.path)])
    evidence.append(folder / "A.npz")

    if hybrid is not None:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.3))
        arms = ("cold", "A_forecast_mean"); arm_labels = ("Cold", "A mean warm")
        means = [hybrid["arms"][arm]["iterations_all_returned"]["mean"] for arm in arms]
        axes[0].bar(arm_labels, means, color=["#777777", COLORS["A"]]); axes[0].set_ylabel("Mean iterations")
        n = hybrid["paired_hours"]
        online = np.array([hybrid["arms"][arm]["request_seconds"]["sum"] / n for arm in arms])
        prep = np.array([0, hybrid["preparation_seconds_per_day"]["sum"] / n])
        axes[1].bar(arm_labels, online, label="Online request", color="#3988b7")
        axes[1].bar(arm_labels, prep, bottom=online, label="Forecast + history", color=COLORS["A"])
        axes[1].legend(fontsize=8); axes[1].set_ylabel("Total work / realized hour (s)")
        success = [hybrid["arms"][arm]["success_rate"] * 100 for arm in arms]
        axes[2].bar(arm_labels, success, color=["#777777", COLORS["A"]]); axes[2].set_ylim(0, 110); axes[2].set_ylabel("Valid solve rate (%)")
        for i, value in enumerate(success): axes[2].text(i, value + 1, f"{value:.2f}%", ha="center")
        fig.suptitle(f"Two-arm hybrid: {n:,} paired realized hours, fixed solver budget", fontsize=15)
        fig.tight_layout()
        save(fig, "figure_5_hybrid", "Cold vs arithmetic mean of A's 64 forecast states; both solve the same realized demand. No advance OPF solves. Warm total includes measured history embeddings and forecast sampling amortized over 24 hours. Shared worker/JIT/model setup is reported separately. Success requires independent physical checks.", [str(c / "hybrid/report.json")])

    probe = read_json(b / "probe/report.json")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for shift, family, label, color in ((-.18, "raw", "Raw features (142)", "#777777"), (.18, "raw_plus_genco", "Raw + GENCO embeddings (526)", COLORS["A"])):
        values = [probe["horizons"][h]["families"][family]["normalized"]["mse"] for h in ("1", "24")]
        bars = ax.bar(np.arange(2) + shift, values, .36, label=label, color=color)
        ax.bar_label(bars, fmt="%.4f", fontsize=9, padding=3)
    ax.set_xticks([0, 1], ["t + 1 h", "t + 24 h"]); ax.set_ylabel("Train-scale normalized MSE"); ax.set_ylim(0, .24)
    ax.legend(fontsize=9); ax.set_title("Frozen embeddings did not improve the linear probe")
    fig.tight_layout()
    save(fig, "figure_6_probe", "Separate linear heads with ridge alpha selected on selection episodes. Raw features include the current solved state and known calendar; embedding model adds the target node's final 384 latents. Test counts 2,012 / 1,920. Unequal feature dimensions; association, not causality. Negative result retained.", [str(b / "probe/report.json")])

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    target = npz(folder / "targets.npz")
    for ax, key, index, label in zip(axes, ("pd_mw", "vm", "loading"), (None, 0, 0), ("Aggregate Pd (MW)", "Vm bus 1 (p.u.)", "Line 1 loading / rating")):
        y = target[key].sum(-1) if index is None else target[key][:, index]
        ax.plot(range(1, 25), y, color="black", lw=2, label="Realized")
        for m in ("P", "S", "A"):
            x = npz(folder / f"{m}.npz")[key]
            x = x.sum(-1) if index is None else x[..., index]
            ax.plot(range(1, 25), x.mean(0), color=COLORS[m], label=m)
            if m != "P":
                lo, hi = np.quantile(x, [.05, .95], axis=0)
                ax.fill_between(range(1, 25), lo, hi, color=COLORS[m], alpha=.16)
        ax.set_ylabel(label); ax.grid(alpha=.2)
    axes[0].legend(ncol=4); axes[-1].set_xlabel("Forecast lead (hours)")
    fig.suptitle(f"Illustrative first test day: {origin['id']} (raw 90% bands)")
    fig.tight_layout()
    save(fig, "supplement_fan", "First test origin, not selected for forecast quality. P mean and S/A means with central 90% ensemble bands. Raw states can violate power balance; numerical refinement is a separate experiment.", [str(folder / f"{m}.npz") for m in ("P", "S", "A")] + [str(folder / "targets.npz")])
    evidence += [folder / f"{m}.npz" for m in ("P", "S")] + [folder / "targets.npz"]
    write_json(out / "manifest.json", {"source_sha256": sha256(__file__), "matplotlib_version": matplotlib.__version__,
        "primary_record": "v3", "complete": hybrid is not None, "pending": [] if hybrid is not None else ["figure_5_hybrid"],
        "figures": figures, "evidence_sha256": {str(p): sha256(p) for p in evidence},
        "v4_status_at_render": read_json(root / "v4/pipeline.json") if (root / "v4/pipeline.json").exists() else None})
    lines = ["# Stage D: static figure package", "", "V3 is the primary record. All figures use Matplotlib and saved artifacts. Reproduce with `python -m scripts.figures_stage_d` in the recorded environment.", ""]
    if hybrid is None: lines += ["**Partial rendering: Figure 5 awaits the hybrid experiment.**", ""]
    for name, record in figures.items():
        lines += [f"## {name}", "", f"![{name}](figures/{name}.png)", "", record["caption"], ""]
    (out / "report.md").write_text("\n".join(lines))
    print({"figures": len(figures), "manifest": str(out / "manifest.json")}, flush=True)


if __name__ == "__main__":
    main()
