"""One v3/v4 comparison figure, rendered only from frozen aggregate reports."""
import argparse
import os
from pathlib import Path

from scripts.headline_numbers import collect
from common import read_json, sha256, write_json

MODES = ("A", "S", "C", "D")
COLORS = {"A": "#b56613", "S": "#287ead", "C": "#626262", "D": "#27836b"}


def render(root, output):
    os.environ.setdefault("MPLCONFIGDIR", str(output / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    output.mkdir(parents=True, exist_ok=True)
    records = collect(root)["headlines"]
    metrics = (("vm_crps", "Headline Vm CRPS (p.u.) ↓", ".5f"),
               ("vm_coverage90_pct", "Vm interval coverage (%) · nominal 90%", ".1f"),
               ("pg_free_mae", "Free-generator Pg MAE (MW) ↓", ".2f"),
               ("p_imbalance", "Mean |P imbalance| (MW/bus), log scale ↓", ".3g"))
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(4, 2, figsize=(13.4, 11.4))
    for i, (key, label, fmt) in enumerate(metrics):
        both = [records[f"{version}.{mode}.{key}"]["value"] for version in ("v3", "v4") for mode in MODES]
        for j, version in enumerate(("v3", "v4")):
            ax = axes[i, j]
            values = [records[f"{version}.{mode}.{key}"]["value"] for mode in MODES]
            if key == "p_imbalance":
                assert min(both) > 0, "Log-axis values must be strictly positive; never floor silently"
                ax.set_xscale("log")
                ax.set_xlim(min(both)/4, max(both)*30)
            else:
                ax.set_xlim(0, 125 if "coverage" in key else max(both)*1.35)
            for y, (mode, value) in enumerate(zip(MODES, values)):
                ax.scatter(value, y, color=COLORS[mode], s=65, zorder=3)
                ax.text(.99, y, format(value, fmt), transform=ax.get_yaxis_transform(), va="center", ha="right",
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
            if "coverage" in key:
                ax.axvline(90, color="#777", linestyle=":", linewidth=1.2)
            ax.set_yticks(np.arange(4), MODES); ax.set_ylim(3.6, -.6)
            ax.grid(axis="x", alpha=.2); ax.set_xlabel(label)
            if i == 0:
                ax.set_title("v3 · nominal load 0.30–1.00 · λ = 0.01" if j == 0 else "v4 · nominal load 0.80–1.00 · λ = 1.0",
                             fontsize=12, pad=14)
    fig.suptitle("Post-training trade-offs on case30 · frozen test results", fontsize=16, y=.985)
    caption = ("V4 changes both nominal load range and physics-loss weight after v3 results; this comparison is confounded, not an ablation. "
               "Absolute joint ES is not comparable across datasets because train-derived scales and targets differ. "
               "Vm uses the seven predeclared buses; Pg uses the two free generators. Balance uses each sample's own demand. "
               "Climatology draws solved states, so its near-zero residual is expected. V4 unresolved state labels stay masked.")
    fig.text(.5, .015, "Two changes: nominal load range and λ. No causal attribution to either change; absolute ES is not comparable across datasets.\n"
             "Shared physical-unit axes; P imbalance is logarithmic. C samples solved states. Missing v4 state labels remain masked.",
             ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .96))
    path = output / "figure_7_v3_v4.png"
    fig.savefig(path, dpi=180, facecolor="white"); plt.close(fig)
    inputs = {str(p.relative_to(root)): sha256(p) for p in (
        root / "stage_b/diagnostics/report.json", root / "stage_c/report.json",
        root / "v4/stage_b/report.json", root / "stage_d/final_release/context.json")}
    write_json(output / "comparison_manifest.json", {"path": str(path), "sha256": sha256(path), "caption": caption,
        "inputs": inputs, "headline_keys": [f"{v}.{m}.{k}" for v in ("v3", "v4") for m in MODES for k, _, _ in metrics],
        "renderer_sha256": sha256(Path(__file__)), "new_experiments": False})
    return path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs"))
    a = p.parse_args()
    print(render(a.root, a.root / "stage_d/final_release"))
