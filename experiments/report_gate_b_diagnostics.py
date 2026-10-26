"""Present the immutable diagnostic JSON, including all headline coverage panels.

Separate presentation pass: no metric calculations, model calls, or changes to
the frozen scorer or to the diagnostic worker protocol.
"""
import xml.etree.ElementTree as ET

import numpy as np

from power_fm_foresight._config import load_config, read_json, sha256, write_json
from experiments.diagnose_stage_b import locations, render


def main():
    root, out = locations(load_config())
    result = read_json(out / "report.json")
    protocol = read_json(out / "protocol.json")
    scales = read_json(out / "coordinate_scales.json")
    assert all(sha256(p) == h for p, h in protocol["immutable_inputs"].items())
    render(out, result, scales)
    text = (out / "report.md").read_text()
    # Literal vertical bars inside Markdown cells would create spurious columns.
    text = text.replace("|P|", "abs(P)").replace("|Q|", "abs(Q)")
    counts = []
    for name in protocol["reference_series"]:
        ref = [v for o in protocol["origins"] if o["episode_id"] == name
               for v in protocol["reference_series"][name][o["origin"]:o["origin"] + 24]]
        counts.append((name, sum(v >= .8 for v in ref)))
    note = ("The retained hours by test episode are " + ", ".join(f"{e}: {n}" for e, n in counts)
            + ". None of the 72 complete forecast days stays entirely above 0.8. "
              "Thus conditional scores are partial-trajectory diagnostics, dominated by one source episode, not 35 independent complete high-load days.\n\n")
    marker = "**P binned same-input approximation.**"
    text = text.replace(marker, note + marker)
    for scope, heading in (("all", "## All forecast hours"), ("ref_ge_0_8", "## Secondary: realized ref_t >= 0.8")):
        lines = ["**Headline intervals.** Active Pd excludes inactive buses; Vm uses buses {1,9,11,13,15,21,22}; loading uses line 1. Entries are 50% / 90% coverage, followed by 50% / 90% interval width in the variable's physical units.", "",
                 "| Mode | Active Pd coverage | Pd width MW | Vm coverage | Vm width p.u. | Line-1 coverage | Loading width |",
                 "|---|---|---|---|---|---|---|"]
        for mode, row in result["scopes"][scope]["modes"].items():
            cells = [mode]
            for name in ("active_pd", "vm", "line_1"):
                v = row["headline"][name]
                cells.extend([f"{v['coverage_50']:.2%} / {v['coverage_90']:.2%}", f"{v['width_50']:.6g} / {v['width_90']:.6g}"])
            lines.append("| " + " | ".join(cells) + " |")
        lines.extend(["", ""])
        start = text.index(heading)
        position = text.index("**Raw balance and Pg forecasts.**", start)
        text = text[:position] + "\n".join(lines) + text[position:]
    modes = result["scopes"]["all"]["modes"]
    contributions = {m: modes[m]["groups"]["other"]["joint_energy_contribution"] / modes[m]["scores"]["joint_energy_score"] for m in ("S", "D")}
    findings = ("**Interpretation.** The profile diagnostic supports a strong load-regime association, not a causal OOD finding: "
        "P's same-input six-generator Pg MAE is 13.7604 / 6.16094 / 1.01409 MW across the three bins. "
        f"Under the documented additive allocation, flow coordinates account for {contributions['S']:.2%} of S's full-set joint ES and {contributions['D']:.2%} of D's. "
        "A retains the lowest joint ES on the >=0.8 slice (0.393577 versus S 0.500862), but S has better headline Vm CRPS "
        "(0.00201205 versus A 0.00280054), better 90% Vm coverage (40.34% versus 24.77%), and lower raw active imbalance "
        "(0.793555 versus A 3.34799 MW/bus). A's aggregate joint-score advantage is not a uniform improvement across quantities or a calibration result.\n\n")
    text = text.replace("**Retrospective conditioning.**", findings + "**Retrospective conditioning.**")
    text = text.replace("Qd is scored as a physical marginal but contributes no joint coordinates.",
                        "Qd is a deterministic baseline-power-factor copy of Pd; it is scored as a physical marginal but contributes no joint coordinates.")
    text = text.replace("`python -m scripts.diagnose_stage_b`.",
                        "`python -m scripts.diagnose_stage_b`, then `python -m scripts.report_gate_b_diagnostics`.")
    suites = ET.parse(out / "tests.xml").getroot()
    tests = sum(int(e.attrib.get("tests", 0)) for e in suites.iter("testsuite"))
    failures = sum(int(e.attrib.get(k, 0)) for e in suites.iter("testsuite") for k in ("failures", "errors"))
    assert tests == 3 and failures == 0
    text += f"\nThree diagnostic tests passed (masking/ES closure, profile reconstruction, and bus-residual aggregation). The CPU job completed with all 12 workers returning zero. See [tests.xml](tests.xml) and [job.json](job.json).\n"
    (out / "report.md").write_text(text)
    # Check Markdown tables have a consistent number of columns after formatting.
    expected = None
    for line in text.splitlines():
        if line.startswith("|"):
            count = line.count("|")
            if expected is None:
                expected = count
            assert count == expected, line
        else:
            expected = None
    write_json(out / "presentation.json", {"source_sha256": sha256(__file__),
        "report_json_sha256": sha256(out / "report.json"), "report_md_sha256": sha256(out / "report.md"),
        "tests": {"passed": tests, "failures": failures, "junit_sha256": sha256(out / "tests.xml")},
        "conditional_hours_per_episode": dict(counts), "conditional_full_days": 0,
        "flow_fraction_of_full_joint_ES_under_allocation": contributions,
        "original_global_input_hashes_unchanged": True})
    print(out / "report.md")


if __name__ == "__main__":
    main()
