"""Draft review documents from named frozen evidence; never publish anything."""
import argparse
import json
from pathlib import Path
import shutil

from common import ROOT, load_config, read_json, sha256, write_json
from scripts.headline_numbers import collect, markdown_list

B = chr(96)
FENCE = B * 3


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--post", action="store_true", help="Render the post once, after validation and cards are ready")
    a = p.parse_args()
    cfg = load_config(ROOT / "config.yaml")
    runs = Path(cfg["paths"]["runs_root"])
    out = runs / "stage_d/final_release"
    bundle = Path(cfg["paths"]["data_root"]).parent / "reproducibility/v0.1.0-rc"
    record = collect(runs)
    H = record["headlines"]
    used = set()
    def n(key, fmt="", index=None):
        value = H[key]["value"]
        if index is not None:
            value = value[index]
        shown = format(value, fmt) if fmt else (json.dumps(value) if isinstance(value, (list, dict)) else str(value))
        used.add(key)
        suffix = "" if index is None else f"[{index}]"
        return shown + f"<!-- headline: {key}{suffix} -->"

    assets = out / "figures"
    assets.mkdir(exist_ok=True)
    figures = {}
    original = read_json(runs / "stage_d/manifest.json")["figures"]
    for name, item in original.items():
        source = Path(item["path"])
        assert sha256(source) == item["sha256"], source
        target = assets / f"{name}.{item['sha256'][:12]}.png"
        if not target.exists():
            shutil.copy2(source, target)
        figures[name] = {**item, "filename": target.name}
    item = read_json(out / "comparison_manifest.json")
    source = Path(item["path"])
    assert sha256(source) == item["sha256"]
    target = assets / f"figure_7_v3_v4.{item['sha256'][:12]}.png"
    if not target.exists():
        shutil.copy2(source, target)
    figures["figure_7_v3_v4"] = {**item, "filename": target.name}
    write_json(assets / "manifest.json", figures)
    link = ROOT / "submission/figures"
    if link.is_symlink():
        assert link.resolve() == assets.resolve()
    else:
        assert not link.exists(), "Do not replace an existing figure directory"
        link.symlink_to(assets, target_is_directory=True)
    def figure(name, alt):
        return f"![{alt}](figures/{figures[name]['filename']})"

    context = read_json(out / "context.json")
    pin = context["pins"]
    evaluation = read_json(runs / "v4/stage_b/report.json")
    base = evaluation["provenance"]

    def prevalence_table(version, resolution="hourly"):
        lines = ["| Event | Train | Selection | Calibration | Test |", "|---|---:|---:|---:|---:|"]
        for event in ("voltage_bus_1", "voltage_bus_9", "voltage_bus_11", "voltage_bus_13",
                      "voltage_bus_15", "voltage_bus_21", "voltage_bus_22", "thermal_line_1"):
            rates = H[f"{version}.prevalence.{event}.{resolution}_pct"]["value"]
            counts = H[f"{version}.prevalence.{event}.{resolution}_counts"]["value"]
            cells = [f"{rate:.2f}% ({row[0]}/{row[1]}; unknown {row[2]})" for rate, row in zip(rates, counts)]
            lines.append("| " + event + " | " + " | ".join(cells) + " |")
        return "\n".join(lines)

    if not a.post:
        dataset_card = f"""---
license: apache-2.0
language: en
task_categories:
- time-series-forecasting
---
# GENCO Foresight: temporal case30 v3 and v4

**Draft for review. Not uploaded.** HF organization: {B}<HF_ORG>{B}; repository:
{B}<HF_ORG>/genco-foresight-temporal-case30{B}. Participants: {B}<NAMES>{B}.
Affiliations: {B}<AFFILIATIONS>{B}. Contact: {B}<EMAIL>{B}. Code: {B}<GITHUB_REPOSITORY_URL>{B}.

## Data and intended use

Synthetic research benchmark, not an ERCOT grid model or dispatch/control product.
Each version has {H['setup.episodes']['value']} episodes of {H['setup.days_per_episode']['value']} days:
{H['v3.labels.attempted']['value']:,} hourly demand/AC-OPF snapshots, with train/selection/calibration/test
episode counts {H['setup.split_episodes']['value']}. Observe the preceding
{H['setup.history_hours']['value']} hours to forecast the following {H['setup.forecast_hours']['value']}.
Calibration and test each have {H['setup.test_days']['value']} midnight origins from four source episodes.
Adjacent days and overlapping histories are correlated.

The historical EIA ERCO {H['setup.profile_year']['value']} hourly aggregate profile is mapped onto
the fixed IEEE/PGLib case30 spatial demand pattern. The source profile, retrieval
provenance and hashes are included. This is a modeled geographic assumption.
Commitment, topology and benchmark costs are fixed; there are no ramps.
PowerModels/Ipopt labels include status, validity masks, residuals, objective,
duals, iterations and runtime. Numerical nonconvergence is not infeasibility.

V3 has {H['v3.labels.valid']['value']:,} valid labels; v4 has {H['v4.labels.valid']['value']:,}
valid labels and {H['v4.labels.unresolved']['value']} unresolved test hours
(test_02, hours 240 and 453). Observed demand remains available; missing solved
states are null/masked, never zeros or negative event labels.

## Construction and split history

All partitions use disjoint 21-day calendar-block pools. The final assignment is:

| Partition | One-based source blocks | Episodes |
|---|---|---:|
| Train | 1,4,8,9,12,16,17 | 10 |
| Selection | 5,13 | 2 |
| Calibration | 2,6,10,14 | 4 |
| Test | 3,7,11,15 | 4 |

Reuse stays within its assigned pool; only training reuses source blocks in the
final allocation. Eight trailing days of the source year are unused.

| Revision | Timing and reason |
|---|---|
| v1 | Initial 12/2/2/4 allocation and disjoint random source pools; pre-score audit exposed prevalence shifts. |
| v2 | Calendar-interleaved 12/2/2/4 pools; pre-score audit still found inadequate seasonal representation in two calibration source blocks. |
| v3 | Final 10/2/4/4 allocation, doubling calibration to four source blocks and 72 days. Prevalence became a reported diagnostic, not a training blocker. |
| v4 | Requested after v3 forecast scores. Same final partitions/noise/profile, nominal load range changes from 0.30–1.00 to 0.80–1.00. Adapter physics weight also changes from 0.01 to 1.0; this is a confounded two-change follow-up. |

V1→v3 design changes preceded forecasting scores; that disclosure does not make
this an untouched prospective benchmark. Full old datasets remain on Lustre;
the package includes their manifests/audit summaries.

Training-only affine profile scaling is applied to held-out profiles without
clipping. Both versions retain independent uniform bus noise ±3% and stationary
lognormal AR(1) factors (rho 0.9, innovation std 0.02). Qd uses baseline signed
power factor; buses with zero baseline Pd retain reactive-only baseline demand.
The original pretrained MVA normalizer is loaded, never refitted.

## Load range and support

OPFData nominal training multipliers were [0.8,1.2]. V3's nominal range extends
below this: only {H['v3.test_hours_high_profile']['value']} of 1,728 forecast hours have
realized ref_t ≥0.8. Frozen GENCO approximation errors increase in lower bins.
V4's 0.80–1.00 endpoints are nominal: noise leaves
{H['v4.test_bus_load_below_support_pct']['value']:.4f}% of test active-bus/hour multipliers
below 0.8, including the history hours. Reference-profile bins are retrospective
diagnostics, never future model inputs or proof of complete feature-support overlap.

## Event prevalence

Voltage stress means the outer 5% of the permitted range or beyond; thermal
stress means loading >95% of rating. Cells show rate, positive/resolved counts,
and unknown counts. Hourly tables use forecast-target hours after history.

### V3 hourly

{prevalence_table('v3')}

### V4 hourly

{prevalence_table('v4')}

V4's frozen headline events are degenerate on resolved labels in every partition:
buses 1/11/13 and line 1 are always positive; buses 9/15/21/22 always negative.
It cannot demonstrate event discrimination. No prevalence-based tuning follows.

### V3 daily-any

{prevalence_table('v3', 'daily_any')}

### V4 daily-any

{prevalence_table('v4', 'daily_any')}

Daily events are unions within joint trajectories. An observed positive resolves
a daily union as positive; a negative requires all hours to be known negative;
otherwise the day remains unknown. V3 daily reliability panels additionally
require calibration prevalence 20–80% and the predeclared class-support rule.
Calibration/test prevalence ratios >1.5 are printed, not suppressed.

## Files, use and reproducibility

The local package contains datasets/temporal_case30_v3 and
datasets/temporal_case30_v4, each with episodes/*.json, manifest.json, generation,
quality and pilot reports. Audits and event counts live in datasets/audit_v3 and
datasets/audit_v4. SHA256SUMS covers the release payload; provenance.json records
source paths/hashes. Episode order, source dates, seeds and sampling law are in
the manifests. Users must split by episode/source pool and preserve masks.

No future solved state is an input. Selection chooses checkpoints; calibration
fits probability maps; test outcomes are reporting-only. Case30 has one generator
per generator bus, so no multi-generator Qg allocation was required.
All copied/adapted source retains attribution in THIRD_PARTY_NOTICES.md.
The source code and generated artifacts use Apache-2.0; upstream network,
checkpoint and profile notices remain applicable. Cite the original GENCO,
OpenGridFM, OPFData and PGLib sources alongside this project.

Dataset cards are publication drafts; local availability does not imply a public
HF repository. All bulk artifacts stay on /lus/flare until the owner publishes.
"""
        (ROOT / "data/DATASET_CARD.md").write_text(dataset_card)
        state_rows = []
        for version in ("v3", "v4"):
            manifest = read_json(bundle / f"model/adapter_{version}/release_manifest.json")
            state_rows.append(f"| {version} | {manifest['best_epoch']} | {manifest['exported_adapter_file_sha256']} | {manifest['adapter_named_tensor_bytes_sha256']} |")
        model_card = f"""---
license: apache-2.0
library_name: pytorch
base_model: gridfm/genco-opfdata
tags:
- time-series
- power-systems
---
# GENCO Foresight: case30 post-training adapters v3 and v4

**Draft for review. Not uploaded.** HF organization: {B}<HF_ORG>{B}; repository:
{B}<HF_ORG>/genco-foresight-adapters{B}. Participants: {B}<NAMES>{B}.
Affiliations: {B}<AFFILIATIONS>{B}. Contact: {B}<EMAIL>{B}. Code: {B}<GITHUB_REPOSITORY_URL>{B}.

## Model and base reference

Both adapters use frozen {B}gridfm/genco-opfdata{B}, case30 seed 42,
revision {B}{pin['checkpoint']}{B}, directory case30_ieee/seed42.
Graphkit commit: {B}{pin['graphkit']}{B}; Datakit 1.1.0.
The original normalizer_stats.pt is included and is never refitted.
Base checkpoint file SHA256: {B}{base['base_checkpoint_file_sha256']}{B}.
Base named-tensor-bytes SHA256: {B}{base['base_named_tensor_bytes_sha256']}{B}.
These hash different byte streams; neither is substituted for the other.

| Adapter | Best selection epoch | Exported adapter file SHA256 | Named-tensor-bytes SHA256 |
|---|---:|---|---|
{chr(10).join(state_rows)}

Each adapter has 91,461 trainable parameters and 884 buffer elements, for 92,345
state-dict elements. Historical manifests that called 92,345 trainable remain
preserved; release_manifest.json records the corrected accounting. Each model
directory includes its original manifest, selection curve, chain log, resume-test
record, base reference and release manifest. V4 additionally includes all
per-origin evaluation hashes. Full forecasts/results are in evidence/runs.

## Method and scope

72 observed hours produce 64 joint 24-hour scenarios. Inputs are past
Pd/Qd/Vm/Va/Pg/Qg and availability masks, frozen final GENCO embeddings, and
known future calendar. Node-type GRUs have width 32; projections add memory
into the backbone's width-48 input latents. Rank-8-plus-diagonal innovations
drive demand and state readouts. Each scenario's own demand reaches its graph;
Qg and branch flows are reconstructed by the model's physics operations.
The backbone and checkpoint normalizer remain frozen.

Loss is fair train-scaled energy score plus normalized squared P/Q bus
imbalance. V3 uses lambda 0.01 and nominal profile endpoints 0.30–1.00; v4 uses
lambda 1.0 and endpoints 0.80–1.00. V4 changes both after seeing v3 scores:
it is not an ablation. Absolute ES across datasets is not a controlled comparison.
Both use one seed, fixed topology/costs/commitment and no ramps.

## Better forecast scores / worse physical consistency

Within v4, A has joint ES {H['v4.A.joint_es']['value']:.6f} versus S
{H['v4.S.joint_es']['value']:.6f} and C {H['v4.C.joint_es']['value']:.6f}.
Headline voltage CRPS is {H['v4.A.vm_crps']['value']:.6f} for A versus
{H['v4.S.vm_crps']['value']:.6f} for S. Demand-family ES still favors S.
Nominal 90% voltage intervals cover only {H['v4.A.vm_coverage90_pct']['value']:.2f}%.
Mean |P imbalance| is {H['v4.A.p_imbalance']['value']:.4f} MW/bus for A versus
{H['v4.D.p_imbalance']['value']:.4f} for D using A's identical demand samples through
frozen GENCO. D controls the demand input; it does not isolate conditioning from
the adapter's stochastic state readouts.

V3's A joint ES is {H['v3.A.joint_es']['value']:.6f}, but headline voltage coverage
is only {H['v3.A.vm_coverage90_pct']['value']:.2f}%. Frozen-GENCO identical-input
error rises below the checkpoint's nominal load regime. Calibration mapping
worsens A's aggregate voltage Brier. V3 warm starts provide no online solver
benefit and cost {H['v3.hybrid.work_ratio']['value']:.2f} times cold-start work including
forecast preparation. The embedding probe is negative. These are main results.

V4 retains two unresolved test labels and all eight frozen stress events are
degenerate. Neither model is demonstrated to provide calibrated operational
uncertainty or feasible dispatch. No deployment/control use is validated.

## Loading and validation

best.pt is a plain adapter state dictionary, without backbone or optimizer/RNG
state. Construct the repository's TemporalAdapter with the pinned frozen base
and the matching dataset's train-coordinate scales, then:

{FENCE}python
state = torch.load(path, map_location="cpu", weights_only=True)
missing, unexpected = adapter.load_state_dict(state, strict=False)
assert not unexpected
assert all(name.startswith("backbone.") for name in missing)
{FENCE}

This export cannot resume training and differs from the wrapped original
training checkpoint. Original checkpoint file hashes and export hashes are
recorded separately. Tensor hashing concatenates sorted UTF-8 names and
contiguous CPU NumPy tensor bytes, including buffers. Base weights are referenced,
not redistributed in this package.

The offline CPU reproduction rescores saved predictions and rebuilds figures;
it does not execute a fresh GENCO inference/training or OPF generation. Independent
non-Aurora and NVIDIA reproduction remain untested. Preserve vendor Torch and
Transformers on Aurora. No repository has been uploaded or published.
"""
        (ROOT / "reproducibility/MODEL_CARD.md").write_text(model_card)
    else:
        assert "reproduction.combined_seconds" in H, "Measure the combined replay before writing the review post"
        table = ["| Mode | Joint ES ↓ | Vm CRPS (p.u.) ↓ | Vm coverage90 | Free-Pg MAE (MW) ↓ |", "|---|---:|---:|---:|---:|"]
        for mode in ("A", "S", "C", "D"):
            table.append(f"| {mode} | {n(f'v4.{mode}.joint_es','.5f')} | {n(f'v4.{mode}.vm_crps','.6f')} | {n(f'v4.{mode}.vm_coverage90_pct','.2f')}% | {n(f'v4.{mode}.pg_free_mae','.3f')} |")
        family = ["| Family (coordinates/hour) | A | S | C | D |", "|---|---:|---:|---:|---:|"]
        for group, label in (("Pd","Pd"),("Vm","Vm"),("Va","Va"),("Pg","free Pg"),("Qg","Qg"),("other","branch-end P/Q flows")):
            family.append(f"| {label} ({n('joint.coordinates.'+group)}) | " + " | ".join(n(f"v4.{m}.family_es.{group}",".4f") for m in ("A","S","C","D")) + " |")
        brier = ["| Mode | Raw voltage Brier | Calibration-mapped |", "|---|---:|---:|"]
        for mode in ("C","P","P0","S","A","D"):
            brier.append(f"| {mode} | {n(f'v3.{mode}.voltage_brier_raw','.5f')} | {n(f'v3.{mode}.voltage_brier_mapped','.5f')} |")
        prevalence = ["| V3 hourly event | Calibration prevalence | Test prevalence |", "|---|---:|---:|"]
        for event in ("voltage_bus_1","voltage_bus_9","voltage_bus_11","voltage_bus_13","voltage_bus_15","voltage_bus_21","voltage_bus_22","thermal_line_1"):
            key = f"v3.prevalence.{event}.hourly_pct"
            prevalence.append(f"| {event} | {n(key,'.2f',2)}% | {n(key,'.2f',3)}% |")
        ood = ["| Realized reference-profile bin | Vm MAE (p.u.) | Pg MAE (MW, all generators) |", "|---|---:|---:|"]
        for key, label in (("lt_0_6","<0.6"),("0_6_to_0_8","0.6–0.8"),("ge_0_8","≥0.8")):
            ood.append(f"| {label} | {n(f'v3.ood.{key}.vm_mae','.6f')} | {n(f'v3.ood.{key}.pg_mw_mae','.3f')} |")
        hybrid = ["| Arm | Valid solves | Mean iterations | Mean request (ms) | Total work (s) |", "|---|---:|---:|---:|---:|"]
        for arm, label in (("cold","Cold"),("A_forecast_mean","A forecast mean")):
            hybrid.append(f"| {label} | {n('v3.hybrid.'+arm+'.valid',',')} | {n('v3.hybrid.'+arm+'.iterations','.3f')} | {n('v3.hybrid.'+arm+'.request_ms','.3f')} | {n('v3.hybrid.'+arm+'.total_work_seconds','.3f')} |")
        used.add("setup.case_buses")
        post = f"""# Case30 post-training with GENCO: forecasting gains, calibration and physical costs
<!-- headline: setup.case_buses -->

**Review draft — not published.** Participants: {B}<NAMES>{B}; affiliations:
{B}<AFFILIATIONS>{B}; contact: {B}<EMAIL>{B}. HF organization: {B}<HF_ORG>{B}.
V3 is the primary record; v4 is the disclosed follow-up.
Numerical evidence keys appear beside values in this Markdown's HTML comments;
the complete readable list is [headline_numbers.md](headline_numbers.md).

## Summary

- **Frozen-GENCO load-regime audit [v3]:** identical-input Vm error changes from
  {n('v3.ood.lt_0_6.vm_mae','.4f')} to {n('v3.ood.ge_0_8.vm_mae','.4f')} p.u.,
  and Pg error from {n('v3.ood.lt_0_6.pg_mw_mae','.1f')} to {n('v3.ood.ge_0_8.pg_mw_mae','.1f')} MW,
  moving from ref_t <{n('setup.ood_bin_edges',index=0)} to ≥{n('setup.ood_bin_edges',index=1)}.
- **Forecasting [v4, nominal in-distribution range]:** A lowers joint ES by
  {n('v4.A_vs_S.joint_es_reduction_pct','.0f')}% versus frozen-sampling S; Vm CRPS is
  {n('v4.A.vm_crps','.5f')} versus {n('v4.S.vm_crps','.5f')} p.u.
- **The cost [v4]:** A's coverage is {n('v4.A.vm_coverage90_pct','.1f')}% for nominal
  {n('setup.nominal_coverage_pct')}% intervals; mean |P imbalance| is
  {n('v4.A.p_imbalance','.2f')} versus {n('v4.D.p_imbalance','.3f')} MW/bus for the
  frozen path D on A's identical demand samples.

## Motivation

GENCO predicts an operating snapshot. An operator anticipating tomorrow needs
correlated demand and operating-state scenarios before the realized demand is
known. We post-train a temporal adapter around GENCO and ask whether its forecasts,
uncertainty and proposed numerical-solver starts are useful. We measure forecast
error, identical-input solver-approximation error and numerical refinement
separately, because success at one does not establish the others.

## GENCO, OpenGridFM and method

{figure('figure_1_workflow','Temporal adapter, frozen GENCO, calibration and numerical verification')}

We use {B}gridfm/genco-opfdata{B}, case30 seed {n('setup.seed')}, with the original
{B}normalizer_stats.pt{B}; neither backbone weights nor normalization statistics
are fitted to the temporal data. Graphkit supplies
{B}GNS_heterogeneous.forward(batch, return_embeddings=True){B}, HeteroData graphs
and physics reconstruction. Datakit supplies
{B}gridfm_datakit.network:Network{B},
{B}gridfm_datakit.network:makeYbus{B}, and
{B}gridfm_datakit.perturbations.load_perturbation:LoadScenariosFromAggProfile.min_max_scale{B}.
Our sequential wrapper adds bus noise and AR dependence; independent
PowerModels/Ipopt solves provide labels and numerical checks.

The encoder observes {n('setup.history_hours')} hours of Pd/Qd/Vm/Va/Pg/Qg,
availability masks and frozen GENCO embeddings. Future calendar is known; future
solved states are labels only. Bus/gen GRUs of width {n('setup.gru_width')} project
memory into width-{n('setup.conditioning_width')} input latents. Rank-{n('setup.covariance_rank')}
plus diagonal innovations affect demand and state readouts. Every one of the
{n('setup.samples')} scenarios passes its own sampled Pd/Qd into its own future graph
over {n('setup.forecast_hours')} hours; Qg and flows follow model reconstruction.
The adapter has {n('setup.trainable_parameters',',')} trainable parameters and
{n('setup.buffer_elements')} buffer elements. Frozen-weight hashes are checked.
Both adapters trained on Aurora XPU; selection chose epoch {n('v3.best_epoch')}
for v3 and {n('v4.best_epoch')} for v4. Resumable short-job chain logs are retained.

The loss is fair, train-scaled joint energy score plus normalized squared P/Q
balance residuals. There is no replay term. P is a per-active-bus ridge mean
with lags {n('setup.ridge_lags')} and calendar/past operating covariates; S uses
{n('setup.ridge_residual_blocks',',')} training-only rolling-origin residual blocks.
C samples training daily solved trajectories. D sends A's exact demand samples
through unconditioned frozen GENCO. P0 repeats P as a zero-spread check.

## Results

### (a) Frozen-GENCO load-regime audit [v3]

{figure('figure_2_ood','Frozen GENCO approximation error by realized reference-profile bin')}

{chr(10).join(ood)}

Neural and numerical predictions use identical predicted demand. The bins use
realized future ref_t only for retrospective auditing, never as a forecast input.
Only {n('v3.test_hours_high_profile')} of {n('setup.forecast_test_hours',',')} forecast
hours lie above the lower nominal OPFData multiplier {n('setup.opfdata_nominal_range',index=0)}.
The binning does not establish full feature-support overlap. The Pg audit above
includes fixed generators; the forecast tables below isolate free generators.

### (b) Forecasting and attribution [v4]

Nominal endpoints are {n('v4.nominal_load_range')}; the original OPFData range is
{n('setup.opfdata_nominal_range')}. Noise still puts
{n('v4.test_bus_load_below_support_pct','.2f')}% of test active-bus/hour multipliers below
the lower support endpoint, including history. “In-distribution” describes the
nominal range, not every graph feature.

{chr(10).join(table)}

{figure('figure_7_v3_v4','V3 and v4 forecast accuracy, interval coverage, free dispatch error and log-scale imbalance')}

V4 changes both nominal load range and physics-loss weight after v3 scores.
The side-by-side panel is confounded; absolute ES is not comparable across
datasets because targets and train-derived coordinate scales differ.

Family-wise ES uses separately normalized coordinate groups:

{chr(10).join(family)}

These scores are not additive. Joint ES contains {n('joint.coordinates.total')}
coordinates per hour; Qd and loading are scored marginally rather than duplicated
in the joint vector. A wins joint ES and voltage CRPS, but S has the better
demand-family ES. A and D share demand samples, so their difference implicates
the conditioned/stochastic state path; it does not isolate temporal conditioning
from learned state spread. Full P/P0 results and physical-unit scores remain
in the saved evaluation report.

### (c) Uncertainty and physics [v3 and v4]

A's v4 voltage coverage is {n('v4.A.vm_coverage90_pct','.1f')}% against nominal
{n('setup.nominal_coverage_pct')}%; C reaches {n('v4.C.vm_coverage90_pct','.1f')}%.
Lower joint ES does not establish calibrated intervals.

{figure('figure_3b_pit','V3 PIT diagnostics for every mode')}

The PIT figure reports v3; v4 marginal PIT counts and interval widths are
retained in its JSON report. Event maps do not change either ensemble's
continuous intervals. V3 hourly voltage reliability uses the predeclared buses
{n('setup.headline_buses')}; stress is the outer {n('setup.voltage_outer_band_pct')}%
of permitted voltage ranges or line loading above {n('setup.thermal_rating_pct')}%.
Maps fit on calibration only and pool supported components by event family.

{figure('figure_3_reliability','V3 raw and mapped reliability with prevalence in each panel')}

{chr(10).join(prevalence)}

Bus 21's calibration/test prevalence ratio is {n('v3.prevalence_ratio.voltage_bus_21','.2f')}×,
above the predeclared {n('setup.prevalence_ratio_flag')}× flag. Its mapped result
is a sensitivity analysis; the panel remains visible.

{chr(10).join(brier)}

Mapping worsens A's aggregate voltage Brier. Its raw advantage is concentrated
at buses 1/11/13; A predicts zero stress at buses 9/15/21/22 throughout v3 test.
V4's frozen stress family is degenerate on resolved labels in every partition:
the former buses and line 1 are always stressed, the latter buses never stressed.
That prevents a discrimination claim. Missing labels are not counted negative.

Physics weight changes from λ={n('v3.physics_weight')} to {n('v4.physics_weight')},
alongside nominal range {n('v3.nominal_load_range')} → {n('v4.nominal_load_range')}.
The change in A's imbalance, {n('v3.A.p_imbalance','.2f')} → {n('v4.A.p_imbalance','.2f')}
MW/bus, cannot be attributed to λ alone. Within v4, D's imbalance is
{n('v4.D.p_imbalance','.4f')} MW/bus. Raw adapter forecasts remain physically
inconsistent despite better forecasting scores. V3 remains the primary record.

### (d) Warm starts and embeddings: negative results [v3]

{figure('figure_5_hybrid','Cold versus forecast-mean warm start on the realized demand')}

{chr(10).join(hybrid)}

Both arms receive identical realized demand and the same solver budget.
Warm starts are A's forecast-mean Vm/Va/Pg/Qg, with no future solved-state input
or advance OPF solve. No online solver benefit was measured; including forecast
preparation, warm-start work is {n('v3.hybrid.work_ratio','.2f')}× cold-start work.
Totals sum request and preparation wall seconds; they are not parallel elapsed
time or energy. Shared setup/JIT is reported separately. The first preflight's
batch-size-dependent embedding mismatch is retained; the completed experiment
uses immutable cached embeddings and verified identical forecast replay.

{figure('figure_6_probe','Frozen-embedding linear probe: negative incremental predictive association')}

Adding embeddings to raw/current-state/calendar features worsens normalized MSE:
{n('v3.probe.h1.raw.normalized_mse','.5f')} → {n('v3.probe.h1.raw_plus_genco.normalized_mse','.5f')}
at t+{n('setup.probe_horizons',index=0)}, and
{n('v3.probe.h24.raw.normalized_mse','.5f')} → {n('v3.probe.h24.raw_plus_genco.normalized_mse','.5f')}
at t+{n('setup.probe_horizons',index=1)}. Feature dimensions differ; this is an
association test, not causal evidence about representations.

## What this gives an operator

A reproducible way to inspect joint future demand/operating scenarios, component
stress probabilities, reliability and physical residuals before considering a
numerical initialization. Load-regime diagnostics identify where a snapshot
model's approximation needs particular scrutiny.

It does not provide calibrated operational guarantees, feasible dispatch or a
demonstrated solver speedup. A prospective operator would still need independent
verification and validation on the intended network and demand regime.

## Limitations

Case30 only; fixed topology, costs and commitment; synthetic ERCOT→IEEE mapping;
{n('setup.training_seed_count')} training seed. The {n('setup.test_days')} correlated test
days come from {n('setup.test_source_episodes')} source episodes, not independent years.
V1→v3 split revisions were pre-score audits and are disclosed in the dataset card.
V4 is a two-change follow-up requested after v3 scores, with
{n('v4.labels.unresolved')} unresolved test labels and degenerate stress events.
These experiments do not establish rare-event, seasonal or multi-grid performance.

## Open science

Code: {B}<GITHUB_REPOSITORY_URL>{B}. HF model: {B}<HF_ORG>/genco-foresight-adapters{B}.
HF dataset: {B}<HF_ORG>/genco-foresight-temporal-case30{B}. Source release tag:
{B}v0.1.0{B} (prepared locally; publication awaits review).
Cards: [dataset](../data/DATASET_CARD.md), [model](../reproducibility/MODEL_CARD.md).
Figures use content-hashed filenames; [figure manifest](figures/manifest.json)
contains full SHA256s. The local v0.1.0-rc package includes source, both adapters,
datasets, per-origin predictions/hashes, failures and training chain logs.
No public availability is claimed until the owner publishes.

From the unpacked package, in an existing CPU Python environment:

{FENCE}bash
bash code/scripts/reproduce_small.sh "$PWD"
python code/scripts/headline_numbers.py --root evidence/runs --list
{FENCE}

The combined saved-forecast CPU replay took
{n('reproduction.combined_seconds','.1f')} seconds on Aurora, verifying payload hashes,
fixed origins from both datasets and the figure renderers. It includes a v4
origin with a masked solved state. It does not retrain, run fresh GENCO inference,
regenerate OPF labels or validate a clean/non-Aurora installation.
{B}headline_numbers.py{B} extracts every numerical claim from saved evidence.
Graphkit is pinned to {B}{pin['graphkit']}{B}; the base checkpoint revision is
{B}{pin['checkpoint']}{B}. Vendor Torch/Transformers were preserved; attribution
and upstream notices remain in the repository.

Supplementary hashed figures: [fan chart](figures/{figures['supplement_fan']['filename']}),
[daily reliability](figures/{figures['supplement_daily_reliability']['filename']}),
[coverage](figures/{figures['figure_3c_coverage']['filename']}),
[stress schematic](figures/{figures['figure_4_network']['filename']}).

## Next steps

Replicate across grids with prepared OPFData snapshot caches for
{n('setup.prepared_snapshot_grids')} buses; temporal generation and model validation
on those grids remain future work. Separate nominal-range and physics-weight
changes, then test demand-only conditioning with a frozen D-mode physics path
and learned demand spread. Explore GENCO as a regime router over per-topology
linear models.
"""
        (ROOT / "submission/post.md").write_text(post)
        write_json(out / "post_claims.json", {"post_sha256": sha256(ROOT / "submission/post.md"), "keys": sorted(used),
            "figure_manifest_sha256": sha256(assets / "manifest.json"), "review_status": "awaiting user review",
            "summary_interpretation": "Three requested quantitative findings; each bundles the comparisons specified by the user."})
    (ROOT / "submission/headline_numbers.md").write_text(markdown_list(H))
    write_json(out / "headline_numbers.json", record)
    print({"headline_keys": len(H), "post_written": a.post, "post_keys": len(used), "figures": len(figures)})


if __name__ == "__main__":
    main()
