# Case30 post-training with GENCO: forecasting gains, calibration and physical costs
<!-- headline: setup.case_buses -->

**Review draft — not published.** Participants: `<NAMES>`; affiliations:
`<AFFILIATIONS>`; contact: `<EMAIL>`. HF organization: `<HF_ORG>`.
V3 is the primary record; v4 is the disclosed follow-up.
Numerical evidence keys appear beside values in this Markdown's HTML comments;
the complete readable list is [headline_numbers.md](headline_numbers.md).

## Summary

- **Frozen-GENCO load-regime audit [v3]:** identical-input Vm error changes from
  0.0060<!-- headline: v3.ood.lt_0_6.vm_mae --> to 0.0023<!-- headline: v3.ood.ge_0_8.vm_mae --> p.u.,
  and Pg error from 13.8<!-- headline: v3.ood.lt_0_6.pg_mw_mae --> to 1.0<!-- headline: v3.ood.ge_0_8.pg_mw_mae --> MW,
  moving from ref_t <0.6<!-- headline: setup.ood_bin_edges[0] --> to ≥0.8<!-- headline: setup.ood_bin_edges[1] -->.
- **Forecasting [v4, nominal in-distribution range]:** A lowers joint ES by
  39<!-- headline: v4.A_vs_S.joint_es_reduction_pct -->% versus frozen-sampling S; Vm CRPS is
  0.00096<!-- headline: v4.A.vm_crps --> versus 0.00147<!-- headline: v4.S.vm_crps --> p.u.
- **The cost [v4]:** A's coverage is 31.7<!-- headline: v4.A.vm_coverage90_pct -->% for nominal
  90<!-- headline: setup.nominal_coverage_pct -->% intervals; mean |P imbalance| is
  1.11<!-- headline: v4.A.p_imbalance --> versus 0.037<!-- headline: v4.D.p_imbalance --> MW/bus for the
  frozen path D on A's identical demand samples.

## Motivation

GENCO predicts an operating snapshot. An operator anticipating tomorrow needs
correlated demand and operating-state scenarios before the realized demand is
known. We post-train a temporal adapter around GENCO and ask whether its forecasts,
uncertainty and proposed numerical-solver starts are useful. We measure forecast
error, identical-input solver-approximation error and numerical refinement
separately, because success at one does not establish the others.

## GENCO, OpenGridFM and method

![Temporal adapter, frozen GENCO, calibration and numerical verification](figures/figure_1_workflow.ff2ed41c83df.png)

We use `gridfm/genco-opfdata`, case30 seed 42<!-- headline: setup.seed -->, with the original
`normalizer_stats.pt`; neither backbone weights nor normalization statistics
are fitted to the temporal data. Graphkit supplies
`GNS_heterogeneous.forward(batch, return_embeddings=True)`, HeteroData graphs
and physics reconstruction. Datakit supplies
`gridfm_datakit.network:Network`,
`gridfm_datakit.network:makeYbus`, and
`gridfm_datakit.perturbations.load_perturbation:LoadScenariosFromAggProfile.min_max_scale`.
Our sequential wrapper adds bus noise and AR dependence; independent
PowerModels/Ipopt solves provide labels and numerical checks.

The encoder observes 72<!-- headline: setup.history_hours --> hours of Pd/Qd/Vm/Va/Pg/Qg,
availability masks and frozen GENCO embeddings. Future calendar is known; future
solved states are labels only. Bus/gen GRUs of width 32<!-- headline: setup.gru_width --> project
memory into width-48<!-- headline: setup.conditioning_width --> input latents. Rank-8<!-- headline: setup.covariance_rank -->
plus diagonal innovations affect demand and state readouts. Every one of the
64<!-- headline: setup.samples --> scenarios passes its own sampled Pd/Qd into its own future graph
over 24<!-- headline: setup.forecast_hours --> hours; Qg and flows follow model reconstruction.
The adapter has 91,461<!-- headline: setup.trainable_parameters --> trainable parameters and
884<!-- headline: setup.buffer_elements --> buffer elements. Frozen-weight hashes are checked.
Both adapters trained on Aurora XPU; selection chose epoch 18<!-- headline: v3.best_epoch -->
for v3 and 14<!-- headline: v4.best_epoch --> for v4. Resumable short-job chain logs are retained.

The loss is fair, train-scaled joint energy score plus normalized squared P/Q
balance residuals. There is no replay term. P is a per-active-bus ridge mean
with lags [1, 2, 24, 48]<!-- headline: setup.ridge_lags --> and calendar/past operating covariates; S uses
3,130<!-- headline: setup.ridge_residual_blocks --> training-only rolling-origin residual blocks.
C samples training daily solved trajectories. D sends A's exact demand samples
through unconditioned frozen GENCO. P0 repeats P as a zero-spread check.

## Results

### (a) Frozen-GENCO load-regime audit [v3]

![Frozen GENCO approximation error by realized reference-profile bin](figures/figure_2_ood.54ab9c6c23fe.png)

| Realized reference-profile bin | Vm MAE (p.u.) | Pg MAE (MW, all generators) |
|---|---:|---:|
| <0.6 | 0.006015<!-- headline: v3.ood.lt_0_6.vm_mae --> | 13.760<!-- headline: v3.ood.lt_0_6.pg_mw_mae --> |
| 0.6–0.8 | 0.004747<!-- headline: v3.ood.0_6_to_0_8.vm_mae --> | 6.161<!-- headline: v3.ood.0_6_to_0_8.pg_mw_mae --> |
| ≥0.8 | 0.002332<!-- headline: v3.ood.ge_0_8.vm_mae --> | 1.014<!-- headline: v3.ood.ge_0_8.pg_mw_mae --> |

Neural and numerical predictions use identical predicted demand. The bins use
realized future ref_t only for retrospective auditing, never as a forecast input.
Only 289<!-- headline: v3.test_hours_high_profile --> of 1,728<!-- headline: setup.forecast_test_hours --> forecast
hours lie above the lower nominal OPFData multiplier 0.8<!-- headline: setup.opfdata_nominal_range[0] -->.
The binning does not establish full feature-support overlap. The Pg audit above
includes fixed generators; the forecast tables below isolate free generators.

### (b) Forecasting and attribution [v4]

Nominal endpoints are [0.8, 1.0]<!-- headline: v4.nominal_load_range -->; the original OPFData range is
[0.8, 1.2]<!-- headline: setup.opfdata_nominal_range -->. Noise still puts
9.75<!-- headline: v4.test_bus_load_below_support_pct -->% of test active-bus/hour multipliers below
the lower support endpoint, including history. “In-distribution” describes the
nominal range, not every graph feature.

| Mode | Joint ES ↓ | Vm CRPS (p.u.) ↓ | Vm coverage90 | Free-Pg MAE (MW) ↓ |
|---|---:|---:|---:|---:|
| A | 0.30334<!-- headline: v4.A.joint_es --> | 0.000958<!-- headline: v4.A.vm_crps --> | 31.71<!-- headline: v4.A.vm_coverage90_pct -->% | 3.155<!-- headline: v4.A.pg_free_mae --> |
| S | 0.49691<!-- headline: v4.S.joint_es --> | 0.001466<!-- headline: v4.S.vm_crps --> | 28.60<!-- headline: v4.S.vm_coverage90_pct -->% | 3.447<!-- headline: v4.S.pg_free_mae --> |
| C | 0.42472<!-- headline: v4.C.joint_es --> | 0.001298<!-- headline: v4.C.vm_crps --> | 80.08<!-- headline: v4.C.vm_coverage90_pct -->% | 6.048<!-- headline: v4.C.pg_free_mae --> |
| D | 0.63410<!-- headline: v4.D.joint_es --> | 0.001821<!-- headline: v4.D.vm_crps --> | 7.03<!-- headline: v4.D.vm_coverage90_pct -->% | 3.770<!-- headline: v4.D.pg_free_mae --> |

![V3 and v4 forecast accuracy, interval coverage, free dispatch error and log-scale imbalance](figures/figure_7_v3_v4.689dad7076b6.png)

V4 changes both nominal load range and physics-loss weight after v3 scores.
The side-by-side panel is confounded; absolute ES is not comparable across
datasets because targets and train-derived coordinate scales differ.

Family-wise ES uses separately normalized coordinate groups:

| Family (coordinates/hour) | A | S | C | D |
|---|---:|---:|---:|---:|
| Pd (21<!-- headline: joint.coordinates.Pd -->) | 0.4961<!-- headline: v4.A.family_es.Pd --> | 0.3340<!-- headline: v4.S.family_es.Pd --> | 0.4316<!-- headline: v4.C.family_es.Pd --> | 0.4961<!-- headline: v4.D.family_es.Pd --> |
| Vm (30<!-- headline: joint.coordinates.Vm -->) | 0.1718<!-- headline: v4.A.family_es.Vm --> | 0.2317<!-- headline: v4.S.family_es.Vm --> | 0.2608<!-- headline: v4.C.family_es.Vm --> | 0.2948<!-- headline: v4.D.family_es.Vm --> |
| Va (29<!-- headline: joint.coordinates.Va -->) | 0.3183<!-- headline: v4.A.family_es.Va --> | 0.3502<!-- headline: v4.S.family_es.Va --> | 0.5705<!-- headline: v4.C.family_es.Va --> | 0.4595<!-- headline: v4.D.family_es.Va --> |
| free Pg (2<!-- headline: joint.coordinates.Pg -->) | 0.4119<!-- headline: v4.A.family_es.Pg --> | 0.4184<!-- headline: v4.S.family_es.Pg --> | 0.6636<!-- headline: v4.C.family_es.Pg --> | 0.5791<!-- headline: v4.D.family_es.Pg --> |
| Qg (6<!-- headline: joint.coordinates.Qg -->) | 0.4443<!-- headline: v4.A.family_es.Qg --> | 1.7816<!-- headline: v4.S.family_es.Qg --> | 0.6057<!-- headline: v4.C.family_es.Qg --> | 1.9811<!-- headline: v4.D.family_es.Qg --> |
| branch-end P/Q flows (164<!-- headline: joint.coordinates.other -->) | 0.3043<!-- headline: v4.A.family_es.other --> | 0.4768<!-- headline: v4.S.family_es.other --> | 0.4026<!-- headline: v4.C.family_es.other --> | 0.6184<!-- headline: v4.D.family_es.other --> |

These scores are not additive. Joint ES contains 252<!-- headline: joint.coordinates.total -->
coordinates per hour; Qd and loading are scored marginally rather than duplicated
in the joint vector. A wins joint ES and voltage CRPS, but S has the better
demand-family ES. A and D share demand samples, so their difference implicates
the conditioned/stochastic state path; it does not isolate temporal conditioning
from learned state spread. Full P/P0 results and physical-unit scores remain
in the saved evaluation report.

### (c) Uncertainty and physics [v3 and v4]

A's v4 voltage coverage is 31.7<!-- headline: v4.A.vm_coverage90_pct -->% against nominal
90<!-- headline: setup.nominal_coverage_pct -->%; C reaches 80.1<!-- headline: v4.C.vm_coverage90_pct -->%.
Lower joint ES does not establish calibrated intervals.

![V3 PIT diagnostics for every mode](figures/figure_3b_pit.af45260ec185.png)

The PIT figure reports v3; v4 marginal PIT counts and interval widths are
retained in its JSON report. Event maps do not change either ensemble's
continuous intervals. V3 hourly voltage reliability uses the predeclared buses
[1, 9, 11, 13, 15, 21, 22]<!-- headline: setup.headline_buses -->; stress is the outer 5<!-- headline: setup.voltage_outer_band_pct -->%
of permitted voltage ranges or line loading above 95<!-- headline: setup.thermal_rating_pct -->%.
Maps fit on calibration only and pool supported components by event family.

![V3 raw and mapped reliability with prevalence in each panel](figures/figure_3_reliability.e8f7c1eb5663.png)

| V3 hourly event | Calibration prevalence | Test prevalence |
|---|---:|---:|
| voltage_bus_1 | 65.28<!-- headline: v3.prevalence.voltage_bus_1.hourly_pct[2] -->% | 64.06<!-- headline: v3.prevalence.voltage_bus_1.hourly_pct[3] -->% |
| voltage_bus_9 | 8.97<!-- headline: v3.prevalence.voltage_bus_9.hourly_pct[2] -->% | 8.22<!-- headline: v3.prevalence.voltage_bus_9.hourly_pct[3] -->% |
| voltage_bus_11 | 32.87<!-- headline: v3.prevalence.voltage_bus_11.hourly_pct[2] -->% | 37.44<!-- headline: v3.prevalence.voltage_bus_11.hourly_pct[3] -->% |
| voltage_bus_13 | 27.78<!-- headline: v3.prevalence.voltage_bus_13.hourly_pct[2] -->% | 33.28<!-- headline: v3.prevalence.voltage_bus_13.hourly_pct[3] -->% |
| voltage_bus_15 | 5.84<!-- headline: v3.prevalence.voltage_bus_15.hourly_pct[2] -->% | 8.22<!-- headline: v3.prevalence.voltage_bus_15.hourly_pct[3] -->% |
| voltage_bus_21 | 3.70<!-- headline: v3.prevalence.voltage_bus_21.hourly_pct[2] -->% | 6.42<!-- headline: v3.prevalence.voltage_bus_21.hourly_pct[3] -->% |
| voltage_bus_22 | 6.37<!-- headline: v3.prevalence.voltage_bus_22.hourly_pct[2] -->% | 8.45<!-- headline: v3.prevalence.voltage_bus_22.hourly_pct[3] -->% |
| thermal_line_1 | 27.03<!-- headline: v3.prevalence.thermal_line_1.hourly_pct[2] -->% | 32.81<!-- headline: v3.prevalence.thermal_line_1.hourly_pct[3] -->% |

Bus 21's calibration/test prevalence ratio is 1.73<!-- headline: v3.prevalence_ratio.voltage_bus_21 -->×,
above the predeclared 1.5<!-- headline: setup.prevalence_ratio_flag -->× flag. Its mapped result
is a sensitivity analysis; the panel remains visible.

| Mode | Raw voltage Brier | Calibration-mapped |
|---|---:|---:|
| C | 0.12341<!-- headline: v3.C.voltage_brier_raw --> | 0.12442<!-- headline: v3.C.voltage_brier_mapped --> |
| P | 0.27918<!-- headline: v3.P.voltage_brier_raw --> | 0.14748<!-- headline: v3.P.voltage_brier_mapped --> |
| P0 | 0.27918<!-- headline: v3.P0.voltage_brier_raw --> | 0.14748<!-- headline: v3.P0.voltage_brier_mapped --> |
| S | 0.26835<!-- headline: v3.S.voltage_brier_raw --> | 0.14877<!-- headline: v3.S.voltage_brier_mapped --> |
| A | 0.08692<!-- headline: v3.A.voltage_brier_raw --> | 0.09536<!-- headline: v3.A.voltage_brier_mapped --> |
| D | 0.27628<!-- headline: v3.D.voltage_brier_raw --> | 0.14644<!-- headline: v3.D.voltage_brier_mapped --> |

Mapping worsens A's aggregate voltage Brier. Its raw advantage is concentrated
at buses 1/11/13; A predicts zero stress at buses 9/15/21/22 throughout v3 test.
V4's frozen stress family is degenerate on resolved labels in every partition:
the former buses and line 1 are always stressed, the latter buses never stressed.
That prevents a discrimination claim. Missing labels are not counted negative.

Physics weight changes from λ=0.01<!-- headline: v3.physics_weight --> to 1.0<!-- headline: v4.physics_weight -->,
alongside nominal range [0.3, 1.0]<!-- headline: v3.nominal_load_range --> → [0.8, 1.0]<!-- headline: v4.nominal_load_range -->.
The change in A's imbalance, 5.46<!-- headline: v3.A.p_imbalance --> → 1.11<!-- headline: v4.A.p_imbalance -->
MW/bus, cannot be attributed to λ alone. Within v4, D's imbalance is
0.0374<!-- headline: v4.D.p_imbalance --> MW/bus. Raw adapter forecasts remain physically
inconsistent despite better forecasting scores. V3 remains the primary record.

### (d) Warm starts and embeddings: negative results [v3]

![Cold versus forecast-mean warm start on the realized demand](figures/figure_5_hybrid.69f169a0f75f.png)

| Arm | Valid solves | Mean iterations | Mean request (ms) | Total work (s) |
|---|---:|---:|---:|---:|
| Cold | 1,728<!-- headline: v3.hybrid.cold.valid --> | 15.960<!-- headline: v3.hybrid.cold.iterations --> | 38.694<!-- headline: v3.hybrid.cold.request_ms --> | 66.864<!-- headline: v3.hybrid.cold.total_work_seconds --> |
| A forecast mean | 1,728<!-- headline: v3.hybrid.A_forecast_mean.valid --> | 16.074<!-- headline: v3.hybrid.A_forecast_mean.iterations --> | 38.701<!-- headline: v3.hybrid.A_forecast_mean.request_ms --> | 196.109<!-- headline: v3.hybrid.A_forecast_mean.total_work_seconds --> |

Both arms receive identical realized demand and the same solver budget.
Warm starts are A's forecast-mean Vm/Va/Pg/Qg, with no future solved-state input
or advance OPF solve. No online solver benefit was measured; including forecast
preparation, warm-start work is 2.93<!-- headline: v3.hybrid.work_ratio -->× cold-start work.
Totals sum request and preparation wall seconds; they are not parallel elapsed
time or energy. Shared setup/JIT is reported separately. The first preflight's
batch-size-dependent embedding mismatch is retained; the completed experiment
uses immutable cached embeddings and verified identical forecast replay.

![Frozen-embedding linear probe: negative incremental predictive association](figures/figure_6_probe.aeed48ead688.png)

Adding embeddings to raw/current-state/calendar features worsens normalized MSE:
0.03426<!-- headline: v3.probe.h1.raw.normalized_mse --> → 0.03604<!-- headline: v3.probe.h1.raw_plus_genco.normalized_mse -->
at t+1<!-- headline: setup.probe_horizons[0] -->, and
0.18436<!-- headline: v3.probe.h24.raw.normalized_mse --> → 0.19562<!-- headline: v3.probe.h24.raw_plus_genco.normalized_mse -->
at t+24<!-- headline: setup.probe_horizons[1] -->. Feature dimensions differ; this is an
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
1<!-- headline: setup.training_seed_count --> training seed. The 72<!-- headline: setup.test_days --> correlated test
days come from 4<!-- headline: setup.test_source_episodes --> source episodes, not independent years.
V1→v3 split revisions were pre-score audits and are disclosed in the dataset card.
V4 is a two-change follow-up requested after v3 scores, with
2<!-- headline: v4.labels.unresolved --> unresolved test labels and degenerate stress events.
These experiments do not establish rare-event, seasonal or multi-grid performance.

## Open science

Code: `<GITHUB_REPOSITORY_URL>`. HF model: `<HF_ORG>/genco-foresight-adapters`.
HF dataset: `<HF_ORG>/genco-foresight-temporal-case30`. Source release tag:
`v0.1.0` (prepared locally; publication awaits review).
Cards: [dataset](../data/DATASET_CARD.md), [model](../reproducibility/MODEL_CARD.md).
Figures use content-hashed filenames; [figure manifest](figures/manifest.json)
contains full SHA256s. The local v0.1.0-rc package includes source, both adapters,
datasets, per-origin predictions/hashes, failures and training chain logs.
No public availability is claimed until the owner publishes.

From the unpacked package, in an existing CPU Python environment:

```bash
bash code/scripts/reproduce_small.sh "$PWD"
python code/scripts/headline_numbers.py --root evidence/runs --list
```

The combined saved-forecast CPU replay took
143.8<!-- headline: reproduction.combined_seconds --> seconds on Aurora, verifying payload hashes,
fixed origins from both datasets and the figure renderers. It includes a v4
origin with a masked solved state. It does not retrain, run fresh GENCO inference,
regenerate OPF labels or validate a clean/non-Aurora installation.
`headline_numbers.py` extracts every numerical claim from saved evidence.
Graphkit is pinned to `f104900245cba26d1380037eea562cf633a97484`; the base checkpoint revision is
`74e19e1e210a175aa17023f4db161f0ae9f753cb`. Vendor Torch/Transformers were preserved; attribution
and upstream notices remain in the repository.

Supplementary hashed figures: [fan chart](figures/supplement_fan.df58b10b5383.png),
[daily reliability](figures/supplement_daily_reliability.44573c648768.png),
[coverage](figures/figure_3c_coverage.9f4099fd6ce5.png),
[stress schematic](figures/figure_4_network.e01be0f8bb78.png).

## Next steps

Replicate across grids with prepared OPFData snapshot caches for
[14, 57, 118, 500]<!-- headline: setup.prepared_snapshot_grids --> buses; temporal generation and model validation
on those grids remain future work. Separate nominal-range and physics-weight
changes, then test demand-only conditioning with a frozen D-mode physics path
and learned demand spread. Explore GENCO as a regime router over per-topology
linear models.
