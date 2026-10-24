# Stage C/D evidence and interpretation

V3 remains the primary experiment. The model, scorer, training data and test
forecasts are unchanged. Stage C adds predictions for the 72 calibration origins,
calibration-only event maps, and the user-confirmed cold versus A forecast-mean
solver comparison. Static figures are generated from saved reports. Live state:
[pipeline.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/stage_c/pipeline.json).

## Calibration findings

| Mode | Hourly voltage Brier, raw | Mapped | Vm 90% coverage |
|---|---:|---:|---:|
| C | 0.12341 | 0.12442 | 77.33% |
| P | 0.27918 | 0.14748 | 0.00% |
| P0 | 0.27918 | 0.14748 | 0.00% |
| S | 0.26835 | 0.14877 | 10.14% |
| A | 0.08692 | 0.09536 | 28.75% |
| D | 0.27628 | 0.14644 | 3.55% |

Voltage Brier averages the seven frozen buses equally; each has 1,728 resolved
test hours. This aggregate is descriptive and does not replace the per-bus
results. A's raw Brier improves on C at buses 1, 11 and 13, but A predicts zero
stress on buses 9, 15, 21 and 22 throughout the test set. Mapping changes those
zeros into nonzero probabilities without adding discrimination. The pooled map
worsens A's aggregate voltage Brier and improves its hourly line-1 Brier from
0.13979 to 0.08614; S's raw line-1 Brier is better (0.07232). Event probability
mapping does not repair continuous undercoverage or AC balance violations.

Calibration uses one L2-regularized sigmoid of clipped probability logits per
mode/family/resolution, pooling supported components. Hyperparameters, bins,
support rules and hashes were recorded before fitting. Maps use calibration
only. No test-based map choice, retraining, scorer change or event-family
selection occurred. All raw/mapped results are retained, including worse maps.
P/P0 have identical event probabilities; their finite-ensemble randomized PITs
differ because their ensemble sizes differ.

The hourly bus-21 prevalence shift exceeds 1.5×; its mapped panel is a sensitivity
analysis, not suppressed. Daily-any is secondary, formed within each joint
trajectory. Bus 1 fails the calibration 20–80% daily-prevalence rule; the other
six buses and line 1 pass. Reliability points require at least 20 observations
in predeclared bins. There are only four source episodes per held-out partition,
so 72 days do not provide strong rare-event or independent-hour guarantees.

Evidence: [full event report](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/stage_c/report.md),
[all metrics and per-episode results](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/stage_c/report.json),
[fitted maps](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/stage_c/maps.json),
[protocol](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/stage_c/protocol.json).

## Hybrid and figures

The paired solver experiment covers 72×24 realized hours. Warm starts are the
arithmetic mean of A's stored 64 Vm/Va/Pg/Qg states; no projection or future
solved state supplies the start. Cold means PowerModels' default initialization.
Both arms receive identical realized demand and the same fixed Ipopt budget.
Arm order alternates. Independent residual/limit checks determine success.
Both arms resolved all 1,728 hours. The warm start provided no measured benefit:

| Arm | Valid / attempted | Mean iterations | Mean online request | Forecast preparation, total | Total work |
|---|---:|---:|---:|---:|---:|
| Cold | 1,728/1,728 | 15.960 | 38.694 ms | 0 s | 66.864 s |
| A forecast mean | 1,728/1,728 | 16.074 | 38.701 ms | 129.233 s | 196.109 s |

Total work sums request wall seconds and measured history/forecast preparation;
it is not parallel elapsed time or energy consumption. The warm/cold ratio is
2.933. Shared worker setup and Julia JIT are recorded separately. No advance
OPF solves or third arm are used. This is a negative result on case30, not an
assertion about larger grids or an implementation with a rolling history cache.

The first timing preflight stopped before any paired solver outcomes: computing
history embeddings in a batch of 72 did not reproduce the immutable cache made
in batches of 192. The failed job/protocol/logs are retained under
`runs/stage_c/hybrid_attempt1_failed`. The retry times 72 historical graphs,
records their numerical deviation, and uses the original cached embeddings for
the actual forecast replay. The maximum latent differences were 0.00862217
(bus) and 0.00170684 (generator). Actual 64-sample forecast replay was
bit-identical to the saved A ensembles for every origin and reported state
field. No model, forecast, solver budget or success criterion was changed.

Retry job 8957813 finished successfully at 2026-10-23 19:04:26 UTC, before the
subsequent cancellation instruction could apply. It is not rerun. V4 training
has exclusive priority for this project's queued-job slot from the user's
priority amendment onward; no Stage C/D PBS work remains.

Completed outputs (six main figures plus four supplementary PNGs):
[hybrid results](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/stage_c/hybrid/report.md),
[figure package](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/stage_d/report.md),
[figure hashes and sources](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/stage_d/manifest.json).
Figure 2 is the OOD curve. Supplements retain the illustrative fan chart,
daily reliability, all-mode PIT and coverage/width table. All PNGs use Matplotlib.

Seven new monitor/configuration/missing-label tests pass. Numerical difficulty
cannot be fit as a two-class classifier: v3 has zero failed training solves at
the fixed budget. Prices and the third hybrid arm are deferred. F has not run.

## Separate v4 follow-up

V4 changes nominal profile endpoints and physics-loss weight together, after v3
scores. It is **in-distribution + physics-weighted**, not an ablation. Stage A
passes with 24 tests, 10,078 valid labels and two masked unresolved test hours.
Every headline event is degenerate in every partition: buses 1/11/13 and line 1
are always stressed, buses 9/15/21/22 never stressed on resolved labels. V4 cannot
demonstrate discriminative event calibration in this regime. Despite
nominal endpoints 0.80–1.00, unchanged noise puts 9.75% of test active-bus
multipliers below 0.8. Neither result triggered tuning. Train/selection labels
are complete; the architecture/optimizer, six-hour cumulative chain and patience
three remain frozen. V3 data, training and evaluation artifacts are preserved.

Training job 1 is **8957840**, running on `x4615c0s6b0n0`. Training was ready
9.47 seconds after job start; the first optimizer step completed at 11.90 seconds.
Its first checkpoint has file timestamp **2026-10-23 14:23:12 CT / 19:23:12 UTC**,
after 3,394 steps, and was successfully loaded on CPU with model, optimizer,
scheduler, RNG, progress and manifest state. The
[checkpoint record](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/adapter_v4/first_checkpoint.json)
contains its SHA256 and timestamp definition. Job **8957873** is the dependent
successor. The
[live chain](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/adapter_v4/chain.md)
accumulates completed-job wall time; an active job is not yet included in that total.
The 200-step XPU kill/resume test passed bit-for-bit, and three mocked scheduler
checks passed for successor reservation and cancellation after training stops.
V4 training and its subsequent automatic evaluation remain in progress.

[V4 audit](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/stage_a_v4/stage_a.json),
[prevalence](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/stage_a_v4/stage_b_prevalence.md),
[support qualification](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/stage_a_v4/support_diagnostic.json),
[pipeline](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v4/pipeline.json).

## Reproduction

Use the preserved Aurora environment; install nothing and do not replace Torch.
Run forecast/hybrid modules on allocated compute nodes. Analysis and rendering
reuse saved artifacts. Bulk outputs remain under the configured Lustre root.

```bash
export GENCO_CONFIG="$PWD/config.yaml"
python -m scripts.forecast_partition --partition risk_calibration
python -m scripts.calibrate_stage_c
python -m scripts.hybrid_stage_c
python -m scripts.figures_stage_d
python scripts/headline_numbers.py
```

For v4, select `GENCO_CONFIG="$PWD/configs/v4.yaml"`; the background driver handles
Stage A, the cache and exact-resume test, the resumable training chain, then
`python -m scripts.evaluate_variant`. Do not launch a duplicate controller;
both pipelines have lock files. Every forecast origin and hybrid hour is saved
independently. Pipeline/chain JSON retains errors and job IDs for recovery.
