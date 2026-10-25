# V2 Stage 1 — production determinism OFF

Gate: **FAIL**. Stage 1 is not merged; determinism ON remains adopted on `main`.
Machine-readable verdict: [stage1.json](stage1.json). Stages 2–4 are not started.

Prerequisite: Stage 0 commit `7c19dd0`, merged into local `main` by
`a6ae620264ff4cdfda9e1b4f31eeca377cb6f938`.
Protocol: [stage1.protocol.json](stage1.protocol.json), `stage1-v1`, frozen SHA256
`71948ce147d2bfcfba0660fe1e315280cdc882de63d9ce63601a9880bb33b04b`.
Evidence root: `/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/stage1_20261025T0605Z`.

## Changes

`stage_b.deterministic_algorithms` is one explicit Boolean setting. Historical
configurations that omit it retain ON. New V2 production and Stage 1 overlays set
OFF. `make_manifest` records the effective setting and strict resume equality
remains enforced. `run(test_mode=True)` forces ON on a configuration copy, preserving
the existing `scripts/resume_test.py` mechanism test. Evaluation entry points are
unchanged and retain their existing ON behavior.

Checkpoint restoration now supports an audit containing hashes of saved/restored
adapter, optimizer, scheduler and all RNG state. A mismatch aborts restoration.
The record also identifies the exact configuration and restored progress. These
checks are exact even when subsequent production kernels are nondeterministic.

The reusable V2 verification entry dispatches Stage 1's bounded job chain. Three
independent workers use isolated flat tiles 0, 1 and 2 and assert that each sees one
device. OFF run B exits after the epoch's 4,090th update, before selection, and a
fresh interpreter resumes it. Atomic periodic saves and the existing validation
guard support additional clean allocation boundaries. The controller records all
worker costs, prevents overlapping job families and stops after failures or caps.

## Protocol

Case30 uses the v4 recipe, seed 42, B1, K8, 24 hours, fp32, unchanged masks/scales,
balance weight and AdamW settings. OFF A, OFF B and ON reference each run three
complete epochs (12,270 origins), with 36 K64 selection origins after every epoch.
Every training update is persisted. Both OFF-vs-OFF and OFF-vs-ON comparisons use
the bounds declared in the protocol: loss/ES/selection ES atol 1e-4, rtol 0.005;
balance atol 1e-5, rtol 0.01; gradient norm atol 1e-4, rtol 0.02. The symmetric
absolute-plus-relative rule avoids division near zero. These are engineering
equivalence limits, not a claim about v4 seed variability or confidence intervals.

The deterministic mechanism test uses 200 uninterrupted steps versus 100 steps,
SIGKILL after atomic save, and 100 resumed steps. A separate small controller
fixture uses four equal selection scores: patience 3 must stop at epoch 4, with
best epoch 1, including a restart before the first validation. Three production
epochs alone cannot exercise patience-3 stopping.

Case30 timing uses steps 101–600. Case118 timing uses 20 warmup steps and measured
steps 21–70, with fresh seed-42 ON/OFF runs and real cached histories. Case118
currently has no populated temporal training dataset, so the protocol includes
the existing solver smoke/200-hour pilot, followed by 10 training and 2 selection
episodes, independent label checks, train-derived scales and cache preparation.
No held-out labels/evaluation are included. This preparation and timing phase may
run only after case30 acceptance passes.

## Budget and execution

At most two one-hour debug allocations for case30 and two for case118; one node per
allocation, 14,400 seconds reserved in total, 3,300-second family guard, 120-second
training checkpoint cadence. Full worker startup, training, validation, checkpoint
and restart costs are recorded; queue time is separate. Missing boundaries are
charged conservatively and stop verification. Crash retries are not authorized.

The initial queued job `8962442` was withdrawn before execution when review found
a clean timing-continuation bug. Its original protocol/preparation/submission
records remain under `runs/v2/stage1_20261025T0600Z`, with `withdrawal.json` and
`withdrawn_queued_job.txt`. PBS recorded no execution host, start time or resource
usage. No numerical outcome existed; thresholds and compute caps were unchanged.
The fix and a continuation test were frozen under the new run identity above.
Replacement job: `8962445.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov`.

## Results and limitations

All **17 focused tests passed** on compute, including the actual loop's patience-3
controller fixture and boundary restoration. The unchanged deterministic mechanism
trial passed: 200 uninterrupted steps matched 100 + SIGKILL + 100 resumed steps
exactly for model, optimizer, scheduler, RNG, initial weights, order and counters.

Production acceptance failed. The OFF runs have identical manifests except run
identity, identical initial adapter hash
`aaedb0c038aec1a770a8e60d075323d12bc0e14c56f000e005e788ec4b34c647`, and matching
origin/noise/counter records throughout the compared prefixes. The ON reference
has the same initial adapter hash. All persisted metrics are finite. Nevertheless:

| Comparison / first failure | Step | Left | Right | Absolute difference | Allowed |
|---|---:|---:|---:|---:|---:|
| OFF A / OFF B, gradient norm | 1 | 366.337250 | 325.895355 | 40.441895 | 7.326845 |
| OFF A / OFF B, loss | 4 | 7.160635 | 7.274521 | 0.113886 | 0.036473 |
| OFF A / ON, gradient norm | 1 | 366.337250 | 290.002014 | 76.335236 | 7.326845 |
| OFF A / ON, loss | 2 | 9.991344 | 9.706055 | 0.285290 | 0.050057 |

The read-only prefix check detected the failure after 819 common OFF steps and
518 common OFF/ON steps. An all-observations numerical requirement is already
false at these prefixes; later epochs cannot repair it. Under the predeclared
stop condition, only this verification job was cancelled. No threshold was
changed, no retry was submitted and no new experiment followed the failure.
Final immutable logs contain 1,089 / 1,120 / 688 steps for OFF A / OFF B / ON;
all three remained in epoch 0. Final offline comparator verdicts also reject the
numerical differences and report the expected incomplete-coverage/missing-final-status
failures after cancellation. The evidence index hashes all retained artifacts.

The specific kernel or numerical mechanism causing the gradient disagreement has
**not** been isolated. This result establishes failure of the declared acceptance
bounds on this setup, not a general claim about every nondeterministic backend.

The predeclared case30 timing window, steps 101–600, was complete in all three runs:

| Mode | Median step s | p90 step s | Startup s | Worker seconds through last persisted step |
|---|---:|---:|---:|---:|
| off_a | 0.167152 | 0.167676 | 9.385 | 195.049 |
| off_b | 0.162514 | 0.162904 | 9.380 | 195.026 |
| on_reference | 0.265441 | 0.265998 | 9.378 | 195.036 |

The measured ON/OFF speedups are **1.588×** (OFF A) and **1.633×** (OFF B), with
three independent workers sharing one node. These are diagnostic production-step
timings; correctness failed, so OFF is not adopted. Full worker teardown costs
were not emitted after cancellation: the measured worker prefixes are lower bounds,
not complete end-to-end costs. PBS reports **423 seconds**, exit 143 from requested
termination. The protocol ledger conservatively charges the full **3,600-second**
allocation because no normal controller boundary was emitted. The withdrawn queued
job consumed zero compute; no second allocation or case118 job was submitted.

Atomic production checkpoints remain at steps 703 / 724 / 444, with matching
checkpoint/run manifests. Logs past those checkpoints remain intact as failure
evidence. The production restart at step 4,090, three complete epochs, selection
curves and final frozen-backbone checks were **not reached**. The separate controller
fixture and mechanism test do not substitute for those production requirements.
Case118 preparation/timing and its helper are unverified; no synthetic timing is
substituted. No XCCL, sample-parallel, origin-batching or bf16 work was started.

The failed Stage 1 code/configuration and this report are committed only on
`multigrid`; there is no Stage 1 merge into `main`. Historical manifests, tag
v0.1.0, vendor dependencies, unrelated edits and the pre-existing staged inventory
remain unchanged. No push or publication was performed.
