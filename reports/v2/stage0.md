# V2 Stage 0 — persisted comparison harness

Gate: **PASS**. The recorded exact-comparison and negative-control requirements passed.
Machine-readable verdict: [stage0.json](stage0.json).

Protocol: [stage0.protocol.json](stage0.protocol.json), schema 1, `stage0-v1`.
Frozen protocol SHA256: `7cc582ad251333737411f79e6be93072319dfe065766725c437d6ec3471bbc22`.
Evidence root: `/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/stage0_20261025T052938Z`.
PBS job: `8962405.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov`.

## Changes

`training/metrics.py` persists loss, fair ES (`energy`), balance, pre-clipping gradient
norm, optimizer steps, origins consumed, epoch position, origin identity, noise digest,
chained order/noise digest and measured timing at a fixed cadence. Training flushes
each record and fsyncs before saving an atomic checkpoint. Recovery validates the
committed cadence and archives uncheckpointed log tails before replay. Completed
selection ES remains in `epochs.json` and checkpoint progress; startup reconciles
that file from the restored checkpoint.

`make_manifest` records world size, origins per update, precision, global K,
sample partition and AdamW/clipping policy. V2 runs also record their unique identity,
resolved configuration, checkpoint/YAML/normalizer hashes and cache manifest hash.
Existing strict manifest equality still rejects incompatible resume. Deterministic
algorithms remain ON. Neither historical manifests nor `scripts/resume_test.py` were
modified.

`scripts/v2/compare_runs.py` requires complete, finite metric curves, expected
counters/configuration and matching origin/noise records. It applies predeclared
absolute-plus-relative bounds and excludes only the stated timing fields. The
verification entry point additionally compares complete final model, optimizer,
scheduler and RNG trees and non-wall-clock progress. A copied fixture changes only
step 150's loss by +1; genuine run artifacts are hashed before and after that check.

`scripts/v2/verify_stage.py` prepares and freezes configuration/source/protocol hashes,
checks the user queue before submission, reserves the allocation cap durably and
launches fresh workers sequentially. `verify.pbs` uses the existing vendor framework
modules and venv. No dependency installation is performed. Bulk/runtime artifacts
stay in the new Lustre V2 directory; existing v4 data, audit and scales are read only.
The cache is copied and hash-checked into V2 on compute before training.

## Protocol and budget

Two independent case30/v4 runs, seed 42, deterministic ON, B1, global K8, 24 hours,
fp32, original masks/scales, balance weight 1 and AdamW lr 1e-4. Each consumes the
first 300 origins in the full epoch permutation, with a record every update. All
numerical bounds are exactly zero. No denominator division is used. The run pair
must match exactly and the altered fixture must fail specifically at step 150 loss.

One debug node, one isolated tile per worker, sequential workers; one allocation
maximum, 3,600 seconds reserved, 3,000-second family guard, 60-second periodic
checkpoints. Allocation accounting includes module setup, cache preparation, tests,
worker startup, training, checkpointing and comparison; queue time is separate.
Worker failures, missing evidence, cap exhaustion or failed comparisons stop the gate.
No automatic retry or extra experiment is declared.

## Results

All **12 focused harness tests passed** on compute. Both runs completed exactly 300
steps and zero epochs. Maximum absolute differences in loss, ES, balance and gradient
norm are all **0.0**. Origin/counter/noise records and final adapter, optimizer,
scheduler, RNG and non-wall-clock progress matched exactly. Both backbone checks
passed. The copied negative fixture failed only at `step 150: loss` with delta 1.0;
genuine artifacts were unchanged. All evidence hashes were rechecked after the job.

| Run | Startup s | Worker s | Median step s | Sum of 300 steps s |
|---|---:|---:|---:|---:|
| on_a | 49.273 | 132.426 | 0.269306 | 82.915 |
| on_b | 17.398 | 102.006 | 0.271130 | 83.483 |

The outer subprocess wall costs were 133.608 and 103.330 seconds. Full allocation
accounting through verification was **263.948 seconds**, within the
3,600-second cap and 3,000-second family guard. PBS reports exit 0 and 269 seconds of wall time, including final teardown;
its accounting is linked and hashed in the gate JSON. These short-run medians include all 300 steps and are diagnostic;
no ON/OFF speedup is claimed. Cache/page residency differs between sequential workers.

The adopted setting remains deterministic ON, B1, world size 1, global K8 and fp32.
Code and this gate are committed on `multigrid`; the subsequent local `main` merge
records integration. Existing staged work is excluded from the task commit.

## Preservation and limitations

The initial inspection matched all 29 inventory hashes and all 65 source excerpts;
see [inspection.json](inspection.json). The pre-existing index and unrelated edits
were preserved. Historical run metadata and the v0.1.0 tag are checked before and
after verification. No push, publication, fleet training or replication is included.

**300 steps do not reach the 4,090-origin epoch boundary.** This gate does not
exercise selection, patience-3 stopping, an epoch-boundary restart, nondeterministic
production acceptance, distributed training, origin batching or bf16. Unit fixtures
exercise selection-record comparison and checkpoint/log recovery, not those model
behaviors. Wall time is measured and is never required to match exactly.
