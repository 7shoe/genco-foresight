# Stage 1 revised epoch acceptance

**FAIL under the predeclared, literal three-seed min/max envelope.** The second OFF
chain's first-epoch ES was **0.5028239936**, below the historical minimum
**0.5118635098**. Lower ES is better, so this is an envelope-policy failure caused
by improvement beyond the recorded range. It is not evidence of inferior OFF
optimization, nonfinite training or an unreliable gradient. The threshold was not
changed after observing this result. The family stopped, and Stage 2 remains gated.

The [single-batch gradient diagnosis](stage1_diagnosis.md) passed the user's >=0.99
cosine rule. This new acceptance run followed that decision, using a new identity
and protocol. Both the original Stage 1 failure and diagnostic evidence remain
unchanged. No Stage 1 production adoption or local merge into main occurred.

## Changes and fixed protocol

Source commit `327e225` adds an explicit epoch-envelope comparison mode while
preserving the original step-equivalence mode. Step losses, ES, balance, gradient
norms and timings are still persisted at every step; nonfinite values fail. Origin
order, noise records, counters, configuration and restoration checks remain exact.
The runner checks persisted epoch scores and stops its worker family on a failed
envelope. It never changes released manifests or resumes the failed old run.

[Protocol](stage1_epoch.protocol.json), SHA256
`86b8591c03c0642bb90201bd6d7eb9ea2db2b6bb56c17f93603adb9146e1ef08`, was frozen
before submission. [Gate JSON](stage1_epoch.json) links the evidence index and
primary records. The interpretation of "within the three-seed spread" was stated
before running: each chain's ES must be between the historical minimum and maximum
at the same epoch, inclusive, with no added slack. This is stricter than a one-sided
non-inferiority criterion and can reject an improvement, as happened here.

Planned acceptance: case30 v4, seed 42, global K8, B1, fp32, two independent OFF
chains and one ON reference, each 12,270 steps / three full epochs; fixed K64
selection over 36 origins every epoch. `off_b` stops after training origin 4,090
and restarts in a new interpreter before validation. The original determinism-ON
200-step interrupted/resumed mechanism test and patience-3 controller test remain.
Matching production timing uses exactly steps 101–600 in each chain.

All bounds came from the already-recorded v4 seed42/1/2 curves, with hashes in the
protocol. Their data, scales, checkpoint and training recipe match case30 here;
their historical source versions differ from the current implementation. Bounds
for epochs 1–3 respectively were [0.5118635098, 0.6171770659],
[0.4213344115, 0.4580226243], [0.3815448309, 0.4392866890]. No new-run result informed
those numbers. Actual early stopping would use the user-specified +/-1 epoch rule;
three epochs do not test patience-3 stopping, and budget stops remain censored.

## Results and stop

| Run | Persisted training steps | Completed selection epochs | Epoch-1 selection ES |
|---|---:|---:|---:|
| OFF A | 4,204 | 1 | 0.519169973830382 — inside |
| OFF B | 4,090 | 1 | 0.502823993563652 — below minimum |
| ON reference | 2,586 | 0 | Not reached |

OFF B is 0.0090395163 below the minimum (about 1.77% better than that historical
best). The gate closed as soon as that persisted result was observed. OFF A's 114
steps in epoch 2 occurred while OFF B was restarting and validating; no work
continued intentionally after the failed gate. The controller terminated only
this family's processes. The planned boundary worker exited normally before its
restart; remaining workers were killed by the declared family-failure handler.
Periodic and epoch checkpoints remain atomic, and raw logs are retained.

All 10,880 persisted loss/ES/balance/gradient-norm records are finite. The two OFF
chains have identical origin/noise/counter records through all 4,090 common steps;
OFF A and ON match through their 2,586 common steps. These are prefix checks, not a
claim of completed three-epoch coverage or final RNG equality across all chains.

The controlled OFF B restore passed exactly for model, optimizer, scheduler and
RNG, with saved counters `steps=4090`, `epoch=0`, `step_in_epoch=4090`. The restored
worker completed the pending selection phase and saved epoch 1. The separate
200-step ON mechanism test also passed bit-for-bit after SIGKILL at step 100.
All 18 harness/controller tests passed on the compute node, including tests that
the new envelope mode still rejects wrong noise, nonfinite metrics and an ES just
outside its fixed bound. Production early stopping did not occur.

## Timing and budget

| Run | Median step, seconds | p90 step, seconds | ON / median |
|---|---:|---:|---:|
| OFF A | 0.160816 | 0.162473 | 1.6663x |
| OFF B | 0.160831 | 0.161425 | 1.6662x |
| ON reference | 0.267970 | 0.268666 | 1.0000x |

Each row uses all 500 predeclared production steps, with real cached histories and
training updates. The independent chains occupied isolated flat tiles 0/1/2 of one
node. These are step timings, not a completed-run speedup or a distributed model.

PBS job `8962608.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov`, node
`x4418c6s3b0n0`, used **925 seconds** (launcher: 924.505) of one reserved 3,600-second
debug allocation. Exit status 1 is the recorded envelope failure. Production
worker wall times, including setup, restart, selection and termination, were
707.153 seconds for OFF A, 706.316 seconds across OFF B's two processes, and 707.230
seconds for ON. Their sum is 2,120.699 worker-seconds because they ran concurrently.
Persisted training-step times sum to 2,036.758 seconds; the remaining 83.941 worker-
seconds include startup, validation, checkpointing and orchestration. Allocation
setup, cache preparation, tests and the mechanism check precede these workers.

The fresh acceptance protocol allowed at most two case30 and two conditional
case118 one-hour allocations (14,400 node-seconds reserved cap). Only the first
case30 allocation was submitted; the failure closed that chain. No continuation,
case118 timing, fleet run or Stage 2 compute was launched.

Across Stage 1 attempts so far: original failed acceptance reserved 3,600 / used
423 seconds; diagnosis reserved 1,200 / used 75 seconds; revised acceptance
reserved 3,600 / used 925 seconds. Total reservations are **8,400 node-seconds**,
actual PBS usage **1,423 node-seconds**. The original abrupt stop was conservatively
charged its full 3,600-second reservation, so the corresponding conservative
completed-job charge is **4,600 seconds**. These earlier costs were not reset.

## Evidence and limitations

Bulk evidence:
`/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/stage1_epoch_20261025_v2`.
`artifact_sha256.json` indexes 62 files, including frozen preparation, raw curves,
checkpoints, epoch failure, restore audit, tests, worker ledger and PBS accounting.
`preservation_audit.json` confirms frozen sources at completion, protected historical
artifacts, unrelated working-tree files, the existing staged inventory and release
tag. No jobs remain active.

Only one complete OFF epoch is available; ON did not reach its first validation.
Consequently this run cannot establish three-epoch acceptance, a full-run cost,
stopping-epoch equivalence or case118 performance. The diagnostic's usable-gradient
finding remains valid for its single prescribed origin. It does not waive a failed
production gate. Main stays at the Stage 0 merge; the tested OFF implementation and
revised gate remain on multigrid. A different acceptance interpretation would need
a newly declared protocol, not a rewritten verdict for these runs.

After verification, the diagnostic launcher was adjusted to disable pytest's
repository cache and place test temporaries under the run's Lustre `tmp` directory.
The three task-created cache entries were removed while preserving prior entries.
This argument-only housekeeping change does not change the recorded computations;
the executed diagnostic source is commit `b49a177`.
