# Stage 1 background confirmation — pending

Job **8962843** was submitted to `debug` on 2026-10-25 UTC: one node, one allocation, maximum 3,600 seconds, graceful family guard at 3,300 seconds. The last submission check showed queued. Acceptance is pending; `passed: null` in the [gate record](stage1_confirmation.json) is not a pass or a failure. The OFF fleet launch does not wait for this confirmation.

The [protocol](stage1_confirmation.protocol.json) was committed and frozen before submission, SHA-256 `72d15b6e27edd51af5e4a3b9245a2e6ced7601303158cdbde16bd4d57c658273`. Implementation commit: `c8b6f6bfee33ef3533a81df347bb3dadde096193`. It supersedes the earlier envelope protocol for this new identity; all earlier protocols and failed verdicts remain intact.

## Changes and protocol

The comparator now supports one-sided non-inferiority, accepts improvements without a lower bound, and requires complete epoch coverage and finite metrics. The runner monitors the fixed upper limits and writes the final case30-only gate without launching case118 or a new ON reference chain. Regression fixtures cover improvement, the inclusive bound, exceedance, nonfinite ES and missing epochs. Syntax and protocol consistency passed on the login node; the harness/controller tests and existing determinism-ON serialization/RNG mechanism test run on compute before the two OFF chains.

Two independent seed-42 OFF chains use the exact case30 v4 recipe from job 8962608, excluding the new V2 identity, protocol hash and output path. Both consume all 4,090 origins for three epochs (12,270 steps), K8/B1/fp32, with full 36-origin K64 selection each epoch. OFF B saves after step 4,090 and restarts in a fresh interpreter before selection. The gate checks exact restoration, configuration, counters, noise/order and final RNG state. Timing window: steps 101–600.

Each OFF selection ES must be at most the three historical v4 ON seeds' epoch maximum plus the fixed 0.005 slack:

| Epoch | Upper limit |
| --- | --- |
| 1 | 0.6221770658757951 |
| 2 | 0.4630226242542267 |
| 3 | 0.4442866889635722 |

The user-declared rationale and historical evidence hashes are in the protocol. The three historical ON seeds are the reference. There is no cache pre-copy before the allocation or spare-tile overlap. Ordinary preparation and checks precede OFF workers on flat tiles 0 and 1. No crash retry or second allocation is authorized by this protocol.

## Evidence and budget

Evidence root: [stage1_confirmation_20261025_v3](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/stage1_confirmation_20261025_v3). The runner writes the final verdict to [gate.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/stage1_confirmation_20261025_v3/gate.json), detailed comparisons to `compare_off_a__off_b.json`, boundary evidence under `off_b/restores/`, and worker costs under `jobs/`. Submission and frozen-input hashes are in the linked gate record. These completion artifacts are pending.

This confirmation reserves 3,600 node-seconds. Earlier Stage 1 attempts reserved 8,400 node-seconds and used 1,423 measured PBS node-seconds (conservative charge 4,600). Actual confirmation cost will come from PBS and the worker ledger after completion. User changes, the staged inventory, release references and main were verified unchanged at submission.

## Results and limitations

No production results are available yet. No main merge has occurred for this pending confirmation. The runner stops on a failed gate or the allocation cap. It does not launch a fleet. Three epochs do not test stopping-epoch agreement; that comparison is deferred to the first full fleet run. No new ON chain or case118 timing is included. The earlier ON timing remains contextual, not a new matched measurement.
