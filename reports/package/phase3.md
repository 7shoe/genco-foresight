# Phase 3: pff CLI — incomplete, compute budget closed

**Phase 3 did not pass.** PBS job **8966340 exited 1** after saving the
step-4,090 checkpoint, when the CLI attempted to write chain status. There are
zero completed epochs and no selection-ES verdict. The user's hard compute stop
closes all remaining verification. No merge to `main`, `v0.3.0-dev` tag, or
Phase 4 work was performed. The conditional waiver required this job to pass;
that condition was not met.

Continuation job **8966377** had been submitted and started before the hard-stop
message arrived. It was cancelled immediately on receipt of that instruction
(PBS exit 143, 18 seconds). It stopped before the resume command: no additional
optimizer updates, selection epochs or restore audits were produced. There are
no active task jobs or automatic follow-up submitters. `STOP.json` records and
the shared budget ledger record the prohibition on further task submissions.

## Implementation and source state

The `cli` branch starts at `v0.2.1-dev`. `pff` and `power-fm-foresight` expose
five argparse verbs: prepare, train, evaluate, report and verify. Configuration
precedence, campaign expansion, immutable command manifests, strict resume
checks, evaluation sharding, compatible mixtures, and persisted-result reports
are implemented. PBS wrappers remain under `experiments/aurora/`, outside the
wheel. Reproduction examples are in [experiments/README.md](../../experiments/README.md).

The numerical training loop, optimizer, masks, physics and loss formulas were
not changed. Existing preparation/parity/event helpers were made importable.
Forecast functions accept checkpoint/sample-count arguments and the scorer
accepts an evaluation seed, retaining historical defaults.

| Commit | Scope |
| --- | --- |
| `85632d4` | Five verbs, package workflows, fixed protocols and initial tests |
| `5421d5a` | Avoid duplicate campaign members in persisted reports |
| `d725ab8` | Preserve recorded coordinate scales when reusing labels |
| `9fa77c7` | Read continuation counters from the committed checkpoint; source used by 8966340 |
| `b567d6e` | Per-chain worker budgeting and persisted status counters |
| `901594d` | Baseline and regression-target provenance checks |
| `9482677` | Correct the cache-variable collision that caused 8966340 to fail |

The last three commits and their additional tests **were not compute-verified**.
The passing suite below applies to `9fa77c7`, not to the final branch tip.
Historical artifacts and their manifests were not rewritten. The training
checkpoint remains tied to its original pinned source; no compatibility bypass
or manifest migration was attempted.

## Recorded protocol and completed checks

The [protocol](phase3.protocol.json) was saved before submissions and remains
unchanged (SHA256 `91f4811db7b8c2ae155b5b775ae99a27b00b3ac94611f3cc9f0a2924fc75246c`).
It specifies case30, seed0, v4-OFF, K8, B1, fp32, lambda 1, LR 1e-4, two epochs,
and a restart immediately before epoch-1 validation. Selection ES must be at
most the maximum of fleet seeds 0/1/2 plus 0.005 at each epoch. The fixed ceilings
are 1.145403378340933 and 0.6283314689662721. Loss differences are diagnostics;
finite metrics and exact boundary restoration are required. No thresholds changed.

| Check | Recorded result |
| --- | --- |
| Fresh editable install with existing torch stack | Passed on `9fa77c7`; vendor packages reused read-only |
| Isolated install without torch | Passed expected-failure check; import error points to README install matrix |
| Wheel and both entry points | Passed on `9fa77c7`; package/resources included, experiment drivers excluded |
| Test suite | **159 tests + 3 subtests passed**, zero failures/errors/skips on `9fa77c7` |
| Full reused-label Stage A audit | Passed: 10,080 hours; 10,078 valid and 2 unresolved labels retained as masked |
| Data identity, demand law, splits and shard hashes | Passed; source labels read-only |
| Event support and headline selection | Completed; calibration-support rule selected voltage bus 12 |
| Subset parity (200) | Passed bit-identically on CPU; seeded IDs recorded first; 54.30 seconds |
| Reused coordinate scales | Exact released bytes/order, SHA256 `29a824afcb46e1dccc00e11aa8c1a0697cf0207db1adfd50a3c86aee2797ca7e` |
| Training prefix | 4,090 updates, K8/B1/fp32/OFF; frozen backbone unchanged |
| Persisted training metrics | All 40 fixed-cadence rows finite, covering steps 100–4,000; final step metrics also persisted in adapter status |
| Controlled boundary checkpoint | Saved at epoch 0, step-in-epoch 4,090, before validation; restore not executed |

The initial preparation allocation, 8966309, passed its installation, 155 tests
and 3 subtests, full audit and subset parity. A subsequent preflight inspection
found that refitting the reused labels changed their coordinate order and was
incompatible with the released embedding cache. No training used those scales.
The corrected preparation verifies the recorded scales' train-shard provenance
and copies their exact bytes into a new output directory. Both preparation
records remain intact.

## Final job failure and retained checkpoint

The cache-reuse loop assigned a tensor-cache object to `saved`, overwriting the
dictionary of chain metadata. After the training function completed its
requested 4,090 updates and saved state, `train_chain()` evaluated
`saved['pilot']` while writing `chain.status.json` and raised `KeyError: 'pilot'`.
The checkpoint and boundary receipt had already been written successfully.
The training loop itself wrote no `failure.json`.

The correction uses a distinct `cached_state` variable. A parameterized test now
covers both a pre-existing cache manifest and copying a reusable embedding
cache. **That correction and test are unverified because compute is closed.**

Retained checkpoint:
[`latest.pt`](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/package/phase3_20261026_v2/campaign/case30_ieee/seed0/conditioned_lambda1_K8/adapter/latest.pt),
SHA256 `b1c0724d11ddf43e2dca730a5c191be9bce69e037c0e8da0663d665562b5385d`.
Its saved counters are 4,090 updates, zero completed epochs. `epochs.json` is
empty and no restore-audit files exist. Last training loss is 0.5471112132072449;
this is not a selection-ES result or an acceptance verdict.

## Outstanding gates

| Required gate | Status |
| --- | --- |
| Finish the two-epoch training and selection-ES acceptance | **not run—compute budget closed** |
| Restore across the epoch/validation boundary and verify exact state/journal recovery | **not run—compute budget closed** |
| CLI evaluation of three origins, C/P/P0/S/A/D and same-input P/Ipopt audit | **not run—compute budget closed** |
| CLI summary report over that evaluation | **not run—compute budget closed** |
| `pff verify --check regression`, metrics to 1e-7 | **not run—compute budget closed** |
| Phase 2 case30 50-hour audit + two-origin adapter-forward regression rerun | **not run—compute budget closed** |
| Phase 2 two-checkpoint, three-origin scorer/loader rerun | **not run—compute budget closed** |
| Final-source suite and install checks after the later fixes | **not run—compute budget closed** |

The completed full-label audit is not a substitute for the explicit recorded
50-hour numerical regression. Earlier Phase 2 passes are historical evidence,
not Phase 3 passes. No heavy login-node computation was substituted.

## Jobs, budget and evidence

| Job | Snapshot commit | Outcome | PBS elapsed | Worker time |
| --- | --- | --- | ---: | ---: |
| 8966309 | `85632d4` | Initial preparation checks passed; no training | 201 s | 199.91 s |
| 8966340 | `9fa77c7` | Exit 1 after step-4,090 checkpoint; final user-designated allocation | 866 s | 865.72 s |
| 8966377 | `9fa77c7` | Submitted before the stop; cancelled, exit 143; resume never started | 18 s | No completed worker record |

Each requested debug, one node, one isolated tile and a one-hour maximum.
Three submissions reserved 10,800 node-seconds; recorded PBS elapsed totals
1,085 node-seconds. Completed worker runtimes total 1,065.63 seconds. The original
four-allocation cap is retained as historical protocol; the user closed the
budget before any fourth submission. No further submissions are authorized.

- Initial evidence: [phase3_20261026_v1](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/package/phase3_20261026_v1).
- Corrected preparation/training evidence: [phase3_20261026_v2](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/package/phase3_20261026_v2).
- Training [source pin](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/package/phase3_20261026_v2/source_pin_final.json): SHA256 `2890d07d749685a05508e08a958308184cede2e6bc6bad0ea3a2349a27f1d422`.
- [Failed job record](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/package/phase3_20261026_v2/jobs/8966340.json) and [failure log](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/package/phase3_20261026_v2/logs/8966340/boundary.log).
- [Compute closure](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/package/phase3_20261026_v2/STOP.json), [budget ledger](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/package/phase3_20261026_v1/budget.json), and [scheduler records](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/package/phase3_20261026_v2/scheduler).
- [Machine-readable incomplete gate](phase3.json) contains evidence paths and hashes.

## Repository handover

Work remains committed on `cli`. `main` remains at
`31445dc60b3b2ac41ce237f94d9bf0bc53adb4d1` (`v0.2.1-dev`). Existing tags are
unchanged. No `v0.3.0-dev` tag, merge, push, publishing, or Phase 4 work was done.
The CLI is not release-accepted; evaluation, regression, full restart acceptance
and the later fixes remain unverified. CUDA and fresh large-grid generation were
not exercised by this acceptance run.
