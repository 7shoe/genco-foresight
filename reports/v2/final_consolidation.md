# Final consolidation on main

PASS for the consolidation regression and index checks. `release-window` was merged with history in **00e7bb7972453c867d9b68c18a4196d23db0219d**. No package restructuring or new training was performed. The tag message for v0.2.0 is “all weekend experiments' code and results index”. Historical failed gates remain failed.

The checkout review found the configuration-path/interpolation fixes already committed in 4ebba7e, and the configuration tests/report corrections in a5f0723. Their `git diff main` was empty; no changes were discarded or recommitted. The remaining Stage B report link correction was preserved while renaming `reprot_stage_B.md` to `report_stage_B.md`. The CLI proposal was committed with that rename in 8e7d24e.

Verification protocol was committed before submission in 8987ed2. Job **8966156**, one debug node with a one-hour cap and no resubmission, tested a pinned snapshot of that commit. The existing vendor environment and PBS modules were used without package changes.

| Check | Result |
|---|---|
| Full test suite | 140 passed + 3 subtests; zero failures, errors or skips. |
| Existing regression test subset | 87 passed. |
| Released case30 v4 audit | All 50 train_00 hours passed; original audit errors compared at atol 1e-12, rtol 1e-10. |
| Released adapter_v4 forward | A/D arrays for test_00_0072 and test_00_0096 passed at atol=rtol=1e-5; K64, noise seeds 10042/10043, determinism ON. |
| PBS result | Finished, exit 0. Reported wall time 00:01:50; one node (0.03056 node-hours), within the one-node-hour cap. |
| Worker check time | Full suite 68.65 seconds; regression 35.06 seconds; total 103.74 seconds. |
| Index coverage | 215 independent run directories; every freeze-inventory directory and discovered training manifest covered; nine failed continuation children included. |
| Metadata validation | 591 input records rehashed; indexed directories/configurations and pinned verification evidence checked. |

The [run index](../../RUNS.md) links each run's configuration and source record, with full snapshot-record SHA256 values and producing commits where recorded. It includes released/replication/fleet/ensemble/A0/λ/pilot/post-freeze/failed outcomes and verification runs. Per-origin arrays, cache shards and temporary unit-test fixtures are components rather than separate experiments. Existing directory aliases are listed explicitly.

Some older manifests record file hashes but no producing commit or complete checkout snapshot. Those entries say **not recorded** and identify the partial source-hash evidence; a release tag is not substituted for a missing producing commit. The four-epoch pilots remain not converged, evaluated pilot snapshots retain their earlier epochs, and all budget-capped chains keep that label.

The final continuation wrapper still has the recorded missing committed metrics-journal defect. Job 8965746 failed before any extra optimizer update; job 8965775 independently completed the case14 evaluation. Passing this regression suite neither tests a repaired continuation nor changes that historical failure. The current user instruction authorizes merging/indexing this code and its failure evidence.

No scientific code changed after the tested snapshot; subsequent changes are reports, status, README and the run index. v0.1.0 and v0.2.0-rc are unchanged. Main and tags are pushed before deleting the merged development branches locally/on origin. The branches' complete history remains in main and in `/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/repository_backups/final_main_20261026T032238Z/v0.2.0-all-branches.bundle`; verification and publication records remain with the evidence below.

[Protocol](final_consolidation.protocol.json) · [Gate/accounting/index summary](final_consolidation.json) · [Full evidence](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/final_consolidation_20261026_v1/) · [Test log](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/final_consolidation_20261026_v1/logs/suite.log) · [Regression gate](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/final_consolidation_20261026_v1/regression/gate.json).
