# Runs and final results

All compute jobs have ended. Evidence and pinned source snapshots remain under `/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO`; repository consolidation did not move or rewrite them. Times below refer to October 25, 2026 CT. Frozen and post-freeze results remain distinct.

| Work | Outcome | Evidence |
|---|---|---|
| Original fleet: case14/30/57/118, seeds 0/1/2 | Twelve runs ended; case118 reached its budget caps. All twelve evaluations and four three-seed mixtures complete. | [Training](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/fleet_20261025_off_v1/), [evaluation](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/evaluation/) |
| Ensemble depth: seeds 3–7 | Eleven early-stopped; nine budget-capped. The final continuation added no training. | [Window evidence](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/), [combined mixture table](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/postfreeze_20261025_final/combined_mixture_sizes.md), [figure](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/postfreeze_20261025_final/combined_mixture_sizes.png) |
| A0 and λ sweep | All five evaluated; A0 case118 budget-capped. Separate single-seed experiments, not fleet estimates. | [Cross-grid and ablation tables](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/cross_grid.md) |
| case500 K8 and case2000 K2 pilots | Four epochs each; not converged. Test scores use the three/two-epoch snapshots. | [Pilot snapshot results](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/pilot_results.json), [final selection curves](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/postfreeze_20261025_final/final_pilot_selection_curves.png) |
| Final training/evaluation: 8965746 / 8965775 | Training failed before optimizer updates; evaluation succeeded. Case14 eight-seed ES 0.3377790401. | [Final report](reports/v2/postfreeze.md), [terminal job accounting](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/postfreeze_20261025_final/final_job_accounting.json) |

The [19:00 freeze inventory](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/FREEZE.md) preserves run directories, snapshot provenance and status at freeze. The [post-freeze handover](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/postfreeze_20261025_final/handover.json) records later outcomes. Neither has been rewritten for the repository cleanup. Consequently, their historical references to pending worktree consolidation are superseded by the [repository consolidation record](reports/v2/repository_consolidation.md).

Historical released artifacts remain at `runs/adapter_v3`, `runs/adapter_v4`, their data/evaluation directories, and `runs/replication` under the same GENCO root. They are read-only evidence, separate from these V2 run identities.
