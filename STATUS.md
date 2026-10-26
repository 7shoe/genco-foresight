# Status

All experiment jobs and final verification job 8966156 have ended. No further submissions are scheduled.

The sole checkout is `/home/siebenschuh/Projects/genco-foresight`, on `main`. Remaining edits were resolved and `release-window` was merged with history in merge commit 00e7bb7. The original v0.1.0 and v0.2.0-rc tags are unchanged. The v0.2.0 release captures all weekend experiments' code and the results index. [Final consolidation and verification](reports/v2/final_consolidation.md).

The original 12 fleet runs and their evaluations are finished. Of 20 ensemble extensions, 11 early-stopped and nine reached their budget cap. All five A0/λ experiments were evaluated; A0 case118 was budget-capped. Both large-grid pilots ended at four epochs without convergence; their test results use the earlier snapshots.

The final training continuation (8965746) failed before any extra optimizer step. Its independent evaluation (8965775) succeeded, including the case14 eight-seed mixture. The failed gate is unchanged. Its code and diagnosis are now included in main; the missing resume-journal implementation defect is documented, not repaired by this consolidation.

Final verification passed: 140 tests and three subtests, with no skips or failures; the existing 50-hour case30 v4 audit and two-origin adapter forward gate also passed. RUNS.md indexes 215 run directories with source records, configurations, outcomes and provenance limitations. Branch history is retained in main and the verified Git bundles; the merged development branches are removed after the authorized push.

[Run locations and results](RUNS.md) · [Report index](reports/INDEX.md) · [Final continuation report](reports/v2/postfreeze.md). Older controller/status files can retain their last observed running state; use the terminal job records and final handover linked here.
