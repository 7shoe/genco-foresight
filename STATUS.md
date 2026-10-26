# Status

All submitted jobs have ended. PBS was empty at the repository consolidation check on October 26, 2026 UTC. No further submissions are scheduled.

The sole checkout is `/home/siebenschuh/Projects/genco-foresight`, on `main`. The four auxiliary worktrees were removed after verified backups and duplicate-edit checks. All branches, original user edits, and both tags are preserved. [Consolidation record](reports/v2/repository_consolidation.md).

The original 12 fleet runs and their evaluations are finished. Of 20 ensemble extensions, 11 early-stopped and nine reached their budget cap. All five A0/λ experiments were evaluated; A0 case118 was budget-capped. Both large-grid pilots ended at four epochs without convergence; their test results use the earlier snapshots.

The final training continuation (8965746) failed before any extra optimizer step. Its independent evaluation (8965775) succeeded, including the case14 eight-seed mixture. The failed gate is unchanged. `release-window` retains the unmerged experiment/continuation work; no v0.2.0 tag has been created.

[Run locations and results](RUNS.md) · [Report index](reports/INDEX.md) · [Final continuation report](reports/v2/postfreeze.md). Older controller/status files can retain their last observed running state; use the terminal job records and final handover linked here.
