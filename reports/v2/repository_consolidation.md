# Single repository consolidation

This records the earlier filesystem-only cleanup. The later [final main consolidation](final_consolidation.md) merges release-window, resolves the preserved edits, verifies main, and archives/deletes merged branches under the user's subsequent authorization.

Completed on October 26, 2026 UTC. `/home/siebenschuh/Projects/genco-foresight` is the only remaining checkout, on `main`. The `-main`, `-fleet`, `-pilots` and `-release` directories were linked Git worktrees sharing this repository, and have been removed. No compute jobs were submitted.

Before removal, all five worktrees were archived with per-file hashes and verified by reading the archives back. A Git bundle preserves every branch/tag, and each worktree's index and staged/unstaged patches were saved. The fleet/pilot uncommitted files were additionally proved byte-identical to files retained in the original checkout. Their duplicate copies could therefore be removed without losing edits. Original modified/untracked files and staged file contents remain byte-identical.

The original staged `reports/V2_IMPL.md` is now clean because its exact contents were already committed on main. The two other staged report edits remain staged; the user's remaining edits and untracked files were not committed or stashed. The new documentation commit contains only the consolidation/status/index/run documents and verbatim copies of the final post-freeze reports.

| Branch | Preserved history |
|---|---|
| `main` | Validated replication, multigrid, fleet and pilot recovery code at 930fbe0, followed only by this documentation handover. |
| `multigrid` | 598542f; original development branch. |
| `multigrid-fleet` | 8954dd1; commits already integrated into main. |
| `multigrid-pilots` | a905fe6; commits already integrated into main. |
| `release-window` | a1b8901; retains A0, ensemble/ablation orchestration, final continuation and its failure diagnosis. Not merged into main. |
| `replication` | 8a27467; original replication branch retained. |

This is filesystem/worktree consolidation, not a new release gate. The failed final continuation remains failed, the release-window branch remains available in this single repository, and no v0.2.0 tag was created. Existing v0.1.0 and v0.2.0-rc tags are unchanged. Nothing was pushed. Pinned runtime snapshots, released artifacts, historical manifests and all results remain untouched on Lustre.

[Preservation archive and Git bundle](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/repository_backups/consolidation_20261026T030048Z/) · [File/hash inventory](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/repository_backups/consolidation_20261026T030048Z/preservation.json) · [Small machine-readable record](repository_consolidation.json) · [Run index](../../RUNS.md).

Verification covered: one remaining worktree/directory; retained branch heads; unchanged tag objects; exact preservation of original edits, staged contents, CLI proposal, configuration test and ignored local files; valid archived worktree hashes and Git bundle; and an empty PBS queue. No scientific implementation changed, so model/training tests were not rerun.
