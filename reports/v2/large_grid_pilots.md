# Large-grid OFF pilots: submission handover

Recorded 2026-10-25T09:06:57.901146+00:00. Job **8963555** is submitted to `debug`, one node, one hour. A detached controller (PID4121869 on aurora-uan-0012) polls PBS every30 seconds and submits the next allocation while authorized work and budget remain. Its source is the immutable snapshot below.

The existing debug-scaling controller (PID3440483) was not modified or restarted. It automatically submitted **8963510** after **8963092** ended with exit0; its second allocation is running. Confirmation8962843 was untouched. No pilot enters the small-grid cross-grid table.

| Pilot | Training seed | Tile (flat) | Global K | B | Precision | Determinism | Epoch cap | Patience |
|---|---:|---:|---:|---:|---|---|---:|---:|
| case500_goc, K=8 pilot | 0 | 0 | 8 | 1 | fp32 | OFF | 6 | 3 |
| case2000_goc, K=2 pilot | 0 | 2 | 2 | 1 | fp32 | OFF | 6 | 3 |

Covariance rank remains8 in both pilots. Selection uses64 samples on the existing selection partition and fixed selection noise. The v4 law0.80–1.00, λ1, AdamW settings, masks, train-derived scales,24 forecast hours, fair ES and frozen backbone with conditioning gradients are retained.

## Protocol and data gates

The protocol was recorded and committed before submission: [large_grid_pilots.protocol.json](large_grid_pilots.protocol.json). Each grid receives new data and artifact directories. No released data or model directory is a destination. The selected grid's seed42 checkpoint, YAML, normalizer and Graphkit pins remain enforced.

After CPU integration parity and checked solver/dual smoke tests, a200-hour solver pilot must pass. Four Ipopt processes per grid then own disjoint episode lists, covering all10080 hours in the20 fixed episodes. Atomic25-hour shard saves and exact demand/time/prefix validation support continuation. Solver worker crashes park that grid. Every completed dataset undergoes the full independent Stage A audit, event-support report, train-only scales and mechanical calibration-support headline selection before training starts.

The explicitly bounded CPU parity check covers the first32 sorted local/official intersection IDs, all independently compared with official parquet, batch1, using the unchanged tolerances. It does not claim full-intersection parity. The temporal audit is complete, including72 held-out daily origins. Embedding cache batches use24 graphs to bound preparation memory; complete cache episodes are immutable and reused.

## Budget and continuation

| Budget line | Per grid/pilot | Family limit |
|---|---:|---:|
| Data preparation, audits and embedding cache | 6 worker hours | 12 worker hours across2 grids |
| Training, selection, startup and saves | 12 worker hours | 24 worker hours across2 pilots |
| PBS reservation | 1 node ×1 hour per allocation | 24 allocations /24 node-hours |

Workers run concurrently, so worker hours and reserved node-hours are different quantities. Queue wait is excluded. Synthetic step measurements imply roughly7.8 hours for six training epochs before validation/startup overhead; twelve hours is a cap, not a completion guarantee. Patience3 may stop earlier. Checkpoints are atomic every15 minutes, each epoch and on the walltime guard. A changed source/configuration, failed audit, nonfinite training or unexpected worker failure parks the affected pilot. A missing boundary charges a full allocation conservatively. No failure relaxes a gate or spends the fleet budget.

## Verification and current result

Five lightweight tests passed in4.39 seconds: pilot/fleet isolation and shard assignment, data gating/budget stop, next-allocation submission, missing-boundary accounting, and an interrupted/resumed shard with exact prefix retention plus tamper rejection. Both generated data/training configurations passed recipe, grid/checkpoint, path and interpolation checks. Python syntax, PBS shell syntax and diff whitespace checks passed. Model/training tests run on compute before data preparation; Stage A and training results remain pending.

The original multigrid worktree and index, including staged reports/V2_IMPL.md and unrelated comments/docstrings, are preserved. Task implementation commit: `a1d0730305837511d9592509f488e30f941344d7` on multigrid-pilots, descended from multigrid via multigrid-fleet. Main and tags remain unchanged; no scientific gate is yet passed for these pilots, and nothing was pushed.

## Evidence

- Runtime status: [/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/large_grid_pilots_20261025_v1/STATUS.md](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/large_grid_pilots_20261025_v1/STATUS.md)
- Source snapshot: `/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/large_grid_pilots_20261025_v1/source` (212 pinned files).
- Protocol SHA256: `2c94249e53a0f16c8bdf9c87f7f0fa6abad382e3bf8f996f2f1dbd1387992995`.
- Source pin SHA256: `821b0460dddf5686fc749d1f166ee9addb68f5fb61774bf7cb11eb9156093df9`.
- Controller ledger/log: `/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/large_grid_pilots_20261025_v1/controller.json`, `/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/large_grid_pilots_20261025_v1/logs/controller.log`.
- PBS log: `/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/large_grid_pilots_20261025_v1/logs/job_01.log`.
- Per-pilot accounting: `/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/large_grid_pilots_20261025_v1/<case>/seed0/budget.json` and `boundaries/`.
- Compute tests: `/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/large_grid_pilots_20261025_v1/tests.json`, `tests.xml`, `logs/tests.log`.
- Gate record: [large_grid_pilots.json](large_grid_pilots.json).

These are pilots, not Stage2 sample-parallel acceptance or full replicated evaluations. The K=2 pilot changes Monte Carlo sample count explicitly and cannot be treated as K8 evidence.
