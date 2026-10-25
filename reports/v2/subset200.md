# Seeded subset parity and independent continuation

Recorded 2026-10-25T09:25:31.200563+00:00. The user authorized **subset parity (200)** for case118, case500 and case2000, CPU, a20-minute cap per grid, and bit-identical output acceptance. Protocol and all600 IDs were saved before execution. NumPy default_rng(42) selects200 without replacement from each official/local test-ID intersection; IDs are sorted for execution. Existing32-snapshot independent official-parquet conversion evidence remains separate and hash-linked. This amendment does not waive solver or full temporal data audits.

The dataset card now labels this scope explicitly. Per-dataset cards are written with actual gate results on pass. Historical released case30 v3/v4 artifacts and all running source snapshots remain unchanged.

## Execution

Case118 runs on node x4211c3s1b0n0 within **8963510**, the existing fleet allocation. The official CPU evaluator's inherited MPI rank context is removed: the original failure log contains PMIx_Fence/MPI_Init_thread failure, then timeout. On strict subset pass the previous parity directory and timeout record are archived, and three case118 slots are released into that same allocation. Training still waits for its data audit and cache.

Pilot job **8963555** ended before this amendment launched. Both32-snapshot parity checks passed there, but both grids failed the cold-solver audit. The authorized200-snapshot check is submitted as **8963648**, debug, one node, one hour; its two CPU checks each have a1200-second subprocess cap. Their wall cost is charged to the original per-grid data budget. Original solver failures remain blockers: case500 has reference-angle error0.0936725 rad; case2000 reports TIME_LIMIT at5.73 seconds with P/Q errors2.24MW/0.895Mvar. No budget or threshold was relaxed to retry these failures.

## Standalone resubmission

[scripts/keep_submitting.py](../../scripts/keep_submitting.py) is running detached on aurora-uan-0012, PID106738, parent PID1 and its own session. It does not depend on Codex or this conversation. It polls both original, locked, pinned family controllers every30 seconds and permits at most12 additional allocations per family, while retaining all original cumulative budgets and scientific stops. The original continuous controller was replaced only after it was stopped; no competing submitter or running compute worker was cancelled.

Runtime: `/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/subset200_20261025_v1/continuation.json` and `continuation.log`. The exact invocation is:

```bash
cd /lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/subset200_20261025_v1/source
/lus/flare/projects/FRAME-IDP/siebenschuh/envs/reason_venv_2026.1.0/bin/python -u -m scripts.keep_submitting --control /lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/subset200_20261025_v1
```

The family locks prevent duplicate controllers. A missing boundary, failed scientific gate or exhausted budget is not a reason to blindly launch12 failing jobs. Thus pilot training will remain parked after its subset check unless its separate solver issues are resolved under an authorized protocol. Independence from this chat does not promise survival of a login-node reboot.

## Verification and evidence

Two lightweight tests passed: seeded unique sampling and removal of parent MPI rank variables while retaining PBS/vendor environment. Numerical parity runs only on compute. Python/PBS syntax and whitespace checks passed. Implementation commit `4e37e2c46fe7d229be1e3e32cd713305e70e378b`;219 snapshot files are hashed. Original multigrid index and unrelated edits remain intact.

- Protocol and IDs: [subset200.protocol.json](subset200.protocol.json).
- Gate record: [subset200.json](subset200.json).
- Immutable source: `/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/subset200_20261025_v1/source`; pin SHA256 `ef37740ce6d37e2b16d083babf3388a4d3324d40905fea46420c7ec76f2112f3`.
- Per-grid evidence: `/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/subset200_20261025_v1/<case>/protocol.json`, `worker.log`, `gate.json`, `promotion.json`.
- Case118 rollout: `/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/subset200_20261025_v1/case118_ieee/rollout.log`, `released.json`, `released_workers.log`.
- Pilot continuation: original pilot `controller.json`, `logs/job_02.log` and budget ledgers.

No scientific result is claimed until the per-grid gate is written. No merge to main or publication is part of this submission record.

## Case118 result

Subset parity (200) passed: CPU outputs Vm/Va/Pg/Qg are bit-identical, every maximum absolute difference0.0. Numerical work took88.49 seconds; full child-worker wall cost98.90 seconds, below1200 seconds. Gate SHA256: `edc236dd8aff866eef2d5fc16b681c9af27852c3f3e38cc9d0c9be8c53580181`. The three case118 slots were released in8963510 at09:24:16 UTC; data generation resumed and training continues automatically after the remaining Stage A gates and cache. The external dataset card records **subset parity (200)**.

## Final subset results

All200-snapshot CPU comparisons passed with identical dtype and output bytes. Full subprocess wall times, including import/preparation, remained below the1200-second cap.

| Grid | Numerical/preparation seconds | Full worker seconds | Maximum absolute differences | Gate |
|---|---:|---:|---|---|
| case118_ieee | 88.49 | 98.90 | 0.0 for Vm/Va/Pg/Qg | pass |
| case500_goc | 200.14 | 210.01 | 0.0 for Vm/Va/Pg/Qg | pass |
| case2000_goc | 611.28 | 621.27 | 0.0 for Vm/Va/Pg/Qg | pass |

Each active parity path now points to its accepted subset evidence; the old parity directories are archived as `stage_a/parity_before_subset200`. All three external dataset cards state **subset parity (200)** and link their IDs and gate results. Gate/card hashes are in subset200.json.

Case118 data generation continues in8963510 after release; its training still waits for the complete temporal audit/cache. Both pilot solver failures remain preserved and blocking. Job8963648 was the successful additional debug allocation for subset verification, not authorization to bypass those failures. The detached resubmission driver remains active for eligible work, bounded by12 further allocations per family and the original cumulative budgets.
