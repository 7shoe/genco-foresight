# Independent recovery status

Updated 2026-10-25T18:05:46.008280+00:00. The original fleet controller, pinned snapshot and run directories remain untouched.

| Item | Status |
|---|---|
| case14/30/57 × seeds0/1/2 evaluation | Complete: all six modes, three-seed summaries, equal-weight mixtures and cross-grid table; job8964735 exited0 |
| case118 × seeds0/1/2 evaluation | All three complete; mixture65/72 origins saved, continuation8964879 submitted and queued |
| case500 seed0 K8 pilot | Running in8964751; full10080-label audit and all Stage A gates passed; finite persisted training metrics |
| case2000 seed0 K2 pilot | Running in8964751; full10080-label audit and all Stage A gates passed; finite persisted training metrics; labeled K=2 pilot |
| Stage1 verdict | Complete: OFFseeds0/1 exceed the historical ON selection range; gap reported, confirmation failure contextual |
| Automatic continuation | Active, detached controllers; maximum12 one-hour allocations per family; combined status watcher active |

[Live status](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/STATUS.md). [Six-mode cross-grid table](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/evaluation/cross_grid.md). [Stage1 verdict](reports/v2/stage1_verdict.md). [Protocol](reports/v2/recovery.protocol.json). [Handover evidence](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/handover.json).

Pilot epoch cap6, patience3, OFF/B1/fp32; per-epoch and15-minute atomic checkpoints. Previous data budget usage is carried forward. The pilots do not enter the cross-grid table. Queued/running work is not claimed finished; the live page updates independently of this conversation.
