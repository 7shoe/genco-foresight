# Stage 1 fleet verdict

OFF exceeds historical ON range; gap recorded.

The requested rule has no added slack: every OFF selected-checkpoint selection ES must be at or below the maximum historical ON value. Test results provide context; they do not change that rule.

Historical ON selected-checkpoint range: [0.278330659, 0.292115940]. All six training runs are complete. OFF test evaluation is still pending; this report is provisional only for the test table.

| Mode | Logical seed (RNG seed) | Selected ES | Best epoch | Last epoch | Last-epoch ES | Gap above ON maximum |
|---|---|---:|---:|---:|---:|---:|
| ON | 0 (42) | 0.278330659 | 14 | 17 | 0.287082683 | 0.000000000 |
| ON | 1 (1) | 0.292115940 | 19 | 19 | 0.292115940 | 0.000000000 |
| ON | 2 (2) | 0.286212626 | 14 | 17 | 0.300692208 | 0.000000000 |
| OFF | 0 (0) | 0.309088704 | 22 | 25 | 0.337181439 | 0.016972764 |
| OFF | 1 (1) | 0.292681884 | 16 | 19 | 0.304906952 | 0.000565944 |
| OFF | 2 (2) | 0.270779078 | 18 | 21 | 0.281821650 | 0.000000000 |

“Selected ES” is selection ES at best.pt, the checkpoint used for test evaluation. Last-epoch ES is shown separately. Historical logical seed0 used RNG42; the new fleet used literal RNG0. This is a historical three-seed comparison, not a paired experiment isolating determinism.

All test headline voltage metrics below are recomputed from the stored frozen-scorer component summaries using the same mechanically selected bus IDs [12]; historical headline bus lists are not mixed with the new list. Joint ES and free-generator masks/scales are unchanged. C/P/P0/S use the released reference forecasts.

| Mode | Seed | C ES | P ES | P0 ES | S ES | A ES | D ES |
|---|---|---:|---:|---:|---:|---:|---:|
| ON | 0 | 0.424719248 | 0.658759767 | 0.658759751 | 0.496909644 | 0.303336547 | 0.634101409 |
| ON | 1 | 0.424719248 | 0.658759767 | 0.658759751 | 0.496909644 | 0.297683424 | 0.633041368 |
| ON | 2 | 0.424719248 | 0.658759767 | 0.658759751 | 0.496909644 | 0.301374938 | 0.644595364 |
| OFF | 0 | pending | pending | pending | pending | pending | pending |
| OFF | 1 | pending | pending | pending | pending | pending | pending |
| OFF | 2 | pending | pending | pending | pending | pending | pending |

Confirmation job8962843 is contextual only. Six epoch comparisons were planned; 4 are persisted. OFF B epoch2 recorded0.484601428 versus the fixed upper threshold0.463022624, a gap0.021578804. The controller stopped at that failure, so epoch3 was not observed. The one-of-six wording refers to the planned comparisons and does not establish five observed passes.

No historical manifest, checkpoint or result was rewritten. No new training or acceptance threshold was introduced. This report does not reverse prior operational authorization to run OFF; it records that the latest requested historical-range criterion is not met.

Bulk evidence and all input SHA256 hashes: [recovery_20261025_v1/stage1_verdict.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/stage1_verdict.json).
Protocol: [recovery.protocol.json](recovery.protocol.json). Report data SHA256: `432dffd172c99a70e81301f6f6e4f2b1b5623750cc0e0812f49e79b427adab7c`.
