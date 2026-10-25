# Stage 1 fleet verdict

OFF exceeds historical ON range; gap recorded.

The requested rule has no added slack: every OFF selected-checkpoint selection ES must be at or below the maximum historical ON value. Test results provide context; they do not change that rule.

Historical ON selected-checkpoint range: [0.278330659, 0.292115940]. All six training runs are complete. Test evaluation is complete.

| Mode | Logical seed (RNG seed) | Selected ES | Best epoch | Last epoch | Last-epoch ES | Gap above ON maximum |
|---|---|---:|---:|---:|---:|---:|
| ON | 0 (42) | 0.278330659 | 14 | 17 | 0.287082683 | 0.000000000 |
| ON | 1 (1) | 0.292115940 | 19 | 19 | 0.292115940 | 0.000000000 |
| ON | 2 (2) | 0.286212626 | 14 | 17 | 0.300692208 | 0.000000000 |
| OFF | 0 (0) | 0.309088704 | 22 | 25 | 0.337181439 | 0.016972764 |
| OFF | 1 (1) | 0.292681884 | 16 | 19 | 0.304906952 | 0.000565944 |
| OFF | 2 (2) | 0.270779078 | 18 | 21 | 0.281821650 | 0.000000000 |

Selected-checkpoint means ± sample SD: ON 0.285553075 ± 0.006916267; OFF 0.290849888 ± 0.019220406. The OFF mean is below the ON maximum, but two individual OFF seeds exceed it; averaging does not hide those gaps.

“Selected ES” is selection ES at best.pt, the checkpoint used for test evaluation. Last-epoch ES is shown separately. Historical logical seed0 used RNG42; the new fleet used literal RNG0. This is a historical three-seed comparison, not a paired experiment isolating determinism.

Historical ON seeds0/2 early-stopped at epoch17; ON seed1 reached its six-hour budget at epoch19. OFF seeds0/1/2 early-stopped at epochs25/19/21. Thus the common epoch19 endpoint for seed1 has different stopping reasons. The earlier stopping-epoch ±1 proposal is not part of this verdict.

All test headline voltage metrics below are recomputed from the stored frozen-scorer component summaries using the same mechanically selected bus IDs [12]; historical headline bus lists are not mixed with the new list. Joint ES and free-generator masks/scales are unchanged. C/P/P0/S use the released reference forecasts.

| Mode | Seed | C ES | P ES | P0 ES | S ES | A ES | D ES |
|---|---|---:|---:|---:|---:|---:|---:|
| ON | 0 | 0.424719248 | 0.658759767 | 0.658759751 | 0.496909644 | 0.303336547 | 0.634101409 |
| ON | 1 | 0.424719248 | 0.658759767 | 0.658759751 | 0.496909644 | 0.297683424 | 0.633041368 |
| ON | 2 | 0.424719248 | 0.658759767 | 0.658759751 | 0.496909644 | 0.301374938 | 0.644595364 |
| OFF | 0 | 0.424719248 | 0.658759767 | 0.658759751 | 0.496909644 | 0.327957972 | 0.618051959 |
| OFF | 1 | 0.424719248 | 0.658759767 | 0.658759751 | 0.496909644 | 0.301397925 | 0.636782924 |
| OFF | 2 | 0.424719248 | 0.658759767 | 0.658759751 | 0.496909644 | 0.292520124 | 0.626925427 |

Historical test A ES range: [0.2976834237653222, 0.3033365471731542]. OFF gaps above its maximum by seed: 0: 0.024621424, 1: 0.000000000, 2: 0.000000000.

| Forecast | Metric | ON mean ± sample SD | OFF mean ± sample SD |
|---|---|---:|---:|
| C | joint_es | 0.424719248 ± 0 | 0.424719248 ± 0 |
| C | vm_crps | 0.00141340883 ± 0 | 0.00141340883 ± 0 |
| C | vm_coverage90_pct | 79.8377752 ± 0 | 79.8377752 ± 0 |
| C | pg_free_mae | 6.04800063 ± 0 | 6.04800063 ± 0 |
| C | p_imbalance | 1.19621972e-13 ± 0 | 1.19621972e-13 ± 0 |
| P | joint_es | 0.658759767 ± 0 | 0.658759767 ± 0 |
| P | vm_crps | 0.00194824584 ± 0 | 0.00194824584 ± 0 |
| P | vm_coverage90_pct | 0 ± 0 | 0 ± 0 |
| P | pg_free_mae | 3.50529186 ± 0 | 3.50529186 ± 0 |
| P | p_imbalance | 0.0358644532 ± 0 | 0.0358644532 ± 0 |
| P0 | joint_es | 0.658759751 ± 0 | 0.658759751 ± 0 |
| P0 | vm_crps | 0.00194824584 ± 0 | 0.00194824584 ± 0 |
| P0 | vm_coverage90_pct | 0 ± 0 | 0 ± 0 |
| P0 | pg_free_mae | 3.50529186 ± 0 | 3.50529186 ± 0 |
| P0 | p_imbalance | 0.0358644532 ± 0 | 0.0358644532 ± 0 |
| S | joint_es | 0.496909644 ± 0 | 0.496909644 ± 0 |
| S | vm_crps | 0.00140033657 ± 0 | 0.00140033657 ± 0 |
| S | vm_coverage90_pct | 39.5133256 ± 0 | 39.5133256 ± 0 |
| S | pg_free_mae | 3.446597 ± 0 | 3.446597 ± 0 |
| S | p_imbalance | 0.049634626 ± 0 | 0.049634626 ± 0 |
| A | joint_es | 0.300798303 ± 0.00287033656 | 0.307292007 ± 0.018439507 |
| A | vm_crps | 0.00100120847 ± 7.92791396e-05 | 0.0014924299 ± 0.000987251843 |
| A | vm_coverage90_pct | 35.4190807 ± 6.28302842 | 27.2692159 ± 18.2327468 |
| A | pg_free_mae | 3.21811468 ± 0.0549236691 | 3.31891046 ± 0.252392962 |
| A | p_imbalance | 1.08671841 ± 0.0574162918 | 1.17665002 ± 0.158005366 |
| D | joint_es | 0.637246047 ± 0.00638672562 | 0.627253437 ± 0.00936978941 |
| D | vm_crps | 0.00185323657 ± 0.000149684088 | 0.00167184805 ± 0.000217150506 |
| D | vm_coverage90_pct | 10.0811124 ± 1.44727648 | 13.2676709 ± 5.59658757 |
| D | pg_free_mae | 3.8876396 ± 0.10942347 | 3.75088153 ± 0.159333953 |
| D | p_imbalance | 0.0353560486 ± 0.00175131365 | 0.0391144755 ± 0.00469849139 |

Confirmation job8962843 is contextual only. Six epoch comparisons were planned; 4 are persisted. OFF B epoch2 recorded0.484601428 versus the fixed upper threshold0.463022624, a gap0.021578804. The controller stopped at that failure, so epoch3 was not observed. The one-of-six wording refers to the planned comparisons and does not establish five observed passes.

No historical manifest, checkpoint or result was rewritten. No new training or acceptance threshold was introduced. This report does not reverse prior operational authorization to run OFF; it records that the latest requested historical-range criterion is not met.

Bulk evidence and all input SHA256 hashes: [recovery_20261025_v1/stage1_verdict.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/stage1_verdict.json).
Protocol: [recovery.protocol.json](recovery.protocol.json). Report data SHA256: `387b8e73b5e44b9564c1ee708041771b81f8598182a7296aac6e5fec202344f7`.
