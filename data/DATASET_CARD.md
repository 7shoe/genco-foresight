---
license: apache-2.0
language: en
task_categories:
- time-series-forecasting
---
# GENCO Foresight: temporal case30 v3 and v4

**Draft for review. Not uploaded.** HF organization: `<HF_ORG>`; repository:
`<HF_ORG>/genco-foresight-temporal-case30`. Participants: `<NAMES>`.
Affiliations: `<AFFILIATIONS>`. Contact: `<EMAIL>`. Code: `<GITHUB_REPOSITORY_URL>`.

## Data and intended use

Synthetic research benchmark, not an ERCOT grid model or dispatch/control product.
Each version has 20 episodes of 21 days:
10,080 hourly demand/AC-OPF snapshots, with train/selection/calibration/test
episode counts [10, 2, 4, 4]. Observe the preceding
72 hours to forecast the following 24.
Calibration and test each have 72 midnight origins from four source episodes.
Adjacent days and overlapping histories are correlated.

The historical EIA ERCO 2025 hourly aggregate profile is mapped onto
the fixed IEEE/PGLib case30 spatial demand pattern. The source profile, retrieval
provenance and hashes are included. This is a modeled geographic assumption.
Commitment, topology and benchmark costs are fixed; there are no ramps.
PowerModels/Ipopt labels include status, validity masks, residuals, objective,
duals, iterations and runtime. Numerical nonconvergence is not infeasibility.

V3 has 10,080 valid labels; v4 has 10,078
valid labels and 2 unresolved test hours
(test_02, hours 240 and 453). Observed demand remains available; missing solved
states are null/masked, never zeros or negative event labels.

## Construction and split history

All partitions use disjoint 21-day calendar-block pools. The final assignment is:

| Partition | One-based source blocks | Episodes |
|---|---|---:|
| Train | 1,4,8,9,12,16,17 | 10 |
| Selection | 5,13 | 2 |
| Calibration | 2,6,10,14 | 4 |
| Test | 3,7,11,15 | 4 |

Reuse stays within its assigned pool; only training reuses source blocks in the
final allocation. Eight trailing days of the source year are unused.

| Revision | Timing and reason |
|---|---|
| v1 | Initial 12/2/2/4 allocation and disjoint random source pools; pre-score audit exposed prevalence shifts. |
| v2 | Calendar-interleaved 12/2/2/4 pools; pre-score audit still found inadequate seasonal representation in two calibration source blocks. |
| v3 | Final 10/2/4/4 allocation, doubling calibration to four source blocks and 72 days. Prevalence became a reported diagnostic, not a training blocker. |
| v4 | Requested after v3 forecast scores. Same final partitions/noise/profile, nominal load range changes from 0.30–1.00 to 0.80–1.00. Adapter physics weight also changes from 0.01 to 1.0; this is a confounded two-change follow-up. |

V1→v3 design changes preceded forecasting scores; that disclosure does not make
this an untouched prospective benchmark. Full old datasets remain on Lustre;
the package includes their manifests/audit summaries.

Training-only affine profile scaling is applied to held-out profiles without
clipping. Both versions retain independent uniform bus noise ±3% and stationary
lognormal AR(1) factors (rho 0.9, innovation std 0.02). Qd uses baseline signed
power factor; buses with zero baseline Pd retain reactive-only baseline demand.
The original pretrained MVA normalizer is loaded, never refitted.

## Load range and support

OPFData nominal training multipliers were [0.8,1.2]. V3's nominal range extends
below this: only 289 of 1,728 forecast hours have
realized ref_t ≥0.8. Frozen GENCO approximation errors increase in lower bins.
V4's 0.80–1.00 endpoints are nominal: noise leaves
9.7506% of test active-bus/hour multipliers
below 0.8, including the history hours. Reference-profile bins are retrospective
diagnostics, never future model inputs or proof of complete feature-support overlap.

## Event prevalence

Voltage stress means the outer 5% of the permitted range or beyond; thermal
stress means loading >95% of rating. Cells show rate, positive/resolved counts,
and unknown counts. Hourly tables use forecast-target hours after history.

### V3 hourly

| Event | Train | Selection | Calibration | Test |
|---|---:|---:|---:|---:|
| voltage_bus_1 | 65.90% (2847/4320; unknown 0) | 61.81% (534/864; unknown 0) | 65.28% (1128/1728; unknown 0) | 64.06% (1107/1728; unknown 0) |
| voltage_bus_9 | 9.26% (400/4320; unknown 0) | 7.29% (63/864; unknown 0) | 8.97% (155/1728; unknown 0) | 8.22% (142/1728; unknown 0) |
| voltage_bus_11 | 33.31% (1439/4320; unknown 0) | 32.06% (277/864; unknown 0) | 32.87% (568/1728; unknown 0) | 37.44% (647/1728; unknown 0) |
| voltage_bus_13 | 28.96% (1251/4320; unknown 0) | 28.12% (243/864; unknown 0) | 27.78% (480/1728; unknown 0) | 33.28% (575/1728; unknown 0) |
| voltage_bus_15 | 6.53% (282/4320; unknown 0) | 7.64% (66/864; unknown 0) | 5.84% (101/1728; unknown 0) | 8.22% (142/1728; unknown 0) |
| voltage_bus_21 | 4.72% (204/4320; unknown 0) | 5.44% (47/864; unknown 0) | 3.70% (64/1728; unknown 0) | 6.42% (111/1728; unknown 0) |
| voltage_bus_22 | 6.83% (295/4320; unknown 0) | 7.52% (65/864; unknown 0) | 6.37% (110/1728; unknown 0) | 8.45% (146/1728; unknown 0) |
| thermal_line_1 | 27.82% (1202/4320; unknown 0) | 27.08% (234/864; unknown 0) | 27.03% (467/1728; unknown 0) | 32.81% (567/1728; unknown 0) |

### V4 hourly

| Event | Train | Selection | Calibration | Test |
|---|---:|---:|---:|---:|
| voltage_bus_1 | 100.00% (4320/4320; unknown 0) | 100.00% (864/864; unknown 0) | 100.00% (1728/1728; unknown 0) | 100.00% (1726/1726; unknown 2) |
| voltage_bus_9 | 0.00% (0/4320; unknown 0) | 0.00% (0/864; unknown 0) | 0.00% (0/1728; unknown 0) | 0.00% (0/1726; unknown 2) |
| voltage_bus_11 | 100.00% (4320/4320; unknown 0) | 100.00% (864/864; unknown 0) | 100.00% (1728/1728; unknown 0) | 100.00% (1726/1726; unknown 2) |
| voltage_bus_13 | 100.00% (4320/4320; unknown 0) | 100.00% (864/864; unknown 0) | 100.00% (1728/1728; unknown 0) | 100.00% (1726/1726; unknown 2) |
| voltage_bus_15 | 0.00% (0/4320; unknown 0) | 0.00% (0/864; unknown 0) | 0.00% (0/1728; unknown 0) | 0.00% (0/1726; unknown 2) |
| voltage_bus_21 | 0.00% (0/4320; unknown 0) | 0.00% (0/864; unknown 0) | 0.00% (0/1728; unknown 0) | 0.00% (0/1726; unknown 2) |
| voltage_bus_22 | 0.00% (0/4320; unknown 0) | 0.00% (0/864; unknown 0) | 0.00% (0/1728; unknown 0) | 0.00% (0/1726; unknown 2) |
| thermal_line_1 | 100.00% (4320/4320; unknown 0) | 100.00% (864/864; unknown 0) | 100.00% (1728/1728; unknown 0) | 100.00% (1726/1726; unknown 2) |

V4's frozen headline events are degenerate on resolved labels in every partition:
buses 1/11/13 and line 1 are always positive; buses 9/15/21/22 always negative.
It cannot demonstrate event discrimination. No prevalence-based tuning follows.

### V3 daily-any

| Event | Train | Selection | Calibration | Test |
|---|---:|---:|---:|---:|
| voltage_bus_1 | 95.56% (172/180; unknown 0) | 94.44% (34/36; unknown 0) | 98.61% (71/72; unknown 0) | 88.89% (64/72; unknown 0) |
| voltage_bus_9 | 61.11% (110/180; unknown 0) | 77.78% (28/36; unknown 0) | 69.44% (50/72; unknown 0) | 63.89% (46/72; unknown 0) |
| voltage_bus_11 | 61.67% (111/180; unknown 0) | 83.33% (30/36; unknown 0) | 69.44% (50/72; unknown 0) | 65.28% (47/72; unknown 0) |
| voltage_bus_13 | 58.89% (106/180; unknown 0) | 66.67% (24/36; unknown 0) | 66.67% (48/72; unknown 0) | 59.72% (43/72; unknown 0) |
| voltage_bus_15 | 28.89% (52/180; unknown 0) | 36.11% (13/36; unknown 0) | 31.94% (23/72; unknown 0) | 34.72% (25/72; unknown 0) |
| voltage_bus_21 | 23.33% (42/180; unknown 0) | 33.33% (12/36; unknown 0) | 20.83% (15/72; unknown 0) | 33.33% (24/72; unknown 0) |
| voltage_bus_22 | 29.44% (53/180; unknown 0) | 36.11% (13/36; unknown 0) | 37.50% (27/72; unknown 0) | 37.50% (27/72; unknown 0) |
| thermal_line_1 | 57.22% (103/180; unknown 0) | 66.67% (24/36; unknown 0) | 63.89% (46/72; unknown 0) | 59.72% (43/72; unknown 0) |

### V4 daily-any

| Event | Train | Selection | Calibration | Test |
|---|---:|---:|---:|---:|
| voltage_bus_1 | 100.00% (180/180; unknown 0) | 100.00% (36/36; unknown 0) | 100.00% (72/72; unknown 0) | 100.00% (72/72; unknown 0) |
| voltage_bus_9 | 0.00% (0/180; unknown 0) | 0.00% (0/36; unknown 0) | 0.00% (0/72; unknown 0) | 0.00% (0/70; unknown 2) |
| voltage_bus_11 | 100.00% (180/180; unknown 0) | 100.00% (36/36; unknown 0) | 100.00% (72/72; unknown 0) | 100.00% (72/72; unknown 0) |
| voltage_bus_13 | 100.00% (180/180; unknown 0) | 100.00% (36/36; unknown 0) | 100.00% (72/72; unknown 0) | 100.00% (72/72; unknown 0) |
| voltage_bus_15 | 0.00% (0/180; unknown 0) | 0.00% (0/36; unknown 0) | 0.00% (0/72; unknown 0) | 0.00% (0/70; unknown 2) |
| voltage_bus_21 | 0.00% (0/180; unknown 0) | 0.00% (0/36; unknown 0) | 0.00% (0/72; unknown 0) | 0.00% (0/70; unknown 2) |
| voltage_bus_22 | 0.00% (0/180; unknown 0) | 0.00% (0/36; unknown 0) | 0.00% (0/72; unknown 0) | 0.00% (0/70; unknown 2) |
| thermal_line_1 | 100.00% (180/180; unknown 0) | 100.00% (36/36; unknown 0) | 100.00% (72/72; unknown 0) | 100.00% (72/72; unknown 0) |

Daily events are unions within joint trajectories. An observed positive resolves
a daily union as positive; a negative requires all hours to be known negative;
otherwise the day remains unknown. V3 daily reliability panels additionally
require calibration prevalence 20–80% and the predeclared class-support rule.
Calibration/test prevalence ratios >1.5 are printed, not suppressed.

## Files, use and reproducibility

The local package contains datasets/temporal_case30_v3 and
datasets/temporal_case30_v4, each with episodes/*.json, manifest.json, generation,
quality and pilot reports. Audits and event counts live in datasets/audit_v3 and
datasets/audit_v4. SHA256SUMS covers the release payload; provenance.json records
source paths/hashes. Episode order, source dates, seeds and sampling law are in
the manifests. Users must split by episode/source pool and preserve masks.

No future solved state is an input. Selection chooses checkpoints; calibration
fits probability maps; test outcomes are reporting-only. Case30 has one generator
per generator bus, so no multi-generator Qg allocation was required.
All copied/adapted source retains attribution in THIRD_PARTY_NOTICES.md.
The source code and generated artifacts use Apache-2.0; upstream network,
checkpoint and profile notices remain applicable. Cite the original GENCO,
OpenGridFM, OPFData and PGLib sources alongside this project.

Dataset cards are publication drafts; local availability does not imply a public
HF repository. All bulk artifacts stay on /lus/flare until the owner publishes.
