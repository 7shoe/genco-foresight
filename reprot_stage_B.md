# Stage B review report

Prepared 23 October 2026 by the implementation agent. This is a disclosure of the
work and its evidence, not an independent certification. **Evaluation completed;
calibrated forecasting was not demonstrated.** A has the lowest joint energy
score, but climatology has better voltage CRPS and coverage. Numerical feasibility
also remains inadequate for direct operational use.

**Task and data.** Forecast 24 hourly demand/AC-OPF states from 72 observed hours
on fixed-topology case30, with benchmark costs, fixed commitment, and no ramps.
AC-OPF means alternating-current optimal power flow; Pd/Qd are active/reactive
demand, Pg/Qg generation, and Vm/Va voltage magnitude/angle. GENCO supplies the
pretrained graph model for snapshot operating-state prediction.
`temporal_case30_v3` contains 20 × 21-day episodes: 10 train / 2 model-selection /
4 calibration / 4 test. Training has 4,090 hourly windows; evaluation has 72
midnight origins, totaling 1,728 target hours. The calibration partition was not
used to fit these forecasts or select A's checkpoint.

Demand applies the EIA ERCO 2025 profile to the IEEE benchmark: train-derived
0.30–1.00 affine scaling (held-out extrapolation allowed), ±3% noise and lognormal
AR(1), rho 0.9/sigma 0.02. Qd follows baseline power factor with a zero-Pd rule.
Ipopt labels passed Stage A physical/dual checks. These are synthetic states,
not ERCOT network measurements.

Source pools are disjoint across partitions; training reuses some blocks. The
four test blocks are not independent years. V1→v2→v3 revisions responded to
calibration/test event-prevalence audits before forecasting scores existed.
This is an exploratory benchmark, not an untouched prospective test; see
[dataset history](data/DATASET_CARD.md).

**Implemented methods.** All modes share the target origins, masks, scoring
scales and metric code. Neither future solved states nor future realized demand
are inference inputs; known future calendar features are allowed.

| Mode | Actual implementation |
|---|---|
| P | Separate ridge models for 21 active demand buses, 24 direct horizon heads, alpha 1, lags 1/2/24/48, calendar and past local Pd/Qd/Vm/Va/bus Pg/Qg; mean demand through frozen GENCO. |
| S | Same ridge, with 64 draws of whole 24-hour/all-bus residual blocks through frozen GENCO. **3,130 blocks** = 10 × 313 hourly origins, explicitly approved instead of the obsolete 3,756. Rolling fits use matured target windows from all training-episode prefixes, not fitted residuals or global UTC chronological fitting. Negative demand is clipped; clipping fractions are saved. |
| A | Frozen GENCO plus observed-variable projections, per-type GRU width 32, memory projection 32→48, demand head and rank-8-plus-diagonal innovations. Each sample's Pd/Qd reaches its own conditioned graph; Qg and flows are derived. History includes past solved states, availability masks and frozen embeddings. |
| D | A's **exact same Pd and Qd samples**, passed through unconditioned frozen GENCO; isolates the combined conditioning/state-innovation effect, not each mechanism separately. |
| C | Added on request after main evaluation: 64 uniform draws with replacement from 210 complete midnight-aligned training-day demand/solved-state trajectories. Unconditional empirical climatology; no GENCO inference. |
| P0 | Added at the same time: 64 identical copies of saved P, scored as a zero-spread ensemble. |

A's loss is fair normalized ES + 0.01 × normalized squared P/Q imbalance; replay
was cut. AdamW: lr 1e-4, weight decay 0.01, clip 1, constant schedule. Eight
training samples form 192 graphs; evaluation uses 64 samples. Ridge has local
lagged features; A also has graph embeddings and full history. Representations
are not matched.

**Training and provenance.** Seed 42 only. Seven Aurora `debug` jobs completed
78,135 updates. The budget ended training: 21,585.3 s worker wall (including
startup/validation/checkpoints), 1,268.7 s queue wait, 22,994.0 s calendar elapsed;
cache preparation/resume tests are additional. Selection ES chose epoch 18
(0.399655). Patience 3 was configured; early stopping did not end this run.

The adapter has **91,461 trainable parameters + 884 buffer elements = 92,345
state-dict elements**. The earlier version of this report incorrectly described
the counts and hashes as discrepancies: it compared parameters with all state
elements, and named-tensor bytes with serialized file bytes. The supplied
`9dbf…` is the correct base **file** SHA256; `df0c…` hashes named tensor contents.
Base parameters/buffers remained frozen. Historical run manifests are retained;
[the accounting correction](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/stage_b/diagnostics/parameter_hash_accounting.json)
records both definitions and every buffer.

| Asset | Recorded identity |
|---|---|
| Graphkit | `f104900245cba26d1380037eea562cf633a97484`; datakit 1.1.0 |
| Base model | `gridfm/genco-opfdata`, revision `74e19e1e210a175aa17023f4db161f0ae9f753cb`, `case30_ieee/seed42`; supplied normalizer restored, never refit |
| Selected adapter checkpoint SHA256 | `4153e4b60eb3b2afd298d7fd6833f882c4fee0ab35cb79ab80b807fee2c543d4` |
| Base checkpoint file SHA256 | `9dbf5596b4eb63a3b0e23a7d9a84427c33f3829a07ad2d7dbd12e99e364545a4` |
| Frozen backbone state-dict SHA256 | `df0c0596b72eed5951189c854ae8f07063c389b08bea6f05d3fe00f8fb9eba4a` (sorted names and CPU tensor bytes, including buffers) |

**Forecast results.** Lower errors/scores are better. Pd uses active buses; Vm
uses the frozen headline buses {1,9,11,13,15,21,22}. These are raw forecasts.

| Mode | Joint ES | Variogram p=.5 | Pd MAE, MW | Vm MAE, p.u. | Vm CRPS, p.u. | Line-1 loading MAE |
|---|---:|---:|---:|---:|---:|---:|
| C | 0.549374 | 0.027077 | 1.8507 | 0.005019 | 0.003147 | 0.14453 |
| P | 6.403819 | 0.142790 | 0.8885 | 0.006693 | 0.006693 | 0.16482 |
| P0 | 6.403819 | 0.142790 | 0.8885 | 0.006693 | 0.006693 | 0.16482 |
| S | 3.576211 | 0.151530 | 0.8945 | 0.006779 | 0.006155 | 0.16350 |
| A | 0.429958 | 0.035465 | 1.0081 | 0.004961 | 0.004164 | 0.15245 |
| D | 5.428913 | 0.168274 | 1.0081 | 0.006868 | 0.006596 | 0.16737 |

Coverage entries below are **50% / 90%** nominal intervals. Loading is a fraction
of rating. A/D demand coverage is identical by construction.

| Mode | Active Pd coverage | Headline Vm coverage | Line-1 loading coverage | Vm 90% width, p.u. |
|---|---:|---:|---:|---:|
| C | 38.2% / 77.8% | 39.6% / 77.3% | 46.1% / 86.3% | 0.012900 |
| P | 0% / 0% | 0% / 0% | 0% / 0% | 0 |
| P0 | 0% / 0% | 0% / 0% | 0% / 0% | 0 |
| S | 45.7% / 86.1% | 4.1% / 10.1% | 8.9% / 20.3% | 0.004169 |
| A | 16.5% / 37.4% | 20.1% / 28.8% | 37.6% / 59.6% | 0.009438 |
| D | 16.5% / 37.4% | 1.4% / 3.6% | 6.5% / 13.8% | 0.001593 |

A improves joint ES but loses to ridge on demand MAE and to C on voltage CRPS,
coverage, variogram score and line-1 loading MAE. C itself undercovers. No
confidence intervals or significance tests establish robustness of these rankings.

The [scorer](scoring.py) reports physical MAE/MSE, fair CRPS, energy/variogram,
widths/coverage and randomized finite-ensemble PIT. Joint ES uses training scales
and division by sqrt(observed dimension); it is scale-dependent, not a calibration
certificate. Variograms use fixed temporal/connected-bus pairs. Missing targets
are masked. All-component coverage includes constant coordinates; headline Pd
coverage excludes inactive buses.

**Numerical audit and probe.** The same-input audit covers all 72 origins but
only leads 1/7/13/19/24 and samples 0/32 (P: sample 0). Of 1,800 unique Ipopt
requests, 1,797 resolved; three S cases remain masked/unresolved, not infeasible.
A/D reuse identical numerical references. All-component paired MAE is:

| Mode | Vm, p.u. | Pg, MW | Loading |
|---|---:|---:|---:|
| P | 0.004926 | 9.3304 | 0.08169 |
| S | 0.005176 | 9.6315 | 0.08671 |
| A | 0.005595 | 12.7208 | 0.02406 |
| D | 0.005442 | 8.7840 | 0.08275 |

A improves loading approximation but worsens Vm/Pg relative to D. Raw mean
absolute active-power imbalance is **5.46 MW per bus**. Raw/refined forecast
errors are separate, matched-subset results, not full 64-sample refinement.
Numerical references do not certify global optimality.

The actual probe compares **raw + calendar (142 features)** with **raw + calendar
+ GENCO latents (526)**, rather than an embedding-only probe. It uses cached
official `return_embeddings=True` outputs, not CLI parquet export. Ridge alpha
is selected on selection episodes. Normalized test MSE worsens from
0.034263→0.036039 at +1 h and 0.184358→0.195620 at +24 h. This is a negative
incremental-feature result, not proof that embeddings contain no temporal
information; unequal feature dimension is a confound.

**Requested Gate B follow-ups: diagnostics only.** These analyses were requested
after the original scores. No training, checkpoint selection, scorer modification,
new ensemble draws, GENCO forwards, or numerical solves followed. The complete
[diagnostic report](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/stage_b/diagnostics/report.md)
repeats every numerical table for all forecast hours and the retrospective
`ref_t >= 0.8` slice, including physical scores, headline intervals, Pg,
same-input approximation, matched raw/refined errors and the saved probe.

The slice contains **289/1,728 forecast hours (16.72%)**, from 35 origins but
zero complete 24-hour days. By test episode, counts are 37/63/189/0. Including
initial history, 322/2,016 test hours (15.97%) qualify. The original train-derived
profile transform is used without clipping/refitting. This is a realized-profile
proxy for checkpoint support, not proof that forecast demand graphs lie within
the pretraining distribution or that OOD causes the errors. Restricted ES retains
the 24-hour layout, masks other hours, normalizes by retained dimensions and
averages equally over nonempty origins; marginal metrics weight valid coordinates.

P's same-input audit has only five existing leads per origin. Its imbalance below
is averaged over those audited samples and 30 buses, using predicted demand.
Pg here averages all six generators.

| Realized ref_t | Forecast hours | Audited hours | Vm MAE, p.u. | Pg MAE, MW | Mean abs(P), MW/bus | Mean abs(Q), MVAr/bus |
|---|---:|---:|---:|---:|---:|---:|
| <0.6 | 1,057 | 205 | 0.006015 | 13.7604 | 8.5036 | 5.3534 |
| 0.6–<0.8 | 382 | 74 | 0.004748 | 6.1609 | 1.8854 | 1.3317 |
| >=0.8 | 289 | 81 | 0.002332 | 1.0141 | 0.2150 | 0.1438 |

Raw balance means on **all saved samples and forecast hours** are shown next;
each cell is **full set / >=0.8 slice**. Four Pg coordinates are fixed zero, so
all-six coverage is inflated; free-generator coverage excludes them. P0 matches P.

| Mode | Mean abs(P), MW/bus | Mean abs(Q), MVAr/bus | Pg MAE, MW, all 6 | Pg 90% coverage, free 2 |
|---|---:|---:|---:|---:|
| C | ~0 / ~0 | ~0 / ~0 | 7.1159 / 13.7754 | 76.24% / 32.70% |
| P | 6.0335 / 0.5184 | 3.8350 / 0.3582 | 10.7506 / 2.4808 | 0% / 0% |
| S | 6.8680 / 0.7936 | 4.2022 / 0.5145 | 10.7530 / 2.5335 | 17.85% / 72.66% |
| A | 5.4641 / 3.3480 | 0.9304 / 0.7993 | 7.7273 / 3.6782 | 53.56% / 69.55% |
| D | 6.6364 / 0.5949 | 4.1810 / 0.4257 | 11.0658 / 3.7638 | 5.82% / 15.22% |

Joint ES uses **Pd 21, Qd 0, Vm 30, Va 29, Pg 2, Qg 6, flows 164** per
hour: 252 total, or 6,048/day. Flows are 41 each of Pf/Qf/Pt/Qt; loading is
excluded. Qd is a deterministic power-factor copy; Va excludes the reference;
Pg excludes fixed generators. Scales are `max(train population std, floor)`
from 5,040 unique training hours, with floors 1 MW/MVAr and 0.01 p.u./rad.
[The coordinate CSV](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/stage_b/diagnostics/coordinates.csv)
gives every index, scale and unit; scales remain unchanged in the restricted slice.

Group-restricted ES below uses the same estimator and scales but each group's own
dimension normalizer. **These group scores do not add to joint ES.** Qd is N/A,
not zero. Again cells show **full set / >=0.8 slice**.

| Mode | Pd | Vm | Va | Pg | Qg | Other: flows |
|---|---:|---:|---:|---:|---:|---:|
| C | .5577 / .9593 | .4536 / .9773 | .6290 / .9697 | .7177 / 1.4572 | .7034 / 1.2042 | .5375 / .8554 |
| S | .3124 / .3679 | .6095 / .3503 | 1.2185 / .2137 | 1.4975 / .4454 | 1.8578 / .6978 | 4.3681 / .5567 |
| A | .4181 / .5314 | .4562 / .3803 | .5104 / .2644 | 1.0554 / .5487 | .5694 / .5311 | .4130 / .4057 |
| D | .4181 / .5314 | .6713 / .4804 | 1.2686 / .3405 | 1.5890 / .7136 | 1.9964 / .8724 | 6.6546 / .8011 |

An additional **exact additive allocation** splits every Euclidean distance by
`group squared distance / total distance`, then applies the unchanged fair-ES
pair subtraction and joint normalization. It sums to the original ES; it is an
algebraic attribution, not a standalone group score or causal decomposition.
Under this allocation flows account for 97.17% of S's full joint ES and 97.85%
of D's. Thus the aggregate comparison is heavily influenced by flow errors.

| Mode | Joint ES, >=0.8 | Headline Vm CRPS, >=0.8 | Vm 90% coverage, >=0.8 |
|---|---:|---:|---:|
| C | 0.910226 | 0.007326 | 34.95% |
| P | 0.725109 | 0.002481 | 0% |
| S | 0.500862 | 0.002012 | 40.34% |
| A | 0.393577 | 0.002801 | 24.77% |
| D | 0.730587 | 0.003105 | 15.22% |

A retains the lowest conditional joint ES, while S has better voltage CRPS,
coverage and active balance in that slice. On the same-input high-profile audit,
A also has worse Vm/Pg/loading approximation than D. These are material limits
on the adaptation claim. The probe remains negative on the slice: raw versus
raw+GENCO normalized MSE is .05921/.07059 (+1 h, 321 target hours) and
.38129/.43748 (+24 h, 312 target hours). Probe filtering uses target-hour ref_t
and its original all-hour origins, without refitting any head.

The diagnostic job used 12 CPU workers. Three added tests passed, all 2,088
original per-origin artifact hashes were checked before/after processing, and
original scorer/model/checkpoint/global-input hashes were unchanged. ES allocation
closure was checked for every mode/origin/scope; full probe scores reproduced.

**Verification and disclosed gaps.**

- Recorded checks passed: six scorer/forecaster tests plus one point-mass test;
  2,088 per-origin artifact hashes; exact A/D demand identity, exact P0 copies,
  and reconstruction of C draws from training-only blocks. Earlier CPU and XPU
  resume checks passed, including the 200-step XPU kill/resume comparison.
  Hashes establish integrity, not scientific correctness; the implementing
  agent produced these checks, and no independent full reproduction is claimed.
- The first P0 run failed an overly strict ES equality assertion. The frozen
  multi-sample ES uses a float32 count square root, while P uses float64.
  The revised check permits relative ES error 1e-7; observed maximum was
  2.57e-8 (4.35e-7 absolute). Marginal/interval differences were ≤3.41e-13.
  The scorer was not changed for this control; the failed attempt is preserved.
  P's M=1 and P0's M=64 randomized rank PIT are intentionally not identical.
- **Unfixed resume-contract issue, identified by static inspection:** evaluation augments `adapter_v3/manifest.json`
  and preserves `manifest.training.json`; `scoring.py` was also extended after
  training. The current [training loader](training/loop.py) still requires an
  exact newly generated manifest match, so resuming this existing run with the
  current files would be rejected. The recorded historical resume test does not
  validate this post-evaluation state. Evaluation uses the archived training
  manifest. No new resume execution or fix was performed for this report.
- Data/headline selection used exploratory event audits; C/P0 were added after
  original test scores. They are disclosed additional controls, not preregistered
  experiments. No A retraining or checkpoint/scorer tuning followed those controls.
- Event Brier/reliability, post-hoc maps, hybrid warm-start timing, F, larger grids,
  outages and repeated seeds remain unperformed in this stage. A low joint ES
  supports neither calibrated stress probabilities nor a solver speedup claim.
- Portable, clean-checkout reproduction remains incomplete.
  Paths and existing Aurora/Julia assets are required; no portable Stage B bundle
  or Git release tag was established. Source hashes are recorded. Vendor Torch
  and Transformers were unchanged. Prediction generation used 12 XPU workers;
  C/P0 used 12 CPU workers on a compute node; final reporting used one CPU thread.

**Evidence and reproduction boundary.** Define the following absolute roots;
all bulk artifacts are on Lustre, not in this checkout:

```text
G = /lus/flare/projects/FRAME-IDP/siebenschuh/GENCO
B = G/runs/stage_b
T = G/runs/adapter_v3
```

| Review question | Evidence |
|---|---|
| Frozen inputs, scales and method identity | `B/protocol.json`, `B/forecaster/manifest.json`, `B/controls/protocol.json`; [configuration](config.yaml), [forecaster](training/forecaster.py), [adapter](models/temporal.py) |
| All variables, counts, widths, PIT and saved samples | `B/{C,P,P0,S,A,D}.json`; `B/origins/<id>/`, `B/controls/origins/<id>/`; [full report](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/stage_b/full_report.md) |
| Numerical references and raw/refined errors | `B/origins/<id>/solver_references.json`, `*_audit.json`, `*_solver_subset.npz` |
| Probe comparison, coefficients and targets | `B/probe/{protocol,report}.json`, `coefficients/`, `predictions_h*.npz` |
| Passed checks and failed control attempt | `B/verification.json`, `tests.xml`, `controls_tests.xml`, `controls_attempt01_float_precision/`; `T/resume_verification.md` |
| Training history and corrected parameter/hash accounting | `T/manifest.json`, `manifest.training.json`, `epochs.json`, `chain.md`, `best.pt`; `B/diagnostics/parameter_hash_accounting.json` supersedes the earlier discrepancy interpretation |
| Post-Gate-B diagnostics and >=0.8 tables | `B/diagnostics/{report.md,report.json,protocol.json,coordinates.csv,tests.xml,job.json}`; [diagnostic worker](scripts/diagnose_stage_b.py), [presentation pass](scripts/report_gate_b_diagnostics.py) |

From this checkout in the existing configured environment:

```bash
# Validate saved artifacts and rebuild tables/figures; does not rescore every NPZ.
python -m scripts.report_stage_b
# On an allocated node, evaluate/resume the frozen experiment and its controls:
python -m scripts.evaluate_stage_b run --device xpu
python -m scripts.embedding_probe
python -m scripts.score_controls
# Diagnostics only, using existing ensembles and solver references:
python -m scripts.diagnose_stage_b
python -m scripts.report_gate_b_diagnostics
```

Completed origins are reused only after matching hashes; these commands are not
a forced fresh independent rerun. A reviewer without Lustre access needs the
referenced artifact bundle. The most consequential review targets are scale and
mask choices, causal feature construction, residual-pool fitting, the limited
numerical audit, and the distinction between joint-score improvement and calibration.
