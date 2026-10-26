---
license: apache-2.0
library_name: pytorch
base_model: gridfm/genco-opfdata
tags:
- time-series
- power-systems
---
# GENCO Foresight adapters

Research models for joint 24-hour power-system forecasts from 72 hours of
history. The implementation is provided by `power-fm-foresight`.

## Case-specific models

Each case uses its own frozen GENCO checkpoint, YAML and saved normalizer from
`gridfm/genco-opfdata`, revision `74e19e1e210a175aa17023f4db161f0ae9f753cb`,
under `<case>/seed42`. An adapter is trained for that case and its matching data
and coordinate scales. There are multiple adapter checkpoints per case because
seeds, recipes and ablations were varied; cross-case adapter transfer is untested.

| Cases | Evidence |
| --- | --- |
| 14, 30, 57, 118 | v4 fleet with three seeds per case; additional ensemble experiments |
| 500, 2000 | Unconverged pilots; case2000 uses K=2 |

Case30 also has historical v3/v4 replications and conditioning/physics-weight
ablations. Checkpoint identities and experiment status are in [RUNS.md](../RUNS.md).

## Method and findings

The temporal adapter conditions frozen GENCO and adds stochastic demand/state
innovations. Training uses fair energy score plus a P/Q balance penalty, normally
with eight trajectories per origin; evaluation uses 64. Data are synthetic,
Ipopt-labelled sequences with fixed topology. The v4 demand law is 0.80–1.00
nominal scaling and its default balance weight is 1.

For case30 v4 OFF across three seeds, adapter test ES is **0.307 ± 0.018** versus
**0.497** for ridge with residual sampling. However, nominal 90% headline voltage intervals
cover only **27.3%**, and mean absolute P imbalance is **1.18 MW/bus**.
Production determinism OFF was adopted with disclosure: the historical ON-range
criterion was not met. See the [comparison](../reports/v2/stage1_verdict.md).

## Limits and compatibility

These models are not validated for operational control or feasible dispatch.
Uncertainty is underdispersed, seed variability matters, and pilot/budget-capped
results do not establish convergence. ES values across cases or data laws are
not directly comparable because their training-derived scales differ.

Evaluation requires the matching case, base assets and recorded configuration.
Historical checkpoints can be evaluated; training resume requires compatible
source and full state. Adapter-only exports lack optimizer/RNG state.
Aurora is verified; CUDA and fresh large-grid generation through the packaged CLI
are untested. Model-hub publication remains pending.
