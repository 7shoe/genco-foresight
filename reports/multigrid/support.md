# Multigrid support

Multigrid support is integrated into `main` in `/home/siebenschuh/Projects/genco-foresight`.
It originated on branch `multigrid`, based on `v0.1.0` (`521c34203d8edb6c2ed3966e50f0ca9ca9a9569e`).
The user's later instruction replaces the original numerical gates with lightweight CPU loading/format checks.
Here, **supported means CPU data-format support**, not demonstrated training convergence or solver performance.

Each case passed Datakit network parsing, strict pinned-checkpoint loading, 200 hours of v4-law demand generation,
normalization and OPF graph preprocessing, and conversion of local OPFData sample 0. The 200 hours contain inputs,
not new Ipopt labels; unavailable states have explicit masks. No parity study, training run, replication, or new
Ipopt solve was performed. Job `8962134` was cancelled while queued, before execution.

## Measured coverage

Generator/branch columns show total/online rows. Reference buses are original network IDs. Dash means unmeasured.

| Grid | CPU format | Buses | Generators | Branches | Reference bus | Bus-Qg coordinates | CPU check seconds | Step seconds | Peak device memory | Ipopt seconds/solve |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
| case14_ieee | supported | 14 | 5/5 | 20/20 | 1 | 5 | 0.45 | — | — | — |
| case30_ieee | supported | 30 | 6/6 | 41/41 | 1 | 6 | 0.42 | — | — | — |
| case57_ieee | supported | 57 | 7/7 | 80/80 | 1 | 7 | 0.39 | — | — | — |
| case118_ieee | supported | 118 | 54/54 | 186/186 | 69 | 54 | 0.41 | — | — | — |
| case500_goc | supported | 500 | 224/171 | 733/728 | 311 | 113 | 0.45 | — | — | — |
| case2000_goc | supported | 2000 | 384/238 | 3639/3633 | 551 | 178 | 0.53 | — | — | — |

The CPU check time includes imports/loading/conversion; it is not an inference benchmark.
Per-case evidence and SHA256s are in [evidence.json](evidence.json). Assets and all generated arrays remain under
`/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/`; per-case outputs use `runs/multigrid/<case>/`.

## Network and coordinate contract

`GridSpec` retains the Datakit `Network`, original equipment rows, bus identities, reference bus, baseMVA,
gen-to-bus map, active-demand buses, rated branches, and online masks. Network dimensions drive graph conversion,
solver checks, adapter tensors, history caches, ridge outputs, and scoring-coordinate masks.
Graphkit compacts active equipment internally; predictions expand back to original rows with NaN plus masks.
Offline equipment is omitted from physical sums and scoring. The reference-angle coordinate comes from bus type,
including buses 69, 311, and 551 in the larger cases.

Pg remains per-generator. Qg scoring uses unique online generator buses in first-generator appearance order,
aggregating Ipopt generator labels. Individual generator Qg labels and history inputs remain intact. Case30's
one-generator-per-bus coordinate order is unchanged. No generator-Qg allocation is invented.
Scale floors and the energy/variogram/CRPS formulas are unchanged; the variogram topology excludes offline branches.
Multigrid event selection uses calibration support (five positive and five negative days, prevalence below 95%);
the short format check makes no event-support or calibration claim.

## Pinned Graphkit evidence

The inspected installation is
`/lus/flare/projects/FRAME-IDP/siebenschuh/envs/reason_venv_2026.1.0/lib/python3.12/site-packages/gridfm_graphkit/`,
at commit `f104900245cba26d1380037eea562cf633a97484`.

- Pg aggregation: `models/gnn_heterogeneous_gns.py:281` and `tasks/opf_task.py:569` use `scatter_add` with the generator-to-bus map.
- Qg reconstruction: `models/utils.py:86–113` produces bus Qg; `tasks/opf_task.py:602` exports bus Qg. This version has **no co-located generator-Qg split**. The user approved aggregated bus-Qg scoring after this inspection.
- Offline masks: `datasets/transforms.py:25–52` removes inactive generators and remaps links; `datasets/transforms.py:67–78` removes inactive branches. `datasets/task_transforms.py:46–47` applies these for OPF.

The HF base revision is `74e19e1e210a175aa17023f4db161f0ae9f753cb`. Per-case model, normalizer, YAML and PGLib
hashes and source paths are recorded in each external `assets.json`. PGLib attribution remains in `data/networks/LICENSE`.

## Removed grid literals

| Location | Replaced assumption |
|---|---|
| `data/loader/network.py` | Bus count 30, unique generator per bus, all equipment online, fixed default network path |
| `solver/adapter.py`, `solver/checks.py` | Fixed case file and 30-element voltage/demand shapes; checks now mask offline rows |
| `models/temporal.py` | Bus/gen/branch sizes 30/6/41, directed flow width 82, cached width 384 and conditioning width 48 |
| `training/inputs.py`, `training/loop.py` | Fixed history/target/noise widths; cache windows derive from loaded episode length |
| `training/forecaster.py` | Six-generator histories, 30-bus residual/forecast arrays, overwrite instead of sum for co-located generators |
| Evaluation/preflight/forecast/hybrid helpers | Fixed node counts and embedding reshapes; sampled outputs use original equipment masks |
| Snapshot/download/parity helpers | Fixed case30 dataset identities, local paths and export names |
| Event/scoring helpers | Fixed component counts, bus-ID-minus-one indexing, unmasked offline flow coordinates |

Numbers such as MATPOWER column index 6, six calendar features, timeout seconds, and historical case30 report text
are not grid dimensions. Historical release/post builders remain tied to the frozen v3/v4 record; they are not the
multigrid reporting entry point. Existing comments/docstrings were retained, with obsolete dimension comments updated.

## Run the lightweight check

```bash
python -m scripts.multigrid_assets
python -m scripts.multigrid_smoke
python -m scripts.multigrid_report
```

To check one grid, use `python -m scripts.multigrid_smoke --case case118_ieee`.
Existing workflows select a case using `GENCO_CONFIG=configs/multigrid/case118_ieee.yaml`; the central loader expands
that selector through the registry and v4 recipe, with separate per-grid output paths. These commands use the existing
environment; they do not install packages or replace Torch/Transformers.

## Validation and limits

The integrated CPU suite passed 38 tests, including selector overrides/interpolation, original row identities, offline masks,
co-located generation sums, Qg coordinate counts, reference angles, data loading, solver checks, and existing scorer
and monitor tests. Before the scope reduction, the case30 adapter/model/solver unit suite also passed (13 tests).
No claim is made about the original 50/200-hour Ipopt audits, 20-step training gates, case2000 backward gate, two-origin
released-number comparison, XPU memory, step time, or solve latency: those were not run after the user reduced scope.
