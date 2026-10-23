# GENCO Foresight — implementation plan

**GENCO Foresight: Post-Training the Grid Solver of Tomorrow**

Repository: https://github.com/7shoe/genco-foresight

Build a small, standalone research demo that turns GENCO's spatial representations into a joint forecast of tomorrow's demand and AC-OPF operating states. Demonstrate calibrated risk estimates and selective numerical verification. Start with case30; expand only after the pilot passes.

This document is the implementation contract and contains four sequential, independently invocable coding-agent prompts. Invoke one stage at a time. The fourth stage includes the subsequently agreed evaluation, risk monitor, and visualization.

## Current evidence and asset inventory

The repository is README-only **according to the user**. This planning session could not inspect it: the GitHub connection returned 404, and the Aurora paths below were not mounted. Their contents remain user-reported until the implementation agent inspects them.

| Asset | Location / reported state | Required action |
|---|---|---|
| Repository | git@github.com:7shoe/genco-foresight.git | Ask for the actual local clone path; inspect README, AGENTS.md, and current changes. |
| Existing OPFData | /lus/flare/projects/FRAME-IDP/siebenschuh/acopf_datasets/ | Read-only; inspect schema and sample files. |
| Fixed topology | processed/T1/ under the data root | Cases 14, 30, 57, 118, 500. |
| Single outage | processed/N1/ under the data root | Same cases plus 2000; no curated N−2. |
| Canonical N1 splits | processed/N1/splits/<canonical-case>/ | Reuse frozen splits and nested 100/1K/10K subsets; discover actual filenames. |
| Existing loader | /home/siebenschuh/Projects/lumina-sdk/distillation/baselines/n1/curated_data.py | Adapt the minimum necessary code with attribution. |
| Loading instructions | /home/siebenschuh/Projects/lumina-sdk/distillation/baselines/artifacts/n1_datasets_20261022/README.md | Read before adapting the loader. |
| Standalone solver | /home/siebenschuh/Projects/acopf | Inspect package, API, tests, license, and installation method. |
| Integrated solver | /home/siebenschuh/Projects/lumina-sdk/distillation/solver/ | Inspect README.md, backends/reference.py, sparse_solver/solver.py. |
| Optional independent reference | /home/siebenschuh/Projects/lumina-sdk/distillation/RAMBO/julia/model.jl | PowerModels + Ipopt; optional, not an MVP runtime dependency. |
| GENCO source | gridfm/gridfm-graphkit, genco-paper-repro branch | Resolve and record a commit SHA; inspect model, physics, normalization, and license. |
| GENCO checkpoint | gridfm/genco-opf-datakit on Hugging Face | Ask for cached weights, matching YAML and normalizer first; resolve a revision. |
| Run outputs — fixed | /lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs | All run directories live here on Aurora. |

Reported OPFData size: 300,000 samples per case, split 240K/30K/30K using the canonical seed-42 split. N1 is randomly split across samples; the same outage may appear in all splits. It is **not** an unseen-outage benchmark. Static samples are not a time series.

Proposed writable locations, to confirm in Stage 1:

- Generated temporal data: /lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/data
- Downloaded model/source assets: /lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/assets

These locations are separate from the existing read-only data. On NVIDIA machines, explicitly configure equivalent paths outside the checkout.

## Final directory structure

Create files when their stage needs them; do not generate empty placeholder modules. Empty package initialization files are omitted below.

~~~text
genco-foresight/
├── README.md
├── PLAN.md
├── config.yaml
├── requirements.txt
├── THIRD_PARTY_NOTICES.md
├── .gitignore
├── models/
│   ├── genco.py
│   └── foresight.py
├── data/
│   ├── download/
│   │   └── prepare.py
│   ├── loader/
│   │   └── dataset.py
│   └── synthetic/
│       └── generate.py
├── solver/
│   ├── adapter.py
│   └── checks.py
├── training/
│   ├── common.py
│   ├── forecaster.py
│   ├── frozen_point/
│   │   └── run.py
│   ├── frozen_interpolation/
│   │   └── run.py
│   ├── frozen_sampling/
│   │   └── run.py
│   ├── adapter_only/
│   │   └── run.py
│   └── full_adaptation/
│       └── run.py
├── monitor.py
├── evaluate.py
├── visualization/
│   ├── app.py
│   ├── grid.py
│   └── plots.py
└── tests/
    ├── test_data.py
    ├── test_solver.py
    ├── test_model.py
    ├── test_monitor.py
    └── test_smoke.py
~~~

There is no repository-local runs directory. Each external run contains only what it uses: configuration, asset/version manifest, checkpoints, predictions, solver records, metrics, and figures.

### Ownership and simplicity

- models/ owns reusable inference. It must not import training/, visualization/, or the user's SDK.
- data/ owns acquisition, graph conversion, temporal generation, and window loading.
- solver/ is a small numerical interface plus independent physical checks.
- training/common.py owns one shared training loop, device handling, losses, checkpointing, and run creation. Recipe files are short entry points.
- monitor.py computes events, fits/calibrates small risk models, and selects scenarios. It does not render the UI.
- evaluate.py produces the same metrics and result format for every mode.
- visualization/ reads those results; it does not train models or redefine metrics.
- Use ordinary functions, dictionaries, PyTorch modules, one YAML configuration, and argparse. No registries, plugin framework, Hydra, database, service layer, mandatory Lightning, or MLflow.
- Allow a small helper file only when an existing module becomes difficult to read. Do not duplicate upstream code to avoid one sensible dependency.
- Keep original license notices and any required license files for copied source.

## Shared scientific and interface contract

**Task.** Default 72 observed hourly states → 24 forecast hours; 168-hour history is configurable. Inputs include available demand, operating measurements, topology, calendar variables, and known future schedules/costs. Never use future solved states as model inputs. The MVP solves hourly snapshot AC-OPF with fixed commitment, not unit commitment or ramp-constrained multi-period dispatch.

**Uncertainty.** Sample coherent trajectories jointly across hours, buses, and free continuous variables. Use low-rank Gaussian coordinates with positive diagonal variance, followed by appropriate bound transformations. Supply discrete outage/repair paths separately. Compute quantiles from complete samples; input quantiles do not automatically become corresponding output quantiles.

**Predicted quantities.** Forecast demand and feasible-range AC-OPF variables. Derive quantities fixed by other variables, including the initial fixed-power-factor Qd construction and branch flows. Inspect GENCO's treatment of Qg and multiple generators at one bus before choosing the exact output schema.

**Plain interfaces.** Agree exact signatures after inspecting assets, keeping these responsibilities:

| Interface | Contract |
|---|---|
| load_snapshot / load_window | Stable component IDs, times, feature schema, units, masks, topology, and targets. |
| GENCO.forward(graph) | Original snapshot behavior and checkpoint compatibility. |
| Foresight.forecast(history, known_future, outage_paths, n_samples) | Joint demand/state trajectories, scenario IDs/weights, and component IDs. |
| solver.solve(graph, initial_state=None, budget=...) | Numerical status, verified physical residuals, solution when available, objective, valid duals, runtime, and iterations. |
| monitor.score(predictions, solver_records) | Named physical/numerical events, calibrated estimates where supported, and verification status. |

Use a dictionary of tensors and a small metadata dictionary, not a hierarchy of schema classes. Missing labels have masks; solver failure never produces zero-valued target substitutes.

**Run reproducibility.** Save source/checkpoint revisions, preprocessing schema, data/split IDs, seed, device/package versions, solver settings, and configuration. Resume generation/training from completed artifacts. Never silently fall back to different data or a different device for a reported experiment.

**Evaluation separation.** Keep forecast error against tomorrow's realized solution separate from GENCO approximation error on an identical input graph. Keep raw model predictions separate from solver-refined results. Nonconvergence is not a certificate of infeasibility.

## Instructions applying to all four stage prompts

1. Read this entire plan and applicable AGENTS.md files. Inspect the repository without discarding user changes.
2. **Before implementation, ask the user the stage's asset question and wait for the answer.** This is an explicit user requirement. Present known paths as defaults; ask for confirmation/corrections and missing assets, not a repetition of the inventory.
3. Inspect confirmed code, sample data, and existing tests. Report briefly: reusable assets, missing assets, and the smallest implementation needed. Do not report user-described assets as inspected.
4. Implement only the requested stage and its necessary tests. Reuse or adapt code; avoid re-downloading, re-curating, or importing the entire lumina-sdk.
5. Run the stage checks. Record passed, failed, skipped, and unverified checks separately. A skipped accelerator or compatibility test is not a pass.
6. Update README with actual runnable commands and this plan's progress checkboxes. Report changed files, test evidence, run location, and remaining blockers. Stop before the next stage.

Each stage prompt below can be used verbatim with PLAN.md available in the repository.

---

## Stage 1 prompt — Supported functionality, setup, and contracts

**Instruction to coding agent:** Implement Stage 1 of PLAN.md only, following the shared instructions.

**Ask first:** “Please confirm the local genco-foresight clone path, the listed Aurora data/solver paths, and the proposed writable data/assets locations. Where are any cached GENCO source, weights, configuration, normalizer, or prior demo code? Which environment and compute allocation can I use for checks?”

### Implement

- Inspect README, repository state, loader documentation, one representative graph, and solver/model interfaces. Record reusable files and unresolved compatibility questions in README; keep this inventory short.
- Create config.yaml with paths, case, history/horizon, seed, device, sampling size, loss settings, solver budget, and event tolerances. Do not invent canonical case filenames. Engineering starting values: case30, history 72, horizon 24, latent memory 32, covariance rank 8. Tune sample counts and batches after measuring memory.
- Document the shared graph and prediction contracts, especially component ordering, baseMVA, normalized versus physical units, angle reference, topology changes, and generator-to-bus mapping.
- Add README installation toggles using HTML details/summary for Aurora and NVIDIA. Keep hardware-specific PyTorch installation separate from common requirements.
- Preserve the user's Aurora environment: Python 3.12.12, vendor torch 2.13.0a0+gitcf30153, PyG 2.8.0.post1, NumPy 2.3.5, SciPy 1.18.0, pandas 3.0.5, PyYAML 6.0.3, huggingface-hub 1.16.1. These are reported versions, not universal package pins.
- Existing executable: /lus/flare/projects/FRAME-IDP/siebenschuh/envs/reason_venv_2026.1.0/bin/python. Do not replace vendor Torch to satisfy an upstream reproduction environment. The login-node report did not test XPU execution.
- Keep common dependencies minimal: existing Torch/PyG stack, numerical libraries, YAML, checkpoint download support, pytest, and verified solver requirements. Add Plotly/Streamlit only when implementing visualization. Graphkit, Datakit, torch-scatter, Lightning, and MLflow are not automatically required.
- Create only the needed portion of training/common.py: configuration, explicit device selection, and external run creation. Add .gitignore and attribution scaffolding with actual source information.

### Acceptance gate

- [ ] Configuration resolves existing read-only inputs and writable external outputs; no automatic writes into OPFData or the checkout.
- [ ] A CPU tensor/PyG smoke check passes; available accelerator executes a tiny forward/backward operation.
- [ ] XPU availability is checked on an allocated compute node. If unavailable, report that gate as pending.
- [ ] Missing assets produce an actionable message, not a silent download or substitute.
- [ ] README describes the five experimental modes and their common input/output contract.

**Deliverable:** A usable minimal scaffold and inspected asset inventory. No forecasting performance claim and no placeholder implementation of later stages.

---

## Stage 2 prompt — Data construction and numerical reference

**Instruction to coding agent:** Implement Stage 2 only after reviewing Stage 1 results.

**Ask first:** “Please confirm access to the canonical OPFData loader/splits and the standalone or integrated solver. Are ChroniX2Grid/RTS-GMLC inputs, synthetic episodes, solved labels, or warm-start examples already available? What pilot solver budget and writable data location should I use?”

### Implement

1. **Existing-data loader — data/loader/dataset.py.** Adapt the SDK reader after inspection. Reuse canonical static splits and nested N1 subsets exactly. Preserve IDs, units, feature layout, and target masks. Keep T1 and N1 availability explicit; case2000 currently has reported N1 data only.
2. **Acquisition — data/download/prepare.py.** Check configured paths/cache first. Explicit download mode fetches only missing assets at recorded revisions and verifies integrity where available. Include reproducible acquisition instructions/scripts for a machine without Lustre; gated sources must fail with instructions. Do not embed datasets or weights in Git.
3. **Temporal generator — data/synthetic/generate.py.** Follow the published ChroniX2Grid injection-generation recipe: seasonal/weekly load profiles plus spatially and temporally correlated variation. Use its implementation where practical; otherwise label the implementation as an inspired approximation and document differences. Record synthetic geographic assumptions. Its dispatch/load-flow outputs do not replace independently checked AC-OPF labels.
4. Begin with fixed topology, case30, fixed benchmark costs, and roughly 10,000 hourly snapshots across multiple independent episodes. Choose episode duration/count before generation so temporal splits contain enough 72→24 windows. Derive Qd from each load's baseline power factor, with explicit handling of zero baseline Pd.
5. Add an optional bounded N1 regime: state-dependent failure hazard and repair durations, with at most one component unavailable at a time. Evaluate intensity from information available at that time; do not use future solved labels. Borrow documented RTS-GMLC assumptions where suitable, recording component mappings. Keep ordinary and intensified-stress episodes separate.
6. Split complete episodes or source time blocks **before window extraction**. Separate training, model-selection validation, risk-calibration validation, and test. Fit all transforms on training only. Never concatenate random OPFData snapshots and call the result temporal data. If using static templates, do not draw training templates from canonical held-out static splits.
7. **Solver adapter — solver/adapter.py.** Inspect supported inputs, warm starts, tolerances, duals, and return fields. Pin an installable standalone solver where possible; otherwise bring over the smallest licensed subset. No runtime sys.path trick or import from distillation.
8. **Checks — solver/checks.py.** Independently check P/Q balance, voltage and generation limits, and thermal limits using actual topology/units. Inspect bus-aggregated versus generator-level Qg. Store solutions, flows, objective, valid duals, status, residuals, iterations, and runtime. Retain unresolved/failed snapshots with missing-label masks.
9. Prices initially come from valid AC-OPF active-balance duals under the benchmark cost curves. Verify sign, scaling, and units with a small demand-perturbation objective check. These are conditional modeled marginal prices, not forecasts of an actual market's full clearing process. Defer variable offer curves to a documented extension using Chatzos et al.
10. Save compact sharded tensors plus one manifest. Generation must resume without overwriting completed records. Start with a small end-to-end batch before spending the full pilot budget.

### Acceptance gate

- [ ] Existing static loader reproduces canonical sample IDs/splits without modifying source data.
- [ ] Temporal windows cannot cross splits; train-only normalization and reproducibility checks pass.
- [ ] Load/outage paths are correlated, reproducible, correctly bounded, and respect N1 availability/repair.
- [ ] Sampled graphs round-trip through the solver mapping without changing costs, topology, IDs, or units.
- [ ] At least one valid solution passes independent checks; controlled failed/unresolved records retain masks.
- [ ] A verified warm start is exercised if the backend supports it; unsupported functionality is reported explicitly.
- [ ] Dual-price units/sign are checked; absent or unreliable duals remain unavailable.
- [ ] A pilot manifest and a short data-quality report are written outside the repository.

**Tests:** test_data.py and test_solver.py. Use existing solver tests when applicable. Tiny fixtures belong in tests; large artifacts remain external.

**Deliverable:** Reusable static data, chronological windows, and numerical reference records. No inferred GENCO-error labels yet: those depend on a specific checkpoint.

---

## Stage 3 prompt — GENCO architecture and adaptation

**Instruction to coding agent:** Implement Stage 3 only after reviewing Stage 2 artifacts and unresolved gates.

**Ask first:** “Where are the exact GENCO checkpoint, matching model configuration/normalizer, and source revision? Is an official working inference environment or golden-output fixture available? Which solver/physics routines can be reused, and what GPU/XPU hours are available for the two training variants?”

### Implement

1. **Faithful core — models/genco.py.** Extract the original heterogeneous GNN and the minimum physics/normalization routines needed for inference, with attribution. Keep matching parameter names and strict checkpoint loading. Do not recreate the network from the paper alone. Audit scatter operations and, where needed for XPU portability, use tested native Torch equivalents.
2. Establish snapshot parity before adding temporal behavior: same graph, preprocessing, checkpoint, eval mode, and agreed floating-point tolerance. Original snapshot forward remains the default. Expose final bus/generator embeddings and an optional temporal-conditioning hook without changing the unconditioned path.
3. **Temporal extension — models/foresight.py.** Encode each observed snapshot with GENCO. Add a small projection for observed operating variables masked by the original OPF task. Pass the sequence through shared per-node-type GRUs, initially memory width 32, retaining stable component identities.
4. For each future hour, combine memory, horizon/calendar features, known schedules/costs, and the supplied outage path. An internal demand head predicts demand coordinates; construct the corresponding future graph. Project temporal memory into GENCO embeddings and use its existing OPF processing/readouts for the state means.
5. Add a shared low-rank covariance head over free demand/state coordinates across all forecast hours. Initially rank 8 plus a positive diagonal; common random factors connect buses, variables, and times. Use reparameterized sampling. Preserve original mean-head bounds through compatible transforms; do not independently clip unrelated quantiles.
6. Sample only independent coordinates. Fix the angle reference, omit constants/offline variables, and derive Qd, Qg where supported, and flows **for each sample using its own demand and topology**. Resolve multi-generator Qg allocation from the inspected schema. Derived values can violate limits: measure those violations instead of hiding them. Return complete joint samples and scenario weights.
7. Use normalized joint energy score as the primary probabilistic loss, evaluated on forecast samples and available targets. State the coordinate scaling. Add modest physical residual penalties and canonical training-split OPFData replay; tune weights on validation. Monitor whether regularization damages spread or calibration. Missing labels must not turn failed examples into fictitious solutions.
8. Implement exactly two initial learned modes with the same architecture/data:
   - **A — adapter_only:** freeze original GENCO parameters; train temporal projections, GRUs, demand and uncertainty heads.
   - **F — full_adaptation:** initialize identically, train those additions and the backbone with a smaller backbone learning rate.
   Share one loop; report trainable parameters, peak memory, and actual compute. Frozen weights can still require activation/backpropagation memory.
9. Use the numerical solver as teacher/verifier. Do not add RL or assume differentiability through an iterative sparse solver. A KKT loss requires valid duals and stationarity/complementarity terms; balance residual alone is not KKT.
10. Save checkpoint metadata, schema, transforms, and source revision. A may save an adapter plus a pinned base reference; F needs the updated backbone. Exported inference must work without the training package or lumina-sdk.

### Acceptance gate

- [ ] Strict official checkpoint loading and snapshot parity pass, including normalization and physical outputs.
- [ ] Tiny-batch training reduces loss; samples have finite gradients, coherent covariance, valid masks, and nonzero spread.
- [ ] A leaves every original parameter unchanged; F updates the intended parameters.
- [ ] No future targets enter inference. Sample-dependent load/topology reaches the physics calculation.
- [ ] Serialization/reload preserves predictions with a fixed seed.
- [ ] Actual CPU and available XPU/CUDA forward/backward smoke checks pass; no unsupported scatter assumption.
- [ ] One case30 pilot per learned mode completes with comparable recorded budgets. F is separately evaluated for snapshot regression.

**Tests:** test_model.py plus relevant tests in test_smoke.py. An unavailable official parity reference is a compatibility blocker, not permission to claim parity.

**Deliverable:** A reusable pretrained GENCO core, a temporal forecasting module, and reproducible A/F checkpoints. Results determine whether post-training helps; architectural novelty alone does not establish it.

---

## Stage 4 prompt — Baselines, evaluation, risk monitor, and demo

**Instruction to coding agent:** Implement Stage 4 only after reviewing Stage 3 checkpoints and evaluation contracts.

**Ask first:** “Which frozen GENCO and A/F checkpoints should be compared? Are baseline forecasts, calibration labels, solver difficulty records, plotting code, or grid coordinates already available? Please confirm event thresholds, numerical budgets, and the demo hardware before running comparisons.”

### Implement the comparison

- **Shared external forecaster — training/forecaster.py.** Use a small seasonal autoregression/ridge baseline with calendar features and available lags such as 1, 2, 24, and 48 hours. Fit on training data. Estimate joint residual uncertainty using whole 24-hour residual blocks from training forecast origins, preserving spatial/temporal dependence. This is the inexpensive ARIMA-style comparator.
- **P — frozen_point:** feed its mean demand trajectory into frozen GENCO, once per hour, with the known/modal outage path. Report deterministic margins, not fabricated probabilities.
- **Q — frozen_interpolation:** evaluate frozen GENCO at representative nodes in joint forecast/scenario space, then interpolate its response at joint forecast draws. Keep scenario weights and discrete topology strata. Use a simple documented interpolation rule, validate approximation error against S, and fall back to direct evaluation where support is inadequate. Never interpolate across incompatible topologies or equate marginal input/output quantiles.
- **S — frozen_sampling:** run frozen GENCO on joint external-forecast draws and retain their empirical output distribution.
- Reuse A/F inference. All modes share the forecast cutoff, available information, evaluation episodes, and scenario assumptions. Start the five-way comparison with fixed topology; then enable the N1 regime and explicitly account for deterministic P's limited uncertainty representation.
- In evaluate.py, report demand/state point error; energy score, marginal CRPS/coverage, interval width and selected joint-event calibration; physical residuals/violation rates; cost; and runtime/memory. State tolerances before test evaluation and report per-variable metrics in physical units.
- Measure GENCO approximation error by comparing outputs with numerical references for the **same sampled inputs**. Do not confuse that error with failure to forecast the realized demand.
- Budget one small case30 ablation removing the added physics/replay regularization together to test its incremental contribution. It supports a combined regularization claim, not separate attribution. Broader claims about pretrained representations require additional controls.
- Use one seed for debugging. Before a strong comparative claim, run repeated seeds and episode-level uncertainty estimates within the agreed compute budget. Report negative results.

### Implement the risk monitor

Keep named quantities separate; no opaque combined safety score.

| Quantity | Definition / evidence |
|---|---|
| Thermal stress | P(line loading > 95% of its valid configured rating). |
| Voltage stress | P(voltage lies within the outer 5% of its allowed range, or beyond a bound). |
| Reactive headroom | Analogous Qg headroom event; exclude offline/fixed variables and respect generator mapping. |
| GENCO physical reliability | Probability of configured physical residual/limit exceedance from samples. |
| GENCO approximation reliability | Separately calibrated probability of reference-error threshold exceedance on the same input. |
| Numerical difficulty | Calibrated probability a specified backend/init/tolerance does not resolve within a fixed budget. |
| Modeled prices | Weighted solver-dual quantiles and configured spike probability, where duals are valid. |

- Freeze each forecasting checkpoint before fitting its error/difficulty models. Use small classifiers or regression models; fit on training episodes and calibrate on the reserved validation partition. Use numerical outcome labels with the exact backend/settings.
- For rare events, display insufficient evidence when calibration cannot be supported. Do not equate stress with a violation, blackout, or proven infeasibility.
- Compute daily “any physical stress” probability from full joint trajectories. Do not convert marginal classifier probabilities into daily probability by assuming independence; display peak hourly risk instead.
- Select diverse, consequential scenarios for a fixed solver budget, reserving representative audit samples. Save checked candidate warm starts and try/refine them on the actual subsequently realized input. Report checks and fallbacks.
- Do not estimate unconditional price/risk distributions from an unweighted, deliberately stress-selected verification subset. Use representative samples or justified sampling weights.
- Compare cold starts, ordinary warm starts where supported, and anticipatory warm starts. Report online latency and total cost including advance solves, as well as success rates and event recall at the same solver budget. No assumed speedup on case30.

### Implement the visualization

Use a small Streamlit app with Plotly, reading saved outputs. Separate **what is calculated** in monitor.py from **how it is shown** here.

- Fixed schematic grid layout unless actual coordinates exist. Edges show thermal-stress probability; buses show voltage-stress probability; a separate view shows generator reactive headroom.
- Hour/mode selectors, a three-row 24-hour timeline, and selected-hour cards for operating stress, GENCO reliability, and numerical difficulty.
- Component drill-down: forecast bands, limits, event probability, verification evidence, and available modeled prices.
- Coherent individual-scenario playback with outages distinguished; keep it separate from aggregate probability maps.
- Solver queue/results and candidate warm-start availability. Label raw versus refined predictions and checked versus unchecked scenarios.
- Show data provenance, modeled assumptions, and calibration availability briefly. A forecast distribution is not fully verified merely because a few selected scenarios were solved.

### Acceptance gate

- [ ] P/Q/S/A/F write the same documented prediction format and evaluate on identical episode IDs.
- [ ] Synthetic fixtures verify joint sampling, scenario weighting, Q interpolation behavior, and no cross-topology interpolation.
- [ ] Calibration has no train/test leakage; risk and price estimates do not inherit selection bias from the solver queue.
- [ ] An end-to-end case30 demo loads history, forecasts, displays stress, checks selected scenarios, and reports valid numerical outcomes.
- [ ] Physics, reliability, and solver-failure metrics remain distinct; missing evidence is visible.
- [ ] Compute comparisons include preparation and advance solver work.
- [ ] A clean checkout can reproduce a small demonstration with documented assets/commands.
- [ ] Only after case30 passes: validate 57, 118, 500, then 2000 with case-specific checkpoints/data support. Unsupported case/checkpoint combinations remain explicitly unsupported.

**Tests:** test_monitor.py and test_smoke.py, supplemented by earlier model/data/solver checks. Test meaningful scientific failure cases, not every UI detail.

**Deliverable:** Reproducible comparison, compact risk monitor, and a working demo. Record which claims are supported and which remain hypotheses.

## Stage completion record

- [ ] Stage 1: inspected assets and working setup.
- [ ] Stage 2: temporal data and verified numerical reference.
- [ ] Stage 3: compatible GENCO core and trained A/F variants.
- [ ] Stage 4: five-mode comparison, risk monitor, visualization.

No stage is complete merely because its files exist. Record actual test commands, results, and remaining platform limitations in README or the external run report.

## Sources and implementation anchors

Inspect the exact versions used; branch names and documentation can change.

- [GENCO paper](https://arxiv.org/abs/2608.09921)
- [GENCO reproduction instructions](https://github.com/gridfm/gridfm-paper-genco/blob/main/repro_instructions/5.4.3_OPF_on_datakit.md)
- [GENCO source branch](https://github.com/gridfm/gridfm-graphkit/tree/genco-paper-repro)
- [GENCO OPFData checkpoints](https://huggingface.co/gridfm/genco-opf-datakit)
- [ChroniX2Grid paper](https://zenodo.org/records/15184237), [code](https://github.com/Grid2op/chronix2grid), and [recipe documentation](https://chronix2grid.readthedocs.io/en/latest/DESCRIPTION.html)
- [Chatzos et al.: Data-driven time series reconstruction for modern power systems research](https://arxiv.org/abs/2110.13772)
- [RTS-GMLC reliability and system data](https://github.com/GridMod/RTS-GMLC)
- [Gneiting and Raftery: proper scoring rules and energy score](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf)
- [ALCF Aurora PyTorch](https://docs.alcf.anl.gov/aurora/data-science/frameworks/pytorch/) and [PyG guidance](https://docs.alcf.anl.gov/aurora/data-science/frameworks/pyg/)

