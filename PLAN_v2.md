# GENCO Foresight — Challenge implementation contract

**Status: proposed for one review; implementation has not started.** This document supersedes `PLAN.md` after that review. Preserve `PLAN.md` as historical context. The Challenge Track is a 72-hour delivery plan for 23–25 October 2026; the Research Track appendix preserves the subsequent four-stage program. Do not start Stage A until the user has reviewed this document. Thereafter, record evidence at each gate and continue within the approved scope; the old requirement to ask the entire asset questionnaire before every stage is superseded. Ask only for missing information that changes the next 24 hours of work.

## 1. Thesis and submission position

GENCO Foresight investigates whether a pretrained spatial AC-OPF solver can become a useful probabilistic day-ahead forecaster. On fixed-topology case30, it uses 72 observed hours to predict 24 hours of joint demand and operating states, tests the calibration of operating-stress probabilities, and uses selected forecast scenarios to prepare numerical-solver initializations. The contribution is measured temporal extension with an auditable uncertainty evaluation. Calibration, approximation quality, and solver savings are hypotheses to test, not promised results. Every deliverable must support this experiment or make its evidence reproducible.

**One-sentence story:** “GENCO is a snapshot solver; we post-train it into a forecaster of tomorrow's joint demand + AC-OPF operating state distribution, with calibrated operating-stress probabilities and anticipatory warm starts for the numerical solver.”

Use that sentence as the intended capability. In the submitted post, qualify “calibrated” and any warm-start benefit by the actual held-out evidence; replace them with “calibration evaluated” or “warm starts tested” if the results require it.

The following entries were read on 23 October UTC. Their results are entrant-reported, not independently reproduced here. Our specific distinction is **learned 72→24-hour joint forecasting and its calibration**; do not claim that other entries never consider future uncertainty or temporal decisions.

| Existing entry | Positioning decision |
|---|---|
| [#7 Trust, but verify](https://huggingface.co/spaces/gridfm/genco-challenge/discussions/7), IREC | Match its pinned artifacts, frozen protocol, numerical checks, and recomputable headline evidence. Its Texas contingency verification and embedding warning are not our headline. |
| [#6 GENCO TO THE MOON](https://huggingface.co/spaces/gridfm/genco-challenge/discussions/6), UIUC | Already develops embedding-based prices and market/outage decisions with ERCOT profiles. No price head; solver-dual prices remain secondary and conditional. |
| [#3 Inside GENCO](https://huggingface.co/spaces/gridfm/genco-challenge/discussions/3) | Correction-step interpretation and parameter sharing are covered. Our inexpensive probe tests future-state information instead. |
| [#2 GENCO-PTDF](https://huggingface.co/spaces/gridfm/genco-challenge/discussions/2) | Retain negative findings and numerical audits as it does; do not add derivative fine-tuning. |
| [#5 Battery control](https://huggingface.co/spaces/gridfm/genco-challenge/discussions/5) | It already uses sequential planning and stochastic profiles. Our experiment learns a forecast distribution rather than a battery policy. |
| [#1 Organizer example](https://huggingface.co/spaces/gridfm/genco-challenge/discussions/1) | Follow its readable evidence-first format; contingency screening is outside this submission. |

## 2. Rules, deadline, and submission checklist

The [organizer page](https://huggingface.co/spaces/gridfm/genco-challenge), including its [published HTML](https://huggingface.co/spaces/gridfm/genco-challenge/blob/main/index.html), was inspected. It specifies **25 October 2026, end of day AoE**, with winners announced 30 October at 17:15 WET in Porto. There is no pitch or Q&A: the Community post must stand alone. Entries must use GENCO and OpenGridFM; this project will demonstrate concrete use of **both graphkit and datakit**. IBM-affiliated individuals are ineligible.

**Calendar correction:** 25 October 2026 is **Sunday**, not Saturday. The deadline is Sunday 25 October 23:59 AoE (UTC−12), equivalently **Monday 26 October 11:59 UTC / 06:59 CDT**. All schedule times below use `America/Chicago`, which is UTC−5 on these dates. The submission target remains **Sunday 25 October, 18:00 CT**, leaving approximately 13 hours of contingency margin. Do not schedule planned work in that margin. The planning session is Thursday 22 October late evening CT / Friday 23 October UTC; “72 hours” is an upper planning envelope, not extra time beyond the Sunday target.

The five judging dimensions are originality/impact, technical quality, exploration/extension, communication/narrative, and open science/reproducibility. Treat communication as equally important in our allocation; the organizer does not publish numeric weights. Reserve **18 hours, 25% of the nominal 72 hours**, for evidence packaging, figures, and writing.

- [ ] One new discussion in the [Community tab](https://huggingface.co/spaces/gridfm/genco-challenge/discussions), with title and short, self-contained write-up.
- [ ] Five core figures plus the embedding-probe figure, with legible captions and denominators.
- [ ] Public code/tag, model, and dataset links, each checked without authentication.
- [ ] Contact email and every team member's name and affiliation; user supplies these before publication.
- [ ] Eligibility checked by the submitting team; do not invent affiliations or contact details.
- [ ] Explicit GENCO, graphkit, and datakit roles, pinned revisions, limitations, and reproduction commands.
- [ ] Every numerical claim linked to a saved result and recomputation path; no unmeasured headline or assumed speedup.
- [ ] Post submitted by Sunday 18:00 CT; submission URL recorded.

This planning task writes only this contract. It does not authorize publication or start any experiment. The user is the submission owner; prepare concrete artifacts before the later publication step.

## 3. Corrected asset inventory and evidence status

**Inspected** means the listed text, source, or remote metadata was read during this revision; it does not imply a runtime check. **Reported** means supplied by the user or old plan and still awaiting local inspection. No weights, binary splits, local solver, Lustre data, or accelerator have been executed or validated in this planning task.

| Asset | Location / selected identity | Evidence and next action |
|---|---|---|
| Checkout | `/home/siebenschuh/Projects/genco-foresight`; `https://github.com/7shoe/genco-foresight` | **Inspected:** `README.md`, full `PLAN.md`, clean initial Git status. Initially only these two tracked documents; no applicable `AGENTS.md` found in the checkout/ancestor locations checked. |
| OPFData | `/lus/flare/projects/FRAME-IDP/siebenschuh/acopf_datasets/` | **Reported:** read-only; inspect one representative graph and metadata in Stage A. |
| Local static regimes | `processed/T1/`, `processed/N1/` beneath that root | **Reported:** T1 cases 14/30/57/118/500; N1 also 2000. Use case30 only now. No curated N−2; N1 split is not unseen-outage evaluation. |
| Local split/subsets | `processed/N1/splits/<canonical-case>/` | **Reported:** 240K/30K/30K seed-42 split and nested 100/1K/10K subsets. Discover filenames and canonical IDs; never regenerate or overwrite. |
| Downloadable static data | [`gridfm/opfdata_case30_ieee`](https://huggingface.co/datasets/gridfm/opfdata_case30_ieee) | **Inspected:** official reproduction guide identifies this converted OPFData source. Resolve its dataset revision/hash in Stage A; reuse local matching data first. |
| Existing reader | `/home/siebenschuh/Projects/lumina-sdk/distillation/baselines/n1/curated_data.py` | **Reported:** adapt the minimum licensed code; no runtime dependency on the SDK. |
| Reader instructions | `/home/siebenschuh/Projects/lumina-sdk/distillation/baselines/artifacts/n1_datasets_20261022/README.md` | **Reported:** read before adapting. |
| Standalone numerical solver | `/home/siebenschuh/Projects/acopf` | **Reported:** preferred AC-OPF/Ipopt backend; inspect install/API/license, warm-start support, and duals. |
| Integrated fallback | `/home/siebenschuh/Projects/lumina-sdk/distillation/solver/` | **Reported:** inspect `README.md`, `backends/reference.py`, `sparse_solver/solver.py`; extract only a necessary licensed subset if standalone installation fails. |
| Optional independent reference | `/home/siebenschuh/Projects/lumina-sdk/distillation/RAMBO/julia/model.jl` | **Reported:** PowerModels/Ipopt; defer a second solver dependency to Research Track. |
| GENCO checkpoint | [`gridfm/genco-opfdata`](https://huggingface.co/gridfm/genco-opfdata), revision `74e19e1e210a175aa17023f4db161f0ae9f753cb`, `case30_ieee/seed42/` | **Inspected:** card/API inventory/YAML. Download only `best_model_state_dict.pt`, `normalizer_stats.pt`, `HGNSQ_penalty_11_OPFData_case30_default.yaml`, and official splits after checking caches. Binary integrity/loading pending. This replaces `genco-opf-datakit`. |
| Checkpoint architecture | GENCO Base, hidden size 48, 8 attention heads, 12 correction steps; approximately 20.1M parameters | **Inspected:** YAML dimensions; **reported:** approximate parameter count. Measure exact base/trainable counts at load. |
| Official OPFData split | Same checkpoint revision, `splits/train.pt`, `val.pt`, `test.pt`, `shuffled_indices.pt` | **Inspected:** filenames, YAML's 5% validation/5% test settings, and official guide's 270K/15K/15K counts; binary membership check pending. Use official/local test intersection for parity and snapshot regression. |
| Normalizer | Checkpoint's `normalizer_stats.pt`; `HeteroDataMVANormalizer`, baseMVA 100 in YAML | **Inspected:** YAML; **reported:** `fit_on_train` provenance. Always restore shipped statistics, never refit the GENCO normalizer. Separate newly learned feature/score scaling uses temporal training data only. |
| Graphkit source | [`main` commit `f104900245cba26d1380037eea562cf633a97484`](https://github.com/gridfm/gridfm-graphkit/tree/f104900245cba26d1380037eea562cf633a97484) | **Inspected:** full SHA, source, metadata. Use this ONE commit for installation, parity, extraction, training, and release. The model card reports agreement with the paper branch at this commit; local parity remains mandatory. |
| Graphkit dependency range | Pinned commit's [`pyproject.toml`](https://github.com/gridfm/gridfm-graphkit/blob/f104900245cba26d1380037eea562cf633a97484/pyproject.toml) | **Inspected:** fetched source declares `torch>=2.10,<2.14`, `torch-scatter>=2.1.2`, `gridfm-datakit==1.1.0`. This differs from the user's reported `main` upper bound `<2.13`; record installed metadata in Stage A. Never use the stale paper-branch `<2.9` requirement or replace vendor Torch. |
| Existing embedding API | `gridfm_graphkit/models/gnn_heterogeneous_gns.py`, `GNS_heterogeneous.forward(batch, return_embeddings=True)` | **Inspected:** returns predictions and final bus/gen head-input latents; one `scatter_add` import and one call site. Reuse it and upstream `predict --get_embeddings`; do not implement another extractor. |
| Datakit and profile | Pin `gridfm-datakit==1.1.0` with package/source hash; ERCOT 2025 hourly EIA profile via datakit | **Inspected:** graphkit dependency and [paper Eq. (1)](https://arxiv.org/html/2608.09921v2#S3.SS1.SSS2); **reported/uninspected:** cached profile, exact datakit functions, package hash. Resolve these in Stage A. |
| case30 generator mapping | One generator per generator bus | **Reported:** assert against the selected network in Stage A; defer general multi-generator Qg allocation. |
| Aurora environment | `/lus/flare/projects/FRAME-IDP/siebenschuh/envs/reason_venv_2026.1.0/bin/python` | **Reported:** Python 3.12.12, torch 2.13.0a0+gitcf30153, PyG 2.8.0.post1, NumPy 2.3.5, SciPy 1.18.0, pandas 3.0.5, PyYAML 6.0.3, huggingface-hub 1.16.1. Preserve; not universal dependency pins. |
| External outputs | `/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/{runs,data,assets}` | **Reported:** runs root fixed; data/assets proposed previously. Check writable status at Stage A; on CPU/CUDA without Lustre use explicit external equivalents. No repository-local runtime directories. |

The matching source and weights are pinned now, subject to successful inspection/loading at Stage A. Any necessary revision change must be recorded and parity rerun; never mix checkpoints/configs/normalizers from different revisions silently. `--config_conversion auto_inline` is the official `main` evaluation route. Preserve the input YAML and resulting configuration in each run.

## 4. Challenge Track: Stages A–D

The scope is **case30, fixed topology, benchmark costs, fixed commitment, hourly snapshot AC-OPF, 72→24 hours**. No ramps or unit commitment. An unmet scientific gate stays failed/pending; reducing scope cannot turn it into a pass. All gates record passed, failed, skipped, and unverified checks separately, with commands, elapsed time, and artifact paths.

### Stage A — pinned setup, temporal data, numerical reference

**Time-box:** Friday 08:00–18:00 CT. Setup/parity target 11:00; first 200-hour shard target 13:00; full generation runs while baseline/scoring work proceeds. Datakit bring-up and optional XPU each receive at most two hours, not serial extensions of the day.

#### A1. Environment and compatibility

1. Inspect confirmed local assets, solver API/license, and a sample graph. Report reusable/missing assets briefly. Create `config.yaml`, explicit device/output paths, minimal common requirements, installation instructions, and third-party notices only as needed. Check caches before downloading. Explicit download mode pins revisions and records SHA256; unavailable assets fail with actionable instructions.
2. Default to CPU or an already available NVIDIA machine. Install the selected graphkit source with `pip install --no-deps`, especially on Aurora. Do not launch an unrestricted resolver against vendor Torch. The older reported `<2.13` range would exclude a 2.13 prerelease; the fetched `<2.14` metadata does not establish runtime compatibility either. Save `pip freeze` before/after and its diff; install only verified missing dependencies. Add Lightning/MLflow with `--no-deps` only when upstream imports/CLI require them, plus explicitly identified dependencies; they are not our training framework or service.
3. Provide a small, explicitly installed `torch_scatter` compatibility shim for the one needed `scatter_add(src, index, dim=0, dim_size=...)` operation using device/dtype-preserving zeros and `index_add_` (approximately five implementation lines plus packaging/tests). Restrict and test the supported signature; do not pretend to supply every torch-scatter operator. Audit PyG's optional-extension detection so the shim does not falsely enable unsupported compiled kernels. Verify both repeated-index sums and gradients, batching offsets, and isolated buses.
4. Verify PyG 2.8 TransformerConv forward/backward uses available native scatter/scatter-reduce paths without a compiled torch-scatter dependency. CPU first; XPU only on an allocated compute node. A login-node availability query is not accelerator evidence. No device switch without a recorded configuration change.
5. Strictly load case30 seed42 weights/config/normalizer. Map canonical IDs and run upstream `gridfm_graphkit evaluate --config_conversion auto_inline` on the **intersection of local test IDs and official test IDs**; compare per-snapshot physical outputs and metrics on that same subset. Match precision, masks, ordering, and preprocessing. Keep original split files intact; use a derived evaluation manifest outside source data. Published full-test metrics cannot establish subset parity. Declare numerical tolerances before comparing.
6. Prefer importing upstream graphkit. Vendoring the four required inference/physics/normalization source files is a documented fallback only if the import path fails inside this setup time-box; inspect the actual dependency closure, preserve licenses, and still demonstrate toolkit use and parity. Do not assume four files are sufficient without inspection.

#### A2. Temporal construction

Generate **20 episodes × 21 days × 24 hours = 10,080 snapshots**. Assign episode IDs and splits before sampling windows:

| Partition | Episodes | Hours | Complete daily 72→24 origins |
|---|---:|---:|---:|
| Training | 12 | 6,048 | 216 |
| Model-selection validation | 2 | 1,008 | 36 |
| Risk-calibration validation | 2 | 1,008 | 36 |
| Test | 4 | 2,016 | **72** |

For a midnight-aligned 504-hour episode, forecast origins are hours 72, 96, …, 480: 18 per episode. Observations are `[origin−72, origin)` and targets `[origin, origin+24)`. Use an explicit fixed hourly time coordinate (UTC by default), save source timestamps and calendar convention, and handle source daylight-saving duplicates/gaps explicitly. Training may use additional within-episode origins if runtime allows; evaluation remains these 72 test days. Never bridge episodes or partitions.

Use datakit's sequential aggregate-profile mechanism: baseline bus load times hourly `ref_t` times local uniform variation. Add a documented per-bus AR(1) multiplicative factor, for example `u[i,t] = rho*u[i,t−1] + eta[i,t]`, with a positive factor `exp(u−Var(u)/2)`, independently initialized at stationarity per episode. Require `abs(rho)<1` and uniform amplitude `0<=sigma<1`; record innovation variance, profile normalization, any bounds, and seeds. The common profile induces spatial dependence; AR(1) adds temporal dependence beyond datakit's i.i.d. local perturbations. Specify any truncation as part of the synthetic distribution, not a hidden repair.

Use ERCOT 2025 via EIA as a **historical generator input**, not as known future information supplied to the forecaster. Record the geographic assumption: an aggregate Texas profile drives an IEEE case30 network with synthetic bus variation. This is not an ERCOT network or field-validated forecast. Determine scaling and generation settings using training/pilot evidence, then freeze them before evaluating held-out data; do not search test settings until stress events look favorable.

**Source-profile reuse:** 420 episode-days exceed the 365 days in the source year. Assign disjoint source-date pools to train/selection/calibration/test before drawing 21-day blocks, allow reuse only within a pool, and record every source block. Use independently seeded AR/uniform paths; call episodes independent stochastic replicates conditional on their source profiles, not independent real years. If this allocation cannot be satisfied, disclose the changed benchmark before running it. Do not silently reuse a test source block in training. Fit any profile rescaling from training pools or a fixed documented upstream convention.

Derive `Qd = Pd * Qd_base / Pd_base` where baseline active demand is nonzero, preserving signed reactive demand. At zero baseline Pd, hold Pd at zero; if Qd is also zero keep it zero, otherwise retain the baseline reactive-only load and flag the exception. Do not divide by zero, invent a power factor, or add these fixed coordinates to the covariance. Verify exactly one generator per generator bus in this case.

**Two-hour datakit fallback:** if its Julia/PowerModels generation pipeline is not working by the limit, implement the documented Eq. (1) load recipe plus the AR extension and use our existing Ipopt solver for labels. Retain actual datakit use through its supplied profile and, where available, network parsing/admittance utilities. Save exact functions/assets and package revision used. Citing a paper or installing an unused package is not toolkit use; if no substantive datakit component works, record that as an unmet submission requirement. ChroniX2Grid is not installed for this track.

#### A3. Labels and physical checks

Use the existing standalone AC-OPF solver with Ipopt, benchmark costs, and a pinned numerical configuration. Before the full run, generate **one resumable 200-hour shard end-to-end**, including graph conversion, demand, solver records, masks, and manifest. It belongs to the eventual assigned partition and is not counted twice. Estimate full-run wall time from this shard; use bounded CPU parallelism if supported. Preserve completed shards and deterministic IDs on resume.

Store status, valid solution/flows, objective, valid duals and their units, residuals, iterations, and runtime for every attempt. Independent checks cover P/Q balance, voltage/Pg/Qg bounds, both-end apparent-power thermal limits, branch angle-difference bounds when configured, shunts, taps, baseMVA, reference angle, and topology. Assert connected fixed topology; unsupported islands fail explicitly. Do not silently alter costs, network limits, or commitment to get a solution.

A solver status alone does not establish a valid label. Failed/unresolved snapshots retain masks and outcomes, never zero target substitutes. Demand remains observed even if its OPF label is missing. History operating-state gaps retain availability masks and use only a documented causal imputation if needed. A valid Ipopt point is a numerically checked reference, not a global optimality certificate.

Verify active-balance dual sign/scaling with a small demand perturbation and objective finite difference. For this check hold Qd fixed: perturbing P and Q together along fixed power factor measures a combination of active and reactive marginal effects. State the objective time/currency convention before labeling duals as modeled currency/MWh. Absent/unreliable duals stay unavailable.

**Acceptance gate, Friday 18:00:** pinned CPU snapshot parity; meaningful graphkit/datakit usage; 200-hour shard verified before expansion; 10,080 attempted snapshots with immutable split/ID manifest; disjoint windows/source pools; unchanged upstream normalizer; graph/solver round-trip; at least one independently valid solution; failure masks; dual check or explicit price unavailability; resume and checksum evidence. Tests concentrate on split leakage, mapping/units, physical violations, and failure records.

**Cut if late:** stop XPU and datakit-Julia debugging at their two-hour limits; use the specified CPU/recipe routes. Defer additional dataset regimes and secondary dual analysis if unsupported. Preserve the 20-episode target; if full labeling misses the budget, publish actual coverage and mark the gate incomplete rather than inventing 72 fully labeled test days.

### Stage B — P/S/A forecasts, attribution, and embedding probe

**Time-box:** Friday 13:00–Saturday 12:00 CT. P/S end-to-end results by Friday evening. A gets a measured training budget, initially at most six hours overnight/early Saturday; the attribution diagnostic and probe finish Saturday morning. No hyperparameter sweep.

#### B1. Baselines and shared scoring

- **P / frozen_point:** seasonal ridge autoregression with lags 1, 2, 24, 48 plus calendar features; mean demand trajectory through frozen GENCO. For multi-step forecasting, recursively use predicted values when a lag falls beyond the cutoff, or fit declared direct horizon regressions using only available lags. Never use future realized demand as a lag. P reports deterministic margins; probability metrics are N/A, not fabricated probabilities.
- **S / frozen_sampling:** the same ridge mean plus sampled complete 24-hour residual blocks across all buses. Obtain residuals from rolling-origin forecasts entirely within training episodes, using the same forecasting procedure as deployment; no fitted residuals. Retain whole blocks to preserve cross-bus/hour dependence. Declare any demand-bound transform and its effect. Apply frozen GENCO to every sampled graph; derive Qd and flows from that graph.
- All modes use identical forecast cutoffs, feature availability, episode IDs, state masks, sample-count conventions, and physical event thresholds. Provide the external baseline the same permissible observed information; document any architecture-specific representation use. Freeze model-selection choices before test evaluation; Friday P/S test results do not authorize later tuning on test outcomes.
- `scoring.py` is the sole implementation of metrics: physical-unit MAE/MSE; marginal sample CRPS; scaled joint energy score; variogram score `p=0.5`; 50%/90% coverage and interval width; finite-ensemble PIT/rank diagnostics; Brier scores and reliability diagrams for named events. Save raw per-origin/component results and counts, not just averages. Use a fixed train-derived coordinate scale for joint metrics and a documented variogram pair set (temporal pairs and connected-bus pairs); hold it constant across modes. Report per-variable physical-unit results alongside dimensionless aggregate scores. [Energy score](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf) and [variogram rationale](https://repository.library.noaa.gov/view/noaa/22327/) support complementary distribution/dependence checks.

#### B2. A / adapter_only

Use the original GENCO implementation and its existing `return_embeddings=True` output. Encode historical graphs in eval mode; separately project observed operating variables hidden by the OPF input mask. Shared GRUs per node type have width 32 and retain stable component identities. Add horizon/calendar information and fixed known costs/topology. No future solved states, future profile values, or future realized loads enter inference.

The temporal-conditioning subclass adds projected memory at the output of `input_proj_bus`/`input_proj_gen`. Use scoped module hooks or an equivalent minimal extension, then call the parent forward; the parent correction loop is not reimplemented. With conditioning absent, call the untouched parent directly, without even an add-zero operation: require bit-identical CPU eval outputs under identical deterministic settings. Clear hook/context state after every call; no cross-batch contamination.

**Sampling construction is fixed to per-sample GENCO evaluation.** The uncertainty head emits rank-8 factors plus strictly positive diagonal variance over free demand coordinates and state-innovation coordinates across the full 24-hour horizon. A common Gaussian factor vector and independent diagonal innovations produce reparameterized joint draws. For each draw:

1. Transform its demand coordinates into physical Pd; derive its Qd and build its own future `HeteroData` graph for each hour.
2. Run memory-conditioned frozen GENCO on those graphs. Its existing state readouts provide sample-specific centers, conditional on that draw's demand.
3. Apply the correlated state innovations to the final free pre-bound readout coordinates (`Vm`, reference-aligned `Va`, `Pg`) through a thin final-readout hook; use existing compatible bound transforms for Vm/Pg. Invoke the existing physics completion on that sample's demand, voltages, and topology. Do not sample Qd, Qg, or flows independently. The base parent path is unchanged when these hooks are disabled.
4. Derive Qg and branch flows for that sample. Remove constants, offline coordinates, and the reference angle from stochastic coordinates. One generator per generator bus resolves Qg for case30; assert this condition.

Thus the low-rank Gaussian is a distribution in driving coordinates; the final demand/state distribution is nonlinear and is **not** claimed to remain Gaussian or rank eight. Demand/state correlation alone does not enforce AC feasibility; record residuals and derived-variable violations. The “central trajectory plus state deviations without sample-specific GENCO” alternative is dropped. Compute reported predictive means/quantiles from complete samples; a transformed latent center need not be the predictive mean.

Loss is normalized joint energy score plus a modest power-balance penalty and canonical OPFData training replay, with scales/weights fixed using model-selection validation. Missing state labels select the scored subvector; all compared samples use the same observed-coordinate mask, and demand-only contributions remain possible. Record scored dimension/coverage. Use an unbiased/fair ensemble estimator where appropriate and fixed sample counts to avoid artificial spread incentives.

**Replay must have an effective gradient.** Use known-demand static OPFData training graphs through the adapter-conditioned state path with a declared null-history context, and penalize departure from their valid labels/physics; never present their solved state as input. Replay IDs come from local canonical training data, excluding official held-out IDs used for snapshot checks. A replay loss on an entirely bypassed, frozen backbone is constant: record it as a regression check rather than claiming regularization. Verify which adapter parameters the replay term reaches before the pilot.

Freeze all original GENCO parameters; train only temporal projections, observed-variable projection, GRUs, demand/uncertainty heads, and necessary conditioning parameters. Cache frozen historical embeddings once where useful. Future passes still need autograd through frozen layers to train inputs/conditioning. A small trainable parameter count is not evidence of cheap training: measure a representative minibatch within 30 minutes, extrapolate runtime, and reduce batch size, training samples per forecast, or epochs if needed. Preserve the task/horizon, inference construction, and calibration holdout. Save actual optimizer steps, sample counts, peak memory, and wall time.

#### B3. Attribution and exploration

- **Diagnostic D:** pass A's exact demand samples through unconditioned frozen GENCO, preserving IDs/weights. Compare D with A for forecast scores and physical residuals. On a fixed representative budget of sampled graphs, obtain numerical references and compare A and frozen GENCO on **identical inputs**. This distinguishes demand-forecast gains from adapted-state gains; it is not a fourth headline mode.
- **Embedding probe:** export final embeddings via upstream `gridfm_graphkit predict --get_embeddings`. Fit linear/ridge probes from hour t to realized state at t+1 and t+24, compared with the same probe family on raw available features. Split by episode, prevent target-crossing at episode ends, use train-only feature scaling and validation-only regularization selection, and give both probes the same calendar/observed information. Match masks/metrics and report feature dimensions. One figure; a negative result is useful. Predictive association is not causal evidence or proof of a general foundation representation.
- Report exact trainable/base counts, unchanged base-parameter and buffer checksums, and A versus frozen same-input approximation error. The unconditioned A wrapper must retain snapshot parity.
- **F / full_adaptation is conditional:** start only if A, its evaluation, and required diagnostics are complete by **Saturday 24 October 12:00 CT**, and the remaining schedule still protects 18 communication hours. Use the same loop/architecture/data, a smaller backbone LR, and the official/local-intersection snapshot-regression check. Freeze configuration and reserve a final test evaluation; deciding whether to run F is a completion/time decision, not test-score shopping. Cap extra training/evaluation at four hours and finish by Saturday 18:00; otherwise defer F entirely.

**Acceptance gate, Saturday 12:00:** P/S/A share saved prediction schema and IDs; tiny-batch A optimization has finite gradients/nonzero spread; no future inputs; sample-specific demand demonstrably reaches GENCO and physics; base weights/buffers unchanged; fixed-seed serialization round-trip; CPU forward/backward; all requested scores saved with masks/counts; D and the probe completed. Test disclosure distinguishes forecast error, same-input approximation error, raw predictions, and solver-refined predictions. An absent official reference is a parity blocker, not permission to claim compatibility.

**Cut if late:** drop F first, then large training budgets, additional probe variants, and extra sampled-reference solves. Preserve P/S, a valid measured A pilot, its attribution control, and honest calibration evaluation. If A cannot train/validate, retain the baselines and document failure; do not describe the submission as a successful post-trained forecaster.

### Stage C — calibration, stress monitor, and anticipatory warm starts

**Time-box:** Saturday 12:00–20:00 CT. Freeze forecasting checkpoints before fitting calibration/error models. At most two hours for numerical-difficulty model bring-up; no complex classifier search. Calibration and the hybrid experiment take priority over secondary prices and optional reliability classifiers.

`monitor.py` computes named events and selects scenarios; `scoring.py` evaluates them. Keep event thresholds fixed before inspecting test outcomes:

| Quantity | Definition and reporting contract |
|---|---|
| Thermal stress | For each valid rated line, `P(max(abs(S_from), abs(S_to))/rating > 0.95)`. Exclude unrated/disabled constraints explicitly; retain both-end checks. |
| Voltage stress | With range `r = Vmax−Vmin`, `P(Vm <= Vmin+0.05*r or Vm >= Vmax−0.05*r)`, including values outside bounds. Exclude fixed-voltage coordinates from headroom summaries. |
| Daily any stress | For each complete sampled trajectory evaluate the union across the declared hours/components, then take its weighted frequency. Never multiply complements of marginal probabilities. |
| Numerical difficulty | Probability a specified backend, initialization policy, tolerances, and fixed runtime/iteration budget fail to return an independently valid solution. Fit on train episodes; calibrate on reserved calibration episodes. Nonconvergence is not infeasibility. |
| Modeled prices | Weighted quantiles of checked active-balance duals, conditional on resolution and dual validity. Publish retained probability mass and unresolved mass; no market-price head or unconditional price claim. |

**Calibration procedure:** retain raw trajectory event frequencies and their Monte Carlo resolution. For S and A, fit at most one simple sigmoid probability calibration map per mode/event family on the 36 reserved calibration days; separately fit a daily-union map if evidence supports it. Pool hourly component events only with a stated weighting and show component/time breakdowns. Freeze map family and binning in advance; do not fit per-component calibrators on a handful of events. Use no calibration data for GRU/checkpoint selection or baseline coefficient fitting.

A post-hoc event map changes an event probability, not the underlying joint samples. Label raw versus calibrated probabilities separately; marginal coverage, PIT, and trajectory plots still describe the actual samples. Do not infer a calibrated joint distribution from a calibrated event classifier, and do not derive daily risk by combining calibrated marginals. Distribution recalibration beyond the learned uncertainty head is deferred.

As an engineering availability rule, require at least five positive and five negative **distinct forecast days** in calibration for an event map, show their episode allocation, and report “insufficient evidence” if absent. This threshold permits fitting only; it is not a statistical guarantee. Use coarse, predeclared reliability bins with counts; suppress unsupported bins. Test Brier scores against a training-prevalence baseline, plot raw/calibrated reliability, PIT/rank histograms, and 50/90 coverage/width. Rare events can remain unsupported even if aggregate scores look good. The post must explicitly say **72 test days from only four episodes, with correlated days, are thin evidence for rare-event calibration**. The calibration set is only 36 days from two episodes. Report per-episode results; any episode-resampling intervals are descriptive, not strong significance claims.

For daily event labels, a verified stress occurrence establishes a positive even if another hour is unresolved; a negative requires all relevant hours/components checked and non-stressed. Otherwise the label is unknown. Report this coverage, rather than turning unresolved labels into negatives. Use the same target masks/denominators across modes and separately describe resolved-only conditional results.

Numerical difficulty uses a small classifier (e.g. logistic regression) on information available before its associated solve: forecast/input features and optional already-computed embeddings. It must not see the measured runtime, iteration count, final residual, or status it predicts. Fit training-episode outcomes under exactly the deployment backend/budget, then calibrate on the reserved partition. If there are no failures or too few outcomes, publish that fact and an unavailable classifier; do not manufacture failure examples by changing the budget after test inspection. Treat inference from subsequently observed demand as an online estimate, distinct from a day-ahead difficulty estimate.

**Hybrid numerical experiment:** on each realized next-day hourly input compare:

1. Cold start using the documented backend default.
2. Warm start from the forecast mean state, with the same checked normalization/reference mapping. An ensemble mean need not be feasible; record its residuals and any mechanical bound adjustment.
3. Anticipatory warm start from a **numerically checked** forecast scenario prepared before realization. Preselect diverse high-stress scenarios under a fixed advance-solve budget, plus a representative audit allocation. When the current hour's actual demand becomes available, choose the nearest prepared scenario using a train-scaled distance on the now-available demand and fixed topology; never use the realized solution, future hours, or hindsight solver outcomes to select it.

Use A if available, otherwise label the S fallback explicitly. Pin candidate count/budget and distance rule using validation. Allocate a nonzero representative audit budget separately from stress selection; stress-selected solves cannot estimate an unconditional price/risk distribution. Representative samples use their known weights; otherwise prices are reported only for the selected subset with no population claim.

Each arm solves the same realized inputs with the same online budget/tolerances and cold fallback policy. Record initialization residuals, solver status, objective quality, iterations, success rate, and median/tail online latency. Include model inference, transfers, candidate selection, advance solves, failed attempts, and fallback work in total cost; show advance and online costs separately. Also report performance within matched total solver budgets where making efficiency claims. A warm start is checked on its candidate graph and then refined on the realized graph; that candidate check never certifies tomorrow's realized state. No assumed speedup on case30. If the backend cannot accept initial states, publish an unsupported result and do not claim a hybrid warm-start benefit.

**Acceptance gate, Saturday 20:00:** named event definitions/weights tested; calibration fitted solely on its partition or clearly unavailable; no marginal-independence shortcut; event/label coverage retained; no selected-sample population estimate; paired three-arm solver records or explicit backend limitation; complete online/advance cost accounting. Freeze result files for writing.

**Cut if late:** omit reactive-headroom displays and GENCO-reliability classifiers; drop secondary price plots if dual evidence is weak; reduce candidate count and audit size prospectively. Numerical-difficulty calibration may be unavailable for lack of events. Preserve the stress-calibration figures and paired hybrid evidence, including negative/unsupported outcomes.

### Stage D — figures, public artifacts, and self-contained post

**Time-box:** begin the narrative skeleton Friday; protect Friday 20:00–22:00, Saturday 20:00–24:00, and Sunday 06:00–18:00 CT (18 hours total). Sunday morning produces the release candidates; a complete post draft exists by Sunday 12:00. Stop new experiments at Sunday 06:00 except a necessary reproduction/error check.

Create **figures, not an app**, using Plotly exported to PNG with a pinned working image-export dependency. Verify one tiny PNG export on Friday so a browser/Kaleido installation issue does not consume Sunday. Keep figure inputs and scripts; static report export is optional after the complete post draft.

| Figure | Required content and evidence source |
|---|---|
| 1. Workflow | 72-hour observations → temporal GENCO → joint 24-hour samples → calibrated stress estimates → selected numerical preparation → realized-input refinement. Show the forecast cutoff and separate training from inference. |
| 2. One test day | Demand, Vm, and line-loading fan charts with P point line, S/A bands, and realized reference. Select the day/component by a rule fixed on validation (or a fixed test ID), not the most favorable result. Mark missing references. |
| 3. Calibration | Reliability diagram with bin counts, PIT/rank panels, and 50/90 coverage/width table; distinguish raw/calibrated event probabilities and show sample/day/episode support. |
| 4. case30 map | Fixed schematic layout, explicitly not geography; peak-hour thermal edge and voltage bus stress probabilities, with rated/unknown components marked. State how peak hour is chosen. |
| 5. Hybrid solver | Cold / forecast-mean / anticipatory results: iterations, success, online latency, advance work, total cost. Negative speedups are retained. |
| 6. Embedding probe | Frozen-embedding versus raw-feature linear probe at +1 and +24 hours, with common targets/splits and per-episode results. Include A's trainable count/base checksum as a nearby table or caption. |

Package the HF adapter model and dataset with cards and manifests, prepare GitHub tag `v0.1.0`, test the one-command CPU demonstration, and recompute every headline from saved results. Protect the contact/affiliation and anonymous-link checks. No figure may conflate solver-refined predictions with the raw forecast or imply that a few checks verify the whole distribution.

**Acceptance gate, Sunday 17:00:** complete Community-ready post; six figures or an explicit documented omission; all public links/cards/checksums ready; headline recomputation passes; clean-checkout CPU demonstration measured under 30 minutes on declared hardware; all claims supported or explicitly qualified. Submission owner posts by 18:00.

**Cut if late:** drop the HF static HTML Space, cosmetic revisions, secondary price graphics, and F discussion before cutting core figures, provenance, or the complete post. Publication with missing scientific evidence must describe the limitation; no fabricated completion statements.

## 5. Interfaces, file ownership, and reproducibility

Use ordinary functions, dictionaries, tensors, PyTorch modules, one YAML configuration, and argparse. No new registries, Hydra, plugin framework, service layer, database, or mandatory MLflow/Lightning training loop. Upstream internals may use their own registry; do not recreate it. `models/` must not import `training/`, visualization, or lumina-sdk. No runtime `sys.path` dependency on another checkout.

| Interface | Contract |
|---|---|
| `load_snapshot(sample_id)` / `load_window(episode_id, origin, history=72, horizon=24)` | Return upstream-compatible `HeteroData` graphs plus tensor dictionaries/metadata: stable IDs/order, times, physical units, masks, topology, available inputs, and targets kept separate. |
| `GNS_heterogeneous.forward(batch, return_embeddings=False)` | Original graphkit `HeteroData`/`Batch` input and checkpoint-compatible predictions. With `True`, return `(predictions, {bus, gen})` through the existing upstream API. |
| GENCO normalizer | Restore `normalizer_stats.pt` once; use pinned upstream transform/inverse-transform behavior. Do not refit it on temporal data. Track physical versus normalized values explicitly. |
| `Foresight.forecast(history, known_future, n_samples, seed)` | Fixed-topology joint demand/state samples, weights, origin/hour/component/scenario IDs, availability/verification flags, and metadata. `known_future` is calendar and genuinely known schedules/costs, never realized future profile/state. Research adds outage scenarios and weights later. |
| `solver.solve(graph, initial_state=None, budget=...)` | Status, independent residuals, valid solution/flows, objective, valid duals, runtime, iterations, and exact backend/init settings. Never silently retry with a different budget. |
| `monitor.score(predictions, solver_records, calibration=None)` | Named raw/calibrated event probabilities, calibration support, numerical-difficulty estimates if supported, conditional prices, and verification coverage. No rendering. |
| `scoring.evaluate(predictions, targets, masks, protocol)` | Common per-origin/per-variable metrics and aggregate tables with physical units, scaling, denominators, mode, raw/refined status, and reference identity. No duplicate metric implementation in figures. |

Samples have documented leading dimensions `[origin, sample, horizon, component, variable]` or equivalent per-node-type tensors. Weights sum to one over complete trajectories per origin. P carries deterministic output with an explicit no-uncertainty flag. Preserve baseMVA, angle units/reference, generator-to-bus mapping, branch direction, topology status, fixed-coordinate masks, and graph batching offsets. Maintain separate input masks and target-availability masks.

Create only files needed now:

| Location | Responsibility |
|---|---|
| `README.md`, `config.yaml`, `requirements.txt`, `LICENSE`, `THIRD_PARTY_NOTICES.md`, `.gitignore` | Runnable instructions, pins, scope/status, Apache-2.0 project licensing with upstream terms retained. |
| `compat/torch_scatter/` | Explicitly installable minimal shim and supported-API documentation; no global environment masquerade. |
| `models/genco.py`, `models/foresight.py` | Thin upstream integration and temporal inference. |
| `data/download/prepare.py`, `data/loader/dataset.py`, `data/synthetic/generate.py` | Acquisition, conversion/windows, resumable temporal generation. |
| `solver/adapter.py`, `solver/checks.py` | Numerical backend and independent physical checks. |
| `training/common.py`, `training/forecaster.py`, short P/S/A entry points | One training loop, explicit device, losses, checkpoints, and external run creation. F file only if its gate opens. |
| `scoring.py`, `monitor.py` | Single metric implementation; events/calibration/scenario selection. |
| `visualization/plots.py`, `visualization/grid.py` | Figures from saved results; no training or alternative metrics. |
| `scripts/reproduce_small.sh`, `scripts/headline_numbers.py`, a small embedding-probe script | Clean-checkout demonstration, claim recomputation, and the requested probe. |
| `tests/test_data.py`, `test_solver.py`, `test_model.py`, `test_monitor.py`, `test_smoke.py` | Meaningful scientific and compatibility failure checks. |
| `submission/post.md`, `submission/claims.csv`, compact result tables/figure sources | Reviewable post and claim-to-file map; large arrays live externally/HF. |

Run directories remain external and contain only used artifacts: config, version/asset manifest, source/profile/split IDs, seeds, package/device versions, preprocessing schema/scales, checkpoints, predictions, solver records, metrics, and figures. Record source revision, checkpoint revision, normalizer hash, solver settings, and generation resume state. Commit small immutable evidence tables needed for `headline_numbers.py`; runtime outputs, weights, and datasets are not a repository-local `runs/` tree. Reuse available Lustre data read-only; download scripts provide the equivalent path without Lustre.

## 6. Scientific contract and adopted reviewer decisions

Future solved states never enter inference. Historical measurements are available only through the cutoff and their masks. Static OPFData snapshots are never concatenated to impersonate time series. Missing labels are never zero targets. Solver nonconvergence is an unresolved numerical outcome, not proof of infeasibility. Stress near a limit is distinct from an actual limit violation, blackout, or certified operational danger.

Keep three evaluation axes separate: **forecast versus realized solution**, **neural approximation versus reference on identical sampled inputs**, and **raw versus solver-refined predictions**. State-reference scores are conditional on valid numerical labels; publish unresolved fractions and masking patterns. Independently assess physical residuals/limits even when normalized forecast scores improve. Costs from infeasible raw predictions do not establish superior optimality. Modeled prices are local numerical dual quantities under benchmark costs and stated resolution coverage.

The base normalizer is immutable; any new AR/probe/adapter-feature/joint-score transforms are fitted on temporal training data only. Model selection uses its own two episodes; event calibration uses its separate two episodes. Test episodes never choose loss weights, thresholds, source scaling, bins, or the best-looking figure. One model seed is permitted for this submission; repeated seeds and generality claims are deferred. Report negative results and actual compute.

| Reviewer finding | Challenge decision | Research follow-up |
|---|---|---|
| 1. Checkpoint/data identity and foundation claims | **Adopted:** case30 seed42 `genco-opfdata`, pinned source/config/stats; local/official test-ID intersection for parity/regression. New temporal episodes define the forecasting benchmark; no broader historical split audit on the critical path. Describe a pretrained case-specific solver. | Study transfer controls before claiming general foundation-model benefits; inspect each new case/checkpoint. |
| 2. Ambiguous demand-to-state sampling | **Adopted:** every demand draw constructs its own graph and passes through GENCO; state innovations and physics use that draw. No central-graph shortcut; latent centers are not predictive means. | Rank/representation/conditional-noise ablations and stronger physical consistency. |
| 3. Causal, dependent outage scenarios | **Deferred by scope:** fixed topology only, no outage oracle. | Generate hazard/repair paths sequentially with demand/stress dependence, using only then-available information; record joint weights and separate oracle experiments. |
| 4. Complete physical checks | **Adopted:** both branch ends, angle limits where configured, shunts/taps/units/reference, connected topology assertion, case30 single-generator assertion. | General islands/reference reassignment and multi-generator Qg allocation. |
| 5. Effective sample size and selective labels | **Adopted:** episode/source-pool splitting, 72 test days/four episodes, 36 calibration days/two episodes, explicit unknown labels/coverage and conditional metrics. | More independent episodes/seasons and rare events before strong calibration claims. |
| 6. Dependence-sensitive scoring and honest residuals | **Adopted:** variogram `p=0.5`, rolling-origin residual blocks, PIT/coverage plus event Brier/reliability; fixed scaling. | Larger calibration studies and distribution recalibration with validated uncertainty intervals. |
| 7. Attribution and price coverage | **Adopted:** A-demand-through-frozen diagnostic and matched-input numerical references; price quantiles conditional on valid resolved weight. | F versus A, scratch/raw temporal controls, separate regularization ablations, and representative price studies. |

No RL, no assumed differentiation through Ipopt, and no “KKT loss” label for a balance penalty. A true KKT extension requires valid duals plus stationarity and complementarity. Bound transforms are not a full feasibility certificate. Never hide calibration failure by selecting only easy/resolved examples or by recalibrating on test data.

## 7. Platform gate and day-by-day schedule

**CPU/available NVIDIA is the default. Aurora is optional.** Reserve one allocated-node XPU attempt Friday, at most two hours including shim/import/forward/backward checks. Run it alongside already-running data/CPU work if resources allow; do not introduce an XPU dependency into Stage A or B. If forward and backward pass, use XPU with measured timing. If only inference passes, say “XPU inference verified; training on CPU/CUDA” with the actual training device. Otherwise say “XPU pending.” Preserve vendor Torch and record environment diffs. These are result labels, not prewritten claims.

Owners below are responsibilities, not assumptions about extra personnel: **Implementation** = the coding agent with user-provided execution access; **Scientific review/submission** = the user; **Writing/release** = agent prepares, user reviews/publishes. Parallel work refers to independent jobs/resources, not an assumption of a staffed team. A lack of XPU allocation never blocks the CPU route.

| CT date/time | Work and exit evidence | Owner | Hard stop / communication allocation |
|---|---|---|---|
| Thu 22 late evening | One review of this contract; resolve only immediate execution blockers | User + agent | No implementation before review |
| Fri 23 AM, 08:00–12:00 | Pinned graphkit, shim, `--no-deps`; CPU checkpoint/parity versus official CLI on intersection; kick off 200-hour shard; optional XPU attempt in parallel | Implementation; user supplies compute access only if missing | XPU two hours; datakit pipeline two hours |
| Fri 23 PM, 12:00–20:00 | Full generation/labels, loader, rolling-origin baseline, `scoring.py`; P/S evaluated on frozen test IDs; A throughput check then training | Implementation | First baseline numbers Friday; no extra platform work |
| Fri 23, 20:00–22:00 | Draft method/limitations/provenance and figure captions while jobs run | Writing/release | **2 protected communication hours** |
| Sat 24 AM, by 12:00 | A trained/evaluated; attribution diagnostic; embedding probe | Implementation | F eligibility fixed at noon based on completion/time |
| Sat 24 PM, 12:00–20:00 | Monitor/calibration and hybrid warm starts; F only if ahead and finished by 18:00 | Implementation + scientific review | Optional additions cannot consume writing time |
| Sat 24, 20:00–24:00 | Freeze headline evidence; assemble results narrative and captions | Writing/release | **4 protected communication hours** |
| Sun 25 AM, 06:00–12:00 | Final PNG figures, complete write-up draft, HF model/data release candidates/cards/checksums; GitHub `v0.1.0` candidate; one-command reproduction | Writing/release + user for account metadata | **6 protected communication/release hours**; no new experiments |
| Sun 25 PM, 12:00–18:00 | Audit all claims/links, review post, publish artifacts/tag and Community entry by 18:00 | User submission owner; agent prepares/verifies | **6 protected communication/release hours**; HTML Space only after draft and gates pass |
| Sun 25 18:00 → Mon 26 06:59 | Deadline contingency margin | Submission owner | Buffer for failed upload/correction only; no planned features |

Use the known clone and default asset paths; do not ask the user to repeat them. During Stage A ask only if read-only inspection cannot resolve a necessary asset/cache, CPU/solver access, writable location, or compute budget. Team metadata/HF namespace can be collected during packaging; it need not delay setup/data construction. Deadline correction is resolved from the organizer page, so no clarification is needed for it.

## 8. Open-science deliverables and release gates

- **GitHub:** Apache-2.0 for our code; retain upstream licenses/notices for every copied/adapted source segment and record source URLs, revisions, paths, and modifications. Check solver/data redistribution terms independently. Tag `v0.1.0` after evidence checks. Pin the full graphkit SHA, checkpoint revision, datakit package/source hash, solver revision, profile identity, and normalizer hash.
- **Installation:** common `requirements.txt` plus README Aurora/NVIDIA `<details>` toggles and a CPU route. Hardware Torch installation is separate; no replacing Aurora vendor packages. Include shim scope, import checks, before/after environment snapshots, and platform limitations.
- **`scripts/reproduce_small.sh`:** one command from a clean checkout on CPU; explicitly downloads missing pinned small assets, verifies checksums, loads the official base plus adapter, forecasts at least one complete 72→24 example in P/S/A, runs a small numerical check, and writes scores/one figure externally. Target **under 30 minutes**, measured on stated CPU hardware with download/setup timing and network assumptions disclosed. Use released weights/data; this is a small demonstration, not a claim that full training fits 30 minutes. Provide a separate full-regeneration command. If the target fails, report measured time and the unmet gate.
- **`scripts/headline_numbers.py`:** recompute every post number from saved compact result files, with no training/solver rerun. Include all headline denominators, runtimes, trainable counts, calibration support, and relevant rounding. `submission/claims.csv` maps each claim to file/hash, metric definition, filtering rule, and script output key; every claim, including qualitative comparisons, has repository evidence.
- **HF model repository:** adapter weights, pinned base reference/revision, matching normalizer and YAML, adapter/calibration configuration, feature schema/scales, manifest, checksum, and model card describing training data, intended use, limits, and licensing. Include F separately only if completed. Never imply untouched GENCO parameters were fine-tuned in A.
- **HF dataset repository:** temporal demand/topology metadata, labels/masks, solver outcomes/duals, episode/source-pool split manifest, generation settings/seeds, SHA256 for every shard, and dataset card with profile provenance, geographic assumptions, sampling law, resolution coverage, and license terms. Include failure records. No full weights/dataset blobs in Git.
- **Meaningful tests:** split/source leakage; future-target/input isolation; sample-dependent demand reaching GENCO and physics; A base weights/buffers unchanged; fixed-seed serialization round-trip. Also exercise independent physical checks, invalid-label masks, scenario weights/daily unions, calibration partition separation, and the supported scatter shim/CPU parity. Tiny fixtures only; no UI-test expansion.
- **Release evidence:** passed/failed/skipped/unverified table, actual commands, device/version manifest, full-run paths, compact results, six PNGs plus their data, and clean-checkout log. Recheck every public link and claim after final artifact revisions.

All publication outputs are prepared for review. This document does not claim that repositories, models, data, figures, or tests already exist.

## 9. Community post outline

**Proposed title:** “GENCO Foresight: forecasting tomorrow's grid stress, then preparing the solver.”

1. **Summary:** the task and three measured numbers: one forecast-skill metric (A versus S with units/denominator), one calibration result (coverage/width or stress Brier with support), and one hybrid result (online change accompanied by total cost). Fill from `headline_numbers.py`; show negative or unavailable results honestly.
2. **Motivation:** a snapshot solver needs known loads; an operator must prepare before tomorrow's loads are known. Explain what uncertainty and advance computation add. Figure 1.
3. **GENCO + OpenGridFM + method:** exact checkpoint/source, datakit profile use/AR extension, episode split and forecast cutoff, P/S/A, per-sample physics, immutable normalizer, and attribution diagnostic. Explain extension of a pretrained case-specific model without broader transfer claims.
4. **Results — Forecasting and temporal representations (Stage B):** point/probabilistic scores, raw/refined distinction, A-demand control, identical-input approximation check, and embedding probe. Figures 2 and 6.
5. **Results — Are operating-stress probabilities calibrated? (Stage C):** raw versus calibrated reliability, Brier, PIT, coverage/width, day/episode counts, missing labels, and insufficient rare-event evidence. Figures 3 and 4. This is the central result.
6. **Results — Does anticipation help the numerical solver? (Stage C):** cold, mean, anticipatory starts, success/iterations/online and total cost, including advance work. Figure 5; no assumed improvement.
7. **What this gives an operator:** tomorrow's time/component stress outlook, visible uncertainty/evidence, and candidate initializations for numerical checks on subsequently observed inputs. It does not certify future safety or automate dispatch.
8. **Limitations:** case30, fixed topology/costs/commitment, synthetic ERCOT-to-IEEE mapping, only 72 test days/four episodes and two calibration episodes, source-profile reuse within splits, one training seed, local numerical references, conditional missing labels/duals, platform and runtime limits. No ramp/market/real-world reliability claim.
9. **Open science:** code/tag, model, dataset, exact reproduction command, claim-to-file map, licenses, contact and full team affiliations.
10. **Next steps:** a short pointer to the Research Track: more calibration evidence, N1 causal paths, stronger controls, larger grids. No unexecuted feature presented as a result.

## 10. Cuts and deferrals

Anything outside the Challenge Track is deferred unless needed to fix a demonstrated correctness/reproducibility defect. Each cut protects calibration evidence and the post deadline.

| Item | Decision and reason |
|---|---|
| N1 hazard/repair, N2/N-k, intensified-stress regimes | Research only; causal scenario construction and topology validation exceed this deadline. |
| Q / frozen interpolation | Cut from challenge; high-dimensional interpolation adds a separate approximation audit. |
| F / full adaptation | Conditional noon-Saturday stretch only; A and calibration have priority. Full F ablations remain research. |
| Central-graph sampling plus deviations | Dropped; every sampled demand must reach GENCO and physics. |
| ChroniX2Grid | Research only; installation/integration risk within 72 hours. |
| RTS-GMLC reliability assumptions | Research only; no outages in the challenge. |
| Reactive-headroom UI and GENCO reliability classifiers | Deferred if time-boxed out; thermal/voltage calibration and numerical difficulty are the selected monitor quantities. Raw physical residual metrics stay. |
| Price prediction head, variable offers, market clearing | No challenge head; dual prices are secondary, and market realism requires a different experiment. |
| Larger grids 57→118→500→2000 | Research only; case/checkpoint/mapping and runtime validation are separate work. |
| Multi-generator Qg allocation and island handling | General implementation deferred; assert supported connected case30 structure now. |
| 168-hour history and alternate horizons | Research only; freeze 72→24 for the submission. |
| Repeated training seeds, scratch controls, separate physics/replay ablations | Research only; qualify one-seed findings now and retain the cheap demand-attribution control. |
| Broad pretraining-contamination audit | Not a challenge prerequisite beyond prescribed static parity/regression IDs; forecasting data are newly constructed, with explicit source splitting. |
| Distribution-wide recalibration / formal guarantees | Research only; the challenge measures sample calibration and fits small supported event maps. |
| Streamlit app, solver queue UI, interactive scenario playback | Research only; the jury needs figures and a complete post. |
| Static Plotly HTML HF Space | Stretch only after the post draft/release gates; no app infrastructure. |
| Independent Julia reference backend | Research only unless already available and indispensable for a correctness issue; independent physics checks stay mandatory. |
| XPU optimization | Two-hour optional gate; never on the critical path. |
| RL, differentiable iterative solver, KKT objective | Out of scope; no validated need or correct implementation budget. |

## Appendix — Research Track after the deadline

Retain the original four-stage structure with evidence-based milestone reviews. Challenge artifacts may satisfy individual gates but do not automatically complete a research milestone. This appendix preserves the substance of original Stages 2–4 with the seven reviewer clarifications and corrected source/checkpoint/normalizer choices applied. It does not allocate work before submission.

### Research Stage 1 — supported functionality, setup, and contracts

Re-audit assets/README/current changes and completed challenge evidence; retain one config, simple interfaces, external artifacts, attribution, and explicit device selection. Expand supported case/checkpoint inventory only after inspection. Resolve XPU issues on allocated compute nodes without replacing vendor Torch. Keep original upstream imports as the default and vendored licensed inference as fallback. Record CPU/accelerator forward/backward and missing-asset behavior separately. Restore configurable 168-hour history only with new window/data sufficiency checks.

**Gate:** inspected inputs and writable external paths; strict checkpoint/normalizer loading; reproducible snapshot parity and explicit unsupported combinations; CPU and available accelerator evidence; README documents the five research modes and shared schema. Review results before Research Stage 2.

### Research Stage 2 — richer temporal data and numerical references

1. Extend the standalone loader across existing T1/N1 availability, preserving canonical local IDs, frozen splits, and nested 100/1K/10K training subsets. Keep the official split intersection for checkpoint regression. Do not modify or recurate source OPFData. Maintain explicit download/cache scripts and license/provenance records for machines without Lustre.
2. Add the published ChroniX2Grid seasonal/weekly/spatiotemporal load recipe as an alternative to datakit-plus-AR. Reuse its implementation where practical; mark any inspired approximation and explain differences. Retain documented geography and source-profile splitting. Dispatch/load-flow outputs do not replace checked AC-OPF labels.
3. Increase independent episodes, seasonal coverage, calibration event counts, and source diversity before rare-event claims. Split complete source blocks/episodes before windows; keep train, model-selection, risk-calibration, and test separate. Fit new transforms only on training; never refit a pretrained checkpoint's normalizer. No randomly concatenated static snapshots and no held-out static templates in training.
4. Add bounded N1 outage/repair paths with at most one unavailable component. Use documented RTS-GMLC assumptions where suitable and explicit component mappings. Evolve state-dependent hazards using then-available demand/operating state; forecast scenarios jointly evolve demand, state, and outages rather than sampling independent paths when hazards depend on stress. Future realized outages belong only to an explicitly labeled oracle comparator. Save probabilities/weights, repair rules, and ordinary versus intensified-stress regimes separately.
5. Extend generator/bus Qg allocation, offline masks, connected components, island detection, and angle references. Check capability bounds and explain any underdetermined allocation. Retain both-end thermal/angle/shunt/tap/unit checks; do not hide derived-variable violations.
6. Pin/install the standalone solver; reuse minimal licensed integrated code only if needed. Inspect warm starts, budgets, tolerances, valid duals, and status semantics; optionally cross-check with PowerModels/Ipopt. Preserve failed attempts with masks and resumable shards/manifests. Quantify selective label coverage instead of evaluating only successful cases silently.
7. Retain marginal-price perturbation checks and conditional valid-dual interpretation. Variable offers/cost time series are a documented extension using [Chatzos et al.](https://arxiv.org/abs/2110.13772); they do not make snapshot OPF equivalent to an actual market's clearing process.

**Gate:** canonical IDs unchanged; train-only new transforms and leakage-safe windows/source pools; reproducible correlated load/causal N1 paths; graph/solver round-trip; valid independently checked solutions and controlled failure masks; supported warm starts exercised; trustworthy dual units or unavailable labels; external data-quality report/manifests. Test data/solver behavior before Research Stage 3. GENCO-error labels always identify the checkpoint used.

### Research Stage 3 — faithful GENCO and controlled adaptation

1. Reuse pinned graphkit architecture, original names, strict weights, matching immutable normalizer, native embedding API, and minimum conditioning hooks. Preserve unconditioned inference and snapshot parity. Audit native-scatter portability with numerical/gradient comparisons rather than recreating the network from a paper.
2. Retain temporal observed-state projections, per-node-type GRUs with stable IDs, future calendar/schedule inputs, and per-sample future graphs. Study rank/memory widths and conditional state noise with explicit controls; all physics uses each sample's own demand/topology. Derive dependent Qd/Qg/flows and omit fixed/offline coordinates/reference angles.
3. Compare **A and F** with identical architecture, initialization of additions, data, and declared compute budgets. A freezes base parameters; F updates them with a smaller LR. Use one loop, exact trainable counts, memory, optimizer steps, and actual compute. Cache history only where gradients are not required. F must pass a separate same-input snapshot-regression evaluation.
4. Keep energy score with train-defined scaling, modest physical penalties, and effective training-split replay. Study calibration/spread damage from regularization. Run the combined removal of physics+replay as the original inexpensive ablation; it supports only a combined effect. Separate ablations require their own budgets. Retain A-demand-through-frozen and add scratch/raw temporal controls before claiming pretrained-representation benefits. Distinguish constant frozen-path replay diagnostics from gradient-carrying replay regularization.
5. Use the solver as teacher/verifier without assumed differentiation through it or RL. KKT work, if justified later, requires valid stationarity/complementarity evidence. Export inference independent of training/SDK, with full schema, scales, model/source revisions, and base references or full adapted weights as appropriate.

**Gate:** upstream parity including physics/normalization; meaningful tiny-batch optimization and sampling gradients; A unchanged and F intended parameter updates; no future target leakage; sample-specific demand/topology reaches physics; fixed-seed serialization; real CPU/available accelerator checks; comparable completed A/F pilots and F regression report. Unavailable official parity remains a blocker. Review before Research Stage 4.

### Research Stage 4 — five-mode comparison, calibrated monitor, and demo

1. Restore **P/Q/S/A/F** with common forecast origins, available information, episodes, topology assumptions, and result schema. P stays deterministic. S uses joint rolling-origin residual blocks. **Q** interpolates frozen GENCO over representative nodes in joint scenario space, within compatible topology strata only; validate against S, preserve weights, and directly evaluate unsupported points. Never equate marginal input/output quantiles.
2. Start with fixed topology, then causal N1 scenarios and explicitly labeled oracle experiments. Add more cases in order **57, 118, 500, then 2000**, only with supported data/checkpoints and new parity/mapping checks. Keep case2000's reported N1-only local availability explicit. Unsupported combinations remain unsupported.
3. Retain all challenge forecast/approximation/raw-refined distinctions and scores: point error, CRPS, energy/variogram, coverage/width/PIT, event Brier/reliability, physical residuals/violation rates, objective/cost, runtime/memory. Report per-variable units and full coverage. Expand representative sampled-input numerical comparisons and quantify unresolved probability mass.
4. Run repeated training seeds and episode-level uncertainty estimates before strong comparative claims. Ensure enough independent source/episode support; additional overlapping windows are not independent evidence. Retain negative results and compare preparation/advance work as well as online cost.
5. Expand the named monitor with reactive headroom, empirical physical-reliability events, separately calibrated same-input approximation-error exceedance, and backend-specific numerical difficulty. Freeze each checkpoint before fitting train-episode models and reserved-partition calibrators. Rare-event support and unknown labels remain visible. Do not collapse these into one safety score or infer daily probability by independent marginals.
6. Select diverse consequential scenarios with representative audit samples and explicit sampling weights. Continue cold/ordinary/anticipatory warm-start comparisons on subsequently realized inputs, including fallback success, event recall, advance work, and matched budgets. Use representative sampling or justified weights for population risk/price estimates; selected stress cases alone do not provide them.
7. Build the deferred **Streamlit + Plotly app** reading saved results: fixed schematic or actual coordinates; hour/mode selectors; three-row 24-hour timeline; thermal/voltage/reactive views; component bands/limits/events/prices; coherent scenario playback with outages; solver queue/results and candidate availability. Separate aggregate probabilities from individual scenarios, raw from refined, and checked from unchecked. Computation remains in `monitor.py`/`scoring.py`; UI does not redefine metrics.

**Gate:** identical episode IDs/schema across all five modes; joint sampling/weights/interpolation fixtures including cross-topology rejection; leakage-free supported calibration and selection-aware estimates; end-to-end case30 demo with valid numerical outcomes; distinct physics/approximation/difficulty measures; full cost accounting; clean-checkout reproduction; expansion only after case30 passes. Tests target scientific failure modes, not every UI detail. Report exactly which claims each experiment supports.

### Completion record

- [ ] Challenge contract reviewed once; Stage A authorized.
- [ ] A: pinned setup, data, and numerical-reference evidence.
- [ ] B: P/S/A, diagnostic, probe, and shared scoring evidence.
- [ ] C: supported calibration and hybrid-solver evidence.
- [ ] D: verified figures/artifacts/post submitted; URL recorded.
- [ ] Research 1: expanded inspected setup/contracts.
- [ ] Research 2: richer temporal data and numerical references.
- [ ] Research 3: controlled A/F adaptation and ablations.
- [ ] Research 4: five-mode evaluation, larger cases, and app.

Files existing is not milestone completion. Update README and external reports with actual checks, outcomes, limitations, and claim support.

### Additional implementation anchors

- [Official GENCO OPFData reproduction instructions](https://github.com/gridfm/gridfm-paper-genco/blob/main/repro_instructions/5.2_OPF_on_opfdata.md): the pinned model card remains the acquisition/evaluation anchor; retain our selected `main` commit rather than switching to the paper branch.
- [ChroniX2Grid recipe](https://chronix2grid.readthedocs.io/en/latest/DESCRIPTION.html), [code](https://github.com/Grid2op/chronix2grid), and [paper](https://zenodo.org/records/15184237): Research Track only.
- [RTS-GMLC data](https://github.com/GridMod/RTS-GMLC): Research Track outage assumptions, with explicit mappings.
- [ALCF Aurora PyTorch](https://docs.alcf.anl.gov/aurora/data-science/frameworks/pytorch/) and [PyG guidance](https://docs.alcf.anl.gov/aurora/data-science/frameworks/pyg/): optional platform guidance; vendor environment is preserved.
