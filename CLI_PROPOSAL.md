# power-fm-foresight: proposed Python package and CLI

Use **`power-fm-foresight`** as the distribution name, **`power_fm_foresight`** as the Python import, and **`pff`** as the command. The long command `power-fm-foresight` can be an alias for the same entry point.

Five commands cover the workflow:

| Command | Purpose |
|---|---|
| `pff prepare` | Prepare or register data and foundation-model assets; run the required audits. |
| `pff train` | Train, resume, or launch a matrix of independent experiments. |
| `pff evaluate` | Evaluate checkpoints and jointly rescore ensembles. |
| `pff report` | Show status or build tables, comparisons, and figures from recorded results. |
| `pff verify` | Run a declared correctness, reproducibility, or performance protocol. |

This is an interface proposal, not an implemented CLI. It covers the experiments already run; sample-parallel training, origin batching, and selective bf16 still require their own implementation and gates.

## Shared conventions

| Argument | Meaning / proposed default |
|---|---|
| `--case INT [INT ...]` | One or more of `14 30 57 118 500 2000`. Maps to the canonical grid identifiers. `200` is not a supported case. |
| `--fm genco` | Foundation model; default `genco`, initially the only choice. |
| `--root PATH` | Storage root for assets, data, runs, logs, and temporary artifacts. Can come from `PFF_ROOT`. |
| `--config PATH` | Existing YAML configuration, including `${base}` interpolation. |
| `--name NAME` | Human-readable campaign/run label; the tool also assigns an immutable unique identity. |
| `--dry-run` | Resolve configurations, enumerate runs, and print resource/budget requirements without launching work. |

For new runs, precedence is **explicit CLI arguments → supplied YAML → selected recipe**. Resume uses the saved configuration and rejects incompatible overrides. Existing historical configurations retain their original semantics.

`--case` and `--fm` select inputs for preparation/new training. Evaluation, reporting, verification of saved runs, and resume infer these from manifests; they should not require users to restate them.

Keep seeds distinct:

- `train --seed`: adapter initialization and training RNG, one integer or a list.
- Data-generation seed: part of the dataset recipe/configuration and dataset identity.
- `evaluate --eval-seed`: evaluation-noise convention, recorded separately.
- The GENCO checkpoint's **seed42** is an asset identity, independent of the adapter training seed.

## 1. `pff prepare`

```text
pff prepare --case CASE... [--fm genco] --dataset PATH [options]
```

Resolve the matching network, checkpoint YAML, weights, and saved normalizer; generate resumable data shards or register existing data; run Stage A physics/data audits, event-support analysis, and the mechanical headline-bus selection. Publish a ready manifest only after the required checks pass.

| Argument | Meaning / default |
|---|---|
| `--dataset PATH` | Dataset collection root; each selected case gets a separate entry and manifest. |
| `--recipe v4` | Data recipe: demand law 0.80–1.00 and the recorded temporal/split rules. |
| `--reuse PATH` | Register and audit an existing dataset read-only; no regeneration. For several cases, specify the mapping in YAML. |
| `--audit-only` | Rerun audits on existing labels without generating labels. |
| `--workers INT` | Concurrent CPU solver processes per node; bounded by the allocated CPUs. |
| `--parity full\|subset` | Official/local snapshot parity scope; default `full`. |
| `--parity-samples INT` | Required for subset parity; e.g. `200`. |
| `--parity-seed INT` | Seed for selecting the official/local test-ID-intersection subset; record the actual IDs. |
| `--parity-budget DURATION` | Parity time cap; e.g. `20m`. Hitting the cap is incomplete, not a pass. |

Grid-specific solver limits, warm starts, episode counts/splits, and pilot size belong in YAML. For example, the case2000 pilot configuration can declare a 600-second per-solve wall limit, 3000 iterations, and 96 solver workers. A half-size dataset must explicitly record 10 episodes and the 5/1/2/2 split in a new dataset identity.

The dataset card must distinguish full parity from **“subset parity (200)”**. Dataset case = network case = checkpoint YAML case remains enforced. Existing case30 v4 data should be registered/reused, not silently regenerated.

## 2. `pff train`

```text
pff train --case CASE... --dataset PATH --seed INT... [options]
pff train --config CAMPAIGN_YAML [options]
pff train --resume RUN_OR_CAMPAIGN [options]
```

This is the command for ordinary training, the small-grid fleet, added ensemble seeds, A0, the λ sweep, and the large-grid pilots.

| Argument | Meaning / default |
|---|---|
| `--recipe v4-off\|v4-on` | Proposed presets: recorded v4 recipe with determinism OFF or ON. Default for new runs: `v4-off`. |
| `--seed INT [INT ...]` | Training seeds, e.g. `--seed 0 1 2`; require an explicit choice. |
| `--variant conditioned\|a0` | Default `conditioned`; `a0` selects the demand-only ablation. |
| `--balance-weight FLOAT [FLOAT ...]` | λ; default `1.0`. A list launches a sweep. |
| `--samples INT` | Global Monte Carlo trajectories per origin, K; default `8`. Case2000 pilot uses `2`. |
| `--lr FLOAT` | AdamW learning rate; default `1e-4`. Remaining AdamW/clipping settings come from the recipe. |
| `--max-epochs INT` | Epoch cap; recipe default, or `6` for the pilots. |
| `--patience INT` | Selection-ES early-stopping patience; default `3`. |
| `--chain-budget DURATION` | Cumulative worker-time cap per training chain, across allocations; e.g. `6h`. |
| `--label TEXT` | Explicit category such as `pilot`, `K=2 pilot`, or `budget-capped`. |
| `--resume RUN_OR_CAMPAIGN` | Continue incomplete chains from committed checkpoints and journals; completed chains are skipped. |
| `--extend-budget DURATION` | Explicit budget extension with `--resume`; creates a new child identity preserving the parent evidence. |

Lists of cases, seeds, and balance weights form a **Cartesian product**, printed before submission. For example, four cases × three seeds = 12 chains. A campaign contains one immutable resolved configuration per chain.

A campaign YAML can instead list heterogeneous runs explicitly, such as case500 K8 and case2000 K2. Matrix flags and an explicit run list cannot be combined ambiguously; `--dry-run` shows the final plan.

The initial implementation remains **B=1, fp32, one tile per independent chain**. Covariance rank stays 8 in the recipe; it is not `--samples`. Adding workers/nodes launches more independent chains and does not distribute one model's gradients. Do not expose unimplemented bf16, B>1, or distributed-training choices as usable options.

For A0, record the variant in the manifest: no latent conditioning, frozen backbone under `no_grad`, state innovations after the backbone, differentiable recomputed physics, and demand-head training through demand ES coordinates. Its zero-innovation path must pass the frozen-GENCO parity check.

Exact historical replication additionally requires the historical source, data, and asset manifests; choosing `v4-on` alone is not a claim of exact released-artifact reproduction.

## 3. `pff evaluate`

```text
pff evaluate --runs RUN_OR_CAMPAIGN... [options]
```

| Argument | Meaning / default |
|---|---|
| `--runs PATH [PATH ...]` | Input runs/campaigns; manifests determine the grids, variants, and training seeds. |
| `--checkpoint best\|latest` | Default `best`, selected by selection-partition ES. |
| `--partition selection\|test` | Default `test`; selection is for optimization comparisons. |
| `--modes C P P0 S A D` | Modes to score; default all six existing modes. Reuse compatible frozen baselines by hash. |
| `--samples INT` | Forecast draws **per member per origin**; default `64`, independent of training K. |
| `--eval-seed INT` | Default `42` under the versioned existing origin-noise convention. |
| `--ensemble-sizes INT [INT ...]` | Optional mixture sizes, e.g. `1 2 3 5 8`; member evaluations remain available separately. |
| `--members PATH` | Optional manifest fixing eligible members and their order; otherwise ascending seed order within each compatible group. |
| `--shards INT` | Number of origin shards; e.g. `4`. Does not change samples or scores. |
| `--allow-budget-capped` | Explicitly include budget-capped checkpoints, labeled as such. Failed runs remain excluded. |
| `--snapshot-at TIMESTAMP` | Fix the checkpoint versions used for a best-so-far evaluation; record observed epochs and capture time. |

Ensembles combine samples with equal member weight and rescore the **joint** sample set, including all cross-member pairs. They are not averages of separately computed ES values. Members must share case, dataset/split, scoring masks/scales, and compatible experimental settings. Different λ values and A0/conditioned runs form separate groups.

Insufficient eligible members produce an unavailable cell; no silent replacement of a missing seed or selection by test score. A frozen member manifest makes mixture curves reproducible. Preserve the fixed evaluation-noise convention, determinism ON for evaluation, and the per-grid same-input P/Ipopt audit. A running model can be evaluated only through a committed immutable checkpoint snapshot.

## 4. `pff report`

```text
pff report --runs RUN_OR_CAMPAIGN... [options]
```

| Argument | Meaning / default |
|---|---|
| `--runs PATH [PATH ...]` | Runs and/or evaluation campaigns to summarize. |
| `--kind status\|summary\|ensemble\|comparison` | Default `summary`; `status` shows jobs, chain progress, stop reasons, and remaining budgets. |
| `--reference PATH` | Reference runs/results for a comparison, e.g. historical v4 ON seeds. |
| `--group-by case` | Default grouping; variants, λ, training K, precision, and pilot categories remain separate. |
| `--format markdown json csv png pdf` | Requested output formats; default Markdown + JSON. Figures apply to curves. |
| `--output PATH` | Destination for the report/artifact collection. |

Produce per-seed results, seed mean ± sample standard deviation, and **separate** jointly scored mixture rows. Include actual seed counts, epochs, budget caps, and unavailable metrics. A single seed has undefined seed standard deviation. Pilots do not enter the primary cross-grid fleet table. This command reads persisted results; missing forecasts require `evaluate`, not implicit compute submission.

## 5. `pff verify`

```text
pff verify --check CHECK --protocol PATH [--runs PATH...] [options]
```

| Argument | Meaning |
|---|---|
| `--check regression\|resume\|reproducibility\|gradients\|timing` | Select the type of verification, not a separate command for each historical experiment. |
| `--protocol PATH` | Required versioned JSON declaring inputs, seeds, origins, steps, restart points, thresholds, resource caps, and stop conditions. |
| `--runs PATH [PATH ...]` | Existing inputs for curve comparisons, reference checks, or resume tests. |
| `--case CASE --fm genco` | Needed when the protocol creates new model runs rather than comparing saved runs. |
| `--output PATH` | Gate JSON, small report, and links/hashes for bulk evidence. |

Examples of protocol contents: Stage 0's two 300-step ON runs plus altered-fixture detection; a boundary-restart mechanism test; fp64 CPU versus fp32 XPU gradients; or matched production timing windows. Keep the deterministic resume mechanism test separate from production acceptance. Performance results can be negative without failing correctness. No thresholds are inferred or relaxed after results are read.

## Execution arguments

The same execution arguments apply to `prepare`, `train`, `evaluate`, and `verify`. They do not change the scientific recipe.

| Argument | Meaning / default |
|---|---|
| `--backend local\|pbs` | Default `local`; `pbs` submits pinned work and returns job/campaign IDs. |
| `--device cpu\|xpu` | Model execution device; CPU preparation remains CPU work. Other accelerators can be added when tested. |
| `--device-index INT` | Local execution only; default `0`. PBS assigns and isolates each worker's tile, which the worker addresses as `xpu:0`. |
| `--queue debug\|debug-scaling` | PBS queue. |
| `--nodes INT` | Nodes per allocation. |
| `--workers-per-node INT` | Independent model workers per node, one visible tile each; not distributed ranks. |
| `--walltime DURATION` | Per-allocation limit; default and maximum `1h` for these queues. |
| `--max-submissions INT` | Total successful submissions for this campaign, **including the first**; default `1`. |
| `--deadline TIMESTAMP` | Optional cutoff after which the campaign may submit no additional jobs. |

PBS continuations must survive closing the CLI/conversation: persist the plan and use scheduler-backed continuation, independent of an interactive session. The controller enforces both submission limits and cumulative per-chain budgets. Data preparation and evaluation have their own budget records; they do not consume an unreported training allowance. Training starts for a case only once that case's dataset is ready.

Pin source/configuration/assets before submission. Save atomically per epoch and every 15 minutes, and stop gracefully before allocation expiry. Resume must restore the **complete committed run state**, including metrics journals, RNG/order, counters, optimizer, and checkpoint-selection/early-stopping state. Replacing only checkpoint files is insufficient. A changed scientific configuration gets a new run identity; historical manifests are never rewritten.

## Example invocations

Assume `PFF_ROOT=/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO`. Names below identify new campaigns; paths are illustrative proposal outputs.

### Data and the original 12-chain fleet

```bash
pff prepare --case 14 30 57 118 --fm genco --recipe v4 \
  --dataset "$PFF_ROOT/datasets/v4" --config configs/data-v4.yaml \
  --parity subset --parity-samples 200 --parity-seed 42 --parity-budget 20m \
  --backend pbs --queue debug --nodes 1 --walltime 1h --max-submissions 12

pff train --case 14 30 57 118 --fm genco --seed 0 1 2 \
  --dataset "$PFF_ROOT/datasets/v4" --recipe v4-off --name fleet \
  --chain-budget 6h --patience 3 --device xpu \
  --backend pbs --queue debug-scaling --nodes 2 --workers-per-node 6 \
  --walltime 1h --max-submissions 12
```

The data YAML registers existing case30 v4 labels and defines generation for the other cases. The subset scope is an explicit protocol choice, not interchangeable with full parity.

### Ensemble depth, A0, and λ sweep: all use `train`

```bash
pff train --case 14 30 57 118 --seed 3 4 5 6 7 \
  --dataset "$PFF_ROOT/datasets/v4" --recipe v4-off --name ensemble \
  --chain-budget 4h30m --device xpu \
  --backend pbs --queue debug-scaling --nodes 2 --workers-per-node 10 \
  --walltime 1h --max-submissions 6

pff train --case 30 57 118 --seed 0 --variant a0 \
  --dataset "$PFF_ROOT/datasets/v4" --recipe v4-off --name a0 \
  --chain-budget 3h18m --label budget-capped --device xpu \
  --backend pbs --queue debug-scaling --nodes 1 --workers-per-node 3 \
  --walltime 1h --max-submissions 4

pff train --case 30 --seed 0 --balance-weight 0.1 10 \
  --dataset "$PFF_ROOT/datasets/v4" --recipe v4-off --name lambda \
  --chain-budget 3h18m --label budget-capped --device xpu \
  --backend pbs --queue debug-scaling --nodes 1 --workers-per-node 2 \
  --walltime 1h --max-submissions 4
```

The λ=1 fleet run can be a labeled reference; its different training budget must remain visible. To reproduce the time-boxed window, add the declared `--deadline` to each submitted campaign.

### Large-grid pilots

```bash
pff prepare --case 500 2000 --dataset "$PFF_ROOT/datasets/pilots" \
  --config configs/large-grid-pilots.yaml --workers 96 \
  --parity subset --parity-samples 200 --parity-seed 42 --parity-budget 20m \
  --backend pbs --queue debug --nodes 1 --walltime 1h --max-submissions 12

pff train --case 500 --seed 0 --samples 8 --max-epochs 6 --patience 3 \
  --dataset "$PFF_ROOT/datasets/pilots" --recipe v4-off \
  --name pilot500 --label pilot --chain-budget 12h --device xpu

pff train --case 2000 --seed 0 --samples 2 --max-epochs 6 --patience 3 \
  --dataset "$PFF_ROOT/datasets/pilots" --recipe v4-off \
  --name pilot2000 --label 'K=2 pilot' --chain-budget 12h --device xpu
```

The last two examples execute locally on an already allocated compute node. To place both in one PBS family, put their two resolved configurations in a campaign YAML and use `pff train --config configs/pilot-campaign.yaml --backend pbs --queue debug --nodes 1 --workers-per-node 2 --walltime 1h --max-submissions 12`. Per-case K belongs in those configurations, not a Cartesian sweep over both K values. The 12-hour training caps are explicit pilot budgets, distinct from fleet budgets and data-generation costs.

### Evaluation, summaries, and status

```bash
pff evaluate --runs "$PFF_ROOT/runs/fleet" "$PFF_ROOT/runs/ensemble" \
  --checkpoint best --partition test --samples 64 --eval-seed 42 \
  --ensemble-sizes 1 2 3 5 8 --shards 4 --device xpu \
  --backend pbs --queue debug-scaling --nodes 2 --workers-per-node 4 \
  --walltime 1h --max-submissions 1

pff report --runs "$PFF_ROOT/runs/fleet" "$PFF_ROOT/runs/ensemble" \
  --kind summary --format markdown json csv --output "$PFF_ROOT/reports/cross-grid"

pff report --runs "$PFF_ROOT/runs/fleet" "$PFF_ROOT/runs/ensemble" \
  --kind ensemble --format markdown png pdf --output "$PFF_ROOT/reports/mixtures"

pff report --runs "$PFF_ROOT/runs/fleet" --kind comparison \
  --reference "$PFF_ROOT/runs/historical-v4-on" \
  --output "$PFF_ROOT/reports/on-off"

pff report --runs "$PFF_ROOT/runs/fleet" --kind status
```

### Verification and controlled continuation

```bash
pff verify --check resume --protocol protocols/resume-on.json \
  --case 30 --fm genco --device xpu \
  --backend pbs --queue debug --nodes 1 --walltime 1h --max-submissions 1

pff train --resume "$PFF_ROOT/runs/ensemble" \
  --extend-budget 1h --name ensemble-final \
  --backend pbs --queue debug-scaling --nodes 2 --workers-per-node 5 \
  --walltime 1h --max-submissions 1
```

The latter authorizes one additional allocation and creates child runs only for eligible unfinished chains. It does not authorize another training retry or implicit evaluation job. Evaluation remains a separate explicit `pff evaluate` invocation.
