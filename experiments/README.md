# Reproducing the weekend experiments with pff

Install the package into the existing hardware-compatible environment first. Set
`PFF_ROOT` to the bulk storage root, `PFF_OPFDATA_ROOT` to the existing OPFData
corpus, and `PFF_JULIA` / `PFF_JULIA_DEPOT` to the existing solver installation. Explicit `--config` YAML takes precedence over
the packaged recipe; command flags take precedence over YAML. Use a new output
root for each campaign. Historical data, checkpoints, manifests and tags stay intact.
These examples reproduce the experiment definitions; they do not promise the same
stopping epoch, wall time or OFF-mode floating-point trajectory.

All five verbs accept `--config`. The YAML may contain the existing full scientific
configuration and a `cli.<verb>` mapping for command defaults. Evaluation origin
subsets live in `evaluation.origins` in that YAML; there is no extra origin-limit
CLI flag. `evaluation.baseline_root` may identify an immutable matching baseline
with saved ridge coefficients, climatology and embedding caches. The evaluator
checks its dataset and coordinate-scale hashes before reuse.

## Data

Case IDs are 14, 30, 57, 118, 500 and **2000**. The default data recipe is v4,
including the 0.80–1.00 profile law and train-derived coordinate scales. Point the
config at the selected grid's existing OPFData N1 cache/splits, pinned checkpoint,
normalizer, network, and Julia executable/depot. The command creates temporal
labels, not a replacement OPFData corpus or a torch environment.

```bash
pff prepare --config experiments/configs/aurora.yaml --case 30 --dataset "$PFF_ROOT/runs/cli/data" \
  --recipe v4 --reuse "$PFF_ROOT/data/temporal_case30_v4"
pff prepare --config experiments/configs/aurora.yaml --case 14 57 118 --dataset "$PFF_ROOT/runs/cli/data" \
  --recipe v4 --workers 32 --parity-samples 200 --device cpu
```

Fresh labels use persistent Ipopt workers and checksummed, atomic hour shards.
Re-running preparation resumes existing shards; it rejects changed provenance.
`--audit-only` requires existing labels and does not invoke label generation.
Reused labels are read-only. Each grid gets a full Stage A label audit, masked event
support, mechanically selected headline components, and a dataset card recording
`subset parity (200)`. Parity records the seeded IDs before comparing exact CPU
outputs. No fallback to an easier parity threshold is allowed.
When the config supplies recorded scales for the reused case, preparation checks
their training-shard hashes and preserves their exact bytes and coordinate order.

For a later independent audit, choose a new destination and reuse the old labels:

```bash
pff prepare --config experiments/configs/aurora.yaml --case 30 --dataset "$PFF_ROOT/runs/cli/data_reaudit" \
  --reuse "$PFF_ROOT/data/temporal_case30_v4" --audit-only
```

## Fleet and ensemble depth

`--case × --seed × --balance-weight` expands a campaign. A chain owns one full
adapter/backbone, optimizer and RNG stream. `--parallel N` runs independent chains
on distinct visible tiles; it is not sample-parallel training. With no parallel
flag the chains run sequentially. `--dry-run` records and prints the plan without
loading data or a model; execute that saved plan with `--resume`.

```bash
pff train --case 14 30 57 118 --dataset "$PFF_ROOT/runs/cli/data" --seed 0 1 2 \
  --recipe v4-off --samples 8 --lr 1e-4 --balance-weight 1 --patience 3 \
  --chain-budget 6h --device xpu --parallel 12 --out "$PFF_ROOT/runs/cli/fleet" --dry-run
pff train --resume "$PFF_ROOT/runs/cli/fleet"

pff train --case 14 30 57 118 --dataset "$PFF_ROOT/runs/cli/data" --seed 3 4 5 6 7 \
  --recipe v4-off --chain-budget 4.5h --patience 3 --device xpu --parallel 12 \
  --label ensemble-depth --out "$PFF_ROOT/runs/cli/ensemble"
```

The original fleet used two nodes with six tiles per node. `pff` has no node
placement backend. Launch two node-local invocations over disjoint chain paths
using the external PBS wrapper below; do not run the same chain in both.
`v4-on` selects the historical deterministic setting for an ON comparison.
`--samples` is the Monte Carlo count; covariance rank remains eight. Training
counts must divide the unchanged 64-sample selection ensemble (2/4/8/16/32/64).

## A0 and lambda sweep

```bash
pff train --case 30 57 118 --dataset "$PFF_ROOT/runs/cli/data" --seed 0 \
  --recipe v4-off --variant a0 --balance-weight 1 --chain-budget 3.3h \
  --label 'budget-capped A0' --device xpu --parallel 3 --out "$PFF_ROOT/runs/cli/a0"
pff train --case 30 --dataset "$PFF_ROOT/runs/cli/data" --seed 0 \
  --recipe v4-off --balance-weight 0.1 10 --chain-budget 3.3h \
  --label 'budget-capped lambda' --device xpu --parallel 2 --out "$PFF_ROOT/runs/cli/lambda"
```

The lambda=1 conditioned result already exists in the fleet. A0 uses the existing
no-latent-conditioning implementation, including its differentiable demand loss
and recomputed physics; this CLI introduces no alternative objective.

## Large-grid pilots

Use the recorded per-grid solver settings in `pilot500.yaml` and `pilot2000.yaml`
(or equivalent full saved configs): the existing repaired case files/reference
metadata, solver budgets and fixed base warm start must be preserved. For case2000,
the recorded policy uses 600 seconds and 3000 iterations per solve. Preparation
obtains PowerModels' parsed reference metadata when it is not already supplied.
The worker count is explicit and can be set to the available CPU allocation.

```bash
pff prepare --config experiments/configs/pilot500.yaml --case 500 --dataset "$PFF_ROOT/runs/cli/pilot_data" --workers 96
pff prepare --config experiments/configs/pilot2000.yaml --case 2000 --dataset "$PFF_ROOT/runs/cli/pilot_data_2000" --workers 96
pff train --case 500 --dataset "$PFF_ROOT/runs/cli/pilot_data" --seed 0 --recipe v4-off \
  --samples 8 --max-epochs 6 --patience 3 --chain-budget 6h --label 'K=8 pilot' \
  --device xpu --out "$PFF_ROOT/runs/cli/pilot500"
pff train --case 2000 --dataset "$PFF_ROOT/runs/cli/pilot_data_2000" --seed 0 --recipe v4-off \
  --samples 2 --max-epochs 6 --patience 3 --chain-budget 6h --label 'K=2 pilot' \
  --device xpu --out "$PFF_ROOT/runs/cli/pilot2000"
```

Existing labels may instead be supplied via `prepare --reuse`. Pilot identities,
K values and budget caps are retained in evaluation/report rows and are not
silently pooled with the ordinary fleet.

## Frozen evaluation, ensemble curves and reports

```bash
pff evaluate --config experiments/configs/aurora.yaml --runs "$PFF_ROOT/runs/cli/fleet" "$PFF_ROOT/runs/cli/ensemble" \
  --checkpoint best --partition test --modes C P P0 S A D --samples 64 --eval-seed 42 \
  --ensemble-sizes 1 2 3 5 8 --device xpu:0 --out "$PFF_ROOT/runs/cli/evaluation"
pff report --runs "$PFF_ROOT/runs/cli/evaluation" --kind summary --format md json csv \
  --out "$PFF_ROOT/runs/cli/summary"
pff report --runs "$PFF_ROOT/runs/cli/evaluation" --kind ensemble --format md json png \
  --out "$PFF_ROOT/runs/cli/ensemble_curve"
pff report --runs "$PFF_ROOT/runs/cli/fleet" --kind status --format md json \
  --out "$PFF_ROOT/runs/cli/status"
```

For independent origin shards, run the same evaluation command/output with
`--shards N --shard i`, for each `i` from zero through N−1. Preparation uses a file
lock and each shard owns disjoint origins. No distributed collectives are used.
The last completed shard can aggregate. Immutable inputs and a new output root are
required when changing checkpoints, partition, samples or evaluation seed.

A/D mixtures use ascending training-seed order within groups matching case, data,
scales, optimizer/variant, precision, K and evaluation protocol. Equal-size member
ensembles are concatenated and jointly rescored, including cross-member pairs.
C/P/P0/S remain shared baselines. An unavailable member count or coordinate stays
unavailable. Same-input P/Ipopt audit results are included in every completed run
report. A failed solve is masked, not relabelled infeasible.

Evaluate pilot or budget-limited snapshots explicitly with `--allow-budget-capped`:

```bash
pff evaluate --config experiments/configs/aurora.yaml --runs "$PFF_ROOT/runs/cli/a0" "$PFF_ROOT/runs/cli/lambda" \
  --checkpoint best --allow-budget-capped --device xpu:0 --out "$PFF_ROOT/runs/cli/ablations_eval"
pff evaluate --config experiments/configs/aurora.yaml --runs "$PFF_ROOT/runs/cli/pilot500" "$PFF_ROOT/runs/cli/pilot2000" \
  --checkpoint best --allow-budget-capped --device xpu:0 --out "$PFF_ROOT/runs/cli/pilots_eval"
pff report --runs "$PFF_ROOT/runs/cli/ablations_eval" --kind comparison \
  --reference "$PFF_ROOT/runs/cli/evaluation" --format md json csv --out "$PFF_ROOT/runs/cli/comparison"
```

Reports read persisted results only. They never start forecasts, training or solves.
A new checkpoint/evaluation requires a new output; historical manifests are not edited.

## Verification

Exactly two protocol templates ship under `power_fm_foresight/resources/protocols`.
Their recorded evidence paths refer to the weekend artifacts under `PFF_ROOT`.
Copy a template to adapt configuration/output locations before running; the gate
constants stay fixed. `regression` compares recorded checkpoint metrics at `1e-7`.
`reproducibility` trains case30 seed0 OFF for two epochs when no `--runs` is given,
using the existing training loop's stop-at-step facility to checkpoint immediately
before epoch-1 validation, then resumes through the normal command. With `--runs`,
it checks the supplied completed chain and its recorded boundary restore.

```bash
pff verify --check regression --protocol regression.json --out "$PFF_ROOT/runs/cli/verify_regression"
pff verify --check reproducibility --protocol reproducibility.json --out "$PFF_ROOT/runs/cli/verify_reproducibility"
```

Selection ES must be at most the maximum of the three recorded fleet seeds plus
0.005 at each epoch. All persisted metrics must be finite; model/optimizer/RNG,
counters and the committed metrics prefix must restore exactly. Loss curves are
reported at common steps and do not define a second acceptance threshold.

## Aurora wrapper

[`aurora/train_chain.pbs`](aurora/train_chain.pbs) calls `pff train --resume` once
per allocation. Submit a pinned source installation and an already planned campaign
or individual chain via `PFF_PYTHON` and `PFF_RESUME`. It neither edits source nor
submits another job. The external [`aurora/chain.sh`](aurora/chain.sh) submits bounded
one-hour allocations and waits for each result. It is outside the package/wheel.
For two nodes, use separate disjoint node-local chain lists from the printed plan.
A launcher-isolated Aurora tile has local index zero; otherwise flat tile indices
are explicit. Keep the same visible-device mapping when resuming.
