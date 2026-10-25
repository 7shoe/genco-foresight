# Small-grid OFF fleet

Submitted **8963092** to `debug-scaling`, two nodes, maximum one hour. This is the first allocation of a persistent resubmission chain. Runtime status is [runs/fleet/STATUS.md](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/fleet/STATUS.md); the [controller ledger](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/fleet_20261025_off_v1/controller.json) records subsequent job IDs and boundaries. Training and evaluation results are pending.

## Frozen implementation and recipe

Development worktree: `/home/siebenschuh/Projects/genco-foresight-fleet`, branch `multigrid-fleet`, based on `multigrid`. Implementation commit `1fec6e805c0eb2203c71df24ffbca3cdf426ca79`. Jobs execute only the [pinned snapshot](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/fleet_20261025_off_v1/source), with file and configuration hashes in [source_pin.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/fleet_20261025_off_v1/source_pin.json), SHA-256 `be99f936e092ae03f3529dc6aba30037cf64cf9ecb92d7606781699ea15b7b70`. Changes to development do not enter this snapshot. The original working tree/index and confirmation job 8962843 were left untouched.

The [protocol](fleet.protocol.json) was recorded before submission; SHA-256 `e7b79bbcc777f6fbb49c7497ccecb0fcaf419e67d2a280c29f8e2020d166f076`. All twelve resolved configurations were checked against v4 for the generation law, optimizer, loss, K, B, model, fair ES, patience and checkpoint cadence. Training RNG seeds are literally **0, 1, 2**, each fresh, for case14/30/57/118. They do not map seed0 to the historical seed42 run. Data seed remains 42. Production determinism is OFF; evaluation remains ON. No packages were installed or upgraded.

One independent process per tile, six per node: flat tiles 0/2/4/6/8/10, externally isolated and addressed as `xpu:0`. Node 0 hosts case14 and case30; node 1 hosts case57 and case118. There is no sample/gradient distribution between models. This fleet is separate from Stage 2 implementation.

## Data gates

case30 reuses the released v4 data, scales and embedding cache read-only by hash, with copies of the audit evidence under the new run. The other grids generate their own 10,080-hour v4-law datasets, retaining the 200-hour pilot prefix. Their chains wait for CPU checkpoint parity, independent conversion, numerical/dual checks, full Stage A audit, event support, scales and embedding-cache completion. Each grid can start as soon as its own gates pass.

Headline buses are selected from risk-calibration support alone: at least five positive days, at least five negative days, and hourly prevalence below 0.95. All eligible components are retained; empty sets remain N/A. The existing case30 v4 event table selects **bus 12** mechanically for this fleet, unlike the historical fixed headline set. Historical reports remain unchanged. Event prevalence is not itself a training blocker.

## Continuation, evaluation and costs

Each chain gets at most 21,600 cumulative training worker seconds, including startup, selection and checkpointing. It stops on patience 3 or that budget. Best checkpoints are selected by ES on the selection partition with 64 fixed-noise samples. Checkpoints are atomic every epoch and every 900 seconds; configurations, RNG, optimizer state and order are restored exactly. Unexpected worker failures park the affected chain/grid; other chains continue. Missing boundaries are charged conservatively.

Data preparation is separately capped at three hours per grid, evaluation at six hours per chain, with a fleet ceiling of 24 two-node one-hour allocations (48 reserved node-hours maximum). These are ceilings, not promised costs or target durations. The controller submits only while work remains and respects PBS limits without waiting for unrelated job families.

As each chain finishes, its slot runs C/P/P0/S/A/D evaluation using the unchanged scorer and numerical audit helpers through a grid-aware replication wrapper. All 72 test origins are included with evaluation seed 42. case30 baselines are copied by hash; other grids use the same baseline recipes. A saved-output scorer reproduction precedes evaluation. Completed origins have atomic hash records and are reused after allocation boundaries. Missing historical covariates or unresolved baseline requirements are reported as failures, never silently imputed by a changed method.

The [cross-grid table](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/fleet_20261025_off_v1/cross_grid.md) will contain mean ± sample SD over all three seeds, all six modes, and the equal-weight 192-draw A/D mixture rescored jointly. Fixed baselines remain the same distributions across seeds. Different grid scales and eligible coordinates limit cross-grid comparisons. A budget-limited stopping epoch is censored; per-chain stop reasons and epochs remain in `adapter/status.json`.

## Verification and current limitations

Five lightweight controller tests passed: placement/seeds, per-grid readiness, budget/evaluation scheduling, mechanical headline selection and rejection of changed pinned inputs. The first allocation runs the broader case-identity, multigrid and V2 harness/controller tests on compute before numerical work. Those results and production results are pending. The initial local test invocation needed its external temporary parent directory created; the corrected invocation passed all five tests.

No main merge or publication has occurred. Existing user edits, staged inventory and release tags were preserved. This report records submission and the predeclared workflow, not a training/evaluation pass.
