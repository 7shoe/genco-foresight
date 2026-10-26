# Phase 2: importable package

Status: implementation prepared; compute gates pending. No merge or new tag yet.

The scientific implementations move to `power_fm_foresight` version 0.2.0.
The public modules are `grids`, `genco`, `temporal`, `scoring`, `solver`, `data`,
and `training`. Private modules preserve original globals and function boundaries.
The A0 flag, numerical operations, RNG ordering, masks and resume rejection remain.

[Protocol](phase2.protocol.json) fixes the tests, origins, tolerances and resource cap
before verification. Bulk evidence: `/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/package/phase2_20261026_v1`.
Maximum: two debug allocations, one node and one hour each; no training.

## Moves

| Old path | New path |
| --- | --- |
| `common.py` | `src/power_fm_foresight/_config.py` |
| `data/grids.py` | `src/power_fm_foresight/grids.py` |
| `models/genco.py` | `src/power_fm_foresight/genco.py` |
| `models/temporal.py` | `src/power_fm_foresight/temporal.py` |
| `scoring.py` | `src/power_fm_foresight/scoring.py` |
| `data/coordinates.py` | `src/power_fm_foresight/_data/coordinates.py` |
| `data/loader/network.py` | `src/power_fm_foresight/_data/network.py` |
| `data/loader/dataset.py` | `src/power_fm_foresight/_data/dataset.py` |
| `data/download/prepare.py` | `src/power_fm_foresight/_data/assets.py` |
| `data/synthetic/generate.py` | `experiments/data_generation.py` |
| `training/loop.py` | `src/power_fm_foresight/_training/loop.py` |
| `training/state.py` | `src/power_fm_foresight/_training/state.py` |
| `training/inputs.py` | `src/power_fm_foresight/_training/inputs.py` |
| `training/metrics.py` | `src/power_fm_foresight/_training/metrics.py` |
| `training/forecaster.py` | `src/power_fm_foresight/_training/forecaster.py` |
| `solver/adapter.py` | `src/power_fm_foresight/_solver/adapter.py` |
| `solver/checks.py` | `src/power_fm_foresight/_solver/checks.py` |
| `scripts/**` | `experiments/**` (same relative driver names) |
| `solver/julia/**` | `src/power_fm_foresight/resources/julia/**` (pinned bytes copied) |
| `data/networks/**` | `src/power_fm_foresight/resources/networks/**` (case30 and license copied) |

The exact driver map and one-line shim list are in [moves.json](moves.json).
Shims emit `DeprecationWarning` and alias module identity, including private names.
Legacy executable module paths dispatch to their experiment driver. These aliases
remain in the checkout; the wheel contains only the new package. Old PBS paths are
symlinks to the relocated scripts.

### Extracted definitions

| Original | Packaged definition |
| --- | --- |
| `data/loader/network.py:GridSpec` | `src/power_fm_foresight/grids.py` |
| `data/synthetic/generate.py:episode_plan` | `src/power_fm_foresight/_data/generate.py` |
| `data/synthetic/generate.py:demands` | `src/power_fm_foresight/_data/generate.py` |
| `scripts/evaluate_stage_b.py:physics` | `src/power_fm_foresight/_training/physics.py` |
| `scripts/evaluate_stage_b.py:qd_from_pd` | `src/power_fm_foresight/_training/physics.py` |
| `scripts/evaluate_stage_b.py:frozen` | `src/power_fm_foresight/_training/physics.py` |
| `scripts/evaluate_stage_b.py:target_values` | `src/power_fm_foresight/_training/physics.py` |
| `scripts/forecast_partition.py:components` | `src/power_fm_foresight/_training/forecast.py` |
| `scripts/forecast_partition.py:adapter_samples` | `src/power_fm_foresight/_training/forecast.py` |
| `scripts/forecast_partition.py:cache_partition` | `src/power_fm_foresight/_training/forecast.py` |
| `scripts/fleet/evaluate.py:score` | `src/power_fm_foresight/scoring.py` |
| `scripts/replication_evaluate.py:merged` | `src/power_fm_foresight/scoring.py` |

### Packaging adjustments

Configuration interpolation is unchanged. Source checkouts retain the legacy v3
config default and explicit configs. Packaged v3/v4 recipes take their storage
and Julia paths from environment/configuration. Registry resource entries include
numeric case IDs and relative asset templates. Solver and case30 network resources
are located relative to the package. New manifests hash the moved implementation;
no historical manifests are edited or made compatible with new training resumes.
The wheel adds a dependency-stack import diagnostic and no CLI entry points.
The mixture helper contains the existing concatenation expression unchanged.

An AST comparison found 439 top-level definitions unchanged after ignoring imports.
Nine reviewed exceptions cover configuration/resource paths, manifest source paths,
the extracted mixture call and an import inside a subprocess source string.
The full case30 regression and saved scorer comparisons remain required evidence.

## Gates

Pending: full 140-test/3-subtest suite; 50-hour audit and 2-origin forward regression;
3-origin released/fleet scorer and loader equivalence; fresh editable installations
with and without the existing torch stack; wheel contents and static path/import checks.

## Limitations

CUDA and CPU wheel install recipes are documented from upstream matrices; this gate
uses Aurora's vendor stack. The fresh positive environment reuses that stack read-only.
Julia and model/data assets remain external prerequisites. Historical experimental
controllers still enforce their original snapshot/hash rules; packaging does not
make historical training checkpoints resumable under changed source hashes.
