# Phase 2: importable package

Status: **PASS**. Reviewed code commit `5b0093d5285bf9f4fc5e7e5717e87b01469ae525`; merged to main and tagged `v0.2.1-dev` locally. Distribution version remains 0.2.0. No CLI added.

The scientific implementations move to `power_fm_foresight` version 0.2.0.
The public modules are `grids`, `genco`, `temporal`, `scoring`, `solver`, `data`,
and `training`. Private modules preserve original globals and function boundaries.
The A0 flag, numerical operations, RNG ordering, masks and resume rejection remain.

[Protocol](phase2.protocol.json) fixes the tests, origins, tolerances and resource cap
before verification. Bulk evidence: [final gate](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/package/phase2_20261026_v1_final/gate.json), [first gate](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/package/phase2_20261026_v1/gate.json).
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
The old asset command dispatches to `experiments/assets.py`; its reusable functions
live in `_data/assets.py`. Scientific code never imports the command wrapper.
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
The compute gates below verify the case30 regression and saved scorer comparisons.

## Gates

| Gate | Result |
| --- | --- |
| Full suite | 140 passed + 3 subtests; zero failures/skips |
| case30 v4 audit | 50 hours passed; maximum difference from recorded audit `0.0` |
| adapter_v4 forward | Two origins, A/D, 64 trajectories; maximum array difference `0.0` (bit-identical) |
| Package loader/scorer | Released adapter_v4 and fleet case30 seed0, three origins each, A/D: 12 comparisons; maximum absolute difference `3.5527136788005009e-15`, threshold `1e-7` |
| Fresh editable install | Import succeeded with the existing stack; isolated environment without torch failed with the README install-matrix pointer |
| Wheel | Only the new package and its resources; no experiment drivers or torch/PyG/scatter requirements |
| Static checks | No `/lus/flare` strings under src; no package imports of experiments; 439 unchanged definitions |

Both allocations passed. The second verifies the final public scorer relocation and
the last direct-package test imports. Same predeclared protocol, inputs and tolerances;
no acceptance thresholds changed. The first used `e3ab375`, the final used `5b0093d`.

| Job | Installation | Full suite | Regression | Three-origin scorer | Total worker |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8966222 | 89.01 s | 26.70 s | 58.55 s | 23.29 s | 200.16 s |
| 8966236 | 44.49 s | 20.42 s | 30.40 s | 22.09 s | 119.51 s |

Both jobs used debug, one node, one isolated XPU tile, and a one-hour wall limit.
Reserved cumulative budget: 7,200 node-seconds (the declared maximum); measured
verification worker time: 319.67 seconds. No training ran.
The positive fresh venv references the vendor site-packages read-only; the negative
venv contains only build tools and this package. No vendor packages were replaced.

The [machine-readable gate](phase2.json) records evidence paths and SHA256 hashes,
including [source pin](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/package/phase2_20261026_v1_final/source_pin.json), [AST audit](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/package/phase2_20261026_v1_final/ast.json),
[reference hashes](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/package/phase2_20261026_v1_final/reference_hashes.json), and [budget ledger](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/package/phase2_20261026_v1/budget.json).
The protocol SHA256 is `10daabc5d474e5659c046f333546e1e985edb622104240c14bddd0b1dfd8a47c`.


## Limitations

CUDA and CPU wheel install recipes are documented from upstream matrices; this gate
uses Aurora's vendor stack. The fresh positive environment reuses that stack read-only.
Julia and model/data assets remain external prerequisites. Historical experimental
controllers still enforce their original snapshot/hash rules; packaging does not
make historical training checkpoints resumable under changed source hashes.
