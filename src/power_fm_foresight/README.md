# Using `power_fm_foresight`

Install the package with the dependencies in the [installation guide](../../README.md#install-matrix).
Import `power_fm_foresight`; `src/` is only the repository layout.

## Python modules

| Module | Purpose |
| --- | --- |
| `grids` | Case registry, grid configuration and `GridSpec` |
| `genco` | Frozen GENCO checkpoint wrapper |
| `temporal` | Case-specific temporal adapter, including A0 |
| `data` | Configuration, datasets, windows and coordinate scales |
| `solver` | Ipopt integration and physics checks |
| `scoring` | Forecast scores and diagnostics |
| `training` | Training, checkpoints, RNG restoration and forecast mixtures |
| `cli` | Prepare, train, evaluate, report and verify workflows |

Load your configuration and select a case:

```python
from power_fm_foresight.data import load_config
from power_fm_foresight.grids import grid_config, registry

print(registry()["case_ids"])  # 14, 30, 57, 118, 500, 2000
cfg = grid_config("case30_ieee", load_config("my_config.yaml"))
```

## Command-line workflows

Use `pff COMMAND --help` for arguments. For example, with an already prepared
and audited dataset and a configured accelerator environment:

```bash
pff train --case 30 --dataset /path/to/prepared --seed 0 --recipe v4-off \
  --device xpu --device-index 0 --out /path/to/run
pff evaluate --runs /path/to/run --device xpu:0 --out /path/to/evaluation
pff report --runs /path/to/evaluation --kind summary --out /path/to/report
```

Set `PFF_ROOT` to your bulk storage root; supply explicit configuration for other
paths. Each case requires its matching GENCO checkpoint, normalizer and data
scales. Historical checkpoints can be evaluated; training resume requires
compatible source and full state. See the [main README](../../README.md) for
preparation, hardware setup and validation limits.
