Project configs use OmegaConf interpolation to share a storage prefix:

```yaml
base: /lus/flare/projects/FRAME-IDP/siebenschuh
paths:
  opfdata_root: ${base}/acopf_datasets
  n1_cache: ${base}/acopf_datasets/processed/N1/pglib_opf_case30_ieee
  assets_root: ${base}/GENCO/assets
  reference_solver_checkout: /home/siebenschuh/Projects/acopf
```

Edit `base` in the selected file to relocate dependent paths. 
`common.load_config()` selects an explicit filename first, then `GENCO_CONFIG`,
then `configs/v3.yaml`.

```python
from common import load_config

cfg = load_config("configs/v3.yaml", overrides={"base": "/lus/flare/projects/example/user"})
```

## v1, v2: initial experiments
Legacy; not included.

## v3: primary experiment
```bash
configs/v3.yaml
```
Case30 temporal data with nominal load scaling 0.30–1.00 and balance-penalty weight 0.01. Defines data generation, training and evaluation settings.

## v4: follow-up experiment
```bash
configs/v4.yaml
```

configs/v3.yaml — primary experiment. 
  • configs/v4.yaml — . Same architecture and partitions, but load scaling 0.80–1.00 and balance weight 1.0. Tests a narrower load regime with stronger physics regularization.
    These are two simultaneous changes, so it is not an isolated ablation.

## `configs/replication/`
Repeats the v3/v4 recipes with seeds 1 and 2, using separate output directories. family.json controls their PBS scheduling, budgets and reporting. 
Currently covers case30 only which should be sufficent as a proof of concept.

## `configs/multigrid/`

Each `grid: case118_ieee` selector expands the v4 recipe through `configs/grids.yaml`.
The selected network determines checkpoint paths and dimensions; generated artifacts
go to `GENCO/runs/multigrid/<case>/` under the configured base. Selectors support
`load_config(path, overrides={"base": "/lus/flare/projects/example/user"})` and
ordinary nested overrides. Interpolation resolves after overrides, and the returned
configuration has no helper `base` key.

Loading rejects disagreement between the configured case, parsed network name,
snapshot manifest, and checkpoint YAML. Temporal manifests must match the network's
`case_sha256`; existing v3/v4 manifests work unchanged. Graphs produced by the data
loaders carry `case_id`, which GENCO checks before preprocessing; unlabelled graphs
are rejected. These checks run before inference or training and do not change weights.
