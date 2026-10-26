# Package `power-fm-foresight`

<img align="left" width="50%" src="https://raw.githubusercontent.com/7shoe/lumina-sdk/main/assets/graph_predict.png" alt="GENCO Foresight prediction graph">

<p>• <b>MODEL</b>: Extends pre-trained GENCO from AC-OPF solver to a probabilistic forecaster</p>
<br>
<p>• <b>DATA</b>: Temporal dimension from EIA ERCOT 2025, operating-stress calibration for voltage and line events</p>
<br>
<p>• <b>TRAINING</b>: Synthetic temporal dataset: 10,080 Ipopt-labeled hours driven by EIA ERCOT 2025 demand</p>
<br>
<p>• <b>OPTIMIZER</b> Anticipatory solver warm starts evaluated against cold starts (no solver benefit observed at 2.93× cost)</p>
<br>
<p>• <b>SYSTEM</b> Reproducible on Aurora (XPU) HPC and NVIDIA (CUDA) systems, with CPU fallback</p>

<br clear="all">

The v0.2.0 tag collects all weekend experiments' code and the [run index](RUNS.md).
The [final consolidation gate](reports/v2/final_consolidation.md) passed the full
test suite and the existing case30 regression. Failed and budget-capped experiments
retain their recorded outcomes; this tag does not assert that every training gate passed.

## Motivation 

## Method

## Empirical Highlights

## Conclusion

## References
Below, the key references.

[GridFM graphkit](https://gridfm.github.io/gridfm-graphkit/) - A library to train, finetune and interact with a foundation model for the electric power grid. (Puech, et al., 2025). 
```bash
@article{puech2025gridfmdatakitv1pythonlibraryscalable,
  title={gridfm-datakit-v1: A Python Library for Scalable and Realistic Power Flow and Optimal Power Flow Data Generation},
  author={Alban Puech and Matteo Mazzonelli and Celia Cintas and Tamara R. Govindasamy and Mangaliso Mngomezulu and Jonas Weiss and Matteo Baù and Anna Varbella and François Mirallès and Kibaek Kim and Le Xie and Hendrik F. Hamann and Etienne Vos and Thomas Brunschwiler},
  journal={arXiv preprint arXiv:2512.14658},
  year={2025},
  url={https://arxiv.org/abs/2512.14658}
}
```

[GENCO](https://arxiv.org/abs/2608.09921) - A Unified Neural Solver Embedded in a Development Framework for Steady-State Grid Analysis; (<u>GE</u>ometric <u>N</u>eural <u>C</u>orrective <u>O</u>ptimizer) (Puech et. al, 2026)

```bash
@article{puech2026gencounifiedneural,
  title={GENCO - A Unified Neural Solver Embedded in a Development Framework for Steady-State Grid Analysis},
  author={Alban Puech and Matteo Mazzonelli and Tamara R. Govindasamy and Mangaliso Mngomezulu and Héctor Maeso-García and Thomas Tolhurst and Javad Bayazi and Ali Moeini and Naomi Simumba and Celia Cintas and David Nelischer and Romeo Kienzler and Jonas Weiss and Anna Varbella and Florian Dörfler and Gabriela Hug and Martin Mevissen and Juan Bernabé-Moreno and François Mirallès and Hendrik F. Hamann and Etienne Vos and Thomas Brunschwiler},
  journal={arXiv preprint arXiv:2608.09921},
  year={2026},
  url={https://arxiv.org/abs/2608.09921}
}
```

[IPOPT](https://cepac.cheme.cmu.edu/pasilectures/biegler/ipopt.pdf) - On the implementation of an interior-point filter line-search
algorithm for large-scale nonlinear programming (Wächter, Biegler, 2005)

```bash
@article{wachter2006implementation,
  title={On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming},
  author={W{\"a}chter, Andreas and Biegler, Lorenz T},
  journal={Mathematical programming},
  volume={106},
  number={1},
  pages={25--57},
  year={2006},
  publisher={Springer}
}
```

[PowerModels.jl](https://arxiv.org/abs/1711.01728) - An Open-Source Framework for Exploring Power Flow Formulations (Coffrin, et al., 2017)
```bash
@inproceedings{coffrin2018powermodels,
  title={Powermodels. jl: An open-source framework for exploring power flow formulations},
  author={Coffrin, Carleton and Bent, Russell and Sundar, Kaarthik and Ng, Yeesian and Lubin, Miles},
  booktitle={2018 Power Systems Computation Conference (PSCC)},
  pages={1--8},
  year={2018},
  organization={IEEE}
}
```

[ERCO: EIA's Hourly Electric Grid Monitor](https://www.eia.gov/electricity/gridmonitor/dashboard/electric_overview/balancing_authority/ERCO) from the Electric Reliability Council of Texas, Inc. (ERCO)

```bash
@misc{eia_erco,
  author = {{U.S. Energy Information Administration}},
  title  = {Hourly Electric Grid Monitor: Electric Reliability Council of Texas ({ERCO})},
  url    = {https://www.eia.gov/electricity/gridmonitor/dashboard/electric_overview/balancing_authority/ERCO},
  note   = {Accessed: 2026-10-24}
}
```

## V2 production policy

Determinism OFF is **adopted with disclosure** for new V2 production runs; use
`configs/v2/production.yaml`. Historical v3/v4 configurations and manifests retain
ON, as do evaluation and resume-mechanism checks. The case30 OFF selection-ES mean
is 0.290849888 versus 0.285553075 for historical ON seeds (Δ = +0.005296813).
Mean test ES is 0.307292007 versus 0.300798303 (Δ = +0.006493704); the three-seed
comparison does not establish statistical equivalence, and some voltage metrics
also differ. Two OFF seeds exceed the historical selected-ES range. This operational
adoption does not turn the earlier failed gates into passes. See the complete
[Stage 1 verdict](reports/v2/stage1_verdict.md), including seed identities, stopping
reasons, all test metrics and the separate confirmation-job exceedance.

## Multigrid CPU format support

The V2 A0 and λ ablations use separate run identities and a 3.3-hour worker cap.
Apply `configs/v2/a0.yaml`, `lambda01.yaml`, or `lambda10.yaml` as overrides through
`load_config`; they are not standalone complete configurations. A0 freezes latent
conditioning and runs GENCO under `no_grad`, then adds final-state innovations and
recomputes differentiable physics. Its demand head still receives direct demand-ES
gradients. The conditioned adapter remains the production model. The frozen GENCO
path and the default adapter forward were checked before these experiments.

This repository supports case14, case30, case57, case118, case500, and case2000.
Select a case with `GENCO_CONFIG=configs/multigrid/case118_ieee.yaml`; the selector
uses the v4 recipe and separate output paths. `GridSpec` supplies dimensions,
reference buses, generator mappings, and offline masks.

Run `python -m scripts.multigrid_smoke` for CPU loading and formatting checks, then
`python -m scripts.multigrid_report` to refresh [the support report](reports/multigrid/support.md).
These checks launch no solver, training, or PBS job. Bulk artifacts stay on `/lus/flare`;
the existing Torch/Transformers environment is retained.

## Python package

The distribution is `power-fm-foresight` (development version 0.3.0.dev0); import it as
`power_fm_foresight`. The Python API is also exposed by the five `pff` verbs below.
`genco`, `temporal`, `grids`, `data`, `solver`, `scoring`, and `training` expose
existing implementations. Private submodules retain the original module boundaries.
Experiment drivers, PBS scripts, gates and controllers live in `experiments/`
and are excluded from the wheel. Historical module paths remain deprecated aliases.
Install the package before using those paths from a checkout.

### Install matrix

Provision the torch stack first. Torch, PyG, scatter and Graphkit are deliberately
absent from `install_requires`; `[cuda]` and `[xpu]` are empty selection markers.
The commands below are pinned installation recipes. The packaging gate runs on
Aurora; CUDA and CPU wheel compatibility is documented, not hardware-validated here.
See the [PyTorch version matrix](https://pytorch.org/get-started/previous-versions/)
and [PyG wheel matrix](https://pytorch-geometric.readthedocs.io/en/stable/notes/installation.html).

| Target | Python | Torch stack | Installation policy |
| --- | --- | --- | --- |
| CUDA 12.8 | 3.12 | Torch 2.10.0, PyG 2.8.0, scatter 2.1.2 | Fresh environment, commands below |
| CPU | 3.12 | Torch 2.10.0, PyG 2.8.0, scatter 2.1.2 | CPU wheels, commands below |
| Aurora XPU | vendor 3.12.12 | Torch 2.13.0a0+gitcf30153, PyG 2.8.0.post1, scatter 2.1.2 | Preserve frameworks/2026.1.0 and base constraints |

CUDA, optionally after `conda env create -f environment-cuda.yml` and
`conda activate power-fm-foresight-cuda`:

```bash
python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install torch-geometric==2.8.0
python -m pip install --no-deps torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.10.0+cu128.html
```

CPU in a fresh Python 3.12 environment:

```bash
python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install torch-geometric==2.8.0
python -m pip install --no-deps torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.10.0+cpu.html
```

For either fresh environment, install the existing GENCO runtime prerequisites
and Graphkit without allowing it to replace the chosen torch stack:

```bash
python -m pip install lightning==2.6.6 opt-einsum==3.4.0 seaborn==0.13.2 plotly==7.1.0
python -m pip install --no-deps "gridfm-graphkit @ git+https://github.com/gridfm/gridfm-graphkit.git@f104900245cba26d1380037eea562cf633a97484"
python -m pip install -e .
```

Aurora uses the existing frameworks modules and environment. Do not install a
PyPI torch wheel or upgrade vendor torch, Transformers or their dependencies.
The installed PyG is a vendor build, not a request to fetch that version from PyPI.
For a new Aurora environment, follow [the recorded installation](install_on_aurora.md)
and preserve its `base-constraints.txt`/`PIP_CONSTRAINT` policy:

```bash
# Only if absent, use the recorded vendor PyG source/build, not a CUDA wheel.
# Scatter is built against the active vendor torch:
MAX_JOBS=2 python -m pip install --no-deps --no-build-isolation --no-binary=torch-scatter torch-scatter==2.1.2
# Graphkit needs newer isolated build tools; the runtime environment stays constrained.
env -u PIP_CONSTRAINT python -m pip install --no-deps "gridfm-graphkit @ git+https://github.com/gridfm/gridfm-graphkit.git@f104900245cba26d1380037eea562cf633a97484"
python -m pip install --no-deps --no-build-isolation -e .
```

The Aurora module setup is recorded in `experiments/package/verify.pbs`.
These commands are instructions for provisioning, not permission to replace
packages in a working vendor environment. Existing runtime dependencies,
including OmegaConf, Datakit, matpowercaseframes and requests, are declared in
`pyproject.toml`. Graphkit's optional CLI/evaluator extras remain separate.
Ipopt uses the packaged pinned Julia project and an explicitly configured Julia
executable/depot; installing the wheel does not install Julia or download assets.

Packaged recipes use `PFF_ROOT` as the bulk storage root and `PFF_JULIA` and
`PFF_JULIA_DEPOT` for the solver. Explicit legacy configs retain their original
`${base}` interpolation and paths. With a wheel, the default remains the v3 recipe;
choose the packaged v4 recipe explicitly:

```python
from importlib.resources import files
from power_fm_foresight.data import load_config
from power_fm_foresight.grids import grid_config, registry

recipe = files("power_fm_foresight").joinpath("resources/v4.yaml")
config = load_config(recipe)
case = registry()["case_ids"][30]
config = grid_config(case, config)
```

Set `PFF_ROOT`, `PFF_JULIA` and `PFF_JULIA_DEPOT` before resolving a packaged
recipe, or supply explicit path overrides. Package import itself does not need
these variables. New manifests hash the relocated source; historical manifests
are never rewritten and incompatible training resumes remain rejected.

## pff CLI (0.3.0.dev0)

`pff` and `power-fm-foresight` invoke the same five argparse commands:
`prepare`, `train`, `evaluate`, `report`, and `verify`. Run `pff VERB --help` for the
supported flags. Configuration precedence is command flags, then `--config` YAML,
then the recipe. Each command records its effective configuration, package/source
identity, device, seeds and paths. The package has no PBS submission backend.

[Weekend reproduction commands](experiments/README.md) cover the fleet, deeper
ensembles, A0, lambda sweep and pilots, plus the external Aurora wrapper. Reports
use persisted results only. CLI verification is documented in
[the Phase 3 report](reports/package/phase3.md). This development phase does not
publish a wheel to TestPyPI or PyPI.
