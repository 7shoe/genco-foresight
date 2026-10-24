# Installation on Aurora

These steps reproduce the installation verified on Aurora with Python 3.12.12, vendor Torch `2.13.0a0+gitcf30153`, PyG `2.8.0.post1`, and existing `pytorch-lightning==2.6.6`. Preserve the installed Torch, Transformers, Torchvision, NumPy, SciPy, and pandas versions. Every installation uses `--no-deps` to prevent automatic dependency changes; the explicitly named packages should be absent or already at the versions below. Run the commands in the same shell with the existing Aurora framework environment loaded.

1. Activate the intended environment and save its package inventory. Preserve any earlier inventory before repeating this step.

   ```bash
   source /lus/flare/projects/FRAME-IDP/siebenschuh/envs/reason_venv_2026.1.0/bin/activate
   python -m pip --version
   python -m pip freeze > /tmp/genco-install-before.txt
   ```

2. Install Graphkit from the verified commit (`0.9.0.3`). Temporarily unset `PIP_CONSTRAINT` for this command because the build requires `setuptools>=83`, while Aurora constrains the installed version to `79.0.1`. Keep build isolation enabled: the newer build tools stay in the temporary build environment.

   ```bash
   env -u PIP_CONSTRAINT python -m pip install --no-deps \
       "gridfm-graphkit @ git+https://github.com/gridfm/gridfm-graphkit.git@f104900245cba26d1380037eea562cf633a97484"
   ```

3. Build `torch-scatter` against the installed vendor Torch. This source build succeeded on the same machine; no shim was needed for the verified CPU operations.

   ```bash
   MAX_JOBS=2 python -m pip install \
       --no-deps \
       --no-build-isolation \
       --no-binary=torch-scatter \
       "torch-scatter==2.1.2"
   ```

4. Install the remaining packages at the versions observed in the successful environment.

   ```bash
   python -m pip install \
       --no-deps \
       --only-binary=:all: \
       "gridfm-datakit==1.1.0" \
       "lightning==2.6.6" \
       "seaborn==0.13.2" \
       "opt-einsum==3.4.0" \
       "plotly==7.1.0" \
       "juliapkg==0.1.24" \
       "semver==3.1.0" \
       "tomli==2.5.0" \
       "tomlkit==0.15.2"
   ```

5. Check the GENCO import and CPU scatter operation, including its gradient.

   ```bash
   python - <<'PY'
   from importlib.metadata import version
   import torch
   import torch_scatter
   from gridfm_graphkit.models.gnn_heterogeneous_gns import GNS_heterogeneous

   for name in (
       "torch", "transformers", "torchvision", "torch-geometric",
       "numpy", "scipy", "pandas", "setuptools",
   ):
       print(f"{name}: {version(name)}", flush=True)
   print("GENCO model import: OK", flush=True)

   x = torch.tensor([1., 2., 3.], requires_grad=True)
   index = torch.tensor([0, 1, 0])
   y = torch_scatter.scatter_add(x, index, dim=0, dim_size=3)
   torch.testing.assert_close(y, torch.tensor([4., 2., 0.]))
   y.sum().backward()
   torch.testing.assert_close(x.grad, torch.ones_like(x))
   print("CPU scatter_add forward/backward: OK", flush=True)
   PY

   python -m gridfm_graphkit --help
   ```

   The CLI should list `train`, `finetune`, `evaluate`, `predict`, and `benchmark`. An XPU device-count warning is expected on a login node.

6. Compare the environment inventories.

   ```bash
   python -m pip freeze > /tmp/genco-install-after.txt
   diff -u /tmp/genco-install-before.txt /tmp/genco-install-after.txt
   ```

   On the verified installation, the diff contained only these 11 additions: Graphkit, Datakit, torch-scatter, lightning, seaborn, opt-einsum, plotly, juliapkg, semver, tomli, and tomlkit. No existing package version changed. `diff` exits with status 1 when it finds differences; inspect the output for replacements or removals.

This reproduces the verified model-import and CLI-help environment. Some declared dependencies remain absent, including MLflow, torchaudio, and JuliaCall; add dependencies only when a required execution path needs them. Python 3.12 already provides `pathlib`. Checkpoint loading, official evaluation, Julia/PowerModels setup, and XPU execution require separate verification; XPU tests must run on a compute node.

Stage A subsequently needed MLflow for the official evaluator. Keep that optional
CLI dependency separate from the existing environment:

```bash
python -m pip install --no-deps --only-binary=:all: \
    --target /lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/assets/cli_deps \
    -r requirements-cli.txt
```

Use an empty target or one already containing these exact pins. The five additions
are MLflow, MLflow Skinny and MLflow Tracing `3.17.2`, Databricks SDK `0.95.0`, and
sqlparse `0.5.5`. `scripts/parity.py` exposes that directory only to the official
evaluator subprocess and sets `MLFLOW_ALLOW_FILE_STORE=true`. The base environment
freeze and this isolated dependency inventory are recorded separately. This is
the tested evaluator path; it does not install every optional MLflow service
dependency. Stage A results are described in the repository README and report.
