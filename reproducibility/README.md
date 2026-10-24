# Local reproducibility package

This directory documents the local review package. Nothing publishes or changes
an environment. The owner controls all later pushes and release uploads.
The source commit/tag `v0.1.0` is prepared locally; the evidence directory keeps
its review label `v0.1.0-rc`.

Package: `/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/reproducibility/v0.1.0-rc/`.
All bulk data remain on Lustre. Copy the complete directory to another machine
when ready; runnable paths are relative. Historical manifests preserve original
Aurora paths as provenance. Later hosting can be entirely on GitHub Release assets.

The release commit is prepared after evaluation. Scientific source files are
checked against hashes saved by the original runs; the experiments were not
launched from this later tag. Reporting and packaging tools are new release work.

## Contents

- `code/`: source snapshot, unchanged scientific model/scorer/evaluation files,
  final post, headline-key list, and content-hashed figure files.
- `datasets/`: complete v3/v4 episodes, manifests, audits and draft dataset card;
  v1/v2 manifests and audit summaries preserve the revision history.
- `model/adapter_v3/` and `model/adapter_v4/`: adapter-only `best.pt`, original
  normalizer, pinned base reference, file/tensor hashes, manifests, selection
  curves, resume-test records and chain logs. Base weights are referenced separately.
- `evidence/runs/`: frozen reports, forecasts, diagnostics, calibration, hybrid
  results and figures; all 72 v4 test origins and their hashes are included.
- `small/`: version-specific training scales and frozen variogram pair lists.
- `bundle.json`, `SHA256SUMS`, `provenance.json`: payload hashes, scope and sources.
- `history/v3_only/`: earlier package manifest, post and CPU validation record.

## Offline CPU check

From the unpacked package, use an existing Python environment with NumPy,
PyTorch and Matplotlib. This command installs nothing and contacts no service:

```bash
bash code/scripts/reproduce_small.sh "$PWD" "$PWD/validation/final"
python code/scripts/headline_numbers.py --root evidence/runs --list
```

From the source checkout on Aurora:

```bash
GENCO_PYTHON=/lus/flare/projects/FRAME-IDP/siebenschuh/envs/reason_venv_2026.1.0/bin/python \
  bash scripts/reproduce_small.sh \
  /lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/reproducibility/v0.1.0-rc
```

The check verifies payload hashes, rescores two fixed origins per dataset for
C/P/P0/S/A/D (24 comparisons), and rebuilds three figures. V4 includes
`test_02_0240`, with an unresolved solved state, to check mask handling. Counts
and PIT bins must match exactly; floating comparisons use rtol 2e-6, atol 2e-8.
It preserves full marginal/component metrics, joint ES and variogram scores.

This is saved-forecast rescoring in the existing environment: no fresh GENCO
inference, retraining, OPF generation or independent non-Aurora installation is
claimed. See [VALIDATION.md](VALIDATION.md) for timings and failures. Validation
outputs are outside the payload hash set so checks can be repeated.

## Environment and device profiles

The root README lists CPU/Aurora/NVIDIA profiles. The quick check always uses
CPU and needs none of the full-model Graphkit/Julia dependencies. For a separate
fresh environment, `requirements-reproduce.txt` records NumPy/Matplotlib pins;
provision the appropriate CPU Torch separately. Do not apply those pins over
the Aurora vendor environment.

Full-model additions are inventoried in `requirements-aurora.txt` and
`requirements-nvidia.txt`. On Aurora follow `install_on_aurora.md`, preserving
Torch, Transformers and the numerical stack with selective `--no-deps` installs.
NVIDIA execution is untested. Training chains support CPU/XPU; CUDA inference
can be selected explicitly with `--device cuda:0`, but the PBS launcher is not
a CUDA training implementation. Configure external asset/data paths before any
new full pipeline run. Full OPF generation needs the pinned Julia environment.

## Local assembly

The original v3 builder requires a new destination; finalization adds frozen v4
artifacts and reseals this existing candidate:

```bash
python -m scripts.package_reproducibility --output /lus/flare/.../NEW_DIRECTORY
python -m scripts.finalize_release --bundle /lus/flare/.../NEW_DIRECTORY --add-v4
python -m scripts.finalize_release --bundle /lus/flare/.../NEW_DIRECTORY --seal
```

Assembly checks frozen source hashes and adapter tensor equality. It submits no
jobs and changes no experiment. Post/cards remain review drafts with participant,
contact and hosting placeholders. The current checkout's figure link points to
Lustre; the portable package contains actual image files at `code/submission/figures`.
