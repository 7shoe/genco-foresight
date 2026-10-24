# GENCO Foresight

Extend pretrained GENCO from snapshot AC-OPF prediction to 72-hour → 24-hour
probabilistic forecasting on case30, with operating-stress calibration and
anticipatory numerical-solver warm starts. [PLAN_v2.md](PLAN_v2.md) defines the
Challenge Track and its acceptance gates.

Local release preparation is in [reproducibility/README.md](reproducibility/README.md).
The self-contained evidence directory stays on Lustre; its quick check runs
offline and installs nothing. Publication is left to the user. The final post
draft is handed over for review; no GitHub or Hugging Face upload has occurred.
The prepared source release is `v0.1.0`; the local evidence package retains its
review label `v0.1.0-rc`.

**Status:** Stage A and Stage B evaluation are complete. The
[full Stage B report](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/stage_b/full_report.md)
covers P/S/A/D, training-only climatology, a zero-spread P control, all physical
variable scores, coverage/widths, PIT, same-input numerical references, and the
embedding probe. A improves joint ES but substantially undercovers; climatology
has better headline voltage CRPS and coverage. These results do not establish
calibrated operating-risk probabilities. Stage C/D is complete: calibration-only
maps, the cold-versus-A-mean hybrid experiment and ten static Matplotlib PNGs.
The map worsens A's aggregate voltage Brier; warm starts show no solver benefit
and cost 2.93× cold-start work including forecast preparation. F has not run.
See [Stage B evidence and commands](reports/stage_b.md),
[Stage C/D findings](reports/stage_c_d.md) and the [post draft](submission/post.md).

The **in-distribution + physics-weighted** v4 follow-up has completed training
and all 72 test origins using [configs/v4.yaml](configs/v4.yaml). It preserves v3's calendar,
noise and partitions; nominal profile endpoints become 0.80–1.00 and the balance
weight becomes 1.0. This is an exploratory two-change follow-up after v3 scores,
not an ablation. V3 remains primary. Follow the
[v4 pipeline](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v4/pipeline.json)
and [revision disclosure](data/DATASET_CARD.md). All bulk outputs stay on Lustre.
Best epoch 14 is selected. A improves joint ES and headline Vm CRPS versus S,
but its nominal 90% Vm intervals cover only 31.7%; physical imbalance exceeds
the frozen path using the same demand samples. All frozen v4 headline stress
events are degenerate; two unresolved test labels remain masked. See the
[comparison and keyed numbers](submission/headline_numbers.md). V3 remains the
primary record; absolute ES is not comparable across these datasets.

[Gate B follow-up diagnostics](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/stage_b/diagnostics/report.md)
add profile-binned approximation errors, balance/Pg coverage, joint-ES coordinate
accounting and group scores, and every numerical table restricted to realized
ref_t >=0.8. That slice contains 289 forecast hours; A's joint-ES advantage
persists, while S has better voltage CRPS/coverage and active balance there.
No retraining or scorer changes were made. See the updated
[review report](reprot_stage_B.md) for limitations and the parameter/hash correction.

The final v3 dataset has 10,080 valid labels/duals, 23 passing Stage A tests,
10/2/4/4 episodes, 4,090 training windows, and 72 calibration days. Its
[dataset card](data/DATASET_CARD.md) records all revisions. The A chain finished
seven short `debug` jobs within six cumulative training-job hours; epoch 18 is
selected. The XPU 200-step kill/resume check passed bit-for-bit. See the
[training chain](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/adapter_v3/chain.md),
[restart verification](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/adapter_v3/resume_verification.md),
and [v3 audit](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/stage_a_v3/stage_a.json).
Earlier CPU parity passed on all 1,496 local/official test-intersection snapshots
with bit-identical exported bus predictions on the same node. Historical
preflight and v1/v2 evidence remain preserved in [reports/](reports/).

Follow [Installation on Aurora](install_on_aurora.md) to reproduce the verified
package additions. Preserve the installed Torch, Transformers, and numerical
stack. XPU is the measured training choice in [config.yaml](config.yaml); CPU
remains the snapshot reference and a verified training fallback. The
configuration selects the paths, fixed benchmark, and generation law.

For the local saved-forecast check (CPU, no install or download):

```bash
bash scripts/reproduce_small.sh /lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/reproducibility/v0.1.0-rc
python scripts/headline_numbers.py --root /lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs --list
```

| Profile | Dependency inventory | Device toggle | Validation status |
|---|---|---|---|
| CPU saved-forecast replay | `requirements-reproduce.txt`; Torch supplied separately | Replay always uses CPU | Timed local artifact replay; not a clean installation |
| Aurora full model | `reproducibility/requirements-aurora.txt`; follow `install_on_aurora.md` | `--device xpu:0` for evaluation; `--device xpu` for training | Measured XPU runs; preserve vendor Torch/Transformers |
| CPU full model | Pinned Graphkit/Datakit in `requirements.txt`, plus compatible Torch/PyG | `--device cpu` | Snapshot reference and training fallback tested |
| NVIDIA full model | `reproducibility/requirements-nvidia.txt` over an existing CUDA Torch/PyG environment | `--device cuda:0` for evaluation workers | Untested; PBS training driver supports only CPU/XPU |

These are explicit profiles, not automatic environment migrations. No requirements
file installs Torch or Transformers. Use selective `--no-deps` additions on
Aurora; do not replace the vendor numerical stack. The portable evidence package
rebuilds scores/figures without loading Graphkit or installing Julia. See the
[validation record](reproducibility/VALIDATION.md) for its exact scope.

```text
common.py            Configuration, atomic JSON, and content hashes
config.yaml          Device, local paths, pinned sources, and task dimensions
data/
  README.md          Inspected OPFData locations and remaining data checks
  download/          Asset acquisition and preparation
  loader/            Snapshot conversion and temporal windows
  synthetic/         Resumable temporal demand and label generation
models/genco.py      Thin upstream model/normalizer integration
training/            Cached histories, atomic restart state, budgeted A loop
solver/              Persistent Julia/Ipopt adapter and independent AC checks
scripts/             Reproduction and evidence commands
tests/               Scientific and compatibility checks
```

Runtime data, weights, and runs
belong under the external paths in `config.yaml`; existing OPFData and split
files are read-only. See [data/README.md](data/README.md) for the confirmed paths.
On Aurora, bulk artifacts must stay outside `/home`. The current dataset is
`/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/data/temporal_case30_v3`;
v1 and v2 remain alongside it. V3 audit reports, test artifacts and job logs live in
`/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/stage_a_v3`.
The old generator/configuration/report archive is under
`/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/data_v2_provenance`.

V3 fixes disjoint calendar-interleaved source pools before labeling and retains
the original load law. Run `python -m scripts.data_v3` on an allocated node
after configuring the existing Julia runtime. It resumes a 200-hour pilot,
generates all 10,080 labels, audits them, reruns tests, and writes
`readiness.json`. The Stage A audit gates training; prevalence is diagnostic.
There will be no further split iteration. Hourly voltage and line-1 events are
primary; daily panels require 20–80% calibration prevalence. Show test raw/mapped
results side by side, with prevalence tables and shift captions.

Training uses `scripts.training_chain` with one-node `debug` jobs, one hour max.
Workers memory-map cached histories, checkpoint each epoch and every 15 minutes,
and exit before minute 55. The chain enforces six cumulative training-job hours
and selection-score early stopping (patience three). All checkpoints, cache files,
resume tests and the per-job chain log live in external `runs/adapter_v3`.
The driver can be restarted with `python -m scripts.training_chain --device xpu`;
it locks the run to prevent duplicate controllers. Use `--once` for a manual check
or submission. A completed audit and successful resume-test report are required.
See [training operations](training/README.md) for restart verification, clock
definitions, and artifact locations.

The selected Ipopt route adapts RAMBO's PowerModels implementation into an
isolated Julia subprocess in this repository. See
[solver/README.md](solver/README.md) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for source pins and attribution.

Prepare assets explicitly in an existing compatible environment:

```bash
python -m data.download.prepare --download --profile
python -m scripts.parity --prepare --download
```

`config.yaml` points to the existing read-only Aurora cache. Elsewhere, use
`--reference-partitions 0 1` with the preparation command to download official
Graphkit-format OPFData parquet; each partition contains 200 snapshots. Content
hashes and pinned revisions are recorded. This format and the local canonical
cache have separate readers/conversion paths. `load_snapshot` uses the downloaded
official partitions when the local cache is absent; that route is tested against
the local conversion.

The official evaluator needs the optional logger packages in
`requirements-cli.txt`. Aurora's run installed these with `--no-deps` into
external `assets/cli_deps`; `scripts.parity` exposes that path only to the
evaluator subprocess. The existing environment is preserved.

Submit CPU reference validation and then the resumable data run on Aurora:

```bash
qsub -v GENCO_STAGE_A_MODULE=scripts.reference_validation scripts/stage_a.pbs
# After parity and the documented training-only generator selection:
qsub -v GENCO_STAGE_A_MODULE=scripts.run_stage_a,GENCO_FULL_GENERATION=1,GENCO_XPU_CHECK=1 scripts/stage_a.pbs
```

The data run requires CPU parity and a verified 200-hour pilot before expansion.
It resumes completed hours and keeps unresolved attempts and masks. On other CPU
machines with the configured Julia executable/depot, use
`GENCO_FULL_GENERATION=1 python -m scripts.run_stage_a`.

```bash
python -m pytest -q tests
python -m scripts.data_audit
python -m scripts.collect_stage_a --snapshot
```

The historical profile is EIA's ERCOT 2025 hourly demand in UTC, driving a
synthetic IEEE network. Generation settings were selected from training-only
pilot evidence, then frozen before held-out labeling. Voltage daily-union
events can be structurally uninformative because cost-minimizing OPF operates
some buses near a bound; report component probabilities and state calibration
separately. Stage B's held-out comparisons and Stage C/D's calibration,
two-arm solver experiment and figures are complete; public release remains pending.
