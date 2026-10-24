# Stage A data inventory

The following locations were inspected read-only on 23 October 2026. This
initial inventory recorded filesystem/text-metadata checks. Stage A now reads
the canonical cache and frozen split arrays and converts them to Graphkit.

| Asset | Confirmed location |
|---|---|
| OPFData root | `/lus/flare/projects/FRAME-IDP/siebenschuh/acopf_datasets/` |
| case30 N1 cache | `processed/N1/pglib_opf_case30_ieee/` beneath that root |
| case30 T1 cache | `processed/T1/pglib_opf_case30_ieee/` beneath that root |
| N1 split files | `processed/N1/splits/pglib_opf_case30_ieee/splits.npz` and `training_subsets.npz` |
| Raw source root | `raw/OPFData/` beneath that root |
| Existing reader | `/home/siebenschuh/Projects/lumina-sdk/distillation/baselines/n1/curated_data.py` |
| Collection documentation | `/home/siebenschuh/Projects/lumina-sdk/distillation/baselines/artifacts/n1_datasets_20261022/README.md` |

Both case30 manifests declare format `lumina-t1-data-v1`, 300,000 samples,
15,000 samples per group, and numeric `example_ID` ordering. Their tensor schema
has been checked against 32 independent official-export snapshots. Equal IDs across T1 and N1 are not
paired physical scenarios. The collection documentation reports 240K/30K/30K
N1 splits; actual membership is checked for complete, disjoint coverage. CPU
parity passed on all 1,496 snapshots in the official/local test intersection.

`loader/dataset.py` adapts minimal read-only cache access and explicitly converts
schema/units without SDK imports. It uses upstream graph construction, masking,
normalization and inference. Adapted source is attributed in
`../THIRD_PARTY_NOTICES.md`.
The same module provides an official-parquet reader for downloaded partitions;
`load_snapshot` selects it when the canonical Lustre cache is unavailable.

Use existing OPFData and split files read-only. Keep acquisition/preparation
source in `download/`, conversion/window source in `loader/`, and temporal
generation source in `synthetic/`. Store downloaded assets, generated shards,
manifests, and run outputs at the external locations in `../config.yaml`.
The external `GENCO/{runs,assets,data}/` directories now exist. The selected
dataset is `GENCO/data/temporal_case30_v3/`; its manifest records the final
calendar-interleaved source pools, source dates, seeds, sampling law, budgets,
implementation hashes, and shard hashes. V1/v2 and their audits remain unchanged at
`GENCO/data/temporal_case30_v1/` and `GENCO/data/temporal_case30_v2/`. All these paths are under
`/lus/flare/projects/FRAME-IDP/siebenschuh/`; do not store runtime datasets,
embeddings, checkpoints or logs in `/home` or this source directory.
V3's external audit output is `GENCO/runs/stage_a_v3/`; its
`readiness.json` records the passed Stage A audit. Prevalence is a reported
diagnostic, not a training blocker. See [the dataset card](DATASET_CARD.md) for
the final allocation, revised reporting contract and complete version history.

The source is a complete 8,760-hour ERCO/D series for 2025 from EIA's public
API. Hourly MWh divided by one hour gives mean MW. Raw responses, URL, retrieval
time and hashes are retained. UTC indexing avoids daylight-saving duplicates.
Packaged Datakit filenames were not assumed to establish 2025 provenance.

Datakit supplies `Network`, `makeYbus`, and the aggregate-profile scaler. Our
sequential wrapper adds uniform bus noise and stationary lognormal AR(1) noise,
derives Qd from baseline power factor, and preserves reactive-only loads.
Seventeen disjoint 21-day source blocks form pools of 7 train / 2 selection /
4 calibration / 4 test blocks; eight trailing source days are unused. Twenty
independently seeded episodes reuse blocks only within their assigned pool.
The 72 test forecast days and 72 calibration days each come from four stochastic
episodes using four source blocks, not independent real years.

The original 0.65–1.15 training pilot remains under `temporal_case30/`. Every
hour had thermal stress. A training-only six-point load diagnostic found valid
non-stressed solutions at lower demand and an unresolved/invalid reference at
scale 1.15. The selected 0.30–1.00 range was frozen before held-out labeling.
V1's revised 200-hour training pilot has 200 valid labels and 34 thermal-stress
hours. These are v1 results; v2 is generated and audited independently.

Generation records every attempt with status, iterations, runtime, objective,
duals, checks and masks. Unresolved target states are null. `scripts/data_audit.py`
rechecks demand reproduction, physical validity, source separation, hashes and
daily support. `load_window` separates past graphs/future calendar from targets;
future demand and solved states never enter the inference input dictionary.

The standalone solver checkout exists at `/home/siebenschuh/Projects/acopf`.
Its inspected reference module defines PYPOWER and SciPy backends. Its
`core/problem.py` changes Torch's default dtype at import, which any later reuse
must handle explicitly.

The Ipopt implementation has now been identified separately in
`/home/siebenschuh/Projects/lumina-sdk/distillation/RAMBO/julia/model.jl`,
accessed through a persistent Julia subprocess. Its minimal licensed OPF path
and Project/Manifest are retained in this repository. Cold/warm and graph
round-trip solves pass independent checks, and the active-dual sign/unit
perturbation passes with Qd fixed. See
[the solver inventory](../solver/README.md) for implementation and evidence.
