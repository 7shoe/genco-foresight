# Third-party sources

Our implementation uses Apache-2.0; dependency and data licenses remain their
own. Copied/adapted source is identified below and in the affected files.

| Source | Pin and local use | Changes |
|---|---|---|
| `lumina-sdk/distillation/RAMBO/backend.py` | SDK `e75ffcd74854c14e2960b9a9411ddd72377e5239`; `solver/adapter.py` | Persistent Julia JSON transport adapted with explicit paths, OPF-only requests, physical units, and failure records. |
| `lumina-sdk/distillation/RAMBO/julia/{model.jl,worker.jl}` | Same SDK pin; `solver/julia/worker.jl` | Generic PowerModels AC-OPF path only; explicit mapping, duals, iterations, and starts. RAMBO's specialized optimization is omitted. |
| RAMBO Julia `Project.toml`, `Manifest.toml` | Same SDK pin; `solver/julia/` | Unmodified resolved dependency metadata. |
| `lumina-sdk/distillation/training/T1_single_topo/data.py` | Same SDK pin; canonical-cache reader in `data/loader/dataset.py` | Lazy read-only group access and manifest checks; no SDK imports. |
| PGLib-OPF `pglib_opf_case30_ieee.m` | Existing RAMBO case, PGLib v21.07; `data/networks/` | Unmodified benchmark data. Copyright/CC BY 4.0 attribution retained in file; accompanying data/software license retained. |
| `gridfm_datakit/network.py:load_net_from_file` | Datakit 1.1.0; MATPOWER-frame dictionary construction in `data/loader/network.py:load_network` | Minimal CaseFrames-to-dictionary adaptation, followed by imported `Network` and `makeYbus`; the automatic Julia network-correction call is omitted. |

The SDK license and NOTICE are retained under `third_party/lumina-sdk/`.
The PGLib case license is retained under `data/networks/LICENSE`.

Graphkit is used by import at commit
`f104900245cba26d1380037eea562cf633a97484`; GENCO weights/config/stats are from
`gridfm/genco-opfdata` revision `74e19e1e210a175aa17023f4db161f0ae9f753cb`.
Datakit 1.1.0 supplies network parsing/admittance utilities and the aggregate
profile scaling mechanism. Those implementations are imported; the small
CaseFrames conversion adaptation is listed above. Its Apache-2.0 license is
retained under `third_party/gridfm-datakit/`.
Asset manifests record downloaded content hashes and profile provenance.

The Aurora module-version list was read from the existing environment's
`reason_env_common.sh`; `scripts/stage_a.pbs` configures those installed modules.
No environment guard or other repository's bootstrap implementation is vendored.
