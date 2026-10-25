# Multigrid step and tile profile

Job `8962223.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov` on `x4112c2s6b0n0`; status **complete_with_capacity_limits**; [raw results](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/multigrid/profile/20261025T041341Z/summary.json). Measured source `d28b9e29527a0fdd1511175dca1becbef752cfb6`. Float32; unchanged `training.loop.training_step`; synthetic inputs/scales, no datasets.
Each XPU process uses one Intel Max 1550 tile; the CPU row uses 16 threads. Times are median / p90 seconds with XPU synchronization at phase boundaries. Forward includes the history encoder, sampling, GENCO and physics; optimizer includes gradient clipping and scheduling; total also includes transfers, RNG and the existing digest. Per-step JSON writes are outside timers; co-tenancy includes those gaps. Memory is peak allocated device GB; reserved/host memory and tile details are in JSON.
Epoch estimates use 4,090 steps; 19-epoch estimates add measured 36×64 forward-only validation once per epoch when available († otherwise training only). Setup, checkpointing, validation scoring and queue wait are excluded. K=1 cannot train with the unchanged fair energy score; smaller K is not run after a larger K fits.

## Table A

| Grid | K | Device / deterministic | Status | Bus / branch / gen | Graphs/step | Fwd s | Loss s | Bwd s | Opt s | Step s | Peak GB | Train s/epoch | Validation s | Hours/19 epochs | Graphs/s |
|---|---:|---|---|---|---:|---|---|---|---|---|---:|---:|---:|---:|---:|
| case14_ieee | 8 | xpu / off | OK | 14/20/5 | 192 | 0.063 / 0.063 | 0.002 / 0.002 | 0.092 / 0.093 | 0.003 / 0.003 | 0.161 / 0.161 | 1.136 | 658.375 | 16.235 | 3.560 | 1192.755 |
| case30_ieee | 8 | xpu / off | OK | 30/41/6 | 192 | 0.066 / 0.066 | 0.002 / 0.002 | 0.095 / 0.095 | 0.003 / 0.003 | 0.166 / 0.167 | 2.111 | 680.239 | 16.154 | 3.675 | 1154.418 |
| case57_ieee | 8 | xpu / off | OK | 57/80/7 | 192 | 0.066 / 0.066 | 0.002 / 0.002 | 0.103 / 0.103 | 0.003 / 0.003 | 0.175 / 0.175 | 3.781 | 714.639 | 18.186 | 3.868 | 1098.849 |
| case118_ieee | 8 | xpu / off | OK | 118/186/54 | 192 | 0.127 / 0.128 | 0.002 / 0.002 | 0.183 / 0.183 | 0.003 / 0.003 | 0.317 / 0.317 | 9.542 | 1295.155 | 38.191 | 7.037 | 606.321 |
| case500_goc | 8 | xpu / off | OK | 500/733/224 | 192 | 0.508 / 0.508 | 0.002 / 0.002 | 0.631 / 0.631 | 0.003 / 0.003 | 1.147 / 1.148 | 36.686 | 4693.116 | 146.713 | 25.544 | 167.326 |
| case2000_goc | 8 | xpu / off | ERROR | 2000/3639/384 | 192 | — / — | — / — | — / — | — / — | — / — | 66.695 | — | — | — | — |
| case2000_goc | 4 | xpu / off | OOM | 2000/3639/384 | 96 | — / — | — / — | — / — | — / — | — / — | 67.742 | — | — | — | — |
| case2000_goc | 2 | xpu / off | OK | 2000/3639/384 | 48 | 0.501 / 0.501 | 0.002 / 0.002 | 0.619 / 0.619 | 0.003 / 0.003 | 1.132 / 1.133 | 37.775 | 4631.574 | 587.142 | 27.543 | 42.387 |
| case30_ieee | 8 | xpu / on | OK | 30/41/6 | 192 | 0.111 / 0.112 | 0.002 / 0.002 | 0.150 / 0.150 | 0.003 / 0.003 | 0.268 / 0.268 | 2.111 | 1094.692 | — | 5.778† | 717.353 |
| case118_ieee | 8 | xpu / on | OK | 118/186/54 | 192 | 0.198 / 0.198 | 0.002 / 0.002 | 0.280 / 0.280 | 0.003 / 0.003 | 0.484 / 0.485 | 9.542 | 1981.396 | — | 10.457† | 396.327 |
| case30_ieee | 8 | cpu / off | OK | 30/41/6 | 192 | 0.369 / 0.371 | 0.001 / 0.001 | 0.424 / 0.427 | 0.003 / 0.003 | 0.796 / 0.800 | — | 3256.294 | — | 17.186† | 241.158 |

`solo_case2000_goc_k8` rank 0: Level Zero OUT_OF_RESOURCES; retained original backend error.

`solo_case2000_goc_k4` rank 0: XPU memory exhausted; allocation details in raw JSON.

case2000_goc validation used K=1 chunks for all 64 samples per origin; training used K=2.

The case2000 smaller-K retries used the same allocation with the suite controller paused; no concurrent timing overlapped the retries. The resource-error fallback was corrected after the step measurements; numerical code was unchanged.

## Table B

| Grid | Fits one tile / K | Single-tile hours | Tiles: 1 h / 3 h / 6 h, ideal | Recommended mode |
|---|---|---:|---|---|
| case14_ieee | Y / 8 | 3.560 | 4 / 2 / 1 | one tile per model |
| case30_ieee | Y / 8 | 3.675 | 4 / 2 / 1 | many models per node |
| case57_ieee | Y / 8 | 3.868 | 4 / 2 / 1 | one tile per model |
| case118_ieee | Y / 8 | 7.037 | 8 / 3 / 2 | DDP across tiles/nodes |
| case500_goc | Y / 8 | 25.544 | 26 / 9 / 5 | DDP across tiles/nodes |
| case2000_goc | Y / 2 | 27.543 | 28 / 10 / 5 | sample-parallel across tiles |

The requested ‘upper bound under ideal scaling’ applies to speedup: ceil(hours/target) is an optimistic **lower bound on required tiles**, not measured scaling or a memory guarantee. Multi-tile recommendations are prospective; no model sharding or DDP was tested.

## Table C

| Concurrent chains | Process / tile | Step median / p90 s | Median as % of solo | Status |
|---:|---:|---|---:|---|
| 1 | 0 | 0.163 / 0.163 | 100.000 | OK |
| 4 | 0 | 0.165 / 0.174 | 101.605 | OK |
| 4 | 1 | 0.164 / 0.172 | 100.632 | OK |
| 4 | 2 | 0.165 / 0.165 | 101.463 | OK |
| 4 | 3 | 0.164 / 0.165 | 100.907 | OK |
| 8 | 0 | 0.170 / 0.171 | 104.291 | OK |
| 8 | 1 | 0.167 / 0.168 | 102.680 | OK |
| 8 | 2 | 0.164 / 0.165 | 100.982 | OK |
| 8 | 3 | 0.168 / 0.169 | 103.093 | OK |
| 8 | 4 | 0.168 / 0.169 | 103.078 | OK |
| 8 | 5 | 0.168 / 0.169 | 103.512 | OK |
| 8 | 6 | 0.172 / 0.173 | 105.778 | OK |
| 8 | 7 | 0.160 / 0.161 | 98.607 | OK |
| 12 | 0 | 0.170 / 0.171 | 104.556 | OK |
| 12 | 1 | 0.172 / 0.173 | 105.900 | OK |
| 12 | 2 | 0.170 / 0.171 | 104.672 | OK |
| 12 | 3 | 0.172 / 0.173 | 105.661 | OK |
| 12 | 4 | 0.169 / 0.170 | 103.735 | OK |
| 12 | 5 | 0.172 / 0.173 | 105.688 | OK |
| 12 | 6 | 0.162 / 0.163 | 99.604 | OK |
| 12 | 7 | 0.164 / 0.166 | 101.087 | OK |
| 12 | 8 | 0.173 / 0.174 | 106.213 | OK |
| 12 | 9 | 0.174 / 0.176 | 107.210 | OK |
| 12 | 10 | 0.172 / 0.173 | 105.687 | OK |
| 12 | 11 | 0.174 / 0.176 | 107.259 | OK |

![Step time by grid and K](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/multigrid/profile/20261025T041341Z/step_time.png)

Single-tile 1 h estimate: none established.  
Requires chained 1 h jobs under this recipe: case14_ieee, case30_ieee, case57_ieee, case118_ieee, case500_goc, case2000_goc.  
Single-tile six-hour target exceeded or a valid step not established: case118_ieee, case500_goc, case2000_goc; synthetic timing does not establish convergence.
