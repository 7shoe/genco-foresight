# Stage A verification record

Stage A passed on 23 October 2026. Machine-readable evidence is in
[`stage_a.json`](stage_a.json); full logs, predictions and solver records remain
at the external paths recorded there. Forecasting/adaptation/calibration have
not been run yet.

| Check | Result |
|---|---|
| Pinned GENCO | Strict case30 seed42 load; 20,069,187 parameters. |
| Immutable normalizer | Original physical base 100 MVA; restored normalization base 75.95927429199219 MVA. |
| CPU snapshot parity | All 1,496 official/local test-intersection snapshots: bit-identical on the same node (`x4418c3s0b0n0`), using the same Python/Torch binaries and float32, versus the unmodified official evaluator. |
| Independent input conversion | 32 official-export spot checks; all local cache group hashes verified. |
| Numerical backend | Repository-owned PowerModels/Ipopt subprocess; cold/warm and graph round-trip checks pass. |
| Active duals | Fixed-Q 0.01 MW perturbation; `-raw_dual/baseMVA` matches the objective derivative to about `1.5e-10` relative error. |
| Training pilot | Revised 200 hours: all valid, 34 thermal-stress hours; original pilot retained. |
| Full data | 20 × 504 = 10,080 valid hourly labels and valid active duals; no unresolved attempts. |
| Physical audit | Maximum active imbalance `1.03e-8 MW`, reactive imbalance `2.38e-7 Mvar`; no bound, thermal or angle violations under the declared `1e-5` checks. |
| Resume/integrity | Demand draws reproduced exactly; source pools disjoint; all shard hashes checked; retained pilot prefix matches its pre-expansion SHA256. |
| Tests | 16 passed, none skipped. Leakage, masks, independent physics, sample-specific demand, frozen-base gradients, embeddings, serialization, the download-backed reader, hourly training origins, and validation before JSON replacement covered. |
| XPU | Scatter, TransformerConv and GENCO forward/backward verified on an allocated compute node without a shim. CPU remains the configured default. |
| Environment | Base `pip freeze` unchanged. Five optional official-CLI logger packages live in an isolated external target. |

The full generation invocation took approximately 395 seconds, excluding the
preceding pilot, queue wait and setup/parity work. This is data-generation
timing, not a forecast or hybrid-solver speedup. The single numerical smoke
example used 18 cold-start versus 19 warm-start iterations; only initialization
support is established.

Allocated-node runs: CPU parity in PBS job `8956235`; full generation and XPU
verification in `8956241`. Commands, earlier failed bring-up logs and final
outputs are retained under the external Stage A run directory.

Follow-up PBS job `8956300` records binary/node/dtype provenance and confirms
byte-for-byte CPU parity. Its profiled XPU GENCO forward uses native PyTorch
scatter operators with no `torch_scatter::` custom operators. The installed
package nevertheless loads `_scatter_cpu.so` and its other CPU libraries at
import; a claim that no compiled extension is loaded would be false.

The benchmark uses EIA's complete 2025 hourly ERCO demand profile and Datakit
network/admittance/scaling utilities. Reference scaling 0.30–1.00, uniform bus
noise ±3%, and stationary lognormal AR(1) with rho 0.9 and innovation standard
deviation 0.02 were frozen after training-only pilot/domain checks. The earlier
0.65–1.15 pilot had thermal stress every hour; the revised range was selected
before held-out labeling. No held-out forecast scores selected the data law.

| Partition | Episodes | Daily forecast origins | Thermal daily-any positive / negative |
|---|---:|---:|---:|
| Train | 12 | 216 | 138 / 78 |
| Model selection | 2 | 36 | 15 / 21 |
| Risk calibration | 2 | 36 | 9 / 27 |
| Test | 4 | 72 | 36 / 36 |

Daily any-voltage-stress is positive throughout the dataset. Individual bus
events vary, but the daily union cannot demonstrate discriminative calibration.
The numerical-difficulty classifier also lacks positive examples at the fixed
solver budget. Report those quantities as unsupported for classifier fitting;
retain component voltage and thermal calibration studies.

The test partition has only four stochastic episodes using three source blocks;
calibration has two episodes/two source blocks. The 72/36 days are correlated,
thin evidence for rare-event calibration. This is a synthetic ERCOT-profile-to-
IEEE-case30 benchmark with fixed topology/costs/commitment and no ramps.
Ipopt labels are independently checked local numerical references, not global
optimality certificates.

To regenerate the compact evidence after the data audit:

```bash
python -m scripts.data_audit
python -m scripts.collect_stage_a --snapshot
```

The pre-Stage-B follow-up is recorded in `stage_a_followup/events.json`, with
all 30 bus and 41 line counts in adjacent CSVs. Only line 1 (bus 1–2) ever has
thermal stress. A train-fitted logistic model of the **realized next-day**
maximum profile has calibration AUROC 1.0: this is an oracle structural
diagnostic, not a permitted forecast or evidence of probabilistic calibration.
No bus exceeds 95% voltage stress over all 10,080 hours; calibration support
must still exclude constant and unsupported component events.

Matplotlib is now fixed for all PNGs; an actual PNG export passed, while
Plotly export in the isolated target failed for missing Kaleido. Next is
Stage B: the rolling-origin ridge baseline, P/S forecasts, common scoring, then
the temporal adapter and its attribution control. No model/data publication or
Community submission has occurred.
