# Stage B amendments and adapter performance gate

The user amendments are frozen in [config.yaml](../config.yaml) and
[stage_b_settings.json](stage_b_settings.json). Headline events are hourly and
daily voltage stress at buses 1, 9, 11, 13, 15, 21, and 22. Continuous diagnostics
are PIT and 50/90% coverage/width for Vm at those buses and line-1 loading.
Thermal line-1 daily-any is secondary; its single-corridor and oracle-profile
limitations are recorded in [the prevalence report](stage_b_prevalence.md).
All PNGs use Matplotlib.

The S pool uses hourly rolling origins 168–480, after the seven-day initial fit:
3,756 complete joint residual blocks across 12 training episodes. Blocks overlap;
they do not constitute independent observations. The fitting loop is pending.

## Measured 192-graph gate

PBS job 8956319 ran on `x4417c6s2b0n0`, using the preserved vendor environment,
float32, one forecast, eight samples and 24 future hours. Every timed forward
uses **one PyG Batch with 192 graphs**. There is no graph microbatch loop.
The model includes observed-variable projections, both GRUs, calendar projection,
temporal conditioning, rank-8 plus diagonal sampling, per-sample demand graphs,
final-readout innovations, native GENCO physics, fair energy score and balance
penalty. History embeddings and normalized graph templates are prepared outside
the timed step. GPU timing synchronizes before and after execution.

| Device | Forward + backward | Optimizer | Total step | Seconds/epoch | Six-hour compute projection |
|---|---:|---:|---:|---:|---:|
| CPU, 16 threads | 0.871364 | 0.001298 | 0.872663 | 4,283.03 | 5.04 epochs; 5 complete |
| XPU, Intel Max 1550 | 0.165868 | 0.001909 | 0.167777 | 823.45 | 26.23 epochs; 26 complete |

**Selected training device: XPU.** These are one measured step after one full
warmup step on a training window. The cold CPU/XPU steps were 1.035/2.396 seconds.
The projection uses 4,908 windows per epoch and excludes full historical-cache
creation, input loading, validation, checkpointing and solver work. It is not
an end-to-end runtime promise; enforce the six-hour wall-time cap.

Peak host RSS was 4.02 GiB on CPU; XPU peak tensor allocation was 1.97 GiB.
The 91,461 adapter parameters received finite gradients, every adapter module
had nonzero gradient mass, and optimizer steps changed adapter weights.
The 20,069,187 frozen backbone parameters and buffers retained checksum
`df0c0596b72eed5951189c854ae8f07063c389b08bea6f05d3fe00f8fb9eba4a`
on both devices. Sample-specific demand reached the physics calculation, and
all scoped hooks were removed after calls. Two timing steps do not establish
learning, convergence, CPU/XPU numerical equivalence, or forecast quality.

Full measurements, source hashes and settings hash:
[stage_b_preflight.json](stage_b_preflight.json). Reproduce on an allocated node:

```bash
qsub -N genco-adapter-gate -v GENCO_STAGE_A_MODULE=scripts.adapter_preflight scripts/stage_a.pbs
```

All 20 tests passed, including sample-specific physics against independent
equations, adapter/base gradient separation, exact unconditioned behavior,
adapter serialization, hook cleanup, fair-energy masking and the strict 1.5×
prevalence boundary. Full forecast loops, other shared scores, held-out forecast
evaluation and calibration fitting remain Stage B/C work.
