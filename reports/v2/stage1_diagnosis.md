# Stage 1 gradient diagnosis

**PASS for the user-specified single-batch directional-gradient criterion.** Every
group in ON and all five OFF evaluations has cosine >= 0.99 against CPU fp64.
No group triggers the < 0.95 selective-fp64 follow-up. OFF is the candidate under
the revised epoch-level acceptance rule; this diagnostic does not establish
three-epoch acceptance, boundary resume or a passing Stage 1 production gate.
The original failed Stage 1 evidence remains unchanged.

## Protocol and provenance

[Protocol](stage1_diagnosis.protocol.json), SHA256
`6536529d037b7a792cebc441646b19f51ca2ab3583e86def5f9d26b4bf341d7e`, was saved and
frozen before submission. [Gate record](stage1_diagnosis.json) links evidence hashes.

Evidence root:
`/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/stage1_diagnosis_20261025_v1`.
All tensor fixtures, gradients, outputs and runtime files are there. The worker
preserved the production sources and all vendor dependencies; only a diagnostic
entry point, PBS launcher and focused diagnostic tests were added.

The fixture is the first failed Stage 1 step: case30 v4, seed 42, origin
`train_09:123`, B1, global K8, 24 hours, covariance rank 8. The original noise was
logged as a hash rather than a tensor. Reconstruction reproduced the recorded
noise hash `0f729e6250810269bc694b25a93c82de179f02ebd3b33d5d6afcd45133534bd1`
and initial adapter hash
`aaedb0c038aec1a770a8e60d075323d12bc0e14c56f000e005e788ec4b34c647` exactly.
The reconstructed tensors and full starting state are saved in `fixture.pt`.
Dataset, coordinate scales, checkpoint files, Graphkit and production source
hashes match the failed run. Copied cache contents match; their manifest paths
are new. The ON loss, ES and balance reproduce the original first step exactly.

CPU fp64 casts the same initial adapter/backbone, floating buffers, graph, cached
history, calendar, target, scales and noise to double. Integer indices and Boolean
masks retain their types. Cached embeddings are promoted, not recomputed. The fair
ES retains all within-origin sample pairs; its dimension normalizer and reductions
are double in the reference. The baseline uses the unmodified fp32 production loss.
The frozen backbone stays frozen and input-conditioning gradients remain enabled.
Each evaluation starts from the same parameters and performs one backward pass,
without clipping, optimizer updates or new noise. No parameters or buffers changed.

Metrics concatenate every trainable parameter in each named group. Norms <= 1e-12
produce an unresolved cosine, never an automatic pass; none occurred at group level.
The supplied >= 0.99 and < 0.95 criteria were fixed before submission. Forward
deltas and norm ratios are descriptive and were not assigned post-hoc thresholds.

## Results

OFF intervals below are minima and maxima across all five repeats.

| Parameter group | ON cosine | OFF cosine range | ON norm / fp64 | OFF norm / fp64 range |
|---|---:|---:|---:|---:|
| Observed projections | 0.999281 | 0.999671–0.999997 | 0.822090 | 1.013360–1.247852 |
| GRUs | 0.997722 | 0.998978–0.999994 | 0.824062 | 1.013619–1.248016 |
| Conditioning | 0.997621 | 0.998955–0.999997 | 0.845835 | 1.015930–1.229315 |
| Calendar | 0.995663 | 0.997991–0.999995 | 0.856813 | 1.016357–1.220757 |
| Heads | 0.999975 | 0.999937–0.999993 | 0.999917 | 1.000164–1.000608 |

The GRUs and conditioning account for 97.51% of ON's squared gradient error and
92.80–97.46% of OFF's. Their directions remain close, while their magnitudes move
substantially. The heads' norm is comparatively stable. Total unclipped gradient
norm is 319.467652 for fp64, 290.002013 for ON, and 322.248711–365.802998 for OFF.
The original production norm uses a different fp32 reduction; the diagnostic total
above concatenates gradients and reduces in fp64 for reporting.

ON is **not uniformly closer** to fp64. OFF has higher cosine in every repeat for
observed projections, GRUs, conditioning and calendar. The heads are mixed: three
OFF repeats have higher cosine than ON, two lower. ON's head norm ratio is closer
to one than every OFF repeat. For the other groups, three OFF repeats have smaller
norm errors than ON, while two have larger errors.

| Forward delta against fp64 | ON | OFF range |
|---|---:|---:|
| Vm mean absolute, p.u. | 5.48314e-7 | 6.30200e-7–6.91633e-7 |
| Vm p95 absolute, p.u. | 1.18179e-6 | 9.51863e-7–1.13234e-6 |
| Vm maximum absolute, p.u. | 2.20649e-4 | 4.92374e-4–6.85612e-4 |
| Joint mean absolute | 1.09265e-3 | 1.04940e-3–1.16131e-3 |
| Joint p95 absolute | 1.44015e-3 | 1.19013e-3–1.30052e-3 |
| Joint maximum absolute | 1.12933 | 1.70206–2.36589 |

Joint coordinates mix physical units and these absolute deltas are not scaled ES.
ON has smaller worst-case Vm and joint errors, despite its less accurate gradient
directions in four groups. Thus forward agreement alone does not rank gradient
agreement. Reference loss is 8.60713796686839; ON is 8.60700511932373; OFF spans
8.606904029846191–8.607023239135742. All losses, outputs and gradients are finite.

Under the user's decision rule, this supports a usable direction with substantial
numerical sensitivity, consistent with reassociation effects. It does not measure
a condition number or prove which kernel causes the sensitivity. The loss64,
heads64 and combined tests were predeclared but not executed because no group
cosine fell below 0.95.

## Epoch evidence and next gate

The three historical v4 runs share data, checkpoint, coordinate scales, training
recipe and their historical source hashes. Current production source hashes differ
from those historical sources, while the case30 data and optimizer/objective
settings match. Their actual per-epoch ES envelopes are:

| Epoch | Three-seed minimum | Three-seed maximum | Spread |
|---|---:|---:|---:|
| 1 | 0.5118635098 | 0.6171770659 | 0.1053135561 |
| 2 | 0.4213344115 | 0.4580226243 | 0.0366882127 |
| 3 | 0.3815448309 | 0.4392866890 | 0.0577418581 |

These are matching per-epoch observations, not final-score variability. Seeds 42
and 2 actually early-stop at epoch 17. Seed 1 reaches epoch 19 and then its time
budget: it supplies no observed early-stopping epoch. A three-epoch comparison
cannot test stopping epoch +/-1; the real controller has a separate patience-3
and epoch-boundary resume test. The revised production protocol must retain exact
configuration/counter/RNG restoration and finite persisted training metrics while
using epoch-level ES acceptance. Stage 2 and a local Stage 1 merge require that gate.

## Cost, checks and limitations

PBS job `8962527.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov`, node
`x4602c3s6b0n0`, exited 0. One debug node was reserved for 1,200 seconds; the worker
guard was 1,080 seconds. PBS accounts 75 seconds, and the launcher records 73.822
seconds including setup and tests. Model-worker duration was 18.165 seconds.
CPU reference evaluation took 1.735 seconds. XPU ON took 2.174 seconds; OFF first
evaluation took 0.480 seconds and subsequent evaluations 0.205–0.208 seconds.
These cold diagnostic timings are not production speedup measurements. Seven
model evaluations were performed; there was no retry or second diagnostic job.

All three diagnostic tests passed on the compute node: metric edge cases,
masked/all-pairs double ES with gradient checking, and head-boundary gradient
preservation. The original Stage 1 tests and mechanism results remain available in
its unchanged evidence. No training acceptance was inferred from this one origin.
ON has only one evaluation; CPU fp64 is a numerical reference, not analytic truth.
Neither the original failed gate nor release manifests/tags were rewritten.
