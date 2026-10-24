---
license: apache-2.0
library_name: pytorch
base_model: gridfm/genco-opfdata
tags:
- time-series
- power-systems
---
# GENCO Foresight: case30 post-training adapters v3 and v4

**Draft for review. Not uploaded.** HF organization: `<HF_ORG>`; repository:
`<HF_ORG>/genco-foresight-adapters`. Participants: `<NAMES>`.
Affiliations: `<AFFILIATIONS>`. Contact: `<EMAIL>`. Code: `<GITHUB_REPOSITORY_URL>`.

## Model and base reference

Both adapters use frozen `gridfm/genco-opfdata`, case30 seed 42,
revision `74e19e1e210a175aa17023f4db161f0ae9f753cb`, directory case30_ieee/seed42.
Graphkit commit: `f104900245cba26d1380037eea562cf633a97484`; Datakit 1.1.0.
The original normalizer_stats.pt is included and is never refitted.
Base checkpoint file SHA256: `9dbf5596b4eb63a3b0e23a7d9a84427c33f3829a07ad2d7dbd12e99e364545a4`.
Base named-tensor-bytes SHA256: `df0c0596b72eed5951189c854ae8f07063c389b08bea6f05d3fe00f8fb9eba4a`.
These hash different byte streams; neither is substituted for the other.

| Adapter | Best selection epoch | Exported adapter file SHA256 | Named-tensor-bytes SHA256 |
|---|---:|---|---|
| v3 | 18 | 2d78bee036ebafe5269e4b650af3bdeac560aa52d421b53682d776e46c559853 | df961f7b9dabcc8c3f76273f81a0656ff9da82c404f7f69074188eec03251c99 |
| v4 | 14 | 362405c131938999c913ea974db24b7bd2a34cc5285d4a43c753e4070a5a5a98 | f4726b63ec6a199269ad3153ccca7e3185d284cb58af966b72e994598f2e2b7c |

Each adapter has 91,461 trainable parameters and 884 buffer elements, for 92,345
state-dict elements. Historical manifests that called 92,345 trainable remain
preserved; release_manifest.json records the corrected accounting. Each model
directory includes its original manifest, selection curve, chain log, resume-test
record, base reference and release manifest. V4 additionally includes all
per-origin evaluation hashes. Full forecasts/results are in evidence/runs.

## Method and scope

72 observed hours produce 64 joint 24-hour scenarios. Inputs are past
Pd/Qd/Vm/Va/Pg/Qg and availability masks, frozen final GENCO embeddings, and
known future calendar. Node-type GRUs have width 32; projections add memory
into the backbone's width-48 input latents. Rank-8-plus-diagonal innovations
drive demand and state readouts. Each scenario's own demand reaches its graph;
Qg and branch flows are reconstructed by the model's physics operations.
The backbone and checkpoint normalizer remain frozen.

Loss is fair train-scaled energy score plus normalized squared P/Q bus
imbalance. V3 uses lambda 0.01 and nominal profile endpoints 0.30–1.00; v4 uses
lambda 1.0 and endpoints 0.80–1.00. V4 changes both after seeing v3 scores:
it is not an ablation. Absolute ES across datasets is not a controlled comparison.
Both use one seed, fixed topology/costs/commitment and no ramps.

## Better forecast scores / worse physical consistency

Within v4, A has joint ES 0.303337 versus S
0.496910 and C 0.424719.
Headline voltage CRPS is 0.000958 for A versus
0.001466 for S. Demand-family ES still favors S.
Nominal 90% voltage intervals cover only 31.71%.
Mean |P imbalance| is 1.1133 MW/bus for A versus
0.0374 for D using A's identical demand samples through
frozen GENCO. D controls the demand input; it does not isolate conditioning from
the adapter's stochastic state readouts.

V3's A joint ES is 0.429958, but headline voltage coverage
is only 28.75%. Frozen-GENCO identical-input
error rises below the checkpoint's nominal load regime. Calibration mapping
worsens A's aggregate voltage Brier. V3 warm starts provide no online solver
benefit and cost 2.93 times cold-start work including
forecast preparation. The embedding probe is negative. These are main results.

V4 retains two unresolved test labels and all eight frozen stress events are
degenerate. Neither model is demonstrated to provide calibrated operational
uncertainty or feasible dispatch. No deployment/control use is validated.

## Loading and validation

best.pt is a plain adapter state dictionary, without backbone or optimizer/RNG
state. Construct the repository's TemporalAdapter with the pinned frozen base
and the matching dataset's train-coordinate scales, then:

```python
state = torch.load(path, map_location="cpu", weights_only=True)
missing, unexpected = adapter.load_state_dict(state, strict=False)
assert not unexpected
assert all(name.startswith("backbone.") for name in missing)
```

This export cannot resume training and differs from the wrapped original
training checkpoint. Original checkpoint file hashes and export hashes are
recorded separately. Tensor hashing concatenates sorted UTF-8 names and
contiguous CPU NumPy tensor bytes, including buffers. Base weights are referenced,
not redistributed in this package.

The offline CPU reproduction rescores saved predictions and rebuilds figures;
it does not execute a fresh GENCO inference/training or OPF generation. Independent
non-Aurora and NVIDIA reproduction remain untested. Preserve vendor Torch and
Transformers on Aurora. No repository has been uploaded or published.
