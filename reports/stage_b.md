# Stage B evidence

Evaluation and the requested additional controls are complete. All runtime
artifacts are under `/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/stage_b`.

- [Full report](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/stage_b/full_report.md)
- [Gate B table](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/stage_b/gate_b.md)
- [Score, coverage and PIT figure](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/stage_b/comparison.png)
- [Embedding probe](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/stage_b/probe/report.json)
- [Verification](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/stage_b/verification.json)
- [Gate B follow-up diagnostics: full and >=0.8 tables](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/stage_b/diagnostics/report.md)
- [Corrected parameter and hash accounting](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/stage_b/diagnostics/parameter_hash_accounting.json)

The requested follow-ups use the saved forecasts only. Just 289/1,728 forecast
hours have realized ref_t >=0.8. P's approximation errors drop sharply there;
A retains the lowest joint ES, but S has better voltage CRPS/coverage and active
balance on that slice. Flow coordinates dominate the full-set S/D joint scores.
The adapter has 91,461 trainable parameters and 92,345 state elements including
884 buffers; file hashes and tensor-content hashes are distinct. The accounting
correction supersedes the historical report's discrepancy wording.

All six modes share 72 midnight test origins and the frozen scorer. S/A/D/C use
64 samples; P uses one point forecast and P0 uses 64 exact copies. S uses the
approved 3,130 rolling residual blocks. Climatology C samples 210 whole training
days, preserving joint demand/state dependence. A/D have exactly identical demand
samples. Seven scoring/control tests and 2,088 artifact-hash checks passed.

A's joint ES is 0.429958 versus C's 0.549374, but nominal 90% voltage coverage
is 28.8% versus C's 77.3%, and C's voltage CRPS is better. A is not demonstrated
to be calibrated. The probe finds no incremental benefit from adding frozen
GENCO latents to raw features. Negative results and all numerical references
are retained; no F run or post-hoc map was added.

The original main evaluation/probe completed before the additional C/P0 work.
The first P0 assertion used an unnecessarily tight tolerance for the frozen
ES count-square-root normalization (float32 versus P's float64 path); its
partial outputs are preserved. The scorer was unchanged. Final marginal and
interval differences are at most 3.41e-13; ES differs by at most 2.57e-8 relative.

On an allocated node in the existing environment:

```bash
python -m scripts.evaluate_stage_b run --device xpu
python -m scripts.embedding_probe
python -m scripts.score_controls
python -m scripts.report_stage_b
# Saved-output diagnostics; no fitting, inference, solver calls or scorer edits:
python -m scripts.diagnose_stage_b
python -m scripts.report_gate_b_diagnostics
```

The first three commands perform computation; the last validates and summarizes
saved files, runs the added point-mass invariant test, and writes Matplotlib PNGs.
Original evaluation used 12 XPU tiles; C/P0 used 12 CPU processes on a compute
node. The final report audit ran on one CPU thread after its queued allocation
was cancelled. No installed environment packages changed.
