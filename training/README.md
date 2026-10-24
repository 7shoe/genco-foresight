# Aurora training jobs

The active A run uses the unchanged vendor Python environment and one-node
`debug` allocations, each requesting one hour. Data, cached histories, optimizer
state, checkpoints and logs stay under
`/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/adapter_v3`.
The dataset is the final `temporal_case30_v3`; v1 and v2 remain on Lustre.

## Start or resume

Run from the repository in the configured existing environment:

```bash
python -m scripts.training_chain --device xpu
```

The driver requires the v3 Stage A audit and successful restart verification.
It holds an exclusive lock, persists `chain.json` and `chain.md`, and submits at
most one dependent successor using `afterany`. If dependencies are rejected, it
waits for completion before submitting. Keep this lightweight driver running on
the login node; rerunning the same command resumes its recorded chain after an
interruption. `--once` checks/submits once for manual operation. A failed worker
stops the chain for inspection rather than silently continuing.

Do not change the frozen configuration or training source during a run. The
checkpoint manifest verifies both. A CPU run uses the same worker and driver
with `--device cpu`, in a separately configured run directory; changing device
inside the active XPU chain is rejected. CPU numerical equality with XPU is not
claimed.

## Restart contract

Frozen final GENCO embeddings, observed operating states and masks are cached
once per episode. Workers memory-map those tensors and take 72-hour views;
selection uses fixed independent sampling noise. Training uses deterministic
algorithms, `PYTHONHASHSEED=42`, and an epoch permutation derived from seed and
epoch. There is no shuffled loader cursor outside the checkpoint.

`latest.pt` is written atomically using a temporary file, flush/fsync and rename,
every 900 seconds and after each epoch. Epoch checkpoints and `best.pt` are also
retained. The portable state includes adapter parameters/buffers, optimizer,
scheduler, Python/NumPy/Torch RNG states, epoch and step counters, selection best
score and patience, cumulative wall time, and the run manifest. Serialized tensors
are on CPU. The state utility includes backbone weights when they are unfrozen,
so a future F loop can reuse it; no F training has been launched.

The production-loop test compares 200 uninterrupted XPU steps with 100 steps,
an actual SIGKILL immediately after an atomic checkpoint, and 100 resumed steps.
The final model, optimizer, scheduler and RNG states match exactly, as do the
window/noise sequence and progress counters. The CPU fallback passes a shorter
four-step test with a kill after step two. An initial nondeterministic XPU failure
is retained alongside the successful deterministic trial.

## Clocks and stopping

The PBS script records its start before module setup. The worker reads that
timestamp and `PBS_WALLTIME`, begins its final save 15 seconds before the
55-minute guard, and exits cleanly. Selection-only early stopping has patience
three. The cumulative cap is 21,600 seconds of worker-job wall time, including
startup, updates, validation and checkpointing. Queue wait, dependency wait,
cache preparation and restart tests are separate costs. Calendar elapsed starts
when the production chain is created.

The chain table sums completed jobs; an active job's live progress is in its log.
`startup.json` describes the latest worker and is overwritten on restart; the
finished per-job JSON retains that job's startup measurement.

## Evidence

Paths below are relative to the external run directory:

| Artifact | Purpose |
|---|---|
| `resume_verification.md`, `resume_test/report.json` | Restart result and detailed evidence links |
| `resume_test_nondeterministic_attempt/` | Preserved negative XPU trial |
| `manifest.json` | Frozen dataset, settings, code hashes and budget definition |
| `cache/manifest.json` | Cached episode files and content hashes |
| `chain.json`, `chain.md` | Job IDs, dependencies, nodes, progress and clocks |
| `chain_after_job_01.json`, `chain_after_job_01.md` | Retained chain-log snapshot after the first successful job |
| `jobs/<numeric-job-id>.json` | Final per-job counters, times and checksums |
| `logs/job_XX.log` | Live steps and selection scores |
| `startup.json` | Latest worker readiness and first optimizer-update timing |
| `epochs.json` | Selection score per completed epoch |
| `latest.pt`, `epoch_NNN.pt`, `best.pt` | Resume, epoch and selected model states |
| `first_periodic_checkpoint_check.json` | Periodic checkpoint readability and CPU storage-tag check |
| `job_01_final.pt`, `job_01_verification.json` | Retained handoff checkpoint and verified restart counters |

Selection energy scores guide training only. Calibration/test scores, stress
maps and the S-versus-A comparison have not yet been evaluated.
