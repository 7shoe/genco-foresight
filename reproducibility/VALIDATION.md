# Local reproducibility validation

The following section preserves the **historical v3-only check**. References to
v4 running, the old post hash and untagged source describe that earlier snapshot.
The final v3/v4 replay and release validation are recorded separately below.

## Historical v3-only check

Passed on 23 October 2026, 20:00:49–20:05:25 UTC, in **276.006 seconds**
(4 minutes 36 seconds), including interpreter startup, file verification,
rescoring and figure rendering. Execution used CPU on `aurora-uan-0012`, Python
3.12.12, vendor Torch `2.13.0a0+gitcf30153` and NumPy 2.3.5. Nothing was installed.

The tested directory is
`/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/reproducibility/v0.1.0-rc/`.
The process ran from its copied `code/` directory with `PYTHONPATH` unset;
this was a source-snapshot check in the existing Aurora environment, not a
fresh-environment or non-Aurora test.

## Checks and evidence

- All 7,092 payload hashes in `bundle.json` matched. `SHA256SUMS` additionally
  records the bundle manifest itself.
- Rescored fixed origins `test_00_0072` and `test_03_0072` for C/P/P0/S/A/D.
  All marginal metrics, component metrics, PIT histograms, joint energy scores
  and variogram scores matched saved results. Integer values matched exactly;
  floating comparisons used rtol 2e-6 and atol 2e-8.
- Rebuilt the OOD and coverage PNGs from saved full-evaluation reports and
  extracted the post's numerical evidence into `validation/headline_numbers.json`.
  The compact figure renderer reproduces the evidence, not the original PNG bytes.
- V3 evaluation's four source hashes and v4 training's five source hashes remain
  unchanged. The submission draft remains unchanged pending user review.

Machine-readable results, timings, source path and figure hashes:
[validation/result.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/reproducibility/v0.1.0-rc/validation/result.json).
The tested bundle SHA256 is
`0304d663fbb7615b10454fc4e38075dcd4ca6283a64480973131e82540b20162`.
The unchanged post SHA256 is
`6618296ecfe68ba5954be40e8b17f8c09e1d878771d9b6e9ec3af2d28fa760a5`.

## What failed or remains untested

Preflight review found that C/P0 forecasts lived in a separate controls directory
omitted by the first packaging draft. Those forecasts and their training-only
climatology pool were added, and the runner's paths were corrected before the
timed test. Relative bundle/output paths were also resolved before changing the
working directory. No frozen scorer or experiment results were changed.
The timed run completed without a runtime failure.

This check rescores saved forecasts; it does not execute GENCO inference,
retrain an adapter, regenerate OPF labels or validate a new package installation.
Independent non-Aurora reproduction has **not been run**. Full model execution
requires the separately referenced pinned base weights and its dependencies;
full data generation additionally requires Julia/Ipopt and path configuration.
V4 training artifacts are excluded while that run continues; its audited data
are included. V1/v2 summaries are included; their full datasets remain in their
original Lustre directories.

Nothing was published, uploaded, committed or tagged. The user controls any
eventual GitHub publication. See [README.md](README.md) for offline commands.

## Final combined v3/v4 saved-forecast replay

Passed 2026-10-24T06:26:31+00:00 to 2026-10-24T06:28:54.774986+00:00 in **143.775 seconds**
(about 2 minutes 24 seconds), including startup, 9,392 payload hash checks,
24 mode/origin comparisons and three figure renders. Host: `aurora-uan-0012`;
CPU, Python 3.12.12, vendor Torch `2.13.0a0+gitcf30153`, NumPy `2.3.5`.
The command ran from the copied `code/` tree with `PYTHONPATH` unset, installed
nothing and accessed no remote service.

V3 origins: `test_00_0072`, `test_03_0072`. V4 origins: `test_00_0072`,
`test_02_0240`. C/P/P0/S/A/D all matched the frozen scorer outputs: marginal
and component metrics, PIT counts, joint ES and variogram scores. Counts match
exactly; floating tolerances are rtol 2e-6 and atol 2e-8. The masked v4 origin
has 690 observed Vm coordinates instead of 720, confirming that the unresolved
state was not replaced by zeros. The comparison PNG reproduces its original
SHA256 exactly; the two compact v3 renderers reproduce the evidence, not the
original image bytes.

No timed-run failures occurred. During packaging preflight, the headline extractor
was adjusted for v3's nested profile-count schema and its absent P/P0 family-ES
fields; figure symlinks were replaced with real files inside the portable package.
These were packaging corrections, not changes to scientific models or scores.

The timed snapshot manifest SHA256 was
`d4f810c059ea8ef4b73fce6196fdbd04de49fc7f848f353d35de148241ab3e93`. Final documentation, headline provenance,
and Git release metadata are added after this replay; the payload manifest is
then resealed and validated separately. This does not claim that the final
documentation-bearing manifest was the manifest timed above. Frozen scientific
sources and rescoring/rendering code are hash-checked separately.

The headline extractor's evidence paths were improved after the timed replay;
all 346 existing values and units match that replay, with the measured runtime
added as a new key. A packaging-log count originally included the manifest as
a payload file; the log was corrected without changing any artifact hashes.

Results: `validation/final/result.json`; full log: `validation/combined-replay.log`;
tested source hashes: `validation/tested_source_sha256.json`; final document and
payload validation: `validation/final_release.json`. The measured result is also
packaged at `evidence/runs/stage_d/final_release/reproduction_result.json`.

This remains an existing-Aurora-environment saved-forecast replay. Fresh installation,
independent non-Aurora execution, fresh model inference and full regeneration
remain untested. No upload, push or publication is part of this validation.
