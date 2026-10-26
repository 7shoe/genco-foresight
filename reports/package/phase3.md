# Phase 3: local pff CLI

Status: implementation complete; compute gates pending. No merge or release tag yet.

The `cli` branch starts at `v0.2.1-dev`. `pff` and `power-fm-foresight`
share five argparse verbs: prepare, train, evaluate, report and verify. Effective
configuration follows command flags, YAML, then the packaged v4 recipe. Commands
persist package/source/configuration identities and reject incompatible resumes.
PBS submission remains in `experiments/aurora/` outside the wheel.

The numerical training loop, optimizer, coordinate masks and physics are unchanged.
Existing preparation/parity/event helpers now have package-owned implementations.
The forecast API accepts checkpoint and sample-count arguments with historical
defaults; scorer randomness accepts an evaluation seed with its historical default.
Evaluation keeps deterministic algorithms enabled and jointly scores equal-weight
mixtures in ascending training-seed order. Reports consume persisted artifacts only.

The fixed protocol is [phase3.protocol.json](phase3.protocol.json). Acceptance is
one-sided per-epoch selection ES against the three recorded case30 fleet seeds,
with the user-declared 0.005 slack. Per-step loss differences are diagnostics.
All metrics must be finite and the step-4090 boundary restore exact. The Phase 2
audit, forward, scorer/loader and installation gates retain their tolerances.

Bulk evidence: `/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/package/phase3_20261026_v1`.
Resource cap: four debug allocations, one node and one tile each, at most one hour
per allocation and 14,400 reserved node-seconds cumulatively. No extra training
beyond the declared case30 seed0 two-epoch chain.

Reproduction commands and storage/launcher conventions are in
[experiments/README.md](../../experiments/README.md). Fresh label generation and
large-grid recipes use the existing solver/settings; the end-to-end acceptance
exercise reuses the released case30 labels read-only.
