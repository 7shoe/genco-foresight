# Final post-freeze continuation

One final two-node, one-hour training job resumes the nine budget-capped chains, then one two-node, one-hour evaluation-only job. No further training or job resubmission. The original 19:00 freeze remains unchanged.

New child runs explicitly extend the cumulative budget by one hour and record parent checkpoint/manifest hashes. Every scientific configuration field and all model, optimizer, RNG and progress state must match exactly before training. Patience3 remains unchanged. Actual state-transfer and standard restoration gates run on compute before optimizer updates.

Evaluation prioritizes case14 seed5 and its eight-seed mixture, then the continued checkpoints and fixed-prefix5/8 mixtures. All checkpoints are labeled with their final stop reason; budget-capped checkpoints may enter this separately authorized supplement. The frozen scorer and coordinate masks remain unchanged. Undefined metrics and allocation-limited omissions remain explicit.

[Protocol](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/postfreeze_20261025_final/protocol.json) · [Live status](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/postfreeze_20261025_final/STATUS.md) · final evidence will be written to /lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/postfreeze_20261025_final/RESULTS.md.
