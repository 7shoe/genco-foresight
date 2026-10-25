# Independent recovery status

Original fleet job8964643 and its own continuations remain untouched. New snapshot commit8d93abb; all outputs below are separate from fleet and released runs.

| Item | Status |
|---|---|
| Nine completed-model evaluations, summaries and mixtures | Running: debug job8964735; automatic continuation active (PID730566, detached) |
| case118 evaluation | Not started; admitted only after its original training/controller finishes |
| case500/case2000 full audit of existing labels and pilots | Submitted: debug job8964751, queued; detached controller PID733698 |
| Stage1 verdict | Selection criterion has gaps for OFF seeds0/1; test table pending; reports/v2/stage1_verdict.md records the current result |

[Live status](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/STATUS.md). [Protocol](reports/v2/recovery.protocol.json). Each family is capped at12 one-hour debug allocations; original pilot cumulative budgets are carried forward.
