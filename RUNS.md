# All experiment runs

Run directory paths below are relative to `/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs`. Each row links its exact directory, source record, configuration and outcome. Full machine-readable provenance: [run_index.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/final_consolidation_20261026_v1/run_index.json), SHA256 `96b0361f95a67d260e33bb821dbddc16bdbb43423280b8fd2a26a3343e5fecaa`.

Independent training, evaluation/mixture, data/audit and verification runs. Per-origin arrays, cache shards, temporary test fixtures and source copies are components, not independent runs.

Source IDs expand in the provenance table below. “Snapshot SHA256” hashes the linked immutable source/pin record; for older runs it hashes the recorded partial source manifest, not a reconstructed complete checkout. Its scope is explicit. A producing commit is shown only when recorded; “not recorded” is a provenance limitation, not an inferred tag/commit. Full source-map and configuration hashes are in the JSON.

Job 8965746 failed before any optimizer update; all nine child runs are labeled failed. Budget-capped chains retain that outcome. Case500 K8 and case2000 K2 pilots ended at four epochs, not converged; their evaluated snapshots are earlier. Post-freeze outputs remain separate from the 19:00 CT freeze. Stage 1 failed gates are unchanged.

| Run directory | Label | Outcome / epochs | Source: snapshot SHA / commit | Config | Job(s) |
|---|---|---|---|---|---|
| [v2/release_20261025_1420/phase_c/a0_case118_ieee_seed0](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/phase_c/a0_case118_ieee_seed0) | A0 | budget-capped; epochs 16, best 15 | [S43](#s43) `5ac13f755178` / `a76c3085f06b` | [a0_case118_ieee_seed0.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/configs/a0_case118_ieee_seed0.yaml) `d6a85dc04987` | 8965435 |
| [v2/release_20261025_1420/phase_c/a0_case30_ieee_seed0](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/phase_c/a0_case30_ieee_seed0) | A0 | complete (early stopping); epochs 14, best 11 | [S43](#s43) `5ac13f755178` / `a76c3085f06b` | [a0_case30_ieee_seed0.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/configs/a0_case30_ieee_seed0.yaml) `7937a926778e` | 8965185 |
| [v2/release_20261025_1420/phase_c/a0_case57_ieee_seed0](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/phase_c/a0_case57_ieee_seed0) | A0 | complete (early stopping); epochs 11, best 8 | [S43](#s43) `5ac13f755178` / `a76c3085f06b` | [a0_case57_ieee_seed0.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/configs/a0_case57_ieee_seed0.yaml) `5d8583f0d333` | 8965185 |
| [v2/release_20261025_1420/snapshots/a0_case118_ieee_seed0_1815](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/snapshots/a0_case118_ieee_seed0_1815) | A0 | evaluation complete | [S43](#s43) `5ac13f755178` / `a76c3085f06b` | [evaluation.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/snapshots/a0_case118_ieee_seed0_1815/evaluation.yaml) `3b3fb164a556` | 8965435 |
| [v2/release_20261025_1420/snapshots/a0_case30_ieee_seed0_1815](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/snapshots/a0_case30_ieee_seed0_1815) | A0 | evaluation complete | [S43](#s43) `5ac13f755178` / `a76c3085f06b` | [evaluation.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/snapshots/a0_case30_ieee_seed0_1815/evaluation.yaml) `851a39a3d4f4` | 8965589 |
| [v2/release_20261025_1420/snapshots/a0_case57_ieee_seed0_1815](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/snapshots/a0_case57_ieee_seed0_1815) | A0 | evaluation complete | [S43](#s43) `5ac13f755178` / `a76c3085f06b` | [evaluation.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/snapshots/a0_case57_ieee_seed0_1815/evaluation.yaml) `ea5f9e7e657a` | 8965435 |
| [v2/release_20261025_1420/ensemble/case118_ieee/seed3](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/ensemble/case118_ieee/seed3) | ensemble | budget-capped; epochs 11, best 11; post-freeze completion | [S33](#s33) `be99f936e092` / `1fec6e805c0e` | [case118_ieee_seed3.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/configs/case118_ieee_seed3.yaml) `005dc7b96298` | 8965589 |
| [v2/release_20261025_1420/ensemble/case118_ieee/seed4](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/ensemble/case118_ieee/seed4) | ensemble | budget-capped; epochs 11, best 10; post-freeze completion | [S33](#s33) `be99f936e092` / `1fec6e805c0e` | [case118_ieee_seed4.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/configs/case118_ieee_seed4.yaml) `94ded4980136` | 8965589 |
| [v2/release_20261025_1420/ensemble/case118_ieee/seed5](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/ensemble/case118_ieee/seed5) | ensemble | budget-capped; epochs 10, best 9; post-freeze completion | [S33](#s33) `be99f936e092` / `1fec6e805c0e` | [case118_ieee_seed5.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/configs/case118_ieee_seed5.yaml) `31161a0f199e` | 8965589 |
| [v2/release_20261025_1420/ensemble/case118_ieee/seed6](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/ensemble/case118_ieee/seed6) | ensemble | budget-capped; epochs 11, best 11; post-freeze completion | [S33](#s33) `be99f936e092` / `1fec6e805c0e` | [case118_ieee_seed6.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/configs/case118_ieee_seed6.yaml) `f66175a3e204` | 8965589 |
| [v2/release_20261025_1420/ensemble/case118_ieee/seed7](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/ensemble/case118_ieee/seed7) | ensemble | budget-capped; epochs 11, best 11; post-freeze completion | [S33](#s33) `be99f936e092` / `1fec6e805c0e` | [case118_ieee_seed7.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/configs/case118_ieee_seed7.yaml) `ca591fef2820` | 8965589 |
| [v2/release_20261025_1420/ensemble/case14_ieee/seed3](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/ensemble/case14_ieee/seed3) | ensemble | complete (early stopping); epochs 14, best 11 | [S33](#s33) `be99f936e092` / `1fec6e805c0e` | [case14_ieee_seed3.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/configs/case14_ieee_seed3.yaml) `3931aa01a7ec` | 8965319 |
| [v2/release_20261025_1420/ensemble/case14_ieee/seed4](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/ensemble/case14_ieee/seed4) | ensemble | complete (early stopping); epochs 13, best 10 | [S33](#s33) `be99f936e092` / `1fec6e805c0e` | [case14_ieee_seed4.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/configs/case14_ieee_seed4.yaml) `843fa38ebf28` | 8965319 |
| [v2/release_20261025_1420/ensemble/case14_ieee/seed5](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/ensemble/case14_ieee/seed5) | ensemble | complete (early stopping); epochs 22, best 19; post-freeze completion | [S33](#s33) `be99f936e092` / `1fec6e805c0e` | [case14_ieee_seed5.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/configs/case14_ieee_seed5.yaml) `8e6890a20c76` | 8965589 |
| [v2/release_20261025_1420/ensemble/case14_ieee/seed6](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/ensemble/case14_ieee/seed6) | ensemble | complete (early stopping); epochs 11, best 8 | [S33](#s33) `be99f936e092` / `1fec6e805c0e` | [case14_ieee_seed6.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/configs/case14_ieee_seed6.yaml) `8a96ef051fa7` | 8965319 |
| [v2/release_20261025_1420/ensemble/case14_ieee/seed7](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/ensemble/case14_ieee/seed7) | ensemble | complete (early stopping); epochs 8, best 5 | [S33](#s33) `be99f936e092` / `1fec6e805c0e` | [case14_ieee_seed7.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/configs/case14_ieee_seed7.yaml) `05c12dee3027` | 8965185 |
| [v2/release_20261025_1420/ensemble/case30_ieee/seed3](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/ensemble/case30_ieee/seed3) | ensemble | complete (early stopping); epochs 20, best 17 | [S33](#s33) `be99f936e092` / `1fec6e805c0e` | [case30_ieee_seed3.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/configs/case30_ieee_seed3.yaml) `e510b050fd4e` | 8965589 |
| [v2/release_20261025_1420/ensemble/case30_ieee/seed4](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/ensemble/case30_ieee/seed4) | ensemble | budget-capped; epochs 24, best 24; post-freeze completion | [S33](#s33) `be99f936e092` / `1fec6e805c0e` | [case30_ieee_seed4.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/configs/case30_ieee_seed4.yaml) `5a38d78f7eb6` | 8965589 |
| [v2/release_20261025_1420/ensemble/case30_ieee/seed5](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/ensemble/case30_ieee/seed5) | ensemble | complete (early stopping); epochs 15, best 12 | [S33](#s33) `be99f936e092` / `1fec6e805c0e` | [case30_ieee_seed5.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/configs/case30_ieee_seed5.yaml) `07eec8bf5f94` | 8965435 |
| [v2/release_20261025_1420/ensemble/case30_ieee/seed6](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/ensemble/case30_ieee/seed6) | ensemble | complete (early stopping); epochs 14, best 11 | [S33](#s33) `be99f936e092` / `1fec6e805c0e` | [case30_ieee_seed6.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/configs/case30_ieee_seed6.yaml) `eff61ab1a77c` | 8965319 |
| [v2/release_20261025_1420/ensemble/case30_ieee/seed7](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/ensemble/case30_ieee/seed7) | ensemble | budget-capped; epochs 23, best 23; post-freeze completion | [S33](#s33) `be99f936e092` / `1fec6e805c0e` | [case30_ieee_seed7.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/configs/case30_ieee_seed7.yaml) `8c156af46eed` | 8965589 |
| [v2/release_20261025_1420/ensemble/case57_ieee/seed3](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/ensemble/case57_ieee/seed3) | ensemble | budget-capped; epochs 21, best 21; post-freeze completion | [S33](#s33) `be99f936e092` / `1fec6e805c0e` | [case57_ieee_seed3.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/configs/case57_ieee_seed3.yaml) `30faf4196ac0` | 8965589 |
| [v2/release_20261025_1420/ensemble/case57_ieee/seed4](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/ensemble/case57_ieee/seed4) | ensemble | complete (early stopping); epochs 16, best 13 | [S33](#s33) `be99f936e092` / `1fec6e805c0e` | [case57_ieee_seed4.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/configs/case57_ieee_seed4.yaml) `55656a57c3f4` | 8965435 |
| [v2/release_20261025_1420/ensemble/case57_ieee/seed5](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/ensemble/case57_ieee/seed5) | ensemble | complete (early stopping); epochs 15, best 12 | [S33](#s33) `be99f936e092` / `1fec6e805c0e` | [case57_ieee_seed5.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/configs/case57_ieee_seed5.yaml) `0a83da60a1f2` | 8965435 |
| [v2/release_20261025_1420/ensemble/case57_ieee/seed6](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/ensemble/case57_ieee/seed6) | ensemble | complete (early stopping); epochs 17, best 14 | [S33](#s33) `be99f936e092` / `1fec6e805c0e` | [case57_ieee_seed6.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/configs/case57_ieee_seed6.yaml) `b59efeeb8591` | 8965435 |
| [v2/release_20261025_1420/ensemble/case57_ieee/seed7](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/ensemble/case57_ieee/seed7) | ensemble | budget-capped; epochs 21, best 19; post-freeze completion | [S33](#s33) `be99f936e092` / `1fec6e805c0e` | [case57_ieee_seed7.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/configs/case57_ieee_seed7.yaml) `c2aeb4577765` | 8965589 |
| [v2/release_20261025_1420/evaluation/case14_ieee/seed3](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/evaluation/case14_ieee/seed3) | ensemble | evaluation complete | [S38](#s38) `6839ea6aa25a` / `8d93abb18139` | [evaluation.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/evaluation/case14_ieee/seed3/evaluation.yaml) `40d87ab4dd46` | 8965435 |
| [v2/release_20261025_1420/evaluation/case14_ieee/seed4](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/evaluation/case14_ieee/seed4) | ensemble | evaluation complete | [S38](#s38) `6839ea6aa25a` / `8d93abb18139` | [evaluation.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/evaluation/case14_ieee/seed4/evaluation.yaml) `918756c4da48` | 8965319 |
| [v2/release_20261025_1420/evaluation/case14_ieee/seed6](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/evaluation/case14_ieee/seed6) | ensemble | evaluation complete | [S38](#s38) `6839ea6aa25a` / `8d93abb18139` | [evaluation.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/evaluation/case14_ieee/seed6/evaluation.yaml) `45cf4b358c5c` | 8965319 |
| [v2/release_20261025_1420/evaluation/case14_ieee/seed7](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/evaluation/case14_ieee/seed7) | ensemble | evaluation complete | [S38](#s38) `6839ea6aa25a` / `8d93abb18139` | [evaluation.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/evaluation/case14_ieee/seed7/evaluation.yaml) `3819e482440b` | 8965185 |
| [v2/release_20261025_1420/evaluation/case30_ieee/seed3](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/evaluation/case30_ieee/seed3) | ensemble | evaluation complete | [S38](#s38) `6839ea6aa25a` / `8d93abb18139` | [evaluation.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/evaluation/case30_ieee/seed3/evaluation.yaml) `6785d27d5b26` | 8965589 |
| [v2/release_20261025_1420/evaluation/case30_ieee/seed5](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/evaluation/case30_ieee/seed5) | ensemble | evaluation complete | [S38](#s38) `6839ea6aa25a` / `8d93abb18139` | [evaluation.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/evaluation/case30_ieee/seed5/evaluation.yaml) `9a5d697b155f` | 8965435 |
| [v2/release_20261025_1420/evaluation/case30_ieee/seed6](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/evaluation/case30_ieee/seed6) | ensemble | evaluation complete | [S38](#s38) `6839ea6aa25a` / `8d93abb18139` | [evaluation.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/evaluation/case30_ieee/seed6/evaluation.yaml) `d330995a123b` | 8965319 |
| [v2/release_20261025_1420/evaluation/case57_ieee/seed4](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/evaluation/case57_ieee/seed4) | ensemble | evaluation complete | [S38](#s38) `6839ea6aa25a` / `8d93abb18139` | [evaluation.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/evaluation/case57_ieee/seed4/evaluation.yaml) `34f4fd5ee172` | 8965435 |
| [v2/release_20261025_1420/evaluation/case57_ieee/seed5](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/evaluation/case57_ieee/seed5) | ensemble | evaluation complete | [S38](#s38) `6839ea6aa25a` / `8d93abb18139` | [evaluation.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/evaluation/case57_ieee/seed5/evaluation.yaml) `f5a359112cbf` | 8965435 |
| [v2/release_20261025_1420/evaluation/case57_ieee/seed6](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/evaluation/case57_ieee/seed6) | ensemble | evaluation complete | [S38](#s38) `6839ea6aa25a` / `8d93abb18139` | [evaluation.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/evaluation/case57_ieee/seed6/evaluation.yaml) `383b1232eebb` | 8965589 |
| [v2/release_20261025_1420/mixtures/case118_ieee/size2](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/mixtures/case118_ieee/size2) | ensemble | evaluation complete | [S38](#s38) `6839ea6aa25a` / `8d93abb18139` | [case118_ieee_seed0.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/evaluation/configs/case118_ieee_seed0.yaml) `0daf918e827a` | 8965054 |
| [v2/release_20261025_1420/mixtures/case14_ieee/size2](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/mixtures/case14_ieee/size2) | ensemble | evaluation complete | [S38](#s38) `6839ea6aa25a` / `8d93abb18139` | [case14_ieee_seed0.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/evaluation/configs/case14_ieee_seed0.yaml) `135745bb595b` | 8965054 |
| [v2/release_20261025_1420/mixtures/case14_ieee/size5](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/mixtures/case14_ieee/size5) | ensemble | evaluation complete | [S38](#s38) `6839ea6aa25a` / `8d93abb18139` | [case14_ieee_seed0.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/evaluation/configs/case14_ieee_seed0.yaml) `135745bb595b` | 8965435 |
| [v2/release_20261025_1420/mixtures/case30_ieee/size2](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/mixtures/case30_ieee/size2) | ensemble | evaluation complete | [S38](#s38) `6839ea6aa25a` / `8d93abb18139` | [case30_ieee_seed0.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/evaluation/configs/case30_ieee_seed0.yaml) `2be7d31fe821` | 8965054 |
| [v2/release_20261025_1420/mixtures/case30_ieee/size5](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/mixtures/case30_ieee/size5) | ensemble | evaluation complete | [S38](#s38) `6839ea6aa25a` / `8d93abb18139` | [case30_ieee_seed0.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/evaluation/configs/case30_ieee_seed0.yaml) `2be7d31fe821` | 8965435, 8965589 |
| [v2/release_20261025_1420/mixtures/case57_ieee/size2](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/mixtures/case57_ieee/size2) | ensemble | evaluation complete | [S38](#s38) `6839ea6aa25a` / `8d93abb18139` | [case57_ieee_seed0.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/evaluation/configs/case57_ieee_seed0.yaml) `d2393ec9f8b2` | 8965054 |
| [v2/release_20261025_1420/mixtures/case57_ieee/size5](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/mixtures/case57_ieee/size5) | ensemble | evaluation complete | [S38](#s38) `6839ea6aa25a` / `8d93abb18139` | [case57_ieee_seed0.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/evaluation/configs/case57_ieee_seed0.yaml) `d2393ec9f8b2` | 8965435, 8965589 |
| [adapter_v3/resume_test_nondeterministic_attempt/cpu/resumed](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/adapter_v3/resume_test_nondeterministic_attempt/cpu/resumed) | failed | historical failed resume mechanism attempt; epochs 0, best None | [S06](#s06) `ab98534088fa` / `commit not recorded` | [manifest.json#settings](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/adapter_v3/resume_test_nondeterministic_attempt/cpu/resumed/manifest.json) `8d6d5246d363` | 8956405 |
| [adapter_v3/resume_test_nondeterministic_attempt/cpu/uninterrupted](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/adapter_v3/resume_test_nondeterministic_attempt/cpu/uninterrupted) | failed | historical failed resume mechanism attempt; epochs 0, best None | [S07](#s07) `ab98534088fa` / `commit not recorded` | [manifest.json#settings](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/adapter_v3/resume_test_nondeterministic_attempt/cpu/uninterrupted/manifest.json) `8d6d5246d363` | 8956405 |
| [adapter_v3/resume_test_nondeterministic_attempt/xpu/resumed](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/adapter_v3/resume_test_nondeterministic_attempt/xpu/resumed) | failed | historical failed resume mechanism attempt; epochs 0, best None | [S08](#s08) `ab98534088fa` / `commit not recorded` | [manifest.json#settings](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/adapter_v3/resume_test_nondeterministic_attempt/xpu/resumed/manifest.json) `8d6d5246d363` | 8956405 |
| [adapter_v3/resume_test_nondeterministic_attempt/xpu/uninterrupted](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/adapter_v3/resume_test_nondeterministic_attempt/xpu/uninterrupted) | failed | historical failed resume mechanism attempt; epochs 0, best None | [S09](#s09) `ab98534088fa` / `commit not recorded` | [manifest.json#settings](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/adapter_v3/resume_test_nondeterministic_attempt/xpu/uninterrupted/manifest.json) `8d6d5246d363` | 8956405 |
| [v2/postfreeze_20261025_final](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/postfreeze_20261025_final) | failed | failed gate: see gate | [S36](#s36) `d17d5a1b4bfe` / `e2463b137d2b` | [protocol.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/postfreeze_20261025_final/protocol.json) `d58e2c79ca1d` | see evidence |
| [v2/postfreeze_20261025_final/ensemble/case118_ieee/seed3](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/postfreeze_20261025_final/ensemble/case118_ieee/seed3) | failed | 8965746: missing committed metrics journal; zero optimizer steps added | [S36](#s36) `d17d5a1b4bfe` / `e2463b137d2b`<br>[S33](#s33) `be99f936e092` / `1fec6e805c0e` | [case118_ieee_seed3.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/postfreeze_20261025_final/configs/case118_ieee_seed3.yaml) `53882706c7d3` | 8965746 |
| [v2/postfreeze_20261025_final/ensemble/case118_ieee/seed4](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/postfreeze_20261025_final/ensemble/case118_ieee/seed4) | failed | 8965746: missing committed metrics journal; zero optimizer steps added | [S36](#s36) `d17d5a1b4bfe` / `e2463b137d2b`<br>[S33](#s33) `be99f936e092` / `1fec6e805c0e` | [case118_ieee_seed4.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/postfreeze_20261025_final/configs/case118_ieee_seed4.yaml) `9eb4fc6fb287` | 8965746 |
| [v2/postfreeze_20261025_final/ensemble/case118_ieee/seed5](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/postfreeze_20261025_final/ensemble/case118_ieee/seed5) | failed | 8965746: missing committed metrics journal; zero optimizer steps added | [S36](#s36) `d17d5a1b4bfe` / `e2463b137d2b`<br>[S33](#s33) `be99f936e092` / `1fec6e805c0e` | [case118_ieee_seed5.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/postfreeze_20261025_final/configs/case118_ieee_seed5.yaml) `67b24f1a6ca3` | 8965746 |
| [v2/postfreeze_20261025_final/ensemble/case118_ieee/seed6](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/postfreeze_20261025_final/ensemble/case118_ieee/seed6) | failed | 8965746: missing committed metrics journal; zero optimizer steps added | [S36](#s36) `d17d5a1b4bfe` / `e2463b137d2b`<br>[S33](#s33) `be99f936e092` / `1fec6e805c0e` | [case118_ieee_seed6.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/postfreeze_20261025_final/configs/case118_ieee_seed6.yaml) `78c024bc7c5c` | 8965746 |
| [v2/postfreeze_20261025_final/ensemble/case118_ieee/seed7](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/postfreeze_20261025_final/ensemble/case118_ieee/seed7) | failed | 8965746: missing committed metrics journal; zero optimizer steps added | [S36](#s36) `d17d5a1b4bfe` / `e2463b137d2b`<br>[S33](#s33) `be99f936e092` / `1fec6e805c0e` | [case118_ieee_seed7.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/postfreeze_20261025_final/configs/case118_ieee_seed7.yaml) `065a12d78c5f` | 8965746 |
| [v2/postfreeze_20261025_final/ensemble/case30_ieee/seed4](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/postfreeze_20261025_final/ensemble/case30_ieee/seed4) | failed | 8965746: missing committed metrics journal; zero optimizer steps added | [S36](#s36) `d17d5a1b4bfe` / `e2463b137d2b`<br>[S33](#s33) `be99f936e092` / `1fec6e805c0e` | [case30_ieee_seed4.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/postfreeze_20261025_final/configs/case30_ieee_seed4.yaml) `2b252fbb212b` | 8965746 |
| [v2/postfreeze_20261025_final/ensemble/case30_ieee/seed7](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/postfreeze_20261025_final/ensemble/case30_ieee/seed7) | failed | 8965746: missing committed metrics journal; zero optimizer steps added | [S36](#s36) `d17d5a1b4bfe` / `e2463b137d2b`<br>[S33](#s33) `be99f936e092` / `1fec6e805c0e` | [case30_ieee_seed7.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/postfreeze_20261025_final/configs/case30_ieee_seed7.yaml) `15988d48a694` | 8965746 |
| [v2/postfreeze_20261025_final/ensemble/case57_ieee/seed3](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/postfreeze_20261025_final/ensemble/case57_ieee/seed3) | failed | 8965746: missing committed metrics journal; zero optimizer steps added | [S36](#s36) `d17d5a1b4bfe` / `e2463b137d2b`<br>[S33](#s33) `be99f936e092` / `1fec6e805c0e` | [case57_ieee_seed3.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/postfreeze_20261025_final/configs/case57_ieee_seed3.yaml) `6fed1fbd1029` | 8965746 |
| [v2/postfreeze_20261025_final/ensemble/case57_ieee/seed7](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/postfreeze_20261025_final/ensemble/case57_ieee/seed7) | failed | 8965746: missing committed metrics journal; zero optimizer steps added | [S36](#s36) `d17d5a1b4bfe` / `e2463b137d2b`<br>[S33](#s33) `be99f936e092` / `1fec6e805c0e` | [case57_ieee_seed7.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/postfreeze_20261025_final/configs/case57_ieee_seed7.yaml) `08f28fd64eb5` | 8965746 |
| [v2/stage1_20261025T0605Z](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/stage1_20261025T0605Z) | failed | stopped after failed acceptance gate | [S47](#s47) `487f4345e85c` / `commit not recorded` | [configuration.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/stage1_20261025T0605Z/configuration.json) `2ab16d45a091` | 8962445 |
| [v2/stage1_20261025T0605Z/off_a](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/stage1_20261025T0605Z/off_a) | failed | stopped after failed acceptance gate | [S47](#s47) `487f4345e85c` / `commit not recorded` | [configuration.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/stage1_20261025T0605Z/configuration.json) `2ab16d45a091` | see evidence |
| [v2/stage1_20261025T0605Z/off_b](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/stage1_20261025T0605Z/off_b) | failed | stopped after failed acceptance gate | [S47](#s47) `487f4345e85c` / `commit not recorded` | [configuration.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/stage1_20261025T0605Z/configuration.json) `2ab16d45a091` | see evidence |
| [v2/stage1_20261025T0605Z/on_reference](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/stage1_20261025T0605Z/on_reference) | failed | stopped after failed acceptance gate | [S47](#s47) `487f4345e85c` / `commit not recorded` | [configuration.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/stage1_20261025T0605Z/configuration.json) `2ab16d45a091` | see evidence |
| [v2/stage1_confirmation_20261025_v3](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/stage1_confirmation_20261025_v3) | failed | stopped after failed acceptance gate | [S48](#s48) `e667cf31cdda` / `commit not recorded` | [configuration.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/stage1_confirmation_20261025_v3/configuration.json) `12e0344c41d6` | 8962843 |
| [v2/stage1_confirmation_20261025_v3/off_a](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/stage1_confirmation_20261025_v3/off_a) | failed | stopped after failed acceptance gate | [S48](#s48) `e667cf31cdda` / `commit not recorded` | [configuration.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/stage1_confirmation_20261025_v3/configuration.json) `12e0344c41d6` | see evidence |
| [v2/stage1_confirmation_20261025_v3/off_b](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/stage1_confirmation_20261025_v3/off_b) | failed | stopped after failed acceptance gate; epochs 0, best None | [S48](#s48) `e667cf31cdda` / `commit not recorded` | [configuration.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/stage1_confirmation_20261025_v3/configuration.json) `12e0344c41d6` | 8962843 |
| [v2/stage1_epoch_20261025_v2](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/stage1_epoch_20261025_v2) | failed | stopped after failed acceptance gate | [S50](#s50) `45867f268707` / `commit not recorded` | [configuration.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/stage1_epoch_20261025_v2/configuration.json) `9d917e70fbe2` | 8962608 |
| [v2/stage1_epoch_20261025_v2/off_a](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/stage1_epoch_20261025_v2/off_a) | failed | stopped after failed acceptance gate | [S50](#s50) `45867f268707` / `commit not recorded` | [configuration.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/stage1_epoch_20261025_v2/configuration.json) `9d917e70fbe2` | see evidence |
| [v2/stage1_epoch_20261025_v2/off_b](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/stage1_epoch_20261025_v2/off_b) | failed | stopped after failed acceptance gate; epochs 0, best None | [S50](#s50) `45867f268707` / `commit not recorded` | [configuration.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/stage1_epoch_20261025_v2/configuration.json) `9d917e70fbe2` | 8962608 |
| [v2/stage1_epoch_20261025_v2/on_reference](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/stage1_epoch_20261025_v2/on_reference) | failed | stopped after failed acceptance gate | [S50](#s50) `45867f268707` / `commit not recorded` | [configuration.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/stage1_epoch_20261025_v2/configuration.json) `9d917e70fbe2` | see evidence |
| [v2/fleet_20261025_off_v1](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/fleet_20261025_off_v1) | fleet | campaign ended; child records below | [S33](#s33) `be99f936e092` / `1fec6e805c0e` | [protocol.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/fleet_20261025_off_v1/protocol.json) `e7b79bbcc777` | see evidence |
| [v2/fleet_20261025_off_v1/case118_ieee/data](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/fleet_20261025_off_v1/case118_ieee/data) | fleet | audited | [S33](#s33) `be99f936e092` / `1fec6e805c0e` | [case118_ieee_data.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/fleet_20261025_off_v1/configs/case118_ieee_data.yaml) `6586a533867e` | see evidence |
| [v2/fleet_20261025_off_v1/case118_ieee/seed0](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/fleet_20261025_off_v1/case118_ieee/seed0) | fleet | budget-capped (six_hour_budget); epochs 16, best 14 | [S33](#s33) `be99f936e092` / `1fec6e805c0e` | [case118_ieee_seed0.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/fleet_20261025_off_v1/configs/case118_ieee_seed0.yaml) `ac7a1a7022ae` | 8964740 |
| [v2/fleet_20261025_off_v1/case118_ieee/seed1](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/fleet_20261025_off_v1/case118_ieee/seed1) | fleet | budget-capped (six_hour_budget); epochs 15, best 13 | [S33](#s33) `be99f936e092` / `1fec6e805c0e` | [case118_ieee_seed1.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/fleet_20261025_off_v1/configs/case118_ieee_seed1.yaml) `467d6edd110d` | 8964740 |
| [v2/fleet_20261025_off_v1/case118_ieee/seed2](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/fleet_20261025_off_v1/case118_ieee/seed2) | fleet | budget-capped (six_hour_budget); epochs 16, best 15 | [S33](#s33) `be99f936e092` / `1fec6e805c0e` | [case118_ieee_seed2.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/fleet_20261025_off_v1/configs/case118_ieee_seed2.yaml) `f005f9ceaae8` | 8964740 |
| [v2/fleet_20261025_off_v1/case118_ieee/stage_a](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/fleet_20261025_off_v1/case118_ieee/stage_a) | fleet | audited | [S33](#s33) `be99f936e092` / `1fec6e805c0e` | [case118_ieee_data.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/fleet_20261025_off_v1/configs/case118_ieee_data.yaml) `6586a533867e` | see evidence |
| [v2/fleet_20261025_off_v1/case14_ieee/data](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/fleet_20261025_off_v1/case14_ieee/data) | fleet | audited | [S33](#s33) `be99f936e092` / `1fec6e805c0e` | [case14_ieee_data.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/fleet_20261025_off_v1/configs/case14_ieee_data.yaml) `8eb648790e47` | see evidence |
| [v2/fleet_20261025_off_v1/case14_ieee/seed0](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/fleet_20261025_off_v1/case14_ieee/seed0) | fleet | early_stopping; epochs 10, best 7 | [S33](#s33) `be99f936e092` / `1fec6e805c0e` | [case14_ieee_seed0.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/fleet_20261025_off_v1/configs/case14_ieee_seed0.yaml) `dab8bb81486f` | 8963810 |
| [v2/fleet_20261025_off_v1/case14_ieee/seed1](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/fleet_20261025_off_v1/case14_ieee/seed1) | fleet | early_stopping; epochs 13, best 10 | [S33](#s33) `be99f936e092` / `1fec6e805c0e` | [case14_ieee_seed1.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/fleet_20261025_off_v1/configs/case14_ieee_seed1.yaml) `2103690e4fa8` | 8963810 |
| [v2/fleet_20261025_off_v1/case14_ieee/seed2](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/fleet_20261025_off_v1/case14_ieee/seed2) | fleet | early_stopping; epochs 8, best 5 | [S33](#s33) `be99f936e092` / `1fec6e805c0e` | [case14_ieee_seed2.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/fleet_20261025_off_v1/configs/case14_ieee_seed2.yaml) `58f19d6ee16f` | 8963510 |
| [v2/fleet_20261025_off_v1/case14_ieee/stage_a](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/fleet_20261025_off_v1/case14_ieee/stage_a) | fleet | audited | [S33](#s33) `be99f936e092` / `1fec6e805c0e` | [case14_ieee_data.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/fleet_20261025_off_v1/configs/case14_ieee_data.yaml) `8eb648790e47` | see evidence |
| [v2/fleet_20261025_off_v1/case30_ieee/seed0](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/fleet_20261025_off_v1/case30_ieee/seed0) | fleet | early_stopping; epochs 25, best 22 | [S33](#s33) `be99f936e092` / `1fec6e805c0e` | [case30_ieee_seed0.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/fleet_20261025_off_v1/configs/case30_ieee_seed0.yaml) `f0a155add371` | 8964232 |
| [v2/fleet_20261025_off_v1/case30_ieee/seed1](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/fleet_20261025_off_v1/case30_ieee/seed1) | fleet | early_stopping; epochs 19, best 16 | [S33](#s33) `be99f936e092` / `1fec6e805c0e` | [case30_ieee_seed1.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/fleet_20261025_off_v1/configs/case30_ieee_seed1.yaml) `6a98497cc030` | 8963984 |
| [v2/fleet_20261025_off_v1/case30_ieee/seed2](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/fleet_20261025_off_v1/case30_ieee/seed2) | fleet | early_stopping; epochs 21, best 18 | [S33](#s33) `be99f936e092` / `1fec6e805c0e` | [case30_ieee_seed2.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/fleet_20261025_off_v1/configs/case30_ieee_seed2.yaml) `c24944976063` | 8964070 |
| [v2/fleet_20261025_off_v1/case30_ieee/stage_a](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/fleet_20261025_off_v1/case30_ieee/stage_a) | fleet | audited | [S33](#s33) `be99f936e092` / `1fec6e805c0e` | [case30_ieee_data.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/fleet_20261025_off_v1/configs/case30_ieee_data.yaml) `4ebb73ed9aa9` | see evidence |
| [v2/fleet_20261025_off_v1/case57_ieee/data](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/fleet_20261025_off_v1/case57_ieee/data) | fleet | audited | [S33](#s33) `be99f936e092` / `1fec6e805c0e` | [case57_ieee_data.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/fleet_20261025_off_v1/configs/case57_ieee_data.yaml) `ae4d8f1a6584` | see evidence |
| [v2/fleet_20261025_off_v1/case57_ieee/seed0](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/fleet_20261025_off_v1/case57_ieee/seed0) | fleet | early_stopping; epochs 15, best 12 | [S33](#s33) `be99f936e092` / `1fec6e805c0e` | [case57_ieee_seed0.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/fleet_20261025_off_v1/configs/case57_ieee_seed0.yaml) `b86a065589ef` | 8963984 |
| [v2/fleet_20261025_off_v1/case57_ieee/seed1](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/fleet_20261025_off_v1/case57_ieee/seed1) | fleet | early_stopping; epochs 14, best 11 | [S33](#s33) `be99f936e092` / `1fec6e805c0e` | [case57_ieee_seed1.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/fleet_20261025_off_v1/configs/case57_ieee_seed1.yaml) `d52bfea48357` | 8963984 |
| [v2/fleet_20261025_off_v1/case57_ieee/seed2](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/fleet_20261025_off_v1/case57_ieee/seed2) | fleet | early_stopping; epochs 21, best 18 | [S33](#s33) `be99f936e092` / `1fec6e805c0e` | [case57_ieee_seed2.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/fleet_20261025_off_v1/configs/case57_ieee_seed2.yaml) `27062f52377f` | 8964232 |
| [v2/fleet_20261025_off_v1/case57_ieee/stage_a](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/fleet_20261025_off_v1/case57_ieee/stage_a) | fleet | audited | [S33](#s33) `be99f936e092` / `1fec6e805c0e` | [case57_ieee_data.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/fleet_20261025_off_v1/configs/case57_ieee_data.yaml) `ae4d8f1a6584` | see evidence |
| [v2/recovery_20261025_v1](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1) | fleet | completed; historical evidence | [S38](#s38) `6839ea6aa25a` / `8d93abb18139` | [protocol.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/protocol.json) `e4fe6d06c284` | see evidence |
| [v2/recovery_20261025_v1/evaluation/case118_ieee/mixture](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/evaluation/case118_ieee/mixture) | fleet | evaluation complete | [S38](#s38) `6839ea6aa25a` / `8d93abb18139` | [protocol.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/protocol.json) `e4fe6d06c284` | see evidence |
| [v2/recovery_20261025_v1/evaluation/case118_ieee/seed0](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/evaluation/case118_ieee/seed0) | fleet | evaluation complete | [S38](#s38) `6839ea6aa25a` / `8d93abb18139` | [case118_ieee_seed0.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/evaluation/configs/case118_ieee_seed0.yaml) `0daf918e827a` | see evidence |
| [v2/recovery_20261025_v1/evaluation/case118_ieee/seed1](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/evaluation/case118_ieee/seed1) | fleet | evaluation complete | [S38](#s38) `6839ea6aa25a` / `8d93abb18139` | [case118_ieee_seed1.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/evaluation/configs/case118_ieee_seed1.yaml) `868c4d91ddeb` | see evidence |
| [v2/recovery_20261025_v1/evaluation/case118_ieee/seed2](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/evaluation/case118_ieee/seed2) | fleet | evaluation complete | [S38](#s38) `6839ea6aa25a` / `8d93abb18139` | [case118_ieee_seed2.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/evaluation/configs/case118_ieee_seed2.yaml) `484726bf3298` | see evidence |
| [v2/recovery_20261025_v1/evaluation/case14_ieee/mixture](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/evaluation/case14_ieee/mixture) | fleet | evaluation complete | [S38](#s38) `6839ea6aa25a` / `8d93abb18139` | [protocol.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/protocol.json) `e4fe6d06c284` | see evidence |
| [v2/recovery_20261025_v1/evaluation/case14_ieee/seed0](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/evaluation/case14_ieee/seed0) | fleet | evaluation complete | [S38](#s38) `6839ea6aa25a` / `8d93abb18139` | [case14_ieee_seed0.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/evaluation/configs/case14_ieee_seed0.yaml) `135745bb595b` | see evidence |
| [v2/recovery_20261025_v1/evaluation/case14_ieee/seed1](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/evaluation/case14_ieee/seed1) | fleet | evaluation complete | [S38](#s38) `6839ea6aa25a` / `8d93abb18139` | [case14_ieee_seed1.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/evaluation/configs/case14_ieee_seed1.yaml) `38ec06d032a1` | see evidence |
| [v2/recovery_20261025_v1/evaluation/case14_ieee/seed2](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/evaluation/case14_ieee/seed2) | fleet | evaluation complete | [S38](#s38) `6839ea6aa25a` / `8d93abb18139` | [case14_ieee_seed2.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/evaluation/configs/case14_ieee_seed2.yaml) `b78dd6d97d57` | see evidence |
| [v2/recovery_20261025_v1/evaluation/case30_ieee/mixture](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/evaluation/case30_ieee/mixture) | fleet | evaluation complete | [S38](#s38) `6839ea6aa25a` / `8d93abb18139` | [protocol.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/protocol.json) `e4fe6d06c284` | see evidence |
| [v2/recovery_20261025_v1/evaluation/case30_ieee/seed0](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/evaluation/case30_ieee/seed0) | fleet | evaluation complete | [S38](#s38) `6839ea6aa25a` / `8d93abb18139` | [case30_ieee_seed0.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/evaluation/configs/case30_ieee_seed0.yaml) `2be7d31fe821` | see evidence |
| [v2/recovery_20261025_v1/evaluation/case30_ieee/seed1](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/evaluation/case30_ieee/seed1) | fleet | evaluation complete | [S38](#s38) `6839ea6aa25a` / `8d93abb18139` | [case30_ieee_seed1.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/evaluation/configs/case30_ieee_seed1.yaml) `cbf0f409a551` | see evidence |
| [v2/recovery_20261025_v1/evaluation/case30_ieee/seed2](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/evaluation/case30_ieee/seed2) | fleet | evaluation complete | [S38](#s38) `6839ea6aa25a` / `8d93abb18139` | [case30_ieee_seed2.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/evaluation/configs/case30_ieee_seed2.yaml) `5bfd6403d966` | see evidence |
| [v2/recovery_20261025_v1/evaluation/case57_ieee/mixture](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/evaluation/case57_ieee/mixture) | fleet | evaluation complete | [S38](#s38) `6839ea6aa25a` / `8d93abb18139` | [protocol.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/protocol.json) `e4fe6d06c284` | see evidence |
| [v2/recovery_20261025_v1/evaluation/case57_ieee/seed0](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/evaluation/case57_ieee/seed0) | fleet | evaluation complete | [S38](#s38) `6839ea6aa25a` / `8d93abb18139` | [case57_ieee_seed0.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/evaluation/configs/case57_ieee_seed0.yaml) `d2393ec9f8b2` | see evidence |
| [v2/recovery_20261025_v1/evaluation/case57_ieee/seed1](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/evaluation/case57_ieee/seed1) | fleet | evaluation complete | [S38](#s38) `6839ea6aa25a` / `8d93abb18139` | [case57_ieee_seed1.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/evaluation/configs/case57_ieee_seed1.yaml) `29b3d615b7da` | see evidence |
| [v2/recovery_20261025_v1/evaluation/case57_ieee/seed2](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/evaluation/case57_ieee/seed2) | fleet | evaluation complete | [S38](#s38) `6839ea6aa25a` / `8d93abb18139` | [case57_ieee_seed2.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/evaluation/configs/case57_ieee_seed2.yaml) `70624bf67f16` | see evidence |
| [v2/large_grid_pilots_20261025_repair_v2](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/large_grid_pilots_20261025_repair_v2) | pilot | campaign ended; child records below | [S34](#s34) `20ce0b709d60` / `1d0ec500e35c` | [protocol.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/large_grid_pilots_20261025_repair_v2/protocol.json) `83f76885910f` | see evidence |
| [v2/large_grid_pilots_20261025_repair_v2/case2000_goc/data](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/large_grid_pilots_20261025_repair_v2/case2000_goc/data) | pilot | preparation attempt; see evidence | [S34](#s34) `20ce0b709d60` / `1d0ec500e35c` | [case2000_goc_data.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/large_grid_pilots_20261025_repair_v2/configs/case2000_goc_data.yaml) `7adac8e92477` | see evidence |
| [v2/large_grid_pilots_20261025_repair_v2/case2000_goc/stage_a](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/large_grid_pilots_20261025_repair_v2/case2000_goc/stage_a) | pilot | preparation attempt; see evidence | [S34](#s34) `20ce0b709d60` / `1d0ec500e35c` | [case2000_goc_data.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/large_grid_pilots_20261025_repair_v2/configs/case2000_goc_data.yaml) `7adac8e92477` | see evidence |
| [v2/large_grid_pilots_20261025_repair_v2/case500_goc/data](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/large_grid_pilots_20261025_repair_v2/case500_goc/data) | pilot | preparation attempt; see evidence | [S34](#s34) `20ce0b709d60` / `1d0ec500e35c` | [case500_goc_data.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/large_grid_pilots_20261025_repair_v2/configs/case500_goc_data.yaml) `987231c62da5` | see evidence |
| [v2/large_grid_pilots_20261025_repair_v2/case500_goc/stage_a](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/large_grid_pilots_20261025_repair_v2/case500_goc/stage_a) | pilot | preparation attempt; see evidence | [S34](#s34) `20ce0b709d60` / `1d0ec500e35c` | [case500_goc_data.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/large_grid_pilots_20261025_repair_v2/configs/case500_goc_data.yaml) `987231c62da5` | see evidence |
| [v2/large_grid_pilots_20261025_v1](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/large_grid_pilots_20261025_v1) | pilot | campaign ended; child records below | [S35](#s35) `821b0460dddf` / `a1d073030583` | [protocol.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/large_grid_pilots_20261025_v1/protocol.json) `2c94249e53a0` | see evidence |
| [v2/large_grid_pilots_20261025_v1/case2000_goc/data](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/large_grid_pilots_20261025_v1/case2000_goc/data) | pilot | preparation attempt; see evidence | [S35](#s35) `821b0460dddf` / `a1d073030583` | [case2000_goc_data.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/large_grid_pilots_20261025_v1/configs/case2000_goc_data.yaml) `82de00c7433a` | see evidence |
| [v2/large_grid_pilots_20261025_v1/case2000_goc/stage_a](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/large_grid_pilots_20261025_v1/case2000_goc/stage_a) | pilot | preparation attempt; see evidence | [S35](#s35) `821b0460dddf` / `a1d073030583` | [case2000_goc_data.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/large_grid_pilots_20261025_v1/configs/case2000_goc_data.yaml) `82de00c7433a` | see evidence |
| [v2/large_grid_pilots_20261025_v1/case500_goc/data](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/large_grid_pilots_20261025_v1/case500_goc/data) | pilot | preparation attempt; see evidence | [S35](#s35) `821b0460dddf` / `a1d073030583` | [case500_goc_data.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/large_grid_pilots_20261025_v1/configs/case500_goc_data.yaml) `67606a8177d7` | see evidence |
| [v2/large_grid_pilots_20261025_v1/case500_goc/stage_a](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/large_grid_pilots_20261025_v1/case500_goc/stage_a) | pilot | preparation attempt; see evidence | [S35](#s35) `821b0460dddf` / `a1d073030583` | [case500_goc_data.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/large_grid_pilots_20261025_v1/configs/case500_goc_data.yaml) `67606a8177d7` | see evidence |
| [v2/recovery_20261025_v1/pilots/case2000_goc/data](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/pilots/case2000_goc/data) | pilot | audited | [S37](#s37) `6839ea6aa25a` / `8d93abb18139` | [case2000_goc_data.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/pilots/configs/case2000_goc_data.yaml) `2bb65c976cff` | see evidence |
| [v2/recovery_20261025_v1/pilots/case2000_goc/seed0](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/pilots/case2000_goc/seed0) | pilot | stopped at freeze; not converged; epochs 4, best 3 | [S37](#s37) `6839ea6aa25a` / `8d93abb18139` | [case2000_goc_seed0.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/pilots/configs/case2000_goc_seed0.yaml) `3d88af0dc3a2` | 8965630 |
| [v2/recovery_20261025_v1/pilots/case2000_goc/stage_a](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/pilots/case2000_goc/stage_a) | pilot | audited | [S37](#s37) `6839ea6aa25a` / `8d93abb18139` | [case2000_goc_data.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/pilots/configs/case2000_goc_data.yaml) `2bb65c976cff` | see evidence |
| [v2/recovery_20261025_v1/pilots/case500_goc/data](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/pilots/case500_goc/data) | pilot | audited | [S37](#s37) `6839ea6aa25a` / `8d93abb18139` | [case500_goc_data.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/pilots/configs/case500_goc_data.yaml) `a63f34019dca` | see evidence |
| [v2/recovery_20261025_v1/pilots/case500_goc/seed0](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/pilots/case500_goc/seed0) | pilot | stopped at freeze; not converged; epochs 4, best 4 | [S37](#s37) `6839ea6aa25a` / `8d93abb18139` | [case500_goc_seed0.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/pilots/configs/case500_goc_seed0.yaml) `57691b7a4477` | 8965630 |
| [v2/recovery_20261025_v1/pilots/case500_goc/stage_a](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/pilots/case500_goc/stage_a) | pilot | audited | [S37](#s37) `6839ea6aa25a` / `8d93abb18139` | [case500_goc_data.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/pilots/configs/case500_goc_data.yaml) `a63f34019dca` | see evidence |
| [v2/release_20261025_1420/snapshots/pilot_case2000_goc_1745](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/snapshots/pilot_case2000_goc_1745) | pilot | evaluation complete | [S44](#s44) `64c084c05244` / `d1b3071fc714` | [evaluation.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/snapshots/pilot_case2000_goc_1745/evaluation.yaml) `c97aa18779b2` | 8965435 |
| [v2/release_20261025_1420/snapshots/pilot_case500_goc_1745](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/snapshots/pilot_case500_goc_1745) | pilot | evaluation complete | [S44](#s44) `64c084c05244` / `d1b3071fc714` | [evaluation.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/snapshots/pilot_case500_goc_1745/evaluation.yaml) `ca318ee0b35e` | 8965435 |
| [v2/postfreeze_20261025_final/evaluation/case14_ieee/seed5](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/postfreeze_20261025_final/evaluation/case14_ieee/seed5) | post-freeze | evaluation complete | [S38](#s38) `6839ea6aa25a` / `8d93abb18139`<br>[S36](#s36) `d17d5a1b4bfe` / `e2463b137d2b` | [evaluation.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/postfreeze_20261025_final/evaluation/case14_ieee/seed5/evaluation.yaml) `c330f3fc324b`<br>[spec.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/postfreeze_20261025_final/evaluation/case14_ieee/seed5/spec.json) `8f148f84168c` | 8965775 |
| [v2/postfreeze_20261025_final/mixtures/case118_ieee/size5](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/postfreeze_20261025_final/mixtures/case118_ieee/size5) | post-freeze | unavailable: failed member continuation excluded | [S36](#s36) `d17d5a1b4bfe` / `e2463b137d2b` | [protocol.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/postfreeze_20261025_final/protocol.json) `d58e2c79ca1d` | 8965775 |
| [v2/postfreeze_20261025_final/mixtures/case118_ieee/size8](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/postfreeze_20261025_final/mixtures/case118_ieee/size8) | post-freeze | unavailable: failed member continuation excluded | [S36](#s36) `d17d5a1b4bfe` / `e2463b137d2b` | [protocol.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/postfreeze_20261025_final/protocol.json) `d58e2c79ca1d` | 8965775 |
| [v2/postfreeze_20261025_final/mixtures/case14_ieee/size5](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/postfreeze_20261025_final/mixtures/case14_ieee/size5) | post-freeze | evaluation complete | [S38](#s38) `6839ea6aa25a` / `8d93abb18139`<br>[S36](#s36) `d17d5a1b4bfe` / `e2463b137d2b` | [case14_ieee_seed0.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/evaluation/configs/case14_ieee_seed0.yaml) `135745bb595b`<br>[spec.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/postfreeze_20261025_final/mixtures/case14_ieee/size5/spec.json) `beaa5dad11ed` | 8965775 |
| [v2/postfreeze_20261025_final/mixtures/case14_ieee/size8](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/postfreeze_20261025_final/mixtures/case14_ieee/size8) | post-freeze | evaluation complete | [S38](#s38) `6839ea6aa25a` / `8d93abb18139`<br>[S36](#s36) `d17d5a1b4bfe` / `e2463b137d2b` | [case14_ieee_seed0.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/evaluation/configs/case14_ieee_seed0.yaml) `135745bb595b`<br>[spec.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/postfreeze_20261025_final/mixtures/case14_ieee/size8/spec.json) `3830716c7698` | 8965775 |
| [v2/postfreeze_20261025_final/mixtures/case30_ieee/size5](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/postfreeze_20261025_final/mixtures/case30_ieee/size5) | post-freeze | unavailable: failed member continuation excluded | [S36](#s36) `d17d5a1b4bfe` / `e2463b137d2b` | [protocol.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/postfreeze_20261025_final/protocol.json) `d58e2c79ca1d` | 8965775 |
| [v2/postfreeze_20261025_final/mixtures/case30_ieee/size8](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/postfreeze_20261025_final/mixtures/case30_ieee/size8) | post-freeze | unavailable: failed member continuation excluded | [S36](#s36) `d17d5a1b4bfe` / `e2463b137d2b` | [protocol.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/postfreeze_20261025_final/protocol.json) `d58e2c79ca1d` | 8965775 |
| [v2/postfreeze_20261025_final/mixtures/case57_ieee/size5](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/postfreeze_20261025_final/mixtures/case57_ieee/size5) | post-freeze | unavailable: failed member continuation excluded | [S36](#s36) `d17d5a1b4bfe` / `e2463b137d2b` | [protocol.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/postfreeze_20261025_final/protocol.json) `d58e2c79ca1d` | 8965775 |
| [v2/postfreeze_20261025_final/mixtures/case57_ieee/size8](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/postfreeze_20261025_final/mixtures/case57_ieee/size8) | post-freeze | unavailable: failed member continuation excluded | [S36](#s36) `d17d5a1b4bfe` / `e2463b137d2b` | [protocol.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/postfreeze_20261025_final/protocol.json) `d58e2c79ca1d` | 8965775 |
| [adapter_v3](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/adapter_v3) | released | budget-capped (six_hour_budget); epochs 19, best 18 | [S01](#s01) `1c4e80a3d21e` / `commit not recorded` | [manifest.json#settings](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/adapter_v3/manifest.json) `8d6d5246d363` | 8957025 |
| [adapter_v4](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/adapter_v4) | released | early_stopping; epochs 17, best 14 | [S10](#s10) `943f81d38dd7` / `commit not recorded` | [manifest.json#settings](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/adapter_v4/manifest.json) `93103b5cf9f2` | 8958590 |
| [stage_a](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/stage_a) | released | completed; historical evidence | [S31](#s31) `8545c072cf24` / `commit not recorded` | [config_frozen.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/stage_a_v3/config_frozen.yaml) `017409360101` | see evidence |
| [stage_a_v2](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/stage_a_v2) | released | completed; historical evidence | [S31](#s31) `8545c072cf24` / `commit not recorded` | [config_frozen.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/stage_a_v2/config_frozen.yaml) `731c3cf6115d` | see evidence |
| [stage_a_v3](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/stage_a_v3) | released | completed; historical evidence | [S31](#s31) `8545c072cf24` / `commit not recorded` | [config_frozen.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/stage_a_v3/config_frozen.yaml) `017409360101` | see evidence |
| [stage_a_v4](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/stage_a_v4) | released | completed; historical evidence | [S31](#s31) `8545c072cf24` / `commit not recorded` | [config_frozen.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/stage_a_v4/config_frozen.yaml) `4dcc765cd3c7` | see evidence |
| [stage_b](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/stage_b) | released | completed; historical evidence | [S29](#s29) `ba7083a12928` / `commit not recorded` | [protocol.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/stage_b/protocol.json) `ba7083a12928` | 8957225 |
| [stage_c](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/stage_c) | released | completed; historical evidence | [S30](#s30) `5f7863ff0697` / `commit not recorded` | [protocol.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/stage_c/protocol.json) `5f7863ff0697` | see evidence |
| [stage_d](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/stage_d) | released | completed; historical evidence | [S31](#s31) `8545c072cf24` / `commit not recorded` | [config_frozen.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/stage_a_v3/config_frozen.yaml) `017409360101` | see evidence |
| [stage_d/final_release](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/stage_d/final_release) | released | completed; historical evidence | [S31](#s31) `8545c072cf24` / `commit not recorded` | [config_frozen.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/stage_a_v3/config_frozen.yaml) `017409360101` | see evidence |
| [v4/stage_b](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v4/stage_b) | released | completed; historical evidence | [S52](#s52) `92833f98b674` / `commit not recorded` | [protocol.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v4/stage_b/protocol.json) `92833f98b674` | see evidence |
| [replication/case118_ieee](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/replication/case118_ieee) | replication | blocked; ['data/loader/network.py:13,20 explicitly loads/requires case30', 'solver/adapter.py:30,87 fixes the casefile and requires 30 demands', 'scripts/parity.py:47,69,124,136 fixes case30 reference paths', 'training/inputs.py:43 fixes 30 buses, 6 generators and embedding width 384', 'models/temporal.py:41,75 fixes dimensions; released and immutable', 'training/loop.py:53,71,95 fixes 192 graphs, 24 hours, 30 buses and 6 generators; K=4 / 12-hour sub-sampling cannot run unchanged'] | [S28](#s28) `fe8561945523` / `commit not recorded` | [protocol.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/replication/protocol.json) `58277558a1ea` | see evidence |
| [replication/case14_ieee](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/replication/case14_ieee) | replication | blocked; ['data/loader/network.py:13,20 explicitly loads/requires case30', 'solver/adapter.py:30,87 fixes the casefile and requires 30 demands', 'scripts/parity.py:47,69,124,136 fixes case30 reference paths', 'training/inputs.py:43 fixes 30 buses, 6 generators and embedding width 384', 'models/temporal.py:41,75 fixes dimensions; released and immutable', 'training/loop.py:53,71,95 fixes 192 graphs, 24 hours, 30 buses and 6 generators; K=4 / 12-hour sub-sampling cannot run unchanged'] | [S28](#s28) `fe8561945523` / `commit not recorded` | [protocol.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/replication/protocol.json) `58277558a1ea` | see evidence |
| [replication/case30_ieee/reproduction](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/replication/case30_ieee/reproduction) | replication | passed | [S28](#s28) `fe8561945523` / `commit not recorded` | [protocol.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/replication/protocol.json) `58277558a1ea` | see evidence |
| [replication/case30_ieee/v3/seed1](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/replication/case30_ieee/v3/seed1) | replication | early_stopping; epochs 19, best 16 | [S16](#s16) `602c619d3434` / `commit not recorded` | [manifest.json#settings](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/replication/case30_ieee/v3/seed1/adapter/manifest.json) `8d6d5246d363` | 8960079 |
| [replication/case30_ieee/v3/seed2](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/replication/case30_ieee/v3/seed2) | replication | budget-capped (six_hour_budget); epochs 19, best 18 | [S19](#s19) `74b91a7881c3` / `commit not recorded` | [manifest.json#settings](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/replication/case30_ieee/v3/seed2/adapter/manifest.json) `8d6d5246d363` | 8960079 |
| [replication/case30_ieee/v4/seed1](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/replication/case30_ieee/v4/seed1) | replication | budget-capped (six_hour_budget); epochs 19, best 19 | [S22](#s22) `16ddf4586c82` / `commit not recorded` | [manifest.json#settings](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/replication/case30_ieee/v4/seed1/adapter/manifest.json) `93103b5cf9f2` | 8960079 |
| [replication/case30_ieee/v4/seed2](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/replication/case30_ieee/v4/seed2) | replication | early_stopping; epochs 17, best 14 | [S25](#s25) `e54642b8fb22` / `commit not recorded` | [manifest.json#settings](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/replication/case30_ieee/v4/seed2/adapter/manifest.json) `93103b5cf9f2` | 8960006 |
| [replication/case500_goc](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/replication/case500_goc) | replication | blocked; ['data/loader/network.py:13,20 explicitly loads/requires case30', 'solver/adapter.py:30,87 fixes the casefile and requires 30 demands', 'scripts/parity.py:47,69,124,136 fixes case30 reference paths', 'training/inputs.py:43 fixes 30 buses, 6 generators and embedding width 384', 'models/temporal.py:41,75 fixes dimensions; released and immutable', 'training/loop.py:53,71,95 fixes 192 graphs, 24 hours, 30 buses and 6 generators; K=4 / 12-hour sub-sampling cannot run unchanged'] | [S28](#s28) `fe8561945523` / `commit not recorded` | [protocol.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/replication/protocol.json) `58277558a1ea` | see evidence |
| [replication/case57_ieee](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/replication/case57_ieee) | replication | blocked; ['data/loader/network.py:13,20 explicitly loads/requires case30', 'solver/adapter.py:30,87 fixes the casefile and requires 30 demands', 'scripts/parity.py:47,69,124,136 fixes case30 reference paths', 'training/inputs.py:43 fixes 30 buses, 6 generators and embedding width 384', 'models/temporal.py:41,75 fixes dimensions; released and immutable', 'training/loop.py:53,71,95 fixes 192 graphs, 24 hours, 30 buses and 6 generators; K=4 / 12-hour sub-sampling cannot run unchanged'] | [S28](#s28) `fe8561945523` / `commit not recorded` | [protocol.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/replication/protocol.json) `58277558a1ea` | see evidence |
| [adapter_v3/resume_test/cpu/resumed](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/adapter_v3/resume_test/cpu/resumed) | verification | resume_test_step_limit; epochs 0, best None | [S02](#s02) `90d341481cc2` / `commit not recorded` | [manifest.json#settings](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/adapter_v3/resume_test/cpu/resumed/manifest.json) `8d6d5246d363` | 8956434 |
| [adapter_v3/resume_test/cpu/uninterrupted](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/adapter_v3/resume_test/cpu/uninterrupted) | verification | resume_test_step_limit; epochs 0, best None | [S03](#s03) `90d341481cc2` / `commit not recorded` | [manifest.json#settings](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/adapter_v3/resume_test/cpu/uninterrupted/manifest.json) `8d6d5246d363` | 8956434 |
| [adapter_v3/resume_test/xpu/resumed](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/adapter_v3/resume_test/xpu/resumed) | verification | resume_test_step_limit; epochs 0, best None | [S04](#s04) `90d341481cc2` / `commit not recorded` | [manifest.json#settings](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/adapter_v3/resume_test/xpu/resumed/manifest.json) `8d6d5246d363` | 8956434 |
| [adapter_v3/resume_test/xpu/uninterrupted](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/adapter_v3/resume_test/xpu/uninterrupted) | verification | resume_test_step_limit; epochs 0, best None | [S05](#s05) `90d341481cc2` / `commit not recorded` | [manifest.json#settings](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/adapter_v3/resume_test/xpu/uninterrupted/manifest.json) `8d6d5246d363` | 8956434 |
| [adapter_v4/resume_test/cpu/resumed](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/adapter_v4/resume_test/cpu/resumed) | verification | resume_test_step_limit; epochs 0, best None | [S11](#s11) `943f81d38dd7` / `commit not recorded` | [manifest.json#settings](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/adapter_v4/resume_test/cpu/resumed/manifest.json) `93103b5cf9f2` | 8957792 |
| [adapter_v4/resume_test/cpu/uninterrupted](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/adapter_v4/resume_test/cpu/uninterrupted) | verification | resume_test_step_limit; epochs 0, best None | [S12](#s12) `943f81d38dd7` / `commit not recorded` | [manifest.json#settings](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/adapter_v4/resume_test/cpu/uninterrupted/manifest.json) `93103b5cf9f2` | 8957792 |
| [adapter_v4/resume_test/xpu/resumed](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/adapter_v4/resume_test/xpu/resumed) | verification | resume_test_step_limit; epochs 0, best None | [S13](#s13) `943f81d38dd7` / `commit not recorded` | [manifest.json#settings](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/adapter_v4/resume_test/xpu/resumed/manifest.json) `93103b5cf9f2` | 8957792 |
| [adapter_v4/resume_test/xpu/uninterrupted](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/adapter_v4/resume_test/xpu/uninterrupted) | verification | resume_test_step_limit; epochs 0, best None | [S14](#s14) `943f81d38dd7` / `commit not recorded` | [manifest.json#settings](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/adapter_v4/resume_test/xpu/uninterrupted/manifest.json) `93103b5cf9f2` | 8957792 |
| [multigrid/profile/20261025T041341Z/concurrent_1](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/multigrid/profile/20261025T041341Z/concurrent_1) | verification | synthetic profile: OK | [S15](#s15) `efafb2c0fb9e` / `d28b9e29527a` | [summary.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/multigrid/profile/20261025T041341Z/summary.json) `efafb2c0fb9e` | 8962223 |
| [multigrid/profile/20261025T041341Z/concurrent_12](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/multigrid/profile/20261025T041341Z/concurrent_12) | verification | synthetic profile: OK | [S15](#s15) `efafb2c0fb9e` / `d28b9e29527a` | [summary.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/multigrid/profile/20261025T041341Z/summary.json) `efafb2c0fb9e` | 8962223 |
| [multigrid/profile/20261025T041341Z/concurrent_4](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/multigrid/profile/20261025T041341Z/concurrent_4) | verification | synthetic profile: OK | [S15](#s15) `efafb2c0fb9e` / `d28b9e29527a` | [summary.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/multigrid/profile/20261025T041341Z/summary.json) `efafb2c0fb9e` | 8962223 |
| [multigrid/profile/20261025T041341Z/concurrent_8](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/multigrid/profile/20261025T041341Z/concurrent_8) | verification | synthetic profile: OK | [S15](#s15) `efafb2c0fb9e` / `d28b9e29527a` | [summary.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/multigrid/profile/20261025T041341Z/summary.json) `efafb2c0fb9e` | 8962223 |
| [multigrid/profile/20261025T041341Z/cpu_case30_ieee](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/multigrid/profile/20261025T041341Z/cpu_case30_ieee) | verification | synthetic profile: OK | [S15](#s15) `efafb2c0fb9e` / `d28b9e29527a` | [summary.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/multigrid/profile/20261025T041341Z/summary.json) `efafb2c0fb9e` | 8962223 |
| [multigrid/profile/20261025T041341Z/deterministic_case118_ieee](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/multigrid/profile/20261025T041341Z/deterministic_case118_ieee) | verification | synthetic profile: OK | [S15](#s15) `efafb2c0fb9e` / `d28b9e29527a` | [summary.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/multigrid/profile/20261025T041341Z/summary.json) `efafb2c0fb9e` | 8962223 |
| [multigrid/profile/20261025T041341Z/deterministic_case30_ieee](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/multigrid/profile/20261025T041341Z/deterministic_case30_ieee) | verification | synthetic profile: OK | [S15](#s15) `efafb2c0fb9e` / `d28b9e29527a` | [summary.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/multigrid/profile/20261025T041341Z/summary.json) `efafb2c0fb9e` | 8962223 |
| [multigrid/profile/20261025T041341Z/solo_case118_ieee_k8](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/multigrid/profile/20261025T041341Z/solo_case118_ieee_k8) | verification | synthetic profile: OK | [S15](#s15) `efafb2c0fb9e` / `d28b9e29527a` | [summary.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/multigrid/profile/20261025T041341Z/summary.json) `efafb2c0fb9e` | 8962223 |
| [multigrid/profile/20261025T041341Z/solo_case14_ieee_k8](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/multigrid/profile/20261025T041341Z/solo_case14_ieee_k8) | verification | synthetic profile: OK | [S15](#s15) `efafb2c0fb9e` / `d28b9e29527a` | [summary.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/multigrid/profile/20261025T041341Z/summary.json) `efafb2c0fb9e` | 8962223 |
| [multigrid/profile/20261025T041341Z/solo_case2000_goc_k2](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/multigrid/profile/20261025T041341Z/solo_case2000_goc_k2) | verification | synthetic profile: OK | [S15](#s15) `efafb2c0fb9e` / `d28b9e29527a` | [summary.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/multigrid/profile/20261025T041341Z/summary.json) `efafb2c0fb9e` | 8962223 |
| [multigrid/profile/20261025T041341Z/solo_case2000_goc_k4](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/multigrid/profile/20261025T041341Z/solo_case2000_goc_k4) | verification | synthetic profile: OOM | [S15](#s15) `efafb2c0fb9e` / `d28b9e29527a` | [summary.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/multigrid/profile/20261025T041341Z/summary.json) `efafb2c0fb9e` | 8962223 |
| [multigrid/profile/20261025T041341Z/solo_case2000_goc_k8](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/multigrid/profile/20261025T041341Z/solo_case2000_goc_k8) | verification | synthetic profile: ERROR | [S15](#s15) `efafb2c0fb9e` / `d28b9e29527a` | [summary.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/multigrid/profile/20261025T041341Z/summary.json) `efafb2c0fb9e` | 8962223 |
| [multigrid/profile/20261025T041341Z/solo_case30_ieee_k8](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/multigrid/profile/20261025T041341Z/solo_case30_ieee_k8) | verification | synthetic profile: OK | [S15](#s15) `efafb2c0fb9e` / `d28b9e29527a` | [summary.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/multigrid/profile/20261025T041341Z/summary.json) `efafb2c0fb9e` | 8962223 |
| [multigrid/profile/20261025T041341Z/solo_case500_goc_k8](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/multigrid/profile/20261025T041341Z/solo_case500_goc_k8) | verification | synthetic profile: OK | [S15](#s15) `efafb2c0fb9e` / `d28b9e29527a` | [summary.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/multigrid/profile/20261025T041341Z/summary.json) `efafb2c0fb9e` | 8962223 |
| [multigrid/profile/20261025T041341Z/solo_case57_ieee_k8](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/multigrid/profile/20261025T041341Z/solo_case57_ieee_k8) | verification | synthetic profile: OK | [S15](#s15) `efafb2c0fb9e` / `d28b9e29527a` | [summary.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/multigrid/profile/20261025T041341Z/summary.json) `efafb2c0fb9e` | 8962223 |
| [multigrid/profile/20261025T041341Z/validation_case118_ieee](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/multigrid/profile/20261025T041341Z/validation_case118_ieee) | verification | synthetic profile: OK | [S15](#s15) `efafb2c0fb9e` / `d28b9e29527a` | [summary.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/multigrid/profile/20261025T041341Z/summary.json) `efafb2c0fb9e` | 8962223 |
| [multigrid/profile/20261025T041341Z/validation_case14_ieee](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/multigrid/profile/20261025T041341Z/validation_case14_ieee) | verification | synthetic profile: OK | [S15](#s15) `efafb2c0fb9e` / `d28b9e29527a` | [summary.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/multigrid/profile/20261025T041341Z/summary.json) `efafb2c0fb9e` | 8962223 |
| [multigrid/profile/20261025T041341Z/validation_case2000_goc](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/multigrid/profile/20261025T041341Z/validation_case2000_goc) | verification | synthetic profile: OK | [S15](#s15) `efafb2c0fb9e` / `d28b9e29527a` | [summary.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/multigrid/profile/20261025T041341Z/summary.json) `efafb2c0fb9e` | 8962223 |
| [multigrid/profile/20261025T041341Z/validation_case30_ieee](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/multigrid/profile/20261025T041341Z/validation_case30_ieee) | verification | synthetic profile: OK | [S15](#s15) `efafb2c0fb9e` / `d28b9e29527a` | [summary.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/multigrid/profile/20261025T041341Z/summary.json) `efafb2c0fb9e` | 8962223 |
| [multigrid/profile/20261025T041341Z/validation_case500_goc](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/multigrid/profile/20261025T041341Z/validation_case500_goc) | verification | synthetic profile: OK | [S15](#s15) `efafb2c0fb9e` / `d28b9e29527a` | [summary.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/multigrid/profile/20261025T041341Z/summary.json) `efafb2c0fb9e` | 8962223 |
| [multigrid/profile/20261025T041341Z/validation_case57_ieee](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/multigrid/profile/20261025T041341Z/validation_case57_ieee) | verification | synthetic profile: OK | [S15](#s15) `efafb2c0fb9e` / `d28b9e29527a` | [summary.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/multigrid/profile/20261025T041341Z/summary.json) `efafb2c0fb9e` | 8962223 |
| [replication/case30_ieee/v3/seed1/adapter/resume_test/xpu/resumed](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/replication/case30_ieee/v3/seed1/adapter/resume_test/xpu/resumed) | verification | resume_test_step_limit; epochs 0, best None | [S17](#s17) `602c619d3434` / `commit not recorded` | [manifest.json#settings](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/replication/case30_ieee/v3/seed1/adapter/resume_test/xpu/resumed/manifest.json) `8d6d5246d363` | 8959682 |
| [replication/case30_ieee/v3/seed1/adapter/resume_test/xpu/uninterrupted](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/replication/case30_ieee/v3/seed1/adapter/resume_test/xpu/uninterrupted) | verification | resume_test_step_limit; epochs 0, best None | [S18](#s18) `602c619d3434` / `commit not recorded` | [manifest.json#settings](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/replication/case30_ieee/v3/seed1/adapter/resume_test/xpu/uninterrupted/manifest.json) `8d6d5246d363` | 8959682 |
| [replication/case30_ieee/v3/seed2/adapter/resume_test/xpu/resumed](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/replication/case30_ieee/v3/seed2/adapter/resume_test/xpu/resumed) | verification | resume_test_step_limit; epochs 0, best None | [S20](#s20) `74b91a7881c3` / `commit not recorded` | [manifest.json#settings](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/replication/case30_ieee/v3/seed2/adapter/resume_test/xpu/resumed/manifest.json) `8d6d5246d363` | 8959682 |
| [replication/case30_ieee/v3/seed2/adapter/resume_test/xpu/uninterrupted](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/replication/case30_ieee/v3/seed2/adapter/resume_test/xpu/uninterrupted) | verification | resume_test_step_limit; epochs 0, best None | [S21](#s21) `74b91a7881c3` / `commit not recorded` | [manifest.json#settings](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/replication/case30_ieee/v3/seed2/adapter/resume_test/xpu/uninterrupted/manifest.json) `8d6d5246d363` | 8959682 |
| [replication/case30_ieee/v4/seed1/adapter/resume_test/xpu/resumed](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/replication/case30_ieee/v4/seed1/adapter/resume_test/xpu/resumed) | verification | resume_test_step_limit; epochs 0, best None | [S23](#s23) `16ddf4586c82` / `commit not recorded` | [manifest.json#settings](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/replication/case30_ieee/v4/seed1/adapter/resume_test/xpu/resumed/manifest.json) `93103b5cf9f2` | 8959682 |
| [replication/case30_ieee/v4/seed1/adapter/resume_test/xpu/uninterrupted](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/replication/case30_ieee/v4/seed1/adapter/resume_test/xpu/uninterrupted) | verification | resume_test_step_limit; epochs 0, best None | [S24](#s24) `16ddf4586c82` / `commit not recorded` | [manifest.json#settings](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/replication/case30_ieee/v4/seed1/adapter/resume_test/xpu/uninterrupted/manifest.json) `93103b5cf9f2` | 8959682 |
| [replication/case30_ieee/v4/seed2/adapter/resume_test/xpu/resumed](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/replication/case30_ieee/v4/seed2/adapter/resume_test/xpu/resumed) | verification | resume_test_step_limit; epochs 0, best None | [S26](#s26) `e54642b8fb22` / `commit not recorded` | [manifest.json#settings](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/replication/case30_ieee/v4/seed2/adapter/resume_test/xpu/resumed/manifest.json) `93103b5cf9f2` | 8959682 |
| [replication/case30_ieee/v4/seed2/adapter/resume_test/xpu/uninterrupted](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/replication/case30_ieee/v4/seed2/adapter/resume_test/xpu/uninterrupted) | verification | resume_test_step_limit; epochs 0, best None | [S27](#s27) `e54642b8fb22` / `commit not recorded` | [manifest.json#settings](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/replication/case30_ieee/v4/seed2/adapter/resume_test/xpu/uninterrupted/manifest.json) `93103b5cf9f2` | 8959682 |
| [v2/final_consolidation_20261026_v1](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/final_consolidation_20261026_v1) | verification | passed | [S32](#s32) `168e7c1c16f2` / `8987ed24356f` | [protocol.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/final_consolidation_20261026_v1/protocol.json) `04df32de0fc9` | 8966156 |
| [v2/release_20261025_1420](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420) | verification | campaign ended; child records below | [S39](#s39) `75f60e506702` / `cee0ef6ac6aa` | [protocol.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/protocol.json) `476d8dd1e3db` | see evidence |
| [v2/release_20261025_1420/phase_b/multigrid](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/phase_b/multigrid) | verification | complete | [S40](#s40) `557ec93fd4c1` / `184b3c0f88fd` | [phase_b.protocol.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/phase_b.protocol.json) `1a8702d2466d` | 8965054 |
| [v2/release_20261025_1420/phase_b/recovery](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/phase_b/recovery) | verification | complete | [S41](#s41) `080243972495` / `a2c72bd4231a` | [phase_b.protocol.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/phase_b.protocol.json) `1a8702d2466d` | 8965054 |
| [v2/release_20261025_1420/phase_b/replication](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/phase_b/replication) | verification | complete | [S42](#s42) `1f1947a5f6bd` / `a6ae620264ff` | [phase_b.protocol.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/phase_b.protocol.json) `1a8702d2466d` | 8965054 |
| [v2/release_20261025_1420/phase_c/default_regression](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/phase_c/default_regression) | verification | complete | [S43](#s43) `5ac13f755178` / `a76c3085f06b` | [phase_b.protocol.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/phase_b.protocol.json) `1a8702d2466d` | 8965054 |
| [v2/release_20261025_1420/phase_c/gate](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/phase_c/gate) | verification | complete | [S43](#s43) `5ac13f755178` / `a76c3085f06b` | [phase_c.protocol.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/phase_c.protocol.json) `f8d5a2fba09a` | 8965054 |
| [v2/stage0_20261025T052938Z](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/stage0_20261025T052938Z) | verification | passed | [S45](#s45) `786b6ee28ebb` / `commit not recorded` | [configuration.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/stage0_20261025T052938Z/configuration.json) `d350414ca915` | 8962405 |
| [v2/stage0_20261025T052938Z/altered_fixture](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/stage0_20261025T052938Z/altered_fixture) | verification | deliberately altered comparison fixture; expected rejection; epochs 0, best None | [S45](#s45) `786b6ee28ebb` / `commit not recorded` | [configuration.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/stage0_20261025T052938Z/configuration.json) `d350414ca915` | 8962405 |
| [v2/stage0_20261025T052938Z/on_a](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/stage0_20261025T052938Z/on_a) | verification | verification_step_limit; epochs 0, best None | [S45](#s45) `786b6ee28ebb` / `commit not recorded` | [configuration.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/stage0_20261025T052938Z/configuration.json) `d350414ca915` | 8962405 |
| [v2/stage0_20261025T052938Z/on_b](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/stage0_20261025T052938Z/on_b) | verification | verification_step_limit; epochs 0, best None | [S45](#s45) `786b6ee28ebb` / `commit not recorded` | [configuration.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/stage0_20261025T052938Z/configuration.json) `d350414ca915` | 8962405 |
| [v2/stage1_20261025T0600Z](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/stage1_20261025T0600Z) | verification | completed; historical evidence | [S46](#s46) `17d41cd5750c` / `commit not recorded` | [configuration.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/stage1_20261025T0600Z/configuration.json) `d5fa1ea1dfa4` | 8962442 |
| [v2/stage1_20261025T0605Z/mechanism/xpu/resumed](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/stage1_20261025T0605Z/mechanism/xpu/resumed) | verification | resume_test_step_limit; epochs 0, best None | [S47](#s47) `487f4345e85c` / `commit not recorded` | [configuration.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/stage1_20261025T0605Z/configuration.json) `2ab16d45a091` | 8962445 |
| [v2/stage1_20261025T0605Z/mechanism/xpu/uninterrupted](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/stage1_20261025T0605Z/mechanism/xpu/uninterrupted) | verification | resume_test_step_limit; epochs 0, best None | [S47](#s47) `487f4345e85c` / `commit not recorded` | [configuration.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/stage1_20261025T0605Z/configuration.json) `2ab16d45a091` | 8962445 |
| [v2/stage1_confirmation_20261025_v3/mechanism/xpu/resumed](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/stage1_confirmation_20261025_v3/mechanism/xpu/resumed) | verification | resume_test_step_limit; epochs 0, best None | [S48](#s48) `e667cf31cdda` / `commit not recorded` | [configuration.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/stage1_confirmation_20261025_v3/configuration.json) `12e0344c41d6` | 8962843 |
| [v2/stage1_confirmation_20261025_v3/mechanism/xpu/uninterrupted](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/stage1_confirmation_20261025_v3/mechanism/xpu/uninterrupted) | verification | resume_test_step_limit; epochs 0, best None | [S48](#s48) `e667cf31cdda` / `commit not recorded` | [configuration.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/stage1_confirmation_20261025_v3/configuration.json) `12e0344c41d6` | 8962843 |
| [v2/stage1_diagnosis_20261025_v1](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/stage1_diagnosis_20261025_v1) | verification | completed; historical evidence | [S49](#s49) `389ca43c7c36` / `commit not recorded` | [configuration.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/stage1_diagnosis_20261025_v1/configuration.json) `1a5d4929d66f` | 8962527 |
| [v2/stage1_epoch_20261025_v2/mechanism/xpu/resumed](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/stage1_epoch_20261025_v2/mechanism/xpu/resumed) | verification | resume_test_step_limit; epochs 0, best None | [S50](#s50) `45867f268707` / `commit not recorded` | [configuration.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/stage1_epoch_20261025_v2/configuration.json) `9d917e70fbe2` | 8962608 |
| [v2/stage1_epoch_20261025_v2/mechanism/xpu/uninterrupted](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/stage1_epoch_20261025_v2/mechanism/xpu/uninterrupted) | verification | resume_test_step_limit; epochs 0, best None | [S50](#s50) `45867f268707` / `commit not recorded` | [configuration.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/stage1_epoch_20261025_v2/configuration.json) `9d917e70fbe2` | 8962608 |
| [v2/subset200_20261025_v1/case118_ieee](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/subset200_20261025_v1/case118_ieee) | verification | subset parity (200) passed | [S51](#s51) `ef37740ce6d3` / `4e37e2c46fe7` | [protocol.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/subset200_20261025_v1/case118_ieee/protocol.json) `9e9665b2b6aa` | see evidence |
| [v2/subset200_20261025_v1/case2000_goc](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/subset200_20261025_v1/case2000_goc) | verification | subset parity (200) passed | [S51](#s51) `ef37740ce6d3` / `4e37e2c46fe7` | [protocol.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/subset200_20261025_v1/case2000_goc/protocol.json) `8df971a7244d` | see evidence |
| [v2/subset200_20261025_v1/case500_goc](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/subset200_20261025_v1/case500_goc) | verification | subset parity (200) passed | [S51](#s51) `ef37740ce6d3` / `4e37e2c46fe7` | [protocol.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/subset200_20261025_v1/case500_goc/protocol.json) `998ed60d058c` | see evidence |
| [v2/release_20261025_1420/phase_c/lambda01_case30_ieee_seed0](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/phase_c/lambda01_case30_ieee_seed0) | λ | complete (early stopping); epochs 15, best 12 | [S43](#s43) `5ac13f755178` / `a76c3085f06b` | [lambda01_case30_ieee_seed0.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/configs/lambda01_case30_ieee_seed0.yaml) `d8968d807f26` | 8965435 |
| [v2/release_20261025_1420/phase_c/lambda10_case30_ieee_seed0](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/phase_c/lambda10_case30_ieee_seed0) | λ | complete (early stopping); epochs 15, best 12 | [S43](#s43) `5ac13f755178` / `a76c3085f06b` | [lambda10_case30_ieee_seed0.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/configs/lambda10_case30_ieee_seed0.yaml) `fab4e1e0a29b` | 8965435 |
| [v2/release_20261025_1420/snapshots/lambda01_case30_ieee_seed0_1815](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/snapshots/lambda01_case30_ieee_seed0_1815) | λ | evaluation complete | [S43](#s43) `5ac13f755178` / `a76c3085f06b` | [evaluation.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/snapshots/lambda01_case30_ieee_seed0_1815/evaluation.yaml) `92ce5088cb52` | 8965589 |
| [v2/release_20261025_1420/snapshots/lambda10_case30_ieee_seed0_1815](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/snapshots/lambda10_case30_ieee_seed0_1815) | λ | evaluation complete | [S43](#s43) `5ac13f755178` / `a76c3085f06b` | [evaluation.yaml](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/snapshots/lambda10_case30_ieee_seed0_1815/evaluation.yaml) `a156d9b4b261` | 8965589 |

## Source snapshots and producing commits

### S01

- Record: [manifest.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/adapter_v3/manifest.json)
- Snapshot SHA256: `1c4e80a3d21e7727e037e74596e00e1a93173fac1d426e2f0969cf8da7edde89`
- Producing commit: **not recorded**.
- Scope: recorded source hashes; scope may be partial; 5 recorded file hashes.

### S02

- Record: [manifest.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/adapter_v3/resume_test/cpu/resumed/manifest.json)
- Snapshot SHA256: `90d341481cc2025a95e65fcde97b4c3509e9e73316b32b79a9edebc17128651e`
- Producing commit: **not recorded**.
- Scope: recorded source hashes; scope may be partial; 5 recorded file hashes.

### S03

- Record: [manifest.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/adapter_v3/resume_test/cpu/uninterrupted/manifest.json)
- Snapshot SHA256: `90d341481cc2025a95e65fcde97b4c3509e9e73316b32b79a9edebc17128651e`
- Producing commit: **not recorded**.
- Scope: recorded source hashes; scope may be partial; 5 recorded file hashes.

### S04

- Record: [manifest.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/adapter_v3/resume_test/xpu/resumed/manifest.json)
- Snapshot SHA256: `90d341481cc2025a95e65fcde97b4c3509e9e73316b32b79a9edebc17128651e`
- Producing commit: **not recorded**.
- Scope: recorded source hashes; scope may be partial; 5 recorded file hashes.

### S05

- Record: [manifest.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/adapter_v3/resume_test/xpu/uninterrupted/manifest.json)
- Snapshot SHA256: `90d341481cc2025a95e65fcde97b4c3509e9e73316b32b79a9edebc17128651e`
- Producing commit: **not recorded**.
- Scope: recorded source hashes; scope may be partial; 5 recorded file hashes.

### S06

- Record: [manifest.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/adapter_v3/resume_test_nondeterministic_attempt/cpu/resumed/manifest.json)
- Snapshot SHA256: `ab98534088faefaf01be1a52f7593c8c621889a4524bf5a0832d802274b46b6f`
- Producing commit: **not recorded**.
- Scope: recorded source hashes; scope may be partial; 5 recorded file hashes.

### S07

- Record: [manifest.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/adapter_v3/resume_test_nondeterministic_attempt/cpu/uninterrupted/manifest.json)
- Snapshot SHA256: `ab98534088faefaf01be1a52f7593c8c621889a4524bf5a0832d802274b46b6f`
- Producing commit: **not recorded**.
- Scope: recorded source hashes; scope may be partial; 5 recorded file hashes.

### S08

- Record: [manifest.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/adapter_v3/resume_test_nondeterministic_attempt/xpu/resumed/manifest.json)
- Snapshot SHA256: `ab98534088faefaf01be1a52f7593c8c621889a4524bf5a0832d802274b46b6f`
- Producing commit: **not recorded**.
- Scope: recorded source hashes; scope may be partial; 5 recorded file hashes.

### S09

- Record: [manifest.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/adapter_v3/resume_test_nondeterministic_attempt/xpu/uninterrupted/manifest.json)
- Snapshot SHA256: `ab98534088faefaf01be1a52f7593c8c621889a4524bf5a0832d802274b46b6f`
- Producing commit: **not recorded**.
- Scope: recorded source hashes; scope may be partial; 5 recorded file hashes.

### S10

- Record: [manifest.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/adapter_v4/manifest.json)
- Snapshot SHA256: `943f81d38dd7db32d7de9dea8c8f13c716acea85afc27d55b781107203b69bb5`
- Producing commit: **not recorded**.
- Scope: recorded source hashes; scope may be partial; 5 recorded file hashes.

### S11

- Record: [manifest.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/adapter_v4/resume_test/cpu/resumed/manifest.json)
- Snapshot SHA256: `943f81d38dd7db32d7de9dea8c8f13c716acea85afc27d55b781107203b69bb5`
- Producing commit: **not recorded**.
- Scope: recorded source hashes; scope may be partial; 5 recorded file hashes.

### S12

- Record: [manifest.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/adapter_v4/resume_test/cpu/uninterrupted/manifest.json)
- Snapshot SHA256: `943f81d38dd7db32d7de9dea8c8f13c716acea85afc27d55b781107203b69bb5`
- Producing commit: **not recorded**.
- Scope: recorded source hashes; scope may be partial; 5 recorded file hashes.

### S13

- Record: [manifest.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/adapter_v4/resume_test/xpu/resumed/manifest.json)
- Snapshot SHA256: `943f81d38dd7db32d7de9dea8c8f13c716acea85afc27d55b781107203b69bb5`
- Producing commit: **not recorded**.
- Scope: recorded source hashes; scope may be partial; 5 recorded file hashes.

### S14

- Record: [manifest.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/adapter_v4/resume_test/xpu/uninterrupted/manifest.json)
- Snapshot SHA256: `943f81d38dd7db32d7de9dea8c8f13c716acea85afc27d55b781107203b69bb5`
- Producing commit: **not recorded**.
- Scope: recorded source hashes; scope may be partial; 5 recorded file hashes.

### S15

- Record: [summary.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/multigrid/profile/20261025T041341Z/summary.json)
- Snapshot SHA256: `efafb2c0fb9e8c275adfd5433c8da302e13da0adf4805c1c49c66073889d7ef5`
- Producing commit: `d28b9e29527a0fdd1511175dca1becbef752cfb6`
- Scope: recorded source hashes; scope may be partial; 1 recorded file hashes.

### S16

- Record: [manifest.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/replication/case30_ieee/v3/seed1/adapter/manifest.json)
- Snapshot SHA256: `602c619d343471996328099d28a95c5b6ba8eac8d618e634f94898a49ada7743`
- Producing commit: **not recorded**.
- Scope: recorded source hashes; scope may be partial; 5 recorded file hashes.

### S17

- Record: [manifest.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/replication/case30_ieee/v3/seed1/adapter/resume_test/xpu/resumed/manifest.json)
- Snapshot SHA256: `602c619d343471996328099d28a95c5b6ba8eac8d618e634f94898a49ada7743`
- Producing commit: **not recorded**.
- Scope: recorded source hashes; scope may be partial; 5 recorded file hashes.

### S18

- Record: [manifest.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/replication/case30_ieee/v3/seed1/adapter/resume_test/xpu/uninterrupted/manifest.json)
- Snapshot SHA256: `602c619d343471996328099d28a95c5b6ba8eac8d618e634f94898a49ada7743`
- Producing commit: **not recorded**.
- Scope: recorded source hashes; scope may be partial; 5 recorded file hashes.

### S19

- Record: [manifest.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/replication/case30_ieee/v3/seed2/adapter/manifest.json)
- Snapshot SHA256: `74b91a7881c383fb1491c2adf7bbfc85cc638b285776b7bdffb964c140d187be`
- Producing commit: **not recorded**.
- Scope: recorded source hashes; scope may be partial; 5 recorded file hashes.

### S20

- Record: [manifest.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/replication/case30_ieee/v3/seed2/adapter/resume_test/xpu/resumed/manifest.json)
- Snapshot SHA256: `74b91a7881c383fb1491c2adf7bbfc85cc638b285776b7bdffb964c140d187be`
- Producing commit: **not recorded**.
- Scope: recorded source hashes; scope may be partial; 5 recorded file hashes.

### S21

- Record: [manifest.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/replication/case30_ieee/v3/seed2/adapter/resume_test/xpu/uninterrupted/manifest.json)
- Snapshot SHA256: `74b91a7881c383fb1491c2adf7bbfc85cc638b285776b7bdffb964c140d187be`
- Producing commit: **not recorded**.
- Scope: recorded source hashes; scope may be partial; 5 recorded file hashes.

### S22

- Record: [manifest.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/replication/case30_ieee/v4/seed1/adapter/manifest.json)
- Snapshot SHA256: `16ddf4586c820a574f3566212df1347cd01656fbc10e005da94785719cf25f22`
- Producing commit: **not recorded**.
- Scope: recorded source hashes; scope may be partial; 5 recorded file hashes.

### S23

- Record: [manifest.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/replication/case30_ieee/v4/seed1/adapter/resume_test/xpu/resumed/manifest.json)
- Snapshot SHA256: `16ddf4586c820a574f3566212df1347cd01656fbc10e005da94785719cf25f22`
- Producing commit: **not recorded**.
- Scope: recorded source hashes; scope may be partial; 5 recorded file hashes.

### S24

- Record: [manifest.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/replication/case30_ieee/v4/seed1/adapter/resume_test/xpu/uninterrupted/manifest.json)
- Snapshot SHA256: `16ddf4586c820a574f3566212df1347cd01656fbc10e005da94785719cf25f22`
- Producing commit: **not recorded**.
- Scope: recorded source hashes; scope may be partial; 5 recorded file hashes.

### S25

- Record: [manifest.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/replication/case30_ieee/v4/seed2/adapter/manifest.json)
- Snapshot SHA256: `e54642b8fb22f86114a411b36be7f71883f624e0ac91f7e103fe41416ee29033`
- Producing commit: **not recorded**.
- Scope: recorded source hashes; scope may be partial; 5 recorded file hashes.

### S26

- Record: [manifest.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/replication/case30_ieee/v4/seed2/adapter/resume_test/xpu/resumed/manifest.json)
- Snapshot SHA256: `e54642b8fb22f86114a411b36be7f71883f624e0ac91f7e103fe41416ee29033`
- Producing commit: **not recorded**.
- Scope: recorded source hashes; scope may be partial; 5 recorded file hashes.

### S27

- Record: [manifest.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/replication/case30_ieee/v4/seed2/adapter/resume_test/xpu/uninterrupted/manifest.json)
- Snapshot SHA256: `e54642b8fb22f86114a411b36be7f71883f624e0ac91f7e103fe41416ee29033`
- Producing commit: **not recorded**.
- Scope: recorded source hashes; scope may be partial; 5 recorded file hashes.

### S28

- Record: [replication_sources.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/replication/replication_sources.json)
- Snapshot SHA256: `fe85619455234cf5c2bbe1f23806414b51187eb6e0e0273210f5271f6f78d0da`
- Producing commit: **not recorded**.
- Scope: recorded source hashes; scope may be partial; 8 recorded file hashes.

### S29

- Record: [protocol.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/stage_b/protocol.json)
- Snapshot SHA256: `ba7083a12928612d339d844ccb32fe1bf2a959bc3b66913141b2e8305afb2c5c`
- Producing commit: **not recorded**.
- Scope: recorded source hashes; scope may be partial; 4 recorded file hashes.

### S30

- Record: [protocol.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/stage_c/protocol.json)
- Snapshot SHA256: `5f7863ff06977d4765d9cfd94a607b07eebb0669636c62b7c69fb203569f4def`
- Producing commit: **not recorded**.
- Scope: recorded source hashes; scope may be partial; 3 recorded file hashes.

### S31

- Record: [manifest.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/stage_d/manifest.json)
- Snapshot SHA256: `8545c072cf2497dfb30eef33bd6a58df1ab0c528d76b8ca701b3af47ab3d71db`
- Producing commit: **not recorded**.
- Scope: recorded source hashes; scope may be partial; 64 recorded file hashes.

### S32

- Record: [source_pin.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/final_consolidation_20261026_v1/source_pin.json)
- Snapshot SHA256: `168e7c1c16f278f789ed3ad5d8920a1fa28069ef0eddbdc44f3763cb276eab95`
- Producing commit: `8987ed24356f2f5d66cf60e561951459e346f90b`
- Scope: pinned snapshot record; 270 recorded file hashes.
- Snapshot location: `/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/final_consolidation_20261026_v1/source`.

### S33

- Record: [source_pin.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/fleet_20261025_off_v1/source_pin.json)
- Snapshot SHA256: `be99f936e092ae03f3529dc6aba30037cf64cf9ecb92d7606781699ea15b7b70`
- Producing commit: `1fec6e805c0eb2203c71df24ffbca3cdf426ca79`
- Scope: pinned snapshot record; 204 recorded file hashes.
- Snapshot location: `/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/fleet_20261025_off_v1/source`.
- The pin records working-tree edits; the file hashes, rather than the base commit alone, define the snapshot.

### S34

- Record: [source_pin.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/large_grid_pilots_20261025_repair_v2/source_pin.json)
- Snapshot SHA256: `20ce0b709d60c3a2e4dd448360067f414097fdd80b4cfb6cbe44c01597cd4b0c`
- Producing commit: `1d0ec500e35c6fcb8657274e18576316ffa56258`
- Scope: pinned snapshot record; 226 recorded file hashes.
- Snapshot location: `/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/large_grid_pilots_20261025_repair_v2/source`.
- The pin records working-tree edits; the file hashes, rather than the base commit alone, define the snapshot.

### S35

- Record: [source_pin.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/large_grid_pilots_20261025_v1/source_pin.json)
- Snapshot SHA256: `821b0460dddf5686fc749d1f166ee9addb68f5fb61774bf7cb11eb9156093df9`
- Producing commit: `a1d0730305837511d9592509f488e30f941344d7`
- Scope: pinned snapshot record; 212 recorded file hashes.
- Snapshot location: `/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/large_grid_pilots_20261025_v1/source`.
- The pin records working-tree edits; the file hashes, rather than the base commit alone, define the snapshot.

### S36

- Record: [pin.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/postfreeze_20261025_final/operations/pin.json)
- Snapshot SHA256: `d17d5a1b4bfea5cd25ab9485a1493bc14a31432e8805441542dbd56535b42b08`
- Producing commit: `e2463b137d2bb03f35445a3403e3fb7ff5fe9647`
- Scope: pinned snapshot record; 26 recorded file hashes.

### S37

- Record: [source_pin.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/pilots/source_pin.json)
- Snapshot SHA256: `6839ea6aa25aeca68dfab036def54e5ba57a5b504c3af47a960ff47cbe41e12c`
- Producing commit: `8d93abb18139e578d51ae8f29703723f5b3fad05`
- Scope: pinned snapshot record; 234 recorded file hashes.
- Snapshot location: `/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/source`.

### S38

- Record: [source_pin.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/source_pin.json)
- Snapshot SHA256: `6839ea6aa25aeca68dfab036def54e5ba57a5b504c3af47a960ff47cbe41e12c`
- Producing commit: `8d93abb18139e578d51ae8f29703723f5b3fad05`
- Scope: pinned snapshot record; 234 recorded file hashes.
- Snapshot location: `/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/recovery_20261025_v1/source`.

### S39

- Record: [pin.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/operations_v2/pin.json)
- Snapshot SHA256: `75f60e506702b4c343a77872e3cd855f7fd29810c6fca212b44f8d9d53b0144e`
- Producing commit: `cee0ef6ac6aa21925b368e4355e13c47456f1fcb`
- Scope: pinned snapshot record; 2 recorded file hashes.

### S40

- Record: [source_pin.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/phase_b/multigrid/source_pin.json)
- Snapshot SHA256: `557ec93fd4c18b0ecfa7c944c65f68cc7b7c527dc1f7be0670619a4573359bca`
- Producing commit: `184b3c0f88fdfd2a8868d8dd25260add78ace53b`
- Scope: recorded source hashes; scope may be partial; 197 recorded file hashes.

### S41

- Record: [source_pin.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/phase_b/recovery/source_pin.json)
- Snapshot SHA256: `0802439724959f624a1963005093af9add083b14d2730ec9d37635d82ac1297a`
- Producing commit: `a2c72bd4231a39455d4eaa03d1e9dd32f6d658c3`
- Scope: recorded source hashes; scope may be partial; 240 recorded file hashes.

### S42

- Record: [source_pin.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/phase_b/replication/source_pin.json)
- Snapshot SHA256: `1f1947a5f6bd55d24bd3ab94ea5f1aea8af19d9183e92667b522eae5ace16e73`
- Producing commit: `a6ae620264ff4cdfda9e1b4f31eeca377cb6f938`
- Scope: recorded source hashes; scope may be partial; 177 recorded file hashes.

### S43

- Record: [source_pin.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/phase_c/source_pin.json)
- Snapshot SHA256: `5ac13f75517849651f504f81a53449dcca1178f6fdfb164bfd63e7c280816f7b`
- Producing commit: `a76c3085f06bef3a68973dbb593fb7823f5644b8`
- Scope: pinned snapshot record; 250 recorded file hashes.
- Snapshot location: `/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/phase_c/source`.

### S44

- Record: [source_pin.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/scheduled_operations/source_pin.json)
- Snapshot SHA256: `64c084c052442a8f6f1ce481942cb793fb5f1c2df4b8ef15c9f2140d850130ee`
- Producing commit: `d1b3071fc7140e08d04dec5ef0a311666fbdd2fe`
- Scope: pinned snapshot record; 253 recorded file hashes.
- Snapshot location: `/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/scheduled_operations/source`.

### S45

- Record: [prepared.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/stage0_20261025T052938Z/prepared.json)
- Snapshot SHA256: `786b6ee28ebb95c7f020967ffec06de51c8175017e578b7f5b3cf70599fdb4f1`
- Producing commit: **not recorded**.
- Scope: recorded source hashes; scope may be partial; 125 recorded file hashes.

### S46

- Record: [prepared.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/stage1_20261025T0600Z/prepared.json)
- Snapshot SHA256: `17d41cd5750c700b4dc7ecc69669b8ddccbe152ffecd3ca729cb1f82c076f4e0`
- Producing commit: **not recorded**.
- Scope: recorded source hashes; scope may be partial; 133 recorded file hashes.

### S47

- Record: [prepared.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/stage1_20261025T0605Z/prepared.json)
- Snapshot SHA256: `487f4345e85c6d3943514c6c8e501248d9cf2cdee9768c990a9f65bbde0ba8cc`
- Producing commit: **not recorded**.
- Scope: recorded source hashes; scope may be partial; 133 recorded file hashes.

### S48

- Record: [prepared.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/stage1_confirmation_20261025_v3/prepared.json)
- Snapshot SHA256: `e667cf31cddaa0cae59a3a1ea03a3788fc9857ea7855926d6ee8d7ac8422c994`
- Producing commit: **not recorded**.
- Scope: recorded source hashes; scope may be partial; 142 recorded file hashes.

### S49

- Record: [prepared.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/stage1_diagnosis_20261025_v1/prepared.json)
- Snapshot SHA256: `389ca43c7c364e6e43304cebaeee31d74792192ad4eef3a345b83f8474ac5f0f`
- Producing commit: **not recorded**.
- Scope: recorded source hashes; scope may be partial; 138 recorded file hashes.
- Recorded parent/base commit: `cc1aedc75d1e835163d738b8810fe560dff0ad81`; this is not asserted to be the exact producing tree.

### S50

- Record: [prepared.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/stage1_epoch_20261025_v2/prepared.json)
- Snapshot SHA256: `45867f26870797f90da92d79966f4b46afb264ba6f3d49308e4c8f8241012b0c`
- Producing commit: **not recorded**.
- Scope: recorded source hashes; scope may be partial; 140 recorded file hashes.

### S51

- Record: [source_pin.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/subset200_20261025_v1/source_pin.json)
- Snapshot SHA256: `ef37740ce6d37e2b16d083babf3388a4d3324d40905fea46420c7ec76f2112f3`
- Producing commit: `4e37e2c46fe7d229be1e3e32cd713305e70e378b`
- Scope: pinned snapshot record; 219 recorded file hashes.
- Snapshot location: `/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/subset200_20261025_v1/source`.

### S52

- Record: [protocol.json](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v4/stage_b/protocol.json)
- Snapshot SHA256: `92833f98b674c249a108ff97141226f4f1a92c43c17a44fdf74cf7c73c1af4db`
- Producing commit: **not recorded**.
- Scope: recorded source hashes; scope may be partial; 5 recorded file hashes.

## Directory aliases

These existing symlinks name the same runs, not additional experiments.

| Alias | Canonical directory |
|---|---|
| `fleet` | `v2/fleet_20261025_off_v1` |
| `pilots` | `v2/large_grid_pilots_20261025_v1` |
| `pilots_repaired` | `v2/large_grid_pilots_20261025_repair_v2` |

Pre-v2/v3 source and configuration backups are retained in `data_v2_provenance` and `data_v3_provenance` under the run root; they are provenance archives, not extra training runs.

## Results and release scope

- [Combined mixture table](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/postfreeze_20261025_final/combined_mixture_sizes.md) and [figure](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/postfreeze_20261025_final/combined_mixture_sizes.png).
- [Cross-grid and A0/λ tables](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/cross_grid.md).
- [Original freeze inventory](/lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/freeze_inventory.json), unchanged.
- [Final continuation failure](reports/v2/postfreeze.md); merging/indexing its code does not turn its failed gate into a pass.
- [Final main regression protocol](reports/v2/final_consolidation.protocol.json).

## Package verification (no training)

The Phase 2 protocol is [recorded here](reports/package/phase2.protocol.json); results and hashes are in [the gate](reports/package/phase2.json).

| Directory under GENCO/runs | Source commit | Snapshot pin SHA256 | Config | Status | Job |
| --- | --- | --- | --- | --- | --- |
| `package/phase2_20261026_v1` | `e3ab3757c8ac4b6cfc56909928a2a2317291594e` | `c764edc06f0c278b8664d0603bcf41feab1f040c14839c524d8cd0b195876c76` | case30 v4 + recorded fleet seed0; [protocol](reports/package/phase2.protocol.json) | verification passed | 8966222 |
| `package/phase2_20261026_v1_final` | `5b0093d5285bf9f4fc5e7e5717e87b01469ae525` | `16de483ccd95ba760937406ca1593be76aa834bc234fef1ea0404f8daf5e69f2` | case30 v4 + recorded fleet seed0; [protocol](reports/package/phase2.protocol.json) | verification passed | 8966236 |
