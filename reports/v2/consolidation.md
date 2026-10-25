# Main consolidation

PASS. Replication was already an ancestor of main; its integrated state was verified first. Multigrid and recovery were then merged separately, each followed by the predeclared compute regression.

| Stage | Tests passed | Commit |
|---|---:|---|
| replication | 59 | a6ae620264ff4cdfda9e1b4f31eeca377cb6f938 |
| multigrid | 66 | 184b3c0f88fdfd2a8868d8dd25260add78ace53b |
| recovery | 87 | a2c72bd4231a39455d4eaa03d1e9dd32f6d658c3 |

Every gate recomputed the same 50 original case30 v4 hours and compared all audit errors at atol=1e-12, rtol=1e-10, then reproduced A/D arrays for the first two released adapter_v4 test origins at atol=rtol=1e-5. The original artifacts were read-only. This is a regression check, not new training acceptance.

Protocol and evidence: /lus/flare/projects/FRAME-IDP/siebenschuh/GENCO/runs/v2/release_20261025_1420/phase_b.protocol.json and phase_b/. Job: 8965054. The gates shared one compute node inside the eight-node release allocation; their cap was 1,800 seconds each.

The V2 inventory remains a historical document: 27/29 recorded hashes still match the multigrid working tree; training/loop.py and training/state.py changed in the recorded Stage 0/1 work. Later recovery source is separately pinned. Historical line references are not asserted current.

v0.1.0 and released artifacts remain unchanged. OFF is operationally adopted with the measured selection/test gaps disclosed; this does not claim statistical equivalence or reverse historical failed gates.
