# Stage A numerical backend

Use PowerModels/Ipopt through a persistent Julia subprocess, adapting the
minimum required source from the existing RAMBO implementation. This route
does not require a Python Ipopt binding or changes to the Python framework
packages. Sources were inspected on 23 October 2026, then the minimal adapter
was exercised on allocated Aurora compute nodes. Cold/warm and graph round-trip
solves pass independent checks. The fixed-Q perturbation validates nodal dual
sign/scaling; full temporal generation uses the same backend.

| Implementation | Source beneath `/home/siebenschuh/Projects/` | Finding |
|---|---|---|
| Standalone reference | `acopf/src/acopf/core/solvers/reference.py` | PYPOWER/PIPS and SciPy trust-constr; no Ipopt in this module. |
| SDK references | `lumina-sdk/distillation/solver/backends/reference.py` | User-reported adapted PYPOWER/SciPy implementations. |
| Sparse CPU solver | `lumina-sdk/distillation/solver/sparse_solver/solver.py` | User-reported custom PIPS-style solver; not Ipopt. |
| Selected Ipopt implementation | `lumina-sdk/distillation/RAMBO/julia/model.jl` | Inspected: `optimizer(seconds)` configures Ipopt; `final_opf(request)` calls `PowerModels.solve_opf` with `ACPPowerModel`. |
| Python bridge | `lumina-sdk/distillation/RAMBO/backend.py` | Inspected: persistent subprocess, JSON requests, request wall deadlines, process cleanup. |
| Julia worker | `lumina-sdk/distillation/RAMBO/julia/worker.jl` | Inspected: routes `stage="opf"` to `final_opf`; also exposes separate RAMBO-specific stages. |

Reuse these existing runtime locations explicitly:

```text
Executable: /lus/flare/projects/FRAME-IDP/siebenschuh/envs/rambo_julia/julia-1.10.10/bin/julia
Depot:      /lus/flare/projects/FRAME-IDP/siebenschuh/envs/rambo_julia/depot
Source:     /home/siebenschuh/Projects/lumina-sdk/distillation/RAMBO
Project:    <Source>/julia/Project.toml
Manifest:   <Source>/julia/Manifest.toml
```

The Manifest records Julia `1.10.10`, Ipopt.jl `1.16.0`, JuMP `1.32.1`, and
PowerModels `0.21.5`. Worker startup subsequently reported those same versions.
Preserve the resolved environment; do not automatically update or install
packages. Pass the
executable and depot explicitly because the original bridge can prefer a
different `julia` on `PATH`.

The original RAMBO `final_opf` wrapper returns status, success, solve time, objective,
and primal state. Its inputs include a MATPOWER casefile, network dimensions
and load-bus mapping, demand arrays, and a CPU-time budget. Our adapter adds:

1. A minimal repository-owned bridge and Julia OPF worker, retaining applicable
   licenses/notices and source attribution. No runtime SDK import or checkout
   dependency. Keep RAMBO's bilevel and sampling formulations outside this path.
2. Stable bus/generator/load identity and unit mapping. The worker checks for
   one load entry per mapped bus and maps sorted explicit solution IDs. The
   selected network has one generator per generator bus and connected topology.
3. Nodal balance dual extraction and sign/unit verification. The final OPF
   response did not export duals. The new worker requests PowerModels nodal
   duals. The physical derivative is `-raw_dual/baseMVA` for this formulation;
   a 0.01 MW fixed-Q perturbation matches to about `1.5e-10` relative error.
   RAMBO-stage multipliers are not used.
4. Iteration counts, independent residuals, and complete failure records.
   Record both solver CPU-time limits and external wall deadlines; they are
   different budgets. A success status alone does not establish a valid label.
5. Explicit primal initialization in this same OPF path through PowerModels
   `vm_start`, `va_start`, `pg_start`, and `qg_start`. Initialization support
   passes the smoke test; the three-arm performance experiment remains Stage C.
   No warm-start speedup is established by this test.

Before the resumable 200-hour training pilot, verify primal mapping/residuals
and test dual export, recording any unavailable outputs explicitly under the
plan's acceptance rules. Validate initial-state support before the hybrid
experiment. Solver nonconvergence remains an unresolved numerical outcome.

Source provenance is SDK revision
`e75ffcd74854c14e2960b9a9411ddd72377e5239`; the five source files below had no
reported modifications in the scoped Git status. SHA256 values at inspection:

| File relative to RAMBO source | SHA256 |
|---|---|
| `backend.py` | `284e62ea292a9645926696c173817ae850450e8b6f0f5b15c7ecef73ce684941` |
| `julia/model.jl` | `c96d4e029b6d88c4a4369f01b8635446b7205c453beea11319340a9ea7ab7757` |
| `julia/worker.jl` | `96cd42ade3bc2d22d5f8fa27206c9cafba0cb309b940231b3b235bdfe7452a82` |
| `julia/Project.toml` | `14480124430e5f7e0b64697123a2e4a5011b076b29847651ca8d2d63f4d3e052` |
| `julia/Manifest.toml` | `424222f23a5d11b2c679f871ef3633b754bb2af6dc3a93188ce877301cd55b1d` |

The SDK root contains an Apache-2.0 license and an Argonne copyright/NOTICE.
They are retained under `third_party/lumina-sdk/`. Adapted/copied source is
identified in `THIRD_PARTY_NOTICES.md`; dependency/data terms remain separate.
