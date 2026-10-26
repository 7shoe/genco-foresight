#!/usr/bin/env bash
# External bounded PBS chaining; the package never submits jobs.
set -euo pipefail
: "${PFF_PYTHON:?Set a pinned package Python}"
: "${PFF_RESUME:?Set a campaign or chain directory}"
allocations="${PFF_ALLOCATIONS:-6}"
case "$allocations" in ''|*[!0-9]*) exit 2;; esac
(( allocations >= 1 && allocations <= 12 )) || exit 2
wrapper="$(cd -- "$(dirname -- "$0")" && pwd)/train_chain.pbs"
mkdir -p "$PFF_RESUME/pbs"
for ((attempt=1; attempt<=allocations; attempt++)); do
    if "$PFF_PYTHON" - "$PFF_RESUME" <<'PY'
from pathlib import Path
import sys
from power_fm_foresight.campaign import resolve_runs
from power_fm_foresight._config import read_json
runs=resolve_runs([sys.argv[1]])
finished=all((p/'adapter/status.json').exists() and read_json(p/'adapter/status.json').get('status')=='finished' for p in runs)
sys.exit(0 if finished else 1)
PY
    then break; fi
    qsub -W block=true -v "PFF_PYTHON=$PFF_PYTHON,PFF_RESUME=$PFF_RESUME" \
        -o "$PFF_RESUME/pbs/allocation_${attempt}.log" "$wrapper"
done
