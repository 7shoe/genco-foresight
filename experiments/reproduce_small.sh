#!/usr/bin/env bash
# Offline saved-forecast rescoring. No installs, uploads or downloads.
set -euo pipefail
task_start=$(date +%s)
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
bundle=${1:-${GENCO_REPRO_BUNDLE:-}}
if [[ -z "$bundle" && -f "$repo_root/../bundle.json" ]]; then
    bundle=$(cd "$repo_root/.." && pwd)
fi
if [[ -z "$bundle" ]]; then
    echo "Usage: bash scripts/reproduce_small.sh BUNDLE_DIRECTORY [OUTPUT_DIRECTORY]" >&2
    exit 2
fi
bundle=$(realpath "$bundle")
work_root=${2:-${GENCO_REPRO_DIR:-$bundle/validation}}
work_root=$(realpath -m "$work_root")
if [[ $(hostname) == *aurora* ]]; then
    case $(realpath -m "$work_root") in
        /lus/flare/*) ;;
        *) echo 'Aurora reproduction outputs must be on /lus/flare.' >&2; exit 2 ;;
    esac
fi
mkdir -p "$work_root"
export MPLCONFIGDIR="$work_root/matplotlib" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
task_python=${GENCO_PYTHON:-python}
cd "$repo_root"
"$task_python" -u -m scripts.reproduce_small --bundle "$bundle" --output "$work_root" --start-epoch "$task_start"
