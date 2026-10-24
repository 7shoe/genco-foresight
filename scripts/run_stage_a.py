"""Allocated-node numerical validation, followed by the accepted pilot/full run."""
import os
import sys

from pathlib import Path
from common import load_config, read_json
from data.synthetic.generate import label
from scripts.solver_smoke import main as solver_smoke


def main():
    config = load_config()
    parity_path = Path(config["paths"]["runs_root"])/"stage_a/parity/parity_report.json"
    if not parity_path.exists() or not read_json(parity_path).get("passed"):
        raise RuntimeError("CPU snapshot parity is required before the full Stage A run")
    solver_smoke()
    pilot_path = Path(config["paths"]["temporal_root"])/"pilot_report.json"
    pilot = read_json(pilot_path) if pilot_path.exists() else label(config, maximum_hours=200)
    if not pilot["accepted"]:
        raise RuntimeError("Pilot gate failed; full generation was not launched")
    if os.environ.get("GENCO_FULL_GENERATION") == "1":
        label(config, maximum_hours=10080)
        from scripts.data_audit import audit as data_audit
        data_audit(config)
    if os.environ.get("GENCO_XPU_CHECK") == "1":
        from scripts.platform_smoke import main as platform_smoke
        platform_smoke()


if __name__ == "__main__":
    main()
