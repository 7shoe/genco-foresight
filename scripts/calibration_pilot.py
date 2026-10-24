"""Training-only load-scale diagnostic and optional XPU check before expansion."""
from pathlib import Path

import numpy as np

from common import load_config, write_json
from data.loader.network import load_network
from solver.adapter import IpoptSolver
from solver.checks import check_solution, physical_quantities
from scripts.platform_smoke import main as platform_smoke


def main():
    config = load_config()
    root = Path(config["paths"]["runs_root"])/"stage_a"
    net = load_network()
    records = []
    with IpoptSolver(config, root/"scale_diagnostic_julia.log") as solver:
        for scale in (0.30, 0.40, 0.50, 0.65, 0.85, 1.15):
            pd, qd = net["bus"][:, 2]*scale, net["bus"][:, 3]*scale
            record = solver.solve(pd, qd)
            check = check_solution(net, pd, qd, record)
            item = {"scale": scale, "checks": check}
            if check["valid"]:
                q = physical_quantities(net, pd, qd, record["state"])
                item["max_loading"] = float(np.nanmax(q["loading"]))
            records.append(item)
    write_json(root/"training_scale_diagnostic.json", {
        "protocol": "prospective six-point demand-support diagnostic before held-out generation",
        "held_out_data_used": False, "records": records})
    print(records, flush=True)
    platform_smoke()


if __name__ == "__main__":
    main()
