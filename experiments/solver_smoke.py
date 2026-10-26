"""One checked cold solve, warm solve, and fixed-Q demand/dual perturbation."""
import argparse
from pathlib import Path

import numpy as np

from power_fm_foresight._config import load_config, write_json
from power_fm_foresight._data.network import graph_for_demand, load_network
from power_fm_foresight._solver.adapter import IpoptSolver
from power_fm_foresight._solver.checks import check_solution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v3.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(config["paths"].get("stage_a_run", Path(config["paths"]["runs_root"])/"stage_a"))
    root.mkdir(parents=True, exist_ok=True)
    network = load_network(config=config)
    pd, qd = network["bus"][:, 2], network["bus"][:, 3]
    result = {"case_sha256": network["case_sha256"], "perturbation_mw": 0.01,
              "independent_tolerance": 1e-5, "objective_convention": "benchmark cost per hour; one-hour snapshots"}
    with IpoptSolver(config, root / "julia.log") as solver:
        result["versions"] = solver.versions
        cold = solver.solve(pd, qd)
        cold["checks"] = check_solution(network, pd, qd, cold)
        result["cold"] = cold
        write_json(root / "solver_smoke.json", result)
        if not cold["checks"]["valid"]:
            raise RuntimeError(f"Cold reference failed: {cold}")
        warm = solver.solve(pd, qd, initial_state=cold["state"])
        warm["checks"] = check_solution(network, pd, qd, warm)
        result["warm"] = warm
        graph = graph_for_demand(network, pd, qd)
        initial = cold["state"] if config["solver"].get("warm_start_base", False) else None
        round_trip = solver.solve(graph, initial_state=initial)
        graph_pd, graph_qd = graph["bus"].x[:, 0].numpy(), graph["bus"].x[:, 1].numpy()
        round_trip["checks"] = check_solution(network, graph_pd, graph_qd, round_trip)
        result["graph_round_trip"] = round_trip
        bus = int(np.flatnonzero(pd > 0)[0])
        perturbed_pd = pd.copy(); perturbed_pd[bus] += result["perturbation_mw"]
        perturbed = solver.solve(perturbed_pd, qd, initial_state=initial)
        perturbed["checks"] = check_solution(network, perturbed_pd, qd, perturbed)
        result["perturbed"] = perturbed
        if perturbed["checks"]["valid"] and "dual_p_raw" in cold:
            derivative = (perturbed["objective"]-cold["objective"])/result["perturbation_mw"]
            raw_per_mw = cold["dual_p_raw"][bus] / network["baseMVA"]
            sign = 1 if abs(derivative-raw_per_mw) <= abs(derivative+raw_per_mw) else -1
            error = abs(derivative-sign*raw_per_mw)/max(1, abs(derivative))
            result["dual_check"] = {"bus_index": bus, "hold_qd_fixed": True, "sign": sign,
                "finite_difference": derivative, "signed_dual_per_mw": sign*raw_per_mw,
                "relative_error": error, "valid": error < 0.02}
        else:
            result["dual_check"] = {"valid": False, "reason": "unavailable_dual_or_reference"}
    write_json(root / "solver_smoke.json", result)
    print({"cold_valid": cold["checks"]["valid"], "warm_valid": warm["checks"]["valid"],
           "dual_check": result["dual_check"]}, flush=True)
    if not warm["checks"]["valid"] or not round_trip["checks"]["valid"] or not result["dual_check"]["valid"]:
        raise RuntimeError("Warm-start or dual validation failed; see solver_smoke.json")


if __name__ == "__main__":
    main()
