"""Record the frozen Stage B design, preserving explicitly superseded contracts."""
import argparse
from power_fm_foresight._config import ROOT, load_config, read_json, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adopt-user-amendments", action="store_true")
    args = parser.parse_args()
    cfg = load_config()
    b = cfg["stage_b"]
    train_episodes = cfg["task"]["split_episodes"]["train"]
    origins = list(range(72, 504-24+1, b["training_origin_stride_hours"]))
    residual_origins = list(range(b["ridge"]["residual_initial_fit_hours"], 504-24+1, b["ridge"]["residual_origin_stride_hours"]))
    scale_path = ROOT/b["coordinate_scales"]
    scales = read_json(scale_path)
    # Exact PyTorch GRU parameter formula: 3HI + 3HH + two 3H biases.
    expected = {"bus_observed_linear_8_to_32": 8*32+32, "gen_observed_linear_4_to_32": 4*32+32,
        "bus_gru_416_to_32": 3*32*416+3*32*32+6*32,
        "gen_gru_416_to_32": 3*32*416+3*32*32+6*32,
        "bus_conditioning_32_to_48": 32*48+48, "gen_conditioning_32_to_48": 32*48+48,
        "shared_future_projection_6_to_32": 6*32+32, "bus_head_32_to_28": 32*28+28,
        "gen_head_32_to_9": 32*9+9}
    value = {"scientific_settings_frozen": True, "implementation_status": "Adapter preflight; forecasting loops and ridge remain pending",
        "settings": b, "training_origins_per_episode": len(origins), "training_origins": train_episodes*len(origins),
        "training_origin_first": origins[0], "training_origin_last": origins[-1],
        "residual_blocks_planned": train_episodes*len(residual_origins), "residual_blocks_generated": 0,
        "residual_origin_first": residual_origins[0], "residual_origin_last": residual_origins[-1],
        "residual_protocol": f"At each hourly origin168 through480, fit only targets ending before the cutoff from all {train_episodes} train prefixes; predict each next24h; retain one complete joint residual block per episode/origin. The {train_episodes*len(residual_origins)} overlapping blocks are not independent observations.",
        "coordinate_scale_sha256": sha256(scale_path), "joint_score_dimension_24h": scales["dimension_24_hours"],
        "energy_score_formula": "mean_k d(Xk,y) - sum_{k!=l} d(Xk,Xl)/(2*K*(K-1)); d=norm((x-y)/scale)/sqrt(observed_dimension)",
        "parameter_count_design": expected, "expected_trainable_parameters": sum(expected.values()),
        "measured_parameter_count_and_throughput": "reports/stage_b_preflight.json",
        "throughput_protocol": {
            "forecast_batch_size": b["forecast_batch_size"], "training_samples": b["training_samples"],
            "one_pyg_batch_future_graphs": b["future_graph_batch_size"],
            "steps_per_epoch": train_episodes*len(origins),
            "epoch_budget_formula": f"floor((21600-setup_evaluation_seconds)/({train_episodes*len(origins)}*measured_seconds_per_step))",
            "required_measurement": "cached-history full adapter sampling+GENCO+loss+backward+optimizer; synchronize XPU; record peak memory; no substitution of Stage A smoke timing"}}
    from pathlib import Path
    destination = Path(cfg["paths"].get("evidence_root", ROOT/"reports"))/"stage_b_settings.json"
    if destination.exists() and read_json(destination) != value:
        if not args.adopt_user_amendments:
            raise ValueError("Stage B contract already frozen; do not silently change it")
        prior = ROOT/"reports"/f"stage_b_superseded_{sha256(destination)[:12]}.json"
        if not prior.exists():
            write_json(prior, read_json(destination))
    write_json(destination, value)
    print({"origins": value["training_origins"], "planned_blocks": value["residual_blocks_planned"],
           "expected_parameters": value["expected_trainable_parameters"]}, flush=True)


if __name__ == "__main__":
    main()
