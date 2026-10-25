"""Record release facts from existing configurations and immutable evidence."""
from pathlib import Path
from data.loader.network import load_network
from common import ROOT, load_config, read_json, sha256, write_json
from scripts.finalize_release import check_frozen


def main():
    cfg = load_config(ROOT / "config.yaml")
    other = load_config(ROOT / "configs/v4.yaml")
    runs = Path(cfg["paths"]["runs_root"])
    out = runs / "stage_d/final_release"
    out.mkdir(parents=True, exist_ok=True)
    claims = {}
    def add(key, value, unit, source):
        claims[key] = {"value": value, "unit": unit, "source": source}
    for key, value, unit in (
        ("setup.case_buses", load_network(config=cfg).n_bus, "buses"), ("setup.seed", cfg["seed"], "seed"),
        ("setup.history_hours", cfg["task"]["history_hours"], "hours"),
        ("setup.forecast_hours", cfg["task"]["horizon_hours"], "hours"),
        ("setup.episodes", cfg["task"]["episodes"], "episodes"),
        ("setup.days_per_episode", cfg["task"]["days_per_episode"], "days"),
        ("setup.split_episodes", list(cfg["task"]["split_episodes"].values()), "train, selection, calibration, test"),
        ("setup.samples", cfg["stage_b"]["forecast_samples"], "samples/origin"),
        ("setup.gru_width", cfg["stage_b"]["adapter"]["gru_hidden_width"], "latent coordinates"),
        ("setup.conditioning_width", cfg["stage_b"]["adapter"]["conditioning_output_width"], "latent coordinates"),
        ("setup.covariance_rank", cfg["stage_b"]["adapter"]["covariance_rank"], "rank"),
        ("setup.embedding_width", cfg["stage_b"]["adapter"]["embedding_width"], "latent coordinates"),
        ("setup.ridge_lags", cfg["stage_b"]["ridge"]["lags_hours"], "hours"),
        ("setup.nominal_coverage_pct", 90, "%"),
        ("setup.interval_levels_pct", [50, 90], "%"),
        ("setup.opfdata_nominal_range", [.8, 1.2], "nominal multiplier; checkpoint training regime reported in PLAN_v2.md"),
        ("setup.ood_bin_edges", [.6, .8], "realized reference multiplier"),
        ("setup.profile_year", 2025, "EIA ERCO source year"),
        ("setup.headline_buses", [1, 9, 11, 13, 15, 21, 22], "one-based bus IDs"),
        ("setup.voltage_outer_band_pct", 5, "% of voltage range at each end"),
        ("setup.thermal_rating_pct", 95, "% rating"),
        ("setup.prevalence_ratio_flag", 1.5, "larger/smaller prevalence"),
        ("setup.test_days", 72, "forecast origins per dataset"),
        ("setup.calibration_days", 72, "forecast origins"),
        ("setup.forecast_test_hours", 1728, "hours per dataset before unresolved masks"),
        ("setup.training_windows", 4090, "hourly origins"),
        ("setup.training_seed_count", 1, "seed"),
        ("setup.ar_rho", cfg["generation"]["rho"], "AR coefficient"),
        ("setup.ar_sigma", cfg["generation"]["innovation_std"], "lognormal innovation std"),
        ("setup.iid_noise_pct", 100*cfg["generation"]["sigma"], "symmetric uniform noise half-width %"),
        ("setup.trainable_parameters", 91461, "parameters"),
        ("setup.buffer_elements", 884, "elements"),
        ("setup.state_elements", 92345, "parameters plus buffers"),
    ):
        add(key, value, unit, "config.yaml / configs/v4.yaml / frozen evaluation contracts")
    frozen = check_frozen(runs)
    protected = {str(p.relative_to(ROOT)): sha256(p) for directory in ("models", "training")
                 for p in (ROOT / directory).rglob("*.py")}
    protected["scoring.py"] = sha256(ROOT / "scoring.py")
    for version, config in (("v3", cfg), ("v4", other)):
        quality = read_json(Path(config["paths"]["temporal_root"]) / "quality_report.json")
        status = read_json(Path(config["paths"]["training_run"]) / "status.json")
        chain = read_json(Path(config["paths"]["training_run"]) / "chain.json")
        add(f"{version}.nominal_load_range", [config["generation"]["scale_low"], config["generation"]["scale_high"]], "nominal multiplier", config["paths"]["temporal_root"])
        add(f"{version}.physics_weight", config["stage_b"]["loss"]["power_balance_weight"], "lambda", "config.yaml" if version == "v3" else "configs/v4.yaml")
        for key in ("attempted", "valid"):
            add(f"{version}.labels.{key}", quality[key], "hours", config["paths"]["temporal_root"] + "/quality_report.json")
        add(f"{version}.labels.unresolved", quality["attempted"]-quality["valid"], "hours", config["paths"]["temporal_root"] + "/quality_report.json")
        add(f"{version}.best_epoch", status["best_epoch"], "epoch", config["paths"]["training_run"] + "/status.json")
        add(f"{version}.training_wall_seconds", chain["cumulative_training_wall_seconds"], "seconds; job wall including startup", config["paths"]["training_run"] + "/chain.json")
        add(f"{version}.queue_wait_seconds", chain["total_queue_wait_seconds"], "seconds; completed jobs, excludes dependency wait", config["paths"]["training_run"] + "/chain.json")
    ridge = read_json(runs / "stage_b/forecaster/manifest.json")
    add("setup.ridge_residual_blocks", 3130, "blocks; audited v3/v4 manifests", "stage_b/forecaster/manifest.json")
    add("setup.probe_horizons", [1, 24], "hours", "stage_b/probe/report.json#protocol/horizons")
    add("setup.test_source_episodes", 4, "episodes", "config.yaml#task/split_episodes/test")
    support = read_json(runs / "stage_a_v4/support_diagnostic.json")
    add("v4.test_bus_load_below_support_pct", 100*support["test"]["actual_bus_fraction_below_0p8"], "% of active bus-hours, including history", "stage_a_v4/support_diagnostic.json#test")
    inventory = {}
    for buses, suffix in ((14, "ieee"), (57, "ieee"), (118, "ieee"), (500, "goc")):
        path = Path(cfg["paths"]["opfdata_root"]) / f"processed/N1/pglib_opf_case{buses}_{suffix}/manifest.json"
        assert path.is_file()
        inventory[str(buses)] = {"path": str(path), "sha256": sha256(path), "status": "snapshot cache present; temporal/model validation not run"}
    add("setup.prepared_snapshot_grids", [int(k) for k in inventory], "buses; snapshot caches only", "multi_grid_inventory")
    bundle = Path(cfg["paths"]["data_root"]).parent / "reproducibility/v0.1.0-rc"
    validation = read_json(bundle / "history/v3_only/validation/result.json")
    add("reproduction.v3_original_seconds", validation["elapsed_seconds_including_setup"], "seconds; prior v3 saved-forecast check", "history/v3_only/validation/result.json")
    # Updated with the measured combined replay before the final post is rendered.
    current = out / "context.json"
    if current.exists():
        old = read_json(current)
        for k, v in old["claims"].items():
            if k.startswith("reproduction.combined_"):
                claims[k] = v
    write_json(current, {"claims": claims, "config_v3": cfg, "config_v4": other,
        "pins": {"graphkit": cfg["upstream"]["graphkit"]["revision"], "checkpoint": cfg["checkpoint"]["revision"]},
        "multi_grid_inventory": inventory, "frozen_evaluation_source_sha256": frozen,
        "protected_source_sha256": protected, "ridge_manifest": ridge,
        "no_new_experiments": True, "publication": "local review only"})
    print({"context": str(current), "claims": len(claims), "protected_sources": len(protected)})


if __name__ == "__main__":
    main()
