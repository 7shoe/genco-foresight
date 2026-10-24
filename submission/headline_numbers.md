# Headline number keys

Generated from frozen saved reports; source paths are relative to evidence/runs in the release package.
Keys next to numbers in post.md are retained in HTML comments for audit without cluttering the rendered post.

| Key | Value | Unit / definition | Evidence |
|---|---|---|---|
| joint.coordinates.Pd | 21 | coordinates/hour | stage_b/diagnostics/report.json#groups |
| joint.coordinates.Pg | 2 | coordinates/hour | stage_b/diagnostics/report.json#groups |
| joint.coordinates.Qd | 0 | coordinates/hour | stage_b/diagnostics/report.json#groups |
| joint.coordinates.Qg | 6 | coordinates/hour | stage_b/diagnostics/report.json#groups |
| joint.coordinates.Va | 29 | coordinates/hour | stage_b/diagnostics/report.json#groups |
| joint.coordinates.Vm | 30 | coordinates/hour | stage_b/diagnostics/report.json#groups |
| joint.coordinates.other | 164 | coordinates/hour | stage_b/diagnostics/report.json#groups |
| joint.coordinates.total | 252 | coordinates/hour | stage_b/diagnostics/report.json#groups |
| reproduction.combined_seconds | 143.77497053146362 | seconds; hashes, saved-forecast CPU rescoring and three figure renders | stage_d/final_release/context.json#claims/reproduction.combined_seconds |
| reproduction.v3_original_seconds | 276.0059931278229 | seconds; prior v3 saved-forecast check | stage_d/final_release/context.json#claims/reproduction.v3_original_seconds |
| setup.ar_rho | 0.9 | AR coefficient | stage_d/final_release/context.json#claims/setup.ar_rho |
| setup.ar_sigma | 0.02 | lognormal innovation std | stage_d/final_release/context.json#claims/setup.ar_sigma |
| setup.buffer_elements | 884 | elements | stage_d/final_release/context.json#claims/setup.buffer_elements |
| setup.calibration_days | 72 | forecast origins | stage_d/final_release/context.json#claims/setup.calibration_days |
| setup.case_buses | 30 | buses | stage_d/final_release/context.json#claims/setup.case_buses |
| setup.conditioning_width | 48 | latent coordinates | stage_d/final_release/context.json#claims/setup.conditioning_width |
| setup.covariance_rank | 8 | rank | stage_d/final_release/context.json#claims/setup.covariance_rank |
| setup.days_per_episode | 21 | days | stage_d/final_release/context.json#claims/setup.days_per_episode |
| setup.embedding_width | 384 | latent coordinates | stage_d/final_release/context.json#claims/setup.embedding_width |
| setup.episodes | 20 | episodes | stage_d/final_release/context.json#claims/setup.episodes |
| setup.forecast_hours | 24 | hours | stage_d/final_release/context.json#claims/setup.forecast_hours |
| setup.forecast_test_hours | 1728 | hours per dataset before unresolved masks | stage_d/final_release/context.json#claims/setup.forecast_test_hours |
| setup.gru_width | 32 | latent coordinates | stage_d/final_release/context.json#claims/setup.gru_width |
| setup.headline_buses | [1, 9, 11, 13, 15, 21, 22] | one-based bus IDs | stage_d/final_release/context.json#claims/setup.headline_buses |
| setup.history_hours | 72 | hours | stage_d/final_release/context.json#claims/setup.history_hours |
| setup.iid_noise_pct | 3.0 | symmetric uniform noise half-width % | stage_d/final_release/context.json#claims/setup.iid_noise_pct |
| setup.interval_levels_pct | [50, 90] | % | stage_d/final_release/context.json#claims/setup.interval_levels_pct |
| setup.nominal_coverage_pct | 90 | % | stage_d/final_release/context.json#claims/setup.nominal_coverage_pct |
| setup.ood_bin_edges | [0.6, 0.8] | realized reference multiplier | stage_d/final_release/context.json#claims/setup.ood_bin_edges |
| setup.opfdata_nominal_range | [0.8, 1.2] | nominal multiplier; checkpoint training regime reported in PLAN_v2.md | stage_d/final_release/context.json#claims/setup.opfdata_nominal_range |
| setup.prepared_snapshot_grids | [14, 57, 118, 500] | buses; snapshot caches only | stage_d/final_release/context.json#claims/setup.prepared_snapshot_grids |
| setup.prevalence_ratio_flag | 1.5 | larger/smaller prevalence | stage_d/final_release/context.json#claims/setup.prevalence_ratio_flag |
| setup.probe_horizons | [1, 24] | hours | stage_d/final_release/context.json#claims/setup.probe_horizons |
| setup.profile_year | 2025 | EIA ERCO source year | stage_d/final_release/context.json#claims/setup.profile_year |
| setup.ridge_lags | [1, 2, 24, 48] | hours | stage_d/final_release/context.json#claims/setup.ridge_lags |
| setup.ridge_residual_blocks | 3130 | blocks; audited v3/v4 manifests | stage_d/final_release/context.json#claims/setup.ridge_residual_blocks |
| setup.samples | 64 | samples/origin | stage_d/final_release/context.json#claims/setup.samples |
| setup.seed | 42 | seed | stage_d/final_release/context.json#claims/setup.seed |
| setup.split_episodes | [10, 2, 4, 4] | train, selection, calibration, test | stage_d/final_release/context.json#claims/setup.split_episodes |
| setup.state_elements | 92345 | parameters plus buffers | stage_d/final_release/context.json#claims/setup.state_elements |
| setup.test_days | 72 | forecast origins per dataset | stage_d/final_release/context.json#claims/setup.test_days |
| setup.test_source_episodes | 4 | episodes | stage_d/final_release/context.json#claims/setup.test_source_episodes |
| setup.thermal_rating_pct | 95 | % rating | stage_d/final_release/context.json#claims/setup.thermal_rating_pct |
| setup.trainable_parameters | 91461 | parameters | stage_d/final_release/context.json#claims/setup.trainable_parameters |
| setup.training_seed_count | 1 | seed | stage_d/final_release/context.json#claims/setup.training_seed_count |
| setup.training_windows | 4090 | hourly origins | stage_d/final_release/context.json#claims/setup.training_windows |
| setup.voltage_outer_band_pct | 5 | % of voltage range at each end | stage_d/final_release/context.json#claims/setup.voltage_outer_band_pct |
| v3.A.family_es.Pd | 0.4181093694022315 | separately train-scaled score | stage_b/diagnostics/report.json#scopes/all/modes/A/groups/Pd |
| v3.A.family_es.Pg | 1.0554463238485599 | separately train-scaled score | stage_b/diagnostics/report.json#scopes/all/modes/A/groups/Pg |
| v3.A.family_es.Qd | null | separately train-scaled score | stage_b/diagnostics/report.json#scopes/all/modes/A/groups/Qd |
| v3.A.family_es.Qg | 0.5693940268381222 | separately train-scaled score | stage_b/diagnostics/report.json#scopes/all/modes/A/groups/Qg |
| v3.A.family_es.Va | 0.5104096610520815 | separately train-scaled score | stage_b/diagnostics/report.json#scopes/all/modes/A/groups/Va |
| v3.A.family_es.Vm | 0.4562277314240942 | separately train-scaled score | stage_b/diagnostics/report.json#scopes/all/modes/A/groups/Vm |
| v3.A.family_es.other | 0.41300973279716796 | separately train-scaled score | stage_b/diagnostics/report.json#scopes/all/modes/A/groups/other |
| v3.A.joint_es | 0.42995763282679167 | train-scaled score | stage_b/A.json#joint_energy_score |
| v3.A.p_imbalance | 5.4641438793186 | MW/bus | stage_b/diagnostics/report.json#scopes/all/modes/A |
| v3.A.pg_free_coverage90_pct | 53.55902777777778 | % | stage_b/diagnostics/report.json#scopes/all/modes/A |
| v3.A.pg_free_mae | 23.18194995077493 | MW | stage_b/diagnostics/report.json#scopes/all/modes/A |
| v3.A.q_imbalance | 0.9303982670180619 | MVAr/bus | stage_b/diagnostics/report.json#scopes/all/modes/A |
| v3.A.vm_coverage50_pct | 20.089285714285715 | % | stage_c/report.json#continuous/test/A (count-weighted headline buses) |
| v3.A.vm_coverage90_pct | 28.753306878306876 | % | stage_c/report.json#continuous/test/A (count-weighted headline buses) |
| v3.A.vm_crps | 0.004164363654725085 | p.u. | stage_c/report.json#continuous/test/A (count-weighted headline buses) |
| v3.A.vm_width90 | 0.009437559224744006 | p.u. | stage_c/report.json#continuous/test/A (count-weighted headline buses) |
| v3.A.voltage_brier_mapped | 0.09536328031960253 | Brier score | stage_c/report.json#hourly-voltage-macro |
| v3.A.voltage_brier_raw | 0.08692003684069115 | Brier score | stage_c/report.json#hourly-voltage-macro |
| v3.C.family_es.Pd | 0.5577313711236522 | separately train-scaled score | stage_b/diagnostics/report.json#scopes/all/modes/C/groups/Pd |
| v3.C.family_es.Pg | 0.7176864205881807 | separately train-scaled score | stage_b/diagnostics/report.json#scopes/all/modes/C/groups/Pg |
| v3.C.family_es.Qd | null | separately train-scaled score | stage_b/diagnostics/report.json#scopes/all/modes/C/groups/Qd |
| v3.C.family_es.Qg | 0.703371942286363 | separately train-scaled score | stage_b/diagnostics/report.json#scopes/all/modes/C/groups/Qg |
| v3.C.family_es.Va | 0.628971856353326 | separately train-scaled score | stage_b/diagnostics/report.json#scopes/all/modes/C/groups/Va |
| v3.C.family_es.Vm | 0.45357528809220304 | separately train-scaled score | stage_b/diagnostics/report.json#scopes/all/modes/C/groups/Vm |
| v3.C.family_es.other | 0.5375066304661403 | separately train-scaled score | stage_b/diagnostics/report.json#scopes/all/modes/C/groups/other |
| v3.C.joint_es | 0.549374137320322 | train-scaled score | stage_b/C.json#joint_energy_score |
| v3.C.p_imbalance | 3.8500634997391384e-11 | MW/bus | stage_b/diagnostics/report.json#scopes/all/modes/C |
| v3.C.pg_free_coverage90_pct | 76.24421296296296 | % | stage_b/diagnostics/report.json#scopes/all/modes/C |
| v3.C.pg_free_mae | 21.34770218415035 | MW | stage_b/diagnostics/report.json#scopes/all/modes/C |
| v3.C.q_imbalance | 1.620643771724153e-10 | MVAr/bus | stage_b/diagnostics/report.json#scopes/all/modes/C |
| v3.C.vm_coverage50_pct | 39.60813492063492 | % | stage_c/report.json#continuous/test/C (count-weighted headline buses) |
| v3.C.vm_coverage90_pct | 77.33134920634922 | % | stage_c/report.json#continuous/test/C (count-weighted headline buses) |
| v3.C.vm_crps | 0.003146985246434156 | p.u. | stage_c/report.json#continuous/test/C (count-weighted headline buses) |
| v3.C.vm_width90 | 0.012900289195393709 | p.u. | stage_c/report.json#continuous/test/C (count-weighted headline buses) |
| v3.C.voltage_brier_mapped | 0.12441506422571617 | Brier score | stage_c/report.json#hourly-voltage-macro |
| v3.C.voltage_brier_raw | 0.12340761618639427 | Brier score | stage_c/report.json#hourly-voltage-macro |
| v3.D.family_es.Pd | 0.4181093694022315 | separately train-scaled score | stage_b/diagnostics/report.json#scopes/all/modes/D/groups/Pd |
| v3.D.family_es.Pg | 1.5889624795064856 | separately train-scaled score | stage_b/diagnostics/report.json#scopes/all/modes/D/groups/Pg |
| v3.D.family_es.Qd | null | separately train-scaled score | stage_b/diagnostics/report.json#scopes/all/modes/D/groups/Qd |
| v3.D.family_es.Qg | 1.9964180894326162 | separately train-scaled score | stage_b/diagnostics/report.json#scopes/all/modes/D/groups/Qg |
| v3.D.family_es.Va | 1.2685518951302226 | separately train-scaled score | stage_b/diagnostics/report.json#scopes/all/modes/D/groups/Va |
| v3.D.family_es.Vm | 0.671292735945657 | separately train-scaled score | stage_b/diagnostics/report.json#scopes/all/modes/D/groups/Vm |
| v3.D.family_es.other | 6.654553887050794 | separately train-scaled score | stage_b/diagnostics/report.json#scopes/all/modes/D/groups/other |
| v3.D.joint_es | 5.428913402550243 | train-scaled score | stage_b/D.json#joint_energy_score |
| v3.D.p_imbalance | 6.636415150201725 | MW/bus | stage_b/diagnostics/report.json#scopes/all/modes/D |
| v3.D.pg_free_coverage90_pct | 5.815972222222222 | % | stage_b/diagnostics/report.json#scopes/all/modes/D |
| v3.D.pg_free_mae | 33.19750020423029 | MW | stage_b/diagnostics/report.json#scopes/all/modes/D |
| v3.D.q_imbalance | 4.180969843211384 | MVAr/bus | stage_b/diagnostics/report.json#scopes/all/modes/D |
| v3.D.vm_coverage50_pct | 1.3971560846560847 | % | stage_c/report.json#continuous/test/D (count-weighted headline buses) |
| v3.D.vm_coverage90_pct | 3.5548941798941804 | % | stage_c/report.json#continuous/test/D (count-weighted headline buses) |
| v3.D.vm_crps | 0.0065957890055979105 | p.u. | stage_c/report.json#continuous/test/D (count-weighted headline buses) |
| v3.D.vm_width90 | 0.0015934017058166243 | p.u. | stage_c/report.json#continuous/test/D (count-weighted headline buses) |
| v3.D.voltage_brier_mapped | 0.14643791870527706 | Brier score | stage_c/report.json#hourly-voltage-macro |
| v3.D.voltage_brier_raw | 0.2762761595387938 | Brier score | stage_c/report.json#hourly-voltage-macro |
| v3.P.joint_es | 6.4038189945097095 | train-scaled score | stage_b/P.json#joint_energy_score |
| v3.P.p_imbalance | 6.033454714870075 | MW/bus | stage_b/diagnostics/report.json#scopes/all/modes/P |
| v3.P.pg_free_coverage90_pct | 0.0 | % | stage_b/diagnostics/report.json#scopes/all/modes/P |
| v3.P.pg_free_mae | 32.25188442602169 | MW | stage_b/diagnostics/report.json#scopes/all/modes/P |
| v3.P.q_imbalance | 3.835031745103113 | MVAr/bus | stage_b/diagnostics/report.json#scopes/all/modes/P |
| v3.P.vm_coverage50_pct | 0.0 | % | stage_c/report.json#continuous/test/P (count-weighted headline buses) |
| v3.P.vm_coverage90_pct | 0.0 | % | stage_c/report.json#continuous/test/P (count-weighted headline buses) |
| v3.P.vm_crps | 0.006693252946955841 | p.u. | stage_c/report.json#continuous/test/P (count-weighted headline buses) |
| v3.P.vm_width90 | 0.0 | p.u. | stage_c/report.json#continuous/test/P (count-weighted headline buses) |
| v3.P.voltage_brier_mapped | 0.14747570105929694 | Brier score | stage_c/report.json#hourly-voltage-macro |
| v3.P.voltage_brier_raw | 0.27918320105820105 | Brier score | stage_c/report.json#hourly-voltage-macro |
| v3.P0.joint_es | 6.403818829824138 | train-scaled score | stage_b/P0.json#joint_energy_score |
| v3.P0.p_imbalance | 6.033454714870075 | MW/bus | stage_b/diagnostics/report.json#scopes/all/modes/P0 |
| v3.P0.pg_free_coverage90_pct | 0.0 | % | stage_b/diagnostics/report.json#scopes/all/modes/P0 |
| v3.P0.pg_free_mae | 32.25188442602169 | MW | stage_b/diagnostics/report.json#scopes/all/modes/P0 |
| v3.P0.q_imbalance | 3.835031745103113 | MVAr/bus | stage_b/diagnostics/report.json#scopes/all/modes/P0 |
| v3.P0.vm_coverage50_pct | 0.0 | % | stage_c/report.json#continuous/test/P0 (count-weighted headline buses) |
| v3.P0.vm_coverage90_pct | 0.0 | % | stage_c/report.json#continuous/test/P0 (count-weighted headline buses) |
| v3.P0.vm_crps | 0.006693252946955841 | p.u. | stage_c/report.json#continuous/test/P0 (count-weighted headline buses) |
| v3.P0.vm_width90 | 0.0 | p.u. | stage_c/report.json#continuous/test/P0 (count-weighted headline buses) |
| v3.P0.voltage_brier_mapped | 0.14747570105929694 | Brier score | stage_c/report.json#hourly-voltage-macro |
| v3.P0.voltage_brier_raw | 0.27918320105820105 | Brier score | stage_c/report.json#hourly-voltage-macro |
| v3.S.family_es.Pd | 0.3123797633369686 | separately train-scaled score | stage_b/diagnostics/report.json#scopes/all/modes/S/groups/Pd |
| v3.S.family_es.Pg | 1.4974750825059737 | separately train-scaled score | stage_b/diagnostics/report.json#scopes/all/modes/S/groups/Pg |
| v3.S.family_es.Qd | null | separately train-scaled score | stage_b/diagnostics/report.json#scopes/all/modes/S/groups/Qd |
| v3.S.family_es.Qg | 1.8578139700635832 | separately train-scaled score | stage_b/diagnostics/report.json#scopes/all/modes/S/groups/Qg |
| v3.S.family_es.Va | 1.218537615410492 | separately train-scaled score | stage_b/diagnostics/report.json#scopes/all/modes/S/groups/Va |
| v3.S.family_es.Vm | 0.6095045037945833 | separately train-scaled score | stage_b/diagnostics/report.json#scopes/all/modes/S/groups/Vm |
| v3.S.family_es.other | 4.368062435536236 | separately train-scaled score | stage_b/diagnostics/report.json#scopes/all/modes/S/groups/other |
| v3.S.joint_es | 3.5762105792062857 | train-scaled score | stage_b/S.json#joint_energy_score |
| v3.S.p_imbalance | 6.868047007764859 | MW/bus | stage_b/diagnostics/report.json#scopes/all/modes/S |
| v3.S.pg_free_coverage90_pct | 17.85300925925926 | % | stage_b/diagnostics/report.json#scopes/all/modes/S |
| v3.S.pg_free_mae | 32.258949610209825 | MW | stage_b/diagnostics/report.json#scopes/all/modes/S |
| v3.S.q_imbalance | 4.202218103282704 | MVAr/bus | stage_b/diagnostics/report.json#scopes/all/modes/S |
| v3.S.vm_coverage50_pct | 4.1005291005291005 | % | stage_c/report.json#continuous/test/S (count-weighted headline buses) |
| v3.S.vm_coverage90_pct | 10.13558201058201 | % | stage_c/report.json#continuous/test/S (count-weighted headline buses) |
| v3.S.vm_crps | 0.006155236234504526 | p.u. | stage_c/report.json#continuous/test/S (count-weighted headline buses) |
| v3.S.vm_width90 | 0.0041692564475828055 | p.u. | stage_c/report.json#continuous/test/S (count-weighted headline buses) |
| v3.S.voltage_brier_mapped | 0.14876752483730565 | Brier score | stage_c/report.json#hourly-voltage-macro |
| v3.S.voltage_brier_raw | 0.2683463323683966 | Brier score | stage_c/report.json#hourly-voltage-macro |
| v3.best_epoch | 18 | epoch | stage_d/final_release/context.json#claims/v3.best_epoch |
| v3.hybrid.A_forecast_mean.iterations | 16.074074074074073 | mean iterations | stage_c/hybrid/report.json#arms/A_forecast_mean |
| v3.hybrid.A_forecast_mean.request_ms | 38.70118095656599 | mean milliseconds | stage_c/hybrid/report.json#arms/A_forecast_mean |
| v3.hybrid.A_forecast_mean.total_work_seconds | 196.10870696540223 | seconds | stage_c/hybrid/report.json#arms/A_forecast_mean |
| v3.hybrid.A_forecast_mean.valid | 1728 | valid solves | stage_c/hybrid/report.json#arms/A_forecast_mean |
| v3.hybrid.cold.iterations | 15.960069444444445 | mean iterations | stage_c/hybrid/report.json#arms/cold |
| v3.hybrid.cold.request_ms | 38.69440129788186 | mean milliseconds | stage_c/hybrid/report.json#arms/cold |
| v3.hybrid.cold.total_work_seconds | 66.86392544273986 | seconds | stage_c/hybrid/report.json#arms/cold |
| v3.hybrid.cold.valid | 1728 | valid solves | stage_c/hybrid/report.json#arms/cold |
| v3.hybrid.work_ratio | 2.93295234563133 | warm/cold | stage_c/hybrid/report.json#arms |
| v3.labels.attempted | 10080 | hours | stage_d/final_release/context.json#claims/v3.labels.attempted |
| v3.labels.unresolved | 0 | hours | stage_d/final_release/context.json#claims/v3.labels.unresolved |
| v3.labels.valid | 10080 | hours | stage_d/final_release/context.json#claims/v3.labels.valid |
| v3.nominal_load_range | [0.3, 1.0] | nominal multiplier | stage_d/final_release/context.json#claims/v3.nominal_load_range |
| v3.ood.0_6_to_0_8.pg_mw_mae | 6.160937495674079 | MW | stage_b/diagnostics/report.json#ood_P/0_6_to_0_8 |
| v3.ood.0_6_to_0_8.vm_mae | 0.004747495694481555 | p.u. | stage_b/diagnostics/report.json#ood_P/0_6_to_0_8 |
| v3.ood.ge_0_8.pg_mw_mae | 1.0140938822546437 | MW | stage_b/diagnostics/report.json#ood_P/ge_0_8 |
| v3.ood.ge_0_8.vm_mae | 0.0023317253534366374 | p.u. | stage_b/diagnostics/report.json#ood_P/ge_0_8 |
| v3.ood.lt_0_6.pg_mw_mae | 13.760412228262135 | MW | stage_b/diagnostics/report.json#ood_P/lt_0_6 |
| v3.ood.lt_0_6.vm_mae | 0.006015373568218938 | p.u. | stage_b/diagnostics/report.json#ood_P/lt_0_6 |
| v3.physics_weight | 0.01 | lambda | stage_d/final_release/context.json#claims/v3.physics_weight |
| v3.prevalence.thermal_line_1.daily_any_counts | [[103, 180, 0], [24, 36, 0], [46, 72, 0], [43, 72, 0]] | [positive,resolved,unknown]; train, selection, calibration, test | stage_a_v3/stage_b_prevalence.json |
| v3.prevalence.thermal_line_1.daily_any_pct | [57.22222222222222, 66.66666666666666, 63.888888888888886, 59.72222222222222] | %; train, selection, calibration, test | stage_a_v3/stage_b_prevalence.json |
| v3.prevalence.thermal_line_1.hourly_counts | [[1202, 4320, 0], [234, 864, 0], [467, 1728, 0], [567, 1728, 0]] | [positive,resolved,unknown]; train, selection, calibration, test | stage_a_v3/stage_b_prevalence.json |
| v3.prevalence.thermal_line_1.hourly_pct | [27.824074074074073, 27.083333333333332, 27.025462962962965, 32.8125] | %; train, selection, calibration, test | stage_a_v3/stage_b_prevalence.json |
| v3.prevalence.voltage_bus_1.daily_any_counts | [[172, 180, 0], [34, 36, 0], [71, 72, 0], [64, 72, 0]] | [positive,resolved,unknown]; train, selection, calibration, test | stage_a_v3/stage_b_prevalence.json |
| v3.prevalence.voltage_bus_1.daily_any_pct | [95.55555555555556, 94.44444444444444, 98.61111111111111, 88.88888888888889] | %; train, selection, calibration, test | stage_a_v3/stage_b_prevalence.json |
| v3.prevalence.voltage_bus_1.hourly_counts | [[2847, 4320, 0], [534, 864, 0], [1128, 1728, 0], [1107, 1728, 0]] | [positive,resolved,unknown]; train, selection, calibration, test | stage_a_v3/stage_b_prevalence.json |
| v3.prevalence.voltage_bus_1.hourly_pct | [65.90277777777777, 61.80555555555556, 65.27777777777779, 64.0625] | %; train, selection, calibration, test | stage_a_v3/stage_b_prevalence.json |
| v3.prevalence.voltage_bus_11.daily_any_counts | [[111, 180, 0], [30, 36, 0], [50, 72, 0], [47, 72, 0]] | [positive,resolved,unknown]; train, selection, calibration, test | stage_a_v3/stage_b_prevalence.json |
| v3.prevalence.voltage_bus_11.daily_any_pct | [61.66666666666667, 83.33333333333334, 69.44444444444444, 65.27777777777779] | %; train, selection, calibration, test | stage_a_v3/stage_b_prevalence.json |
| v3.prevalence.voltage_bus_11.hourly_counts | [[1439, 4320, 0], [277, 864, 0], [568, 1728, 0], [647, 1728, 0]] | [positive,resolved,unknown]; train, selection, calibration, test | stage_a_v3/stage_b_prevalence.json |
| v3.prevalence.voltage_bus_11.hourly_pct | [33.31018518518519, 32.06018518518518, 32.870370370370374, 37.442129629629626] | %; train, selection, calibration, test | stage_a_v3/stage_b_prevalence.json |
| v3.prevalence.voltage_bus_13.daily_any_counts | [[106, 180, 0], [24, 36, 0], [48, 72, 0], [43, 72, 0]] | [positive,resolved,unknown]; train, selection, calibration, test | stage_a_v3/stage_b_prevalence.json |
| v3.prevalence.voltage_bus_13.daily_any_pct | [58.88888888888889, 66.66666666666666, 66.66666666666666, 59.72222222222222] | %; train, selection, calibration, test | stage_a_v3/stage_b_prevalence.json |
| v3.prevalence.voltage_bus_13.hourly_counts | [[1251, 4320, 0], [243, 864, 0], [480, 1728, 0], [575, 1728, 0]] | [positive,resolved,unknown]; train, selection, calibration, test | stage_a_v3/stage_b_prevalence.json |
| v3.prevalence.voltage_bus_13.hourly_pct | [28.958333333333336, 28.125, 27.77777777777778, 33.27546296296296] | %; train, selection, calibration, test | stage_a_v3/stage_b_prevalence.json |
| v3.prevalence.voltage_bus_15.daily_any_counts | [[52, 180, 0], [13, 36, 0], [23, 72, 0], [25, 72, 0]] | [positive,resolved,unknown]; train, selection, calibration, test | stage_a_v3/stage_b_prevalence.json |
| v3.prevalence.voltage_bus_15.daily_any_pct | [28.888888888888886, 36.11111111111111, 31.944444444444443, 34.72222222222222] | %; train, selection, calibration, test | stage_a_v3/stage_b_prevalence.json |
| v3.prevalence.voltage_bus_15.hourly_counts | [[282, 4320, 0], [66, 864, 0], [101, 1728, 0], [142, 1728, 0]] | [positive,resolved,unknown]; train, selection, calibration, test | stage_a_v3/stage_b_prevalence.json |
| v3.prevalence.voltage_bus_15.hourly_pct | [6.527777777777779, 7.638888888888889, 5.844907407407407, 8.217592592592593] | %; train, selection, calibration, test | stage_a_v3/stage_b_prevalence.json |
| v3.prevalence.voltage_bus_21.daily_any_counts | [[42, 180, 0], [12, 36, 0], [15, 72, 0], [24, 72, 0]] | [positive,resolved,unknown]; train, selection, calibration, test | stage_a_v3/stage_b_prevalence.json |
| v3.prevalence.voltage_bus_21.daily_any_pct | [23.333333333333332, 33.33333333333333, 20.833333333333336, 33.33333333333333] | %; train, selection, calibration, test | stage_a_v3/stage_b_prevalence.json |
| v3.prevalence.voltage_bus_21.hourly_counts | [[204, 4320, 0], [47, 864, 0], [64, 1728, 0], [111, 1728, 0]] | [positive,resolved,unknown]; train, selection, calibration, test | stage_a_v3/stage_b_prevalence.json |
| v3.prevalence.voltage_bus_21.hourly_pct | [4.722222222222222, 5.439814814814815, 3.7037037037037033, 6.423611111111111] | %; train, selection, calibration, test | stage_a_v3/stage_b_prevalence.json |
| v3.prevalence.voltage_bus_22.daily_any_counts | [[53, 180, 0], [13, 36, 0], [27, 72, 0], [27, 72, 0]] | [positive,resolved,unknown]; train, selection, calibration, test | stage_a_v3/stage_b_prevalence.json |
| v3.prevalence.voltage_bus_22.daily_any_pct | [29.444444444444446, 36.11111111111111, 37.5, 37.5] | %; train, selection, calibration, test | stage_a_v3/stage_b_prevalence.json |
| v3.prevalence.voltage_bus_22.hourly_counts | [[295, 4320, 0], [65, 864, 0], [110, 1728, 0], [146, 1728, 0]] | [positive,resolved,unknown]; train, selection, calibration, test | stage_a_v3/stage_b_prevalence.json |
| v3.prevalence.voltage_bus_22.hourly_pct | [6.828703703703703, 7.523148148148148, 6.365740740740741, 8.449074074074074] | %; train, selection, calibration, test | stage_a_v3/stage_b_prevalence.json |
| v3.prevalence.voltage_bus_9.daily_any_counts | [[110, 180, 0], [28, 36, 0], [50, 72, 0], [46, 72, 0]] | [positive,resolved,unknown]; train, selection, calibration, test | stage_a_v3/stage_b_prevalence.json |
| v3.prevalence.voltage_bus_9.daily_any_pct | [61.111111111111114, 77.77777777777779, 69.44444444444444, 63.888888888888886] | %; train, selection, calibration, test | stage_a_v3/stage_b_prevalence.json |
| v3.prevalence.voltage_bus_9.hourly_counts | [[400, 4320, 0], [63, 864, 0], [155, 1728, 0], [142, 1728, 0]] | [positive,resolved,unknown]; train, selection, calibration, test | stage_a_v3/stage_b_prevalence.json |
| v3.prevalence.voltage_bus_9.hourly_pct | [9.25925925925926, 7.291666666666667, 8.969907407407407, 8.217592592592593] | %; train, selection, calibration, test | stage_a_v3/stage_b_prevalence.json |
| v3.prevalence_ratio.thermal_line_1 | 1.2141327623126337 | larger/smaller | stage_c/report.json#events |
| v3.prevalence_ratio.voltage_bus_1 | 1.018970189701897 | larger/smaller | stage_c/report.json#events |
| v3.prevalence_ratio.voltage_bus_11 | 1.1390845070422535 | larger/smaller | stage_c/report.json#events |
| v3.prevalence_ratio.voltage_bus_13 | 1.1979166666666667 | larger/smaller | stage_c/report.json#events |
| v3.prevalence_ratio.voltage_bus_15 | 1.4059405940594059 | larger/smaller | stage_c/report.json#events |
| v3.prevalence_ratio.voltage_bus_21 | 1.734375 | larger/smaller | stage_c/report.json#events |
| v3.prevalence_ratio.voltage_bus_22 | 1.3272727272727272 | larger/smaller | stage_c/report.json#events |
| v3.prevalence_ratio.voltage_bus_9 | 1.0915492957746478 | larger/smaller | stage_c/report.json#events |
| v3.probe.h1.raw.normalized_mse | 0.034262705630202436 | train-scaled MSE | stage_b/probe/report.json#horizons/1 |
| v3.probe.h1.raw_plus_genco.normalized_mse | 0.03603915356460208 | train-scaled MSE | stage_b/probe/report.json#horizons/1 |
| v3.probe.h24.raw.normalized_mse | 0.18435845627732347 | train-scaled MSE | stage_b/probe/report.json#horizons/24 |
| v3.probe.h24.raw_plus_genco.normalized_mse | 0.19562049654703303 | train-scaled MSE | stage_b/probe/report.json#horizons/24 |
| v3.queue_wait_seconds | 1268.7373235225677 | seconds; completed jobs, excludes dependency wait | stage_d/final_release/context.json#claims/v3.queue_wait_seconds |
| v3.test_hours_high_profile | 289 | hours | stage_b/diagnostics/report.json#ref_counts |
| v3.training_wall_seconds | 21585.33293390274 | seconds; job wall including startup | stage_d/final_release/context.json#claims/v3.training_wall_seconds |
| v4.A.family_es.Pd | 0.4961371180659714 | separately train-scaled score | v4/stage_b/report.json#modes/A/family_energy/Pd |
| v4.A.family_es.Pg | 0.41189473742731575 | separately train-scaled score | v4/stage_b/report.json#modes/A/family_energy/Pg |
| v4.A.family_es.Qd | null | separately train-scaled score | v4/stage_b/report.json#modes/A/family_energy/Qd |
| v4.A.family_es.Qg | 0.4443447181816371 | separately train-scaled score | v4/stage_b/report.json#modes/A/family_energy/Qg |
| v4.A.family_es.Va | 0.3182929424261368 | separately train-scaled score | v4/stage_b/report.json#modes/A/family_energy/Va |
| v4.A.family_es.Vm | 0.17184184487282528 | separately train-scaled score | v4/stage_b/report.json#modes/A/family_energy/Vm |
| v4.A.family_es.other | 0.30431089399889816 | separately train-scaled score | v4/stage_b/report.json#modes/A/family_energy/other |
| v4.A.joint_es | 0.3033365471731542 | train-scaled score | v4/stage_b/report.json#modes/A |
| v4.A.p_imbalance | 1.113322703081177 | MW/bus | v4/stage_b/report.json#modes/A |
| v4.A.pg_free_coverage90_pct | 61.326767091541136 | % | v4/stage_b/report.json#modes/A |
| v4.A.pg_free_mae | 3.1547447790240826 | MW | v4/stage_b/report.json#modes/A |
| v4.A.q_imbalance | 0.28372386030045843 | MVAr/bus | v4/stage_b/report.json#modes/A |
| v4.A.vm_coverage50_pct | 14.517463996027146 | % | v4/stage_b/report.json#modes/A |
| v4.A.vm_coverage90_pct | 31.708326436020528 | % | v4/stage_b/report.json#modes/A |
| v4.A.vm_crps | 0.0009578046187918492 | p.u. | v4/stage_b/report.json#modes/A |
| v4.A.vm_width90 | 0.00198866921413657 | p.u. | v4/stage_b/report.json#modes/A |
| v4.A_vs_S.joint_es_reduction_pct | 38.9553913910881 | % | v4/stage_b/report.json#1-A.joint_energy_score/S.joint_energy_score |
| v4.C.family_es.Pd | 0.4315783700942591 | separately train-scaled score | v4/stage_b/report.json#modes/C/family_energy/Pd |
| v4.C.family_es.Pg | 0.663572675425662 | separately train-scaled score | v4/stage_b/report.json#modes/C/family_energy/Pg |
| v4.C.family_es.Qd | null | separately train-scaled score | v4/stage_b/report.json#modes/C/family_energy/Qd |
| v4.C.family_es.Qg | 0.6057206094513999 | separately train-scaled score | v4/stage_b/report.json#modes/C/family_energy/Qg |
| v4.C.family_es.Va | 0.5705163063573417 | separately train-scaled score | v4/stage_b/report.json#modes/C/family_energy/Va |
| v4.C.family_es.Vm | 0.26078082864853613 | separately train-scaled score | v4/stage_b/report.json#modes/C/family_energy/Vm |
| v4.C.family_es.other | 0.402610076117625 | separately train-scaled score | v4/stage_b/report.json#modes/C/family_energy/other |
| v4.C.joint_es | 0.42471924816790174 | train-scaled score | v4/stage_b/report.json#modes/C |
| v4.C.p_imbalance | 1.1962197243974366e-13 | MW/bus | v4/stage_b/report.json#modes/C |
| v4.C.pg_free_coverage90_pct | 79.28736964078796 | % | v4/stage_b/report.json#modes/C |
| v4.C.pg_free_mae | 6.048000630715082 | MW | v4/stage_b/report.json#modes/C |
| v4.C.q_imbalance | 2.839119304170612e-13 | MVAr/bus | v4/stage_b/report.json#modes/C |
| v4.C.vm_coverage50_pct | 41.4252607184241 | % | v4/stage_b/report.json#modes/C |
| v4.C.vm_coverage90_pct | 80.07780168846217 | % | v4/stage_b/report.json#modes/C |
| v4.C.vm_crps | 0.0012983292262371455 | p.u. | v4/stage_b/report.json#modes/C |
| v4.C.vm_width90 | 0.005832988251992502 | p.u. | v4/stage_b/report.json#modes/C |
| v4.D.family_es.Pd | 0.4961371180659714 | separately train-scaled score | v4/stage_b/report.json#modes/D/family_energy/Pd |
| v4.D.family_es.Pg | 0.5791233711315456 | separately train-scaled score | v4/stage_b/report.json#modes/D/family_energy/Pg |
| v4.D.family_es.Qd | null | separately train-scaled score | v4/stage_b/report.json#modes/D/family_energy/Qd |
| v4.D.family_es.Qg | 1.9811448463230095 | separately train-scaled score | v4/stage_b/report.json#modes/D/family_energy/Qg |
| v4.D.family_es.Va | 0.459460407679777 | separately train-scaled score | v4/stage_b/report.json#modes/D/family_energy/Va |
| v4.D.family_es.Vm | 0.29482764012888957 | separately train-scaled score | v4/stage_b/report.json#modes/D/family_energy/Vm |
| v4.D.family_es.other | 0.6184018202454294 | separately train-scaled score | v4/stage_b/report.json#modes/D/family_energy/other |
| v4.D.joint_es | 0.6341014093833078 | train-scaled score | v4/stage_b/report.json#modes/D |
| v4.D.p_imbalance | 0.03736919916212871 | MW/bus | v4/stage_b/report.json#modes/D |
| v4.D.pg_free_coverage90_pct | 18.13441483198146 | % | v4/stage_b/report.json#modes/D |
| v4.D.pg_free_mae | 3.7703969091645146 | MW | v4/stage_b/report.json#modes/D |
| v4.D.q_imbalance | 0.03641998319040234 | MVAr/bus | v4/stage_b/report.json#modes/D |
| v4.D.vm_coverage50_pct | 2.789273299122662 | % | v4/stage_b/report.json#modes/D |
| v4.D.vm_coverage90_pct | 7.0269822877007115 | % | v4/stage_b/report.json#modes/D |
| v4.D.vm_crps | 0.001821009464737999 | p.u. | v4/stage_b/report.json#modes/D |
| v4.D.vm_width90 | 0.000847657516467962 | p.u. | v4/stage_b/report.json#modes/D |
| v4.P.family_es.Pd | 0.46939650512203557 | separately train-scaled score | v4/stage_b/report.json#modes/P/family_energy/Pd |
| v4.P.family_es.Pg | 0.6085871425444213 | separately train-scaled score | v4/stage_b/report.json#modes/P/family_energy/Pg |
| v4.P.family_es.Qd | null | separately train-scaled score | v4/stage_b/report.json#modes/P/family_energy/Qd |
| v4.P.family_es.Qg | 2.041032479214797 | separately train-scaled score | v4/stage_b/report.json#modes/P/family_energy/Qg |
| v4.P.family_es.Va | 0.5228669885849206 | separately train-scaled score | v4/stage_b/report.json#modes/P/family_energy/Va |
| v4.P.family_es.Vm | 0.3287693067545328 | separately train-scaled score | v4/stage_b/report.json#modes/P/family_energy/Vm |
| v4.P.family_es.other | 0.6354295508958729 | separately train-scaled score | v4/stage_b/report.json#modes/P/family_energy/other |
| v4.P.joint_es | 0.658759766565152 | train-scaled score | v4/stage_b/report.json#modes/P |
| v4.P.p_imbalance | 0.03586445316858845 | MW/bus | v4/stage_b/report.json#modes/P |
| v4.P.pg_free_coverage90_pct | 0.0 | % | v4/stage_b/report.json#modes/P |
| v4.P.pg_free_mae | 3.5052918618308735 | MW | v4/stage_b/report.json#modes/P |
| v4.P.q_imbalance | 0.03644897746244008 | MVAr/bus | v4/stage_b/report.json#modes/P |
| v4.P.vm_coverage50_pct | 0.0 | % | v4/stage_b/report.json#modes/P |
| v4.P.vm_coverage90_pct | 0.0 | % | v4/stage_b/report.json#modes/P |
| v4.P.vm_crps | 0.0020004954700381427 | p.u. | v4/stage_b/report.json#modes/P |
| v4.P.vm_width90 | 0.0 | p.u. | v4/stage_b/report.json#modes/P |
| v4.P0.family_es.Pd | 0.46939652139232485 | separately train-scaled score | v4/stage_b/report.json#modes/P0/family_energy/Pd |
| v4.P0.family_es.Pg | 0.6085871531629139 | separately train-scaled score | v4/stage_b/report.json#modes/P0/family_energy/Pg |
| v4.P0.family_es.Qd | null | separately train-scaled score | v4/stage_b/report.json#modes/P0/family_energy/Qd |
| v4.P0.family_es.Qg | 2.0410324789307452 | separately train-scaled score | v4/stage_b/report.json#modes/P0/family_energy/Qg |
| v4.P0.family_es.Va | 0.5228670034863558 | separately train-scaled score | v4/stage_b/report.json#modes/P0/family_energy/Va |
| v4.P0.family_es.Vm | 0.32876931351815636 | separately train-scaled score | v4/stage_b/report.json#modes/P0/family_energy/Vm |
| v4.P0.family_es.other | 0.6354295481538605 | separately train-scaled score | v4/stage_b/report.json#modes/P0/family_energy/other |
| v4.P0.joint_es | 0.6587597506545725 | train-scaled score | v4/stage_b/report.json#modes/P0 |
| v4.P0.p_imbalance | 0.03586445316858845 | MW/bus | v4/stage_b/report.json#modes/P0 |
| v4.P0.pg_free_coverage90_pct | 0.0 | % | v4/stage_b/report.json#modes/P0 |
| v4.P0.pg_free_mae | 3.5052918618308735 | MW | v4/stage_b/report.json#modes/P0 |
| v4.P0.q_imbalance | 0.03644897746244008 | MVAr/bus | v4/stage_b/report.json#modes/P0 |
| v4.P0.vm_coverage50_pct | 0.0 | % | v4/stage_b/report.json#modes/P0 |
| v4.P0.vm_coverage90_pct | 0.0 | % | v4/stage_b/report.json#modes/P0 |
| v4.P0.vm_crps | 0.0020004954700381427 | p.u. | v4/stage_b/report.json#modes/P0 |
| v4.P0.vm_width90 | 0.0 | p.u. | v4/stage_b/report.json#modes/P0 |
| v4.S.family_es.Pd | 0.3339893041168392 | separately train-scaled score | v4/stage_b/report.json#modes/S/family_energy/Pd |
| v4.S.family_es.Pg | 0.4183512285298083 | separately train-scaled score | v4/stage_b/report.json#modes/S/family_energy/Pg |
| v4.S.family_es.Qd | null | separately train-scaled score | v4/stage_b/report.json#modes/S/family_energy/Qd |
| v4.S.family_es.Qg | 1.7816470332219052 | separately train-scaled score | v4/stage_b/report.json#modes/S/family_energy/Qg |
| v4.S.family_es.Va | 0.3502493864642214 | separately train-scaled score | v4/stage_b/report.json#modes/S/family_energy/Va |
| v4.S.family_es.Vm | 0.2317307916733543 | separately train-scaled score | v4/stage_b/report.json#modes/S/family_energy/Vm |
| v4.S.family_es.other | 0.47678491465245937 | separately train-scaled score | v4/stage_b/report.json#modes/S/family_energy/other |
| v4.S.joint_es | 0.49690964375987523 | train-scaled score | v4/stage_b/report.json#modes/S |
| v4.S.p_imbalance | 0.049634626011868625 | MW/bus | v4/stage_b/report.json#modes/S |
| v4.S.pg_free_coverage90_pct | 82.79258400926999 | % | v4/stage_b/report.json#modes/S |
| v4.S.pg_free_mae | 3.4465969994672365 | MW | v4/stage_b/report.json#modes/S |
| v4.S.q_imbalance | 0.04146036583703948 | MVAr/bus | v4/stage_b/report.json#modes/S |
| v4.S.vm_coverage50_pct | 10.453567290183745 | % | v4/stage_b/report.json#modes/S |
| v4.S.vm_coverage90_pct | 28.604535672901836 | % | v4/stage_b/report.json#modes/S |
| v4.S.vm_crps | 0.0014660415444228262 | p.u. | v4/stage_b/report.json#modes/S |
| v4.S.vm_width90 | 0.0035275971113974385 | p.u. | v4/stage_b/report.json#modes/S |
| v4.best_epoch | 14 | epoch | stage_d/final_release/context.json#claims/v4.best_epoch |
| v4.labels.attempted | 10080 | hours | stage_d/final_release/context.json#claims/v4.labels.attempted |
| v4.labels.unresolved | 2 | hours | stage_d/final_release/context.json#claims/v4.labels.unresolved |
| v4.labels.valid | 10078 | hours | stage_d/final_release/context.json#claims/v4.labels.valid |
| v4.nominal_load_range | [0.8, 1.0] | nominal multiplier | stage_d/final_release/context.json#claims/v4.nominal_load_range |
| v4.physics_weight | 1.0 | lambda | stage_d/final_release/context.json#claims/v4.physics_weight |
| v4.prevalence.thermal_line_1.daily_any_counts | [[180, 180, 0], [36, 36, 0], [72, 72, 0], [72, 72, 0]] | [positive,resolved,unknown]; train, selection, calibration, test | stage_a_v4/stage_b_prevalence.json |
| v4.prevalence.thermal_line_1.daily_any_pct | [100.0, 100.0, 100.0, 100.0] | %; train, selection, calibration, test | stage_a_v4/stage_b_prevalence.json |
| v4.prevalence.thermal_line_1.hourly_counts | [[4320, 4320, 0], [864, 864, 0], [1728, 1728, 0], [1726, 1726, 2]] | [positive,resolved,unknown]; train, selection, calibration, test | stage_a_v4/stage_b_prevalence.json |
| v4.prevalence.thermal_line_1.hourly_pct | [100.0, 100.0, 100.0, 100.0] | %; train, selection, calibration, test | stage_a_v4/stage_b_prevalence.json |
| v4.prevalence.voltage_bus_1.daily_any_counts | [[180, 180, 0], [36, 36, 0], [72, 72, 0], [72, 72, 0]] | [positive,resolved,unknown]; train, selection, calibration, test | stage_a_v4/stage_b_prevalence.json |
| v4.prevalence.voltage_bus_1.daily_any_pct | [100.0, 100.0, 100.0, 100.0] | %; train, selection, calibration, test | stage_a_v4/stage_b_prevalence.json |
| v4.prevalence.voltage_bus_1.hourly_counts | [[4320, 4320, 0], [864, 864, 0], [1728, 1728, 0], [1726, 1726, 2]] | [positive,resolved,unknown]; train, selection, calibration, test | stage_a_v4/stage_b_prevalence.json |
| v4.prevalence.voltage_bus_1.hourly_pct | [100.0, 100.0, 100.0, 100.0] | %; train, selection, calibration, test | stage_a_v4/stage_b_prevalence.json |
| v4.prevalence.voltage_bus_11.daily_any_counts | [[180, 180, 0], [36, 36, 0], [72, 72, 0], [72, 72, 0]] | [positive,resolved,unknown]; train, selection, calibration, test | stage_a_v4/stage_b_prevalence.json |
| v4.prevalence.voltage_bus_11.daily_any_pct | [100.0, 100.0, 100.0, 100.0] | %; train, selection, calibration, test | stage_a_v4/stage_b_prevalence.json |
| v4.prevalence.voltage_bus_11.hourly_counts | [[4320, 4320, 0], [864, 864, 0], [1728, 1728, 0], [1726, 1726, 2]] | [positive,resolved,unknown]; train, selection, calibration, test | stage_a_v4/stage_b_prevalence.json |
| v4.prevalence.voltage_bus_11.hourly_pct | [100.0, 100.0, 100.0, 100.0] | %; train, selection, calibration, test | stage_a_v4/stage_b_prevalence.json |
| v4.prevalence.voltage_bus_13.daily_any_counts | [[180, 180, 0], [36, 36, 0], [72, 72, 0], [72, 72, 0]] | [positive,resolved,unknown]; train, selection, calibration, test | stage_a_v4/stage_b_prevalence.json |
| v4.prevalence.voltage_bus_13.daily_any_pct | [100.0, 100.0, 100.0, 100.0] | %; train, selection, calibration, test | stage_a_v4/stage_b_prevalence.json |
| v4.prevalence.voltage_bus_13.hourly_counts | [[4320, 4320, 0], [864, 864, 0], [1728, 1728, 0], [1726, 1726, 2]] | [positive,resolved,unknown]; train, selection, calibration, test | stage_a_v4/stage_b_prevalence.json |
| v4.prevalence.voltage_bus_13.hourly_pct | [100.0, 100.0, 100.0, 100.0] | %; train, selection, calibration, test | stage_a_v4/stage_b_prevalence.json |
| v4.prevalence.voltage_bus_15.daily_any_counts | [[0, 180, 0], [0, 36, 0], [0, 72, 0], [0, 70, 2]] | [positive,resolved,unknown]; train, selection, calibration, test | stage_a_v4/stage_b_prevalence.json |
| v4.prevalence.voltage_bus_15.daily_any_pct | [0.0, 0.0, 0.0, 0.0] | %; train, selection, calibration, test | stage_a_v4/stage_b_prevalence.json |
| v4.prevalence.voltage_bus_15.hourly_counts | [[0, 4320, 0], [0, 864, 0], [0, 1728, 0], [0, 1726, 2]] | [positive,resolved,unknown]; train, selection, calibration, test | stage_a_v4/stage_b_prevalence.json |
| v4.prevalence.voltage_bus_15.hourly_pct | [0.0, 0.0, 0.0, 0.0] | %; train, selection, calibration, test | stage_a_v4/stage_b_prevalence.json |
| v4.prevalence.voltage_bus_21.daily_any_counts | [[0, 180, 0], [0, 36, 0], [0, 72, 0], [0, 70, 2]] | [positive,resolved,unknown]; train, selection, calibration, test | stage_a_v4/stage_b_prevalence.json |
| v4.prevalence.voltage_bus_21.daily_any_pct | [0.0, 0.0, 0.0, 0.0] | %; train, selection, calibration, test | stage_a_v4/stage_b_prevalence.json |
| v4.prevalence.voltage_bus_21.hourly_counts | [[0, 4320, 0], [0, 864, 0], [0, 1728, 0], [0, 1726, 2]] | [positive,resolved,unknown]; train, selection, calibration, test | stage_a_v4/stage_b_prevalence.json |
| v4.prevalence.voltage_bus_21.hourly_pct | [0.0, 0.0, 0.0, 0.0] | %; train, selection, calibration, test | stage_a_v4/stage_b_prevalence.json |
| v4.prevalence.voltage_bus_22.daily_any_counts | [[0, 180, 0], [0, 36, 0], [0, 72, 0], [0, 70, 2]] | [positive,resolved,unknown]; train, selection, calibration, test | stage_a_v4/stage_b_prevalence.json |
| v4.prevalence.voltage_bus_22.daily_any_pct | [0.0, 0.0, 0.0, 0.0] | %; train, selection, calibration, test | stage_a_v4/stage_b_prevalence.json |
| v4.prevalence.voltage_bus_22.hourly_counts | [[0, 4320, 0], [0, 864, 0], [0, 1728, 0], [0, 1726, 2]] | [positive,resolved,unknown]; train, selection, calibration, test | stage_a_v4/stage_b_prevalence.json |
| v4.prevalence.voltage_bus_22.hourly_pct | [0.0, 0.0, 0.0, 0.0] | %; train, selection, calibration, test | stage_a_v4/stage_b_prevalence.json |
| v4.prevalence.voltage_bus_9.daily_any_counts | [[0, 180, 0], [0, 36, 0], [0, 72, 0], [0, 70, 2]] | [positive,resolved,unknown]; train, selection, calibration, test | stage_a_v4/stage_b_prevalence.json |
| v4.prevalence.voltage_bus_9.daily_any_pct | [0.0, 0.0, 0.0, 0.0] | %; train, selection, calibration, test | stage_a_v4/stage_b_prevalence.json |
| v4.prevalence.voltage_bus_9.hourly_counts | [[0, 4320, 0], [0, 864, 0], [0, 1728, 0], [0, 1726, 2]] | [positive,resolved,unknown]; train, selection, calibration, test | stage_a_v4/stage_b_prevalence.json |
| v4.prevalence.voltage_bus_9.hourly_pct | [0.0, 0.0, 0.0, 0.0] | %; train, selection, calibration, test | stage_a_v4/stage_b_prevalence.json |
| v4.queue_wait_seconds | 1315.3907716274261 | seconds; completed jobs, excludes dependency wait | stage_d/final_release/context.json#claims/v4.queue_wait_seconds |
| v4.test_bus_load_below_support_pct | 9.750566893424036 | % of active bus-hours, including history | stage_d/final_release/context.json#claims/v4.test_bus_load_below_support_pct |
| v4.training_wall_seconds | 18975.72829437256 | seconds; job wall including startup | stage_d/final_release/context.json#claims/v4.training_wall_seconds |
