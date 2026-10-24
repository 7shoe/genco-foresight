# Frozen calibration event prevalence

Rates use forecast-target hours and complete forecast days. Each cell is hourly / daily-any percent.

| Event | Train | Model selection | Risk calibration | Test | >1.5× shift: hourly / daily |
|---|---:|---:|---:|---:|---|
| voltage_bus_1 | 68.71 / 95.37 | 47.45 / 88.89 | 34.72 / 83.33 | 58.74 / 97.22 | yes / no |
| voltage_bus_9 | 9.45 / 65.74 | 7.06 / 47.22 | 4.63 / 41.67 | 6.94 / 54.17 | no / no |
| voltage_bus_11 | 37.60 / 66.67 | 18.40 / 52.78 | 6.37 / 52.78 | 26.68 / 54.17 | yes / no |
| voltage_bus_13 | 32.89 / 64.35 | 14.12 / 41.67 | 2.89 / 22.22 | 23.21 / 50.00 | yes / yes |
| voltage_bus_15 | 5.83 / 25.00 | 10.88 / 47.22 | 17.13 / 77.78 | 7.41 / 37.50 | yes / yes |
| voltage_bus_21 | 4.42 / 19.91 | 7.75 / 36.11 | 12.27 / 69.44 | 4.80 / 30.56 | yes / yes |
| voltage_bus_22 | 6.10 / 24.54 | 10.88 / 47.22 | 17.82 / 80.56 | 7.47 / 36.11 | yes / yes |
| thermal_line_1 | 31.91 / 63.89 | 14.24 / 41.67 | 2.20 / 25.00 | 22.69 / 50.00 | yes / yes |

For every flagged event/resolution, raw ensemble frequencies are primary; post-hoc maps are sensitivity results only. Full counts and all-hour rates are in the adjacent JSON/CSV.

Thermal line 1 is secondary: it is the single binding corridor. The realized daily maximum profile separates calibration outcomes perfectly (AUROC 1.0), as recorded in [events.json](stage_a_followup/events.json). This is an oracle diagnostic, not a permitted future input.
