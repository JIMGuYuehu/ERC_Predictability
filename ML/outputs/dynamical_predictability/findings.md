# Dynamical ozone-loss predictability scan

## Research question

How early can winter stratospheric dynamics identify and quantify severe Arctic ozone-loss years in the longrun?

## Predictability definition

Predictability is defined in two complementary ways. Event predictability asks whether a year enters the lowest 25% of March-April minimum polar-cap O3. Intensity predictability asks how severe the year is, using both March-April mean ozone-loss anomaly (`MA_O3_loss`) and the annual March-April minimum O3 depth (`MA_O3_min_DU`).

The stable lead-time gate uses event skill plus seasonal-loss regression skill. Minimum-O3 depth is reported separately because a single-day minimum is noisier, but it directly answers whether we can estimate roughly how low the ozone gets.

O3 is not evaluated as a standalone predictor set. When included, it is treated as a passive tracer inside `DYN_PLUS_PASSIVE_TRACER`.

## Stable B2000WCN lead windows

| model_family      | predictor_set           | earliest_single_usable_lead_days | earliest_7day_stable_usable_lead_days | best_auc | best_loss_correlation | best_loss_mse_skill | best_min_o3_correlation | best_min_o3_mse_skill |
| ----------------- | ----------------------- | -------------------------------- | ------------------------------------- | -------- | --------------------- | ------------------- | ----------------------- | --------------------- |
| random_forest     | DYN_PLUS_PASSIVE_TRACER | 22.000                           | 15.000                                | 0.875    | 0.827                 | 0.684               | 0.716                   | 0.512                 |
| gradient_boosting | DYN_PLUS_PASSIVE_TRACER | 20.000                           | 15.000                                | 0.878    | 0.825                 | 0.679               | 0.712                   | 0.506                 |
| linear_l2         | DYN_PLUS_PASSIVE_TRACER | 15.000                           | 15.000                                | 0.903    | 0.858                 | 0.736               | 0.725                   | 0.525                 |
| gradient_boosting | DYN_COUPLED             | nan                              | nan                                   | 0.814    | 0.688                 | 0.472               | 0.583                   | 0.339                 |
| random_forest     | DYN_COUPLED             | nan                              | nan                                   | 0.786    | 0.665                 | 0.441               | 0.578                   | 0.334                 |
| gradient_boosting | VORTEX_STATE            | nan                              | nan                                   | 0.785    | 0.630                 | 0.396               | 0.554                   | 0.307                 |
| linear_l2         | DYN_COUPLED             | nan                              | nan                                   | 0.784    | 0.696                 | 0.484               | 0.593                   | 0.351                 |
| random_forest     | VORTEX_STATE            | nan                              | nan                                   | 0.783    | 0.604                 | 0.361               | 0.538                   | 0.285                 |
| gradient_boosting | WAVE_FORCING            | nan                              | nan                                   | 0.782    | 0.626                 | 0.392               | 0.527                   | 0.275                 |
| linear_l2         | VORTEX_STATE            | nan                              | nan                                   | 0.777    | 0.609                 | 0.370               | 0.534                   | 0.285                 |
| linear_l2         | WAVE_FORCING            | nan                              | nan                                   | 0.770    | 0.654                 | 0.428               | 0.563                   | 0.317                 |
| random_forest     | WAVE_FORCING            | nan                              | nan                                   | 0.766    | 0.617                 | 0.379               | 0.533                   | 0.283                 |

## BWCN external check

| model_family      | predictor_set           | best_auc | best_loss_correlation | best_min_o3_correlation | best_top25_hit_rate | best_top25_false_alarm_rate |
| ----------------- | ----------------------- | -------- | --------------------- | ----------------------- | ------------------- | --------------------------- |
| gradient_boosting | DYN_PLUS_PASSIVE_TRACER | 1.000    | 0.951                 | 0.926                   | 1.000               | 0.000                       |
| linear_l2         | DYN_PLUS_PASSIVE_TRACER | 1.000    | 0.928                 | 0.874                   | 1.000               | 0.000                       |
| random_forest     | VORTEX_STATE            | 0.989    | 0.923                 | 0.904                   | 0.800               | 0.056                       |
| gradient_boosting | DYN_COUPLED             | 0.989    | 0.886                 | 0.852                   | 0.800               | 0.056                       |
| linear_l2         | DYN_COUPLED             | 0.989    | 0.863                 | 0.815                   | 0.800               | 0.056                       |
| random_forest     | WAVE_FORCING            | 0.989    | 0.851                 | 0.810                   | 0.800               | 0.056                       |
| random_forest     | DYN_PLUS_PASSIVE_TRACER | 0.988    | 0.954                 | 0.924                   | 0.800               | 0.056                       |
| linear_l2         | WAVE_FORCING            | 0.988    | 0.833                 | 0.798                   | 0.800               | 0.056                       |
| gradient_boosting | VORTEX_STATE            | 0.978    | 0.913                 | 0.901                   | 0.800               | 0.056                       |
| gradient_boosting | WAVE_FORCING            | 0.976    | 0.861                 | 0.810                   | 0.800               | 0.056                       |
| random_forest     | DYN_COUPLED             | 0.967    | 0.895                 | 0.857                   | 0.800               | 0.056                       |
| linear_l2         | VORTEX_STATE            | 0.967    | 0.830                 | 0.802                   | 0.800               | 0.056                       |

## BWCN0008 top-quartile alarms

| model_family      | predictor_set           | earliest_top25_alarm_lead_days |
| ----------------- | ----------------------- | ------------------------------ |
| gradient_boosting | VORTEX_STATE            | 59                             |
| linear_l2         | VORTEX_STATE            | 59                             |
| linear_l2         | DYN_PLUS_PASSIVE_TRACER | 59                             |
| linear_l2         | DYN_COUPLED             | 59                             |
| random_forest     | VORTEX_STATE            | 59                             |
| linear_l2         | WAVE_FORCING            | 59                             |
| gradient_boosting | DYN_COUPLED             | 58                             |
| gradient_boosting | WAVE_FORCING            | 57                             |
| random_forest     | WAVE_FORCING            | 57                             |
| random_forest     | DYN_COUPLED             | 57                             |
| random_forest     | DYN_PLUS_PASSIVE_TRACER | 53                             |
| gradient_boosting | DYN_PLUS_PASSIVE_TRACER | 52                             |

## Figure guide

- `fig_leadtime_auc.png`: lead time versus severe-event AUC; answers when low-O3 events become separable.
- `fig_leadtime_reg_correlation.png`: lead time versus `MA_O3_loss` correlation; answers when seasonal loss magnitude becomes estimable.
- `fig_leadtime_min_o3_correlation.png`: lead time versus `MA_O3_min_DU` correlation; answers when the approximate event depth becomes estimable.
- `fig_bwcn0008_probability.png`: BWCN0008 event probability versus lead time; answers whether a known severe case is flagged early.

## Interpretation

- The vortex-only and wave-only partitions test whether the early signal sits primarily in the vortex state or in wave forcing.
- `DYN_COUPLED` tests the physical pathway in which wave forcing modulates the vortex and therefore later ozone loss.
- `DYN_PLUS_PASSIVE_TRACER` tests whether ozone as a passive tracer of prior transport/chemistry sharpens alarms; it should not be interpreted as a separate non-dynamical source of predictability.
- True vortex morphology is not yet diagnosed here. NAM vertical coupling is only a compact proxy; a next step would compute vortex moment diagnostics from geopotential height or PV-like fields.
