# Dynamical ozone-loss predictability scan

Severe event target: lowest quartile of March-April minimum 5-day-smoothed 60-90N, 30-70 hPa partial ozone. O3 is not evaluated as a standalone predictor set; when used, it is treated as a passive tracer within `DYN_PLUS_PASSIVE_TRACER`.

## Stable B2000WCN lead windows

| model_family      | predictor_set           | earliest_single_usable_lead_days | earliest_7day_stable_usable_lead_days | best_auc | best_reg_correlation | best_reg_mse_skill |
| ----------------- | ----------------------- | -------------------------------- | ------------------------------------- | -------- | -------------------- | ------------------ |
| random_forest     | DYN_PLUS_PASSIVE_TRACER | 22.000                           | 15.000                                | 0.876    | 0.827                | 0.683              |
| gradient_boosting | DYN_PLUS_PASSIVE_TRACER | 20.000                           | 15.000                                | 0.880    | 0.832                | 0.691              |
| linear_l2         | DYN_PLUS_PASSIVE_TRACER | 15.000                           | 15.000                                | 0.906    | 0.858                | 0.736              |
| gradient_boosting | DYN_COUPLED             | nan                              | nan                                   | 0.794    | 0.687                | 0.471              |
| random_forest     | DYN_COUPLED             | nan                              | nan                                   | 0.787    | 0.658                | 0.433              |
| gradient_boosting | VORTEX_STATE            | nan                              | nan                                   | 0.785    | 0.630                | 0.396              |
| linear_l2         | DYN_COUPLED             | nan                              | nan                                   | 0.784    | 0.689                | 0.473              |
| random_forest     | VORTEX_STATE            | nan                              | nan                                   | 0.783    | 0.604                | 0.361              |
| linear_l2         | VORTEX_STATE            | nan                              | nan                                   | 0.777    | 0.609                | 0.370              |
| gradient_boosting | WAVE_FORCING            | nan                              | nan                                   | 0.777    | 0.613                | 0.376              |
| linear_l2         | WAVE_FORCING            | nan                              | nan                                   | 0.767    | 0.645                | 0.415              |
| random_forest     | WAVE_FORCING            | nan                              | nan                                   | 0.766    | 0.609                | 0.370              |

## BWCN external check

| model_family      | predictor_set           | best_auc | best_reg_correlation | best_top25_hit_rate | best_top25_false_alarm_rate |
| ----------------- | ----------------------- | -------- | -------------------- | ------------------- | --------------------------- |
| gradient_boosting | DYN_PLUS_PASSIVE_TRACER | 1.000    | 0.953                | 1.000               | 0.000                       |
| random_forest     | DYN_PLUS_PASSIVE_TRACER | 1.000    | 0.952                | 1.000               | 0.000                       |
| linear_l2         | DYN_PLUS_PASSIVE_TRACER | 1.000    | 0.927                | 1.000               | 0.000                       |
| random_forest     | WAVE_FORCING            | 1.000    | 0.862                | 1.000               | 0.000                       |
| gradient_boosting | WAVE_FORCING            | 1.000    | 0.860                | 1.000               | 0.000                       |
| random_forest     | VORTEX_STATE            | 0.989    | 0.923                | 0.800               | 0.056                       |
| gradient_boosting | DYN_COUPLED             | 0.989    | 0.891                | 0.800               | 0.056                       |
| linear_l2         | DYN_COUPLED             | 0.989    | 0.866                | 0.800               | 0.056                       |
| linear_l2         | WAVE_FORCING            | 0.988    | 0.833                | 0.800               | 0.056                       |
| gradient_boosting | VORTEX_STATE            | 0.978    | 0.913                | 0.800               | 0.056                       |
| random_forest     | DYN_COUPLED             | 0.978    | 0.885                | 0.800               | 0.056                       |
| linear_l2         | VORTEX_STATE            | 0.967    | 0.830                | 0.800               | 0.056                       |

## BWCN0008 top-quartile alarms

| model_family      | predictor_set           | earliest_top25_alarm_lead_days |
| ----------------- | ----------------------- | ------------------------------ |
| gradient_boosting | VORTEX_STATE            | 59                             |
| linear_l2         | VORTEX_STATE            | 59                             |
| linear_l2         | DYN_PLUS_PASSIVE_TRACER | 59                             |
| linear_l2         | DYN_COUPLED             | 59                             |
| random_forest     | VORTEX_STATE            | 59                             |
| linear_l2         | WAVE_FORCING            | 59                             |
| gradient_boosting | WAVE_FORCING            | 58                             |
| random_forest     | DYN_COUPLED             | 58                             |
| gradient_boosting | DYN_COUPLED             | 57                             |
| random_forest     | DYN_PLUS_PASSIVE_TRACER | 56                             |
| random_forest     | WAVE_FORCING            | 56                             |
| gradient_boosting | DYN_PLUS_PASSIVE_TRACER | 53                             |

## Interpretation

- The vortex-only and wave-only partitions test whether the early signal sits primarily in the vortex state or in wave forcing.
- `DYN_COUPLED` tests the physical pathway in which wave forcing modulates the vortex and therefore later ozone loss.
- `DYN_PLUS_PASSIVE_TRACER` tests whether ozone as a passive tracer of prior transport/chemistry sharpens alarms; it should not be interpreted as a separate non-dynamical source of predictability.
- True vortex morphology is not yet diagnosed here. NAM vertical coupling is only a compact proxy; a next step would compute vortex moment diagnostics from geopotential height or PV-like fields.
