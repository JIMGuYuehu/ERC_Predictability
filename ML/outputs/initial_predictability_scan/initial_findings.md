# Initial Longrun Ozone-Loss Predictability Scan

## Operational Definition

A severe ozone-loss event is the lowest quartile of March-April minimum 5-day-smoothed 60-90N, 30-70 hPa partial ozone. Predictability is called usable when B2000WCN out-of-fold skill passes the thresholds documented in `ML/README.md`.

## Earliest Usable Lead

| model_family  | predictor_set | earliest_single_usable_lead_days | earliest_7day_stable_usable_lead_days | best_auc | best_reg_correlation | best_reg_mse_skill |
| ------------- | ------------- | -------------------------------- | ------------------------------------- | -------- | -------------------- | ------------------ |
| linear        | O3_MEMORY     | 36.000                           | 36.000                                | 0.903    | 0.862                | 0.742              |
| random_forest | O3_MEMORY     | 31.000                           | 24.000                                | 0.881    | 0.824                | 0.678              |
| random_forest | DYN_PLUS_O3   | 23.000                           | 23.000                                | 0.874    | 0.827                | 0.683              |
| linear        | DYN_PLUS_O3   | 17.000                           | 17.000                                | 0.911    | 0.861                | 0.741              |
| linear        | DYN           | nan                              | nan                                   | 0.794    | 0.650                | 0.422              |
| random_forest | DYN           | nan                              | nan                                   | 0.770    | 0.635                | 0.402              |

Best stable window: linear / O3_MEMORY, 36 days before 1 March.

## BWCN External Check

| model_family  | predictor_set | best_auc | best_reg_correlation | best_top25_hit_rate | best_top25_false_alarm_rate |
| ------------- | ------------- | -------- | -------------------- | ------------------- | --------------------------- |
| linear        | DYN           | 0.965    | 0.864                | 0.800               | 0.056                       |
| linear        | DYN_PLUS_O3   | 1.000    | 0.923                | 1.000               | 0.000                       |
| linear        | O3_MEMORY     | 0.988    | 0.911                | 0.800               | 0.056                       |
| random_forest | DYN           | 0.965    | 0.892                | 0.800               | 0.056                       |
| random_forest | DYN_PLUS_O3   | 1.000    | 0.949                | 1.000               | 0.000                       |
| random_forest | O3_MEMORY     | 0.976    | 0.914                | 0.800               | 0.056                       |

## BWCN0008 Trace

| model_family  | predictor_set | earliest_top25_alarm_lead_days |
| ------------- | ------------- | ------------------------------ |
| linear        | DYN           | 59                             |
| linear        | DYN_PLUS_O3   | 59                             |
| linear        | O3_MEMORY     | 59                             |
| random_forest | DYN           | 58                             |
| random_forest | O3_MEMORY     | 51                             |
| random_forest | DYN_PLUS_O3   | 50                             |

## Output Files

- `b2000wcn_cv_skill_summary.csv`
- `b2000wcn_cv_predictions.csv`
- `bwcn_external_skill_summary.csv`
- `bwcn_external_predictions.csv`
- `bwcn0008_trace.csv`
- `predictability_windows.csv`
- `leadtime_auc.png`
- `leadtime_reg_correlation.png`
- `bwcn0008_probability_trace.png`
