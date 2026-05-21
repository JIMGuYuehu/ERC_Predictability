# Longrun Ozone-Loss Predictability ML

This directory contains the cleaned ML exploration for longrun Arctic ozone-loss
predictability.

Research question:

```text
How early before 1 March can a severe March-April polar-cap ozone-loss year be
identified from physically interpretable winter information?
```

Initial operational definition:

- target: March-April mean ozone loss, defined as negative March-April mean
  anomaly of 5-day smoothed `60-90N`, `30-70 hPa` partial ozone.
- event: years in the lowest 25% of March-April minimum ozone.
- lead time: days before 1 March, using cutoffs from 1 January to 28 February.
- useful prediction: out-of-fold B2000WCN skill with event AUC >= 0.75,
  positive Brier skill, top-quartile hit rate >= 0.60, top-quartile false alarm
  rate <= 0.35, regression correlation >= 0.50, and positive regression MSE
  skill.

Feature groups are intentionally small:

- `DYN`: 50 hPa NAM short-window mean/trend plus 100 hPa EP-flux long-window
  mean.
- `O3_MEMORY`: ozone anomaly short-window mean/trend.
- `DYN_PLUS_O3`: the union of the two groups.

Run:

```bash
cd /home/weiji/restart_exam/code_cleaned
/home/weiji/miniconda3/envs/jimnew/bin/python ML/ozone_predictability/run_initial_predictability_scan.py
```

Outputs are written to `ML/outputs/initial_predictability_scan/`.
