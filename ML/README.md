# Longrun Ozone-Loss Predictability ML

This directory is now notebook-first. The cleaned exploration is:

- `Longrun_Ozone_Dynamical_Predictability.ipynb`

The research framing is dynamical:

- predict severe March-April polar-cap ozone loss in the longrun,
- split predictors into vortex state, wave forcing, coupled dynamics, and a
  dynamics-plus-passive-tracer set,
- treat ozone itself only as a passive tracer of prior stratospheric
  transport/chemistry, not as a standalone non-dynamical predictor set.

Run from the repository root:

```bash
cd /home/weiji/restart_exam/code_cleaned
jupyter nbconvert --execute --inplace ML/Longrun_Ozone_Dynamical_Predictability.ipynb
```

Compact outputs are written to:

```text
ML/outputs/dynamical_predictability/
```

The old script package and large intermediate CSV outputs were removed so this
directory stays readable.
