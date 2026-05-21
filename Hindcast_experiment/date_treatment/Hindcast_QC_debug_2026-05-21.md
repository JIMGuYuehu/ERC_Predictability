# Hindcast QC Debug Notes

Date: 2026-05-21  
Environment: `jimnew`

## EPFlux Legacy QC Error

The command

```bash
/home/weiji/miniconda3/envs/jimnew/bin/python Hindcast_experiment/date_treatment/scripts/compare_hindcast_epflux_legacy.py
```

failed because `area_weights_for_lat()` returned `NaN` outside the requested
latitude band. `xarray.DataArray.weighted()` rejects weights containing missing
values.

Fix:

- `Hindcast_experiment/date_treatment/scripts/hindcast_common.py` now returns
  zero weight outside the selected latitude band.
- `compare_hindcast_epflux_legacy.py` now accepts `--qc-output-root`, so QC CSVs
  can be written inside `code_cleaned` during debugging.

## Current Hindcast Output Status

Checked `/mnt/soclim0/public_data/weiji/Hindcast` after the current runs:

| Product | Complete cases | Member rows | Status |
| --- | ---: | ---: | --- |
| `EPflux_daily_ubar` | `24/24` | `719` | all processing summaries are `ok` |
| `Eddyheatflux_daily` | `24/24` | `719` | all processing summaries are `ok` |
| `final_warming_date` | `24/24` | `719` | all processing summaries are `ok` |
| `partial_O3` main NetCDF | `0/24` | n/a | previous run failed from the latitude-weight NaN bug |
| `NAM_B2000WCN_projection` | `0/24` | n/a | not generated yet |

The partial O3 bug is the same root cause as the EPFlux QC error. A no-write
single-member test after the fix produced valid `1-100 hPa`, `30-70 hPa`, and
`30-100 hPa` partial-column variables.

## OMEGA / EPFlux Consistency

OMEGA extraction is incomplete:

| Case | OMEGA member files |
| --- | ---: |
| `0008-02` | `29` |
| `0008-02_v2` | `29` |

All other checked Hindcast cases have zero extracted OMEGA files. The current
Hindcast EPFlux script is therefore intentionally consistent across all cases:

- `USE_OMEGA = False`
- output directory: `EPflux_daily_ubar`
- NetCDF method attribute: `ComputeEPfluxDiv with DO_UBAR=True and w=None`
- NetCDF `use_omega_w_correction = False`

This means the current EPFlux product is not omega-corrected, including the two
cases where OMEGA files exist.

## EPFlux Legacy QC Result

QC CSVs were written to:

```text
Hindcast_experiment/date_treatment/qc/epflux_legacy_compare/
```

Summary:

- `23` case CSVs were produced.
- `659` legacy-member comparisons were `ok`.
- `0019-03_NOCOUPL` was skipped because it has no legacy
  `EPflux_daily/*.Fz.nc` files to compare against.
- Correlation range: `0.9629` to `0.9873`.
- Median correlation: `0.9796`.
- Median RMSE: `0.00282`.

The case-level summary is:

```text
Hindcast_experiment/date_treatment/qc/epflux_legacy_compare/epflux_legacy_compare_case_summary.csv
```

## AO/NAM Mode Scripts

There are two similarly named mode-saving files:

- `Longrun/date_treatment/Save_AO_NAM_mode1_patterns.ipynb`
  - Notebook wrapper only.
  - It locates and runs `save_longrun_ao_nam_mode1_patterns.py`.
- `Longrun/date_treatment/save_longrun_ao_nam_mode1_patterns.py`
  - Real script.
  - Rebuilds and saves first-mode AO/NAM patterns for the three requested
    longrun cases:
    `B2000WCN001002`, `B2000WCN007009010011_Clim3D`, and
    `B2000WCN_NOCOUPL001002`.

For Hindcast AO/NAM, the relevant script is:

```text
Hindcast_experiment/date_treatment/scripts/compute_hindcast_ao_nam_b2000wcn_projection.py
```

It rebuilds AO/NAM reference modes from
`/mnt/soclim0/public_data/weiji/B2000WCN001002_timefixed` and projects Hindcast
Z3 onto those B2000WCN001002 first EOF modes. It does not train Hindcast EOFs.
However, no Hindcast `NAM_B2000WCN_projection` outputs are present yet, so this
step still needs to be run.

## Next Commands

Because partial O3 and AO/NAM are not complete, run these next:

```bash
cd /home/weiji/restart_exam/code_cleaned

/home/weiji/miniconda3/envs/jimnew/bin/python \
  Hindcast_experiment/date_treatment/scripts/compute_hindcast_partial_o3.py \
  --overwrite --max-workers 4

/home/weiji/miniconda3/envs/jimnew/bin/python \
  Hindcast_experiment/date_treatment/scripts/compute_hindcast_ao_nam_b2000wcn_projection.py \
  --overwrite --max-workers 4
```

Then rerun inventory/QC:

```bash
/home/weiji/miniconda3/envs/jimnew/bin/python \
  Hindcast_experiment/date_treatment/scripts/hindcast_inventory.py

/home/weiji/miniconda3/envs/jimnew/bin/python \
  Hindcast_experiment/date_treatment/scripts/compare_hindcast_epflux_legacy.py \
  --overwrite
```

