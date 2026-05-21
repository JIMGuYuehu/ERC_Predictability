# SWOOSH Data Process

This directory contains the reproducible check/fill workflow for the urgent SWOOSH
NetCDF issue reported in May 2026.

The old development notebooks under `code/20260323swoosh` were used only as
read-only reference material. No files in `code/` are modified by this workflow.

## Target File

Default input:

```text
/mnt/soclim0/public_data/weiji/swoosh/processed_like_input4MIPs/vmro3_SWOOSHv02.72_combinedo3q_neg2miss_remapped_to_CMIP6_template_201912-202101.nc
```

This is the SWOOSH v02.72 `combinedo3q` ozone file previously remapped to the
CMIP6-style `vmro3` template grid for 2019-12 through 2021-01.

## Quick Commands

Run with the repository Conda environment:

```bash
/home/weiji/miniconda3/envs/jimnew/bin/python SWOOSH_DATA_PROCESS/check_swoosh_missing.py
/home/weiji/miniconda3/envs/jimnew/bin/python SWOOSH_DATA_PROCESS/fill_swoosh_target.py
```

The filled NetCDF is written to:

```text
SWOOSH_DATA_PROCESS/outputs/vmro3_SWOOSHv02.72_combinedo3q_neg2miss_remapped_to_CMIP6_template_201912-202101_filled_no_missing.nc
```

NetCDF outputs in `outputs/` are intentionally ignored by Git. Small run reports
are written to `SWOOSH_DATA_PROCESS/reports/`.

## Current Run Result

The original target file does have actual missing values:

```text
vmro3 missing/non-finite = 1064448 / 12773376 = 8.333333%
```

The missing values are concentrated in eight complete polar latitude bands:

```text
-90.0, -88.10526275634766, -86.21052551269531, -84.31578826904297,
 84.31578826904297, 86.21052551269531, 88.10526275634766, 90.0
```

After running `fill_swoosh_target.py`, readback verification found:

```text
all checked numeric variables/coords/bounds missing/non-finite = 0
vmro3 missing/non-finite = 0 / 12773376 = 0.000000%
```

## Fill Method

The `vmro3` data are opened with CF masking enabled, so encoded fill values such
as `1.0e20` are treated as actual missing values. Missing/non-finite values are
then filled dimension by dimension:

1. `lat`
2. `lon`
3. `plev`
4. `time`

For each one-dimensional profile, internal gaps are linearly interpolated and
values outside the valid coordinate span are filled from the nearest valid edge.
This avoids introducing slope-based polar extrapolation while still removing
holes that can crash IFS.
