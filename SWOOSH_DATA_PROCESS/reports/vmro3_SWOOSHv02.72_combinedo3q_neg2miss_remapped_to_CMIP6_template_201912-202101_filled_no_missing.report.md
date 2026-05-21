# SWOOSH Missing-Value Fill Report

- Input: `/mnt/soclim0/public_data/weiji/swoosh/processed_like_input4MIPs/vmro3_SWOOSHv02.72_combinedo3q_neg2miss_remapped_to_CMIP6_template_201912-202101.nc`
- Output: `/home/weiji/restart_exam/code_cleaned/SWOOSH_DATA_PROCESS/outputs/vmro3_SWOOSHv02.72_combinedo3q_neg2miss_remapped_to_CMIP6_template_201912-202101_filled_no_missing.nc`
- Variable: `vmro3`
- Missing before: `1064448` / `12773376` (8.333333%)
- Missing after: `0` / `12773376` (0.000000%)
- Fill dimensions: `lat, lon, plev, time`

## Fill Steps

- `lat`: 1064448 -> 0 missing; filled 1064448

## Interpretation

The original target file did contain missing/non-finite values in `vmro3`. In this run all missing values were removed during the latitude pass, which is consistent with polar latitude rows lying outside the valid SWOOSH source-grid support after remapping to the CMIP6-style template grid.
