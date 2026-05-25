# SWOOSH Additional-Year Plot Consistency

- Created UTC: `2026-05-25T13:03:56.713678+00:00`
- SWOOSH source: `/mnt/soclim0/public_data/weiji/swoosh/swoosh-v02.72-198401-202601-lonlatpress-20deg-5deg-L31.nc`
- Reference template: `/mnt/soclim0/andreas/vmro3_input4MIPs_ozone_CMIP6_UReading-CCMI_2020.nc`

| target year | window | plot | original missing | filled missing | finite-value max delta |
|---:|---|---|---:|---:|---:|
| 2011 | 201012-201201 | `/home/weiji/restart_exam/code_cleaned/SWOOSH_DATA_PROCESS/plots/O3_NHpolar_2011_original_vs_filled_target_ppmv.png` | 1064448 | 0 | 0.0 |
| 2018 | 201712-201901 | `/home/weiji/restart_exam/code_cleaned/SWOOSH_DATA_PROCESS/plots/O3_NHpolar_2018_original_vs_filled_target_ppmv.png` | 1064448 | 0 | 0.0 |
| 2023 | 202212-202401 | `/home/weiji/restart_exam/code_cleaned/SWOOSH_DATA_PROCESS/plots/O3_NHpolar_2023_original_vs_filled_target_ppmv.png` | 1064448 | 0 | 0.0 |

Finite-value max delta is computed only where the remapped pre-fill field already had finite values, after casting that pre-fill field to the filled file dtype. A value of `0.0` means the fill step changed only the original missing/non-finite grid cells.
