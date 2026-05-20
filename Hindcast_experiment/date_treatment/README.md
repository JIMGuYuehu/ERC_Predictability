# Hindcast Longrun-Style Diagnostics

This directory contains the Hindcast branch of the cleaned workflow. All code is
written to be run with the `jimnew` environment.

## What Was Confirmed

- Existing `Longrun/date_treatment/Nam_calculation.ipynb` sets `BWCN` with `use_reference=True`, so BWCN AO/NAM is projected onto the `B2000WCN001002` reference modes.
- In the current Longrun notebook, `B2000WCN_NOCOUPL001002` and `B2000WCN007009010011_Clim3D` are not projected onto B2000WCN; they train their own EOFs.
- Existing Hindcast `EPflux_daily/*.Fz.nc` files are legacy products. Their NetCDF metadata says `w=None`; they also are not split into all/wave1/wave2/rest.
- A source-file header scan of `/mnt/backup_ETH/lens/*/*.cam.h3.*.nc*` on 2026-05-20 found no `OMEGA` in NOCOUPL Hindcast files: 8 NOCOUPL cases, 720 h3 files, 0 with `OMEGA`. The scan can be reproduced with `scripts/audit_hindcast_raw_omega.py`; its latest CSV is `hindcast_raw_omega_inventory.csv`.
- The same scan found sparse coupled `OMEGA` coverage: 16 coupled cases, 1347 h3 files, 116 with `OMEGA`, limited to partial `0008-02` and `0008-02_v2` coverage.
- Therefore the production Hindcast EP-flux workflow intentionally abandons the OMEGA pressure-velocity correction and uses `w=None` for both coupled and NOCOUPL members, while keeping Longrun pressure-grid, `DO_UBAR=True`, and wave-split conventions.

## Variables To Extract

Run this first if any required Hindcast diagnostic variable is missing:

```bash
cd /home/weiji/restart_exam/code_cleaned
MAX_JOBS=16 ./Hindcast_experiment/date_treatment/extract_hindcast_longrun_vars_with_omega.sh
```

Required variables for the Hindcast diagnostics are:

- `U`, `V`, `T`, `PS`, `Z3`, `O3`

`OMEGA` is not required for the production Hindcast EP-flux product. To archive
any available coupled `OMEGA` segments for later inspection, run the extractor
with `EXTRACT_OMEGA=1`; NOCOUPL members will still log as missing `OMEGA`.

## Run Order

Use `/home/weiji/miniconda3/envs/jimnew/bin/python`.

1. Inventory current Hindcast data:

```bash
/home/weiji/miniconda3/envs/jimnew/bin/python Hindcast_experiment/date_treatment/scripts/hindcast_inventory.py
```

2. Preserve Longrun AO/NAM first-mode patterns for the three requested longrun cases:

```bash
/home/weiji/miniconda3/envs/jimnew/bin/python Longrun/date_treatment/save_longrun_ao_nam_mode1_patterns.py
```

3. Compute new Hindcast EP flux with Longrun pressure/wave conventions and no OMEGA correction:

```bash
/home/weiji/miniconda3/envs/jimnew/bin/python Hindcast_experiment/date_treatment/scripts/compute_hindcast_epflux.py --max-workers 4
```

4. Compute Hindcast eddy heat flux:

```bash
/home/weiji/miniconda3/envs/jimnew/bin/python Hindcast_experiment/date_treatment/scripts/compute_hindcast_eddyheatflux.py --max-workers 4
```

5. Compute Hindcast AO/NAM using B2000WCN projection:

```bash
/home/weiji/miniconda3/envs/jimnew/bin/python Hindcast_experiment/date_treatment/scripts/compute_hindcast_ao_nam_b2000wcn_projection.py --max-workers 4
```

6. Compute Hindcast partial O3 without ranking:

```bash
/home/weiji/miniconda3/envs/jimnew/bin/python Hindcast_experiment/date_treatment/scripts/compute_hindcast_partial_o3.py --max-workers 4
```

7. Compute Hindcast final warming date:

```bash
/home/weiji/miniconda3/envs/jimnew/bin/python Hindcast_experiment/date_treatment/scripts/compute_hindcast_fwd.py --max-workers 4
```

8. Compare old and new EP flux products:

```bash
/home/weiji/miniconda3/envs/jimnew/bin/python Hindcast_experiment/date_treatment/scripts/compare_hindcast_epflux_legacy.py
```

## Why Recompute

- Recompute `EPflux`: old Hindcast EP flux does not provide the Longrun
  all/wave1/wave2/rest split or this workflow's consolidated member file.
  OMEGA correction is deliberately disabled because NOCOUPL source files have
  no `OMEGA` and coupled coverage is sparse.
- Recompute `AO/NAM`: the new Hindcast requirement is B2000WCN first-mode
  projection for every member, not member/self-trained EOFs.
- Recompute `Eddyheatflux`: the new output includes both `v'T'` and
  `v'theta'` on the Longrun pressure grid.
- Recompute `partial_O3`: the new output uses the Longrun hybrid-interface
  overlap method and writes `1-100 hPa`, `30-70 hPa`, and `30-100 hPa` ranges
  without ranking.
- Recompute `FWD`: the new output is pressure-level resolved over the same
  compact 1-50 hPa grid as Longrun.

## Output Locations

Outputs are written below each `/mnt/soclim0/public_data/weiji/Hindcast/<case>`:

- `EPflux_daily_ubar/{all_waves,wave1,wave2,wave_rest}/`
- `Eddyheatflux_daily/`
- `NAM_B2000WCN_projection/`
- `partial_O3/`
- `final_warming_date/`

Quality-control comparison CSVs are written under:

- `EPflux_daily_ubar/quality_control/`
