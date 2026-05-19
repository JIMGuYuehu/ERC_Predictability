# Hindcast Longrun-Style Diagnostics

This directory contains the Hindcast branch of the cleaned workflow. All code is
written to be run with the `jimnew` environment.

## What Was Confirmed

- Existing `Longrun/date_treatment/Nam_calculation.ipynb` sets `BWCN` with `use_reference=True`, so BWCN AO/NAM is projected onto the `B2000WCN001002` reference modes.
- In the current Longrun notebook, `B2000WCN_NOCOUPL001002` and `B2000WCN007009010011_Clim3D` are not projected onto B2000WCN; they train their own EOFs.
- Existing Hindcast `EPflux_daily/*.Fz.nc` files are legacy products. Their NetCDF metadata says `w=None`, so they do not include the Longrun OMEGA correction and they are not split into all/wave1/wave2/rest.
- Hindcast raw h3 files have mixed OMEGA availability. For example, some coupled `0008-02` members contain `OMEGA`, while checked NOCOUPL examples did not. The extraction script logs missing OMEGA member-by-member.

## Variables To Extract

Run this first if `OMEGA` or any Longrun-required variable is missing:

```bash
cd /home/weiji/restart_exam/code_cleaned
MAX_JOBS=16 ./Hindcast_experiment/date_treatment/extract_hindcast_longrun_vars_with_omega.sh
```

Required variables for the Longrun-consistent Hindcast diagnostics are:

- `U`, `V`, `T`, `OMEGA`, `PS`, `Z3`, `O3`

`OMEGA` is required for the preferred EP-flux product. Members without `OMEGA`
are skipped by default in the new EP-flux script so the output stays comparable
to Longrun `EPflux_daily_ubar_wcorr`.

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

3. Compute new Hindcast EP flux, Longrun-style:

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

- Recompute `EPflux`: old Hindcast EP flux does not include the OMEGA `-[u'w']`
  term and does not provide the Longrun all/wave1/wave2/rest split.
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

- `EPflux_daily_ubar_wcorr/{all_waves,wave1,wave2,wave_rest}/`
- `Eddyheatflux_daily/`
- `NAM_B2000WCN_projection/`
- `partial_O3/`
- `final_warming_date/`

Quality-control comparison CSVs are written under:

- `EPflux_daily_ubar_wcorr/quality_control/`

