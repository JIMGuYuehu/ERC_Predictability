# ERC Predictability

Cleaned analysis notebooks and helper scripts for long-run WACCM/CESM-style
experiments on Arctic ozone, final stratospheric warming dates, EP-flux
diagnostics, NAM/AO indices, and related climatologies.

The repository currently contains code only. The large NetCDF inputs and derived
diagnostic products are stored outside the repository on local/shared storage,
for example under:

- `/mnt/soclim0/public_data/weiji`
- `/mnt/backup_ETH/Marina`

## Repository Layout

```text
code_cleaned/
└── Longrun/
    ├── diagnose_B2000WCN_time.log
    └── date_treatment/
        ├── Climatology.ipynb
        ├── Data_nan_fill.ipynb
        ├── Eddyheatflux.ipynb
        ├── Epflux_calculation.ipynb
        ├── Finalwarmingdate.ipynb
        ├── Nam_calculation.ipynb
        ├── Partial_O3_with_ranking.ipynb
        ├── TEST_NEW_DATA.ipynb
        ├── aostools_functions.py
        ├── fix_B2000WCN_run2_time_offset.sh
        └── repair_partial_o3_zero_days.py
```

## Main Workflows

- `Data_nan_fill.ipynb` documents preprocessing and missing-value handling for
  extracted WACCM variables.
- `Partial_O3_with_ranking.ipynb` computes Arctic partial-column ozone metrics
  and raw/5-day-running-mean high- and low-ozone year rankings.
- `Finalwarmingdate.ipynb` computes final warming dates on selected pressure
  levels.
- `Epflux_calculation.ipynb` computes EP-flux components and EP-flux divergence
  after hybrid-to-pressure interpolation.
- `Eddyheatflux.ipynb` computes eddy heat flux diagnostics.
- `Nam_calculation.ipynb` computes NAM/AO diagnostics.
- `Climatology.ipynb` builds all-year and ozone-extreme climatologies in shared
  NetCDF outputs.
- `TEST_NEW_DATA.ipynb` contains validation/debugging plots and Friedel et al.
  style diagnostics.

## Experiment Labels

The notebooks use these working labels:

- `INT-3D`: primarily `B2000WCN001002_timefixed`
- `CLIM-2D`: primarily `B2000WCN_NOCOUPL001002_timefixed`
- `CLIM-3D`: primarily `B2000WCN007009010011_Clim3D_timefixed`
- `BWCN`: shorter supporting run used in selected comparisons

Some debug blocks compare against Marina/Friedel et al. processed data under
`/mnt/backup_ETH/Marina`.

## Notes

- Paths are currently absolute and machine-specific.
- Notebooks are intended to be run from the project root or from
  `code_cleaned/Longrun/date_treatment`.
- Heavy computations are intentionally not rerun automatically when existing
  output files are detected.
- The included paper PDF, `acp-22-13997-2022.pdf`, is used as a local reference
  for reproducing Friedel et al. (2022)-style diagnostics.

## Environment

The code has been developed on the local `jimnew`/`jim2_py310` conda-style
environments with common scientific Python packages:

- `numpy`
- `pandas`
- `xarray`
- `netCDF4`
- `scipy`
- `matplotlib`
- `tqdm`

NCO/CDO/NCL utilities are used in some shell preprocessing and diagnostics.

