# Marina Figure 2 Reproduction Note

Script:

```bash
cd /home/weiji/restart_exam/code_cleaned
/home/weiji/miniconda3/envs/jimnew/bin/python Longrun/date_treatment/reproduce_marina_figure2.py
```

Main output:

- `Longrun/Visualization/plots/TEST_FWD_plots/marina_figure2_waccm_panel_reproduction.png`
- `Longrun/Visualization/plots/TEST_FWD_plots/marina_figure2_waccm_panel_reproduction.pdf`
- support table: `Longrun/date_treatment/clim3d_marina_repro_report/marina_figure2_waccm_panel_reproduction.csv`

## What Marina's Figure 2 WACCM Panel Uses

- Data source: Marina's WACCM `INT-3D` and `CLIM-3D` files, not the cleaned
  timefixed products.
- O3 selection: 60-90N, pressure-level discrete 30-70 hPa partial column from
  the 5-day running-mean O3 files.
- FWD source: Marina's saved `FW_vertical_newthreshIII_0.npy` and
  `FW_vertical_newthreshIII_1.npy`.
- Plotted quantity: selected-year FWD minus the all-year mean FWD at each
  pressure level.
- Layout: one WACCM vertical-profile panel with `INT-3D` solid and `CLIM-3D`
  dotted. Low-O3 years are blue; high-O3 years are red.
- Shading: Marina's plotting cell shades only the `INT-3D` low/high groups with
  plus/minus one standard deviation.

## Why Our Earlier Plots Did Not Look Like Marina's Figure 2

The earlier cleaned-data figure was answering a different question. It used our
cleaned FWD/O3 products and split windows such as March-April versus March-FWD.
Marina's Figure 2 WACCM panel instead uses her saved vertical FWD arrays and a
single low/high O3-year selection, then plots FWD anomalies relative to each
level's all-year mean.

The difference is therefore not just styling. The data source, O3 ranking
diagnostic, FWD source, anomaly convention, panel layout, and uncertainty
shading convention are all different.

## Why Marina And Cleaned Partial O3 Differ In Absolute Value

Marina's nominal 30-70 hPa O3 partial column is a pressure-level discrete sum
after `ml2pl` interpolation. On the WACCM pressure grid she selects the 30, 50,
and 70 hPa levels and assigns each selected level a backward pressure increment:

| selected level | weight | effective layer |
| ---: | ---: | ---: |
| 30 hPa | 10 hPa | 20-30 hPa |
| 50 hPa | 20 hPa | 30-50 hPa |
| 70 hPa | 20 hPa | 50-70 hPa |

So the Marina "30-70 hPa" diagnostic effectively integrates 20-70 hPa with a
total 50 hPa pressure thickness. The cleaned `partial_O3` product instead uses
CAM/WACCM hybrid interface pressures and exact layer overlap with the requested
30-70 hPa interval, so it is a true 30-70 hPa column with 40 hPa thickness.

In the feature-matched CLIM-3D years, this explains the absolute offset: Marina
March-April O3 averages about `131.1 DU`, while the cleaned hybrid-overlap
30-70 hPa product averages about `109.5 DU`; the mean offset is about `+21.7 DU`.
The year-to-year ranking is still highly correlated, but the absolute
climatology is not meant to be identical.

## Implementation Detail To Remember

In Marina's notebook-era `find_ozone_extremes_FW` code, the active slice after
selecting March-April is `group[0:FW_date]`, where `FW_date` is still a Jan-Jun
day index. For WACCM this effectively keeps the March-April window for most
years. The reproduction script keeps this behavior explicitly so that the plot
matches the Figure 2 logic rather than a corrected March-to-FWD interpretation.
