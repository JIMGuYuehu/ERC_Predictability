#!/usr/bin/env python
"""Reproduce the WACCM panel of Marina/Friedel Figure 2 from Marina data.

This script intentionally follows the plotting/data conventions in
``/home/gchiodo/code/Marina/Marina_home/FW_paper/FW_vertical.ipynb``:

* WACCM INT-3D and CLIM-3D only.
* Ozone years are selected from Marina's 5-day running-mean pressure-level O3.
* The vertical FWD profiles are Marina's saved ``FW_vertical_newthreshIII`` npy
  arrays.
* The plotted quantity is FWD anomaly at each pressure level, i.e. selected
  low/high-O3-year FWD minus the all-year mean FWD at that level.

Outputs are written below
``Longrun/Visualization/plots/TEST_FWD_plots``.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from scipy.stats import ttest_ind


ROOT = Path("/home/weiji/restart_exam/code_cleaned")
PLOT_DIR = ROOT / "Longrun/Visualization/plots/TEST_FWD_plots"
TABLE_DIR = ROOT / "Longrun/date_treatment/clim3d_marina_repro_report"

MARINA_INT_O3 = Path(
    "/mnt/backup_ETH/Marina/ozone_extremes/WACCM/INT_O3_2000/"
    "CO2x1SmidEmin_yBWCN.cam.h1.0101-0300.O3.isobar.zm.runmean5d.nc"
)
MARINA_CLIM3D_O3 = Path(
    "/mnt/backup_ETH/Marina/WACCM/NOCHEM_2000_3D/combined/"
    "B2000WCN.NOCOUPL.e122.f19_g16.int.zm.runmean5.nc"
)
MARINA_FW_DATE_DIR = Path("/home/gchiodo/code/Marina/Marina_home/FW_paper/FW_dates")
MARINA_ORIGINAL_FIG = Path("/home/gchiodo/code/Marina/Marina_home/FW_paper/Plots/FW_vertical_mean.png")

M_AIR = 28.964 / (6.022e23)
GRAVITY_CGS = 980.6
DU_FACTOR = 2.687e16
TARGET_LAT = np.linspace(-90.0, 90.0, 73)
POLAR_TARGET_LAT = TARGET_LAT[TARGET_LAT >= 60.0]
COMMON_PLEV_HPA = np.array([1, 2, 3, 5, 10, 20, 30, 50], dtype=float)


def savefig(fig: plt.Figure, stem: str) -> None:
    """Save a figure as png/pdf/svg in the TEST_FWD plot directory."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in {
        ".png": {"dpi": 300},
        ".pdf": {},
        ".svg": {},
    }.items():
        fig.savefig(PLOT_DIR / f"{stem}{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def interp_to_mfriedel_lat(data: np.ndarray, source_lat: np.ndarray) -> np.ndarray:
    """Linearly interpolate ``(..., lat)`` data to Marina's 2.5-degree grid."""
    out = np.empty(data.shape[:-1] + (len(POLAR_TARGET_LAT),), dtype=np.float64)
    for j, target in enumerate(POLAR_TARGET_LAT):
        hi = int(np.searchsorted(source_lat, target, side="left"))
        if hi == 0:
            out[..., j] = data[..., 0]
        elif hi >= len(source_lat):
            out[..., j] = data[..., -1]
        else:
            lo = hi - 1
            w = (target - source_lat[lo]) / (source_lat[hi] - source_lat[lo])
            out[..., j] = data[..., lo] * (1.0 - w) + data[..., hi] * w
    return out


def marina_pressure_weights_pa(plev: np.ndarray, selected: np.ndarray, units: str) -> np.ndarray:
    """Return Marina's backward pressure increments in Pa for selected levels."""
    delta = np.zeros_like(plev, dtype=float)
    for idx in range(1, len(plev)):
        delta[idx] = plev[idx] - plev[idx - 1]
    weights = delta[selected]
    if units == "hPa":
        weights = weights * 100.0
    return weights.astype(float)


def time_year_and_doy(ds: Dataset, n_years: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Return 1-based model-year index and no-leap day-of-year arrays."""
    if "date" in ds.variables:
        date = np.asarray(ds.variables["date"][:], dtype=np.int64)
        raw_year = date // 10000
        unique = np.unique(raw_year)
        year_lookup = {year: idx + 1 for idx, year in enumerate(unique[:n_years])}
        year_index = np.array([year_lookup.get(year, -1) for year in raw_year], dtype=int)
        mmdd = date % 10000
        month = mmdd // 100
        day = mmdd % 100
        month_ends = np.array([31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365])
        doy = np.empty_like(year_index)
        for m in range(1, 13):
            mask = month == m
            prev = month_ends[m - 2] if m > 1 else 0
            doy[mask] = prev + day[mask]
        return year_index, doy.astype(int)

    time = np.asarray(ds.variables["time"][:], dtype=float)
    day0 = int(round(float(time[0])))
    offset = np.rint(time - day0).astype(int)
    year_index = offset // 365 + 1
    doy = offset % 365 + 1
    return year_index.astype(int), doy.astype(int)


def marina_o3_partial_column(path: Path, units: str, n_years: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Compute Marina-style 60-90N, 30-70 hPa partial O3 for all days.

    The pressure integral follows Marina's pressure-level discrete sum, not the
    cleaned hybrid-interface method.  Returned arrays are shaped
    ``(n_years, 365)``.
    """
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = "int3d" if "CO2x1SmidEmin" in path.name else "clim3d"
    cache_path = TABLE_DIR / f"marina_figure2_o3col_60_90N_30_70hPa_MA_{cache_key}.npy"
    if cache_path.exists():
        return np.load(cache_path), np.arange(1, n_years + 1, dtype=int)

    out = np.full((n_years, 365), np.nan, dtype=float)
    with Dataset(path) as ds:
        plev = np.asarray(ds.variables["plev"][:], dtype=float)
        lat = np.asarray(ds.variables["lat"][:], dtype=float)
        year_index, doy = time_year_and_doy(ds, n_years=n_years)

        if units == "hPa":
            keep = np.where(np.isin(np.round(plev, 6), [30.0, 50.0, 70.0]))[0]
        else:
            keep = np.where(np.isin(np.round(plev, 6), [3000.0, 5000.0, 7000.0]))[0]
        if len(keep) != 3:
            raise ValueError(f"Could not identify 30/50/70 hPa levels in {path}")

        lat_weights = np.cos(np.deg2rad(POLAR_TARGET_LAT))
        dp_pa = marina_pressure_weights_pa(plev, keep, units=units)
        o3_var = ds.variables["O3"]
        o3_var.set_auto_mask(False)
        fill_value = getattr(o3_var, "_FillValue", getattr(o3_var, "missing_value", None))

        for year in range(1, n_years + 1):
            idx = np.where((year_index == year) & (doy >= 60) & (doy <= 120))[0]
            if len(idx) == 0:
                continue
            # March-April is contiguous in these no-leap files.  Reading it as
            # a slice is much faster than fancy-indexing the whole 200-year
            # time axis or reading January-December when Figure 2 only ranks
            # March-April ozone.
            start = int(idx[0])
            stop = int(idx[-1]) + 1
            raw = np.asarray(o3_var[start:stop, keep[0] : keep[-1] + 1, ...], dtype=np.float64)
            if fill_value is not None:
                raw[np.isclose(raw, float(fill_value))] = np.nan
            raw[np.abs(raw) > 1e20] = np.nan
            if raw.ndim == 4:
                raw = raw[..., 0]

            polar = interp_to_mfriedel_lat(raw, lat)
            polar_mean = np.average(polar, axis=-1, weights=lat_weights)
            column = polar_mean * dp_pa[None, :] * 10.0 / (GRAVITY_CGS * M_AIR)
            column = np.nansum(column, axis=1) / DU_FACTOR
            out[year - 1, doy[start:stop] - 1] = column
    np.save(cache_path, out)
    return out, np.arange(1, n_years + 1, dtype=int)


def select_ozone_extreme_years(
    o3_year_day: np.ndarray,
    fwd50_janjun0: np.ndarray,
    extreme_years: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select low/high O3 years with Marina's Figure-2 logic.

    Marina's function first selects March-April, then attempts to limit by FWD.
    Because the active slice is ``group[0:FW_date]`` in the notebook-era code,
    and ``FW_date`` is a Jan-Jun day index, this is effectively all March-April
    for almost every model year.  This function keeps that behavior explicitly.
    """
    ma = o3_year_day[:, 59:120]
    annual_min = np.full(ma.shape[0], np.nan, dtype=float)
    annual_max = np.full(ma.shape[0], np.nan, dtype=float)
    for idx in range(ma.shape[0]):
        group = ma[idx]
        if np.isfinite(fwd50_janjun0[idx]) and 0 < fwd50_janjun0[idx] - 58 <= len(group):
            group_new = group[: int(fwd50_janjun0[idx])]
        else:
            group_new = group
        if np.isfinite(group_new).any():
            annual_min[idx] = np.nanmin(group_new)
            annual_max[idx] = np.nanmax(group_new)

    valid_min = np.where(np.isfinite(annual_min))[0]
    valid_max = np.where(np.isfinite(annual_max))[0]
    if len(valid_min) < extreme_years or len(valid_max) < extreme_years:
        raise ValueError(
            f"Only {len(valid_min)} finite low-O3 and {len(valid_max)} finite high-O3 candidate years; "
            f"need {extreme_years}."
        )
    low = valid_min[np.argsort(annual_min[valid_min])[:extreme_years]]
    high = valid_max[np.argsort(annual_max[valid_max])[-extreme_years:][::-1]]
    return low.astype(int), high.astype(int), annual_min


def vertical_anomalies(fwd_vertical: np.ndarray, low: np.ndarray, high: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return FWD anomalies for low/high ozone years, matching Marina's helper."""
    all_mean = np.nanmean(fwd_vertical, axis=1, keepdims=True)
    low_anom = fwd_vertical[:, low] - all_mean
    high_anom = fwd_vertical[:, high] - all_mean
    return low_anom, high_anom


def build_waccm_reproduction() -> pd.DataFrame:
    """Build and save the WACCM Figure-2 reproduction and support table."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    cases = [
        {
            "name": "INT-3D",
            "o3_path": MARINA_INT_O3,
            "units": "hPa",
            "fwd_path": MARINA_FW_DATE_DIR / "FW_vertical_newthreshIII_0.npy",
            "line": "-",
            "shade": True,
        },
        {
            "name": "CLIM-3D",
            "o3_path": MARINA_CLIM3D_O3,
            "units": "Pa",
            "fwd_path": MARINA_FW_DATE_DIR / "FW_vertical_newthreshIII_1.npy",
            "line": ":",
            "shade": False,
        },
    ]

    records: list[dict[str, float | str | int]] = []
    prepared = []
    for case in cases:
        fwd = np.load(case["fwd_path"]).astype(float)
        if fwd.shape[0] != 10:
            raise ValueError(f"Expected level x year FWD array at {case['fwd_path']}, got {fwd.shape}")
        o3, years = marina_o3_partial_column(case["o3_path"], units=case["units"])
        low, high, annual_min = select_ozone_extreme_years(o3, fwd[-1], extreme_years=50)
        low_anom, high_anom = vertical_anomalies(fwd, low, high)
        prepared.append((case, low_anom, high_anom, annual_min, low, high))

        for level_idx, plev in enumerate([0.1, 0.5, *COMMON_PLEV_HPA.tolist()]):
            for group, subset, anom in [
                ("low_O3", low, low_anom[level_idx]),
                ("high_O3", high, high_anom[level_idx]),
            ]:
                records.append(
                    {
                        "case": case["name"],
                        "group": group,
                        "plev_hpa": float(plev),
                        "n": int(len(subset)),
                        "mean_fwd_anomaly_days": float(np.nanmean(anom)),
                        "std_fwd_anomaly_days": float(np.nanstd(anom)),
                        "mean_absolute_fwd_day_janjun0": float(np.nanmean(fwd[level_idx, subset])),
                        "all_year_mean_fwd_day_janjun0": float(np.nanmean(fwd[level_idx])),
                    }
                )

    fig, ax = plt.subplots(figsize=(6.5 / 2.54, 6.0 / 2.54), constrained_layout=True)
    plev_plot = COMMON_PLEV_HPA
    y = np.log(plev_plot)

    for case, low_anom, high_anom, *_ in prepared:
        low_plot = np.nanmean(low_anom[2:, :], axis=1)
        high_plot = np.nanmean(high_anom[2:, :], axis=1)
        ax.plot(low_plot, y, color="darkblue", linestyle=case["line"], linewidth=1.0, label=f"{case['name']}, low O3")
        ax.plot(high_plot, y, color="firebrick", linestyle=case["line"], linewidth=1.0, label=f"{case['name']}, high O3")

        if case["shade"]:
            ax.fill_betweenx(
                y,
                low_plot - np.nanstd(low_anom[2:, :], axis=1),
                low_plot + np.nanstd(low_anom[2:, :], axis=1),
                color="royalblue",
                alpha=0.1,
                linewidth=0,
            )
            ax.fill_betweenx(
                y,
                high_plot - np.nanstd(high_anom[2:, :], axis=1),
                high_plot + np.nanstd(high_anom[2:, :], axis=1),
                color="indianred",
                alpha=0.1,
                linewidth=0,
            )

    int_low, int_high = prepared[0][1], prepared[0][2]
    clim_low, clim_high = prepared[1][1], prepared[1][2]
    _, p_low = ttest_ind(int_low[2:, :], clim_low[2:, :], axis=1, equal_var=False, nan_policy="omit")
    _, p_high = ttest_ind(int_high[2:, :], clim_high[2:, :], axis=1, equal_var=False, nan_policy="omit")
    int_low_mean = np.nanmean(int_low[2:, :], axis=1)
    int_high_mean = np.nanmean(int_high[2:, :], axis=1)
    for idx, p in enumerate(p_low):
        if p < 0.05:
            ax.plot(
                int_low_mean[idx],
                y[idx],
                marker="o",
                color="none",
                markersize=5,
                markeredgewidth=0.3,
                markeredgecolor="darkblue",
            )
    for idx, p in enumerate(p_high):
        if p < 0.05:
            ax.plot(
                int_high_mean[idx],
                y[idx],
                marker="o",
                color="none",
                markersize=5,
                markeredgewidth=0.3,
                markeredgecolor="firebrick",
            )

    ax.axvline(0, color="k", linewidth=0.5)
    ax.set_xlim(-35, 35)
    ax.set_xticks([-30, -20, -10, 0, 10, 20, 30])
    ax.set_ylim(np.log(50), np.log(1))
    ax.invert_yaxis()
    ax.set_yticks([np.log(50), np.log(10), np.log(5), np.log(1)])
    ax.set_yticklabels(["50", "10", "5", "1"])
    ax.set_xlabel("Deviation from mean FSW date (days)")
    ax.set_ylabel("altitude (hPa)")
    ax.set_title("WACCM", fontweight="bold")
    ax.legend(loc="center left", fontsize=6, frameon=False)
    ax.text(0.04, 0.91, "(a)", transform=ax.transAxes, fontsize=8, bbox={"boxstyle": "square", "ec": "1", "fc": "1"})
    savefig(fig, "marina_figure2_waccm_panel_reproduction")

    summary = pd.DataFrame.from_records(records)
    summary_path = TABLE_DIR / "marina_figure2_waccm_panel_reproduction.csv"
    summary.to_csv(summary_path, index=False)

    note_path = TABLE_DIR / "marina_figure2_waccm_panel_reproduction_notes.md"
    note_path.write_text(
        "\n".join(
            [
                "# Marina Figure 2 WACCM Panel Reproduction Notes",
                "",
                "This reproduction uses Marina's own WACCM INT-3D and CLIM-3D files and",
                "her saved `FW_vertical_newthreshIII_0/1.npy` vertical FWD arrays.",
                "",
                "Important implementation details:",
                "",
                "- The plotted quantity is FWD anomaly at each level:",
                "  selected-year FWD minus all-year mean FWD at that level.",
                "- O3 years are ranked with Marina's pressure-level discrete 30-70 hPa",
                "  polar-cap partial-column method on the 5-day running-mean O3 files.",
                "- The notebook-era `find_ozone_extremes_FW` active slice is effectively",
                "  March-April for WACCM because it uses `group[0:FW_date]` after already",
                "  selecting March-April.",
                "- INT-3D is solid, CLIM-3D is dotted; low O3 is blue, high O3 is red.",
                "- Blue/red shading is shown only for INT-3D, matching Marina's Figure 2",
                "  plotting cell.",
                "",
                f"Original Marina figure checked at: `{MARINA_ORIGINAL_FIG}`",
            ]
        )
        + "\n"
    )
    return summary


def main() -> None:
    summary = build_waccm_reproduction()
    print("[WRITE]", PLOT_DIR / "marina_figure2_waccm_panel_reproduction.png")
    print("[WRITE]", TABLE_DIR / "marina_figure2_waccm_panel_reproduction.csv")
    print(summary.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
