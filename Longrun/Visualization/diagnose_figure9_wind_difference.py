#!/usr/bin/env python3
"""Diagnose WACCM Figure-9 wind-difference discrepancies.

This script reproduces the WACCM low/high-O3 wind-difference panels from
Marina Friedel's FW_atmos workflow and compares them with the current
timefixed climatology inputs used by Plots_for_Longrun.ipynb.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
PLOT_DIR = REPO_ROOT / "Longrun" / "Visualization" / "plots" / "figure9_wind_debug"
TMP_DIR = Path("/tmp") / "figure9_wind_debug"

MARINA_INT_O3 = Path(
    "/mnt/backup_ETH/Marina/ozone_extremes/WACCM/INT_O3_2000/"
    "CO2x1SmidEmin_yBWCN.cam.h1.0101-0300.O3.isobar.zm.runmean5d.nc"
)
MARINA_INT_U = Path("/mnt/backup_ETH/Marina/ozone_extremes/WACCM/INT_O3_2000/U.101-300.zm.nc")
MARINA_CLIM3D_O3 = Path(
    "/mnt/backup_ETH/Marina/WACCM/NOCHEM_2000_3D/combined/"
    "B2000WCN.NOCOUPL.e122.f19_g16.int.zm.runmean5.nc"
)
MARINA_CLIM3D_U = Path(
    "/mnt/backup_ETH/Marina/WACCM/NOCHEM_2000_3D/combined/"
    "B2000WCN.NOCOUPL.e122.f19_g16.int.zm.nc"
)

TIMEFIXED_ROOT = Path("/mnt/soclim0/public_data/weiji")
OUR_CASES = {
    "BWCN": TIMEFIXED_ROOT / "BWCN",
    "B2000WCN": TIMEFIXED_ROOT / "B2000WCN001002_timefixed",
    "CLIM3D": TIMEFIXED_ROOT / "B2000WCN007009010011_Clim3D_timefixed",
}

OUR_RANKING = {
    "BWCN": OUR_CASES["BWCN"] / "partial_O3" / "BWCN_partial_O3_ranking_MarApr_min_60_90N.csv",
    "B2000WCN": OUR_CASES["B2000WCN"] / "partial_O3" / "B2000WCN_partial_O3_ranking_MarApr_min_60_90N.csv",
    "CLIM3D": OUR_CASES["CLIM3D"]
    / "partial_O3"
    / "B2000WCN007009010011_Clim3D_partial_O3_ranking_MarApr_min_60_90N.csv",
}

YEARS = 200
EXTREME_YEARS = 50
LAT_TARGET = np.arange(55.0, 75.0 + 0.1, 2.5)
MAR_MAY = slice(59, 151)
MAR_APR = slice(59, 120)
MONTH_TICKS = [0, 31, 61, 91]
MONTH_LABELS = ["Mar", "Apr", "May", "Jun"]
PLEV_TICKS = [1000, 100, 10, 1]
O3_PRESSURE_TAG = "30_70hPa"
M_AIR = 28.964 / (6.022e23)
GRAV = 980.6
DU = 2.687e16


@dataclass(frozen=True)
class ExtremeYears:
    low0: np.ndarray
    high0: np.ndarray
    low_values: np.ndarray
    high_values: np.ndarray
    low_days: np.ndarray
    high_days: np.ndarray


def as_float(values) -> np.ndarray:
    arr = np.asarray(values)
    if np.ma.isMaskedArray(arr):
        arr = arr.filled(np.nan)
    arr = arr.astype(float)
    return np.where(np.abs(arr) > 1.0e30, np.nan, arr)


def normalize_plev_to_hpa(plev: np.ndarray) -> np.ndarray:
    plev = np.asarray(plev, dtype=float)
    return plev / 100.0 if np.nanmax(np.abs(plev)) > 1000.0 else plev


def interp_lat(data: np.ndarray, src_lat: np.ndarray, target_lat: np.ndarray) -> np.ndarray:
    src_lat = np.asarray(src_lat, dtype=float)
    target_lat = np.asarray(target_lat, dtype=float)
    if src_lat[0] > src_lat[-1]:
        src_lat = src_lat[::-1]
        data = data[..., ::-1]
    hi = np.searchsorted(src_lat, target_lat, side="left")
    hi = np.clip(hi, 1, len(src_lat) - 1)
    lo = hi - 1
    weight = (target_lat - src_lat[lo]) / (src_lat[hi] - src_lat[lo])
    return data[..., lo] * (1.0 - weight) + data[..., hi] * weight


def lat_weighted_mean_after_interp(data: np.ndarray, lat: np.ndarray) -> np.ndarray:
    data_i = interp_lat(data, lat, LAT_TARGET)
    weights = np.cos(np.deg2rad(LAT_TARGET))
    return np.average(data_i, weights=weights, axis=-1)


def run_ncks_subset(src: Path, dst: Path, levels: tuple[float, float, float]) -> None:
    if dst.exists():
        return
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ncks",
        "-O",
        "-v",
        "O3",
        "-d",
        f"plev,{levels[0]}",
        "-d",
        f"plev,{levels[1]}",
        "-d",
        f"plev,{levels[2]}",
        str(src),
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def pressure_weights_like_marina(plev: np.ndarray) -> np.ndarray:
    p_hpa = normalize_plev_to_hpa(plev)
    if np.nanmax(np.abs(plev)) > 1000.0:
        # Full Marina WACCM pressure grid is in Pa; selected 30/50/70 hPa
        # levels have previous-level deltas 1000/2000/2000 Pa.
        return np.array([1000.0, 2000.0, 2000.0])
    # Full Marina INT pressure grid is in hPa; selected 30/50/70 hPa levels
    # have previous-level deltas 10/20/20 hPa, then Marina converts to Pa.
    if not np.allclose(p_hpa, [30.0, 50.0, 70.0]):
        raise ValueError(f"Unexpected selected pressure levels: {p_hpa}")
    return np.array([10.0, 20.0, 20.0]) * 100.0


def marina_o3_extremes(subset_path: Path) -> ExtremeYears:
    with Dataset(subset_path) as ds:
        lat = as_float(ds.variables["lat"][:])
        plev = as_float(ds.variables["plev"][:])
        o3 = as_float(ds.variables["O3"][:])
    if o3.ndim == 4:
        o3 = o3[..., 0]

    o3 = o3.reshape(YEARS, 365, 3, o3.shape[-1])[:, MAR_APR, :, :]
    o3 = interp_lat(o3, lat, np.linspace(60.0, 90.0, 13))
    polar_weights = np.cos(np.deg2rad(np.linspace(60.0, 90.0, 13)))
    o3_polar = np.average(o3, weights=polar_weights, axis=-1)
    weights_p = pressure_weights_like_marina(plev)
    partial_col = np.sum(o3_polar * weights_p[None, None, :] * 10.0 / (M_AIR * GRAV) / DU, axis=-1)

    # Marina's FW-truncation branch effectively uses full Mar-Apr for these
    # WACCM files in FW_atmos.ipynb; this reproduces the notebook's printed
    # mean low/high O3 days exactly.
    low_values = np.nanmin(partial_col, axis=1).astype(int)
    high_values = np.nanmax(partial_col, axis=1).astype(int)
    low_days = np.nanargmin(partial_col, axis=1)
    high_days = np.nanargmax(partial_col, axis=1)
    low0 = np.argsort(low_values)[:EXTREME_YEARS]
    high0 = np.argsort(high_values)[-EXTREME_YEARS:][::-1]
    return ExtremeYears(low0, high0, low_values, high_values, low_days, high_days)


def marina_u_composite(path: Path, years0: np.ndarray) -> xr.DataArray:
    pieces = []
    with Dataset(path) as ds:
        lat = as_float(ds.variables["lat"][:])
        plev_hpa = normalize_plev_to_hpa(as_float(ds.variables["plev"][:]))
        var = ds.variables["U"]
        for year0 in np.asarray(years0, dtype=int):
            arr = as_float(var[year0 * 365 + MAR_MAY.start : year0 * 365 + MAR_MAY.stop, :, ...])
            if arr.ndim == 4:
                arr = arr[..., 0]
            pieces.append(lat_weighted_mean_after_interp(arr, lat))
    comp = np.nanmean(np.stack(pieces, axis=0), axis=0)
    return xr.DataArray(
        comp.T,
        dims=("plev", "plot_day"),
        coords={"plev": plev_hpa, "plot_day": np.arange(MAR_MAY.stop - MAR_MAY.start)},
    )


def sample_flag(sample: str) -> str:
    method = "rm5" if sample.endswith("_rm5") else "raw"
    if sample.startswith("low"):
        return f"is_lowest25pct_{method}"
    if sample.startswith("high"):
        return f"is_highest25pct_{method}"
    raise ValueError(sample)


def our_sample_years(case: str, sample: str) -> list[int]:
    df = pd.read_csv(OUR_RANKING[case])
    if "pressure_range" in df.columns:
        df = df[df["pressure_range"].astype(str) == O3_PRESSURE_TAG]
    flag = sample_flag(sample)
    return sorted(df.loc[df[flag].astype(bool), "year"].astype(int).unique().tolist())


def our_sample_count(case: str, sample: str) -> int:
    return len(our_sample_years(case, sample))


def our_u_case(case: str, sample: str) -> xr.DataArray:
    path = OUR_CASES[case] / "climatology" / "U_climatology_plev_doy.nc"
    with xr.open_dataset(path, decode_times=False) as ds:
        da = ds[f"U_clim_{sample}"]
        plev_hpa = normalize_plev_to_hpa(da["plev"].values)
        da = da.assign_coords(plev=plev_hpa)
        if "lon" in da.dims:
            da = da.mean("lon", skipna=True)
        lat_values = da["lat"].values
        da = da.sel(lat=slice(55.0, 75.0) if lat_values[0] <= lat_values[-1] else slice(75.0, 55.0))
        da = da.weighted(np.cos(np.deg2rad(da["lat"]))).mean("lat")
        da = da.isel(doy=MAR_MAY).rename({"doy": "plot_day"}).load()
    da = da.assign_coords(plot_day=np.arange(MAR_MAY.stop - MAR_MAY.start))
    return da.transpose("plev", "plot_day")


def our_int3d_u(sample: str) -> xr.DataArray:
    bwcn = our_u_case("BWCN", sample)
    b2000 = our_u_case("B2000WCN", sample)
    bwcn, b2000 = xr.align(bwcn, b2000, join="inner")
    w_bwcn = our_sample_count("BWCN", sample)
    w_b2000 = our_sample_count("B2000WCN", sample)
    return (bwcn * w_bwcn + b2000 * w_b2000) / float(w_bwcn + w_b2000)


def our_clim3d_u(sample: str) -> xr.DataArray:
    return our_u_case("CLIM3D", sample)


def format_axis(ax, show_ylabel: bool = False, show_xlabel: bool = False) -> None:
    ax.set_ylim(np.log(1000.0), np.log(1.0))
    ax.set_yticks(np.log(PLEV_TICKS))
    ax.set_yticklabels([str(v) for v in PLEV_TICKS])
    ax.set_xticks(MONTH_TICKS)
    ax.set_xticklabels(MONTH_LABELS)
    if show_ylabel:
        ax.set_ylabel("pressure (hPa)")
    if show_xlabel:
        ax.set_xlabel("")


def panel(ax, da: xr.DataArray, title: str, levels: np.ndarray):
    x = da["plot_day"].values
    p = da["plev"].values
    z = da.values
    cf = ax.contourf(x, np.log(p), z, levels=levels, cmap="RdBu_r", extend="both")
    ax.contour(x, np.log(p), z, levels=[0.0], colors="0.35", linewidths=0.7)
    ax.set_title(title, fontsize=10)
    format_axis(ax)
    return cf


def aligned_diff(a: xr.DataArray, b: xr.DataArray) -> xr.DataArray:
    a, b = xr.align(a, b, join="inner")
    return a - b


def stat_rows(name: str, da: xr.DataArray) -> list[dict[str, float | str]]:
    rows = []
    plev_mask = (da["plev"].values[:, None] >= 30.0) & (da["plev"].values[:, None] <= 70.0)
    masks = {
        "full_MarMay_1-1000hPa": np.ones_like(da.values, dtype=bool),
        "MarMay_30-70hPa": np.broadcast_to(plev_mask, da.values.shape),
        "Apr1-May15_50hPa": np.zeros_like(da.values, dtype=bool),
    }
    lev50 = int(np.nanargmin(np.abs(da["plev"].values - 50.0)))
    masks["Apr1-May15_50hPa"][lev50, 31:76] = True
    for region, mask in masks.items():
        vals = da.values[mask]
        rows.append(
            {
                "case": name,
                "region": region,
                "mean": float(np.nanmean(vals)),
                "mae": float(np.nanmean(np.abs(vals))),
                "max_abs": float(np.nanmax(np.abs(vals))),
                "p05": float(np.nanpercentile(vals, 5)),
                "p95": float(np.nanpercentile(vals, 95)),
            }
        )
    return rows


def marina_clim3d_physical_tuple(marina_year: int) -> tuple[str, int]:
    if 1 <= marina_year <= 52:
        return ("007", marina_year + 4)
    if 53 <= marina_year <= 100:
        return ("010", marina_year - 53 + 57)
    if 101 <= marina_year <= 152:
        return ("009", marina_year - 101 + 5)
    if 153 <= marina_year <= 200:
        return ("011", marina_year - 153 + 57)
    raise ValueError(marina_year)


def our_clim3d_physical_tuple(our_year: int) -> tuple[str, int]:
    if 1 <= our_year <= 57:
        return ("007", our_year)
    if 58 <= our_year <= 113:
        return ("009", our_year - 58 + 1)
    if 114 <= our_year <= 162:
        return ("010", our_year - 114 + 57)
    if 163 <= our_year <= 216:
        return ("011", our_year - 163 + 57)
    raise ValueError(our_year)


def marina_index_from_physical(case: str, source_year: int) -> int | None:
    if case == "007" and 5 <= source_year <= 56:
        return source_year - 4
    if case == "010" and 57 <= source_year <= 104:
        return 53 + source_year - 57
    if case == "009" and 5 <= source_year <= 56:
        return 101 + source_year - 5
    if case == "011" and 57 <= source_year <= 104:
        return 153 + source_year - 57
    return None


def write_mapping_debug(marina_ext: ExtremeYears) -> None:
    rows = []
    marina_low = set((marina_ext.low0 + 1).astype(int))
    marina_high = set((marina_ext.high0 + 1).astype(int))
    for sample in ["low25_rm5", "high25_rm5"]:
        selected = set(our_sample_years("CLIM3D", sample))
        for our_year in sorted(selected):
            case, source_year = our_clim3d_physical_tuple(our_year)
            marina_year = marina_index_from_physical(case, source_year)
            rows.append(
                {
                    "sample": sample,
                    "our_timefixed_year": our_year,
                    "physical_case": case,
                    "physical_source_year": source_year,
                    "marina_combined_year": marina_year,
                    "exists_in_marina_200yr": marina_year is not None,
                    "also_marina_low25": marina_year in marina_low if marina_year is not None else False,
                    "also_marina_high25": marina_year in marina_high if marina_year is not None else False,
                }
            )
    pd.DataFrame(rows).to_csv(PLOT_DIR / "clim3d_timefixed_to_marina_year_mapping.csv", index=False)


def mapping_summary() -> pd.DataFrame:
    df = pd.read_csv(PLOT_DIR / "clim3d_timefixed_to_marina_year_mapping.csv")
    rows = []
    for sample, group in df.groupby("sample", sort=True):
        rows.append(
            {
                "sample": sample,
                "our_selected_years": int(len(group)),
                "mapped_to_marina_200yr": int(group["exists_in_marina_200yr"].sum()),
                "mapped_overlap_marina_low25": int(group["also_marina_low25"].sum()),
                "mapped_overlap_marina_high25": int(group["also_marina_high25"].sum()),
            }
        )
    return pd.DataFrame(rows)


def source_order_rows() -> pd.DataFrame:
    rows = []
    for marina_year in range(1, 201):
        case, source_year = marina_clim3d_physical_tuple(marina_year)
        rows.append({"dataset": "Marina combined", "combined_year": marina_year, "physical_case": case, "source_year": source_year})
    for our_year in range(1, 217):
        case, source_year = our_clim3d_physical_tuple(our_year)
        rows.append({"dataset": "timefixed", "combined_year": our_year, "physical_case": case, "source_year": source_year})
    return pd.DataFrame(rows)


def verify_required_paths() -> None:
    paths = [MARINA_INT_O3, MARINA_INT_U, MARINA_CLIM3D_O3, MARINA_CLIM3D_U]
    paths += [OUR_CASES[c] / "climatology" / "U_climatology_plev_doy.nc" for c in OUR_CASES]
    paths += list(OUR_RANKING.values())
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("\n".join(missing))


def main() -> None:
    verify_required_paths()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    int_o3_subset = TMP_DIR / "marina_int_o3_30_50_70.nc"
    clim_o3_subset = TMP_DIR / "marina_clim3d_o3_30_50_70.nc"
    run_ncks_subset(MARINA_INT_O3, int_o3_subset, (30.0, 50.0, 70.0))
    run_ncks_subset(MARINA_CLIM3D_O3, clim_o3_subset, (3000.0, 5000.0, 7000.0))

    marina_int_ext = marina_o3_extremes(int_o3_subset)
    marina_clim_ext = marina_o3_extremes(clim_o3_subset)
    write_mapping_debug(marina_clim_ext)
    source_order_rows().to_csv(PLOT_DIR / "clim3d_source_order_marina_vs_timefixed.csv", index=False)

    # Marina reproduction.
    marina_int_low = marina_u_composite(MARINA_INT_U, marina_int_ext.low0)
    marina_clim_low = marina_u_composite(MARINA_CLIM3D_U, marina_clim_ext.low0)
    marina_int_high = marina_u_composite(MARINA_INT_U, marina_int_ext.high0)
    marina_clim_high = marina_u_composite(MARINA_CLIM3D_U, marina_clim_ext.high0)
    marina_low_diff = aligned_diff(marina_int_low, marina_clim_low)
    marina_high_diff = aligned_diff(marina_int_high, marina_clim_high)

    # Current notebook inputs.
    our_int_low = our_int3d_u("low25_rm5")
    our_clim_low = our_clim3d_u("low25_rm5")
    our_int_high = our_int3d_u("high25_rm5")
    our_clim_high = our_clim3d_u("high25_rm5")
    our_low_diff = aligned_diff(our_int_low, our_clim_low)
    our_high_diff = aligned_diff(our_int_high, our_clim_high)

    low_net = aligned_diff(our_low_diff, marina_low_diff)
    high_net = aligned_diff(our_high_diff, marina_high_diff)
    low_int_contrib = aligned_diff(our_int_low, marina_int_low)
    low_clim_contrib = aligned_diff(our_clim_low, marina_clim_low)
    high_int_contrib = aligned_diff(our_int_high, marina_int_high)
    high_clim_contrib = aligned_diff(our_clim_high, marina_clim_high)

    stats = []
    for name, da in [
        ("Marina low INT-CLIM3D", marina_low_diff),
        ("Our low INT-CLIM3D", our_low_diff),
        ("Our-Marina low net", low_net),
        ("Low INT contribution: our INT - Marina INT", low_int_contrib),
        ("Low CLIM contribution: our CLIM - Marina CLIM", low_clim_contrib),
        ("Marina high INT-CLIM3D", marina_high_diff),
        ("Our high INT-CLIM3D", our_high_diff),
        ("Our-Marina high net", high_net),
        ("High INT contribution: our INT - Marina INT", high_int_contrib),
        ("High CLIM contribution: our CLIM - Marina CLIM", high_clim_contrib),
    ]:
        stats.extend(stat_rows(name, da))
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(PLOT_DIR / "figure9_wind_difference_debug_stats.csv", index=False)

    year_rows = []
    for label, ext in [("Marina INT", marina_int_ext), ("Marina CLIM3D", marina_clim_ext)]:
        year_rows.append(
            {
                "dataset": label,
                "sample": "low25",
                "years_1based": " ".join(map(str, sorted((ext.low0 + 1).astype(int)))),
                "mean_event_day_from_mar1": float(np.mean(ext.low_days[ext.low0])),
                "std_event_day_from_mar1": float(np.std(ext.low_days[ext.low0])),
            }
        )
        year_rows.append(
            {
                "dataset": label,
                "sample": "high25",
                "years_1based": " ".join(map(str, sorted((ext.high0 + 1).astype(int)))),
                "mean_event_day_from_mar1": float(np.mean(ext.high_days[ext.high0])),
                "std_event_day_from_mar1": float(np.std(ext.high_days[ext.high0])),
            }
        )
    pd.DataFrame(year_rows).to_csv(PLOT_DIR / "marina_reproduced_o3_extreme_years.csv", index=False)
    mapping_summary().to_csv(PLOT_DIR / "clim3d_mapping_overlap_summary.csv", index=False)

    fig, axes = plt.subplots(3, 4, figsize=(15.5, 9.0), constrained_layout=True)
    abs_levels = np.linspace(-10.0, 10.0, 21)
    delta_levels = np.linspace(-6.0, 6.0, 25)
    panels = [
        (marina_low_diff, "A Marina low\nINT-CLIM3D", abs_levels),
        (our_low_diff, "B Our low\nINT-CLIM3D", abs_levels),
        (low_net, "C Our - Marina\nlow net", delta_levels),
        (low_int_contrib, "D low INT term\nour - Marina", delta_levels),
        (-low_clim_contrib, "E low -CLIM term\n-(our - Marina)", delta_levels),
        (marina_high_diff, "F Marina high\nINT-CLIM3D", abs_levels),
        (our_high_diff, "G Our high\nINT-CLIM3D", abs_levels),
        (high_net, "H Our - Marina\nhigh net", delta_levels),
        (high_int_contrib, "I high INT term\nour - Marina", delta_levels),
        (-high_clim_contrib, "J high -CLIM term\n-(our - Marina)", delta_levels),
    ]
    for ax in axes.flat:
        ax.set_visible(False)
    cfs = []
    for ax, (da, title, levels) in zip(axes.flat, panels):
        ax.set_visible(True)
        cfs.append(panel(ax, da, title, levels))
    for ax in axes[:, 0]:
        if ax.get_visible():
            ax.set_ylabel("pressure (hPa)")
    for ax in axes[-1, :]:
        if ax.get_visible():
            ax.set_xlabel("")
    fig.colorbar(cfs[0], ax=axes[:, :2], orientation="horizontal", shrink=0.75, pad=0.04, label="U difference (m s$^{-1}$)")
    fig.colorbar(cfs[2], ax=axes[:, 2:], orientation="horizontal", shrink=0.75, pad=0.04, label="diagnostic difference (m s$^{-1}$)")
    fig.suptitle("Figure 9 wind-difference diagnostic: Marina reproduction vs current timefixed inputs", fontsize=13)
    fig.savefig(PLOT_DIR / "figure9_wind_difference_decomposition.png", dpi=220)
    fig.savefig(PLOT_DIR / "figure9_wind_difference_decomposition.svg")
    plt.close(fig)

    focus = stats_df[stats_df["region"].eq("Apr1-May15_50hPa")]
    print(focus.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))
    print(f"\nWrote diagnostics to {PLOT_DIR}")


if __name__ == "__main__":
    main()
