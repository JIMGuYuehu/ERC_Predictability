#!/usr/bin/env python
"""Build TEST_FWD plot bundle with corrected CLIM-3D/Marina mapping.

The script is intentionally figure-focused and reads existing diagnostics where
possible.  It writes the report figures to:

    Longrun/Visualization/plots/TEST_FWD_plots

The key correction relative to the older clim3d_marina_repro plots is that the
mapped O3 and FWD comparisons both use the feature/fingerprint year mapping
from fwd_clim3d_feature_mapping_test.py.
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
import xarray as xr
from scipy.stats import ttest_ind

import fwd_clim3d_low25_source_test as src
import fwd_clim3d_marina_report_figures as report_figs


ROOT = Path("/home/weiji/restart_exam/code_cleaned")
PLOT_DIR = ROOT / "Longrun/Visualization/plots/TEST_FWD_plots"
TABLE_DIR = ROOT / "Longrun/date_treatment/clim3d_marina_repro_report"

SOURCE_SUMMARY_CSV = TABLE_DIR / "source_isolation_summary.csv"
FEATURE_FWD50_CSV = ROOT / "Longrun/date_treatment/clim3d_feature_mapping_test/feature_matched_fwd50_by_pair.csv"
FEATURE_FWD_BY_LEVEL_CSV = (
    ROOT / "Longrun/date_treatment/clim3d_feature_mapping_test/feature_matched_fwd_by_level_pair.csv"
)


FWD_BWCN_FWD_NC = Path("/mnt/soclim0/public_data/weiji/BWCN/final_warming_date/BWCN_FWD_plev_year.nc")
FWD_BWCN_PARTIAL_O3_NC = Path(
    "/mnt/soclim0/public_data/weiji/BWCN/partial_O3/BWCN_partial_O3_all_ranges.nc"
)
FWD_INT3D_FWD_NC = Path(
    "/mnt/soclim0/public_data/weiji/B2000WCN001002_timefixed/final_warming_date/"
    "B2000WCN001002_FWD_plev_year.nc"
)
FWD_INT3D_PARTIAL_O3_NC = Path(
    "/mnt/soclim0/public_data/weiji/B2000WCN001002_timefixed/partial_O3/"
    "B2000WCN_partial_O3_all_ranges.nc"
)
FWD_CLIM2D_FWD_NC = Path(
    "/mnt/soclim0/public_data/weiji/B2000WCN_NOCOUPL001002_timefixed/final_warming_date/"
    "B2000WCN_NOCOUPL001002_FWD_plev_year.nc"
)
FWD_CLIM2D_PARTIAL_O3_NC = Path(
    "/mnt/soclim0/public_data/weiji/B2000WCN_NOCOUPL001002_timefixed/partial_O3/"
    "B2000WCN_NOCOUPL_partial_O3_all_ranges.nc"
)
FWD_CLIM3D_OUR_FWD_NC = Path(
    "/mnt/soclim0/public_data/weiji/B2000WCN007009010011_Clim3D_timefixed/final_warming_date/"
    "B2000WCN007009010011_Clim3D_FWD_plev_year.nc"
)
FWD_CLIM3D_OUR_PARTIAL_O3_NC = Path(
    "/mnt/soclim0/public_data/weiji/B2000WCN007009010011_Clim3D_timefixed/partial_O3/"
    "B2000WCN007009010011_Clim3D_partial_O3_all_ranges.nc"
)
MARINA_CLIM3D_RUNMEAN5_O3_NC = Path(
    "/mnt/backup_ETH/Marina/WACCM/NOCHEM_2000_3D/combined/"
    "B2000WCN.NOCOUPL.e122.f19_g16.int.zm.runmean5.nc"
)
MARINA_CLIM3D_U_O3_NC = Path(
    "/mnt/backup_ETH/Marina/WACCM/NOCHEM_2000_3D/combined/"
    "B2000WCN.NOCOUPL.e122.f19_g16.int.zm.nc"
)
MARINA_SAVED_FWD_NPY = src.MARINA_SAVED_FWD_NPY
MARINA_O3_FEATURE_CACHE = ROOT / "Longrun/date_treatment/clim3d_feature_mapping_test/marina_o3col_60_90N_mar_may.npy"


FWD_O3_PRESSURE_RANGE = "30_70hPa"
FWD_ONSET_LEVEL_HPA = 50.0
FWD_O3_MIN_VALID_DAYS = 5
FWD_O3_VALID_MIN_DU = 10.0
FWD_TTEST_EQUAL_VAR = False
FWD_LOW_COLOR = "navy"
FWD_HIGH_COLOR = "firebrick"
FWD_COMMON_LEVELS_HPA = np.array([1, 2, 3, 5, 10, 20, 30, 50], dtype=float)
FWD_O3_WINDOW_SPECS = {
    "MA": {"title": "MA", "start_doy": 60, "end_doy": 120, "needs_fwd": False},
    "M-FWD": {"title": "M-FWD", "start_doy": 60, "end_doy": None, "needs_fwd": True},
}
FWD_O3_WINDOW_ORDER = ["MA", "M-FWD"]
FWD_COMPARE_ROWS = ["CLIM-3D", "CLIM-2D"]
MONTH_ENDS_NOLEAP = np.array([31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365])


def ensure_dirs() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, stem: str) -> None:
    for suffix, kwargs in {
        ".png": {"dpi": 300},
        ".pdf": {},
        ".svg": {},
    }.items():
        fig.savefig(PLOT_DIR / f"{stem}{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def save_current_figure(stem: str) -> None:
    fig = plt.gcf()
    save_figure(fig, stem)


def doy_to_month_day(doy: float) -> str:
    return src.doy_to_month_day(doy)


def date_to_doy_no_leap(date_values: np.ndarray) -> np.ndarray:
    date_values = np.asarray(date_values, dtype=np.int64)
    mmdd = date_values % 10000
    month = (mmdd // 100).astype(np.int16)
    day = (mmdd % 100).astype(np.int16)
    doy = np.full(date_values.shape, -9999, dtype=np.int16)
    for m in range(1, 13):
        mask = month == m
        if np.any(mask):
            prev = int(MONTH_ENDS_NOLEAP[m - 2]) if m > 1 else 0
            doy[mask] = prev + day[mask]
    return doy


def calendar_from_ds(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    if "date" in ds:
        dates = ds["date"].values.astype(np.int64)
        years = (dates // 10000).astype(np.int32)
        doys = date_to_doy_no_leap(dates)
        return years.astype(int), doys.astype(int)
    day_index = np.asarray(ds["time"].values, dtype=float).astype(np.int64)
    years = (day_index // 365 + 1).astype(np.int32)
    doys = (day_index % 365 + 1).astype(np.int16)
    return years.astype(int), doys.astype(int)


def fwd_select_common_levels(fwd: xr.DataArray) -> xr.DataArray:
    p_hpa = fwd["plev_hpa"].values.astype(float)
    keep: list[int] = []
    seen: set[float] = set()
    for target in FWD_COMMON_LEVELS_HPA:
        idx = int(np.nanargmin(np.abs(p_hpa - target)))
        key = round(float(p_hpa[idx]), 6)
        if key not in seen:
            keep.append(idx)
            seen.add(key)
    return fwd.isel(plev=keep)


def load_fwd_for_case(cfg: dict[str, object]) -> xr.DataArray:
    if "marina_saved_fwd_npy" in cfg:
        # Marina's merged CLIM-3D pressure levels are fixed.  Avoid opening the
        # huge NetCDF just to read plev; its global history attribute is very
        # large and can make lightweight tests surprisingly slow.
        p_hpa = np.array([0.1, 0.5, 1, 2, 3, 5, 10, 20, 30, 50], dtype=float)
        arr = np.load(Path(cfg["marina_saved_fwd_npy"])).astype(float)
        if arr.shape[0] == len(p_hpa):
            arr = arr.T
        years = np.arange(1, arr.shape[0] + 1, dtype=int)
        fwd = xr.DataArray(
            arr,
            dims=("year", "plev"),
            coords={"year": years, "plev": np.arange(len(p_hpa)), "plev_hpa": ("plev", p_hpa)},
            name="FWD_dayofyear",
        )
        return fwd_select_common_levels(fwd)

    ds = xr.open_dataset(Path(cfg["fwd_nc"]))
    try:
        fwd = ds["FWD_dayofyear"].load()
        if "plev_hpa" in ds:
            p_hpa = ds["plev_hpa"].values.astype(float)
        else:
            p_hpa = ds["plev"].values.astype(float)
            if np.nanmax(np.abs(p_hpa)) > 1000:
                p_hpa = p_hpa / 100.0
        fwd = fwd.assign_coords(plev_hpa=("plev", p_hpa))
        return fwd_select_common_levels(fwd)
    finally:
        ds.close()


def load_partial_o3_series(partial_o3_nc: Path, rolling_days: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ds = xr.open_dataset(partial_o3_nc, decode_times=False)
    try:
        if "O3_partial_60_90N" in ds:
            da = ds["O3_partial_60_90N"].sel(pressure_range=FWD_O3_PRESSURE_RANGE)
        else:
            da = ds[f"O3_partial_60_90N_{FWD_O3_PRESSURE_RANGE}"]
        years, doys = calendar_from_ds(ds)
        da = da.where(da > FWD_O3_VALID_MIN_DU)
        if rolling_days is not None and int(rolling_days) > 1:
            da = da.rolling(time=int(rolling_days), center=True, min_periods=int(rolling_days)).mean()
        vals = da.load().values.astype(float)
        return vals, years, doys
    finally:
        ds.close()


def load_marina_o3_series(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Prefer the compact Marina O3 column cache generated by
    # fwd_clim3d_feature_mapping_test.py.  It contains 200 years x 92 Mar-May
    # days and avoids repeatedly interpolating the large 73,000-day NetCDF.
    if MARINA_O3_FEATURE_CACHE.exists():
        raw = np.load(MARINA_O3_FEATURE_CACHE).astype(float)
        smooth = np.full_like(raw, np.nan, dtype=float)
        for i in range(raw.shape[0]):
            smooth[i, :] = (
                pd.Series(raw[i, :])
                .rolling(5, center=True, min_periods=5)
                .mean()
                .to_numpy(dtype=float)
            )
        years = np.repeat(np.arange(1, raw.shape[0] + 1, dtype=int), raw.shape[1])
        doys = np.tile(np.arange(60, 60 + raw.shape[1], dtype=int), raw.shape[0])
        return smooth.reshape(-1), years, doys

    ds = xr.open_dataset(path, decode_times=False)
    try:
        plev = ds["plev"]
        p_values = plev.values.astype(float)
        delta_p = np.zeros(len(p_values), dtype=float)
        delta_p[1:] = np.diff(p_values)

        o3 = ds["O3"].sel(plev=slice(3000.0, 7000.0))
        o3 = o3.interp(lat=np.linspace(-90.0, 90.0, 73)).sel(lat=slice(60.0, 90.0))
        weights_lat = np.cos(np.deg2rad(o3["lat"]))
        o3 = o3.weighted(weights_lat).mean(dim="lat")

        weights_p = xr.DataArray(delta_p, dims=["plev"], coords={"plev": plev})
        o3 = o3 * weights_p.sel(plev=o3["plev"]) * 10.0 / (980.6 * (28.964 / 6.022e23))
        o3 = o3.sum(dim="plev") / 2.687e16
        if "lon" in o3.dims:
            o3 = o3.mean("lon", skipna=True)
        vals = o3.where(o3 > FWD_O3_VALID_MIN_DU).load().values.astype(float)
    finally:
        ds.close()
    day_index = np.arange(vals.shape[0], dtype=np.int64)
    years = (day_index // 365 + 1).astype(int)
    doys = (day_index % 365 + 1).astype(int)
    return vals, years, doys


def o3_series_for_case(label: str, cfg: dict[str, object], rolling_days: int | None, cache: dict) -> tuple:
    key = (label, cfg.get("partial_o3_nc"), cfg.get("marina_runmean5_o3_nc"), rolling_days)
    if key in cache:
        return cache[key]
    if "marina_runmean5_o3_nc" in cfg:
        out = (*load_marina_o3_series(Path(cfg["marina_runmean5_o3_nc"])), "Marina CLIM-3D runmean5 O3")
    else:
        out = (
            *load_partial_o3_series(Path(cfg["partial_o3_nc"]), rolling_days=rolling_days),
            cfg.get("ranking_family", "partial_O3"),
        )
    cache[key] = out
    return out


def onset_series_for_years(fwd: xr.DataArray, years: list[int]) -> pd.Series:
    p_hpa = fwd["plev_hpa"].values.astype(float)
    idx = int(np.nanargmin(np.abs(p_hpa - FWD_ONSET_LEVEL_HPA)))
    selected = fwd.isel(plev=idx).sel(year=list(years))
    return pd.Series(selected.values.astype(float), index=np.asarray(years, dtype=int)).dropna()


def load_low_high_from_series(
    o3_values: np.ndarray,
    years_by_time: np.ndarray,
    doys_by_time: np.ndarray,
    window_mode: str,
    candidate_years: list[int],
    onset_by_year: pd.Series | None = None,
) -> tuple[list[int], list[int], pd.DataFrame]:
    spec = FWD_O3_WINDOW_SPECS[window_mode]
    available_years = set(np.asarray(years_by_time, dtype=int).tolist())
    candidate_years = sorted(int(y) for y in candidate_years if int(y) in available_years)
    records = []
    for year in candidate_years:
        start_doy = int(spec["start_doy"])
        if spec["needs_fwd"]:
            if onset_by_year is None or year not in onset_by_year.index:
                continue
            onset_doy = float(onset_by_year.loc[year])
            if not np.isfinite(onset_doy) or onset_doy < start_doy:
                continue
            end_doy = int(round(onset_doy))
        else:
            onset_doy = np.nan
            end_doy = int(spec["end_doy"])
        mask = (years_by_time == int(year)) & (doys_by_time >= start_doy) & (doys_by_time <= end_doy)
        vals = np.asarray(o3_values[mask], dtype=float)
        vals = vals[np.isfinite(vals) & (vals > FWD_O3_VALID_MIN_DU)]
        if vals.size < FWD_O3_MIN_VALID_DAYS:
            continue
        records.append(
            {
                "year": int(year),
                "window_min_DU": float(np.min(vals)),
                "window_max_DU": float(np.max(vals)),
                "window_mean_DU": float(np.mean(vals)),
                "n_valid_o3_days": int(vals.size),
                "window_start_doy": start_doy,
                "window_end_doy": end_doy,
                "fsw_onset_doy": onset_doy,
            }
        )
    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError(f"No valid O3 windows for {window_mode}")
    n25 = max(int(np.floor(0.25 * len(df))), 1)
    low_years = sorted(df.nsmallest(n25, "window_min_DU")["year"].astype(int).tolist())
    high_years = sorted(df.nlargest(n25, "window_max_DU")["year"].astype(int).tolist())
    df["is_low_o3_spring"] = df["year"].isin(low_years)
    df["is_high_o3_spring"] = df["year"].isin(high_years)
    return low_years, high_years, df.sort_values("year").reset_index(drop=True)


def fwd_values_from_rows(rows: pd.DataFrame, fwd_by_component: dict[str, xr.DataArray]) -> tuple[np.ndarray, list[str]]:
    rows = rows.sort_values(["component", "year"]).reset_index(drop=True)
    pieces = []
    labels = []
    for component, sub in rows.groupby("component", sort=False):
        years = sub["year"].astype(int).tolist()
        if not years:
            continue
        da = fwd_by_component[component].sel(year=years).transpose("year", "plev")
        pieces.append(da.values.astype(float))
        labels.extend([f"{component}:{year}" for year in years])
    if not pieces:
        nlev = next(iter(fwd_by_component.values())).sizes["plev"]
        return np.empty((0, nlev), dtype=float), labels
    return np.concatenate(pieces, axis=0), labels


def profile_from_value_arrays(
    label: str,
    cfg: dict[str, object],
    window_mode: str,
    rolling_days: int | None,
    ranking_source: str,
    p_hpa: np.ndarray,
    all_values: np.ndarray,
    low_values: np.ndarray,
    high_values: np.ndarray,
    rank_df: pd.DataFrame,
    all_years: list[str] | list[int],
    low_years: list[str] | list[int],
    high_years: list[str] | list[int],
) -> dict[str, object]:
    all_mean = np.nanmean(all_values, axis=0)
    low_dev_values = low_values - all_mean[None, :]
    high_dev_values = high_values - all_mean[None, :]
    return {
        "label": label,
        "window_mode": window_mode,
        "rolling_days": rolling_days,
        "ranking_source": ranking_source,
        "p_hpa": p_hpa,
        "all_mean": all_mean.astype(float),
        "all_std": np.nanstd(all_values, axis=0).astype(float),
        "low_mean_abs": np.nanmean(low_values, axis=0).astype(float),
        "high_mean_abs": np.nanmean(high_values, axis=0).astype(float),
        "low_mean_dev": np.nanmean(low_dev_values, axis=0).astype(float),
        "high_mean_dev": np.nanmean(high_dev_values, axis=0).astype(float),
        "low_values": low_values.astype(float),
        "high_values": high_values.astype(float),
        "low_dev_values": low_dev_values.astype(float),
        "high_dev_values": high_dev_values.astype(float),
        "rank_df": rank_df,
        "all_years": all_years,
        "low_years": low_years,
        "high_years": high_years,
        "linestyle": cfg["linestyle"],
        "shade": cfg["shade"],
    }


def load_single_profile(label: str, cfg: dict[str, object], window_mode: str, rolling_days: int | None, cache: dict) -> dict:
    fwd = load_fwd_for_case(cfg)
    p_hpa = fwd["plev_hpa"].values.astype(float)
    available_fwd_years = sorted(fwd["year"].values.astype(int).tolist())
    onset_by_year = onset_series_for_years(fwd, available_fwd_years)
    o3_values, years_by_time, doys_by_time, ranking_source = o3_series_for_case(label, cfg, rolling_days, cache)
    low_years, high_years, rank_df = load_low_high_from_series(
        o3_values,
        years_by_time,
        doys_by_time,
        window_mode=window_mode,
        candidate_years=available_fwd_years,
        onset_by_year=onset_by_year,
    )
    all_years = rank_df["year"].astype(int).tolist()
    all_values = fwd.sel(year=all_years).transpose("year", "plev").values.astype(float)
    low_values = fwd.sel(year=low_years).transpose("year", "plev").values.astype(float)
    high_values = fwd.sel(year=high_years).transpose("year", "plev").values.astype(float)
    return profile_from_value_arrays(
        label,
        cfg,
        window_mode,
        rolling_days,
        ranking_source,
        p_hpa,
        all_values,
        low_values,
        high_values,
        rank_df,
        all_years,
        low_years,
        high_years,
    )


def load_combined_profile(label: str, cfg: dict[str, object], window_mode: str, rolling_days: int | None, cache: dict) -> dict:
    records = []
    fwd_by_component = {}
    p_hpa_ref = None
    ranking_sources = []

    for component_cfg in cfg["components"]:
        component = component_cfg["name"]
        merged_cfg = {**component_cfg, "linestyle": cfg["linestyle"], "shade": cfg["shade"]}
        fwd = load_fwd_for_case(merged_cfg)
        p_hpa = fwd["plev_hpa"].values.astype(float)
        if p_hpa_ref is None:
            p_hpa_ref = p_hpa
        elif not np.allclose(p_hpa_ref, p_hpa, rtol=0, atol=1e-6):
            raise ValueError(f"{label}: FWD pressure levels differ for {component}")

        available_fwd_years = sorted(fwd["year"].values.astype(int).tolist())
        onset_by_year = onset_series_for_years(fwd, available_fwd_years)
        o3_values, years_by_time, doys_by_time, ranking_source = o3_series_for_case(
            component, merged_cfg, rolling_days, cache
        )
        _, _, rank_df = load_low_high_from_series(
            o3_values,
            years_by_time,
            doys_by_time,
            window_mode=window_mode,
            candidate_years=available_fwd_years,
            onset_by_year=onset_by_year,
        )
        rank_df = rank_df.copy()
        rank_df["component"] = component
        records.append(rank_df)
        fwd_by_component[component] = fwd
        ranking_sources.append(ranking_source)

    rank_all = pd.concat(records, ignore_index=True)
    n25 = max(int(np.floor(0.25 * len(rank_all))), 1)
    low_keys = set(
        zip(
            rank_all.nsmallest(n25, "window_min_DU")["component"].astype(str),
            rank_all.nsmallest(n25, "window_min_DU")["year"].astype(int),
        )
    )
    high_keys = set(
        zip(
            rank_all.nlargest(n25, "window_max_DU")["component"].astype(str),
            rank_all.nlargest(n25, "window_max_DU")["year"].astype(int),
        )
    )
    rank_all["is_low_o3_spring"] = [
        (str(component), int(year)) in low_keys
        for component, year in zip(rank_all["component"], rank_all["year"])
    ]
    rank_all["is_high_o3_spring"] = [
        (str(component), int(year)) in high_keys
        for component, year in zip(rank_all["component"], rank_all["year"])
    ]
    rank_all = rank_all.sort_values(["component", "year"]).reset_index(drop=True)

    all_values, all_years = fwd_values_from_rows(rank_all, fwd_by_component)
    low_values, low_years = fwd_values_from_rows(rank_all[rank_all["is_low_o3_spring"]], fwd_by_component)
    high_values, high_years = fwd_values_from_rows(rank_all[rank_all["is_high_o3_spring"]], fwd_by_component)
    return profile_from_value_arrays(
        label,
        cfg,
        window_mode,
        rolling_days,
        " + ".join(ranking_sources),
        p_hpa_ref,
        all_values,
        low_values,
        high_values,
        rank_all,
        all_years,
        low_years,
        high_years,
    )


def load_profile(label: str, cfg: dict[str, object], window_mode: str, rolling_days: int | None, cache: dict) -> dict:
    if "components" in cfg:
        return load_combined_profile(label, cfg, window_mode, rolling_days, cache)
    return load_single_profile(label, cfg, window_mode, rolling_days, cache)


def figure2_cases(use_marina_clim3d: bool = False) -> dict[str, dict[str, object]]:
    clim3d_cfg: dict[str, object]
    if use_marina_clim3d:
        clim3d_cfg = {
            "marina_saved_fwd_npy": MARINA_SAVED_FWD_NPY,
            "marina_runmean5_o3_nc": MARINA_CLIM3D_RUNMEAN5_O3_NC,
            "linestyle": ":",
            "shade": False,
            "ranking_family": "Marina CLIM-3D runmean5 O3",
        }
    else:
        clim3d_cfg = {
            "fwd_nc": FWD_CLIM3D_OUR_FWD_NC,
            "partial_o3_nc": FWD_CLIM3D_OUR_PARTIAL_O3_NC,
            "linestyle": ":",
            "shade": False,
            "ranking_family": "our CLIM-3D partial_O3",
        }

    return {
        "INT-3D": {
            "components": [
                {
                    "name": "BWCN",
                    "fwd_nc": FWD_BWCN_FWD_NC,
                    "partial_o3_nc": FWD_BWCN_PARTIAL_O3_NC,
                    "ranking_family": "BWCN partial_O3",
                },
                {
                    "name": "B2000WCN",
                    "fwd_nc": FWD_INT3D_FWD_NC,
                    "partial_o3_nc": FWD_INT3D_PARTIAL_O3_NC,
                    "ranking_family": "B2000WCN partial_O3",
                },
            ],
            "linestyle": "-",
            "shade": True,
            "ranking_family": "BWCN + B2000WCN partial_O3",
        },
        "CLIM-3D": clim3d_cfg,
        "CLIM-2D": {
            "fwd_nc": FWD_CLIM2D_FWD_NC,
            "partial_o3_nc": FWD_CLIM2D_PARTIAL_O3_NC,
            "linestyle": ":",
            "shade": False,
            "ranking_family": "our CLIM-2D partial_O3",
        },
    }


def load_profiles_by_window(rolling_days: int | None = None, use_marina_clim3d: bool = False) -> dict:
    cache: dict = {}
    cases = figure2_cases(use_marina_clim3d=use_marina_clim3d)
    profiles: dict[str, dict[str, dict]] = {}
    for window_mode in FWD_O3_WINDOW_ORDER:
        profiles[window_mode] = {}
        for label, cfg in cases.items():
            profiles[window_mode][label] = load_profile(label, cfg, window_mode, rolling_days, cache)
    return profiles


def ttest_mask(values_a: np.ndarray, values_b: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    nlev = values_a.shape[1]
    mask = np.zeros(nlev, dtype=bool)
    pvals = np.full(nlev, np.nan, dtype=float)
    for i in range(nlev):
        a = values_a[:, i]
        b = values_b[:, i]
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if len(a) < 2 or len(b) < 2:
            continue
        _, p = ttest_ind(a, b, equal_var=FWD_TTEST_EQUAL_VAR, nan_policy="omit")
        pvals[i] = p
        mask[i] = np.isfinite(p) and p < alpha
    return mask, pvals


def response_xmax(profiles_by_window: dict) -> float:
    vals = []
    for window_profiles in profiles_by_window.values():
        for prof in window_profiles.values():
            vals.extend(prof["low_mean_dev"][np.isfinite(prof["low_mean_dev"])])
            vals.extend(prof["high_mean_dev"][np.isfinite(prof["high_mean_dev"])])
    xmax = max(10.0, np.ceil((np.nanmax(np.abs(vals)) + 3.0) / 5.0) * 5.0)
    return min(max(xmax, 25.0), 40.0)


def draw_response_panel(
    ax: plt.Axes,
    profiles: dict[str, dict],
    compare_label: str,
    panel_label: str,
    title: str,
    xmax: float,
    show_ylabel: bool = False,
    show_legend: bool = False,
) -> None:
    int3d = profiles["INT-3D"]
    comp = profiles[compare_label]
    p_hpa = int3d["p_hpa"]
    if int3d["shade"]:
        ax.fill_betweenx(p_hpa, -int3d["all_std"], int3d["all_std"], color="0.85", alpha=0.40, lw=0)
    ax.plot(int3d["low_mean_dev"], p_hpa, color=FWD_LOW_COLOR, lw=2.0, ls=int3d["linestyle"], label="INT-3D low O3")
    ax.plot(int3d["high_mean_dev"], p_hpa, color=FWD_HIGH_COLOR, lw=2.0, ls=int3d["linestyle"], label="INT-3D high O3")
    ax.plot(comp["low_mean_dev"], comp["p_hpa"], color=FWD_LOW_COLOR, lw=2.0, ls=comp["linestyle"], label=f"{compare_label} low O3")
    ax.plot(comp["high_mean_dev"], comp["p_hpa"], color=FWD_HIGH_COLOR, lw=2.0, ls=comp["linestyle"], label=f"{compare_label} high O3")

    low_sig, _ = ttest_mask(int3d["low_dev_values"], comp["low_dev_values"])
    high_sig, _ = ttest_mask(int3d["high_dev_values"], comp["high_dev_values"])
    ax.scatter(comp["low_mean_dev"][low_sig], comp["p_hpa"][low_sig], s=48, facecolors="none", edgecolors=FWD_LOW_COLOR, lw=1.1, zorder=6)
    ax.scatter(comp["high_mean_dev"][high_sig], comp["p_hpa"][high_sig], s=48, facecolors="none", edgecolors=FWD_HIGH_COLOR, lw=1.1, zorder=6)

    ax.axvline(0, color="k", lw=1.0)
    ax.set_xlim(-xmax, xmax)
    ax.set_yscale("log")
    ax.set_ylim(55.0, 0.8)
    ax.set_yticks([1, 5, 10, 50])
    ax.set_yticklabels(["1", "5", "10", "50"])
    ax.set_title(title, fontsize=12, pad=8)
    ax.text(0.04, 0.95, panel_label, transform=ax.transAxes, fontsize=12, va="top")
    ax.tick_params(axis="both", labelsize=10, width=1.1, length=5)
    if show_ylabel:
        ax.set_ylabel("altitude (hPa)", fontsize=11)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
    if show_legend:
        ax.legend(loc="center left", fontsize=7.8, frameon=True, handlelength=2.6)


def plot_waccm_four_panel_group(profiles_by_window: dict, figure_title: str) -> plt.Figure:
    xmax = response_xmax(profiles_by_window)
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 8.0), sharex=True, sharey=True)
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]
    k = 0
    for r, compare_label in enumerate(FWD_COMPARE_ROWS):
        for c, window_mode in enumerate(FWD_O3_WINDOW_ORDER):
            draw_response_panel(
                axes[r, c],
                profiles_by_window[window_mode],
                compare_label=compare_label,
                panel_label=panel_labels[k],
                title=f"INT-3D vs {compare_label} | {window_mode}",
                xmax=xmax,
                show_ylabel=(c == 0),
                show_legend=(r == 0 and c == 0),
            )
            k += 1
    fig.suptitle(figure_title, fontsize=14, y=0.995)
    fig.supxlabel("Deviation from mean FSW date (days)", fontsize=12, y=0.02)
    fig.tight_layout(rect=(0, 0.04, 1, 0.965))
    return fig


def our_rm5_metric() -> pd.DataFrame:
    df = pd.read_csv(src.OUR_RANKING_CSV)
    df = df[df["pressure_range"].astype(str).eq(FWD_O3_PRESSURE_RANGE)].copy()
    return pd.DataFrame(
        {
            "year": pd.to_numeric(df["year"], errors="coerce").astype("Int64"),
            "our_o3_rm5_min_DU": pd.to_numeric(df["marapr_min_DU_rm5"], errors="coerce"),
        }
    ).dropna().astype({"year": int})


def marina_rm5_metric() -> pd.DataFrame:
    vals, years, doys = load_marina_o3_series(MARINA_CLIM3D_RUNMEAN5_O3_NC)
    records = []
    for year in np.unique(years):
        mask = (years == int(year)) & (doys >= src.MA_START_DOY) & (doys <= src.MA_END_DOY)
        sub = vals[mask]
        sub = sub[np.isfinite(sub) & (sub > FWD_O3_VALID_MIN_DU)]
        if len(sub) >= src.MIN_VALID_DAYS:
            records.append({"year": int(year), "marina_o3_rm5_min_DU": float(np.min(sub))})
    return pd.DataFrame(records)


def build_feature_matched_rm5_pair_table() -> pd.DataFrame:
    mapping = report_figs.load_feature_matched_fwd_pair_table()
    our = our_rm5_metric()
    marina = marina_rm5_metric()
    pair_df = mapping.merge(our, left_on="our_year", right_on="year", how="left").drop(columns=["year"])
    pair_df = pair_df.merge(marina, left_on="marina_year", right_on="year", how="left").drop(columns=["year"])
    valid = pair_df.dropna(subset=["our_o3_rm5_min_DU", "marina_o3_rm5_min_DU"]).copy()
    n_low = max(int(np.floor(0.25 * len(valid))), 1)
    our_low = set(valid.nsmallest(n_low, "our_o3_rm5_min_DU")["pair_id"].astype(int))
    marina_low = set(valid.nsmallest(n_low, "marina_o3_rm5_min_DU")["pair_id"].astype(int))

    def membership(pair_id: int) -> str:
        in_our = pair_id in our_low
        in_marina = pair_id in marina_low
        if in_our and in_marina:
            return "both LOW25"
        if in_marina:
            return "Marina-only LOW25"
        if in_our:
            return "Our-only LOW25"
        return "not LOW25"

    pair_df["rm5_low25_membership"] = pair_df["pair_id"].map(membership)
    pair_df["is_our_rm5_low25"] = pair_df["pair_id"].isin(our_low)
    pair_df["is_marina_rm5_low25"] = pair_df["pair_id"].isin(marina_low)
    pair_df["valid_for_rm5_o3_low25"] = pair_df["pair_id"].isin(set(valid["pair_id"].astype(int)))
    return pair_df


def _replace_mapped_summary_row(
    summary: pd.DataFrame,
    test_name: str,
    pair_df: pd.DataFrame,
    metric_col: str,
    fwd_col: str,
    fwd_source: str,
    o3_source: str,
) -> None:
    valid = pair_df.dropna(subset=[metric_col, fwd_col]).copy()
    n_low = max(int(np.floor(0.25 * len(valid))), 1)
    low = valid.nsmallest(n_low, metric_col)
    mean_doy = float(low[fwd_col].mean())
    mask = summary["test"].eq(test_name)
    if not mask.any():
        summary.loc[len(summary)] = {
            "test": test_name,
            "fwd_source": fwd_source,
            "o3_source_for_low25": o3_source,
            "candidate_rows": len(valid),
            "n_low25": len(low),
            "mean_low25_50hpa_doy": round(mean_doy, 2),
            "mean_low25_50hpa_date": doy_to_month_day(mean_doy),
            "delta_vs_paper_days": round(mean_doy - src.PAPER_CLIM3D_LOW25_DOY, 2),
            "low_ids_head": ",".join(str(int(x)) for x in low["pair_id"].head(8)),
        }
        return
    summary.loc[mask, "fwd_source"] = fwd_source
    summary.loc[mask, "o3_source_for_low25"] = o3_source
    summary.loc[mask, "candidate_rows"] = len(valid)
    summary.loc[mask, "n_low25"] = len(low)
    summary.loc[mask, "mean_low25_50hpa_doy"] = round(mean_doy, 2)
    summary.loc[mask, "mean_low25_50hpa_date"] = doy_to_month_day(mean_doy)
    summary.loc[mask, "delta_vs_paper_days"] = round(mean_doy - src.PAPER_CLIM3D_LOW25_DOY, 2)
    summary.loc[mask, "low_ids_head"] = ",".join(str(int(x)) for x in low["pair_id"].head(8))


def plot_source_summary(pair_df: pd.DataFrame) -> None:
    df = pd.read_csv(SOURCE_SUMMARY_CSV)
    _replace_mapped_summary_row(
        df,
        "Mapped pair: our FWD + Marina O3 rm5_file",
        pair_df,
        metric_col="marina_o3_rm5_min_DU",
        fwd_col="our_fwd50_doy",
        fwd_source="our generated FWD on feature-matched pair",
        o3_source="Marina rm5 O3 on feature-matched pair",
    )
    _replace_mapped_summary_row(
        df,
        "Mapped pair: Marina FWD + our O3 csv_rm5",
        pair_df,
        metric_col="our_o3_rm5_min_DU",
        fwd_col="marina_fwd50_doy",
        fwd_source="Marina saved FWD on feature-matched pair",
        o3_source="our partial_O3 rm5 on feature-matched pair",
    )
    df.to_csv(TABLE_DIR / "source_isolation_summary_feature_matched_for_test_fwd_plots.csv", index=False)
    # Reuse the compact source-isolation figure design, but write it to the TEST_FWD bundle.
    old_dir = report_figs.PLOT_DIR
    report_figs.PLOT_DIR = PLOT_DIR
    try:
        report_figs.plot_variant_summary(df)
    finally:
        report_figs.PLOT_DIR = old_dir


def plot_corrected_mapping_scatters(pair_df: pd.DataFrame) -> None:
    old_dir = report_figs.PLOT_DIR
    report_figs.PLOT_DIR = PLOT_DIR
    try:
        report_figs.plot_mapped_fwd_scatter(report_figs.load_feature_matched_fwd_pair_table())
        report_figs.plot_mapped_o3_scatter(pair_df)
    finally:
        report_figs.PLOT_DIR = old_dir
    # Also refresh the legacy clim3d_marina_repro output path with corrected O3 mapping.
    old_dir = report_figs.PLOT_DIR
    report_figs.PLOT_DIR = ROOT / "Longrun/Visualization/plots/clim3d_marina_repro"
    try:
        report_figs.plot_mapped_o3_scatter(pair_df)
    finally:
        report_figs.PLOT_DIR = old_dir


def plot_fwd_by_level_comparison() -> None:
    df = pd.read_csv(FEATURE_FWD_BY_LEVEL_CSV)
    levels = [1, 2, 3, 5, 10, 20, 30, 50]
    colors = {1: "#1b9e77", 2: "#d95f02", 3: "#7570b3", 4: "#e7298a"}
    summary = (
        df.groupby("plev_hpa", as_index=False)
        .agg(
            n=("abs_diff_days", "count"),
            bias_days=("diff_our_minus_marina_days", "mean"),
            mae_days=("abs_diff_days", "mean"),
            max_abs_days=("abs_diff_days", "max"),
        )
        .sort_values("plev_hpa")
    )
    summary.to_csv(TABLE_DIR / "feature_matched_fwd_by_level_summary.csv", index=False)

    fig, axes = plt.subplots(2, 4, figsize=(12.2, 6.0), sharex=True, sharey=True)
    for ax, level in zip(axes.ravel(), levels):
        sub = df[np.isclose(df["plev_hpa"], level)].copy()
        for chunk_id, group in sub.groupby("chunk_id"):
            ax.scatter(
                group["marina_fwd_doy"],
                group["our_fwd_doy"],
                s=15,
                alpha=0.75,
                color=colors.get(int(chunk_id), "0.45"),
                label=f"chunk {int(chunk_id)}",
            )
        lo = min(sub["marina_fwd_doy"].min(), sub["our_fwd_doy"].min()) - 3
        hi = max(sub["marina_fwd_doy"].max(), sub["our_fwd_doy"].max()) + 3
        ax.plot([lo, hi], [lo, hi], color="0.2", lw=0.8)
        ax.set_title(f"{level:g} hPa", fontsize=10)
        row = summary[np.isclose(summary["plev_hpa"], level)].iloc[0]
        ax.text(
            0.04,
            0.94,
            f"MAE={row.mae_days:.2f} d\nmax={row.max_abs_days:.0f} d",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9, "pad": 2},
        )
        ax.grid(color="0.9", lw=0.7)
    axes[1, 0].set_xlabel("Marina saved FWD (DOY)")
    axes[1, 1].set_xlabel("Marina saved FWD (DOY)")
    axes[1, 2].set_xlabel("Marina saved FWD (DOY)")
    axes[1, 3].set_xlabel("Marina saved FWD (DOY)")
    axes[0, 0].set_ylabel("Our generated FWD (DOY)")
    axes[1, 0].set_ylabel("Our generated FWD (DOY)")
    axes[0, 3].legend(loc="lower right", fontsize=7, frameon=False)
    fig.suptitle("Feature-matched CLIM-3D FWD by pressure level", fontsize=14)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    save_figure(fig, "clim3d_feature_matched_fwd_by_level_scatter")

    fig, ax = plt.subplots(figsize=(5.3, 5.2))
    ax.plot(summary["mae_days"], summary["plev_hpa"], marker="o", color="0.2", label="MAE")
    ax.plot(summary["max_abs_days"], summary["plev_hpa"], marker="s", color="#b2182b", label="max |diff|")
    ax.axvline(1.0, color="0.55", lw=0.9, ls="--")
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.set_yticks(levels)
    ax.set_yticklabels([f"{p:g}" for p in levels])
    ax.set_xlabel("|our - Marina| FWD difference (days)")
    ax.set_ylabel("Pressure (hPa)")
    ax.set_title("Feature-matched FWD mismatch profile")
    ax.grid(color="0.9", lw=0.8)
    ax.legend(frameon=False)
    save_figure(fig, "clim3d_feature_matched_fwd_by_level_error_profile")


def plot_figure2_variants() -> None:
    local_profiles = load_profiles_by_window(rolling_days=5, use_marina_clim3d=False)
    fig = plot_waccm_four_panel_group(local_profiles, "WACCM FWD response | 5-day running-mean O3-year selection")
    save_figure(fig, "figure2_fwd_response_rm5_o3_4panel")

    # Only CLIM-3D changes in this sensitivity figure.  Reuse the already loaded
    # INT-3D and CLIM-2D profiles so the script does not reread all local data.
    marina_profiles = {window: dict(profiles) for window, profiles in local_profiles.items()}
    marina_cache: dict = {}
    marina_cfg = figure2_cases(use_marina_clim3d=True)["CLIM-3D"]
    for window_mode in FWD_O3_WINDOW_ORDER:
        marina_profiles[window_mode]["CLIM-3D"] = load_profile(
            "CLIM-3D",
            marina_cfg,
            window_mode,
            5,
            marina_cache,
        )
    fig = plot_waccm_four_panel_group(
        marina_profiles,
        "WACCM FWD response | CLIM-3D replaced by Marina saved FWD + runmean5 O3",
    )
    save_figure(fig, "figure2_fwd_response_rm5_o3_4panel_marina_clim3d")

    rows = []
    for tag, profiles in [("local_clim3d", local_profiles), ("marina_clim3d", marina_profiles)]:
        for window_mode, cases in profiles.items():
            for label, prof in cases.items():
                for p, all_mean, low_mean, high_mean in zip(
                    prof["p_hpa"], prof["all_mean"], prof["low_mean_abs"], prof["high_mean_abs"]
                ):
                    rows.append(
                        {
                            "figure_variant": tag,
                            "window_mode": window_mode,
                            "case": label,
                            "plev_hpa": float(p),
                            "all_mean_fwd_doy": float(all_mean),
                            "low25_mean_fwd_doy": float(low_mean),
                            "high25_mean_fwd_doy": float(high_mean),
                            "low25_dev_days": float(low_mean - all_mean),
                            "high25_dev_days": float(high_mean - all_mean),
                            "n_all": len(prof["all_years"]),
                            "n_low25": len(prof["low_years"]),
                            "n_high25": len(prof["high_years"]),
                            "ranking_source": prof["ranking_source"],
                        }
                    )
    pd.DataFrame(rows).to_csv(TABLE_DIR / "figure2_rm5_o3_local_vs_marina_clim3d_profiles.csv", index=False)


def main() -> None:
    ensure_dirs()
    print("[1/5] Building feature-matched O3 pair table", flush=True)
    pair_df = build_feature_matched_rm5_pair_table()
    pair_df.to_csv(TABLE_DIR / "mapped_pair_rm5_feature_matched_details.csv", index=False)
    print("[2/5] Plotting source-isolation summary", flush=True)
    plot_source_summary(pair_df)
    print("[3/5] Plotting corrected mapped O3/FWD scatters", flush=True)
    plot_corrected_mapping_scatters(pair_df)
    print("[4/5] Plotting FWD by-level comparisons", flush=True)
    plot_fwd_by_level_comparison()
    print("[5/5] Plotting Figure 2 local and Marina-CLIM3D variants", flush=True)
    plot_figure2_variants()

    valid = pair_df.dropna(subset=["our_o3_rm5_min_DU", "marina_o3_rm5_min_DU"]).copy()
    n_low = max(int(np.floor(0.25 * len(valid))), 1)
    overlap = int((valid["is_our_rm5_low25"] & valid["is_marina_rm5_low25"]).sum())
    corr = float(valid[["marina_o3_rm5_min_DU", "our_o3_rm5_min_DU"]].corr().iloc[0, 1])
    mae = float(np.mean(np.abs(valid["our_o3_rm5_min_DU"] - valid["marina_o3_rm5_min_DU"])))
    print(f"Wrote TEST_FWD plots to {PLOT_DIR}")
    print(f"Feature-matched rm5 O3: n={len(valid)}, corr={corr:.4f}, MAE={mae:.2f} DU, LOW25 overlap={overlap}/{n_low}")


if __name__ == "__main__":
    main()
