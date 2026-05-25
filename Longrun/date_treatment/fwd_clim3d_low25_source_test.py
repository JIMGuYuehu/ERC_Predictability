#!/usr/bin/env python
"""Source-isolation test for CLIM-3D LOW25% 50 hPa FWD.

This is a read-only diagnostic used by TEST_NEW_DATA.ipynb.  It compares
Marina's CLIM-3D O3/FWD products with the local CLIM-3D products by holding
either the O3-year selection or the FWD series fixed.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import numpy as np
import pandas as pd
import xarray as xr


OUR_FWD_NC = Path(
    "/mnt/soclim0/public_data/weiji/"
    "B2000WCN007009010011_Clim3D_timefixed/final_warming_date/"
    "B2000WCN007009010011_Clim3D_FWD_plev_year.nc"
)
OUR_PARTIAL_O3_NC = Path(
    "/mnt/soclim0/public_data/weiji/"
    "B2000WCN007009010011_Clim3D_timefixed/partial_O3/"
    "B2000WCN007009010011_Clim3D_partial_O3_all_ranges.nc"
)
OUR_RANKING_CSV = Path(
    "/mnt/soclim0/public_data/weiji/"
    "B2000WCN007009010011_Clim3D_timefixed/partial_O3/"
    "B2000WCN007009010011_Clim3D_partial_O3_ranking_MarApr_min_60_90N.csv"
)

MARINA_U_O3_NC = Path(
    "/mnt/backup_ETH/Marina/WACCM/NOCHEM_2000_3D/combined/"
    "B2000WCN.NOCOUPL.e122.f19_g16.int.zm.nc"
)
MARINA_RUNMEAN5_O3_NC = Path(
    "/mnt/backup_ETH/Marina/WACCM/NOCHEM_2000_3D/combined/"
    "B2000WCN.NOCOUPL.e122.f19_g16.int.zm.runmean5.nc"
)
MARINA_SAVED_FWD_NPY = Path(
    "/home/gchiodo/code/Marina/python-scripts-main/"
    'Friedel et al., 2022: "Effects of Arctic ozone on the stratospheric '
    'spring onset and its surface impact"/FW_dates/FW_vertical_newthreshIII_1.npy'
)

PRESSURE_RANGE = "30_70hPa"
VALID_MIN_DU = 10.0
MA_START_DOY = 60
MA_END_DOY = 120
MIN_VALID_DAYS = 5
TARGET_LEVEL_HPA = 50.0
PAPER_CLIM3D_LOW25_DOY = 117.0  # Friedel et al. (2022) Table 1: 27 April.

MONTH_ENDS_NOLEAP = np.array([31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365])
MONTH_NAMES = np.array(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
)

CLIM3D_CHUNKS = [
    # Fingerprint-matched chunk order from fwd_clim3d_feature_mapping_test.py.
    # The older hand mapping duplicated Marina years 50-58 and produced
    # artificial 50 hPa FWD outliers in the source-isolation scatter.
    {"our_start": 5, "our_end": 56, "marina_start": 1, "marina_end": 52},
    {"our_start": 114, "our_end": 161, "marina_start": 53, "marina_end": 100},
    {"our_start": 62, "our_end": 113, "marina_start": 101, "marina_end": 152},
    {"our_start": 163, "our_end": 210, "marina_start": 153, "marina_end": 200},
]


def doy_to_month_day(doy: float) -> str:
    if not np.isfinite(doy):
        return "nan"
    day = int(np.round(float(doy)))
    day = min(max(day, 1), 365)
    month_idx = int(np.searchsorted(MONTH_ENDS_NOLEAP, day, side="left"))
    prev = int(MONTH_ENDS_NOLEAP[month_idx - 1]) if month_idx > 0 else 0
    return f"{day - prev:02d} {MONTH_NAMES[month_idx]}"


def centered_rolling(values: np.ndarray, window: int | None) -> np.ndarray:
    if window is None or int(window) <= 1:
        return np.asarray(values, dtype=float)
    return (
        pd.Series(np.asarray(values, dtype=float))
        .rolling(int(window), center=True, min_periods=int(window))
        .mean()
        .to_numpy(dtype=float)
    )


def fwd_date_to_doy(date_values: np.ndarray) -> np.ndarray:
    dates = np.asarray(date_values, dtype=np.int64)
    mmdd = dates % 10000
    month = (mmdd // 100).astype(np.int16)
    day = (mmdd % 100).astype(np.int16)
    out = np.full(dates.shape, -9999, dtype=np.int16)
    for m in range(1, 13):
        mask = month == m
        if np.any(mask):
            prev = int(MONTH_ENDS_NOLEAP[m - 2]) if m > 1 else 0
            out[mask] = prev + day[mask]
    return out


def years_doys_from_ds(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    if "date" in ds:
        dates = ds["date"].values.astype(np.int64)
        return (dates // 10000).astype(int), fwd_date_to_doy(dates).astype(int)
    day_index = np.asarray(ds["time"].values, dtype=float).astype(np.int64)
    return (day_index // 365 + 1).astype(int), (day_index % 365 + 1).astype(int)


def mapping_pairs() -> pd.DataFrame:
    rows = []
    pair_id = 1
    for chunk in CLIM3D_CHUNKS:
        n_our = chunk["our_end"] - chunk["our_start"] + 1
        n_marina = chunk["marina_end"] - chunk["marina_start"] + 1
        if n_our != n_marina:
            raise ValueError(f"Chunk length mismatch: {chunk}")
        for offset in range(n_our):
            rows.append(
                {
                    "pair_id": pair_id,
                    "our_year": chunk["our_start"] + offset,
                    "marina_year": chunk["marina_start"] + offset,
                }
            )
            pair_id += 1
    return pd.DataFrame(rows)


def nearest_index(values: np.ndarray, target: float) -> int:
    values = np.asarray(values, dtype=float)
    return int(np.nanargmin(np.abs(values - target)))


def load_our_fwd_50hpa() -> pd.Series:
    ds = xr.open_dataset(OUR_FWD_NC, decode_times=False)
    try:
        fwd = ds["FWD_dayofyear"]
        if "plev_hpa" in ds:
            p_hpa = ds["plev_hpa"].values.astype(float)
        elif "plev_hpa" in fwd.coords:
            p_hpa = fwd["plev_hpa"].values.astype(float)
        else:
            p_hpa = ds["plev"].values.astype(float) / 100.0
        idx = nearest_index(p_hpa, TARGET_LEVEL_HPA)
        years = fwd["year"].values.astype(int)
        vals = fwd.isel(plev=idx).load().values.astype(float)
    finally:
        ds.close()
    return pd.Series(vals, index=years, name="our_fwd_50hpa_doy").dropna()


def marina_saved_levels_hpa() -> np.ndarray:
    ds = xr.open_dataset(MARINA_U_O3_NC, decode_times=False)
    try:
        p = ds["plev"].values.astype(float)
    finally:
        ds.close()
    if np.nanmax(p) > 1000:
        return p[(p >= 10.0) & (p <= 5000.0)] / 100.0
    return p[(p >= 0.1) & (p <= 50.0)]


def load_marina_saved_fwd_50hpa() -> pd.Series:
    p_hpa = marina_saved_levels_hpa()
    arr = np.load(MARINA_SAVED_FWD_NPY).astype(float)
    if arr.shape[0] == len(p_hpa):
        arr = arr.T
    if arr.shape[1] != len(p_hpa):
        raise ValueError(f"Unexpected saved FWD shape {arr.shape} for {len(p_hpa)} levels")
    idx = nearest_index(p_hpa, TARGET_LEVEL_HPA)
    years = np.arange(1, arr.shape[0] + 1, dtype=int)
    return pd.Series(arr[:, idx], index=years, name="marina_saved_fwd_50hpa_doy").dropna()


def load_our_partial_o3_base() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ds = xr.open_dataset(OUR_PARTIAL_O3_NC, decode_times=False)
    try:
        if "O3_partial_60_90N" in ds:
            da = ds["O3_partial_60_90N"].sel(pressure_range=PRESSURE_RANGE)
        else:
            da = ds[f"O3_partial_60_90N_{PRESSURE_RANGE}"]
        years, doys = years_doys_from_ds(ds)
        values = da.where(da > VALID_MIN_DU).load().values.astype(float)
    finally:
        ds.close()
    return values, years, doys


def load_marina_o3_base(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        values = o3.where(o3 > VALID_MIN_DU).load().values.astype(float)
    finally:
        ds.close()
    ntime = values.shape[0]
    day_index = np.arange(ntime, dtype=np.int64)
    years = (day_index // 365 + 1).astype(int)
    doys = (day_index % 365 + 1).astype(int)
    return values, years, doys


def ma_min_metric_by_year(
    values: np.ndarray,
    years: np.ndarray,
    doys: np.ndarray,
    candidate_years: set[int] | None = None,
) -> pd.DataFrame:
    if candidate_years is None:
        candidate_years = set(np.unique(years).astype(int).tolist())
    records = []
    for year in sorted(candidate_years):
        mask = (years == int(year)) & (doys >= MA_START_DOY) & (doys <= MA_END_DOY)
        vals = np.asarray(values[mask], dtype=float)
        vals = vals[np.isfinite(vals) & (vals > VALID_MIN_DU)]
        if vals.size < MIN_VALID_DAYS:
            continue
        records.append(
            {
                "year": int(year),
                "window_min_DU": float(np.min(vals)),
                "window_max_DU": float(np.max(vals)),
                "window_mean_DU": float(np.mean(vals)),
                "n_valid_days": int(vals.size),
            }
        )
    return pd.DataFrame(records).sort_values("year").reset_index(drop=True)


def ma_min_metric_from_our_csv(method: str) -> pd.DataFrame:
    df = pd.read_csv(OUR_RANKING_CSV)
    df = df[df["pressure_range"].astype(str).eq(PRESSURE_RANGE)].copy()
    min_col = f"marapr_min_DU_{method}"
    max_col = f"marapr_max_DU_{method}"
    mean_col = f"marapr_mean_DU_{method}"
    out = pd.DataFrame(
        {
            "year": pd.to_numeric(df["year"], errors="coerce").astype("Int64"),
            "window_min_DU": pd.to_numeric(df[min_col], errors="coerce"),
            "window_max_DU": pd.to_numeric(df[max_col], errors="coerce"),
            "window_mean_DU": pd.to_numeric(df[mean_col], errors="coerce"),
            "n_valid_days": np.nan,
        }
    ).dropna(subset=["year", "window_min_DU"])
    out["year"] = out["year"].astype(int)
    return out.sort_values("year").reset_index(drop=True)


def metric_dict(metric_df: pd.DataFrame) -> dict[int, float]:
    return {
        int(row.year): float(row.window_min_DU)
        for row in metric_df.itertuples(index=False)
        if np.isfinite(row.window_min_DU)
    }


def low25_rows(df: pd.DataFrame, metric_col: str = "window_min_DU") -> pd.DataFrame:
    clean = df[np.isfinite(df[metric_col]) & np.isfinite(df["fwd_doy"])].copy()
    n_low = max(int(np.floor(0.25 * len(clean))), 1)
    return clean.nsmallest(n_low, metric_col).copy()


def summarize_low25(name: str, df: pd.DataFrame, fwd_source: str, o3_source: str) -> dict[str, object]:
    low = low25_rows(df)
    mean_doy = float(low["fwd_doy"].mean())
    return {
        "test": name,
        "fwd_source": fwd_source,
        "o3_source_for_low25": o3_source,
        "candidate_rows": int(len(df)),
        "n_low25": int(len(low)),
        "mean_low25_50hpa_doy": round(mean_doy, 2),
        "mean_low25_50hpa_date": doy_to_month_day(mean_doy),
        "delta_vs_paper_days": round(mean_doy - PAPER_CLIM3D_LOW25_DOY, 2),
        "low_ids_head": ",".join(str(int(x)) for x in low["display_year"].head(8)),
    }


def native_eval(
    name: str,
    metrics: dict[int, float],
    fwd: pd.Series,
    fwd_source: str,
    o3_source: str,
    years: set[int] | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    if years is None:
        years = set(metrics).intersection(set(fwd.index.astype(int)))
    rows = []
    for year in sorted(years):
        if year in metrics and year in fwd.index:
            rows.append(
                {
                    "display_year": int(year),
                    "year": int(year),
                    "window_min_DU": float(metrics[year]),
                    "fwd_doy": float(fwd.loc[year]),
                }
            )
    df = pd.DataFrame(rows)
    return summarize_low25(name, df, fwd_source, o3_source), df


def mapped_eval(
    name: str,
    mapping: pd.DataFrame,
    metrics: dict[int, float],
    fwd: pd.Series,
    metric_side: str,
    fwd_side: str,
    fwd_source: str,
    o3_source: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    rows = []
    for row in mapping.itertuples(index=False):
        metric_year = int(row.our_year if metric_side == "our" else row.marina_year)
        fwd_year = int(row.our_year if fwd_side == "our" else row.marina_year)
        if metric_year in metrics and fwd_year in fwd.index:
            rows.append(
                {
                    "display_year": int(row.pair_id),
                    "pair_id": int(row.pair_id),
                    "our_year": int(row.our_year),
                    "marina_year": int(row.marina_year),
                    "metric_year": metric_year,
                    "fwd_year": fwd_year,
                    "window_min_DU": float(metrics[metric_year]),
                    "fwd_doy": float(fwd.loc[fwd_year]),
                }
            )
    df = pd.DataFrame(rows)
    return summarize_low25(name, df, fwd_source, o3_source), df


def mapped_pair_diagnostics(
    mapping: pd.DataFrame,
    our_fwd: pd.Series,
    marina_fwd: pd.Series,
    our_metrics_by_label: dict[str, dict[int, float]],
    marina_metrics_by_label: dict[str, dict[int, float]],
) -> pd.DataFrame:
    fwd_rows = []
    for row in mapping.itertuples(index=False):
        if row.our_year in our_fwd.index and row.marina_year in marina_fwd.index:
            fwd_rows.append((float(our_fwd.loc[row.our_year]), float(marina_fwd.loc[row.marina_year])))
    fwd_arr = np.asarray(fwd_rows, dtype=float)
    records = [
        {
            "diagnostic": "mapped 50hPa FWD our-vs-Marina",
            "variant": "FWD",
            "n_pairs": int(fwd_arr.shape[0]),
            "corr": round(float(np.corrcoef(fwd_arr[:, 0], fwd_arr[:, 1])[0, 1]), 3),
            "mean_abs_diff": round(float(np.mean(np.abs(fwd_arr[:, 0] - fwd_arr[:, 1]))), 2),
            "same_low25_pairs": np.nan,
        }
    ]

    metric_pair_specs = [
        ("raw", "raw", "raw"),
        ("rm5_from_raw", "rm5", "rm5_from_raw"),
        ("rm5_file", "rm5", "rm5_file"),
        ("rm15", "rm15", "rm15"),
    ]
    for variant, our_label, marina_label in metric_pair_specs:
        if our_label not in our_metrics_by_label or marina_label not in marina_metrics_by_label:
            continue
        rows = []
        for row in mapping.itertuples(index=False):
            o_val = our_metrics_by_label[our_label].get(int(row.our_year), np.nan)
            m_val = marina_metrics_by_label[marina_label].get(int(row.marina_year), np.nan)
            if np.isfinite(o_val) and np.isfinite(m_val):
                rows.append((int(row.pair_id), float(o_val), float(m_val)))
        arr = pd.DataFrame(rows, columns=["pair_id", "our_metric", "marina_metric"])
        if arr.empty:
            continue
        n_low = max(int(np.floor(0.25 * len(arr))), 1)
        our_low = set(arr.nsmallest(n_low, "our_metric")["pair_id"].astype(int).tolist())
        marina_low = set(arr.nsmallest(n_low, "marina_metric")["pair_id"].astype(int).tolist())
        records.append(
            {
                "diagnostic": "mapped Mar-Apr O3 minima our-vs-Marina",
                "variant": variant,
                "n_pairs": int(len(arr)),
                "corr": round(float(arr[["our_metric", "marina_metric"]].corr().iloc[0, 1]), 3),
                "mean_abs_diff": round(float(np.mean(np.abs(arr["our_metric"] - arr["marina_metric"]))), 2),
                "same_low25_pairs": int(len(our_low.intersection(marina_low))),
            }
        )
    return pd.DataFrame(records)


def build_metrics() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    our_values, our_years, our_doys = load_our_partial_o3_base()
    marina_values, marina_years, marina_doys = load_marina_o3_base(MARINA_U_O3_NC)
    marina_rm5_values, marina_rm5_years, marina_rm5_doys = load_marina_o3_base(MARINA_RUNMEAN5_O3_NC)

    our_metric_frames = {
        "raw": ma_min_metric_by_year(our_values, our_years, our_doys),
        "rm5": ma_min_metric_by_year(centered_rolling(our_values, 5), our_years, our_doys),
        "rm15": ma_min_metric_by_year(centered_rolling(our_values, 15), our_years, our_doys),
        "csv_raw": ma_min_metric_from_our_csv("raw"),
        "csv_rm5": ma_min_metric_from_our_csv("rm5"),
    }
    marina_metric_frames = {
        "raw": ma_min_metric_by_year(marina_values, marina_years, marina_doys),
        "rm5_from_raw": ma_min_metric_by_year(centered_rolling(marina_values, 5), marina_years, marina_doys),
        "rm15": ma_min_metric_by_year(centered_rolling(marina_values, 15), marina_years, marina_doys),
        "rm5_file": ma_min_metric_by_year(marina_rm5_values, marina_rm5_years, marina_rm5_doys),
    }
    return our_metric_frames, marina_metric_frames


def run_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping = mapping_pairs()
    our_fwd = load_our_fwd_50hpa()
    marina_fwd = load_marina_saved_fwd_50hpa()
    our_metric_frames, marina_metric_frames = build_metrics()
    our_metrics = {key: metric_dict(df) for key, df in our_metric_frames.items()}
    marina_metrics = {key: metric_dict(df) for key, df in marina_metric_frames.items()}

    summaries: list[dict[str, object]] = [
        {
            "test": "paper Table 1 reference",
            "fwd_source": "Friedel et al. 2022",
            "o3_source_for_low25": "WACCM CLIM-3D low-ozone springs",
            "candidate_rows": 200,
            "n_low25": 50,
            "mean_low25_50hpa_doy": PAPER_CLIM3D_LOW25_DOY,
            "mean_low25_50hpa_date": doy_to_month_day(PAPER_CLIM3D_LOW25_DOY),
            "delta_vs_paper_days": 0.0,
            "low_ids_head": "",
        }
    ]

    for key in ["raw", "rm5_from_raw", "rm5_file", "rm15"]:
        summary, _ = native_eval(
            f"Marina native saved FWD + Marina O3 {key}",
            marina_metrics[key],
            marina_fwd,
            "Marina saved FW_vertical_newthreshIII_1.npy",
            f"Marina {key}",
        )
        summaries.append(summary)

    for key in ["raw", "rm5", "rm15", "csv_raw", "csv_rm5"]:
        summary, _ = native_eval(
            f"Our native FWD + our O3 {key}",
            our_metrics[key],
            our_fwd,
            "our generated CLIM-3D FWD",
            f"our partial_O3 {key}",
        )
        summaries.append(summary)

    for key in ["raw", "rm5", "rm15", "csv_raw", "csv_rm5"]:
        summary, _ = mapped_eval(
            f"Mapped pair: Marina FWD + our O3 {key}",
            mapping,
            our_metrics[key],
            marina_fwd,
            metric_side="our",
            fwd_side="marina",
            fwd_source="Marina saved FWD on mapped pair",
            o3_source=f"our partial_O3 {key} on mapped pair",
        )
        summaries.append(summary)

    for key in ["raw", "rm5_from_raw", "rm5_file", "rm15"]:
        summary, _ = mapped_eval(
            f"Mapped pair: our FWD + Marina O3 {key}",
            mapping,
            marina_metrics[key],
            our_fwd,
            metric_side="marina",
            fwd_side="our",
            fwd_source="our generated FWD on mapped pair",
            o3_source=f"Marina {key} on mapped pair",
        )
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    diag_df = mapped_pair_diagnostics(mapping, our_fwd, marina_fwd, our_metrics, marina_metrics)
    return summary_df, diag_df


def main() -> None:
    print("CLIM-3D LOW25% 50 hPa FWD source-isolation test")
    print(f"Paper target: {PAPER_CLIM3D_LOW25_DOY:.0f} DOY = {doy_to_month_day(PAPER_CLIM3D_LOW25_DOY)}")
    print("Selection target: Mar-Apr minimum partial O3, LOW25%, pressure_range=30_70hPa")
    print()
    summary_df, diag_df = run_test()
    ordered = summary_df.sort_values("delta_vs_paper_days", key=lambda s: s.abs()).reset_index(drop=True)
    print("Summary sorted by absolute delta from Friedel Table 1:")
    print(ordered.to_string(index=False))
    print()
    print("Mapped-pair diagnostics:")
    print(diag_df.to_string(index=False))
    print()
    closest = ordered.iloc[1] if len(ordered) > 1 else ordered.iloc[0]
    print(
        "Closest non-reference row: "
        f"{closest['test']} -> {closest['mean_low25_50hpa_date']} "
        f"(delta {closest['delta_vs_paper_days']} d)."
    )
    print(
        "Interpretation guide: if Marina-native rows are near 27 Apr but mapped/local rows move away, "
        "the mismatch comes from the local chunk mapping/data products rather than a 5-day or 15-day "
        "O3 smoothing choice."
    )


if __name__ == "__main__":
    main()
