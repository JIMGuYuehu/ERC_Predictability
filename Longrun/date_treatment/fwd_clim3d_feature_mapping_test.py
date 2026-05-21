"""Feature-based CLIM-3D year mapping test against Marina products.

This script answers one narrow debugging question: does the local cleaned
CLIM-3D year order match Marina's merged CLIM-3D order when the match is made
from physical fields first, before looking at final warming dates?

Default fingerprint:
    daily polar ozone, 60-90N, March-May. This uses compact products and is
    fast enough to rerun from TEST_NEW_DATA.ipynb.

Optional fingerprint:
    daily U at about 10 hPa, 55-75N, days 1-181. Use
    ``--primary-feature u`` when the large yearly U files can be read.

The resulting chunk mapping is then used to compare local FWD against Marina's
saved FWD product. Outputs are written below this directory so the notebook can
show the result without re-implementing the logic.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
LONGRUN_DIR = SCRIPT_DIR.parent
OUT_DIR = SCRIPT_DIR / "clim3d_feature_mapping_test"
PLOT_DIR = LONGRUN_DIR / "Visualization" / "plots" / "clim3d_feature_mapping_test"

MARINA_U_O3_NC = Path(
    "/mnt/backup_ETH/Marina/WACCM/NOCHEM_2000_3D/combined/"
    "B2000WCN.NOCOUPL.e122.f19_g16.int.zm.nc"
)
MARINA_O3_COL_NC = Path(
    "/mnt/backup_ETH/Marina/WACCM/NOCHEM_2000_3D/combined/"
    "B2000WCN.NOCOUPL.e122.f19_g16.int.zm.O3_col.nc"
)
MARINA_SAVED_FWD_NPY = Path(
    "/home/gchiodo/code/Marina/python-scripts-main/"
    'Friedel et al., 2022: "Effects of Arctic ozone on the stratospheric '
    'spring onset and its surface impact"/FW_dates/FW_vertical_newthreshIII_1.npy'
)

LOCAL_ROOT = Path(
    "/mnt/soclim0/public_data/weiji/"
    "B2000WCN007009010011_Clim3D_timefixed"
)
LOCAL_U_DIR = LOCAL_ROOT / "U"
LOCAL_PARTIAL_O3_NC = (
    LOCAL_ROOT
    / "partial_O3"
    / "B2000WCN007009010011_Clim3D_partial_O3_all_ranges.nc"
)
LOCAL_FWD_NC = (
    LOCAL_ROOT
    / "final_warming_date"
    / "B2000WCN007009010011_Clim3D_FWD_plev_year.nc"
)

U_TARGET_HPA = 10.0
U_LAT_BAND = (55.0, 75.0)
U_FINGERPRINT_DAYS = 181
O3_LAT_BAND = (60.0, 90.0)
O3_DOY_SLICE = slice(59, 151)  # Mar 1-May 31, zero-based indexing.
TARGET_FWD_HPA = 50.0


@dataclass(frozen=True)
class FeatureBundle:
    years: np.ndarray
    u: np.ndarray | None
    o3: np.ndarray


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)


def corr_1d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return math.nan
    aa = a[mask] - np.nanmean(a[mask])
    bb = b[mask] - np.nanmean(b[mask])
    denom = float(np.sqrt(np.sum(aa * aa) * np.sum(bb * bb)))
    if denom == 0.0:
        return math.nan
    return float(np.sum(aa * bb) / denom)


def nearest_idx(values: np.ndarray, target: float) -> int:
    arr = np.asarray(values, dtype=float)
    return int(np.nanargmin(np.abs(arr - target)))


def weighted_lat_mean(da: xr.DataArray, lat_name: str = "lat") -> xr.DataArray:
    weights = np.cos(np.deg2rad(da[lat_name]))
    return da.weighted(weights).mean(lat_name, skipna=True)


def parse_marina_history_chunks() -> pd.DataFrame:
    with Dataset(MARINA_U_O3_NC) as ds:
        history = str(getattr(ds, "history", ""))

    pattern = re.compile(
        r"B2000WCN\.NOCOUPL\.e122\.f19_g16\.(?P<run>\d{3})"
        r"\.cam\.h3\.(?P<start>\d{4})-(?P<end>\d{4})"
        r"\.int(?P<shift>\.shift200)?\.nc"
    )
    rows = []
    seen: set[tuple[str, int, int, bool]] = set()
    marina_start = 1
    for match in pattern.finditer(history):
        run = match.group("run")
        source_start = int(match.group("start"))
        source_end = int(match.group("end"))
        shifted = bool(match.group("shift"))
        key = (run, source_start, source_end, shifted)
        if key in seen:
            continue
        seen.add(key)
        n_years = source_end - source_start + 1
        rows.append(
            {
                "chunk_id": len(rows) + 1,
                "marina_run": run,
                "source_start_year": source_start,
                "source_end_year": source_end,
                "source_n_years": n_years,
                "marina_start_year": marina_start,
                "marina_end_year": marina_start + n_years - 1,
                "shift200_file": shifted,
            }
        )
        marina_start += n_years
        if len(rows) == 4:
            break

    if len(rows) != 4:
        raise RuntimeError(f"Expected four Marina history chunks, found {len(rows)}")
    return pd.DataFrame(rows)


def local_file_for_year(year: int) -> Path:
    return LOCAL_U_DIR / f"B2000WCN.NOCOUPL.sample.cam.h3.{year:04d}.U.nc"


def parse_local_source_table(n_years: int = 216) -> pd.DataFrame:
    pattern = re.compile(
        r"B2000WCN\.NOCOUPL\.e122\.f19_g16\.(?P<run>\d{3})"
        r"\.cam\.h3\.(?P<year>\d{4})-\d{2}-\d{2}"
    )
    rows = []
    for our_year in range(1, n_years + 1):
        path = local_file_for_year(our_year)
        with Dataset(path) as ds:
            case = str(getattr(ds, "case", ""))
            history = str(getattr(ds, "history", ""))
        case_run = case.split(".")[-1] if case else ""
        match = pattern.search(history)
        source_run = match.group("run") if match else case_run
        source_year = int(match.group("year")) if match else np.nan
        rows.append(
            {
                "our_year": our_year,
                "our_run": case_run,
                "source_run_from_history": source_run,
                "source_year_from_history": source_year,
            }
        )
    return pd.DataFrame(rows)


def load_marina_u_features(force: bool = False) -> np.ndarray:
    cache = OUT_DIR / "marina_u10_55_75N_days1_181.npy"
    if cache.exists() and not force:
        return np.load(cache)

    ds = xr.open_dataset(MARINA_U_O3_NC, decode_times=False)
    try:
        u = ds["U"].sel(plev=U_TARGET_HPA * 100.0, method="nearest")
        u = u.sel(lat=slice(U_LAT_BAND[0], U_LAT_BAND[1]))
        if "lon" in u.dims:
            u = u.mean("lon", skipna=True)
        u = weighted_lat_mean(u)
        values = u.load().values.astype("float32")
    finally:
        ds.close()

    out = values.reshape(-1, 365)[:, :U_FINGERPRINT_DAYS]
    np.save(cache, out)
    return out


def load_local_u_features(local_table: pd.DataFrame, force: bool = False) -> np.ndarray:
    cache = OUT_DIR / "local_u10_55_75N_days1_181.npy"
    if cache.exists() and not force:
        return np.load(cache)

    first_file = local_file_for_year(int(local_table["our_year"].iloc[0]))
    with Dataset(first_file) as ds:
        lev = np.asarray(ds.variables["lev"][:], dtype=float)
        lat = np.asarray(ds.variables["lat"][:], dtype=float)
    lev_idx = nearest_idx(lev, U_TARGET_HPA)
    lat_idx = np.where((lat >= U_LAT_BAND[0]) & (lat <= U_LAT_BAND[1]))[0]
    lat_weights = np.cos(np.deg2rad(lat[lat_idx])).astype(float)
    lat_weights = lat_weights / lat_weights.sum()

    values = np.full((len(local_table), U_FINGERPRINT_DAYS), np.nan, dtype="float32")
    for row in local_table.itertuples(index=False):
        path = local_file_for_year(int(row.our_year))
        with Dataset(path) as ds:
            u = ds.variables["U"][:U_FINGERPRINT_DAYS, lev_idx, lat_idx, :]
            arr = np.ma.filled(u, np.nan).astype(float)
        zonal = np.nanmean(arr, axis=2)
        values[int(row.our_year) - 1, :] = np.nansum(zonal * lat_weights[None, :], axis=1)

    np.save(cache, values)
    return values


def load_marina_o3_features(force: bool = False) -> np.ndarray:
    cache = OUT_DIR / "marina_o3col_60_90N_mar_may.npy"
    if cache.exists() and not force:
        return np.load(cache)

    ds = xr.open_dataset(MARINA_O3_COL_NC, decode_times=False)
    try:
        o3 = ds["O3_col"].sel(lat=slice(O3_LAT_BAND[0], O3_LAT_BAND[1]))
        o3 = weighted_lat_mean(o3)
        values = o3.load().values.astype("float32")
    finally:
        ds.close()

    out = values.reshape(-1, 365)[:, O3_DOY_SLICE]
    np.save(cache, out)
    return out


def load_local_o3_features(force: bool = False) -> np.ndarray:
    cache = OUT_DIR / "local_partial_o3_60_90N_30_70hPa_mar_may.npy"
    if cache.exists() and not force:
        return np.load(cache)

    ds = xr.open_dataset(LOCAL_PARTIAL_O3_NC, decode_times=False)
    try:
        name = "O3_partial_60_90N_30_70hPa"
        values = ds[name].load().values.astype("float32")
    finally:
        ds.close()

    out = values.reshape(-1, 365)[:, O3_DOY_SLICE]
    np.save(cache, out)
    return out


def load_feature_bundles(
    force: bool = False,
    include_u: bool = False,
) -> tuple[FeatureBundle, FeatureBundle, pd.DataFrame]:
    local_table = parse_local_source_table()
    marina = FeatureBundle(
        years=np.arange(1, 201, dtype=int),
        u=load_marina_u_features(force=force) if include_u else None,
        o3=load_marina_o3_features(force=force),
    )
    local = FeatureBundle(
        years=local_table["our_year"].to_numpy(dtype=int),
        u=load_local_u_features(local_table, force=force) if include_u else None,
        o3=load_local_o3_features(force=force),
    )
    return marina, local, local_table


def chunk_window_scores(
    marina_feature: np.ndarray,
    local_feature: np.ndarray,
    chunk: pd.Series,
    metric_name: str,
) -> pd.DataFrame:
    m0 = int(chunk.marina_start_year) - 1
    m1 = int(chunk.marina_end_year)
    length = int(chunk.source_n_years)
    marina_vec = marina_feature[m0:m1]

    rows = []
    for local_start0 in range(0, local_feature.shape[0] - length + 1):
        local_vec = local_feature[local_start0 : local_start0 + length]
        rows.append(
            {
                "chunk_id": int(chunk.chunk_id),
                "metric": metric_name,
                "marina_start_year": int(chunk.marina_start_year),
                "marina_end_year": int(chunk.marina_end_year),
                "local_start_year": local_start0 + 1,
                "local_end_year": local_start0 + length,
                "correlation": corr_1d(marina_vec, local_vec),
            }
        )
    return pd.DataFrame(rows)


def build_feature_mapping(
    chunks: pd.DataFrame,
    marina: FeatureBundle,
    local: FeatureBundle,
    primary_metric: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_frames = []
    for chunk in chunks.itertuples(index=False):
        chunk_s = pd.Series(chunk._asdict())
        if marina.u is not None and local.u is not None:
            score_frames.append(chunk_window_scores(marina.u, local.u, chunk_s, "U10_daily"))
        score_frames.append(chunk_window_scores(marina.o3, local.o3, chunk_s, "O3_daily"))
    scores = pd.concat(score_frames, ignore_index=True)

    if primary_metric not in set(scores["metric"]):
        raise ValueError(f"Requested primary metric {primary_metric!r}, available={sorted(scores['metric'].unique())}")
    primary_scores = scores[scores["metric"].eq(primary_metric)].copy()
    best = (
        primary_scores.sort_values(["chunk_id", "correlation"], ascending=[True, False])
        .groupby("chunk_id", as_index=False)
        .head(1)
        .copy()
    )

    rows = []
    for row in best.itertuples(index=False):
        chunk = chunks.loc[chunks["chunk_id"].eq(int(row.chunk_id))].iloc[0]
        local_start = int(row.local_start_year)
        for offset in range(int(chunk.source_n_years)):
            rows.append(
                {
                    "pair_id": len(rows) + 1,
                    "chunk_id": int(chunk.chunk_id),
                    "marina_run": chunk.marina_run,
                    "marina_source_year": int(chunk.source_start_year) + offset,
                    "marina_year": int(chunk.marina_start_year) + offset,
                    "our_year": local_start + offset,
                    "matched_by": f"{primary_metric}_sliding_window",
                    "chunk_match_correlation": float(row.correlation),
                }
            )
    mapping = pd.DataFrame(rows)
    return mapping, scores


def add_local_source_info(mapping: pd.DataFrame, local_table: pd.DataFrame) -> pd.DataFrame:
    out = mapping.merge(local_table, on="our_year", how="left")
    out["run_matches_history"] = out["marina_run"].astype(str).eq(out["our_run"].astype(str))
    out["source_year_matches_history"] = (
        out["marina_source_year"].astype(float).eq(out["source_year_from_history"].astype(float))
    )
    return out


def marina_saved_levels_hpa() -> np.ndarray:
    with Dataset(MARINA_U_O3_NC) as ds:
        p_pa = np.asarray(ds.variables["plev"][:], dtype=float)
    p_hpa = p_pa / 100.0
    return p_hpa[(p_hpa >= 0.1) & (p_hpa <= 50.0)]


def load_marina_fwd_profiles() -> tuple[np.ndarray, np.ndarray]:
    p_hpa = marina_saved_levels_hpa()
    arr = np.load(MARINA_SAVED_FWD_NPY).astype(float)
    if arr.shape[0] == len(p_hpa):
        arr = arr.T
    if arr.shape[1] != len(p_hpa):
        raise ValueError(f"Unexpected Marina saved FWD shape {arr.shape}; levels={len(p_hpa)}")
    return arr, p_hpa


def load_local_fwd_profiles() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ds = xr.open_dataset(LOCAL_FWD_NC, decode_times=False)
    try:
        fwd = ds["FWD_dayofyear"].load().values.astype(float)
        years = ds["year"].values.astype(int)
        if "plev_hpa" in ds:
            p_hpa = ds["plev_hpa"].values.astype(float)
        elif "plev_hpa" in ds["FWD_dayofyear"].coords:
            p_hpa = ds["FWD_dayofyear"]["plev_hpa"].values.astype(float)
        else:
            p_hpa = ds["plev"].values.astype(float) / 100.0
    finally:
        ds.close()
    return fwd, p_hpa, years


def summarize_diff(df: pd.DataFrame, diff_col: str) -> dict[str, float]:
    vals = pd.to_numeric(df[diff_col], errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {
            "n": 0,
            "bias_days": math.nan,
            "mae_days": math.nan,
            "rmse_days": math.nan,
            "max_abs_days": math.nan,
            "correlation": math.nan,
        }
    return {
        "n": int(vals.size),
        "bias_days": float(np.nanmean(vals)),
        "mae_days": float(np.nanmean(np.abs(vals))),
        "rmse_days": float(np.sqrt(np.nanmean(vals * vals))),
        "max_abs_days": float(np.nanmax(np.abs(vals))),
    }


def compare_fwd(mapping: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    marina_fwd, marina_p = load_marina_fwd_profiles()
    local_fwd, local_p, local_years = load_local_fwd_profiles()
    local_year_to_idx = {int(y): i for i, y in enumerate(local_years)}

    marina_50_idx = nearest_idx(marina_p, TARGET_FWD_HPA)
    local_50_idx = nearest_idx(local_p, TARGET_FWD_HPA)

    pair_rows = []
    level_rows = []
    for row in mapping.itertuples(index=False):
        marina_idx = int(row.marina_year) - 1
        local_idx = local_year_to_idx[int(row.our_year)]
        marina_50 = float(marina_fwd[marina_idx, marina_50_idx])
        local_50 = float(local_fwd[local_idx, local_50_idx])
        pair_rows.append(
            {
                **row._asdict(),
                "marina_fwd_50hpa_doy": marina_50,
                "our_fwd_50hpa_doy": local_50,
                "fwd_50hpa_diff_our_minus_marina_days": local_50 - marina_50,
                "fwd_50hpa_abs_diff_days": abs(local_50 - marina_50),
            }
        )

        for mp_idx, mp in enumerate(marina_p):
            lp_idx = nearest_idx(local_p, float(mp))
            if abs(local_p[lp_idx] - mp) > max(0.05, 0.01 * float(mp)):
                continue
            marina_val = float(marina_fwd[marina_idx, mp_idx])
            local_val = float(local_fwd[local_idx, lp_idx])
            level_rows.append(
                {
                    "pair_id": int(row.pair_id),
                    "chunk_id": int(row.chunk_id),
                    "marina_year": int(row.marina_year),
                    "our_year": int(row.our_year),
                    "plev_hpa": float(mp),
                    "marina_fwd_doy": marina_val,
                    "our_fwd_doy": local_val,
                    "diff_our_minus_marina_days": local_val - marina_val,
                    "abs_diff_days": abs(local_val - marina_val),
                }
            )

    pairs = pd.DataFrame(pair_rows)
    by_level = pd.DataFrame(level_rows)

    summary_rows = []
    if not pairs.empty:
        s = summarize_diff(pairs, "fwd_50hpa_diff_our_minus_marina_days")
        corr = corr_1d(pairs["marina_fwd_50hpa_doy"].to_numpy(), pairs["our_fwd_50hpa_doy"].to_numpy())
        summary_rows.append({"group": "all_pairs_50hpa", "plev_hpa": TARGET_FWD_HPA, **s, "correlation": corr})

        for chunk_id, group in pairs.groupby("chunk_id"):
            s = summarize_diff(group, "fwd_50hpa_diff_our_minus_marina_days")
            corr = corr_1d(group["marina_fwd_50hpa_doy"].to_numpy(), group["our_fwd_50hpa_doy"].to_numpy())
            summary_rows.append(
                {
                    "group": f"chunk_{int(chunk_id)}_50hpa",
                    "plev_hpa": TARGET_FWD_HPA,
                    **s,
                    "correlation": corr,
                }
            )

    if not by_level.empty:
        for level, group in by_level.groupby("plev_hpa"):
            s = summarize_diff(group, "diff_our_minus_marina_days")
            corr = corr_1d(group["marina_fwd_doy"].to_numpy(), group["our_fwd_doy"].to_numpy())
            summary_rows.append(
                {
                    "group": "all_pairs_by_level",
                    "plev_hpa": float(level),
                    **s,
                    "correlation": corr,
                }
            )

    return pairs, by_level, pd.DataFrame(summary_rows)


def save_figures(pairs: pd.DataFrame, summary: pd.DataFrame, scores: pd.DataFrame, primary_metric: str) -> None:
    if pairs.empty:
        return

    all50 = summary[summary["group"].eq("all_pairs_50hpa")].iloc[0]
    colors = {1: "#1b9e77", 2: "#d95f02", 3: "#7570b3", 4: "#e7298a"}
    labels = {
        1: "chunk 1: 007 0005-0056",
        2: "chunk 2: 010 0057-0104",
        3: "chunk 3: 009 0005-0056",
        4: "chunk 4: 011 0057-0104",
    }

    fig, ax = plt.subplots(figsize=(6.4, 5.8), constrained_layout=True)
    for chunk_id, group in pairs.groupby("chunk_id"):
        ax.scatter(
            group["marina_fwd_50hpa_doy"],
            group["our_fwd_50hpa_doy"],
            s=28,
            alpha=0.82,
            color=colors.get(int(chunk_id), "0.4"),
            label=labels.get(int(chunk_id), f"chunk {chunk_id}"),
        )
    lo = min(pairs["marina_fwd_50hpa_doy"].min(), pairs["our_fwd_50hpa_doy"].min()) - 3
    hi = max(pairs["marina_fwd_50hpa_doy"].max(), pairs["our_fwd_50hpa_doy"].max()) + 3
    ax.plot([lo, hi], [lo, hi], color="0.25", lw=1.0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Marina saved 50 hPa FWD (DOY)")
    ax.set_ylabel("Local cleaned 50 hPa FWD (DOY)")
    ax.set_title("Feature-matched CLIM-3D yearly FWD")
    ax.text(
        0.03,
        0.97,
        (
            f"N={int(all50.n)}\n"
            f"MAE={all50.mae_days:.2f} d\n"
            f"bias={all50.bias_days:.2f} d\n"
            f"r={all50.correlation:.3f}"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="0.75", alpha=0.9),
    )
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.savefig(PLOT_DIR / "clim3d_feature_matched_fwd50_scatter.png", dpi=180)
    fig.savefig(PLOT_DIR / "clim3d_feature_matched_fwd50_scatter.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 3.8), constrained_layout=True)
    for chunk_id, group in pairs.groupby("chunk_id"):
        ax.scatter(
            group["pair_id"],
            group["fwd_50hpa_abs_diff_days"],
            s=18,
            color=colors.get(int(chunk_id), "0.4"),
            label=labels.get(int(chunk_id), f"chunk {chunk_id}"),
        )
    ax.axhline(float(all50.mae_days), color="0.15", lw=1, ls="--", label=f"mean abs = {all50.mae_days:.2f} d")
    ax.set_xlabel("Mapped pair id")
    ax.set_ylabel("|local - Marina| FWD at 50 hPa (days)")
    ax.set_title("FWD mismatch after U-field year matching")
    ax.legend(loc="upper right", fontsize=8, ncol=2, frameon=False)
    fig.savefig(PLOT_DIR / "clim3d_feature_matched_fwd50_abs_error.png", dpi=180)
    fig.savefig(PLOT_DIR / "clim3d_feature_matched_fwd50_abs_error.svg")
    plt.close(fig)

    metric_top = (
        scores[scores["metric"].eq(primary_metric)]
        .sort_values(["chunk_id", "correlation"], ascending=[True, False])
        .groupby("chunk_id", as_index=False)
        .head(8)
        .copy()
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.2), constrained_layout=True, sharey=True)
    for ax, (chunk_id, group) in zip(axes.ravel(), metric_top.groupby("chunk_id")):
        group = group.sort_values("correlation", ascending=True)
        ax.barh(
            group["local_start_year"].astype(str) + "-" + group["local_end_year"].astype(str),
            group["correlation"],
            color=colors.get(int(chunk_id), "0.4"),
            alpha=0.82,
        )
        ax.set_title(labels.get(int(chunk_id), f"chunk {chunk_id}"), fontsize=10)
        ax.set_xlabel(f"{primary_metric} fingerprint correlation")
        ax.set_xlim(max(0.0, float(group["correlation"].min()) - 0.02), 1.001)
    fig.savefig(PLOT_DIR / f"clim3d_{primary_metric.lower()}_fingerprint_top_windows.png", dpi=180)
    fig.savefig(PLOT_DIR / f"clim3d_{primary_metric.lower()}_fingerprint_top_windows.svg")
    plt.close(fig)


def write_figure8_variant_inventory() -> pd.DataFrame:
    rows = []
    for path in sorted((LONGRUN_DIR / "Visualization" / "plots").glob("figure*_ozone_wind_epfdiv_rm5_*.*")):
        if "doubar_omega" in path.name:
            variant = "DO_UBAR_plus_omega"
        elif "no_doubar" in path.name:
            variant = "legacy_no_DO_UBAR_no_omega"
        else:
            variant = "unknown"
        rows.append(
            {
                "figure_file": str(path.relative_to(LONGRUN_DIR)),
                "variant": variant,
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else 0,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "figure8_epflux_variant_inventory.csv", index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force-feature-cache",
        action="store_true",
        help="Recompute U/O3 feature arrays even when cached .npy files exist.",
    )
    parser.add_argument(
        "--primary-feature",
        choices=["o3", "u"],
        default="o3",
        help=(
            "Feature used to choose the year mapping. The default O3 path uses "
            "existing compact products; u reads the large yearly U files."
        ),
    )
    args = parser.parse_args()
    primary_metric = "U10_daily" if args.primary_feature == "u" else "O3_daily"

    ensure_dirs()
    chunks = parse_marina_history_chunks()
    marina, local, local_table = load_feature_bundles(
        force=args.force_feature_cache,
        include_u=args.primary_feature == "u",
    )
    mapping, scores = build_feature_mapping(chunks, marina, local, primary_metric=primary_metric)
    mapping = add_local_source_info(mapping, local_table)
    pairs, by_level, summary = compare_fwd(mapping)
    fig8_inventory = write_figure8_variant_inventory()

    chunks.to_csv(OUT_DIR / "marina_merge_history_chunks.csv", index=False)
    local_table.to_csv(OUT_DIR / "local_year_source_table.csv", index=False)
    mapping.to_csv(OUT_DIR / "feature_matched_year_mapping.csv", index=False)
    scores.to_csv(OUT_DIR / "feature_chunk_sliding_window_scores.csv", index=False)
    pairs.to_csv(OUT_DIR / "feature_matched_fwd50_by_pair.csv", index=False)
    by_level.to_csv(OUT_DIR / "feature_matched_fwd_by_level_pair.csv", index=False)
    summary.to_csv(OUT_DIR / "feature_matched_fwd_summary.csv", index=False)
    save_figures(pairs, summary, scores, primary_metric=primary_metric)

    all50 = summary[summary["group"].eq("all_pairs_50hpa")].iloc[0]
    best_windows = (
        scores[scores["metric"].eq(primary_metric)]
        .sort_values(["chunk_id", "correlation"], ascending=[True, False])
        .groupby("chunk_id", as_index=False)
        .head(1)
    )

    print("CLIM-3D feature-based year mapping test")
    if primary_metric == "U10_daily":
        print("Primary match: daily U at about 10 hPa, 55-75N, days 1-181")
    else:
        print("Primary match: daily polar O3, 60-90N, March-May")
    print("\nMarina merge-history chunks:")
    print(chunks.to_string(index=False))
    print(f"\nBest local windows from {primary_metric} fingerprint:")
    print(best_windows.to_string(index=False))
    print("\nFWD 50 hPa summary after feature matching:")
    print(
        summary[summary["group"].str.contains("50hpa")]
        .round(3)
        .to_string(index=False)
    )
    print("\nRun/source-year checks in the chosen mapping:")
    print(
        mapping[["run_matches_history", "source_year_matches_history"]]
        .mean(numeric_only=False)
        .to_string()
    )
    print("\nFigure-8 EP-flux variant inventory:")
    print(fig8_inventory.to_string(index=False))
    print("\nKey result:")
    print(
        f"N={int(all50.n)} matched years; 50 hPa FWD MAE={all50.mae_days:.2f} d, "
        f"bias={all50.bias_days:.2f} d, r={all50.correlation:.3f}."
    )


if __name__ == "__main__":
    main()
