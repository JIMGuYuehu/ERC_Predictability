from __future__ import annotations

import shutil
import sys
from contextlib import ExitStack
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.ticker import ScalarFormatter


REPO_ROOT = Path("/home/weiji/restart_exam/code_cleaned")
TEST_ROOT = REPO_ROOT / "Hindcast_experiment" / "TEST_TROPOS"

# Keep the split notebooks with the rest of the 0008-01 tropospheric analysis.
OUT_ROOT = TEST_ROOT / "outputs" / "0008-01"
FIG_BASE = OUT_ROOT / "figures"
TABLE_BASE = OUT_ROOT / "tables"
CACHE_BASE = OUT_ROOT / "cache"

OMEGA_TAG = "BWCN0008_omega_epflux_compare"
TABLE_DIR = TABLE_BASE / OMEGA_TAG
CACHE_DIR = CACHE_BASE / OMEGA_TAG
for directory in [FIG_BASE, TABLE_DIR, CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

AOSTOOLS_DIR = REPO_ROOT / "Longrun" / "date_treatment"
if str(AOSTOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(AOSTOOLS_DIR))
from aostools_functions import ComputeEPfluxDiv  # noqa: E402


DATA_ROOT = Path("/mnt/soclim0/public_data/weiji")
BWCN_ROOT = DATA_ROOT / "BWCN"
YEAR = 8
DATE_WINDOW = ((1, 1), (5, 30))
SUMMARY_EP_WINDOW = ((1, 20), (2, 10))
LAT_BAND = (40.0, 80.0)
PLEV_SUMMARY_PA = 10000.0
PLOT_PLEV_RANGE_HPA = (1.0, 100.0)

PLEV_STD_PA = np.array(
    [
        10,
        50,
        100,
        200,
        300,
        500,
        1000,
        2000,
        3000,
        5000,
        7000,
        10000,
        15000,
        20000,
        25000,
        30000,
        40000,
        50000,
        60000,
        70000,
        85000,
        92500,
        100000,
    ],
    dtype=float,
)

DATE_WINDOW_TOKEN = (
    f"M{DATE_WINDOW[0][0]:02d}{DATE_WINDOW[0][1]:02d}_"
    f"M{DATE_WINDOW[1][0]:02d}{DATE_WINDOW[1][1]:02d}"
)
OFFICIAL_WITH_OMEGA = (
    BWCN_ROOT
    / "EPflux_daily_ubar_wcorr"
    / "all_waves"
    / "EPFLUX_all_waves_24yr_time_plev_lat.nc"
)
NOOMEGA_CACHE = CACHE_DIR / f"BWCN{YEAR:04d}_EPFLUX_all_waves_ubar_noomega_{DATE_WINDOW_TOKEN}.nc"
COMBINED_CACHE = CACHE_DIR / f"BWCN{YEAR:04d}_omega_vs_noomega_combined_{DATE_WINDOW_TOKEN}.nc"
SUMMARY_CSV = TABLE_DIR / f"BWCN{YEAR:04d}_omega_vs_noomega_EPFlux_summary_{DATE_WINDOW_TOKEN}.csv"
LEGACY_NOOMEGA_CACHE = (
    TEST_ROOT
    / "outputs"
    / OMEGA_TAG
    / "cache"
    / f"BWCN{YEAR:04d}_EPFLUX_all_waves_ubar_noomega_{DATE_WINDOW_TOKEN}.nc"
)


plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.32,
    }
)


def figure_dir(notebook_stem: str, content_tag: str) -> Path:
    out = FIG_BASE / notebook_stem / content_tag
    out.mkdir(parents=True, exist_ok=True)
    return out


def open_cam_dataset(path: Path) -> xr.Dataset:
    try:
        coder = xr.coders.CFDatetimeCoder(use_cftime=True)
        return xr.open_dataset(path, decode_times=coder)
    except Exception:
        return xr.open_dataset(path, decode_times=True, use_cftime=True)


def date_parts(date_values):
    arr = np.asarray(date_values, dtype=np.int64)
    yy = arr // 10000
    mmdd = arr % 10000
    month = mmdd // 100
    day = mmdd % 100
    return yy, month, day


def date_mask_from_int(date_values, start=DATE_WINDOW[0], end=DATE_WINDOW[1], year=None):
    yy, mm, dd = date_parts(date_values)
    key = mm * 100 + dd
    start_key = start[0] * 100 + start[1]
    end_key = end[0] * 100 + end[1]
    if end_key < start_key:
        raise ValueError("Cross-year windows are not supported in this quick test.")
    mask = (key >= start_key) & (key <= end_key)
    if year is not None:
        mask = mask & (yy == int(year))
    return mask


def date_int_from_time(time_da: xr.DataArray) -> np.ndarray:
    year = np.asarray(time_da.dt.year.values, dtype=np.int64)
    month = np.asarray(time_da.dt.month.values, dtype=np.int64)
    day = np.asarray(time_da.dt.day.values, dtype=np.int64)
    return year * 10000 + month * 100 + day


def doy_from_date_int(date_values) -> np.ndarray:
    month_lengths = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=np.int64)
    _, month, day = date_parts(date_values)
    return np.array(
        [month_lengths[: int(m) - 1].sum() + int(d) for m, d in zip(month, day)],
        dtype=np.int64,
    )


def window_label(start_end):
    (sm, sd), (em, ed) = start_end
    names = {
        1: "Jan",
        2: "Feb",
        3: "Mar",
        4: "Apr",
        5: "May",
        6: "Jun",
        7: "Jul",
        8: "Aug",
        9: "Sep",
        10: "Oct",
        11: "Nov",
        12: "Dec",
    }
    return f"{names[sm]}{sd:02d}-{names[em]}{ed:02d}"


def coslat_mean(da: xr.DataArray, lat_range=LAT_BAND) -> xr.DataArray:
    lat = da["lat"]
    descending = float(lat.values[0]) > float(lat.values[-1])
    lo, hi = lat_range
    sub = da.sel(lat=slice(hi, lo) if descending else slice(lo, hi))
    weights = np.cos(np.deg2rad(sub["lat"])).clip(0, 1)
    return sub.weighted(weights.fillna(0)).mean("lat", skipna=True)


def compute_pressure_mid(ds: xr.Dataset) -> xr.DataArray:
    return (ds["hyam"] * ds["P0"] + ds["hybm"] * ds["PS"]).transpose("time", "lat", "lon", "lev")


def interp_profile_logp_4d(v_hyb: xr.DataArray, p_hyb: xr.DataArray, p_tgt_pa: np.ndarray) -> xr.DataArray:
    p_tgt_pa = np.asarray(p_tgt_pa, dtype=float)
    v_in = v_hyb.transpose("time", "lat", "lon", "lev")
    p_in = p_hyb.transpose("time", "lat", "lon", "lev")

    def _interp_col(vcol, pcol):
        vcol = np.asarray(vcol, dtype=float)
        pcol = np.asarray(pcol, dtype=float)
        mask = np.isfinite(vcol) & np.isfinite(pcol) & (pcol > 0)
        if mask.sum() < 2:
            return np.full(p_tgt_pa.shape, np.nan, dtype=float)
        p_use = pcol[mask]
        v_use = vcol[mask]
        order = np.argsort(p_use)
        return np.interp(
            np.log(p_tgt_pa),
            np.log(p_use[order]),
            v_use[order],
            left=np.nan,
            right=np.nan,
        )

    out = xr.apply_ufunc(
        _interp_col,
        v_in,
        p_in,
        input_core_dims=[["lev"], ["lev"]],
        output_core_dims=[["plev"]],
        vectorize=True,
        dask="allowed",
        output_dtypes=[float],
    )
    return out.assign_coords(plev=("plev", p_tgt_pa)).transpose("time", "plev", "lat", "lon")


def savefig(fig, fig_dir: Path, name: str):
    fig_dir.mkdir(parents=True, exist_ok=True)
    png = fig_dir / f"{name}.png"
    pdf = fig_dir / f"{name}.pdf"
    fig.savefig(png, dpi=260, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print("Saved:", png)
    return png, pdf


def compute_noomega_epflux(force=False) -> xr.Dataset:
    if NOOMEGA_CACHE.exists() and not force:
        print("Loading cached no-omega EPFlux:", NOOMEGA_CACHE)
        return xr.open_dataset(NOOMEGA_CACHE, decode_times=False).load()
    if LEGACY_NOOMEGA_CACHE.exists() and not force:
        print("Reusing legacy no-omega EPFlux cache:", LEGACY_NOOMEGA_CACHE)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(LEGACY_NOOMEGA_CACHE, NOOMEGA_CACHE)
        return xr.open_dataset(NOOMEGA_CACHE, decode_times=False).load()

    files = {
        "U": BWCN_ROOT / "U" / f"BWCN.cam.h3.{YEAR:04d}.U.nc",
        "V": BWCN_ROOT / "V" / f"BWCN.cam.h3.{YEAR:04d}.V.nc",
        "T": BWCN_ROOT / "T" / f"BWCN.cam.h3.{YEAR:04d}.T.nc",
    }
    missing = [str(v) for v in files.values() if not v.exists()]
    if missing:
        raise FileNotFoundError("Missing raw files for no-omega calculation: " + ", ".join(missing))

    with ExitStack() as stack:
        ds_u = stack.enter_context(open_cam_dataset(files["U"]))
        ds_v = stack.enter_context(open_cam_dataset(files["V"]))
        ds_t = stack.enter_context(open_cam_dataset(files["T"]))

        date = np.asarray(ds_u["date"].values, dtype=np.int64)
        mask = date_mask_from_int(date, start=DATE_WINDOW[0], end=DATE_WINDOW[1], year=YEAR)
        if mask.sum() == 0:
            raise ValueError("No raw BWCN dates selected by DATE_WINDOW.")

        ds_u = ds_u.isel(time=mask)
        ds_v = ds_v.isel(time=mask)
        ds_t = ds_t.isel(time=mask)
        date_sel = date[mask]
        p_mid = compute_pressure_mid(ds_u)

        print(f"Interpolating U/V/T for BWCN{YEAR:04d}, {window_label(DATE_WINDOW)}, n_days={mask.sum()}")
        u_std = interp_profile_logp_4d(ds_u["U"], p_mid, PLEV_STD_PA)
        v_std = interp_profile_logp_4d(ds_v["V"], p_mid, PLEV_STD_PA)
        t_std = interp_profile_logp_4d(ds_t["T"], p_mid, PLEV_STD_PA)

        u_np = u_std.values
        v_np = v_std.values
        t_np = t_std.values
        lat_np = np.asarray(ds_u["lat"].values, dtype=float)
        time_coord = ds_u["time"].load()

    print("Computing all-wave EPFlux without omega: DO_UBAR=True, w=None")
    ep1, ep2, div1, div2 = ComputeEPfluxDiv(
        lat=lat_np,
        pres=PLEV_STD_PA / 100.0,
        u=u_np,
        v=v_np,
        t=t_np,
        w=None,
        do_ubar=True,
        wave=-1,
    )

    ds_out = xr.Dataset(
        data_vars={
            "ep1_no_omega": (("time", "plev", "lat"), ep1),
            "ep2_no_omega": (("time", "plev", "lat"), ep2),
            "div1_no_omega": (("time", "plev", "lat"), div1),
            "div2_no_omega": (("time", "plev", "lat"), div2),
            "div_no_omega": (("time", "plev", "lat"), div1 + div2),
            "date": (("time",), date_sel),
        },
        coords={"time": time_coord, "plev": PLEV_STD_PA, "lat": lat_np},
        attrs={
            "description": "BWCN0008 all-wave EPFlux recomputed for omega-correction sensitivity test",
            "year": f"{YEAR:04d}",
            "date_window": str(DATE_WINDOW),
            "do_ubar": "True",
            "use_omega_w_correction": "False",
            "Fz_plot_definition": "Fz_up = -ep2; positive follows old upward-Fz plotting convention",
            "divergence_definition": "EPFDIV = div1 + div2",
            "units_ep2": "hPa*m/s2",
            "units_div": "m/s/day",
        },
    )
    ds_out["plev"].attrs.update({"units": "Pa", "positive": "down", "long_name": "pressure"})
    ds_out.to_netcdf(NOOMEGA_CACHE)
    print("Saved no-omega cache:", NOOMEGA_CACHE)
    return ds_out.load()


def load_official_withomega() -> xr.Dataset:
    if not OFFICIAL_WITH_OMEGA.exists():
        raise FileNotFoundError(OFFICIAL_WITH_OMEGA)
    with open_cam_dataset(OFFICIAL_WITH_OMEGA) as ds:
        date_int = date_int_from_time(ds["time"])
        mask = date_mask_from_int(date_int, start=DATE_WINDOW[0], end=DATE_WINDOW[1], year=YEAR)
        if mask.sum() == 0:
            raise ValueError("No official EPFlux times selected by DATE_WINDOW.")
        sub = ds.isel(time=mask).load()
        date_sel = date_int[mask]
    out = xr.Dataset(
        data_vars={
            "ep2_with_omega": sub["ep2"],
            "div1_with_omega": sub["div1"],
            "div2_with_omega": sub["div2"],
            "div_with_omega": sub["div1"] + sub["div2"],
            "date": (("time",), date_sel),
        },
        coords={"time": sub["time"], "plev": sub["plev"], "lat": sub["lat"]},
        attrs={
            "description": "BWCN0008 with-omega subset loaded from official EPflux_daily_ubar_wcorr all_waves file",
            "source_file": str(OFFICIAL_WITH_OMEGA),
            "do_ubar": sub.attrs.get("do_ubar", "unknown"),
            "use_omega_w_correction": sub.attrs.get("use_omega_w_correction", "unknown"),
        },
    )
    return out.load()


def load_combined_epflux(force_recompute_noomega=False, force_rebuild_combined=False) -> xr.Dataset:
    if COMBINED_CACHE.exists() and not force_recompute_noomega and not force_rebuild_combined:
        print("Loading combined omega comparison cache:", COMBINED_CACHE)
        return xr.open_dataset(COMBINED_CACHE, decode_times=False).load()

    noomega = compute_noomega_epflux(force=force_recompute_noomega)
    withomega = load_official_withomega()

    n = min(noomega.sizes["time"], withomega.sizes["time"])
    noomega = noomega.isel(time=slice(0, n))
    if "date" in noomega.data_vars:
        noomega = noomega.set_coords("date")
    withomega = withomega.isel(time=slice(0, n))
    if "date" in withomega.data_vars:
        withomega = withomega.drop_vars("date")
    withomega = withomega.assign_coords(time=noomega["time"], date=("time", noomega["date"].values))
    combined = xr.merge([noomega, withomega], compat="override")
    combined.attrs.update(
        {
            "description": "BWCN0008 with-omega and no-omega all-wave EPFlux comparison subset",
            "date_window": str(DATE_WINDOW),
            "source_with_omega": str(OFFICIAL_WITH_OMEGA),
            "source_no_omega": str(NOOMEGA_CACHE),
        }
    )
    combined.to_netcdf(COMBINED_CACHE)
    print("Saved combined cache:", COMBINED_CACHE)
    return combined.load()


def reduce_time_plev(da: xr.DataArray) -> xr.DataArray:
    return coslat_mean(da, LAT_BAND).load()


def select_summary_metric(
    combined: xr.Dataset,
    da: xr.DataArray,
    start_end=SUMMARY_EP_WINDOW,
    plev_pa=PLEV_SUMMARY_PA,
) -> xr.DataArray:
    date = np.asarray(combined["date"].values, dtype=np.int64)
    mask = date_mask_from_int(date, start=start_end[0], end=start_end[1], year=YEAR)
    return reduce_time_plev(da.sel(plev=plev_pa, method="nearest")).isel(time=mask).mean("time", skipna=True)


def build_summary(combined: xr.Dataset, save=True) -> pd.DataFrame:
    summary_rows = []
    fields = [
        (
            "Fz_up=-ep2",
            -combined["ep2_no_omega"],
            -combined["ep2_with_omega"],
            -(combined["ep2_with_omega"] - combined["ep2_no_omega"]),
            "hPa m s-2",
        ),
        (
            "EPFDIV=div1+div2",
            combined["div_no_omega"],
            combined["div_with_omega"],
            combined["div_with_omega"] - combined["div_no_omega"],
            "m s-1 day-1",
        ),
    ]
    for name, no_da, yes_da, _diff_da, units in fields:
        no_metric = select_summary_metric(combined, no_da)
        yes_metric = select_summary_metric(combined, yes_da)
        diff_metric = yes_metric - no_metric
        rel = float(abs(diff_metric) / max(abs(float(yes_metric)), 1e-30) * 100.0)
        summary_rows.append(
            {
                "quantity": name,
                "summary_window": window_label(SUMMARY_EP_WINDOW),
                "pressure_hPa": PLEV_SUMMARY_PA / 100.0,
                "lat_band": f"{LAT_BAND[0]:.0f}-{LAT_BAND[1]:.0f}N",
                "no_omega_mean": float(no_metric),
                "with_omega_mean": float(yes_metric),
                "with_minus_no": float(diff_metric),
                "relative_to_with_omega_percent": rel,
                "units": units,
            }
        )
    summary = pd.DataFrame(summary_rows)
    if save:
        summary.to_csv(SUMMARY_CSV, index=False)
        print("Saved:", SUMMARY_CSV)
    return summary


def plot_level_slice(da: xr.DataArray) -> xr.DataArray:
    return da.where(
        (da["plev"] >= PLOT_PLEV_RANGE_HPA[0] * 100.0)
        & (da["plev"] <= PLOT_PLEV_RANGE_HPA[1] * 100.0),
        drop=True,
    )


def plot_omega_vertical_compare(
    combined: xr.Dataset,
    summary: pd.DataFrame,
    fig_dir: Path,
    fig_name: str | None = None,
):
    fz_no = reduce_time_plev(-combined["ep2_no_omega"])
    fz_yes = reduce_time_plev(-combined["ep2_with_omega"])
    fz_diff = fz_yes - fz_no

    div_no = reduce_time_plev(combined["div_no_omega"])
    div_yes = reduce_time_plev(combined["div_with_omega"])
    div_diff = div_yes - div_no

    x = doy_from_date_int(combined["date"].values)
    fig, axes = plt.subplots(2, 3, figsize=(15.4, 8.2), constrained_layout=True, sharex=True, sharey=True)
    rows = [
        ("Fz_up = -ep2", fz_no, fz_yes, fz_diff, "hPa m s$^{-2}$", "RdBu_r"),
        ("EPFDIV = div1 + div2", div_no, div_yes, div_diff, "m s$^{-1}$ day$^{-1}$", "RdBu_r"),
    ]
    col_titles = ["No omega", "With omega", "With - no"]
    for r, (row_title, no_da, yes_da, diff_da, units, cmap) in enumerate(rows):
        no_plot = plot_level_slice(no_da)
        yes_plot = plot_level_slice(yes_da)
        diff_plot = plot_level_slice(diff_da)
        pair = np.concatenate([no_plot.values.ravel(), yes_plot.values.ravel()])
        vlim = np.nanpercentile(np.abs(pair), 98)
        levels = np.linspace(-vlim, vlim, 21)
        row_cf = None
        for c, da in enumerate([no_plot, yes_plot, diff_plot]):
            ax = axes[r, c]
            y = da["plev"].values / 100.0
            field = da.transpose("plev", "time").values
            row_cf = ax.contourf(x, y, field, levels=levels, cmap=cmap, extend="both")
            ax.contour(x, y, field, levels=levels[::4], colors="0.25", linewidths=0.35, alpha=0.45)
            ax.set_yscale("log")
            ax.set_ylim(PLOT_PLEV_RANGE_HPA[1], PLOT_PLEV_RANGE_HPA[0])
            ax.set_yticks([100, 70, 50, 30, 20, 10, 5, 2, 1])
            ax.set_title(f"{row_title}: {col_titles[c]}", fontsize=10)
            ax.set_xlabel("Day of year")
            if c == 0:
                ax.set_ylabel("Pressure (hPa)")
            ax.yaxis.set_major_formatter(ScalarFormatter())
        cbar = fig.colorbar(row_cf, ax=axes[r, :], shrink=0.86, pad=0.015)
        cbar.set_label(units, fontsize=8)
        cbar.ax.tick_params(labelsize=8)

    summary_text = "100 hPa, 40-80N, Jan20-Feb10: "
    summary_text += "; ".join(
        [
            f"{row['quantity']} Δ={row['with_minus_no']:.3g} "
            f"({row['relative_to_with_omega_percent']:.1f}% of with-omega)"
            for _, row in summary.iterrows()
        ]
    )
    fig.suptitle(
        f"BWCN{YEAR:04d} all-wave EPFlux omega-correction sensitivity "
        f"({window_label(DATE_WINDOW)}, 40-80N mean, 1-100 hPa shown)\n{summary_text}",
        fontsize=11,
    )
    if fig_name is None:
        fig_name = f"BWCN{YEAR:04d}_omega_vs_noomega_EPFlux_Fz_divergence_{DATE_WINDOW_TOKEN}"
    return savefig(fig, fig_dir, fig_name)
