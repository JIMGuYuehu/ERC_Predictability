#!/usr/bin/env python3
"""Plot SWOOSH target before/after missing-value filling."""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from check_swoosh_missing import DEFAULT_TARGET
from fill_swoosh_target import default_output_path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_FILLED = default_output_path(DEFAULT_TARGET)
DEFAULT_PLOT = (
    SCRIPT_DIR
    / "plots"
    / "O3_NHpolar_2020_original_vs_filled_target_ppmv.png"
)


def infer_month_datetimes(time_values: np.ndarray) -> list[datetime]:
    dates: list[datetime] = []
    for raw in time_values:
        month_index = int(np.floor(float(raw)))
        year = 1850 + month_index // 12
        month = month_index % 12 + 1
        dates.append(datetime(year, month, 15))
    return dates


def weighted_polar_mean(
    da: xr.DataArray,
    lat_min: float,
    lat_max: float,
    plev_min: float,
    plev_max: float,
) -> xr.DataArray:
    subset = da.where(
        (da["lat"] >= lat_min)
        & (da["lat"] <= lat_max)
        & (da["plev"] >= plev_min)
        & (da["plev"] <= plev_max),
        drop=True,
    )
    weights = np.cos(np.deg2rad(subset["lat"]))
    mean = subset.weighted(weights).mean(dim=("lat", "lon"), skipna=True)
    return mean.transpose("plev", "time")


def nice_range(values: np.ndarray, lower: float = 2.0, upper: float = 98.0) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.nanpercentile(finite, [lower, upper])
    if np.isclose(vmin, vmax):
        pad = abs(vmin) * 0.05 if vmin else 1.0
        return float(vmin - pad), float(vmax + pad)
    return float(vmin), float(vmax)


def plot_panel(
    ax: plt.Axes,
    x_dates: list[datetime],
    plev: np.ndarray,
    values: np.ndarray,
    title: str,
    levels: np.ndarray,
    cmap: str,
) -> matplotlib.contour.QuadContourSet:
    cf = ax.contourf(x_dates, plev, values, levels=levels, cmap=cmap, extend="both")
    contour = ax.contour(x_dates, plev, values, levels=levels[::3], colors="k", linewidths=0.5, alpha=0.75)
    ax.clabel(contour, contour.levels[::2], inline=True, fontsize=7, fmt="%.2g")
    ax.set_title(title, fontsize=13)
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    return cf


def make_plot(
    original_path: Path,
    filled_path: Path,
    output_path: Path,
    variable: str,
    lat_min: float,
    lat_max: float,
    plev_min: float,
    plev_max: float,
) -> Path:
    if not original_path.exists():
        raise FileNotFoundError(original_path)
    if not filled_path.exists():
        raise FileNotFoundError(f"{filled_path} does not exist; run fill_swoosh_target.py first")

    with xr.open_dataset(original_path, decode_times=False, mask_and_scale=True) as ds_original:
        with xr.open_dataset(filled_path, decode_times=False, mask_and_scale=True) as ds_filled:
            if variable not in ds_original:
                raise KeyError(f"{variable!r} not found in {original_path}")
            if variable not in ds_filled:
                raise KeyError(f"{variable!r} not found in {filled_path}")

            original = weighted_polar_mean(
                ds_original[variable], lat_min, lat_max, plev_min, plev_max
            )
            filled = weighted_polar_mean(
                ds_filled[variable], lat_min, lat_max, plev_min, plev_max
            )
            original_ppmv = original * 1.0e6
            filled_ppmv = filled * 1.0e6
            diff_ppmv = filled_ppmv - original_ppmv

            x_dates = infer_month_datetimes(ds_filled["time"].values)
            plev = filled_ppmv["plev"].values

            common_values = np.concatenate(
                [original_ppmv.values.ravel(), filled_ppmv.values.ravel()]
            )
            vmin, vmax = nice_range(common_values, lower=1.0, upper=99.0)
            common_levels = np.linspace(vmin, vmax, 25)

            diff_abs = np.nanpercentile(np.abs(diff_ppmv.values), 99.0)
            if not np.isfinite(diff_abs) or diff_abs <= 0:
                diff_abs = 0.01
            diff_levels = np.linspace(-float(diff_abs), float(diff_abs), 25)

            fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True, constrained_layout=True)
            cf0 = plot_panel(
                axes[0],
                x_dates,
                plev,
                original_ppmv.values,
                "Original target",
                common_levels,
                "viridis",
            )
            plot_panel(
                axes[1],
                x_dates,
                plev,
                filled_ppmv.values,
                "Filled target",
                common_levels,
                "viridis",
            )
            cf2 = plot_panel(
                axes[2],
                x_dates,
                plev,
                diff_ppmv.values,
                "Filled - original",
                diff_levels,
                "RdBu_r",
            )
            axes[0].set_ylabel("Pressure (hPa)")
            for ax in axes:
                ax.set_xlabel("Month")
            fig.colorbar(cf0, ax=axes[:2], orientation="vertical", shrink=0.86, label="O3 (ppmv)")
            fig.colorbar(cf2, ax=axes[2], orientation="vertical", shrink=0.86, label="Difference (ppmv)")

            fig.suptitle(
                f"NH polar-cap weighted mean O3 ({lat_min:.0f}-{lat_max:.0f}N), "
                f"{plev_min:g}-{plev_max:g} hPa, Dec 2019-Jan 2021",
                fontsize=15,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=180)
            plt.close(fig)

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--filled", type=Path, default=DEFAULT_FILLED)
    parser.add_argument("--output", type=Path, default=DEFAULT_PLOT)
    parser.add_argument("--var", default="vmro3")
    parser.add_argument("--lat-min", type=float, default=60.0)
    parser.add_argument("--lat-max", type=float, default=90.0)
    parser.add_argument("--plev-min", type=float, default=1.0)
    parser.add_argument("--plev-max", type=float, default=100.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = make_plot(
        original_path=args.original,
        filled_path=args.filled,
        output_path=args.output,
        variable=args.var,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        plev_min=args.plev_min,
        plev_max=args.plev_max,
    )
    print(f"Saved plot: {out}")


if __name__ == "__main__":
    main()
