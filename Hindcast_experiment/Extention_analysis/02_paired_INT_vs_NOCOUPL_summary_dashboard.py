#!/usr/bin/env python
"""Build one dashboard summary for 02 paired INT-vs-NOCOUPL evolution.

The original 02 notebook intentionally writes one detailed evolution figure per
paired case.  This script does not change that workflow.  It reads the existing
paired evolution support CSVs and summarizes them into one cross-case figure.

Metrics
-------
Panel A, signed pathway difference:
    mean(H INT-3D ensemble mean - H Clim 3D ensemble mean)
    for early and Apr-May windows.  Early is Feb-Mar for February initializations
    and March only for March initializations.

Panel B, skill difference:
    RMSE(H INT-3D ensemble mean, BWCN reference)
    - RMSE(H Clim 3D ensemble mean, BWCN reference)
    over initialization through May 30.  Negative means INT is closer to BWCN.

Panel C, spread difference:
    mean(H INT-3D ensemble standard deviation)
    - mean(H Clim 3D ensemble standard deviation)
    over initialization through May 30.  Positive means INT has larger spread.

Outputs
-------
Figures:
    outputs/figures/02_evolution/paired_INT_vs_NOCOUPL/summary/
Tables:
    outputs/tables/02_evolution/paired_INT_vs_NOCOUPL/summary/
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hindcast_extension_utils import (
    MONTH_ABBR,
    TAB_DIR,
    figure_dir,
    mmdd_to_doy,
    savefig,
    table_dir,
)


VERSION_TAG = "v2"
NOTEBOOK_SOURCE = "02_paired_INT_vs_NOCOUPL_summary_dashboard.py"

SUMMARY_DIR = table_dir("02_evolution", "paired_INT_vs_NOCOUPL", "summary")
FIGURE_DIR = figure_dir("02_evolution", "paired_INT_vs_NOCOUPL", "summary")
PAIR_SUMMARY_PATTERN = "02_paired_INT_vs_NOCOUPL_evolution_summary_v*.csv"

VARIABLE_LABELS = {
    "O3": "O3",
    "U60N1": "U1",
    "U60N10": "U10",
    "U60N50": "U50",
}

VARIABLE_UNITS = {
    "O3": "DU",
    "U60N1": "m/s",
    "U60N10": "m/s",
    "U60N50": "m/s",
}


def _version_number(path: Path) -> int:
    match = re.search(r"_v(\d+)\.csv$", path.name)
    return int(match.group(1)) if match else -1


def latest_pair_summary() -> Path:
    """Return the newest 02 paired evolution summary CSV."""
    candidates = sorted((TAB_DIR / "02_evolution").glob(PAIR_SUMMARY_PATTERN), key=_version_number)
    if not candidates:
        raise FileNotFoundError(f"No paired evolution summary matching {PAIR_SUMMARY_PATTERN}")
    return candidates[-1]


def _display_case(int_case: str) -> str:
    """Compact row label for a paired hindcast case."""
    match = re.match(r"(?P<year>\d{4})-(?P<month>\d{2})", str(int_case))
    if not match:
        return str(int_case)
    year = match.group("year")[-2:]
    month = MONTH_ABBR[int(match.group("month")) - 1]
    return f"{year}-{month}"


def _window_defs(init_month: int) -> dict[str, tuple[int, int]]:
    """Return early, late, and full windows in day-of-year coordinates."""
    init_start = mmdd_to_doy(init_month, 1)
    if init_month <= 2:
        early = (mmdd_to_doy(2, 1), mmdd_to_doy(3, 31))
        early_label = "Feb-Mar"
    elif init_month == 3:
        early = (mmdd_to_doy(3, 1), mmdd_to_doy(3, 31))
        early_label = "Mar"
    else:
        early = (init_start, min(mmdd_to_doy(init_month, 30), mmdd_to_doy(5, 30)))
        early_label = f"{MONTH_ABBR[init_month - 1]}"
    return {
        "early": early,
        "late": (mmdd_to_doy(4, 1), mmdd_to_doy(5, 30)),
        "full": (init_start, mmdd_to_doy(5, 30)),
        "early_label": early_label,
    }


def _value_column(df: pd.DataFrame, source: str) -> str:
    """Return the numeric value column for forecast or reference rows."""
    return "ensemble_mean" if source in {"H INT-3D", "H Clim 3D"} else "value"


def _series(df: pd.DataFrame, source: str, variable: str, start: int, end: int, column: str | None = None) -> pd.Series:
    """Extract a source-variable time series for one day-of-year window."""
    col = column or _value_column(df, source)
    sub = df[
        (df["source"] == source)
        & (df["variable"] == variable)
        & (df["doy"].between(start, end))
    ][["doy", col]].dropna()
    if sub.empty:
        return pd.Series(dtype=float)
    return sub.groupby("doy")[col].mean().sort_index()


def _aligned(a: pd.Series, b: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    idx = a.index.intersection(b.index)
    if len(idx) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    x = a.loc[idx].to_numpy(dtype=float)
    y = b.loc[idx].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _mean_diff(df: pd.DataFrame, variable: str, start: int, end: int) -> float:
    int_s = _series(df, "H INT-3D", variable, start, end)
    clim_s = _series(df, "H Clim 3D", variable, start, end)
    x, y = _aligned(int_s, clim_s)
    return float(np.nanmean(x - y)) if len(x) else np.nan


def _rmse_diff(df: pd.DataFrame, variable: str, start: int, end: int) -> tuple[float, float, float]:
    ref_s = _series(df, "Reference", variable, start, end)
    int_s = _series(df, "H INT-3D", variable, start, end)
    clim_s = _series(df, "H Clim 3D", variable, start, end)
    int_x, ref_i = _aligned(int_s, ref_s)
    clim_x, ref_c = _aligned(clim_s, ref_s)
    int_rmse = float(np.sqrt(np.nanmean((int_x - ref_i) ** 2))) if len(int_x) else np.nan
    clim_rmse = float(np.sqrt(np.nanmean((clim_x - ref_c) ** 2))) if len(clim_x) else np.nan
    diff = int_rmse - clim_rmse if np.isfinite(int_rmse) and np.isfinite(clim_rmse) else np.nan
    return diff, int_rmse, clim_rmse


def _spread_diff(df: pd.DataFrame, variable: str, start: int, end: int) -> tuple[float, float, float]:
    int_s = _series(df, "H INT-3D", variable, start, end, column="ensemble_std")
    clim_s = _series(df, "H Clim 3D", variable, start, end, column="ensemble_std")
    x, y = _aligned(int_s, clim_s)
    int_spread = float(np.nanmean(x)) if len(x) else np.nan
    clim_spread = float(np.nanmean(y)) if len(y) else np.nan
    diff = int_spread - clim_spread if np.isfinite(int_spread) and np.isfinite(clim_spread) else np.nan
    return diff, int_spread, clim_spread


def build_metrics() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build long metric table and three wide matrices for plotting."""
    pair_summary_path = latest_pair_summary()
    pair_summary = pd.read_csv(pair_summary_path)
    rows: list[dict] = []
    for _, pair in pair_summary[pair_summary["status"].eq("plotted")].iterrows():
        csv_path = Path(pair["table"])
        if not csv_path.exists():
            continue
        int_case = str(pair["int_case"])
        noc_case = str(pair["nocoupl_case"])
        init_month = int(pair["init_month"])
        windows = _window_defs(init_month)
        df = pd.read_csv(csv_path)
        variables = [v for v in ["O3", "U60N1", "U60N10", "U60N50"] if v in set(df["variable"])]
        row_base = {
            "row_label": _display_case(int_case),
            "int_case": int_case,
            "nocoupl_case": noc_case,
            "year": str(pair["year"]).zfill(4),
            "init_month": f"{init_month:02d}",
            "early_window_label": windows["early_label"],
            "input_table": str(csv_path),
        }

        for variable in variables:
            for window_key, metric_label in [("early", "Early"), ("late", "Apr-May")]:
                start, end = windows[window_key]
                actual_label = windows["early_label"] if window_key == "early" else "Apr-May"
                diff = _mean_diff(df, variable, start, end)
                rows.append({
                    **row_base,
                    "metric_family": "mean_difference",
                    "metric": f"{VARIABLE_LABELS[variable]} {metric_label}",
                    "variable": variable,
                    "window": actual_label,
                    "display_window": metric_label,
                    "start_doy": start,
                    "end_doy": end,
                    "value": diff,
                    "abs_value": abs(diff) if np.isfinite(diff) else np.nan,
                    "units": VARIABLE_UNITS[variable],
                    "definition": "mean(H INT-3D ensemble mean - H Clim 3D ensemble mean)",
                })

            start, end = windows["full"]
            rmse_diff, int_rmse, clim_rmse = _rmse_diff(df, variable, start, end)
            rows.append({
                **row_base,
                "metric_family": "rmse_difference",
                "metric": f"{VARIABLE_LABELS[variable]} init-May",
                "variable": variable,
                "window": "init-May",
                "start_doy": start,
                "end_doy": end,
                "value": rmse_diff,
                "abs_value": abs(rmse_diff) if np.isfinite(rmse_diff) else np.nan,
                "int_value": int_rmse,
                "clim_value": clim_rmse,
                "units": VARIABLE_UNITS[variable],
                "definition": "RMSE(H INT-3D ensemble mean, BWCN reference) - RMSE(H Clim 3D ensemble mean, BWCN reference)",
            })
            spread_diff, int_spread, clim_spread = _spread_diff(df, variable, start, end)
            rows.append({
                **row_base,
                "metric_family": "spread_difference",
                "metric": f"{VARIABLE_LABELS[variable]} init-May",
                "variable": variable,
                "window": "init-May",
                "start_doy": start,
                "end_doy": end,
                "value": spread_diff,
                "abs_value": abs(spread_diff) if np.isfinite(spread_diff) else np.nan,
                "int_value": int_spread,
                "clim_value": clim_spread,
                "units": VARIABLE_UNITS[variable],
                "definition": "mean(H INT-3D ensemble std) - mean(H Clim 3D ensemble std)",
            })

    metrics = pd.DataFrame(rows)
    if metrics.empty:
        raise RuntimeError("No paired evolution metrics were generated.")

    order = list(dict.fromkeys(metrics["row_label"].tolist()))
    metrics["row_label"] = pd.Categorical(metrics["row_label"], categories=order, ordered=True)

    def matrix(family: str, metric_order: list[str]) -> pd.DataFrame:
        sub = metrics[metrics["metric_family"].eq(family)].copy()
        mat = sub.pivot_table(index="row_label", columns="metric", values="value", observed=False)
        cols = [c for c in metric_order if c in mat.columns]
        return mat.reindex(index=order)[cols]

    mean_order = [
        "O3 Early", "O3 Apr-May",
        "U1 Early", "U1 Apr-May",
        "U10 Early", "U10 Apr-May",
        "U50 Early", "U50 Apr-May",
    ]
    rmse_order = ["O3 init-May", "U1 init-May", "U10 init-May", "U50 init-May"]
    mean_mat = matrix("mean_difference", mean_order)
    rmse_mat = matrix("rmse_difference", rmse_order)
    spread_mat = matrix("spread_difference", rmse_order)
    return metrics, mean_mat, rmse_mat, spread_mat


def _column_normalized(mat: pd.DataFrame) -> np.ndarray:
    arr = mat.to_numpy(dtype=float)
    out = np.full_like(arr, np.nan, dtype=float)
    for j in range(arr.shape[1]):
        col = arr[:, j]
        vmax = np.nanmax(np.abs(col)) if np.isfinite(col).any() else np.nan
        if np.isfinite(vmax) and vmax > 0:
            out[:, j] = col / vmax
    return out


def _fmt_value(value: float) -> str:
    if not np.isfinite(value):
        return ""
    if abs(value) < 0.005:
        value = 0.0
    if abs(value) < 0.1:
        return f"{value:+.2f}"
    if abs(value) < 10:
        return f"{value:+.1f}"
    return f"{value:+.0f}"


def _set_plot_style() -> None:
    """Use a compact journal-style matplotlib theme for the dashboard."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "0.15",
        "axes.linewidth": 0.6,
        "axes.titlesize": 8.5,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.9,
        "ytick.labelsize": 7.2,
        "font.size": 7.2,
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _pretty_columns(columns: pd.Index) -> list[str]:
    labels = []
    for col in columns:
        parts = str(col).split(" ", 1)
        labels.append("\n".join(parts) if len(parts) == 2 else str(col))
    return labels


def _plot_matrix(ax, mat: pd.DataFrame, title: str, panel: str) -> None:
    normed = _column_normalized(mat)
    im = ax.imshow(normed, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title(title, loc="left", pad=7, fontweight="bold")
    ax.text(
        -0.055,
        1.08,
        panel,
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        va="bottom",
        ha="right",
    )
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels(_pretty_columns(mat.columns), rotation=0, ha="center")
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(mat.index)
    ax.set_xticks(np.arange(-0.5, mat.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, mat.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.9)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_color("0.2")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            value = mat.iloc[i, j]
            if np.isfinite(value):
                ax.text(j, i, _fmt_value(value), ha="center", va="center", fontsize=6.7, color="black")
    return im


def plot_dashboard(metrics: pd.DataFrame, mean_mat: pd.DataFrame, rmse_mat: pd.DataFrame, spread_mat: pd.DataFrame) -> Path:
    """Plot and save the paired INT-vs-NOCOUPL dashboard."""
    _set_plot_style()
    csv = SUMMARY_DIR / f"02_paired_INT_vs_NOCOUPL_summary_dashboard_{VERSION_TAG}.csv"
    metrics.to_csv(csv, index=False)
    mean_mat.to_csv(SUMMARY_DIR / f"02_paired_INT_vs_NOCOUPL_mean_difference_matrix_{VERSION_TAG}.csv")
    rmse_mat.to_csv(SUMMARY_DIR / f"02_paired_INT_vs_NOCOUPL_rmse_difference_matrix_{VERSION_TAG}.csv")
    spread_mat.to_csv(SUMMARY_DIR / f"02_paired_INT_vs_NOCOUPL_spread_difference_matrix_{VERSION_TAG}.csv")

    nrows = max(len(mean_mat), 1)
    fig_height = max(6.9, 2.2 + nrows * 0.40)
    fig = plt.figure(figsize=(7.35, fig_height))
    gs = fig.add_gridspec(
        3,
        2,
        width_ratios=[1.0, 0.027],
        height_ratios=[1.18, 1.0, 1.0],
        left=0.11,
        right=0.91,
        bottom=0.17,
        top=0.90,
        hspace=0.62,
        wspace=0.035,
    )
    axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    cax = fig.add_subplot(gs[:, 1])
    im = _plot_matrix(
        axes[0],
        mean_mat,
        "Pathway shift: H INT-3D minus H Clim 3D",
        "a",
    )
    _plot_matrix(
        axes[1],
        rmse_mat,
        "Skill shift: RMSE difference relative to BWCN",
        "b",
    )
    _plot_matrix(
        axes[2],
        spread_mat,
        "Ensemble-spread shift",
        "c",
    )
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Column-normalized\nsigned difference", fontsize=7.0)
    cbar.ax.tick_params(labelsize=6.8, length=2.5, width=0.5)
    fig.suptitle("Paired INT-3D minus Clim-3D evolution summary", fontsize=9.6, fontweight="bold", y=0.958)
    fig.text(
        0.11,
        0.103,
        "Annotated numbers are physical values. O3: 60-90N, 30-70 hPa (DU); winds: U60N at 1, 10 and 50 hPa (m s$^{-1}$). "
        "Early = Feb-Mar for February starts and March for March starts; Apr-May is fixed; RMSE/spread use initialization-May30.",
        ha="left",
        va="top",
        fontsize=6.55,
        color="0.25",
        wrap=True,
    )
    fig.text(
        0.11,
        0.060,
        "Reading: negative RMSE shift means INT-3D is closer to BWCN; positive spread shift means INT-3D has larger ensemble spread. "
        "Color intensity is normalized within each column because columns mix DU and m s$^{-1}$.",
        ha="left",
        va="top",
        fontsize=6.55,
        color="0.25",
        wrap=True,
    )
    stem = f"02_paired_INT_vs_NOCOUPL_summary_dashboard_{VERSION_TAG}"
    savefig(
        fig,
        stem,
        fig_dir=FIGURE_DIR,
        notebook=NOTEBOOK_SOURCE,
        scientific_question="Can paired INT-vs-CLIM hindcasts be summarized as cross-case pathway, skill, and spread differences instead of many separate evolution panels?",
        variables_windows=(
            "Mean difference windows: Feb-Mar for Feb initialized cases, Mar for Mar initialized cases, and Apr-May. "
            "RMSE/spread windows: initialization date through May 30. "
            "O3=60-90N 30-70hPa; U=60N zonal-mean winds at 1/10/50 hPa."
        ),
        interpretation=(
            "Panel A shows the signed pathway shift. Panel B shows whether INT or CLIM is closer to BWCN reference "
            "(negative favors INT). Panel C shows whether INT or CLIM has larger ensemble spread."
        ),
        caveat="The dashboard compares ensemble statistics, not one-to-one member pairs. Color intensity is column-normalized because units differ across O3 and wind metrics; annotated numbers are the physical values.",
        csv_table=csv,
    )
    plt.close(fig)
    return csv


def main() -> None:
    metrics, mean_mat, rmse_mat, spread_mat = build_metrics()
    csv = plot_dashboard(metrics, mean_mat, rmse_mat, spread_mat)
    print(f"[WRITE] {csv}")
    print(metrics.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
