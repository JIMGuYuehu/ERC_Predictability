#!/usr/bin/env python
"""Generate figures and tables for the CLIM-3D Marina reproducibility report."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fwd_clim3d_low25_source_test as src


ROOT = Path("/home/weiji/restart_exam/code_cleaned")
PLOT_DIR = ROOT / "Longrun/Visualization/plots/clim3d_marina_repro"
TABLE_DIR = ROOT / "Longrun/date_treatment/clim3d_marina_repro_report"
FEATURE_MATCHED_FWD50_CSV = (
    ROOT / "Longrun/date_treatment/clim3d_feature_mapping_test/feature_matched_fwd50_by_pair.csv"
)


def ensure_dirs() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(PLOT_DIR / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(PLOT_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(PLOT_DIR / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def load_marina_saved_fwd_profile() -> tuple[np.ndarray, pd.DataFrame]:
    p_hpa = src.marina_saved_levels_hpa()
    arr = np.load(src.MARINA_SAVED_FWD_NPY).astype(float)
    if arr.shape[0] == len(p_hpa):
        arr = arr.T
    if arr.shape[1] != len(p_hpa):
        raise ValueError(f"Unexpected saved FWD shape {arr.shape} for {len(p_hpa)} levels")
    years = np.arange(1, arr.shape[0] + 1, dtype=int)
    fwd_df = pd.DataFrame(arr, index=years, columns=p_hpa)
    fwd_df.index.name = "marina_year"
    return p_hpa, fwd_df


def metric_dict(metric_df: pd.DataFrame, col: str) -> dict[int, float]:
    return {
        int(row.year): float(getattr(row, col))
        for row in metric_df.itertuples(index=False)
        if np.isfinite(getattr(row, col))
    }


def build_rm5_pair_table(
    mapping: pd.DataFrame,
    our_fwd: pd.Series,
    marina_fwd: pd.Series,
    our_metric_df: pd.DataFrame,
    marina_metric_df: pd.DataFrame,
) -> pd.DataFrame:
    our_min = metric_dict(our_metric_df, "window_min_DU")
    marina_min = metric_dict(marina_metric_df, "window_min_DU")
    rows = []
    for row in mapping.itertuples(index=False):
        rows.append(
            {
                "pair_id": int(row.pair_id),
                "our_year": int(row.our_year),
                "marina_year": int(row.marina_year),
                "our_o3_rm5_min_DU": our_min.get(int(row.our_year), np.nan),
                "marina_o3_rm5_min_DU": marina_min.get(int(row.marina_year), np.nan),
                "our_fwd50_doy": float(our_fwd.loc[row.our_year])
                if row.our_year in our_fwd.index
                else np.nan,
                "marina_fwd50_doy": float(marina_fwd.loc[row.marina_year])
                if row.marina_year in marina_fwd.index
                else np.nan,
            }
        )
    paired = pd.DataFrame(rows)
    valid_o3 = paired.dropna(subset=["our_o3_rm5_min_DU", "marina_o3_rm5_min_DU"]).copy()
    n_low = max(int(np.floor(0.25 * len(valid_o3))), 1)
    our_low = set(valid_o3.nsmallest(n_low, "our_o3_rm5_min_DU")["pair_id"].astype(int))
    marina_low = set(valid_o3.nsmallest(n_low, "marina_o3_rm5_min_DU")["pair_id"].astype(int))

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

    paired["rm5_low25_membership"] = paired["pair_id"].map(membership)
    paired["is_our_rm5_low25"] = paired["pair_id"].isin(our_low)
    paired["is_marina_rm5_low25"] = paired["pair_id"].isin(marina_low)
    paired["valid_for_rm5_o3_low25"] = paired["pair_id"].isin(set(valid_o3["pair_id"].astype(int)))
    return paired


def load_feature_matched_fwd_pair_table() -> pd.DataFrame:
    """Load the independently fingerprint-matched CLIM-3D FWD pair table.

    The rm5 O3 source-isolation table is still built from the active mapping in
    fwd_clim3d_low25_source_test.py.  The 50 hPa FWD scatter is stricter: it
    should use the mapping inferred directly from field fingerprints in
    fwd_clim3d_feature_mapping_test.py, because an older hand-written chunk map
    duplicated Marina years 50-58 and created artificial FWD outliers.
    """
    if not FEATURE_MATCHED_FWD50_CSV.exists():
        raise FileNotFoundError(
            f"Missing feature-matched FWD table: {FEATURE_MATCHED_FWD50_CSV}. "
            "Run Longrun/date_treatment/fwd_clim3d_feature_mapping_test.py first."
        )
    df = pd.read_csv(FEATURE_MATCHED_FWD50_CSV)
    required = {
        "pair_id",
        "our_year",
        "marina_year",
        "our_fwd_50hpa_doy",
        "marina_fwd_50hpa_doy",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{FEATURE_MATCHED_FWD50_CSV} is missing columns: {sorted(missing)}")
    out = df[
        [
            "pair_id",
            "our_year",
            "marina_year",
            "our_fwd_50hpa_doy",
            "marina_fwd_50hpa_doy",
        ]
    ].copy()
    out = out.rename(
        columns={
            "our_fwd_50hpa_doy": "our_fwd50_doy",
            "marina_fwd_50hpa_doy": "marina_fwd50_doy",
        }
    )
    out["mapping_source"] = "feature_matched_field_fingerprint"
    return out


def plot_variant_summary(summary_df: pd.DataFrame) -> None:
    tests = [
        ("paper Table 1 reference", "Paper Table 1"),
        ("Marina native saved FWD + Marina O3 rm5_file", "Marina native rm5"),
        ("Marina native saved FWD + Marina O3 raw", "Marina native raw"),
        ("Marina native saved FWD + Marina O3 rm15", "Marina native rm15"),
        ("Mapped pair: our FWD + Marina O3 rm5_file", "Mapped: our FWD + Marina rm5"),
        ("Mapped pair: Marina FWD + our O3 csv_rm5", "Mapped: Marina FWD + our rm5"),
        ("Our native FWD + our O3 csv_rm5", "Our native rm5"),
    ]
    rows = []
    for test, label in tests:
        row = summary_df.loc[summary_df["test"].eq(test)].iloc[0].copy()
        row["short_label"] = label
        rows.append(row)
    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    y = np.arange(len(plot_df))
    colors = ["0.25", "#2c7fb8", "#74a9cf", "#a6bddb", "#fdae6b", "#fd8d3c", "#de2d26"]
    ax.barh(y, plot_df["mean_low25_50hpa_doy"], color=colors, edgecolor="0.25", linewidth=0.5)
    ax.axvline(src.PAPER_CLIM3D_LOW25_DOY, color="black", linewidth=1.1, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["short_label"])
    ax.invert_yaxis()
    ax.set_xlabel("LOW25 mean 50 hPa FWD day of year")
    ax.set_title("CLIM-3D LOW25 50 hPa FWD: source-isolation summary")
    ticks = [112, 117, 122]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t}\n{src.doy_to_month_day(t)}" for t in ticks])
    for yi, row in enumerate(plot_df.itertuples(index=False)):
        ax.text(
            float(row.mean_low25_50hpa_doy) + 0.15,
            yi,
            f"{row.mean_low25_50hpa_date} ({row.delta_vs_paper_days:+.2f} d)",
            va="center",
            fontsize=8.5,
        )
    ax.grid(axis="x", color="0.88", linewidth=0.8)
    ax.set_axisbelow(True)
    save_figure(fig, "clim3d_low25_50hpa_mean_variants")


def plot_mapped_o3_scatter(pair_df: pd.DataFrame) -> None:
    valid = pair_df.dropna(subset=["our_o3_rm5_min_DU", "marina_o3_rm5_min_DU"]).copy()
    corr = valid[["marina_o3_rm5_min_DU", "our_o3_rm5_min_DU"]].corr().iloc[0, 1]
    mae = float(np.mean(np.abs(valid["our_o3_rm5_min_DU"] - valid["marina_o3_rm5_min_DU"])))
    n_low = max(int(np.floor(0.25 * len(valid))), 1)
    overlap = int((valid["is_our_rm5_low25"] & valid["is_marina_rm5_low25"]).sum())

    colors = {
        "both LOW25": "#1b9e77",
        "Marina-only LOW25": "#377eb8",
        "Our-only LOW25": "#e6550d",
        "not LOW25": "0.72",
    }
    sizes = {
        "both LOW25": 44,
        "Marina-only LOW25": 52,
        "Our-only LOW25": 52,
        "not LOW25": 22,
    }

    fig, ax = plt.subplots(figsize=(5.7, 5.0))
    for label in ["not LOW25", "both LOW25", "Marina-only LOW25", "Our-only LOW25"]:
        sub = valid[valid["rm5_low25_membership"].eq(label)]
        ax.scatter(
            sub["marina_o3_rm5_min_DU"],
            sub["our_o3_rm5_min_DU"],
            s=sizes[label],
            color=colors[label],
            alpha=0.88,
            edgecolor="white",
            linewidth=0.45,
            label=f"{label} (n={len(sub)})",
        )
    ax.set_xlabel("Marina rm5 Mar-Apr O3 minimum (DU)")
    ax.set_ylabel("Our rm5 Mar-Apr partial O3 minimum (DU)")
    ax.set_title("Mapped CLIM-3D O3 ranking feature")
    ax.text(
        0.03,
        0.97,
        f"n={len(valid)}, corr={corr:.3f}, MAE={mae:.2f} DU\nLOW25 overlap={overlap}/{n_low}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.75", "alpha": 0.92},
    )
    ax.grid(color="0.9", linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    save_figure(fig, "clim3d_mapped_o3_min_rm5_scatter")


def plot_mapped_fwd_scatter(pair_df: pd.DataFrame) -> None:
    valid = pair_df.dropna(subset=["our_fwd50_doy", "marina_fwd50_doy"]).copy()
    corr = valid[["marina_fwd50_doy", "our_fwd50_doy"]].corr().iloc[0, 1]
    absdiff = np.abs(valid["our_fwd50_doy"] - valid["marina_fwd50_doy"])
    mae = float(np.mean(absdiff))
    max_abs = float(np.max(absdiff))
    fig, ax = plt.subplots(figsize=(5.3, 5.0))
    ax.scatter(
        valid["marina_fwd50_doy"],
        valid["our_fwd50_doy"],
        s=30,
        color="#636363",
        alpha=0.75,
        edgecolor="white",
        linewidth=0.4,
    )
    lim_min = float(np.nanmin([valid["marina_fwd50_doy"].min(), valid["our_fwd50_doy"].min()])) - 2
    lim_max = float(np.nanmax([valid["marina_fwd50_doy"].max(), valid["our_fwd50_doy"].max()])) + 2
    ax.plot([lim_min, lim_max], [lim_min, lim_max], color="black", linestyle="--", linewidth=1)
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_xlabel("Marina saved 50 hPa FWD (DOY)")
    ax.set_ylabel("Our generated 50 hPa FWD (DOY)")
    ax.set_title("Feature-matched CLIM-3D 50 hPa FWD")
    ax.text(
        0.03,
        0.97,
        f"n={len(valid)}, corr={corr:.3f}\nMAE={mae:.2f} d, max={max_abs:.0f} d",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.75", "alpha": 0.92},
    )
    ticks = [90, 105, 120, 135, 150]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.grid(color="0.9", linewidth=0.8)
    save_figure(fig, "clim3d_mapped_fwd50_scatter")


def plot_marina_profile(marina_metric_df: pd.DataFrame) -> pd.DataFrame:
    p_hpa, fwd_df = load_marina_saved_fwd_profile()
    metric = marina_metric_df[marina_metric_df["year"].isin(fwd_df.index)].copy()
    n_low = max(int(np.floor(0.25 * len(metric))), 1)
    low_years = metric.nsmallest(n_low, "window_min_DU")["year"].astype(int).tolist()
    high_years = metric.nlargest(n_low, "window_max_DU")["year"].astype(int).tolist()
    all_years = metric["year"].astype(int).tolist()

    profile_df = pd.DataFrame(
        {
            "plev_hpa": p_hpa,
            "all_mean_doy": fwd_df.loc[all_years].mean(axis=0).values,
            "low25_mean_doy": fwd_df.loc[low_years].mean(axis=0).values,
            "high25_mean_doy": fwd_df.loc[high_years].mean(axis=0).values,
        }
    )

    fig, ax = plt.subplots(figsize=(5.2, 5.8))
    ax.plot(profile_df["all_mean_doy"], profile_df["plev_hpa"], color="black", label="all years")
    ax.plot(profile_df["low25_mean_doy"], profile_df["plev_hpa"], color="#2166ac", label="LOW25 O3")
    ax.plot(profile_df["high25_mean_doy"], profile_df["plev_hpa"], color="#b2182b", label="HIGH25 O3")
    ax.axvline(src.PAPER_CLIM3D_LOW25_DOY, color="#2166ac", linestyle="--", linewidth=0.9)
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.set_yticks(p_hpa)
    ax.set_yticklabels([f"{p:g}" for p in p_hpa])
    ticks = [100, 110, 117, 125, 135]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t}\n{src.doy_to_month_day(t)}" for t in ticks])
    ax.set_xlabel("Mean FWD day of year")
    ax.set_ylabel("Pressure (hPa)")
    ax.set_title("Marina native CLIM-3D saved FWD profiles\nselected by Marina rm5 O3")
    ax.grid(color="0.9", linewidth=0.8)
    ax.legend(frameon=False, loc="upper right")
    save_figure(fig, "clim3d_marina_saved_fwd_profiles_rm5")
    return profile_df


def main() -> None:
    ensure_dirs()
    summary_df, diag_df = src.run_test()
    mapping = src.mapping_pairs()
    our_fwd = src.load_our_fwd_50hpa()
    marina_fwd = src.load_marina_saved_fwd_50hpa()
    our_metric_frames, marina_metric_frames = src.build_metrics()

    pair_df = build_rm5_pair_table(
        mapping,
        our_fwd,
        marina_fwd,
        our_metric_frames["rm5"],
        marina_metric_frames["rm5_file"],
    )
    profile_df = plot_marina_profile(marina_metric_frames["rm5_file"])
    plot_variant_summary(summary_df)
    plot_mapped_o3_scatter(pair_df)
    fwd_pair_df = load_feature_matched_fwd_pair_table()
    plot_mapped_fwd_scatter(fwd_pair_df)

    summary_df.to_csv(TABLE_DIR / "source_isolation_summary.csv", index=False)
    diag_df.to_csv(TABLE_DIR / "mapped_pair_diagnostics.csv", index=False)
    mapping.to_csv(TABLE_DIR / "chunk_mapping_pairs.csv", index=False)
    pair_df.to_csv(TABLE_DIR / "mapped_pair_rm5_details.csv", index=False)
    fwd_pair_df.to_csv(TABLE_DIR / "mapped_pair_fwd50_feature_matched_details.csv", index=False)
    profile_df.to_csv(TABLE_DIR / "marina_saved_fwd_profiles_rm5.csv", index=False)

    valid_o3 = pair_df[pair_df["valid_for_rm5_o3_low25"]].copy()
    n_low = max(int(np.floor(0.25 * len(valid_o3))), 1)
    overlap = int((valid_o3["is_our_rm5_low25"] & valid_o3["is_marina_rm5_low25"]).sum())
    key_numbers = pd.DataFrame(
        [
            {
                "metric": "rm5_valid_mapped_o3_pairs",
                "value": len(valid_o3),
            },
            {
                "metric": "rm5_low25_n",
                "value": n_low,
            },
            {
                "metric": "rm5_low25_overlap_pairs",
                "value": overlap,
            },
            {
                "metric": "rm5_o3_min_corr",
                "value": valid_o3[["marina_o3_rm5_min_DU", "our_o3_rm5_min_DU"]].corr().iloc[0, 1],
            },
            {
                "metric": "rm5_o3_min_mae_DU",
                "value": float(
                    np.mean(np.abs(valid_o3["our_o3_rm5_min_DU"] - valid_o3["marina_o3_rm5_min_DU"]))
                ),
            },
        ]
    )
    key_numbers.to_csv(TABLE_DIR / "key_numbers.csv", index=False)

    print(f"Wrote tables to {TABLE_DIR}")
    print(f"Wrote figures to {PLOT_DIR}")
    print(key_numbers.to_string(index=False))


if __name__ == "__main__":
    main()
