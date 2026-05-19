#!/usr/bin/env python3
"""Repair zero-filled partial-O3 days without recomputing the expensive O3 integral.

The old Partial_O3_with_ranking.ipynb could convert all-NaN filled days to 0 DU
when PS was NaN, because xr.where(overlap > 0, ..., 0.0) treats NaN overlap as
False.  This script edits existing partial_O3_all_ranges.nc files in place:

1. Find exact-zero polar-cap partial-O3 days for every pressure range.
2. Set those days to NaN in both polar-cap time series and gridded partial columns.
3. Rebuild the ranking variables and ranking CSV from the repaired daily series.

Default mode is dry-run.  Use --apply to modify files.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import netCDF4 as nc
import numpy as np


CASES = {
    "BWCN": {
        "case_name": "BWCN",
        "nc": Path("/mnt/soclim0/public_data/weiji/BWCN/partial_O3/BWCN_partial_O3_all_ranges.nc"),
        "csv": Path("/mnt/soclim0/public_data/weiji/BWCN/partial_O3/BWCN_partial_O3_ranking_MarApr_min_60_90N.csv"),
    },
    "B2000WCN": {
        "case_name": "B2000WCN",
        "nc": Path("/mnt/soclim0/public_data/weiji/B2000WCN001002_timefixed/partial_O3/B2000WCN_partial_O3_all_ranges.nc"),
        "csv": Path("/mnt/soclim0/public_data/weiji/B2000WCN001002_timefixed/partial_O3/B2000WCN_partial_O3_ranking_MarApr_min_60_90N.csv"),
    },
    "CLIM-2D": {
        "case_name": "B2000WCN_NOCOUPL",
        "nc": Path("/mnt/soclim0/public_data/weiji/B2000WCN_NOCOUPL001002_timefixed/partial_O3/B2000WCN_NOCOUPL_partial_O3_all_ranges.nc"),
        "csv": Path("/mnt/soclim0/public_data/weiji/B2000WCN_NOCOUPL001002_timefixed/partial_O3/B2000WCN_NOCOUPL_partial_O3_ranking_MarApr_min_60_90N.csv"),
    },
    "CLIM-3D": {
        "case_name": "B2000WCN007009010011_Clim3D",
        "nc": Path("/mnt/soclim0/public_data/weiji/B2000WCN007009010011_Clim3D_timefixed/partial_O3/B2000WCN007009010011_Clim3D_partial_O3_all_ranges.nc"),
        "csv": Path("/mnt/soclim0/public_data/weiji/B2000WCN007009010011_Clim3D_timefixed/partial_O3/B2000WCN007009010011_Clim3D_partial_O3_ranking_MarApr_min_60_90N.csv"),
    },
}

RANKING_MONTHS = {3, 4}
RANKING_MIN_VALID_DAYS = 58
RANKING_MIN_VALID_DU = 10.0
ZERO_EPS = 1.0e-12
RANKING_CSV_COLUMNS = [
    "case_name",
    "pressure_range",
    "rank_low_o3",
    "year",
    "marapr_min_DU",
    "marapr_mean_DU",
    "is_lowest10",
    "is_lowest25pct",
]


def read_array(var):
    arr = var[:]
    if np.ma.isMaskedArray(arr):
        arr = arr.filled(np.nan)
    return np.asarray(arr)


def pressure_tags(ds):
    if "pressure_range" in ds.variables:
        raw = ds.variables["pressure_range"][:]
        if getattr(raw, "dtype", None) is not None and raw.dtype.kind in {"S", "U", "O"}:
            return [str(x) for x in raw.tolist()]
        return [str(x) for x in nc.chartostring(raw).tolist()]

    tags = []
    prefix = "O3_partial_60_90N_"
    for name in ds.variables:
        if name.startswith(prefix):
            tags.append(name[len(prefix):])
    return sorted(tags)


def ts_var_info(ds, tag, ip):
    if "O3_partial_60_90N" in ds.variables:
        return ds.variables["O3_partial_60_90N"], (ip, slice(None))
    return ds.variables[f"O3_partial_60_90N_{tag}"], (slice(None),)


def col_var_info(ds, tag, ip):
    if "O3_partial_column" in ds.variables:
        return ds.variables["O3_partial_column"], "stacked"
    return ds.variables[f"O3_partial_column_{tag}"], "per_tag"


def years_months_from_date(ds):
    dates = read_array(ds.variables["date"]).astype(np.int64)
    valid = dates > 0
    years = dates // 10000
    months = (dates // 100) % 100
    return dates, years.astype(np.int32), months.astype(np.int16), valid


def grouped_zero_dates(dates, zero_idx):
    if len(zero_idx) == 0:
        return "none"
    years = dates[zero_idx] // 10000
    rows = []
    for year in np.unique(years):
        idx = zero_idx[years == year]
        rows.append(f"{int(year)}:n={len(idx)},first={int(dates[idx[0]])},last={int(dates[idx[-1]])}")
    return "; ".join(rows[:12])


def clean_ts_for_ranking(ts):
    out = np.asarray(ts, dtype=np.float64).copy()
    out[np.isfinite(out) & (np.abs(out) <= ZERO_EPS)] = np.nan
    return out


def ranking_records(ts, years, months, valid_time):
    ts = clean_ts_for_ranking(ts)
    records = []
    for year in sorted(np.unique(years[valid_time]).astype(int)):
        mask = (years == year) & valid_time & np.isin(months, list(RANKING_MONTHS))
        vals = ts[mask]
        vals = vals[np.isfinite(vals) & (vals > RANKING_MIN_VALID_DU)]
        if vals.size < RANKING_MIN_VALID_DAYS:
            continue
        records.append((int(year), float(np.min(vals)), float(np.mean(vals))))
    records.sort(key=lambda item: item[1])
    return records


def rebuild_ranking_arrays(ds, tags, years, months, valid_time):
    max_rank = len(ds.dimensions["rank_low_o3"])
    n_pressure = len(tags)

    ranked_year = np.full((n_pressure, max_rank), -9999, dtype=np.int32)
    ranked_min = np.full((n_pressure, max_rank), np.nan, dtype=np.float32)
    ranked_mean = np.full((n_pressure, max_rank), np.nan, dtype=np.float32)
    is_lowest10 = np.zeros((n_pressure, max_rank), dtype=np.int8)
    is_lowest25pct = np.zeros((n_pressure, max_rank), dtype=np.int8)
    csv_rows = []

    for ip, tag in enumerate(tags):
        ts_var, ts_sel = ts_var_info(ds, tag, ip)
        ts = read_array(ts_var[ts_sel]).astype(np.float64)
        records = ranking_records(ts, years, months, valid_time)
        n_low10 = min(10, len(records))
        n_low25 = max(int(0.25 * len(records)), 1) if records else 0

        for ir, (year, min_val, mean_val) in enumerate(records[:max_rank]):
            ranked_year[ip, ir] = year
            ranked_min[ip, ir] = min_val
            ranked_mean[ip, ir] = mean_val
            is_lowest10[ip, ir] = int(ir < n_low10)
            is_lowest25pct[ip, ir] = int(ir < n_low25)
            csv_rows.append({
                "pressure_range": tag,
                "rank_low_o3": ir + 1,
                "year": year,
                "marapr_min_DU": min_val,
                "marapr_mean_DU": mean_val,
                "is_lowest10": bool(ir < n_low10),
                "is_lowest25pct": bool(ir < n_low25),
            })

        print(f"    ranking {tag}: valid_years={len(records)}, low25={n_low25}, first10={[r[0] for r in records[:10]]}")

    return ranked_year, ranked_min, ranked_mean, is_lowest10, is_lowest25pct, csv_rows


def maybe_backup(path: Path):
    backup = path.with_name(path.name + ".bak_zero_to_nan")
    if backup.exists():
        print(f"  backup exists, keeping it: {backup}")
        return
    print(f"  backup: {backup}")
    shutil.copy2(path, backup)


def repair_case(label, cfg, apply=False, backup=False):
    nc_path = cfg["nc"]
    csv_path = cfg["csv"]
    print(f"\n===== {label} =====")
    print(nc_path)

    if not nc_path.exists():
        print("  missing; skip")
        return

    mode = "r+" if apply else "r"
    with nc.Dataset(nc_path, mode) as ds:
        tags = pressure_tags(ds)
        dates, years, months, valid_time = years_months_from_date(ds)

        zero_by_tag = {}
        for ip, tag in enumerate(tags):
            ts_var, ts_sel = ts_var_info(ds, tag, ip)
            ts = read_array(ts_var[ts_sel]).astype(np.float64)
            zero_idx = np.where(np.isfinite(ts) & (np.abs(ts) <= ZERO_EPS))[0]
            zero_by_tag[tag] = zero_idx
            print(f"  {tag}: zero_days={len(zero_idx)} | {grouped_zero_dates(dates, zero_idx)}")

        total_zero = sum(len(v) for v in zero_by_tag.values())
        if total_zero == 0 and not apply:
            return
        if total_zero == 0 and apply:
            print("  no zero days; ranking unchanged")
            return

        if apply and backup:
            # Close/reopen around backup would be cleaner, so this function backs up before writes.
            pass

        if apply:
            for ip, tag in enumerate(tags):
                zero_idx = zero_by_tag[tag]
                if len(zero_idx) == 0:
                    continue
                ts_var, ts_sel = ts_var_info(ds, tag, ip)
                ts = read_array(ts_var[ts_sel]).astype(np.float32)
                ts[zero_idx] = np.nan
                ts_var[ts_sel] = ts

                col_var, layout = col_var_info(ds, tag, ip)
                print(f"    writing NaN into {col_var.name} for {len(zero_idx)} days")
                for idx in zero_idx:
                    if layout == "stacked":
                        col_var[ip, idx, :, :] = np.nan
                    else:
                        col_var[idx, :, :] = np.nan

            ranked_year, ranked_min, ranked_mean, low10, low25, csv_rows = rebuild_ranking_arrays(
                ds, tags, years, months, valid_time
            )
            ds.variables["ranked_year"][:, :] = ranked_year
            ds.variables["ranked_marapr_min_DU"][:, :] = ranked_min
            ds.variables["ranked_marapr_mean_DU"][:, :] = ranked_mean
            ds.variables["ranked_is_lowest10"][:, :] = low10
            ds.variables["ranked_is_lowest25pct"][:, :] = low25
            ds.setncattr("zero_day_repair", "exact-zero partial O3 days set to NaN; rankings rebuilt")

    if apply:
        if backup and csv_path.exists():
            maybe_backup(csv_path)
        csv_rows_full = []
        with nc.Dataset(nc_path, "r") as ds:
            tags = pressure_tags(ds)
            _, years, months, valid_time = years_months_from_date(ds)
            _, _, _, _, _, csv_rows = rebuild_ranking_arrays(ds, tags, years, months, valid_time)
            for row in csv_rows:
                csv_rows_full.append({"case_name": cfg["case_name"], **row})
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=RANKING_CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(csv_rows_full)
        print(f"  wrote ranking CSV: {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="modify NetCDF/CSV files in place")
    parser.add_argument("--backup", action="store_true", help="copy .bak_zero_to_nan backups before modifying")
    parser.add_argument("--case", action="append", choices=sorted(CASES), help="case(s) to process; default all")
    args = parser.parse_args()

    selected = args.case or list(CASES)
    if args.apply and args.backup:
        for label in selected:
            cfg = CASES[label]
            maybe_backup(cfg["nc"])
    for label in selected:
        repair_case(label, CASES[label], apply=args.apply, backup=args.backup)

    if not args.apply:
        print("\nDry run only. Re-run with --apply to modify files.")


if __name__ == "__main__":
    main()
