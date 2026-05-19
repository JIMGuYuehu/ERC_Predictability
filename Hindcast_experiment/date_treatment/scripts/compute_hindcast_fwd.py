#!/usr/bin/env python3
"""Compute Hindcast final warming dates on the Longrun compact pressure grid."""

from __future__ import annotations

import argparse
import csv
import gc
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hindcast_common import (
    add_default_args,
    clean_field,
    compute_pressure_mid,
    discover_member_inputs,
    interp_profile_logp,
    open_dataset,
    parse_case_list,
    write_netcdf_atomic,
)


OUTPUT_SUBDIR = "final_warming_date"
PLEV_FWD_HPA = np.array([1, 2, 3, 4, 5, 7, 9, 10, 11, 13, 16, 20, 24, 30, 36, 43, 50], dtype=np.float64)
PLEV_FWD_PA = PLEV_FWD_HPA * 100.0
TARGET_LAT = 60.0
WESTERLY_RUN_LENGTH = 10
FILL_INT = np.int32(-9999)


def date_to_month_day_doy(date_values) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    date_values = np.asarray(date_values, dtype=np.int64)
    month_lengths = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=np.int16)
    month_ends = np.cumsum(month_lengths)
    mmdd = date_values % 10000
    month = (mmdd // 100).astype(np.int16)
    day = (mmdd % 100).astype(np.int16)
    doy = np.zeros(date_values.shape, dtype=np.int16)
    for m in range(1, 13):
        mask = month == m
        if np.any(mask):
            prev = int(month_ends[m - 2]) if m > 1 else 0
            doy[mask] = prev + day[mask]
    return month, day, doy


def threshold_for_pressure(plev_pa):
    plev_pa = np.asarray(plev_pa, dtype=np.float64)
    return np.where(plev_pa <= 1000.0, 0.0, 7.0).astype(np.float32)


def lat_bracket_indices(lat_values, target_lat):
    lat_values = np.asarray(lat_values, dtype=np.float64)
    order = np.argsort(lat_values)
    lat_sorted = lat_values[order]
    exact = np.where(np.isclose(lat_sorted, target_lat, rtol=0.0, atol=1.0e-10))[0]
    if exact.size:
        return [int(order[exact[0]])], 0.0
    pos = int(np.searchsorted(lat_sorted, target_lat, side="left"))
    if pos <= 0 or pos >= len(lat_sorted):
        raise ValueError(f"{target_lat}N is outside latitude range {lat_sorted[0]}..{lat_sorted[-1]}")
    lower = pos - 1
    upper = pos
    weight = float((target_lat - lat_sorted[lower]) / (lat_sorted[upper] - lat_sorted[lower]))
    return [int(order[lower]), int(order[upper])], weight


def collapse_selected_lat(da: xr.DataArray, weight: float) -> xr.DataArray:
    if da.sizes.get("lat", 1) == 1:
        return da.isel(lat=0, drop=True)
    return (1.0 - weight) * da.isel(lat=0, drop=True) + weight * da.isel(lat=1, drop=True)


def load_u60_plev(ds_u: xr.Dataset) -> xr.DataArray:
    lat_indices, lat_weight = lat_bracket_indices(ds_u["lat"].values, TARGET_LAT)
    u_zm = clean_field(ds_u["U"]).isel(lat=lat_indices).mean("lon", skipna=True).transpose("time", "lat", "lev")
    p_mid = compute_pressure_mid(ds_u)
    p_zm = p_mid.isel(lat=lat_indices).mean("lon", skipna=True).transpose("time", "lat", "lev")
    u_plev = interp_profile_logp(u_zm, p_zm, PLEV_FWD_PA).transpose("time", "lat", "plev")
    u60 = collapse_selected_lat(u_plev, lat_weight).transpose("time", "plev")
    u60 = u60.assign_coords(plev=("plev", PLEV_FWD_PA))
    return u60


def has_later_westerly_return(values: np.ndarray, start: int, threshold: float) -> bool:
    run = 0
    for val in values[start + 1 :]:
        if np.isfinite(val) and val >= threshold:
            run += 1
            if run >= WESTERLY_RUN_LENGTH:
                return True
        else:
            run = 0
    return False


def find_final_warming(values: np.ndarray, threshold: float, require_complete: bool) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.nan
    if require_complete and np.any(~np.isfinite(values)):
        return np.nan
    for i, val in enumerate(values):
        if np.isfinite(val) and val < threshold:
            if not has_later_westerly_return(values, i, threshold):
                return float(i)
    return np.nan


def compute_one_member(payload) -> Tuple[str, Optional[xr.Dataset], Dict[str, str]]:
    case_name, member_id, paths, require_complete = payload
    record = {"case": case_name, "member": member_id, "status": "ok", "message": ""}
    ds_u = None
    try:
        ds_u = open_dataset(paths["U"])
        if "date" not in ds_u:
            raise KeyError("U file has no CAM date variable; FWD needs calendar month/day")
        u60 = load_u60_plev(ds_u).load()
        dates = ds_u["date"].values.astype(np.int32)
        month, day, doy = date_to_month_day_doy(dates)
        janjun = month <= 6
        thresholds = threshold_for_pressure(PLEV_FWD_PA)

        fwd_index = np.full(len(PLEV_FWD_PA), np.nan, dtype=np.float32)
        fwd_doy = np.full(len(PLEV_FWD_PA), np.nan, dtype=np.float32)
        fwd_date = np.full(len(PLEV_FWD_PA), FILL_INT, dtype=np.int32)
        fwd_month = np.full(len(PLEV_FWD_PA), FILL_INT, dtype=np.int32)
        fwd_day = np.full(len(PLEV_FWD_PA), FILL_INT, dtype=np.int32)
        n_valid = np.zeros(len(PLEV_FWD_PA), dtype=np.int16)
        has_nan = np.zeros(len(PLEV_FWD_PA), dtype=np.int8)

        for ip, threshold in enumerate(thresholds):
            series = u60.isel(plev=ip).values[janjun]
            valid_dates = dates[janjun]
            valid_month = month[janjun]
            valid_day = day[janjun]
            valid_doy = doy[janjun]
            n_valid[ip] = int(np.isfinite(series).sum())
            has_nan[ip] = int(np.any(~np.isfinite(series)))
            local_index = find_final_warming(series, float(threshold), require_complete)
            if np.isfinite(local_index):
                ii = int(local_index)
                fwd_index[ip] = local_index
                fwd_doy[ip] = float(valid_doy[ii])
                fwd_date[ip] = int(valid_dates[ii])
                fwd_month[ip] = int(valid_month[ii])
                fwd_day[ip] = int(valid_day[ii])

        ds_out = xr.Dataset(
            {
                "FWD_day_index_janjun0": ("plev", fwd_index),
                "FWD_dayofyear": ("plev", fwd_doy),
                "FWD_date": ("plev", fwd_date),
                "FWD_month": ("plev", fwd_month),
                "FWD_day_of_month": ("plev", fwd_day),
                "n_valid_janjun": ("plev", n_valid),
                "has_nan_janjun": ("plev", has_nan),
                "threshold_u": ("plev", thresholds),
            },
            coords={"plev": PLEV_FWD_PA, "plev_hpa": ("plev", PLEV_FWD_HPA)},
        )
        ds_out["plev"].attrs.update({"units": "Pa", "positive": "down"})
        ds_out["FWD_day_index_janjun0"].attrs["description"] = "0-based index within available Jan-Jun forecast days"
        return member_id, ds_out, record
    except Exception as exc:
        record["status"] = "error"
        record["message"] = f"{type(exc).__name__}: {exc}"
        return member_id, None, record
    finally:
        if ds_u is not None:
            ds_u.close()
        gc.collect()


def write_summary(case_root: Path, records: List[Dict[str, str]], overwrite: bool) -> Path:
    out_dir = case_root / OUTPUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{case_root.name}_FWD_processing_summary.csv"
    if out_file.exists() and not overwrite:
        out_file = out_dir / f"{case_root.name}_FWD_processing_summary.latest.csv"
    with out_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case", "member", "status", "message"])
        writer.writeheader()
        writer.writerows(records)
    return out_file


def process_case(case_root: Path, args) -> None:
    out_file = case_root / OUTPUT_SUBDIR / f"{case_root.name}_FWD_plev_member.nc"
    if out_file.exists() and out_file.stat().st_size > 0 and not args.overwrite:
        print(f"[SKIP] {case_root.name}: existing {out_file}")
        return

    inputs = discover_member_inputs(case_root, required_vars=("U",), members=args.members)
    if not inputs:
        print(f"[SKIP] {case_root.name}: no U members")
        return

    payloads = [(case_root.name, mid, paths, args.require_complete_janjun) for mid, paths in inputs.items()]
    records: List[Dict[str, str]] = []
    results: List[Tuple[str, xr.Dataset]] = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        futs = [ex.submit(compute_one_member, payload) for payload in payloads]
        for fut in tqdm(as_completed(futs), total=len(futs), desc=f"FWD {case_root.name}"):
            member_id, ds, rec = fut.result()
            records.append(rec)
            if ds is not None:
                results.append((member_id, ds))

    print(f"[SUMMARY] {write_summary(case_root, sorted(records, key=lambda r: r['member']), args.overwrite)}")
    if not results:
        print(f"[WARN] {case_root.name}: no FWD outputs collected")
        return

    results.sort(key=lambda item: item[0])
    ds_full = xr.concat(
        [item[1] for item in results],
        dim=xr.DataArray([item[0] for item in results], dims="member", name="member"),
        join="outer",
    )
    ds_full.attrs.update(
        {
            "title": f"Hindcast final warming date, {case_root.name}",
            "case_name": case_root.name,
            "target_latitude_degrees_north": TARGET_LAT,
            "algorithm": f"first available Jan-Jun day below threshold with no later {WESTERLY_RUN_LENGTH}-day westerly return",
            "threshold_definition": "0 m/s for p <= 10 hPa; 7 m/s for p > 10 hPa",
            "require_complete_janjun": str(args.require_complete_janjun),
        }
    )
    write_netcdf_atomic(ds_full, out_file, overwrite=args.overwrite)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_default_args(parser)
    parser.add_argument(
        "--require-complete-janjun",
        action="store_true",
        help="Mark member/level missing if any available Jan-Jun U value is NaN.",
    )
    args = parser.parse_args(argv)
    for case_root in parse_case_list(args.root, args.cases):
        process_case(case_root, args)


if __name__ == "__main__":
    main()
