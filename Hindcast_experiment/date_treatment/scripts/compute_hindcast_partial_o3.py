#!/usr/bin/env python3
"""Compute Hindcast partial O3 columns with the Longrun hybrid-overlap method.

Ranking is intentionally omitted. Pressure ranges are de-duplicated, so the
requested repeated 1-100 hPa range is written once.
"""

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
    area_weights_for_lat,
    compute_pressure_interface,
    discover_member_inputs,
    open_dataset,
    parse_case_list,
    write_netcdf_atomic,
)


OUTPUT_SUBDIR = "partial_O3"
PRESSURE_RANGES = (
    ("1_100hPa", 1.0, 100.0),
    ("30_70hPa", 30.0, 70.0),
    ("30_100hPa", 30.0, 100.0),
)

G = 9.80665
M_AIR = 28.964 / 1000.0
NA = 6.02214e23
DU_MOLECULES_PER_M2 = 2.687e20
DU_FACTOR = NA / (G * M_AIR * DU_MOLECULES_PER_M2)


def add_lead_coords(ds: xr.Dataset, src: xr.Dataset) -> xr.Dataset:
    ntime = int(src.sizes["time"])
    ds = ds.rename({"time": "lead_time"})
    ds = ds.assign_coords(lead_time=np.arange(ntime, dtype=np.int16))
    if "date" in src:
        ds["date"] = ("lead_time", src["date"].values.astype(np.int32))
    return ds


def build_hybrid_overlap_dp(ds: xr.Dataset, p_top_hpa: float, p_bot_hpa: float) -> xr.DataArray:
    if p_top_hpa >= p_bot_hpa:
        raise ValueError("p_top_hpa must be smaller than p_bot_hpa")

    p_interface = compute_pressure_interface(ds)
    p_i = p_interface.isel(ilev=slice(0, -1)).rename({"ilev": "lev"})
    p_ip1 = p_interface.isel(ilev=slice(1, None)).rename({"ilev": "lev"})
    if "lev" in ds.coords:
        p_i = p_i.assign_coords(lev=ds["lev"])
        p_ip1 = p_ip1.assign_coords(lev=ds["lev"])

    p_layer_top = xr.where(p_i < p_ip1, p_i, p_ip1)
    p_layer_bot = xr.where(p_i > p_ip1, p_i, p_ip1)
    p_top = p_top_hpa * 100.0
    p_bot = p_bot_hpa * 100.0
    upper = xr.where(p_layer_top > p_top, p_layer_top, p_top)
    lower = xr.where(p_layer_bot < p_bot, p_layer_bot, p_bot)
    overlap = xr.where(lower > upper, lower - upper, 0.0)
    overlap.name = "hybrid_overlap_dp"
    overlap.attrs.update({"units": "Pa", "p_top_hpa": p_top_hpa, "p_bot_hpa": p_bot_hpa})
    return overlap


def partial_o3_column(ds: xr.Dataset, p_top_hpa: float, p_bot_hpa: float) -> xr.DataArray:
    overlap = build_hybrid_overlap_dp(ds, p_top_hpa, p_bot_hpa)
    valid_overlap = np.isfinite(overlap)
    in_target_layer = valid_overlap & (overlap > 0)
    weighted_o3 = xr.where(
        valid_overlap,
        xr.where(in_target_layer, ds["O3"] * overlap * DU_FACTOR, 0.0),
        np.nan,
    )
    col = weighted_o3.sum(dim="lev", skipna=False)
    col.attrs.update(
        {
            "long_name": f"Partial O3 column, {p_top_hpa:g}-{p_bot_hpa:g} hPa",
            "units": "DU",
            "p_top_hpa": float(p_top_hpa),
            "p_bot_hpa": float(p_bot_hpa),
            "method": "CAM/WACCM hybrid interface pressure overlap dp",
        }
    )
    return col


def open_o3_with_ps(paths: Dict[str, Path]) -> xr.Dataset:
    ds = open_dataset(paths["O3"])
    if "PS" not in ds:
        if "PS" not in paths:
            raise FileNotFoundError(f"O3 file has no PS and no matching PS file was discovered: {paths['O3']}")
        ds_ps = open_dataset(paths["PS"])
        keep = [name for name in ["PS", "P0", "hyai", "hybi", "hyam", "hybm", "date", "datesec", "time_bnds", "gw"] if name in ds_ps]
        ds = xr.merge([ds, ds_ps[keep]], compat="override")
    for required in ("O3", "P0", "PS", "hyai", "hybi"):
        if required not in ds:
            raise KeyError(f"{paths['O3']} missing required variable {required}")
    return ds


def compute_one_member(payload) -> Tuple[str, Optional[xr.Dataset], Dict[str, str]]:
    case_name, member_id, paths = payload
    record = {"case": case_name, "member": member_id, "status": "ok", "message": ""}
    ds = None
    try:
        ds = open_o3_with_ps(paths)
        data_vars = {}
        for tag, p_top, p_bot in PRESSURE_RANGES:
            col = partial_o3_column(ds, p_top, p_bot)
            col_name = f"O3_partial_column_{tag}"
            ts_name = f"O3_partial_60_90N_{tag}"
            data_vars[col_name] = col.astype(np.float32)
            weights = area_weights_for_lat(col["lat"], 60.0, 90.0)
            data_vars[ts_name] = col.weighted(weights).mean(dim=["lat", "lon"]).astype(np.float32)
        ds_out = xr.Dataset(data_vars)
        ds_out = add_lead_coords(ds_out, ds).load()
        return member_id, ds_out, record
    except Exception as exc:
        record["status"] = "error"
        record["message"] = f"{type(exc).__name__}: {exc}"
        return member_id, None, record
    finally:
        if ds is not None:
            ds.close()
        gc.collect()


def write_summary(case_root: Path, records: List[Dict[str, str]], overwrite: bool) -> Path:
    out_dir = case_root / OUTPUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{case_root.name}_partial_O3_processing_summary.csv"
    if out_file.exists() and not overwrite:
        out_file = out_dir / f"{case_root.name}_partial_O3_processing_summary.latest.csv"
    with out_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case", "member", "status", "message"])
        writer.writeheader()
        writer.writerows(records)
    return out_file


def process_case(case_root: Path, args) -> None:
    out_file = case_root / OUTPUT_SUBDIR / f"{case_root.name}_partial_O3_all_ranges_members.nc"
    if out_file.exists() and out_file.stat().st_size > 0 and not args.overwrite:
        print(f"[SKIP] {case_root.name}: existing {out_file}")
        return

    inputs = discover_member_inputs(case_root, required_vars=("O3",), optional_vars=("PS",), members=args.members)
    if not inputs:
        print(f"[SKIP] {case_root.name}: no O3 members")
        return

    payloads = [(case_root.name, member_id, paths) for member_id, paths in inputs.items()]
    results: List[Tuple[str, xr.Dataset]] = []
    records: List[Dict[str, str]] = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        futs = [ex.submit(compute_one_member, payload) for payload in payloads]
        for fut in tqdm(as_completed(futs), total=len(futs), desc=f"partial O3 {case_root.name}"):
            member_id, ds, rec = fut.result()
            records.append(rec)
            if ds is not None:
                results.append((member_id, ds))

    print(f"[SUMMARY] {write_summary(case_root, sorted(records, key=lambda r: r['member']), args.overwrite)}")
    if not results:
        print(f"[WARN] {case_root.name}: no partial O3 outputs collected")
        return

    results.sort(key=lambda item: item[0])
    ds_full = xr.concat(
        [item[1] for item in results],
        dim=xr.DataArray([item[0] for item in results], dims="member", name="member"),
        join="outer",
    )
    ds_full.attrs.update(
        {
            "title": f"Hindcast partial O3 columns, {case_root.name}",
            "case_name": case_root.name,
            "pressure_ranges": ",".join(tag for tag, _, _ in PRESSURE_RANGES),
            "ranking": "not computed",
        }
    )
    write_netcdf_atomic(ds_full, out_file, overwrite=args.overwrite)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_default_args(parser)
    args = parser.parse_args(argv)
    for case_root in parse_case_list(args.root, args.cases):
        process_case(case_root, args)


if __name__ == "__main__":
    main()
