#!/usr/bin/env python3
"""Compute Longrun-consistent Hindcast EP flux diagnostics.

Default method matches Longrun/date_treatment/Epflux_calculation.ipynb:
DO_UBAR=True and OMEGA pressure-velocity correction enabled. Members without
OMEGA are skipped by default and recorded in a summary CSV.
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
LONGRUN_DIR = SCRIPT_DIR.parents[2] / "Longrun" / "date_treatment"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(LONGRUN_DIR) not in sys.path:
    sys.path.insert(0, str(LONGRUN_DIR))

from aostools_functions import ComputeEPfluxDiv
from hindcast_common import (
    PLEV_STD_PA,
    add_default_args,
    clean_field,
    compute_pressure_mid,
    discover_member_inputs,
    interp_profile_logp,
    open_dataset,
    parse_case_list,
    write_netcdf_atomic,
)


DO_UBAR = True
USE_OMEGA = True
OUTPUT_SUBDIR = "EPflux_daily_ubar_wcorr"
WAVES = {"all_waves": -1, "wave1": 1, "wave2": 2}
COMPONENTS = ("ep1", "ep2", "div1", "div2")


def compute_ep_components(lat_np, u_np, v_np, t_np, w_np, wave):
    return ComputeEPfluxDiv(
        lat=lat_np,
        pres=PLEV_STD_PA / 100.0,
        u=u_np,
        v=v_np,
        t=t_np,
        w=w_np,
        do_ubar=DO_UBAR,
        wave=wave,
    )


def add_lead_coords(ds: xr.Dataset, src: xr.Dataset) -> xr.Dataset:
    ntime = int(src.sizes["time"])
    ds = ds.rename({"time": "lead_time"})
    ds = ds.assign_coords(lead_time=np.arange(ntime, dtype=np.int16))
    ds["lead_time"].attrs.update({"long_name": "0-based lead-time index from each member initial date"})
    if "date" in src:
        ds["date"] = ("lead_time", src["date"].values.astype(np.int32))
    if "datesec" in src:
        ds["datesec"] = ("lead_time", src["datesec"].values.astype(np.int32))
    return ds


def compute_one_member(payload) -> Tuple[str, Optional[xr.Dataset], Dict[str, str]]:
    case_name, member_id, paths, allow_missing_omega = payload
    record = {
        "case": case_name,
        "member": member_id,
        "status": "ok",
        "message": "",
        "has_omega": str("OMEGA" in paths),
    }
    ds_u = ds_v = ds_t = ds_w = None
    try:
        if "OMEGA" not in paths and not allow_missing_omega:
            record["status"] = "skip_missing_omega"
            record["message"] = "OMEGA input is required for Longrun-consistent EP flux"
            return member_id, None, record

        ds_u = open_dataset(paths["U"])
        ds_v = open_dataset(paths["V"])
        ds_t = open_dataset(paths["T"])
        ds_w = open_dataset(paths["OMEGA"]) if "OMEGA" in paths else None

        p_mid = compute_pressure_mid(ds_u)
        u_std = interp_profile_logp(clean_field(ds_u["U"]), p_mid, PLEV_STD_PA)
        v_std = interp_profile_logp(clean_field(ds_v["V"]), p_mid, PLEV_STD_PA)
        t_std = interp_profile_logp(clean_field(ds_t["T"]), p_mid, PLEV_STD_PA)
        if ds_w is not None:
            w_std = interp_profile_logp(clean_field(ds_w["OMEGA"]) / 100.0, p_mid, PLEV_STD_PA)
            w_np = w_std.transpose("time", "plev", "lat", "lon").values
        else:
            w_np = None

        u_np = u_std.transpose("time", "plev", "lat", "lon").values
        v_np = v_std.transpose("time", "plev", "lat", "lon").values
        t_np = t_std.transpose("time", "plev", "lat", "lon").values
        lat_np = ds_u["lat"].values

        ep = {}
        for label, wave in WAVES.items():
            ep[label] = compute_ep_components(lat_np, u_np, v_np, t_np, w_np, wave)

        ep["wave_rest"] = tuple(ep["all_waves"][i] - ep["wave1"][i] - ep["wave2"][i] for i in range(4))
        coords = {"time": ds_u["time"], "plev": PLEV_STD_PA, "lat": lat_np}
        data_vars = {}
        for label, arrays in ep.items():
            for component, arr in zip(COMPONENTS, arrays):
                data_vars[f"{component}_{label}"] = (("time", "plev", "lat"), arr.astype(np.float32))

        ds_out = xr.Dataset(data_vars=data_vars, coords=coords)
        ds_out["plev"].attrs.update({"units": "Pa", "positive": "down", "long_name": "pressure"})
        ds_out["lat"].attrs.update(ds_u["lat"].attrs)
        ds_out = add_lead_coords(ds_out, ds_u).load()
        return member_id, ds_out, record
    except Exception as exc:
        record["status"] = "error"
        record["message"] = f"{type(exc).__name__}: {exc}"
        return member_id, None, record
    finally:
        for ds in (ds_u, ds_v, ds_t, ds_w):
            if ds is not None:
                ds.close()
        gc.collect()


def write_summary(case_root: Path, records: List[Dict[str, str]], overwrite: bool) -> Path:
    out_dir = case_root / OUTPUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{case_root.name}_EPFLUX_processing_summary.csv"
    if out_file.exists() and not overwrite:
        out_file = out_dir / f"{case_root.name}_EPFLUX_processing_summary.latest.csv"
    with out_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case", "member", "status", "message", "has_omega"])
        writer.writeheader()
        writer.writerows(records)
    return out_file


def write_wave_outputs(case_root: Path, ds_full: xr.Dataset, overwrite: bool) -> None:
    out_base = case_root / OUTPUT_SUBDIR
    selections = {
        "all_waves": "all_waves",
        "wave1": "wave1",
        "wave2": "wave2",
        "wave_rest": "wave_rest",
    }
    for subdir, suffix in selections.items():
        rename = {f"{component}_{suffix}": component for component in COMPONENTS}
        ds = ds_full[list(rename)].rename(rename)
        ds.attrs.update(
            {
                "title": f"Hindcast EP flux, {case_root.name}, {subdir}",
                "case_name": case_root.name,
                "method": "ComputeEPfluxDiv with DO_UBAR=True and OMEGA correction when available",
                "do_ubar": str(DO_UBAR),
                "use_omega_w_correction": str(USE_OMEGA),
                "omega_units_passed_to_aostools": "hPa/s",
                "wave_selection": subdir,
                "source": "Hindcast member U/V/T/OMEGA files on CAM hybrid levels, log-p interpolated to Longrun standard pressure grid",
            }
        )
        out_file = out_base / subdir / f"EPFLUX_{subdir}_{case_root.name}_members_time_plev_lat.nc"
        write_netcdf_atomic(ds, out_file, overwrite=overwrite)


def process_case(case_root: Path, args) -> None:
    inputs = discover_member_inputs(
        case_root,
        required_vars=("U", "V", "T"),
        optional_vars=("OMEGA",),
        members=args.members,
    )
    if not inputs:
        print(f"[SKIP] {case_root.name}: no U/V/T member intersection")
        return

    expected = case_root / OUTPUT_SUBDIR / "all_waves" / f"EPFLUX_all_waves_{case_root.name}_members_time_plev_lat.nc"
    if expected.exists() and expected.stat().st_size > 0 and not args.overwrite:
        print(f"[SKIP] {case_root.name}: existing EPFLUX output {expected}")
        return

    print(f"[CASE] {case_root.name}: {len(inputs)} U/V/T members")
    payloads = [
        (case_root.name, member_id, paths, args.allow_missing_omega)
        for member_id, paths in inputs.items()
    ]
    results: List[Tuple[str, xr.Dataset]] = []
    records: List[Dict[str, str]] = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        futs = [ex.submit(compute_one_member, payload) for payload in payloads]
        for fut in tqdm(as_completed(futs), total=len(futs), desc=f"EPflux {case_root.name}"):
            member_id, ds, rec = fut.result()
            records.append(rec)
            if ds is not None:
                results.append((member_id, ds))

    summary = write_summary(case_root, sorted(records, key=lambda r: r["member"]), overwrite=args.overwrite)
    print(f"[SUMMARY] {summary}")
    if not results:
        print(f"[WARN] {case_root.name}: no EP flux outputs collected")
        return

    results.sort(key=lambda item: item[0])
    member_ids = [item[0] for item in results]
    datasets = [item[1] for item in results]
    ds_full = xr.concat(datasets, dim=xr.DataArray(member_ids, dims="member", name="member"), join="outer")
    ds_full["member"].attrs["description"] = "Hindcast member id"
    write_wave_outputs(case_root, ds_full, overwrite=args.overwrite)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_default_args(parser)
    parser.add_argument(
        "--allow-missing-omega",
        action="store_true",
        help="Compute members without OMEGA using w=None. Default skips them to keep Longrun consistency.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    for case_root in parse_case_list(args.root, args.cases):
        process_case(case_root, args)


if __name__ == "__main__":
    main()
