#!/usr/bin/env python3
"""Compute Hindcast daily eddy heat flux on the Longrun pressure grid.

Outputs contain both v'T' and v'theta' as member x lead_time x plev x lat.
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


OUTPUT_SUBDIR = "Eddyheatflux_daily"
KAPPA_DRY_AIR = 287.0 / 1004.0
P0_THETA_PA = 100000.0


def add_lead_coords(ds: xr.Dataset, src: xr.Dataset) -> xr.Dataset:
    ntime = int(src.sizes["time"])
    ds = ds.rename({"time": "lead_time"})
    ds = ds.assign_coords(lead_time=np.arange(ntime, dtype=np.int16))
    if "date" in src:
        ds["date"] = ("lead_time", src["date"].values.astype(np.int32))
    return ds


def compute_one_member(payload) -> Tuple[str, Optional[xr.Dataset], Dict[str, str]]:
    case_name, member_id, paths = payload
    record = {"case": case_name, "member": member_id, "status": "ok", "message": ""}
    ds_v = ds_t = None
    try:
        ds_v = open_dataset(paths["V"])
        ds_t = open_dataset(paths["T"])

        V = clean_field(ds_v["V"])
        T = clean_field(ds_t["T"])
        p_mid = compute_pressure_mid(ds_v)
        p_mid = p_mid.where(np.isfinite(p_mid) & (p_mid > 0.0))

        theta = T * (P0_THETA_PA / p_mid) ** KAPPA_DRY_AIR
        theta = clean_field(theta)

        valid_pair = V.notnull() & T.notnull() & theta.notnull() & p_mid.notnull()
        Vp = V.where(valid_pair)
        Tp = T.where(valid_pair)
        Thetap = theta.where(valid_pair)

        V_zm = Vp.mean("lon", skipna=True)
        T_zm = Tp.mean("lon", skipna=True)
        Theta_zm = Thetap.mean("lon", skipna=True)
        VT_zm = (Vp * Tp).mean("lon", skipna=True)
        VTheta_zm = (Vp * Thetap).mean("lon", skipna=True)
        p_zm = p_mid.mean("lon", skipna=True).transpose("time", "lat", "lev")

        vt_hyb = (VT_zm - V_zm * T_zm).transpose("time", "lat", "lev")
        vtheta_hyb = (VTheta_zm - V_zm * Theta_zm).transpose("time", "lat", "lev")

        vt_std = interp_profile_logp(vt_hyb, p_zm, PLEV_STD_PA).transpose("time", "plev", "lat")
        vtheta_std = interp_profile_logp(vtheta_hyb, p_zm, PLEV_STD_PA).transpose("time", "plev", "lat")

        ds_out = xr.Dataset(
            {
                "EHF_vTprime": vt_std.astype(np.float32),
                "EHF_vThetaprime": vtheta_std.astype(np.float32),
            }
        )
        ds_out["EHF_vTprime"].attrs.update(
            {
                "long_name": "Daily zonal-mean eddy heat flux v'T'",
                "units": "K m s-1",
                "definition": "[V*T]-[V][T], hybrid levels then log-p interpolation",
            }
        )
        ds_out["EHF_vThetaprime"].attrs.update(
            {
                "long_name": "Daily zonal-mean eddy heat flux v'theta'",
                "units": "K m s-1",
                "definition": "[V*theta]-[V][theta], theta=T*(100000 Pa/p_mid)^(R_d/c_p)",
                "theta_reference_pressure_pa": P0_THETA_PA,
                "kappa": KAPPA_DRY_AIR,
            }
        )
        ds_out["plev"].attrs.update({"units": "Pa", "positive": "down", "long_name": "pressure"})
        ds_out = add_lead_coords(ds_out, ds_v).load()
        return member_id, ds_out, record
    except Exception as exc:
        record["status"] = "error"
        record["message"] = f"{type(exc).__name__}: {exc}"
        return member_id, None, record
    finally:
        for ds in (ds_v, ds_t):
            if ds is not None:
                ds.close()
        gc.collect()


def write_summary(case_root: Path, records: List[Dict[str, str]], overwrite: bool) -> Path:
    out_dir = case_root / OUTPUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{case_root.name}_EHF_processing_summary.csv"
    if out_file.exists() and not overwrite:
        out_file = out_dir / f"{case_root.name}_EHF_processing_summary.latest.csv"
    with out_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case", "member", "status", "message"])
        writer.writeheader()
        writer.writerows(records)
    return out_file


def process_case(case_root: Path, args) -> None:
    out_file = case_root / OUTPUT_SUBDIR / f"EHF_{case_root.name}_members_time_plev_lat.nc"
    if out_file.exists() and out_file.stat().st_size > 0 and not args.overwrite:
        print(f"[SKIP] {case_root.name}: existing {out_file}")
        return

    inputs = discover_member_inputs(case_root, required_vars=("V", "T"), members=args.members)
    if not inputs:
        print(f"[SKIP] {case_root.name}: no V/T member intersection")
        return

    payloads = [(case_root.name, member_id, paths) for member_id, paths in inputs.items()]
    results: List[Tuple[str, xr.Dataset]] = []
    records: List[Dict[str, str]] = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        futs = [ex.submit(compute_one_member, payload) for payload in payloads]
        for fut in tqdm(as_completed(futs), total=len(futs), desc=f"EHF {case_root.name}"):
            member_id, ds, rec = fut.result()
            records.append(rec)
            if ds is not None:
                results.append((member_id, ds))

    print(f"[SUMMARY] {write_summary(case_root, sorted(records, key=lambda r: r['member']), args.overwrite)}")
    if not results:
        print(f"[WARN] {case_root.name}: no EHF outputs collected")
        return

    results.sort(key=lambda item: item[0])
    ds_full = xr.concat(
        [item[1] for item in results],
        dim=xr.DataArray([item[0] for item in results], dims="member", name="member"),
        join="outer",
    )
    ds_full.attrs.update(
        {
            "title": f"Hindcast eddy heat flux, {case_root.name}",
            "case_name": case_root.name,
            "source": "Hindcast member V/T files on CAM hybrid levels",
            "pressure_interpolation": "linear in log pressure to Longrun standard pressure grid",
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
