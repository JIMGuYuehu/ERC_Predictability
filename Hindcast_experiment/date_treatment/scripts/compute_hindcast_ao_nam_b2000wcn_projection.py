#!/usr/bin/env python3
"""Project Hindcast AO/NAM onto B2000WCN001002 first EOF modes.

The script first rebuilds the B2000WCN001002 AO/NAM reference modes from Longrun
Z3/PS files and saves the mode patterns for traceability. Hindcast members are
then projected onto those modes. This matches the Longrun BWCN use_reference=True
logic; it deliberately avoids training separate Hindcast EOFs.
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
from eofs.standard import Eof as StandardEof
from eofs.xarray import Eof as XarrayEof
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hindcast_common import (
    B2000WCN_ROOT,
    TARGET_NAM_PLEV_HPA,
    TARGET_NAM_PLEV_PA,
    add_default_args,
    clean_field,
    compute_pressure_mid,
    discover_member_inputs,
    interp_profile_logp,
    open_dataset,
    parse_case_list,
    write_netcdf_atomic,
)


OUTPUT_SUBDIR = "NAM_B2000WCN_projection"
LAT_MIN = 20.0
LAT_MAX = 90.0
AO_PLEV_HPA = 1000.0
AO_TARGET_LAT = np.arange(20.0, 90.0 + 0.001, 2.5)
YEAR_RE = re.compile(r"\.cam\.h3\.(\d{4})\.")


def date_to_doy(date_values) -> np.ndarray:
    month_lengths = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=np.int16)
    month_ends = np.cumsum(month_lengths)
    date_values = np.asarray(date_values, dtype=np.int64)
    mmdd = date_values % 10000
    month = (mmdd // 100).astype(np.int16)
    day = (mmdd % 100).astype(np.int16)
    doy = np.zeros(date_values.shape, dtype=np.int16)
    for m in range(1, 13):
        mask = month == m
        if np.any(mask):
            prev = int(month_ends[m - 2]) if m > 1 else 0
            doy[mask] = prev + day[mask]
    return doy


def parse_year(path: Path) -> Optional[int]:
    m = YEAR_RE.search(path.name)
    return int(m.group(1)) if m else None


def discover_longrun_pairs(root: Path) -> List[Tuple[int, Path, Path]]:
    z3_files = sorted((root / "Z3").glob("*.Z3.nc"), key=lambda p: (parse_year(p) or 10**9, p.name))
    ps_files = {parse_year(p): p for p in (root / "PS").glob("*.PS.nc") if parse_year(p) is not None}
    pairs = []
    for z3 in z3_files:
        year = parse_year(z3)
        if year is not None and year in ps_files:
            pairs.append((year, z3, ps_files[year]))
    return pairs


def open_cftime(path: Path) -> xr.Dataset:
    try:
        coder = xr.coders.CFDatetimeCoder(use_cftime=True)
        return xr.open_dataset(path, decode_times=coder)
    except Exception:
        return xr.open_dataset(path, use_cftime=True)


def load_longrun_year_z3_zm(payload) -> Optional[xr.DataArray]:
    year, z3_file, ps_file = payload
    ds_z = ds_ps = None
    try:
        ds_z = open_cftime(z3_file)
        ds_ps = open_cftime(ps_file)
        p_mid = ds_z["hyam"] * ds_z["P0"] + ds_z["hybm"] * ds_ps["PS"]
        z3 = clean_field(ds_z["Z3"]).sel(lat=slice(LAT_MIN, LAT_MAX))
        p_mid = p_mid.sel(lat=slice(LAT_MIN, LAT_MAX))
        z_plev = interp_profile_logp(z3, p_mid, TARGET_NAM_PLEV_PA)
        z_zm = z_plev.mean("lon", skipna=True).transpose("time", "plev", "lat")
        z_zm = z_zm.rename({"plev": "lev"}).assign_coords(lev=("lev", TARGET_NAM_PLEV_HPA))
        return z_zm.load()
    except Exception as exc:
        print(f"[WARN] longrun year {year}: {type(exc).__name__}: {exc}")
        return None
    finally:
        for ds in (ds_z, ds_ps):
            if ds is not None:
                ds.close()
        gc.collect()


def load_b2000wcn_z3_zm(root: Path, max_workers: int, max_years: Optional[int]) -> xr.DataArray:
    pairs = discover_longrun_pairs(root)
    if max_years:
        pairs = pairs[: int(max_years)]
    if not pairs:
        raise FileNotFoundError(f"No Z3/PS longrun pairs found under {root}")
    print(f"[REFERENCE] loading {len(pairs)} B2000WCN years from {root}")
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(load_longrun_year_z3_zm, pair) for pair in pairs]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="B2000WCN Z3"):
            da = fut.result()
            if da is not None:
                results.append(da)
    if not results:
        raise RuntimeError("No B2000WCN Z3 years loaded")
    return xr.concat(results, dim="time").sortby("time")


def valid_time_mask(da: xr.DataArray) -> xr.DataArray:
    finite = xr.apply_ufunc(np.isfinite, da)
    for dim in [d for d in da.dims if d != "time"]:
        finite = finite.all(dim)
    return finite


def decide_ao_flip(gh_layer: np.ndarray, ao_raw: np.ndarray) -> int:
    max_idx = int(np.nanargmax(ao_raw))
    field = gh_layer[max_idx]
    mid = field.size // 2
    # Positive AO convention: low Arctic height and higher midlatitude height.
    return -1 if np.nanmean(field[mid:]) > np.nanmean(field[:mid]) else 1


def take_clim_by_doy(clim: xr.DataArray, doy: np.ndarray, time_coord: xr.DataArray) -> xr.DataArray:
    selected = clim.sel(dayofyear=xr.DataArray(doy, dims="time"))
    if "dayofyear" in selected.coords:
        selected = selected.drop_vars("dayofyear")
    return selected.assign_coords(time=time_coord)


def build_reference(z3_zm_all: xr.DataArray):
    ref: Dict[str, object] = {}

    z_ao = z3_zm_all.sel(lev=AO_PLEV_HPA, method="nearest").interp(lat=AO_TARGET_LAT)
    mask = valid_time_mask(z_ao)
    z_ao = z_ao.sel(time=mask)
    clim_ao = z_ao.groupby("time.dayofyear").mean("time")
    ao_anom = z_ao.groupby("time.dayofyear") - clim_ao
    if "dayofyear" in ao_anom.coords:
        ao_anom = ao_anom.drop_vars("dayofyear")
    gh_layer = np.asarray(ao_anom, dtype=np.float64)
    lat_target = ao_anom["lat"].values.astype(np.float64)
    weights = np.sqrt(np.cos(np.deg2rad(lat_target)).clip(0.0, 1.0))
    ao_solver = StandardEof(gh_layer, weights=weights, center=True)
    ao_raw = np.reshape(ao_solver.pcs(npcs=1, pcscaling=1), (gh_layer.shape[0],))
    ao_flip = decide_ao_flip(gh_layer, ao_raw)

    ref["AO_clim_doy"] = clim_ao
    ref["AO_solver"] = ao_solver
    ref["AO_flip"] = ao_flip
    ref["AO_lat_target"] = lat_target
    ref["AO_eof1_unscaled"] = xr.DataArray(
        np.squeeze(ao_solver.eofs(neofs=1, eofscaling=0)) * ao_flip,
        dims=("lat",),
        coords={"lat": lat_target},
        name="AO_eof1_unscaled",
    )
    ref["AO_eof1_project_scaling"] = xr.DataArray(
        np.squeeze(ao_solver.eofs(neofs=1, eofscaling=1)) * ao_flip,
        dims=("lat",),
        coords={"lat": lat_target},
        name="AO_eof1_project_scaling",
    )

    z_valid = z3_zm_all.sel(time=valid_time_mask(z3_zm_all))
    z_month = z_valid.resample(time="1MS").mean().dropna(dim="time", how="all")
    clim_mon = z_month.groupby("time.month").mean("time")
    anom_mon = z_month.groupby("time.month") - clim_mon
    clim_doy = z_valid.groupby("time.dayofyear").mean("time")
    clim_doy_smooth = clim_doy.rolling(dayofyear=21, center=True, min_periods=1).mean()
    anom_doy = z_valid.groupby("time.dayofyear") - clim_doy_smooth
    if "dayofyear" in anom_doy.coords:
        anom_doy = anom_doy.drop_vars("dayofyear")

    weights_da = np.sqrt(np.cos(np.deg2rad(z_valid["lat"])).clip(0.0, 1.0))
    nam_levels = {}
    eof_list = []
    pc_mean = []
    pc_std = []
    flip_list = []
    for lev in tqdm(z_valid.lev.values, desc="B2000WCN NAM EOF"):
        da_mon = anom_mon.sel(lev=lev)
        solver = XarrayEof(da_mon, weights=weights_da.values)
        eof1 = solver.eofs(neofs=1, eofscaling=0).squeeze()
        pc1 = solver.pcs(npcs=1, pcscaling=0).squeeze()
        flip = -1 if float(eof1.sel(lat=80.0, method="nearest").values) > 0 else 1
        pc1 = pc1 * flip
        mean = pc1.mean("time")
        std = pc1.std("time")
        nam_levels[float(lev)] = {"solver": solver, "flip": flip, "pc_mean": mean, "pc_std": std}
        eof_list.append((eof1 * flip).assign_coords(lev=float(lev)).expand_dims("lev"))
        pc_mean.append(float(mean.values))
        pc_std.append(float(std.values))
        flip_list.append(int(flip))

    ref["NAM_clim_doy_smooth"] = clim_doy_smooth
    ref["NAM_levels"] = nam_levels
    ref["NAM_eof1_unscaled"] = xr.concat(eof_list, dim="lev").rename("NAM_eof1_unscaled")
    ref["NAM_pc_mean"] = xr.DataArray(pc_mean, dims=("lev",), coords={"lev": z_valid.lev.values}, name="NAM_pc1_mean")
    ref["NAM_pc_std"] = xr.DataArray(pc_std, dims=("lev",), coords={"lev": z_valid.lev.values}, name="NAM_pc1_std")
    ref["NAM_flip"] = xr.DataArray(flip_list, dims=("lev",), coords={"lev": z_valid.lev.values}, name="NAM_flip_factor")
    return ref


def save_reference_modes(ref: Dict[str, object], out_file: Path, overwrite: bool) -> None:
    ds = xr.Dataset(
        {
            "AO_eof1_unscaled": ref["AO_eof1_unscaled"],
            "AO_eof1_project_scaling": ref["AO_eof1_project_scaling"],
            "NAM_eof1_unscaled": ref["NAM_eof1_unscaled"],
            "NAM_pc1_mean": ref["NAM_pc_mean"],
            "NAM_pc1_std": ref["NAM_pc_std"],
            "NAM_flip_factor": ref["NAM_flip"],
        }
    )
    ds.attrs.update(
        {
            "title": "B2000WCN001002 AO/NAM first EOF modes used for Hindcast projection",
            "source": str(B2000WCN_ROOT),
            "AO_method": "Longrun Code-B modified weights, 1000 hPa, 20-90N, zonal mean",
            "NAM_method": "Longrun monthly EOF by pressure level, daily projection climatology",
        }
    )
    write_netcdf_atomic(ds, out_file, overwrite=overwrite)


def load_member_z3_zm(paths: Dict[str, Path]) -> Tuple[xr.DataArray, xr.Dataset]:
    ds_z = open_dataset(paths["Z3"])
    if "PS" not in ds_z:
        if "PS" not in paths:
            raise FileNotFoundError("Z3 file has no PS and no matching PS file was discovered")
        ds_ps = open_dataset(paths["PS"])
        keep = [name for name in ["PS", "P0", "hyam", "hybm", "date", "datesec", "time_bnds", "gw"] if name in ds_ps]
        ds_z = xr.merge([ds_z, ds_ps[keep]], compat="override")
    p_mid = compute_pressure_mid(ds_z)
    z3 = clean_field(ds_z["Z3"]).sel(lat=slice(LAT_MIN, LAT_MAX))
    p_mid = p_mid.sel(lat=slice(LAT_MIN, LAT_MAX))
    z_plev = interp_profile_logp(z3, p_mid, TARGET_NAM_PLEV_PA)
    z_zm = z_plev.mean("lon", skipna=True).transpose("time", "plev", "lat")
    z_zm = z_zm.rename({"plev": "lev"}).assign_coords(lev=("lev", TARGET_NAM_PLEV_HPA))
    return z_zm.load(), ds_z


def project_member(payload) -> Tuple[str, Optional[xr.Dataset], Dict[str, str]]:
    case_name, member_id, paths, ref = payload
    record = {"case": case_name, "member": member_id, "status": "ok", "message": ""}
    ds_z = None
    try:
        z_zm, ds_z = load_member_z3_zm(paths)
        if "date" not in ds_z:
            raise KeyError("Z3 file has no CAM date variable")
        doy = date_to_doy(ds_z["date"].values)
        lead_time = np.arange(z_zm.sizes["time"], dtype=np.int16)

        z_ao = z_zm.sel(lev=AO_PLEV_HPA, method="nearest").interp(lat=ref["AO_lat_target"])
        ao_anom = z_ao - take_clim_by_doy(ref["AO_clim_doy"], doy, z_ao["time"])
        gh_layer = np.asarray(ao_anom, dtype=np.float64)
        ao = ref["AO_solver"].projectField(gh_layer, neofs=1, eofscaling=1, weighted=True)
        ao = np.reshape(np.asarray(ao), (gh_layer.shape[0],)) * int(ref["AO_flip"])

        nam_levels = []
        anom_doy = z_zm - take_clim_by_doy(ref["NAM_clim_doy_smooth"], doy, z_zm["time"])
        for lev in z_zm.lev.values:
            da_day = anom_doy.sel(lev=lev)
            ref_level = ref["NAM_levels"][float(lev)]
            projected = ref_level["solver"].projectField(da_day, neofs=1, eofscaling=0).squeeze()
            nam_day = (projected * ref_level["flip"] - ref_level["pc_mean"]) / ref_level["pc_std"]
            nam_levels.append(nam_day.assign_coords(lev=float(lev)).expand_dims("lev"))
        nam = xr.concat(nam_levels, dim="lev").transpose("time", "lev")

        ds_out = xr.Dataset(
            {
                "AO_Index": ("lead_time", ao.astype(np.float32)),
                "NAM_Vertical": (("lead_time", "lev"), nam.values.astype(np.float32)),
                "date": ("lead_time", ds_z["date"].values.astype(np.int32)),
            },
            coords={"lead_time": lead_time, "lev": z_zm.lev.values.astype(np.float64)},
        )
        ds_out["lev"].attrs.update({"units": "hPa", "long_name": "pressure"})
        ds_out["AO_Index"].attrs["reference"] = "B2000WCN001002 AO EOF1"
        ds_out["NAM_Vertical"].attrs["reference"] = "B2000WCN001002 NAM EOF1 by pressure level"
        return member_id, ds_out.load(), record
    except Exception as exc:
        record["status"] = "error"
        record["message"] = f"{type(exc).__name__}: {exc}"
        return member_id, None, record
    finally:
        if ds_z is not None:
            ds_z.close()
        gc.collect()


def write_summary(case_root: Path, records: List[Dict[str, str]], overwrite: bool) -> Path:
    out_dir = case_root / OUTPUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{case_root.name}_AO_NAM_projection_summary.csv"
    if out_file.exists() and not overwrite:
        out_file = out_dir / f"{case_root.name}_AO_NAM_projection_summary.latest.csv"
    with out_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case", "member", "status", "message"])
        writer.writeheader()
        writer.writerows(records)
    return out_file


def process_case(case_root: Path, args, ref) -> None:
    out_file = case_root / OUTPUT_SUBDIR / f"{case_root.name}_AO_NAM_B2000WCN_projection_members.nc"
    if out_file.exists() and out_file.stat().st_size > 0 and not args.overwrite:
        print(f"[SKIP] {case_root.name}: existing {out_file}")
        return
    inputs = discover_member_inputs(case_root, required_vars=("Z3",), optional_vars=("PS",), members=args.members)
    if not inputs:
        print(f"[SKIP] {case_root.name}: no Z3 members")
        return
    payloads = [(case_root.name, mid, paths, ref) for mid, paths in inputs.items()]
    results: List[Tuple[str, xr.Dataset]] = []
    records: List[Dict[str, str]] = []
    # Reference solver objects are not pickle-friendly; use one worker unless the
    # user later ports projection to a saved-pattern formula.
    for payload in tqdm(payloads, desc=f"AO/NAM {case_root.name}"):
        member_id, ds, rec = project_member(payload)
        records.append(rec)
        if ds is not None:
            results.append((member_id, ds))
    print(f"[SUMMARY] {write_summary(case_root, sorted(records, key=lambda r: r['member']), args.overwrite)}")
    if not results:
        print(f"[WARN] {case_root.name}: no AO/NAM outputs collected")
        return
    results.sort(key=lambda item: item[0])
    ds_full = xr.concat(
        [item[1] for item in results],
        dim=xr.DataArray([item[0] for item in results], dims="member", name="member"),
        join="outer",
    )
    ds_full.attrs.update(
        {
            "title": f"Hindcast AO/NAM projected onto B2000WCN001002 first EOF modes, {case_root.name}",
            "case_name": case_root.name,
            "reference_case": "B2000WCN001002_timefixed",
            "projection_note": "Hindcast AO/NAM are not self-trained EOFs; they use B2000WCN001002 reference modes.",
        }
    )
    write_netcdf_atomic(ds_full, out_file, overwrite=args.overwrite)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_default_args(parser)
    parser.add_argument("--reference-root", type=Path, default=B2000WCN_ROOT)
    parser.add_argument(
        "--reference-mode-file",
        type=Path,
        default=B2000WCN_ROOT / "NAM" / "B2000WCN001002_AO_NAM_mode1_patterns.nc",
        help="Where to save the B2000WCN mode pattern NetCDF.",
    )
    parser.add_argument(
        "--max-reference-years",
        type=int,
        default=None,
        help="Debug option: use only the first N B2000WCN years when rebuilding modes.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    z3_ref = load_b2000wcn_z3_zm(args.reference_root, max_workers=args.max_workers, max_years=args.max_reference_years)
    ref = build_reference(z3_ref)
    save_reference_modes(ref, args.reference_mode_file, overwrite=args.overwrite)
    for case_root in parse_case_list(args.root, args.cases):
        process_case(case_root, args, ref)


if __name__ == "__main__":
    main()
