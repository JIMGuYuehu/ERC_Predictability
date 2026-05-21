#!/usr/bin/env python3
"""Shared helpers for Hindcast diagnostics.

All paths default to the data layout under /mnt/soclim0/public_data/weiji/Hindcast.
The functions here only write when a caller explicitly asks them to save an output.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import xarray as xr


PUBLIC_DATA_ROOT = Path("/mnt/soclim0/public_data/weiji")
HINDCAST_ROOT = PUBLIC_DATA_ROOT / "Hindcast"
B2000WCN_ROOT = PUBLIC_DATA_ROOT / "B2000WCN001002_timefixed"

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
    dtype=np.float64,
)
PLEV_STD_HPA = PLEV_STD_PA / 100.0

TARGET_NAM_PLEV_HPA = np.array(
    [
        1000.0,
        950.0,
        900.0,
        850.0,
        800.0,
        750.0,
        700.0,
        600.0,
        550.0,
        500.0,
        450.0,
        400.0,
        350.0,
        300.0,
        250.0,
        225.0,
        200.0,
        175.0,
        150.0,
        125.0,
        100.0,
        70.0,
        50.0,
        30.0,
        20.0,
        10.0,
        7.0,
        5.0,
        3.0,
        2.0,
        1.0,
    ],
    dtype=np.float64,
)
TARGET_NAM_PLEV_PA = TARGET_NAM_PLEV_HPA * 100.0

VALID_ABS_MAX = 1.0e20
NETCDF_ENGINE = "netcdf4"
COMPLEVEL = 1
YEAR_RE = re.compile(r"\.h3\.(\d{4})")


def add_default_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--root", type=Path, default=HINDCAST_ROOT, help="Hindcast root directory.")
    parser.add_argument("--cases", nargs="*", default=None, help="Optional case names to process.")
    parser.add_argument("--members", nargs="*", default=None, help="Optional member ids or numeric member suffixes.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing non-empty outputs.")
    parser.add_argument("--max-workers", type=int, default=4, help="Worker count for scripts that parallelize by member.")
    return parser


def parse_case_list(root: Path, cases: Optional[Sequence[str]]) -> List[Path]:
    root = Path(root)
    if cases:
        case_dirs = [root / case for case in cases]
    else:
        case_dirs = [p for p in sorted(root.iterdir()) if p.is_dir() and re.match(r"^\d{4}-\d{2}", p.name)]
    missing = [str(p) for p in case_dirs if not p.is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing Hindcast case directories: {missing}")
    return case_dirs


def parse_year_from_name(path: Path) -> Optional[int]:
    m = YEAR_RE.search(path.name)
    return int(m.group(1)) if m else None


def member_id_from_var_file(path: Path, var: str) -> str:
    suffix = f".{var}.nc"
    name = Path(path).name
    if not name.endswith(suffix):
        raise ValueError(f"Cannot strip variable suffix {suffix!r} from {name!r}")
    return name[: -len(suffix)]


def member_number(member_id: str) -> Optional[str]:
    m = re.search(r"\.(\d{3})\.cam\.h3$", member_id)
    return m.group(1) if m else None


def normalize_member_filter(members: Optional[Sequence[str]]) -> Optional[set[str]]:
    if not members:
        return None
    out: set[str] = set()
    for item in members:
        out.add(str(item))
        if str(item).isdigit():
            out.add(f"{int(item):03d}")
    return out


def discover_member_inputs(
    case_root: Path,
    required_vars: Sequence[str],
    optional_vars: Sequence[str] = (),
    members: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, Path]]:
    """Return member -> variable -> file mapping.

    Required variables must exist for the member to be returned. Optional variables
    are included when present.
    """

    case_root = Path(case_root)
    member_filter = normalize_member_filter(members)
    by_var: Dict[str, Dict[str, Path]] = {}
    for var in list(required_vars) + list(optional_vars):
        var_dir = case_root / var
        mapping: Dict[str, Path] = {}
        if var_dir.is_dir():
            for path in sorted(var_dir.glob(f"*.{var}.nc")):
                mid = member_id_from_var_file(path, var)
                num = member_number(mid)
                if member_filter and mid not in member_filter and (num not in member_filter):
                    continue
                mapping[mid] = path
        by_var[var] = mapping

    common: Optional[set[str]] = None
    for var in required_vars:
        ids = set(by_var[var])
        common = ids if common is None else common & ids
    common = common or set()

    out: Dict[str, Dict[str, Path]] = {}
    for mid in sorted(common):
        entry = {var: by_var[var][mid] for var in required_vars}
        for var in optional_vars:
            if mid in by_var[var]:
                entry[var] = by_var[var][mid]
        out[mid] = entry
    return out


def open_dataset(path: Path) -> xr.Dataset:
    try:
        return xr.open_dataset(path, decode_times=False, engine=NETCDF_ENGINE)
    except Exception:
        return xr.open_dataset(path, decode_times=False)


def clean_field(da: xr.DataArray) -> xr.DataArray:
    return da.where(np.isfinite(da) & (np.abs(da) < VALID_ABS_MAX))


def compute_pressure_mid(ds: xr.Dataset, ps_da: Optional[xr.DataArray] = None) -> xr.DataArray:
    ps = ds["PS"] if ps_da is None else ps_da
    p0 = ds["P0"] if "P0" in ds else xr.DataArray(100000.0)
    return ds["hyam"] * p0 + ds["hybm"] * ps


def compute_pressure_interface(ds: xr.Dataset, ps_da: Optional[xr.DataArray] = None) -> xr.DataArray:
    ps = ds["PS"] if ps_da is None else ps_da
    p0 = ds["P0"] if "P0" in ds else xr.DataArray(100000.0)
    return ds["hyai"] * p0 + ds["hybi"] * ps


def interp_profile_logp(v_hyb: xr.DataArray, p_hyb: xr.DataArray, p_tgt_pa: Sequence[float]) -> xr.DataArray:
    p_tgt_pa = np.asarray(p_tgt_pa, dtype=np.float64)

    def _interp_col(vcol, pcol):
        vcol = np.asarray(vcol, dtype=np.float64)
        pcol = np.asarray(pcol, dtype=np.float64)
        valid = np.isfinite(vcol) & np.isfinite(pcol) & (pcol > 0.0)
        if valid.sum() < 2:
            return np.full(p_tgt_pa.shape, np.nan, dtype=np.float64)
        p_use = pcol[valid]
        v_use = vcol[valid]
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
        v_hyb,
        p_hyb,
        input_core_dims=[["lev"], ["lev"]],
        output_core_dims=[["plev"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64],
        dask_gufunc_kwargs={"output_sizes": {"plev": len(p_tgt_pa)}},
    )
    return out.assign_coords(plev=("plev", p_tgt_pa))


def area_weights_for_lat(lat: xr.DataArray, lat_min: float = 60.0, lat_max: float = 90.0) -> xr.DataArray:
    weights = np.cos(np.deg2rad(lat))
    return weights.where((lat >= lat_min) & (lat <= lat_max), 0.0).fillna(0.0)


def make_encoding(ds: xr.Dataset, float_dtype: str = "float32") -> Dict[str, Mapping[str, object]]:
    encoding: Dict[str, Mapping[str, object]] = {}
    for name, da in ds.variables.items():
        enc: Dict[str, object] = {}
        if name in ds.data_vars:
            enc.update({"zlib": True, "complevel": COMPLEVEL})
        if np.issubdtype(da.dtype, np.floating) and name in ds.data_vars:
            enc.update({"dtype": float_dtype, "_FillValue": np.float32(np.nan)})
        encoding[name] = enc
    return encoding


def write_netcdf_atomic(ds: xr.Dataset, out_file: Path, overwrite: bool = False) -> Path:
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if out_file.exists() and out_file.stat().st_size > 0 and not overwrite:
        print(f"[SKIP] existing: {out_file}")
        return out_file
    tmp = out_file.with_name(out_file.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    print(f"[WRITE] {out_file}")
    ds.to_netcdf(tmp, engine=NETCDF_ENGINE, format="NETCDF4", encoding=make_encoding(ds))
    tmp.replace(out_file)
    return out_file


def dataset_from_member_arrays(
    data_vars: Mapping[str, Tuple[Tuple[str, ...], np.ndarray]],
    member_ids: Sequence[str],
    attrs: Optional[Mapping[str, object]] = None,
    extra_coords: Optional[Mapping[str, object]] = None,
) -> xr.Dataset:
    coords = {"member": np.asarray(member_ids, dtype=object)}
    if extra_coords:
        coords.update(extra_coords)
    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=dict(attrs or {}))
    ds["member"].attrs["description"] = "Hindcast ensemble member id parsed from file prefix"
    return ds
