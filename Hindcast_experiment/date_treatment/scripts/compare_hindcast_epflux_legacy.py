#!/usr/bin/env python3
"""Compare legacy Hindcast EPflux_daily/Fz files with new EPflux_daily_ubar_wcorr.

The legacy Fz files are 40-80N cos-lat means and were documented as w=None.
This script reduces the new all_waves ep2 to the same latitude band and writes
per-member statistics for quality control, not pass/fail equivalence.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import xarray as xr

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hindcast_common import HINDCAST_ROOT, area_weights_for_lat, parse_case_list


def member_id_from_fz(path: Path) -> str:
    suffix = ".Fz.nc"
    name = path.name
    if not name.endswith(suffix):
        raise ValueError(f"Unexpected Fz filename: {name}")
    return name[: -len(suffix)]


def corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def compare_case(case_root: Path, overwrite: bool) -> Optional[Path]:
    new_file = case_root / "EPflux_daily_ubar_wcorr" / "all_waves" / f"EPFLUX_all_waves_{case_root.name}_members_time_plev_lat.nc"
    legacy_dir = case_root / "EPflux_daily"
    if not new_file.exists() or not legacy_dir.is_dir():
        print(f"[SKIP] {case_root.name}: missing new EPFLUX or legacy EPflux_daily")
        return None

    out_dir = case_root / "EPflux_daily_ubar_wcorr" / "quality_control"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{case_root.name}_legacy_Fz_vs_new_ep2_40_80N.csv"
    if out_file.exists() and not overwrite:
        out_file = out_dir / f"{case_root.name}_legacy_Fz_vs_new_ep2_40_80N.latest.csv"

    rows: List[Dict[str, object]] = []
    with xr.open_dataset(new_file, decode_times=False) as ds_new:
        weights = area_weights_for_lat(ds_new["lat"], 40.0, 80.0)
        new_fz = ds_new["ep2"].weighted(weights).mean("lat")
        member_values = {str(v) for v in ds_new["member"].values}

        for old_file in sorted(legacy_dir.glob("*.Fz.nc")):
            member = member_id_from_fz(old_file)
            if member not in member_values:
                rows.append({"case": case_root.name, "member": member, "status": "missing_new_member"})
                continue
            try:
                with xr.open_dataset(old_file, decode_times=False) as ds_old:
                    old = ds_old["Fz"]
                    new = new_fz.sel(member=member)
                    ntime = min(int(old.sizes["time"]), int(new.sizes["lead_time"]))
                    nplev = min(int(old.sizes["plev"]), int(new.sizes["plev"]))
                    old_arr = old.isel(time=slice(0, ntime), plev=slice(0, nplev)).values.astype(float)
                    new_arr = new.isel(lead_time=slice(0, ntime), plev=slice(0, nplev)).values.astype(float)
                    diff = new_arr - old_arr
                    rows.append(
                        {
                            "case": case_root.name,
                            "member": member,
                            "status": "ok",
                            "ntime": ntime,
                            "nplev": nplev,
                            "old_mean": float(np.nanmean(old_arr)),
                            "new_mean": float(np.nanmean(new_arr)),
                            "bias_new_minus_old": float(np.nanmean(diff)),
                            "rmse": float(np.sqrt(np.nanmean(diff**2))),
                            "max_abs_diff": float(np.nanmax(np.abs(diff))),
                            "corr": corrcoef(old_arr.ravel(), new_arr.ravel()),
                        }
                    )
            except Exception as exc:
                rows.append({"case": case_root.name, "member": member, "status": "error", "message": f"{type(exc).__name__}: {exc}"})

    fieldnames = [
        "case",
        "member",
        "status",
        "ntime",
        "nplev",
        "old_mean",
        "new_mean",
        "bias_new_minus_old",
        "rmse",
        "max_abs_diff",
        "corr",
        "message",
    ]
    with out_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[WRITE] {out_file}")
    return out_file


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=HINDCAST_ROOT)
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    for case_root in parse_case_list(args.root, args.cases):
        compare_case(case_root, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
