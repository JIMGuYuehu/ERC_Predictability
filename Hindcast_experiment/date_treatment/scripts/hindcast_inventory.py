#!/usr/bin/env python3
"""Inventory Hindcast inputs and Longrun-style derived outputs."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hindcast_common import HINDCAST_ROOT, parse_case_list


INPUT_VARS = ("U", "V", "T", "OMEGA", "PS", "Z3", "O3")


def count_files(case_root: Path, subdir: str, pattern: str = "*.nc") -> int:
    path = case_root / subdir
    return len(list(path.glob(pattern))) if path.is_dir() else 0


def inventory_case(case_root: Path) -> dict:
    row = {"case": case_root.name}
    for var in INPUT_VARS:
        row[f"n_{var}"] = count_files(case_root, var, f"*.{var}.nc")
    row["n_legacy_EPflux_Fz"] = count_files(case_root, "EPflux_daily", "*.Fz.nc")
    row["has_new_EPflux_ubar_wcorr"] = int(
        (case_root / "EPflux_daily_ubar_wcorr" / "all_waves" / f"EPFLUX_all_waves_{case_root.name}_members_time_plev_lat.nc").exists()
    )
    row["has_EHF"] = int((case_root / "Eddyheatflux_daily" / f"EHF_{case_root.name}_members_time_plev_lat.nc").exists())
    row["has_AO_NAM_B2000WCN_projection"] = int(
        (case_root / "NAM_B2000WCN_projection" / f"{case_root.name}_AO_NAM_B2000WCN_projection_members.nc").exists()
    )
    row["has_partial_O3"] = int((case_root / "partial_O3" / f"{case_root.name}_partial_O3_all_ranges_members.nc").exists())
    row["has_FWD"] = int((case_root / "final_warming_date" / f"{case_root.name}_FWD_plev_member.nc").exists())
    row["needs_OMEGA_extract"] = int(row["n_OMEGA"] < row["n_U"])
    row["needs_new_EPflux"] = int(not row["has_new_EPflux_ubar_wcorr"])
    row["needs_EHF"] = int(not row["has_EHF"])
    row["needs_AO_NAM"] = int(not row["has_AO_NAM_B2000WCN_projection"])
    row["needs_partial_O3"] = int(not row["has_partial_O3"])
    row["needs_FWD"] = int(not row["has_FWD"])
    return row


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=HINDCAST_ROOT)
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--out", type=Path, default=Path("Hindcast_experiment/date_treatment/hindcast_inventory.csv"))
    args = parser.parse_args(argv)

    rows = [inventory_case(case_root) for case_root in parse_case_list(args.root, args.cases)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["case"]
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[WRITE] {args.out}")
    for row in rows:
        print(
            row["case"],
            "OMEGA",
            f"{row['n_OMEGA']}/{row['n_U']}",
            "new_EPflux",
            row["has_new_EPflux_ubar_wcorr"],
            "EHF",
            row["has_EHF"],
            "AO/NAM",
            row["has_AO_NAM_B2000WCN_projection"],
            "partial_O3",
            row["has_partial_O3"],
            "FWD",
            row["has_FWD"],
        )


if __name__ == "__main__":
    main()
