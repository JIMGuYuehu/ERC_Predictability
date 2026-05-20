#!/usr/bin/env python3
"""Audit OMEGA availability in raw Hindcast h3 source files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Optional

from netCDF4 import Dataset


DEFAULT_SOURCE_ROOT = Path("/mnt/backup_ETH/lens")
DEFAULT_OUT = Path("Hindcast_experiment/date_treatment/hindcast_raw_omega_inventory.csv")


def has_variable(path: Path, varname: str) -> bool:
    with Dataset(path, "r") as ds:
        return varname in ds.variables


def audit_case(case_dir: Path, varname: str) -> dict:
    files = sorted(case_dir.glob("*.cam.h3.*.nc*"))
    n_var = 0
    n_error = 0
    first_with = ""
    first_without = ""
    first_error = ""

    for path in files:
        try:
            has_var = has_variable(path, varname)
        except Exception as exc:
            n_error += 1
            if not first_error:
                first_error = f"{path.name}: {type(exc).__name__}: {exc}"
            continue

        if has_var:
            n_var += 1
            if not first_with:
                first_with = path.name
        elif not first_without:
            first_without = path.name

    return {
        "case": case_dir.name,
        "is_nocoupl": int("NOCOUPL" in case_dir.name),
        "n_h3": len(files),
        f"n_{varname}": n_var,
        "n_error": n_error,
        f"first_with_{varname}": first_with,
        f"first_without_{varname}": first_without,
        "first_error": first_error,
    }


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--varname", default="OMEGA")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    case_dirs = sorted(path for path in args.source_root.iterdir() if path.is_dir())
    rows = [audit_case(case_dir, args.varname) for case_dir in case_dirs]
    rows = [row for row in rows if row["n_h3"] > 0]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["case"]
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    nocoupl = [row for row in rows if row["is_nocoupl"]]
    coupled = [row for row in rows if not row["is_nocoupl"]]
    n_key = f"n_{args.varname}"
    print(f"[WRITE] {args.out}")
    print(
        "NOCOUPL",
        "cases",
        len(nocoupl),
        "h3_files",
        sum(row["n_h3"] for row in nocoupl),
        args.varname,
        sum(row[n_key] for row in nocoupl),
    )
    print(
        "COUPLED",
        "cases",
        len(coupled),
        "h3_files",
        sum(row["n_h3"] for row in coupled),
        args.varname,
        sum(row[n_key] for row in coupled),
    )


if __name__ == "__main__":
    main()
