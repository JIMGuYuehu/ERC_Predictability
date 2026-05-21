#!/usr/bin/env python3
"""Check actual missing/non-finite values in the SWOOSH-derived NetCDF file."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import numpy as np
import xarray as xr


DEFAULT_TARGET = Path(
    "/mnt/soclim0/public_data/weiji/swoosh/processed_like_input4MIPs/"
    "vmro3_SWOOSHv02.72_combinedo3q_neg2miss_remapped_to_CMIP6_template_201912-202101.nc"
)


def as_jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def numeric_variables(ds: xr.Dataset, include_coords: bool) -> list[str]:
    names = list(ds.data_vars)
    if include_coords:
        for name in ds.coords:
            if name not in names:
                names.append(name)
        for name in ds.variables:
            if name not in names:
                names.append(name)
    return [name for name in names if np.issubdtype(ds[name].dtype, np.number)]


def variable_summary(da: xr.DataArray, sample_limit: int = 24) -> dict[str, Any]:
    values = da.values
    missing_mask = ~np.isfinite(values)
    n_total = int(values.size)
    n_missing = int(missing_mask.sum())
    finite = values[np.isfinite(values)]

    summary: dict[str, Any] = {
        "dims": list(da.dims),
        "shape": [int(v) for v in da.shape],
        "dtype": str(da.dtype),
        "n_total": n_total,
        "n_missing_or_nonfinite": n_missing,
        "n_nan": int(np.isnan(values).sum()) if np.issubdtype(values.dtype, np.floating) else 0,
        "n_posinf": int(np.isposinf(values).sum()) if np.issubdtype(values.dtype, np.floating) else 0,
        "n_neginf": int(np.isneginf(values).sum()) if np.issubdtype(values.dtype, np.floating) else 0,
        "missing_fraction": float(n_missing / n_total) if n_total else 0.0,
        "n_negative_finite": int((finite < 0).sum()) if finite.size else 0,
        "finite_min": float(finite.min()) if finite.size else None,
        "finite_max": float(finite.max()) if finite.size else None,
        "finite_mean": float(finite.mean()) if finite.size else None,
        "encoded_fill_value": as_jsonable(da.encoding.get("_FillValue")),
        "attrs_missing_value": as_jsonable(da.attrs.get("missing_value")),
    }

    if n_missing and da.ndim:
        by_dim: dict[str, Any] = {}
        mask_da = xr.DataArray(missing_mask, dims=da.dims, coords=da.coords)
        for dim in da.dims:
            other_dims = [d for d in da.dims if d != dim]
            if not other_dims:
                all_missing = mask_da
            else:
                all_missing = mask_da.all(dim=other_dims)
            idx = np.flatnonzero(all_missing.values)
            coord_values = da[dim].values[idx] if dim in da.coords else idx
            by_dim[dim] = {
                "n_positions_all_missing": int(idx.size),
                "positions_sample": [
                    {"index": int(i), "value": as_jsonable(coord_values[j])}
                    for j, i in enumerate(idx[:sample_limit])
                ],
            }
        summary["all_missing_positions_by_dim"] = by_dim

    return summary


def build_report(
    input_path: Path,
    variables: list[str] | None,
    include_coords: bool,
    sample_limit: int,
) -> dict[str, Any]:
    with xr.open_dataset(input_path, decode_times=False, mask_and_scale=True) as ds:
        selected = variables or numeric_variables(ds, include_coords=include_coords)
        missing = [name for name in selected if name not in ds.variables]
        if missing:
            raise KeyError(f"Variables not found in {input_path}: {missing}")

        var_reports = {name: variable_summary(ds[name], sample_limit=sample_limit) for name in selected}
        total_missing = int(sum(item["n_missing_or_nonfinite"] for item in var_reports.values()))
        return {
            "input_file": str(input_path),
            "checked_variables": selected,
            "n_checked_variables": len(selected),
            "total_missing_or_nonfinite": total_missing,
            "has_missing_or_nonfinite": bool(total_missing),
            "variables": var_reports,
        }


def print_human(report: dict[str, Any]) -> None:
    print(f"Input: {report['input_file']}")
    print(f"Checked variables: {report['n_checked_variables']}")
    print(f"Total missing/non-finite: {report['total_missing_or_nonfinite']}")
    for name, item in report["variables"].items():
        print(
            f"- {name}: missing={item['n_missing_or_nonfinite']}/{item['n_total']} "
            f"({item['missing_fraction']:.6%}), "
            f"finite_min={item['finite_min']}, finite_max={item['finite_max']}"
        )
        for dim, dim_info in item.get("all_missing_positions_by_dim", {}).items():
            count = dim_info["n_positions_all_missing"]
            if count:
                sample = ", ".join(
                    f"{entry['value']}" for entry in dim_info["positions_sample"][:8]
                )
                print(f"  all-missing {dim} positions: {count}; sample: {sample}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_TARGET)
    parser.add_argument(
        "--var",
        action="append",
        dest="variables",
        help="Variable to check. Repeat to check multiple variables. Defaults to numeric data variables.",
    )
    parser.add_argument(
        "--include-coords",
        action="store_true",
        help="Also check numeric coordinates and bounds variables.",
    )
    parser.add_argument("--json-out", type=Path, help="Optional path for the JSON report.")
    parser.add_argument("--sample-limit", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(
        input_path=args.input,
        variables=args.variables,
        include_coords=args.include_coords,
        sample_limit=args.sample_limit,
    )
    print_human(report)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, default=as_jsonable) + "\n")


if __name__ == "__main__":
    main()
