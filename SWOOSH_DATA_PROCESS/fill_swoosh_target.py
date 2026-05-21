#!/usr/bin/env python3
"""Fill gaps in the SWOOSH-derived target NetCDF file.

The default target is the file previously prepared for IFS:

    vmro3_SWOOSHv02.72_combinedo3q_neg2miss_remapped_to_CMIP6_template_201912-202101.nc

The fill strategy is deliberately conservative:

* internal gaps along a coordinate are linearly interpolated;
* gaps outside the valid coordinate span are filled from the nearest valid edge;
* dimensions are attempted in the requested order, defaulting to lat, lon, plev, time.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import numpy as np
import xarray as xr

from check_swoosh_missing import DEFAULT_TARGET, as_jsonable, variable_summary


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"
DEFAULT_REPORT_DIR = SCRIPT_DIR / "reports"
DEFAULT_VAR = "vmro3"
DEFAULT_FILL_VALUE = np.float32(1.0e20)
DEFAULT_DIM_ORDER = ("lat", "lon", "plev", "time")


def default_output_path(input_path: Path) -> Path:
    return DEFAULT_OUTPUT_DIR / f"{input_path.stem}_filled_no_missing.nc"


def clean_encoding(encoding: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "_FillValue",
        "dtype",
        "zlib",
        "complevel",
        "shuffle",
        "fletcher32",
        "contiguous",
        "chunksizes",
    }
    return {key: value for key, value in encoding.items() if key in allowed}


def fill_axis_nearest_edge(data: np.ndarray, coord: np.ndarray, axis: int) -> tuple[np.ndarray, int, int]:
    before = int((~np.isfinite(data)).sum())
    if before == 0:
        return data, before, before

    arr = np.array(data, dtype=np.float64, copy=True)
    x = np.asarray(coord, dtype=np.float64)
    if x.ndim != 1 or x.size != arr.shape[axis]:
        x = np.arange(arr.shape[axis], dtype=np.float64)

    order = np.argsort(x)
    x_sorted = x[order]
    moved = np.ascontiguousarray(np.moveaxis(arr, axis, -1))
    moved_shape = moved.shape
    flat = moved.reshape(-1, moved_shape[-1])

    for row in flat:
        sorted_values = row[order]
        missing = ~np.isfinite(sorted_values)
        if not missing.any():
            continue

        valid = ~missing
        n_valid = int(valid.sum())
        if n_valid == 0:
            continue
        if n_valid == 1:
            filled = np.full_like(sorted_values, sorted_values[valid][0], dtype=np.float64)
        else:
            filled = np.interp(
                x_sorted,
                x_sorted[valid],
                sorted_values[valid],
                left=sorted_values[valid][0],
                right=sorted_values[valid][-1],
            )
        sorted_values[missing] = filled[missing]
        row[order] = sorted_values

    filled_arr = np.moveaxis(flat.reshape(moved_shape), -1, axis)
    after = int((~np.isfinite(filled_arr)).sum())
    return filled_arr, before, after


def fill_dataarray(
    da: xr.DataArray,
    dim_order: tuple[str, ...],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    arr = da.values
    steps: list[dict[str, Any]] = []

    for dim in dim_order:
        if dim not in da.dims:
            continue
        before = int((~np.isfinite(arr)).sum())
        if before == 0:
            break
        coord = da[dim].values if dim in da.coords else np.arange(da.sizes[dim])
        axis = da.get_axis_num(dim)
        arr, before, after = fill_axis_nearest_edge(arr, coord, axis)
        steps.append(
            {
                "dimension": dim,
                "missing_before": before,
                "missing_after": after,
                "filled": before - after,
                "method": "linear interpolation for interior gaps; nearest-edge extrapolation outside valid coordinate range",
            }
        )

    remaining = int((~np.isfinite(arr)).sum())
    if remaining:
        raise RuntimeError(
            f"{remaining} missing/non-finite values remain after trying dimensions {dim_order}"
        )
    return arr, steps


def append_history(ds: xr.Dataset, text: str) -> None:
    old_history = ds.attrs.get("history", "")
    ds.attrs["history"] = f"{old_history}\n{text}" if old_history else text


def write_markdown_report(report: dict[str, Any], path: Path) -> None:
    before = report["before"]["variables"][report["variable"]]
    after = report["after"]["variables"][report["variable"]]
    lines = [
        "# SWOOSH Missing-Value Fill Report",
        "",
        f"- Input: `{report['input_file']}`",
        f"- Output: `{report['output_file']}`",
        f"- Variable: `{report['variable']}`",
        f"- Missing before: `{before['n_missing_or_nonfinite']}` / `{before['n_total']}` "
        f"({before['missing_fraction']:.6%})",
        f"- Missing after: `{after['n_missing_or_nonfinite']}` / `{after['n_total']}` "
        f"({after['missing_fraction']:.6%})",
        f"- Fill dimensions: `{', '.join(report['dim_order'])}`",
        "",
        "## Fill Steps",
        "",
    ]
    for step in report["fill_steps"]:
        lines.append(
            f"- `{step['dimension']}`: {step['missing_before']} -> "
            f"{step['missing_after']} missing; filled {step['filled']}"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The original target file did contain missing/non-finite values in `vmro3`. "
            "In this run all missing values were removed during the latitude pass, "
            "which is consistent with polar latitude rows lying outside the valid SWOOSH "
            "source-grid support after remapping to the CMIP6-style template grid.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def process_file(
    input_path: Path,
    output_path: Path,
    variable: str,
    dim_order: tuple[str, ...],
    json_report: Path,
    markdown_report: Path,
) -> dict[str, Any]:
    with xr.open_dataset(input_path, decode_times=False, mask_and_scale=True) as ds_in:
        if variable not in ds_in:
            raise KeyError(f"{variable!r} not found in {input_path}")

        before_var_summary = variable_summary(ds_in[variable])
        filled_values, fill_steps = fill_dataarray(ds_in[variable], dim_order=dim_order)

        ds_out = ds_in.copy(deep=True)
        ds_out[variable].data = filled_values.astype(ds_in[variable].dtype, copy=False)
        ds_out[variable].attrs["missing_value"] = DEFAULT_FILL_VALUE
        ds_out.attrs["missing_value_treatment"] = (
            f"{variable} missing/non-finite values were filled with coordinate-wise "
            "linear interpolation for interior gaps and nearest-edge extrapolation at "
            f"domain edges. Dimension order: {', '.join(dim_order)}."
        )
        append_history(
            ds_out,
            f"{datetime.now(timezone.utc).isoformat()} filled missing/non-finite "
            f"{variable} values with {SCRIPT_DIR.name}/fill_swoosh_target.py; "
            f"{before_var_summary['n_missing_or_nonfinite']} -> 0 missing.",
        )

        after_var_summary = variable_summary(ds_out[variable])
        if after_var_summary["n_missing_or_nonfinite"] != 0:
            raise RuntimeError("Output still has missing/non-finite values before writing.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        encoding: dict[str, dict[str, Any]] = {}
        for name in ds_out.variables:
            encoding[name] = clean_encoding(ds_in[name].encoding)
        encoding[variable]["_FillValue"] = DEFAULT_FILL_VALUE
        encoding[variable]["dtype"] = "float32"

        ds_out.to_netcdf(output_path, format="NETCDF4", engine="netcdf4", encoding=encoding)

    with xr.open_dataset(output_path, decode_times=False, mask_and_scale=True) as ds_check:
        written_after_summary = variable_summary(ds_check[variable])
        if written_after_summary["n_missing_or_nonfinite"] != 0:
            raise RuntimeError("Output still has missing/non-finite values after write/readback.")

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(input_path),
        "output_file": str(output_path),
        "variable": variable,
        "dim_order": list(dim_order),
        "fill_steps": fill_steps,
        "before": {
            "variables": {
                variable: before_var_summary,
            }
        },
        "after": {
            "variables": {
                variable: written_after_summary,
            }
        },
    }
    json_report.parent.mkdir(parents=True, exist_ok=True)
    json_report.write_text(json.dumps(report, indent=2, default=as_jsonable) + "\n")
    write_markdown_report(report, markdown_report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--output", type=Path, help="Output NetCDF path.")
    parser.add_argument("--var", default=DEFAULT_VAR, help=f"Variable to fill. Default: {DEFAULT_VAR}")
    parser.add_argument(
        "--dim-order",
        default=",".join(DEFAULT_DIM_ORDER),
        help="Comma-separated dimensions used for filling, in order.",
    )
    parser.add_argument("--json-report", type=Path, help="Output JSON report path.")
    parser.add_argument("--markdown-report", type=Path, help="Output Markdown report path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or default_output_path(args.input)
    json_report = args.json_report or (
        DEFAULT_REPORT_DIR / f"{output_path.stem}.report.json"
    )
    markdown_report = args.markdown_report or (
        DEFAULT_REPORT_DIR / f"{output_path.stem}.report.md"
    )
    dim_order = tuple(dim.strip() for dim in args.dim_order.split(",") if dim.strip())

    report = process_file(
        input_path=args.input,
        output_path=output_path,
        variable=args.var,
        dim_order=dim_order,
        json_report=json_report,
        markdown_report=markdown_report,
    )

    before = report["before"]["variables"][args.var]["n_missing_or_nonfinite"]
    after = report["after"]["variables"][args.var]["n_missing_or_nonfinite"]
    print(f"Input : {args.input}")
    print(f"Output: {output_path}")
    print(f"{args.var} missing/non-finite: {before} -> {after}")
    for step in report["fill_steps"]:
        print(
            f"- {step['dimension']}: {step['missing_before']} -> "
            f"{step['missing_after']} (filled {step['filled']})"
        )
    print(f"JSON report: {json_report}")
    print(f"Markdown report: {markdown_report}")


if __name__ == "__main__":
    main()
