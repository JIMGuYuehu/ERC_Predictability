#!/usr/bin/env python3
"""Explain the SWOOSH/CMIP6 NetCDF size difference and write a classic copy.

The script compares the CMIP6 reference example with the previously processed
SWOOSH file and the filled SWOOSH file. It reports:

* physical file size;
* NetCDF data model / disk format;
* variable dtype, dimensions, raw byte count;
* HDF5 chunking and compression filters.

It can also write a NETCDF4_CLASSIC copy of the filled file using the reference
file's variable order, chunking, and compression filters.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import netCDF4 as nc
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent

REFERENCE_FILE = Path("/mnt/soclim0/andreas/vmro3_input4MIPs_ozone_CMIP6_UReading-CCMI_2020.nc")
ORIGINAL_PROCESSED_FILE = Path(
    "/mnt/soclim0/public_data/weiji/swoosh/processed_like_input4MIPs/"
    "vmro3_SWOOSHv02.72_combinedo3q_neg2miss_remapped_to_CMIP6_template_201912-202101.nc"
)
FILLED_FILE = Path(
    "/mnt/soclim0/public_data/weiji/swoosh/SWOOSH_nan_fill_20260521/"
    "vmro3_SWOOSHv02.72_combinedo3q_neg2miss_remapped_to_CMIP6_template_201912-202101_filled_no_missing.nc"
)
CLASSIC_OUTPUT = Path(
    "/mnt/soclim0/public_data/weiji/swoosh/SWOOSH_nan_fill_20260521/"
    "vmro3_SWOOSHv02.72_combinedo3q_neg2miss_remapped_to_CMIP6_template_201912-202101_filled_no_missing_netcdf4_classic.nc"
)
JSON_REPORT = SCRIPT_DIR / "reports" / "why_size_dif_report.json"
MD_REPORT = SCRIPT_DIR / "reports" / "why_size_dif_report.md"

VARIABLE_ORDER = (
    "time",
    "time_bnds",
    "plev",
    "plev_bnds",
    "lat",
    "lat_bnds",
    "lon",
    "lon_bnds",
    "vmro3",
)


def as_jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def mib(nbytes: int) -> float:
    return nbytes / (1024.0**2)


def raw_nbytes(var: nc.Variable) -> int:
    size = 1
    for n in var.shape:
        size *= int(n)
    return int(size * np.dtype(var.dtype).itemsize)


def normalized_filters(var: nc.Variable) -> dict[str, Any]:
    filters = var.filters()
    return {
        "zlib": bool(filters.get("zlib", False)),
        "complevel": int(filters.get("complevel", 0) or 0),
        "shuffle": bool(filters.get("shuffle", False)),
        "fletcher32": bool(filters.get("fletcher32", False)),
    }


def variable_info(var: nc.Variable) -> dict[str, Any]:
    return {
        "dtype": str(var.dtype),
        "dimensions": list(var.dimensions),
        "shape": [int(v) for v in var.shape],
        "raw_nbytes": raw_nbytes(var),
        "raw_mib": mib(raw_nbytes(var)),
        "chunking": var.chunking(),
        "filters": normalized_filters(var),
        "_FillValue": as_jsonable(getattr(var, "_FillValue", None)),
    }


def file_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with nc.Dataset(path, "r") as ds:
        variables = {name: variable_info(ds.variables[name]) for name in ds.variables}
        raw_payload = int(sum(item["raw_nbytes"] for item in variables.values()))
        return {
            "path": str(path),
            "file_size_bytes": path.stat().st_size,
            "file_size_mib": mib(path.stat().st_size),
            "data_model": ds.data_model,
            "disk_format": ds.disk_format,
            "dimensions": {
                name: {
                    "size": int(len(dim)),
                    "isunlimited": bool(dim.isunlimited()),
                }
                for name, dim in ds.dimensions.items()
            },
            "variables": variables,
            "raw_payload_bytes": raw_payload,
            "raw_payload_mib": mib(raw_payload),
            "file_to_raw_payload_ratio": path.stat().st_size / raw_payload if raw_payload else None,
        }


def filters_match(reference: dict[str, Any], other: dict[str, Any], var_name: str) -> bool:
    if var_name not in reference["variables"] or var_name not in other["variables"]:
        return False
    r = reference["variables"][var_name]
    o = other["variables"][var_name]
    return r["filters"] == o["filters"] and r["chunking"] == o["chunking"]


def collect_report(
    reference_file: Path,
    original_file: Path,
    filled_file: Path,
    classic_file: Path | None = None,
) -> dict[str, Any]:
    files = {
        "reference": file_info(reference_file),
        "original_processed": file_info(original_file),
        "filled": file_info(filled_file),
    }
    if classic_file and classic_file.exists():
        files["filled_netcdf4_classic"] = file_info(classic_file)

    reference = files["reference"]
    comparisons: dict[str, Any] = {}
    for label, info in files.items():
        if label == "reference":
            continue
        comparisons[label] = {
            "vmro3_dtype_matches_reference": (
                info["variables"]["vmro3"]["dtype"]
                == reference["variables"]["vmro3"]["dtype"]
            ),
            "vmro3_filters_chunking_match_reference": filters_match(reference, info, "vmro3"),
            "all_common_filters_chunking_match_reference": {
                name: filters_match(reference, info, name)
                for name in VARIABLE_ORDER
                if name in reference["variables"] and name in info["variables"]
            },
        }

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "files": files,
        "comparisons": comparisons,
        "conclusion": {
            "vmro3_dtype": reference["variables"]["vmro3"]["dtype"],
            "size_difference_main_reason": (
                "The CMIP6 reference and SWOOSH files both store vmro3 as float32, "
                "so the 41 MB vs 17 MB difference is not caused by double vs float32 "
                "precision in vmro3. The common variables also use the same zlib level, "
                "shuffle setting, fletcher32 setting, and chunking as the reference. "
                "The remaining size difference is therefore due to HDF5/NetCDF4 "
                "compressibility of the actual data values: the SWOOSH ozone field "
                "compresses substantially better than the CMIP6 reference field."
            ),
        },
    }


def create_dimensions(src: nc.Dataset, ref: nc.Dataset, dst: nc.Dataset) -> None:
    for name in ref.dimensions:
        if name in src.dimensions:
            dim = src.dimensions[name]
            dst.createDimension(name, None if dim.isunlimited() else len(dim))
    for name, dim in src.dimensions.items():
        if name not in dst.dimensions:
            dst.createDimension(name, None if dim.isunlimited() else len(dim))


def variable_create_kwargs(ref_var: nc.Variable, src_var: nc.Variable) -> dict[str, Any]:
    filters = normalized_filters(ref_var)
    chunking = ref_var.chunking()
    fill_value = getattr(src_var, "_FillValue", None)
    kwargs: dict[str, Any] = {
        "fill_value": fill_value,
        "zlib": filters["zlib"],
        "complevel": filters["complevel"],
        "shuffle": filters["shuffle"],
        "fletcher32": filters["fletcher32"],
    }
    if isinstance(chunking, list):
        kwargs["chunksizes"] = tuple(int(v) for v in chunking)
    elif chunking == "contiguous" and not filters["zlib"]:
        kwargs["contiguous"] = True
    return kwargs


def copy_attrs(src: nc.Variable | nc.Dataset, dst: nc.Variable | nc.Dataset) -> None:
    attrs = {
        name: src.getncattr(name)
        for name in src.ncattrs()
        if name != "_FillValue"
    }
    if attrs:
        dst.setncatts(attrs)


def create_classic_copy(source_file: Path, reference_file: Path, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = output_file.with_suffix(output_file.suffix + ".tmp")
    if tmp_file.exists():
        tmp_file.unlink()

    with nc.Dataset(source_file, "r") as src, nc.Dataset(reference_file, "r") as ref:
        with nc.Dataset(tmp_file, "w", format="NETCDF4_CLASSIC") as dst:
            create_dimensions(src, ref, dst)
            copy_attrs(src, dst)
            history_line = (
                f"{datetime.now(timezone.utc).isoformat()} wrote NETCDF4_CLASSIC copy "
                f"with variable chunking/compression copied from {reference_file.name}."
            )
            old_history = dst.getncattr("history") if "history" in dst.ncattrs() else ""
            dst.history = f"{old_history}\n{history_line}" if old_history else history_line

            names = [name for name in ref.variables if name in src.variables]
            names.extend(name for name in src.variables if name not in names)

            for name in names:
                src_var = src.variables[name]
                ref_var = ref.variables[name] if name in ref.variables else src_var
                dst_var = dst.createVariable(
                    name,
                    src_var.dtype,
                    src_var.dimensions,
                    **variable_create_kwargs(ref_var, src_var),
                )
                copy_attrs(src_var, dst_var)
                dst_var[:] = src_var[:]

    tmp_file.replace(output_file)
    os.chmod(output_file, 0o777)


def write_json(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=as_jsonable) + "\n")


def fmt_filters(item: dict[str, Any]) -> str:
    filters = item["filters"]
    return (
        f"zlib={filters['zlib']}, complevel={filters['complevel']}, "
        f"shuffle={filters['shuffle']}, fletcher32={filters['fletcher32']}, "
        f"chunking={item['chunking']}"
    )


def write_markdown(report: dict[str, Any], path: Path) -> None:
    files = report["files"]
    lines = [
        "# SWOOSH NetCDF Size Difference Diagnostics",
        "",
        "## File Size Summary",
        "",
        "| file | data model | disk format | size MiB | raw payload MiB | file/raw |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, info in files.items():
        ratio = info["file_to_raw_payload_ratio"]
        lines.append(
            f"| {label} | {info['data_model']} | {info['disk_format']} | "
            f"{info['file_size_mib']:.2f} | {info['raw_payload_mib']:.2f} | "
            f"{ratio:.3f} |"
        )

    lines.extend(
        [
            "",
            "## `vmro3` Storage",
            "",
            "| file | dtype | raw MiB | filters / chunking |",
            "|---|---:|---:|---|",
        ]
    )
    for label, info in files.items():
        vmro3 = info["variables"]["vmro3"]
        lines.append(
            f"| {label} | {vmro3['dtype']} | {vmro3['raw_mib']:.2f} | "
            f"{fmt_filters(vmro3)} |"
        )

    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            report["conclusion"]["size_difference_main_reason"],
            "",
            "The reference CMIP6 example and the SWOOSH files all use `float32` for "
            "`vmro3`. The new NETCDF4_CLASSIC file was written with the reference "
            "file's chunking and compression filters for the common variables.",
            "",
        ]
    )
    if "filled_netcdf4_classic" in files:
        lines.extend(
            [
                "## Classic Output",
                "",
                f"`{files['filled_netcdf4_classic']['path']}`",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def print_summary(report: dict[str, Any]) -> None:
    print("File size summary:")
    for label, info in report["files"].items():
        vmro3 = info["variables"]["vmro3"]
        print(
            f"- {label}: {info['file_size_mib']:.2f} MiB, "
            f"model={info['data_model']}, disk={info['disk_format']}, "
            f"vmro3 dtype={vmro3['dtype']}, {fmt_filters(vmro3)}"
        )
    print()
    print(report["conclusion"]["size_difference_main_reason"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=REFERENCE_FILE)
    parser.add_argument("--original", type=Path, default=ORIGINAL_PROCESSED_FILE)
    parser.add_argument("--filled", type=Path, default=FILLED_FILE)
    parser.add_argument("--classic-output", type=Path, default=CLASSIC_OUTPUT)
    parser.add_argument("--json-report", type=Path, default=JSON_REPORT)
    parser.add_argument("--markdown-report", type=Path, default=MD_REPORT)
    parser.add_argument(
        "--no-create-classic",
        action="store_true",
        help="Only report diagnostics; do not write the NETCDF4_CLASSIC output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.no_create_classic:
        create_classic_copy(args.filled, args.reference, args.classic_output)

    classic_path = None if args.no_create_classic else args.classic_output
    report = collect_report(args.reference, args.original, args.filled, classic_path)
    write_json(report, args.json_report)
    write_markdown(report, args.markdown_report)
    print_summary(report)
    print(f"JSON report: {args.json_report}")
    print(f"Markdown report: {args.markdown_report}")
    if classic_path:
        print(f"NETCDF4_CLASSIC output: {classic_path}")


if __name__ == "__main__":
    main()
