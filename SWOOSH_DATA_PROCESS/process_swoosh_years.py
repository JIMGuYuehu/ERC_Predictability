#!/usr/bin/env python3
"""Create filled SWOOSH vmro3 files for additional target years.

This script reuses the same processing choices as the 2020 SWOOSH file:

* SWOOSH v02.72 ``combinedo3q`` is read in ppmv and converted to mole mole-1;
* negative ozone values are treated as missing before interpolation;
* horizontal interpolation is linear on the template lat/lon grid;
* vertical interpolation is linear in log-pressure, with nearest-edge pressure
  extrapolation outside the SWOOSH pressure range;
* remaining missing values are filled with the same conservative
  linear/nearest-edge method used for the urgent 2020 no-missing file;
* NetCDF compression/chunking follows the CMIP6 example file.

For target year YYYY, the output contains 14 monthly fields:
YYYY-1 December through YYYY+1 January.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import netCDF4 as nc
import numpy as np
import xarray as xr

from check_swoosh_missing import as_jsonable, variable_summary
from fill_swoosh_target import DEFAULT_DIM_ORDER, DEFAULT_FILL_VALUE, fill_dataarray
from why_size_dif import create_classic_copy


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"
DEFAULT_REPORT_DIR = SCRIPT_DIR / "reports"
DEFAULT_SHARE_DIR = Path("/mnt/soclim0/public_data/weiji/swoosh/SWOOSH_nan_fill_20260521")

SWOOSH_FILE = Path(
    "/mnt/soclim0/public_data/weiji/swoosh/"
    "swoosh-v02.72-198401-202601-lonlatpress-20deg-5deg-L31.nc"
)
REFERENCE_FILE = Path("/mnt/soclim0/andreas/vmro3_input4MIPs_ozone_CMIP6_UReading-CCMI_2020.nc")

SOURCE_VAR = "combinedo3q"
TARGET_VAR = "vmro3"
FILLVAL = DEFAULT_FILL_VALUE
SOURCE_UNITS_TO_TARGET = 1.0e-6
SOURCE_REFERENCE_YEAR = 2020


def year_month_window(target_year: int) -> list[tuple[int, int]]:
    return (
        [(target_year - 1, 12)]
        + [(target_year, month) for month in range(1, 13)]
        + [(target_year + 1, 1)]
    )


def date_range_label(months: list[tuple[int, int]]) -> str:
    start_year, start_month = months[0]
    end_year, end_month = months[-1]
    return f"{start_year:04d}{start_month:02d}-{end_year:04d}{end_month:02d}"


def target_filename(target_year: int, classic: bool = False) -> str:
    label = date_range_label(year_month_window(target_year))
    suffix = "_netcdf4_classic" if classic else ""
    return (
        "vmro3_SWOOSHv02.72_combinedo3q_neg2miss_remapped_to_"
        f"CMIP6_template_{label}_filled_no_missing{suffix}.nc"
    )


def open_dataset(path: Path) -> xr.Dataset:
    return xr.open_dataset(path, decode_times=False, mask_and_scale=True)


def build_swoosh_time_lookup(ds_sw: xr.Dataset) -> dict[tuple[int, int], int]:
    years = ds_sw["year"].values.astype(int)
    months = ds_sw["month"].values.astype(int)
    return {(int(y), int(m)): i for i, (y, m) in enumerate(zip(years, months))}


def shifted_template_time(ds_tp: xr.Dataset, target_year: int) -> tuple[xr.DataArray, xr.DataArray]:
    """Shift the 2020 template time axis to another target year.

    The reference file stores ``time`` as month-like values and ``time_bnds`` as
    the original template's raw bounds. To stay structurally consistent with the
    2020 file, both arrays are shifted by the target-year offset rather than
    rebuilt with a different convention.
    """

    year_offset = target_year - SOURCE_REFERENCE_YEAR
    time_shift = year_offset * 12.0
    time = xr.DataArray(
        ds_tp["time"].values + time_shift,
        dims=ds_tp["time"].dims,
        attrs=dict(ds_tp["time"].attrs),
        name="time",
    )

    if "time_bnds" not in ds_tp:
        return time, xr.DataArray()

    # The reference time_bnds values are day-like raw numbers even though the
    # time units say "months since". Shift them by actual calendar days so their
    # raw convention remains parallel to the 2020 file.
    bounds = ds_tp["time_bnds"].values
    shifted_bounds = np.empty_like(bounds, dtype=np.float64)
    units = "days since 1850-01-01 00:00:00"
    for index, value in np.ndenumerate(bounds):
        date = nc.num2date(float(value), units, calendar="standard")
        shifted_date = date.replace(year=date.year + year_offset)
        shifted_bounds[index] = nc.date2num(shifted_date, units, calendar="standard")

    time_bnds = xr.DataArray(
        shifted_bounds,
        dims=ds_tp["time_bnds"].dims,
        attrs=dict(ds_tp["time_bnds"].attrs),
        name="time_bnds",
    )
    return time, time_bnds


def add_cyclic_lon(da: xr.DataArray) -> xr.DataArray:
    lon = da["lon"].values.astype(float)
    if not np.all(np.diff(lon) > 0):
        raise ValueError("Longitude must be strictly increasing before cyclic padding.")

    left = da.isel(lon=-1).copy(deep=True).expand_dims(lon=[float(lon[-1] - 360.0)])
    right = da.isel(lon=0).copy(deep=True).expand_dims(lon=[float(lon[0] + 360.0)])
    return xr.concat([left, da, right], dim="lon").sortby("lon")


def add_lat_edges(da: xr.DataArray) -> xr.DataArray:
    lat = da["lat"].values.astype(float)
    if not np.all(np.diff(lat) > 0):
        raise ValueError("Latitude must be strictly increasing before edge padding.")

    pieces: list[xr.DataArray] = []
    if lat[0] > -90.0:
        pieces.append(da.isel(lat=0).copy(deep=True).expand_dims(lat=[-90.0]))
    pieces.append(da)
    if lat[-1] < 90.0:
        pieces.append(da.isel(lat=-1).copy(deep=True).expand_dims(lat=[90.0]))
    return xr.concat(pieces, dim="lat").sortby("lat")


def interp_profile_logp(
    profile: np.ndarray,
    src_p: np.ndarray,
    dst_p: np.ndarray,
) -> np.ndarray:
    prof = np.asarray(profile, dtype=np.float64)
    src_p = np.asarray(src_p, dtype=np.float64)
    dst_p = np.asarray(dst_p, dtype=np.float64)
    out = np.full(dst_p.shape, np.nan, dtype=np.float64)

    valid = np.isfinite(prof) & np.isfinite(src_p) & (src_p > 0)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return out

    p_valid = src_p[valid]
    v_valid = prof[valid]
    order = np.argsort(p_valid)
    p_valid = p_valid[order]
    v_valid = v_valid[order]

    if n_valid == 1:
        out[:] = v_valid[0]
        return out

    pmin = p_valid.min()
    pmax = p_valid.max()
    inside = (dst_p >= pmin) & (dst_p <= pmax)
    if inside.any():
        out[inside] = np.interp(np.log(dst_p[inside]), np.log(p_valid), v_valid)

    low_out = dst_p > pmax
    if low_out.any():
        out[low_out] = v_valid[-1]

    high_out = dst_p < pmin
    if high_out.any():
        out[high_out] = v_valid[0]

    return out


def remap_swoosh_to_template(
    ds_sw: xr.Dataset,
    ds_tp: xr.Dataset,
    target_year: int,
) -> tuple[xr.DataArray, list[dict[str, Any]]]:
    months = year_month_window(target_year)
    lookup = build_swoosh_time_lookup(ds_sw)
    missing = [month for month in months if month not in lookup]
    if missing:
        raise KeyError(f"SWOOSH source is missing requested months: {missing}")

    sw_indices = [lookup[month] for month in months]
    src_raw = ds_sw[SOURCE_VAR].isel(time=sw_indices).astype(np.float64) * SOURCE_UNITS_TO_TARGET
    src = src_raw.where(src_raw >= 0.0, np.nan)

    src = src.assign_coords(lon=(src["lon"].values % 360.0)).sortby("lon")
    src = src.rename(level="plev")
    src = add_cyclic_lon(src)
    src = add_lat_edges(src)

    src_horiz = src.interp(
        lat=ds_tp["lat"],
        lon=ds_tp["lon"],
        method="linear",
        kwargs={"fill_value": np.nan},
    ).transpose("time", "plev", "lat", "lon")

    src_plev = src_horiz["plev"].values.astype(np.float64)
    dst_plev = ds_tp["plev"].values.astype(np.float64)
    src_prof = src_horiz.transpose("time", "lat", "lon", "plev")

    vert = xr.apply_ufunc(
        interp_profile_logp,
        src_prof,
        xr.DataArray(src_plev, dims=["plev"]),
        xr.DataArray(dst_plev, dims=["plev_out"]),
        input_core_dims=[["plev"], ["plev"], ["plev_out"]],
        output_core_dims=[["plev_out"]],
        vectorize=True,
        dask="forbidden",
        output_dtypes=[np.float64],
        dask_gufunc_kwargs={"output_sizes": {"plev_out": len(dst_plev)}},
    )

    time_coord, _ = shifted_template_time(ds_tp, target_year)
    out_var = (
        vert.rename({"plev_out": "plev"})
        .assign_coords(
            time=time_coord.values,
            plev=ds_tp["plev"].values,
            lat=ds_tp["lat"].values,
            lon=ds_tp["lon"].values,
        )
        .transpose("time", "plev", "lat", "lon")
    )
    out_var.name = TARGET_VAR
    out_var.attrs = dict(ds_tp[TARGET_VAR].attrs)
    out_var.attrs["missing_value"] = np.float32(FILLVAL)
    out_var.attrs.pop("_FillValue", None)

    prefill_summary = variable_summary(out_var)
    filled_values, fill_steps = fill_dataarray(out_var, DEFAULT_DIM_ORDER)
    out_filled = out_var.copy(deep=False)
    out_filled.data = filled_values.astype(np.float32, copy=False)
    out_filled.attrs = dict(out_var.attrs)
    out_filled.attrs["missing_value"] = np.float32(FILLVAL)

    fill_steps = [
        {
            **step,
            "target_year": target_year,
            "window": date_range_label(months),
        }
        for step in fill_steps
    ]
    fill_steps.insert(
        0,
        {
            "target_year": target_year,
            "window": date_range_label(months),
            "stage": "after_remap_before_fill",
            "missing_before": prefill_summary["n_missing_or_nonfinite"],
            "missing_fraction_before": prefill_summary["missing_fraction"],
        },
    )
    return out_filled, fill_steps


def assemble_dataset(
    ds_tp: xr.Dataset,
    out_var: xr.DataArray,
    target_year: int,
) -> xr.Dataset:
    time_coord, time_bnds = shifted_template_time(ds_tp, target_year)
    months = year_month_window(target_year)
    label = date_range_label(months)

    out_ds = xr.Dataset(
        coords={
            "time": time_coord,
            "plev": ds_tp["plev"],
            "lat": ds_tp["lat"],
            "lon": ds_tp["lon"],
        }
    )
    if "time_bnds" in ds_tp:
        out_ds["time_bnds"] = time_bnds
    for bname in ["plev_bnds", "lat_bnds", "lon_bnds"]:
        if bname in ds_tp:
            out_ds[bname] = ds_tp[bname]

    out_ds[TARGET_VAR] = out_var.astype(np.float32)
    out_ds[TARGET_VAR].attrs = dict(ds_tp[TARGET_VAR].attrs)
    out_ds[TARGET_VAR].attrs["missing_value"] = np.float32(FILLVAL)
    out_ds[TARGET_VAR].attrs.pop("_FillValue", None)

    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out_ds.attrs = {
        "Conventions": ds_tp.attrs.get("Conventions", "CF-1.6"),
        "title": "SWOOSH ozone remapped to CMIP6-style vmro3 template grid with no missing values",
        "source": "SWOOSH v02.72 combinedo3q",
        "comment": (
            "Monthly mean ozone field constructed from SWOOSH v02.72 variable "
            "'combinedo3q' and remapped onto the grid and pressure levels of the "
            "reference CMIP6-style vmro3 template file. Units converted from ppmv "
            "to mole mole-1. Source negative ozone values were treated as missing "
            "before interpolation. Remaining target-grid missing values were filled "
            "by coordinate-wise linear interpolation and nearest-edge extrapolation."
        ),
        "institution": "Derived from NOAA SWOOSH data and remapped by Weiji Hu",
        "created_by": "Weiji Hu",
        "creation_date": created,
        "variable_id": TARGET_VAR,
        "dataset_category": "ozone",
        "frequency": "mon",
        "grid": f"Template lat-lon-pressure grid based on {REFERENCE_FILE.name}",
        "target_year": str(target_year),
        "target_month_window": label,
        "missing_value_treatment": (
            f"{TARGET_VAR} missing/non-finite values were filled with coordinate-wise "
            "linear interpolation for interior gaps and nearest-edge extrapolation "
            f"at domain edges. Dimension order: {', '.join(DEFAULT_DIM_ORDER)}."
        ),
        "history": (
            f"{created} created from SWOOSH v02.72 {SOURCE_VAR}; target window {label}; "
            "unit conversion ppmv -> mole mole-1; source negative values converted "
            "to missing before interpolation; horizontal interpolation to template "
            "lat/lon; log-pressure interpolation to template plev; remaining "
            "missing values filled by nearest-edge/interior-linear method."
        ),
    }
    return out_ds


def output_encoding() -> dict[str, dict[str, Any]]:
    return {
        TARGET_VAR: {
            "zlib": True,
            "complevel": 9,
            "shuffle": False,
            "dtype": "float32",
            "_FillValue": np.float32(FILLVAL),
            "chunksizes": (7, 33, 48, 72),
        },
        "time": {"dtype": "float64", "_FillValue": np.nan, "zlib": False},
        "plev": {"dtype": "float32", "_FillValue": np.nan, "zlib": False},
        "lat": {
            "dtype": "float32",
            "_FillValue": np.nan,
            "zlib": True,
            "complevel": 9,
            "shuffle": False,
            "chunksizes": (96,),
        },
        "lon": {
            "dtype": "float32",
            "_FillValue": np.nan,
            "zlib": True,
            "complevel": 9,
            "shuffle": False,
            "chunksizes": (144,),
        },
        "time_bnds": {
            "dtype": "float64",
            "_FillValue": np.nan,
            "zlib": True,
            "complevel": 9,
            "shuffle": False,
            "chunksizes": (14, 2),
        },
        "plev_bnds": {"dtype": "float32", "_FillValue": np.nan, "zlib": False},
        "lat_bnds": {
            "dtype": "float64",
            "_FillValue": np.nan,
            "zlib": True,
            "complevel": 9,
            "shuffle": False,
            "chunksizes": (96, 2),
        },
        "lon_bnds": {
            "dtype": "float64",
            "_FillValue": np.nan,
            "zlib": True,
            "complevel": 9,
            "shuffle": False,
            "chunksizes": (144, 2),
        },
    }


def write_dataset(ds: xr.Dataset, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    ds.to_netcdf(tmp_path, format="NETCDF4", engine="netcdf4", encoding=output_encoding())
    tmp_path.replace(output_path)


def check_written_file(path: Path) -> dict[str, Any]:
    with open_dataset(path) as ds:
        checked = {
            name: variable_summary(ds[name])
            for name in ["time", "time_bnds", "plev", "plev_bnds", "lat", "lat_bnds", "lon", "lon_bnds", TARGET_VAR]
            if name in ds.variables
        }
    total_missing = int(sum(item["n_missing_or_nonfinite"] for item in checked.values()))
    if total_missing:
        raise RuntimeError(f"{path} still has {total_missing} missing/non-finite values")
    return {
        "path": str(path),
        "file_size_bytes": path.stat().st_size,
        "checked_variables": checked,
        "total_missing_or_nonfinite": total_missing,
    }


def create_year_outputs(
    target_year: int,
    ds_sw: xr.Dataset,
    ds_tp: xr.Dataset,
    output_dir: Path,
    share_dir: Path | None,
    create_classic: bool,
) -> dict[str, Any]:
    print(f"\n=== Processing target year {target_year} ===")
    out_var, fill_steps = remap_swoosh_to_template(ds_sw, ds_tp, target_year)
    out_ds = assemble_dataset(ds_tp, out_var, target_year)

    local_output = output_dir / target_filename(target_year)
    write_dataset(out_ds, local_output)
    local_check = check_written_file(local_output)
    print(f"Wrote local: {local_output}")
    print(f"Readback missing/non-finite: {local_check['total_missing_or_nonfinite']}")

    share_output = None
    share_check = None
    if share_dir is not None:
        share_output = share_dir / target_filename(target_year)
        write_dataset(out_ds, share_output)
        os.chmod(share_output, 0o777)
        share_check = check_written_file(share_output)
        print(f"Wrote shared: {share_output}")

    classic_outputs: dict[str, Any] = {}
    if create_classic:
        local_classic = output_dir / target_filename(target_year, classic=True)
        create_classic_copy(local_output, REFERENCE_FILE, local_classic)
        local_classic_check = check_written_file(local_classic)
        classic_outputs["local"] = local_classic_check
        print(f"Wrote local classic: {local_classic}")

        if share_dir is not None:
            share_classic = share_dir / target_filename(target_year, classic=True)
            create_classic_copy(share_output or local_output, REFERENCE_FILE, share_classic)
            os.chmod(share_classic, 0o777)
            share_classic_check = check_written_file(share_classic)
            classic_outputs["shared"] = share_classic_check
            print(f"Wrote shared classic: {share_classic}")

    return {
        "target_year": target_year,
        "target_months": [f"{year:04d}-{month:02d}" for year, month in year_month_window(target_year)],
        "window": date_range_label(year_month_window(target_year)),
        "fill_steps": fill_steps,
        "local_output": local_check,
        "shared_output": share_check,
        "classic_outputs": classic_outputs,
    }


def write_report(report: dict[str, Any], json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, default=as_jsonable) + "\n")

    lines = [
        "# SWOOSH Additional Year Outputs",
        "",
        f"- Created UTC: `{report['created_utc']}`",
        f"- SWOOSH source: `{report['swoosh_file']}`",
        f"- Reference template: `{report['reference_file']}`",
        f"- Shared directory: `{report.get('share_dir') or 'not used'}`",
        "",
        "| target year | window | local file | shared file | missing/non-finite |",
        "|---:|---|---|---|---:|",
    ]
    for item in report["years"]:
        local_file = item["local_output"]["path"]
        shared_file = item["shared_output"]["path"] if item["shared_output"] else ""
        missing = item["local_output"]["total_missing_or_nonfinite"]
        lines.append(
            f"| {item['target_year']} | {item['window']} | `{local_file}` | "
            f"`{shared_file}` | {missing} |"
        )

    lines.extend(["", "## Fill Summary", ""])
    for item in report["years"]:
        lines.append(f"### {item['target_year']} ({item['window']})")
        for step in item["fill_steps"]:
            if step.get("stage") == "after_remap_before_fill":
                lines.append(
                    f"- Before fill: {step['missing_before']} missing/non-finite "
                    f"({step['missing_fraction_before']:.6%})"
                )
            else:
                lines.append(
                    f"- `{step['dimension']}`: {step['missing_before']} -> "
                    f"{step['missing_after']} missing; filled {step['filled']}"
                )
        lines.append("")

    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=int, nargs="+", default=[2011, 2018, 2023])
    parser.add_argument("--swoosh-file", type=Path, default=SWOOSH_FILE)
    parser.add_argument("--reference-file", type=Path, default=REFERENCE_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--share-dir", type=Path, default=DEFAULT_SHARE_DIR)
    parser.add_argument("--no-share", action="store_true")
    parser.add_argument("--no-classic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global SWOOSH_FILE, REFERENCE_FILE
    SWOOSH_FILE = args.swoosh_file
    REFERENCE_FILE = args.reference_file

    output_dir = args.output_dir
    report_dir = args.report_dir
    share_dir = None if args.no_share else args.share_dir
    if share_dir is not None:
        share_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(share_dir, 0o777)

    with open_dataset(SWOOSH_FILE) as ds_sw, open_dataset(REFERENCE_FILE) as ds_tp:
        results = [
            create_year_outputs(
                target_year=year,
                ds_sw=ds_sw,
                ds_tp=ds_tp,
                output_dir=output_dir,
                share_dir=share_dir,
                create_classic=not args.no_classic,
            )
            for year in args.years
        ]

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "swoosh_file": str(SWOOSH_FILE),
        "reference_file": str(REFERENCE_FILE),
        "share_dir": str(share_dir) if share_dir is not None else None,
        "years": results,
    }
    label = "_".join(str(year) for year in args.years)
    json_path = report_dir / f"swoosh_additional_years_{label}.json"
    markdown_path = report_dir / f"swoosh_additional_years_{label}.md"
    write_report(report, json_path, markdown_path)
    print(f"\nJSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")


if __name__ == "__main__":
    main()
