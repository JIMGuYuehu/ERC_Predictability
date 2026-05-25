#!/usr/bin/env python3
"""Plot additional SWOOSH filled files and verify fill-only differences."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import numpy as np
import xarray as xr

from check_swoosh_missing import as_jsonable
from plot_swoosh_fill_diagnostics import make_plot_from_dataarrays
from process_swoosh_years import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REPORT_DIR,
    DEFAULT_SHARE_DIR,
    REFERENCE_FILE,
    SWOOSH_FILE,
    TARGET_VAR,
    date_range_label,
    remap_swoosh_to_template_unfilled,
    target_filename,
    year_month_window,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PLOT_DIR = SCRIPT_DIR / "plots"


def plot_filename(target_year: int) -> str:
    return f"O3_NHpolar_{target_year}_original_vs_filled_target_ppmv.png"


def finite_fill_consistency(original: xr.DataArray, filled: xr.DataArray) -> dict[str, Any]:
    original_values = np.asarray(original.values, dtype=np.float64)
    filled_raw_values = np.asarray(filled.values)
    filled_values = np.asarray(filled_raw_values, dtype=np.float64)
    if original_values.shape != filled_values.shape:
        raise ValueError(f"Shape mismatch: {original_values.shape} vs {filled_values.shape}")

    original_finite = np.isfinite(original_values)
    filled_finite = np.isfinite(filled_values)
    if np.issubdtype(filled_raw_values.dtype, np.floating):
        comparable_original = original_values.astype(filled_raw_values.dtype).astype(np.float64)
    else:
        comparable_original = original_values
    if original_finite.any():
        max_abs_delta_on_original_finite = float(
            np.max(
                np.abs(
                    filled_values[original_finite]
                    - comparable_original[original_finite]
                )
            )
        )
    else:
        max_abs_delta_on_original_finite = None

    coord_checks: dict[str, Any] = {}
    for coord in ("time", "plev", "lat", "lon"):
        if coord not in original.coords or coord not in filled.coords:
            coord_checks[coord] = {"present_in_both": False}
            continue
        orig_coord = np.asarray(original[coord].values, dtype=np.float64)
        fill_coord = np.asarray(filled[coord].values, dtype=np.float64)
        same_shape = orig_coord.shape == fill_coord.shape
        coord_checks[coord] = {
            "present_in_both": True,
            "same_shape": same_shape,
            "exact_equal": bool(same_shape and np.array_equal(orig_coord, fill_coord)),
            "max_abs_diff": (
                float(np.max(np.abs(orig_coord - fill_coord))) if same_shape else None
            ),
        }

    return {
        "shape": [int(v) for v in original_values.shape],
        "dtype_filled": str(filled.dtype),
        "original_missing_or_nonfinite": int((~original_finite).sum()),
        "original_missing_fraction": float((~original_finite).sum() / original_values.size),
        "filled_missing_or_nonfinite": int((~filled_finite).sum()),
        "filled_missing_fraction": float((~filled_finite).sum() / filled_values.size),
        "max_abs_delta_on_original_finite": max_abs_delta_on_original_finite,
        "finite_delta_note": "original finite values are compared after casting to the filled file dtype",
        "finite_values_unchanged": bool(max_abs_delta_on_original_finite == 0.0),
        "coordinate_checks": coord_checks,
    }


def process_year(
    target_year: int,
    ds_sw: xr.Dataset,
    ds_ref: xr.Dataset,
    filled_dir: Path,
    plot_dir: Path,
    share_dir: Path | None,
    lat_min: float,
    lat_max: float,
    plev_min: float,
    plev_max: float,
) -> dict[str, Any]:
    filled_path = filled_dir / target_filename(target_year)
    if not filled_path.exists():
        raise FileNotFoundError(filled_path)

    original = remap_swoosh_to_template_unfilled(ds_sw, ds_ref, target_year)
    with xr.open_dataset(filled_path, decode_times=False, mask_and_scale=True) as ds_filled:
        filled = ds_filled[TARGET_VAR].load()

    consistency = finite_fill_consistency(original, filled)
    if not consistency["finite_values_unchanged"]:
        raise RuntimeError(
            f"{target_year}: filled file changed finite pre-fill values; "
            f"max abs delta={consistency['max_abs_delta_on_original_finite']}"
        )
    if consistency["filled_missing_or_nonfinite"] != 0:
        raise RuntimeError(f"{target_year}: filled file still contains missing/non-finite values")

    plot_path = plot_dir / plot_filename(target_year)
    make_plot_from_dataarrays(
        original_da=original,
        filled_da=filled,
        time_values=filled["time"].values,
        output_path=plot_path,
        lat_min=lat_min,
        lat_max=lat_max,
        plev_min=plev_min,
        plev_max=plev_max,
    )

    shared_plot = None
    if share_dir is not None:
        share_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(share_dir, 0o777)
        shared_plot = share_dir / plot_path.name
        shutil.copy2(plot_path, shared_plot)
        os.chmod(shared_plot, 0o777)

    return {
        "target_year": target_year,
        "window": date_range_label(year_month_window(target_year)),
        "filled_file": str(filled_path),
        "plot_file": str(plot_path),
        "shared_plot_file": str(shared_plot) if shared_plot is not None else None,
        "consistency": consistency,
    }


def write_reports(report: dict[str, Any], json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, default=as_jsonable) + "\n")

    lines = [
        "# SWOOSH Additional-Year Plot Consistency",
        "",
        f"- Created UTC: `{report['created_utc']}`",
        f"- SWOOSH source: `{report['swoosh_file']}`",
        f"- Reference template: `{report['reference_file']}`",
        "",
        "| target year | window | plot | original missing | filled missing | finite-value max delta |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for item in report["years"]:
        c = item["consistency"]
        lines.append(
            f"| {item['target_year']} | {item['window']} | `{item['plot_file']}` | "
            f"{c['original_missing_or_nonfinite']} | {c['filled_missing_or_nonfinite']} | "
            f"{c['max_abs_delta_on_original_finite']} |"
        )
    lines.extend(
        [
            "",
            "Finite-value max delta is computed only where the remapped pre-fill field "
            "already had finite values, after casting that pre-fill field to the "
            "filled file dtype. A value of `0.0` means the fill step changed only "
            "the original missing/non-finite grid cells.",
            "",
        ]
    )
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=int, nargs="+", default=[2011, 2018, 2023])
    parser.add_argument("--swoosh-file", type=Path, default=SWOOSH_FILE)
    parser.add_argument("--reference-file", type=Path, default=REFERENCE_FILE)
    parser.add_argument("--filled-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--share-dir", type=Path, default=DEFAULT_SHARE_DIR)
    parser.add_argument("--no-share", action="store_true")
    parser.add_argument("--lat-min", type=float, default=60.0)
    parser.add_argument("--lat-max", type=float, default=90.0)
    parser.add_argument("--plev-min", type=float, default=1.0)
    parser.add_argument("--plev-max", type=float, default=100.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    share_dir = None if args.no_share else args.share_dir

    with xr.open_dataset(args.swoosh_file, decode_times=False, mask_and_scale=True) as ds_sw:
        with xr.open_dataset(args.reference_file, decode_times=False, mask_and_scale=True) as ds_ref:
            results = [
                process_year(
                    target_year=year,
                    ds_sw=ds_sw,
                    ds_ref=ds_ref,
                    filled_dir=args.filled_dir,
                    plot_dir=args.plot_dir,
                    share_dir=share_dir,
                    lat_min=args.lat_min,
                    lat_max=args.lat_max,
                    plev_min=args.plev_min,
                    plev_max=args.plev_max,
                )
                for year in args.years
            ]

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "swoosh_file": str(args.swoosh_file),
        "reference_file": str(args.reference_file),
        "share_dir": str(share_dir) if share_dir is not None else None,
        "years": results,
    }
    label = "_".join(str(year) for year in args.years)
    json_path = args.report_dir / f"swoosh_plot_consistency_{label}.json"
    markdown_path = args.report_dir / f"swoosh_plot_consistency_{label}.md"
    write_reports(report, json_path, markdown_path)

    for item in results:
        c = item["consistency"]
        print(
            f"{item['target_year']}: plot={item['plot_file']}; "
            f"missing {c['original_missing_or_nonfinite']} -> "
            f"{c['filled_missing_or_nonfinite']}; "
            f"finite max delta={c['max_abs_delta_on_original_finite']}"
        )
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")


if __name__ == "__main__":
    main()
