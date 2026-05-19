#!/usr/bin/env python3
"""Save Longrun AO/NAM first EOF mode patterns for traceability.

This preserves the mode-1 data behind the existing AO/NAM index products. The
default cases are the three longrun experiments requested for later projection
work. It does not overwrite existing index files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[2]
HINDCAST_SCRIPTS = REPO_ROOT / "Hindcast_experiment" / "date_treatment" / "scripts"
if str(HINDCAST_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(HINDCAST_SCRIPTS))

from compute_hindcast_ao_nam_b2000wcn_projection import build_reference, load_b2000wcn_z3_zm
from hindcast_common import write_netcdf_atomic


DEFAULT_CASES: Dict[str, Path] = {
    "B2000WCN001002": Path("/mnt/soclim0/public_data/weiji/B2000WCN001002_timefixed"),
    "B2000WCN007009010011_Clim3D": Path("/mnt/soclim0/public_data/weiji/B2000WCN007009010011_Clim3D_timefixed"),
    "B2000WCN_NOCOUPL001002": Path("/mnt/soclim0/public_data/weiji/B2000WCN_NOCOUPL001002_timefixed"),
}


def save_modes_for_case(case_name: str, root: Path, max_workers: int, max_years: Optional[int], overwrite: bool) -> Path:
    z3_zm = load_b2000wcn_z3_zm(root, max_workers=max_workers, max_years=max_years)
    ref = build_reference(z3_zm)
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
            "title": f"{case_name} AO/NAM first EOF mode patterns",
            "case_name": case_name,
            "source_root": str(root),
            "purpose": "Preserve mode-1 patterns used by AO/NAM calculations for later projection and traceability.",
        }
    )
    out_file = root / "NAM" / f"{case_name}_AO_NAM_mode1_patterns.nc"
    write_netcdf_atomic(ds, out_file, overwrite=overwrite)
    return out_file


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="*", choices=sorted(DEFAULT_CASES), default=list(DEFAULT_CASES))
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-years", type=int, default=None, help="Debug option: use only first N years.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    for case_name in args.cases:
        out = save_modes_for_case(case_name, DEFAULT_CASES[case_name], args.max_workers, args.max_years, args.overwrite)
        print(f"[MODE1] {case_name}: {out}")


if __name__ == "__main__":
    main()
