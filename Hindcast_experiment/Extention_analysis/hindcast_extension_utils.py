from __future__ import annotations

import json
import math
import re
import warnings
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.ticker import FixedLocator
from matplotlib.ticker import ScalarFormatter
from scipy.stats import pearsonr


DATA_ROOT = Path("/mnt/soclim0/public_data/weiji")
HINDCAST_ROOT = DATA_ROOT / "Hindcast"
BWCN_ROOT = DATA_ROOT / "BWCN"
B2000_ROOT = DATA_ROOT / "B2000WCN001002_timefixed"

WORK_ROOT = Path("/home/weiji/restart_exam/code_cleaned/Hindcast_experiment/Extention_analysis")
OUT_ROOT = WORK_ROOT / "outputs"
FIG_DIR = OUT_ROOT / "figures"
TAB_DIR = OUT_ROOT / "tables"
CACHE_DIR = OUT_ROOT / "cache"
LOG_DIR = OUT_ROOT / "logs"

for d in [FIG_DIR, TAB_DIR, CACHE_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


SELECTED_YEARS = ["0003", "0008", "0013", "0014", "0019"]
WAVES = ["all_waves", "wave1", "wave2", "wave_rest"]
WAVE_LABELS = {
    "all_waves": "All waves",
    "wave1": "Wave 1",
    "wave2": "Wave 2",
    "wave_rest": "Rest / synoptic waves",
    "wave1_plus_wave2": "Wave 1 + Wave 2",
}

O3_VAR = "O3_partial_60_90N_30_70hPa"
LAT_EP = (40.0, 80.0)
LAT_POLAR = (60.0, 90.0)
LAT_Z300 = (20.0, 90.0)
PLEV_EP_HPA = 100.0
PLEV_Z300_HPA = 300.0
U60_THRESH = 7.0
FWD_PERSIST_DAYS = 10
MONTH_START_DOY = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334], dtype=int)
MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH_WINDOWS = {
    "Jan": ((1, 1), (1, 31)),
    "Feb": ((2, 1), (2, 28)),
    "Mar": ((3, 1), (3, 31)),
    "Apr": ((4, 1), (4, 30)),
    "May": ((5, 1), (5, 30)),
}
MONTH_TICK_DOYS = [1, 32, 60, 91, 121, 152]
MONTH_TICK_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]

plt.rcParams.update({
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.35,
})


def log_message(message: str, log_name: str = "missing_products.log") -> None:
    """Append a message to an analysis log file under LOG_DIR."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with (LOG_DIR / log_name).open("a") as f:
        f.write(str(message).rstrip() + "\n")


def member_short_id(member) -> str:
    """Return a compact three-digit member id from filenames or member coordinates.

    Parameters
    ----------
    member:
        Filename-like or coordinate-like member identifier.

    Returns
    -------
    str
        Three-digit member id when detectable; otherwise the original text.
    """
    text = str(member)
    # Hindcast filenames often include an ensemble branch before the
    # year-month token, e.g. ``...f19_g16.002.0013-02.001...``.  The actual
    # perturbed member is the number after ``YYYY-MM``, not the branch id.
    for pattern in [
        r"\.\d{4}-\d{2}\.(\d{3})(?:\.|_)",
        r"\.(\d{3})\.cam\.h3",
        r"-(\d{3})\.cam",
        r"_(\d{3})$",
        r"\.(\d{3})\.",
    ]:
        m = re.search(pattern, text)
        if m:
            return m.group(1)
    return text


def parse_case_name(case: str) -> dict:
    """Parse hindcast case names into year, init month, configuration, and perturbation."""
    m = re.match(r"^(?P<year>\d{4})-(?P<init_month>\d{2})(?P<suffix>.*)$", str(case))
    if not m:
        return {
            "case": case,
            "year": None,
            "init_month": None,
            "config": "UNKNOWN",
            "perturbation": "unknown",
            "special_branch": False,
        }
    suffix = m.group("suffix") or ""
    config = "NOCOUPL" if "NOCOUPL" in suffix else "INT"
    if "v2" in suffix:
        perturbation = "v2_large_temperature"
    elif "v3" in suffix:
        perturbation = "v3_humidity"
    else:
        perturbation = "small_temperature"
    return {
        "case": case,
        "year": m.group("year"),
        "init_month": m.group("init_month"),
        "config": config,
        "perturbation": perturbation,
        "special_branch": m.group("year") == "0003",
    }


def _product_path(case: str, product: str, wave: Optional[str] = None) -> Path:
    root = HINDCAST_ROOT / case
    if product == "partial_O3":
        return root / "partial_O3" / f"{case}_partial_O3_all_ranges_members.nc"
    if product == "EPflux" and wave is not None:
        return root / "EPflux_daily_ubar" / wave / f"EPFLUX_{wave}_{case}_members_time_plev_lat.nc"
    if product == "FWD":
        return root / "final_warming_date" / f"{case}_FWD_plev_member.nc"
    if product == "AO_NAM":
        return root / "NAM_B2000WCN_projection" / f"{case}_AO_NAM_B2000WCN_projection_members.nc"
    return root / product


def _count_members(case: str) -> int:
    path = _product_path(case, "partial_O3")
    if path.exists():
        try:
            with xr.open_dataset(path, decode_times=False) as ds:
                if "member" in ds.sizes:
                    return int(ds.sizes["member"])
        except Exception as exc:
            log_message(f"{case}: failed member count from partial_O3: {exc}")
    return len(list((HINDCAST_ROOT / case / "U").glob("*.U.nc")))


def discover_hindcast_cases(root: Path = HINDCAST_ROOT) -> pd.DataFrame:
    """Scan HINDCAST_ROOT and summarize available cases and cleaned products.

    Parameters
    ----------
    root:
        Directory containing hindcast case subdirectories.

    Returns
    -------
    pandas.DataFrame
        One row per case with parsed metadata, member count, product flags, and paths.
    """
    rows = []
    for p in sorted(root.glob("*")):
        if not p.is_dir() or not re.match(r"^\d{4}-\d{2}", p.name):
            continue
        meta = parse_case_name(p.name)
        if meta["year"] not in SELECTED_YEARS:
            continue
        row = dict(meta)
        row.update({
            "path": str(p),
            "n_members": _count_members(p.name),
            "has_partial_O3": _product_path(p.name, "partial_O3").exists(),
            "has_U": (p / "U").exists(),
            "has_T": (p / "T").exists(),
            "has_Z3": (p / "Z3").exists(),
            "has_FWD": _product_path(p.name, "FWD").exists(),
            "has_AO_NAM": _product_path(p.name, "AO_NAM").exists(),
        })
        for wave in WAVES:
            row[f"has_EP_{wave}"] = _product_path(p.name, "EPflux", wave).exists()
        row["can_source_diagnose"] = row["has_partial_O3"] and row["has_Z3"] and all(row[f"has_EP_{w}"] for w in WAVES)
        row["can_feedback_pair"] = row["config"] == "INT" and row["perturbation"] == "small_temperature"
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["year", "init_month", "config", "perturbation", "case"]).reset_index(drop=True)
    return df


def _append_figure_manifest(
    name: str,
    notebook: str,
    scientific_question: str,
    variables_windows: str,
    interpretation: str,
    caveat: str,
    csv_table: str,
    png: Path,
    pdf: Path,
) -> None:
    manifest = TAB_DIR / "figure_manifest.csv"
    row = pd.DataFrame([{
        "figure": png.name,
        "png": str(png),
        "pdf": str(pdf),
        "notebook_source": notebook,
        "scientific_question": scientific_question,
        "variables_and_windows": variables_windows,
        "short_interpretation": interpretation,
        "caveat": caveat,
        "supporting_csv_table": csv_table,
    }])
    if manifest.exists():
        old = pd.read_csv(manifest)
        old = old.loc[old["png"] != str(png)]
        row = pd.concat([old, row], ignore_index=True)
    row.to_csv(manifest, index=False)


def savefig(
    fig,
    name: str,
    fig_dir: Path = FIG_DIR,
    notebook: str = "",
    scientific_question: str = "",
    variables_windows: str = "",
    interpretation: str = "",
    caveat: str = "",
    csv_table: str | Path = "",
):
    """Save a matplotlib figure as both PNG and PDF and record the manifest row.

    Parameters
    ----------
    fig:
        Matplotlib figure object.
    name:
        Basename without extension.
    fig_dir:
        Output directory.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        PNG and PDF paths.
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    png = fig_dir / f"{name}.png"
    pdf = fig_dir / f"{name}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    if csv_table:
        _append_figure_manifest(
            name,
            notebook,
            scientific_question,
            variables_windows,
            interpretation,
            caveat,
            str(csv_table),
            png,
            pdf,
        )
    print(f"Saved: {png}")
    return png, pdf


def figure_dir(*parts: str) -> Path:
    """Return and create a grouped figure directory below outputs/figures."""
    path = FIG_DIR.joinpath(*[str(p) for p in parts if str(p)])
    path.mkdir(parents=True, exist_ok=True)
    return path


def table_dir(*parts: str) -> Path:
    """Return and create a grouped table directory below outputs/tables."""
    path = TAB_DIR.joinpath(*[str(p) for p in parts if str(p)])
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_dir(*parts: str) -> Path:
    """Return and create a grouped cache directory below outputs/cache."""
    path = CACHE_DIR.joinpath(*[str(p) for p in parts if str(p)])
    path.mkdir(parents=True, exist_ok=True)
    return path


def finite_corr(x, y) -> dict:
    """Compute Pearson correlation for finite paired values.

    Parameters
    ----------
    x, y:
        Array-like vectors with matching length.

    Returns
    -------
    dict
        Keys are R, p, and n.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < 3:
        return {"R": np.nan, "p": np.nan, "n": n}
    r, p = pearsonr(x[mask], y[mask])
    return {"R": float(r), "p": float(p), "n": n}


def select_latband(da: xr.DataArray, lat_range: Tuple[float, float], lat_name: str = "lat") -> xr.DataArray:
    lat = da[lat_name]
    lo, hi = lat_range
    if float(lat.values[0]) > float(lat.values[-1]):
        return da.sel({lat_name: slice(hi, lo)})
    return da.sel({lat_name: slice(lo, hi)})


def coslat_weighted_mean(da: xr.DataArray, lat_range: Tuple[float, float]):
    """Cosine-latitude weighted mean over a latitude band.

    Parameters
    ----------
    da:
        DataArray with a lat coordinate.
    lat_range:
        Inclusive latitude range in degrees north.

    Returns
    -------
    xarray.DataArray
        Input averaged over latitude.
    """
    sub = select_latband(da, lat_range)
    weights = np.cos(np.deg2rad(sub["lat"])).clip(0, 1)
    return sub.weighted(weights.fillna(0)).mean("lat", skipna=True)


def date_parts(date_values):
    arr = np.asarray(date_values, dtype=np.int64)
    year = arr // 10000
    mmdd = arr % 10000
    month = mmdd // 100
    day = mmdd % 100
    return year, month, day


def _valid_calendar_or_mmdd(values) -> bool:
    raw = np.asarray(values, dtype=np.int64)
    if raw.size == 0:
        return False
    _, month, day = date_parts(raw)
    calendar = raw.max() >= 10000
    mmdd_only = raw.max() < 10000 and raw.min() >= 101
    return bool(
        (calendar or mmdd_only)
        and np.all((month >= 1) & (month <= 12) & (day >= 1) & (day <= 31))
    )


def mmdd_to_doy(month: int, day: int) -> int:
    return int(MONTH_START_DOY[int(month) - 1] + int(day))


def doy_to_mmdd(doy: int) -> tuple[int, int]:
    doy = int(doy)
    month = int(np.searchsorted(MONTH_START_DOY + 1, doy, side="right"))
    month = min(max(month, 1), 12)
    return month, int(doy - MONTH_START_DOY[month - 1])


def doy_label(doy: int) -> str:
    m, d = doy_to_mmdd(int(doy))
    return f"{MONTH_ABBR[m - 1]}{d:02d}"


def date_to_doy(date_values) -> np.ndarray:
    _, month, day = date_parts(date_values)
    month = np.asarray(month, dtype=int)
    day = np.asarray(day, dtype=int)
    return MONTH_START_DOY[month - 1] + day


def init_doy_for_case(case: str) -> int:
    """Return 1-based day-of-year for the first day of the case init month."""
    meta = parse_case_name(case)
    month = int(meta.get("init_month") or 1)
    return mmdd_to_doy(month, 1)


def case_time_doy(case: str, time_values) -> np.ndarray:
    """Convert real dates or 0-based lead days to a case-aware day-of-year axis."""
    if isinstance(time_values, xr.DataArray):
        values = time_values.values
    else:
        values = np.asarray(time_values)
    if np.issubdtype(np.asarray(values).dtype, np.datetime64):
        dates = pd.to_datetime(values)
        values = dates.year * 10000 + dates.month * 100 + dates.day
    raw = np.asarray(values, dtype=np.int64)
    if _valid_calendar_or_mmdd(raw):
        return date_to_doy(raw)
    return init_doy_for_case(case) + np.arange(len(raw), dtype=int)


def date_mask_for_case_window(case: str, time, start, end) -> np.ndarray:
    """Build a case-aware mask from MM-DD calendar bounds or lead-day bounds.

    Calendar bounds still work for derived files whose time coordinate is only
    0-based lead day; lead day 0 is anchored to the case initialization month.
    """
    if isinstance(time, xr.DataArray):
        values = time.values
    else:
        values = np.asarray(time)
    if isinstance(start, (tuple, list)) and isinstance(end, (tuple, list)):
        doy = case_time_doy(case, values)
        return (doy >= mmdd_to_doy(*start)) & (doy <= mmdd_to_doy(*end))
    lead = np.arange(len(values))
    return (lead >= int(start)) & (lead <= int(end))


def set_month_axis(ax, start_doy: int = 1, end_doy: int = 151, label: str = "Month"):
    """Style a day-of-year x-axis with month labels."""
    ticks = [d for d in MONTH_TICK_DOYS if start_doy <= d <= end_doy + 15]
    labels = [MONTH_TICK_LABELS[MONTH_TICK_DOYS.index(d)] for d in ticks]
    ax.set_xlim(start_doy, end_doy)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.set_xticklabels(labels)
    ax.set_xlabel(label)


def case_month_window_available(case: str, month_label: str) -> bool:
    """Return True if the case starts no later than the requested month."""
    meta = parse_case_name(case)
    init_month = int(meta.get("init_month") or 1)
    month_index = MONTH_ABBR.index(month_label) + 1
    return init_month <= month_index


def date_mask_from_mmdd_or_leadday(time, start, end):
    """Build a time mask from either calendar MM-DD tuples or lead-day indices.

    Parameters
    ----------
    time:
        Date integer array (YYYYMMDD) or xarray coordinate with date values.
    start, end:
        Either MM-DD tuples such as (1, 20), or integer lead-day bounds.

    Returns
    -------
    numpy.ndarray
        Boolean mask over the time axis.
    """
    if isinstance(time, xr.DataArray):
        values = time.values
    else:
        values = np.asarray(time)
    if np.issubdtype(np.asarray(values).dtype, np.datetime64):
        # Fallback for decoded cftime/datetime arrays.
        dates = pd.to_datetime(values)
        values = dates.year * 10000 + dates.month * 100 + dates.day
    if isinstance(start, (tuple, list)) and isinstance(end, (tuple, list)):
        raw = np.asarray(values, dtype=np.int64)
        year, month, day = date_parts(raw)
        valid_calendar = bool(
            raw.size
            and raw.max() >= 10000
            and np.all((month >= 1) & (month <= 12) & (day >= 1) & (day <= 31))
        )
        valid_mmdd = bool(
            raw.size
            and raw.max() < 10000
            and raw.min() >= 101
            and np.all((month >= 1) & (month <= 12) & (day >= 1) & (day <= 31))
        )
        if not (valid_calendar or valid_mmdd):
            # Some derived EP-flux products only retain a 0-based lead-time
            # coordinate.  For calendar windows in those products, treat lead
            # day 0 as Jan 1.  Non-January initialized cases use explicit lead
            # windows elsewhere in this workflow.
            lead = np.arange(len(raw))
            return (lead >= mmdd_to_doy(*start) - 1) & (lead <= mmdd_to_doy(*end) - 1)
        _, month, day = date_parts(values)
        key = month * 100 + day
        return (key >= start[0] * 100 + start[1]) & (key <= end[0] * 100 + end[1])
    lead = np.arange(len(values))
    return (lead >= int(start)) & (lead <= int(end))


def window_to_label(window) -> str:
    if isinstance(window[0], (tuple, list)):
        return f"{MONTH_ABBR[window[0][0]-1]}{window[0][1]:02d}-{MONTH_ABBR[window[1][0]-1]}{window[1][1]:02d}"
    return f"L{int(window[0]):03d}-L{int(window[1]):03d}"


def init_date_for_case(case: str) -> tuple[int, int]:
    meta = parse_case_name(case)
    month = int(meta.get("init_month") or 1)
    return month, 1


def target_window_for_case(case: str) -> tuple[tuple[int, int], tuple[int, int]]:
    return init_date_for_case(case), (5, 30)


def source_windows_for_case(case: str) -> dict:
    """Return primary and alternate source windows for EP/Z300 source tests."""
    if case == "0008-01":
        return {"primary": ((1, 20), (2, 10)), "lead0_30": (0, 30)}
    return {"primary": (10, 30), "lead0_30": (0, 30)}


def _assign_member_short(da: xr.DataArray) -> xr.DataArray:
    if "member" not in da.dims:
        return da
    shorts = [member_short_id(v) for v in da["member"].values]
    return da.assign_coords(member_short=("member", shorts))


def _has_unique_member_ids(da: xr.DataArray) -> bool:
    if "member" not in da.dims:
        return True
    ids = [member_short_id(v) for v in da["member"].values]
    return len(ids) == len(set(ids))


def _one_dim_date(ds_or_da) -> np.ndarray:
    if "date" in ds_or_da.coords:
        date = ds_or_da["date"]
    elif "date" in ds_or_da:
        date = ds_or_da["date"]
    else:
        return np.arange(ds_or_da.sizes.get("lead_time", ds_or_da.sizes.get("time", 0)))
    if "member" in date.dims:
        date = date.isel(member=0)
    return np.asarray(date.values, dtype=np.int64)


def load_hindcast_o3(case: str) -> tuple[Optional[xr.DataArray], Optional[np.ndarray]]:
    """Load cleaned hindcast partial O3 column for one case.

    Dimensions
    ----------
    member, lead_time

    Units
    -----
    Dobson Units for 60-90N, 30-70 hPa partial column.

    Returns
    -------
    tuple[xarray.DataArray | None, numpy.ndarray | None]
        O3 DataArray and integer date array. Missing products are logged.
    """
    path = _product_path(case, "partial_O3")
    if not path.exists():
        log_message(f"{case}: missing partial_O3 {path}")
        return None, None
    try:
        with xr.open_dataset(path, decode_times=False) as ds:
            if O3_VAR not in ds:
                log_message(f"{case}: missing variable {O3_VAR} in {path}")
                return None, None
            da = _assign_member_short(ds[O3_VAR]).load()
            date = _one_dim_date(ds)
        da = da.assign_coords(date=("lead_time", date[: da.sizes["lead_time"]]))
        return da, date[: da.sizes["lead_time"]]
    except Exception as exc:
        log_message(f"{case}: failed load_hindcast_o3: {exc}")
        return None, None


def load_bwcn_reference_o3(year: int | str) -> tuple[Optional[xr.DataArray], Optional[np.ndarray]]:
    """Load BWCN reference O3 for one model year.

    Parameters
    ----------
    year:
        Model year, e.g. 8 or "0008".

    Returns
    -------
    tuple[xarray.DataArray | None, numpy.ndarray | None]
        Reference O3 time series and date vector for the requested year.
    """
    path = BWCN_ROOT / "partial_O3" / "BWCN_partial_O3_all_ranges.nc"
    if not path.exists():
        log_message(f"BWCN: missing reference O3 {path}")
        return None, None
    year_int = int(year)
    try:
        with xr.open_dataset(path, decode_times=False) as ds:
            date = np.asarray(ds["date"].values, dtype=np.int64)
            mask = date_parts(date)[0] == year_int
            if mask.sum() == 0:
                log_message(f"BWCN: no reference O3 dates for year {year_int:04d}")
                return None, None
            da = ds[O3_VAR].isel(time=mask).load().rename({"time": "lead_time"})
            date = date[mask]
        da = da.assign_coords(lead_time=np.arange(len(date)), date=("lead_time", date))
        return da, date
    except Exception as exc:
        log_message(f"BWCN year {year_int:04d}: failed reference O3 load: {exc}")
        return None, None


def compute_o3_rmse(hind_o3, ref_o3, start, end) -> pd.DataFrame:
    """Compute member O3 RMSE relative to a reference curve over a time window.

    Parameters
    ----------
    hind_o3:
        DataArray with member and lead_time dimensions.
    ref_o3:
        Reference DataArray with lead_time dimension.
    start, end:
        MM-DD tuples or lead-day integer bounds.

    Returns
    -------
    pandas.DataFrame
        Columns member, O3_RMSE, and n_days.
    """
    if hind_o3 is None or ref_o3 is None:
        return pd.DataFrame(columns=["member", "O3_RMSE", "n_days"])
    hind_date = np.asarray(hind_o3["date"].values if "date" in hind_o3.coords else np.arange(hind_o3.sizes["lead_time"]))
    ref_date = np.asarray(ref_o3["date"].values if "date" in ref_o3.coords else np.arange(ref_o3.sizes["lead_time"]))
    mh = date_mask_from_mmdd_or_leadday(hind_date, start, end)
    mr = date_mask_from_mmdd_or_leadday(ref_date, start, end)
    if isinstance(start, (tuple, list)) and isinstance(end, (tuple, list)) and _valid_calendar_or_mmdd(hind_date) and _valid_calendar_or_mmdd(ref_date):
        hind_doy = date_to_doy(hind_date[mh])
        ref_doy = date_to_doy(ref_date[mr])
        common_doy = np.intersect1d(hind_doy, ref_doy)
        if len(common_doy):
            hind_index = np.where(mh)[0][np.isin(hind_doy, common_doy)]
            ref_index = np.where(mr)[0][np.isin(ref_doy, common_doy)]
            h = hind_o3.isel(lead_time=hind_index)
            r = ref_o3.isel(lead_time=ref_index)
        else:
            h = hind_o3.isel(lead_time=mh)
            r = ref_o3.isel(lead_time=mr)
    else:
        h = hind_o3.isel(lead_time=mh)
        r = ref_o3.isel(lead_time=mr)
    n = min(h.sizes.get("lead_time", 0), r.sizes.get("lead_time", 0))
    if n == 0:
        return pd.DataFrame(columns=["member", "O3_RMSE", "n_days"])
    # Reset positional coordinates after selecting matching calendar days.
    # Otherwise xarray aligns an Apr-initialized hindcast lead_time 0..N with
    # reference lead_time 90..N and silently produces an empty comparison.
    h = h.isel(lead_time=slice(0, n)).assign_coords(lead_time=np.arange(n))
    r = r.isel(lead_time=slice(0, n)).assign_coords(lead_time=np.arange(n))
    rmse = np.sqrt(((h - r) ** 2).mean("lead_time", skipna=True))
    members = [str(v) for v in rmse["member_short"].values] if "member_short" in rmse.coords else [member_short_id(v) for v in rmse["member"].values]
    return pd.DataFrame({"member": members, "O3_RMSE": rmse.values.astype(float), "n_days": n})


def load_epflux(case: str, wave: str):
    """Load cleaned EP-flux ep2 for one case and wave component.

    Dimensions
    ----------
    member, lead_time, plev, lat

    Sign convention
    ---------------
    Raw ep2 is loaded. Use -ep2 for positive upward wave activity.
    """
    path = _product_path(case, "EPflux", wave)
    if not path.exists():
        log_message(f"{case}: missing EPflux {wave} {path}")
        return None, None
    try:
        with xr.open_dataset(path, decode_times=False) as ds:
            ep2 = _assign_member_short(ds["ep2"]).load()
            date = _one_dim_date(ds)
        ep2 = ep2.assign_coords(date=("lead_time", date[: ep2.sizes["lead_time"]]))
        return ep2, date[: ep2.sizes["lead_time"]]
    except Exception as exc:
        log_message(f"{case}: failed load_epflux {wave}: {exc}")
        return None, None


def compute_ep100(case: str, wave: str, window, plev_hpa: float = 100, lat_range: tuple = LAT_EP) -> pd.DataFrame:
    """Compute EP100 member metric for one wave and window.

    Definition
    ----------
    EP100 = mean(-ep2), nearest requested pressure, cos-lat mean over 40-80N by
    default. Positive values mean stronger upward wave activity. This is not
    EP-flux divergence.

    Returns
    -------
    pandas.DataFrame
        Columns member, wave, EP100.
    """
    ep2, date = load_epflux(case, wave)
    if ep2 is None:
        return pd.DataFrame(columns=["member", "wave", "EP100"])
    mask = date_mask_for_case_window(case, date, window[0], window[1])
    if mask.sum() == 0:
        log_message(f"{case}: EP100 {wave} empty window {window}")
        return pd.DataFrame(columns=["member", "wave", "EP100"])
    da = -ep2.sel(plev=float(plev_hpa) * 100.0, method="nearest")
    metric = coslat_weighted_mean(da, lat_range).isel(lead_time=mask).mean("lead_time", skipna=True)
    members = [str(v) for v in metric["member_short"].values]
    return pd.DataFrame({"member": members, "wave": wave, "EP100": metric.values.astype(float)})


def compute_all_ep100(case: str, window) -> pd.DataFrame:
    columns = ["member"] + [f"EP100_{wave}" for wave in WAVES] + ["EP100_wave1_plus_wave2"]
    frames = [compute_ep100(case, wave, window) for wave in WAVES]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=columns)
    long = pd.concat(frames, ignore_index=True)
    if long.duplicated(["member", "wave"]).any():
        log_message(
            f"{case}: duplicate EP100 member-wave rows detected; averaging duplicates before reshape."
        )
        long = long.groupby(["member", "wave"], as_index=False)["EP100"].mean()
    wide = long.pivot(index="member", columns="wave", values="EP100").reset_index()
    for wave in WAVES:
        if wave not in wide:
            wide[wave] = np.nan
    wide = wide.rename(columns={wave: f"EP100_{wave}" for wave in WAVES})
    wide["EP100_wave1_plus_wave2"] = wide["EP100_wave1"] + wide["EP100_wave2"]
    return wide


def compute_ep_vertical_profile(
    case: str,
    wave: str,
    window,
    levels_hpa: Sequence[float] = (300, 200, 150, 100, 70, 50),
    lat_range: tuple = LAT_EP,
) -> pd.DataFrame:
    """Compute member EPFz profile for one wave over a source window.

    Returns
    -------
    pandas.DataFrame
        Columns member, wave, plev_hpa, EPFz. EPFz is mean(-ep2).
    """
    ep2, date = load_epflux(case, wave)
    if ep2 is None:
        return pd.DataFrame(columns=["member", "wave", "plev_hpa", "EPFz"])
    mask = date_mask_for_case_window(case, date, window[0], window[1])
    rows = []
    for lev in levels_hpa:
        da = -ep2.sel(plev=float(lev) * 100.0, method="nearest")
        metric = coslat_weighted_mean(da, lat_range).isel(lead_time=mask).mean("lead_time", skipna=True)
        for member, val in zip(metric["member_short"].values, metric.values):
            rows.append({"member": str(member), "wave": wave, "plev_hpa": float(lev), "EPFz": float(val)})
    return pd.DataFrame(rows)


def _interp_profile_logp(da_var: xr.DataArray, p_hyb: xr.DataArray, p_tgt_pa: float) -> xr.DataArray:
    target = np.array([float(p_tgt_pa)], dtype=float)

    def _interp_col(vcol, pcol):
        vcol = np.asarray(vcol, dtype=float)
        pcol = np.asarray(pcol, dtype=float)
        mask = np.isfinite(vcol) & np.isfinite(pcol) & (pcol > 0)
        if mask.sum() < 2:
            return np.array([np.nan], dtype=float)
        p = pcol[mask]
        v = vcol[mask]
        idx = np.argsort(p)
        return np.interp(np.log(target), np.log(p[idx]), v[idx], left=np.nan, right=np.nan)

    out = xr.apply_ufunc(
        _interp_col,
        da_var,
        p_hyb,
        input_core_dims=[["lev"], ["lev"]],
        output_core_dims=[["plev"]],
        vectorize=True,
        dask="allowed",
        output_dtypes=[float],
    )
    return out.assign_coords(plev=("plev", target)).isel(plev=0, drop=True)


def compute_u60(case: str, plev_hpa: float):
    """Compute or load zonal-mean U at 60N and a pressure level.

    Dimensions
    ----------
    member, lead_time

    Units
    -----
    m s-1.
    """
    cache = CACHE_DIR / f"{case}_U60N{int(plev_hpa)}.nc"
    if cache.exists():
        da = xr.open_dataarray(cache).load()
        if _has_unique_member_ids(da):
            date = np.asarray(da["date"].values, dtype=np.int64) if "date" in da.coords else np.arange(da.sizes["lead_time"])
            return da, date
        log_message(f"{case}: rebuilding stale U60N{plev_hpa} cache with duplicate member ids.")
        da.close()
    files = sorted((HINDCAST_ROOT / case / "U").glob("*.U.nc"))
    if not files:
        log_message(f"{case}: missing U files for U60N{plev_hpa}")
        return None, None
    das, mids = [], []
    for f in files:
        try:
            with xr.open_dataset(f, decode_times=False) as ds:
                date = np.asarray(ds["date"].values, dtype=np.int64)
                # Small level slice around requested pressure for speed.
                sub = ds.sel(lat=60.0, method="nearest")
                p_mid = sub["hyam"] * sub["P0"] + sub["hybm"] * sub["PS"]
                u = _interp_profile_logp(
                    sub["U"].transpose("time", "lon", "lev"),
                    p_mid.transpose("time", "lon", "lev"),
                    float(plev_hpa) * 100.0,
                ).mean("lon", skipna=True).load()
                u = u.rename({"time": "lead_time"}).assign_coords(date=("lead_time", date))
            das.append(u.astype(np.float32))
            mids.append(member_short_id(f.name))
        except Exception as exc:
            log_message(f"{case}: failed U60 member {f.name}: {exc}")
    if not das:
        return None, None
    da = xr.concat(das, dim=pd.Index(mids, name="member"))
    da = _assign_member_short(da)
    da.name = f"U60N{int(plev_hpa)}"
    da.attrs.update({"units": "m s-1", "plev_hpa": float(plev_hpa), "lat": 60.0})
    da.to_netcdf(cache)
    return da, np.asarray(da["date"].values, dtype=np.int64)


def load_bwcn_reference_u60(year: str | int, plev_hpa: float):
    """Load or compute BWCN same-year U60N at one pressure level."""
    year_i = int(year)
    cache = CACHE_DIR / f"BWCN_{year_i:04d}_U60N{int(plev_hpa)}.nc"
    if cache.exists():
        da = xr.open_dataarray(cache).load()
        date = np.asarray(da["date"].values, dtype=np.int64) if "date" in da.coords else np.arange(da.sizes["lead_time"])
        return da, date
    path = BWCN_ROOT / "U" / f"BWCN.cam.h3.{year_i:04d}.U.nc"
    if not path.exists():
        log_message(f"BWCN year {year_i:04d}: missing U file {path}")
        return None, None
    try:
        with xr.open_dataset(path, decode_times=False) as ds:
            date = np.asarray(ds["date"].values, dtype=np.int64)
            sub = ds.sel(lat=60.0, method="nearest")
            p_mid = sub["hyam"] * sub["P0"] + sub["hybm"] * sub["PS"]
            u = _interp_profile_logp(
                sub["U"].transpose("time", "lon", "lev"),
                p_mid.transpose("time", "lon", "lev"),
                float(plev_hpa) * 100.0,
            ).mean("lon", skipna=True).load()
        da = u.rename({"time": "lead_time"}).assign_coords(date=("lead_time", date))
        da.name = f"BWCN_U60N{int(plev_hpa)}"
        da.attrs.update({"units": "m s-1", "plev_hpa": float(plev_hpa), "lat": 60.0})
        da.astype(np.float32).to_netcdf(cache)
        return da, date
    except Exception as exc:
        log_message(f"BWCN year {year_i:04d}: failed U60N{plev_hpa}: {exc}")
        return None, None


def load_b2000_u60_climatology(plev_hpa: float):
    """Load B2000 all-year climatological zonal-mean U at 60N."""
    cache = CACHE_DIR / f"B2000_U60N{int(plev_hpa)}_climatology_doy.nc"
    if cache.exists():
        return xr.open_dataarray(cache).load()
    path = B2000_ROOT / "climatology" / "U_climatology_plev_doy.nc"
    if not path.exists():
        log_message(f"missing B2000 U climatology {path}")
        return None
    try:
        with xr.open_dataset(path, decode_times=False) as ds:
            da = (
                ds["U_clim_all"]
                .sel(plev=float(plev_hpa) * 100.0, method="nearest")
                .sel(lat=60.0, method="nearest")
                .mean("lon", skipna=True)
                .load()
            )
        da = da.assign_coords(doy=np.asarray(da["doy"].values, dtype=int))
        da.name = f"B2000_U60N{int(plev_hpa)}_climatology"
        da.attrs.update({"units": "m s-1", "plev_hpa": float(plev_hpa), "lat": 60.0})
        da.astype(np.float32).to_netcdf(cache)
        return da
    except Exception as exc:
        log_message(f"failed B2000 U60N{plev_hpa} climatology: {exc}")
        return None


def compute_fwd_from_u60n50(u60n50) -> pd.DataFrame:
    """Compute final warming date from member U60N50.

    Definition
    ----------
    First day when U60N50 < 7 m/s and remains below that threshold for at least
    10 consecutive days.

    Returns
    -------
    pandas.DataFrame
        Columns member and FWD_DOY.
    """
    if u60n50 is None:
        return pd.DataFrame(columns=["member", "FWD_DOY"])
    dates = np.asarray(u60n50["date"].values if "date" in u60n50.coords else np.arange(u60n50.sizes["lead_time"]))
    doys = date_to_doy(dates) if dates.max() > 1000 else np.arange(len(dates)) + 1
    rows = []
    for i, mid in enumerate(u60n50["member"].values):
        vals = np.asarray(u60n50.isel(member=i).values, dtype=float)
        fwd = np.nan
        below = vals < U60_THRESH
        for t in range(0, max(0, len(vals) - FWD_PERSIST_DAYS + 1)):
            if below[t] and below[t:t + FWD_PERSIST_DAYS].all():
                fwd = float(doys[t])
                break
        rows.append({"member": member_short_id(mid), "FWD_DOY": fwd})
    return pd.DataFrame(rows)


def load_fwd_product(case: str) -> pd.DataFrame:
    """Load existing cleaned FWD product when available."""
    path = _product_path(case, "FWD")
    if not path.exists():
        log_message(f"{case}: missing FWD product {path}")
        return pd.DataFrame(columns=["member", "plev_hpa", "FWD_DOY"])
    try:
        with xr.open_dataset(path, decode_times=False) as ds:
            var = list(ds.data_vars)[0]
            da = _assign_member_short(ds[var]).load()
        # The exact variable name differs across experiments; keep a long table.
        plev_name = "plev" if "plev" in da.dims else ("pressure" if "pressure" in da.dims else None)
        rows = []
        if plev_name is not None and "member" in da.dims:
            for lev in da[plev_name].values:
                for mi, mid in enumerate(da["member"].values):
                    rows.append({"member": member_short_id(mid), "plev_hpa": float(lev) / 100.0 if float(lev) > 2000 else float(lev), "FWD_DOY": float(da.sel({plev_name: lev}).isel(member=mi))})
        return pd.DataFrame(rows)
    except Exception as exc:
        log_message(f"{case}: failed FWD load: {exc}")
        return pd.DataFrame(columns=["member", "plev_hpa", "FWD_DOY"])


def compute_spread_onset(da_member_time) -> dict:
    """Detect ensemble-spread onset from member-time data.

    Definition
    ----------
    spread(t) is member standard deviation. After subtracting the initial spread
    and standardizing, onset is the first 5-day-running-mean value exceeding 50%
    of the maximum and persisting at least 5 days.

    Returns
    -------
    dict
        onset_index, onset_doy, and diagnostic spread arrays.
    """
    if da_member_time is None or "member" not in da_member_time.dims:
        return {"onset_index": np.nan, "onset_doy": np.nan}
    time_dim = "lead_time" if "lead_time" in da_member_time.dims else "time"
    spread = da_member_time.std("member", skipna=True)
    vals = np.asarray(spread.values, dtype=float)
    if len(vals) == 0 or not np.isfinite(vals).any():
        return {"onset_index": np.nan, "onset_doy": np.nan}
    delta = vals - vals[0]
    scale = np.nanstd(delta[1:])
    if not np.isfinite(scale) or scale == 0:
        scale = 1.0
    standardized = delta / scale
    rm = pd.Series(standardized).rolling(5, center=True, min_periods=1).mean().to_numpy()
    thresh = 0.5 * np.nanmax(rm)
    onset = np.nan
    for i in range(0, max(0, len(rm) - 4)):
        if np.all(rm[i:i + 5] >= thresh):
            onset = float(i)
            break
    dates = np.asarray(da_member_time["date"].values if "date" in da_member_time.coords else np.arange(len(vals)))
    doys = date_to_doy(dates) if dates.max() > 1000 else np.arange(len(vals)) + 1
    return {
        "onset_index": onset,
        "onset_lead_day": onset,
        "onset_doy": float(doys[int(onset)]) if np.isfinite(onset) else np.nan,
        "threshold": float(thresh),
        "spread": vals,
        "standardized": standardized,
        "running5": rm,
    }


def _window_token(window) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", window_to_label(window)).strip("_")


def load_or_build_z300(case: str, window):
    """Compute or load member 300 hPa height for a case and time window.

    Dimensions
    ----------
    member, lat, lon

    Returns
    -------
    xarray.DataArray | None
        Time-mean 300 hPa height in meters.
    """
    cache = CACHE_DIR / f"{case}_Z300_{_window_token(window)}.nc"
    if cache.exists():
        da = xr.open_dataarray(cache).load()
        if _has_unique_member_ids(da):
            return da
        log_message(f"{case}: rebuilding stale Z300 cache with duplicate member ids.")
        da.close()
    files = sorted((HINDCAST_ROOT / case / "Z3").glob("*.Z3.nc"))
    if not files:
        log_message(f"{case}: missing Z3 files")
        return None
    das, mids = [], []
    for f in files:
        try:
            with xr.open_dataset(f, decode_times=False) as ds:
                date = np.asarray(ds["date"].values, dtype=np.int64)
                mask = date_mask_for_case_window(case, date, window[0], window[1])
                sub = ds.isel(time=mask).sel(lat=slice(LAT_Z300[0], LAT_Z300[1]))
                p_mid = sub["hyam"] * sub["P0"] + sub["hybm"] * sub["PS"]
                z = _interp_profile_logp(
                    sub["Z3"].transpose("time", "lat", "lon", "lev"),
                    p_mid.transpose("time", "lat", "lon", "lev"),
                    PLEV_Z300_HPA * 100.0,
                ).mean("time", skipna=True).load()
            das.append(z.astype(np.float32))
            mids.append(member_short_id(f.name))
        except Exception as exc:
            log_message(f"{case}: failed Z300 member {f.name}: {exc}")
    if not das:
        return None
    da = xr.concat(das, dim=pd.Index(mids, name="member"))
    da = _assign_member_short(da)
    da.name = "Z300"
    da.attrs.update({"units": "m", "plev_hpa": PLEV_Z300_HPA, "window": str(window)})
    da.to_netcdf(cache)
    return da


def load_or_build_bwcn_z300(year: str | int, window):
    """Load or compute BWCN same-year 300 hPa height for one window."""
    year_i = int(year)
    cache = CACHE_DIR / f"BWCN_{year_i:04d}_Z300_{_window_token(window)}.nc"
    if cache.exists():
        return xr.open_dataarray(cache).load()
    path = BWCN_ROOT / "Z3" / f"BWCN.cam.h3.{year_i:04d}.Z3.nc"
    if not path.exists():
        log_message(f"BWCN year {year_i:04d}: missing Z3 file {path}")
        return None
    try:
        with xr.open_dataset(path, decode_times=False) as ds:
            date = np.asarray(ds["date"].values, dtype=np.int64)
            mask = date_mask_for_case_window(f"{year_i:04d}-01", date, window[0], window[1])
            if mask.sum() == 0:
                log_message(f"BWCN year {year_i:04d}: empty Z300 window {window}")
                return None
            sub = ds.isel(time=mask).sel(lat=slice(LAT_Z300[0], LAT_Z300[1]))
            p_mid = sub["hyam"] * sub["P0"] + sub["hybm"] * sub["PS"]
            z = _interp_profile_logp(
                sub["Z3"].transpose("time", "lat", "lon", "lev"),
                p_mid.transpose("time", "lat", "lon", "lev"),
                PLEV_Z300_HPA * 100.0,
            ).mean("time", skipna=True).load()
        z.name = "BWCN_Z300"
        z.attrs.update({"units": "m", "plev_hpa": PLEV_Z300_HPA, "window": str(window)})
        z.astype(np.float32).to_netcdf(cache)
        return z
    except Exception as exc:
        log_message(f"BWCN year {year_i:04d}: failed Z300 {window}: {exc}")
        return None


def compute_z300_stationary_anomaly(z300):
    """Remove the zonal mean from Z300.

    Parameters
    ----------
    z300:
        DataArray with lon coordinate and optionally member dimension.

    Returns
    -------
    xarray.DataArray
        Stationary-wave anomaly.
    """
    if z300 is None:
        return None
    return z300 - z300.mean("lon", skipna=True)


def compute_z300_climatological_stationary_target(month_or_window):
    """Build B2000 climatological 300 hPa stationary-wave target.

    Parameters
    ----------
    month_or_window:
        Month label such as "Jan" or a MM-DD window.

    Returns
    -------
    xarray.DataArray | None
        Lat-lon stationary-wave climatology with zonal mean removed.
    """
    cache = CACHE_DIR / f"B2000_Z300_stationary_target_{_window_token(MONTH_WINDOWS.get(month_or_window, month_or_window))}.nc"
    if cache.exists():
        return xr.open_dataarray(cache).load()
    clim_file = B2000_ROOT / "climatology" / "Z3_climatology_plev_doy.nc"
    if not clim_file.exists():
        log_message(f"missing B2000 Z3 climatology {clim_file}")
        return None
    window = MONTH_WINDOWS[month_or_window] if isinstance(month_or_window, str) and month_or_window in MONTH_WINDOWS else month_or_window
    if isinstance(window[0], int):
        start_doy, end_doy = int(window[0]), int(window[1])
    else:
        start_doy, end_doy = mmdd_to_doy(*window[0]), mmdd_to_doy(*window[1])
    doys = np.arange(start_doy, end_doy + 1)
    try:
        with xr.open_dataset(clim_file, decode_times=False) as ds:
            z = ds["Z3_clim_all"].sel(plev=PLEV_Z300_HPA * 100.0, method="nearest")
            avail = np.intersect1d(z["doy"].values.astype(int), doys)
            if len(avail) == 0:
                log_message(f"B2000 Z3 climatology empty doys {start_doy}-{end_doy}")
                return None
            z = select_latband(z.sel(doy=avail).mean("doy", skipna=True), LAT_Z300).load()
        target = (z - z.mean("lon", skipna=True)).astype(np.float32)
        target.name = "Z300_B2000_stationary_target"
        target.to_netcdf(cache)
        return target
    except Exception as exc:
        log_message(f"failed B2000 Z300 target {month_or_window}: {exc}")
        return None


def load_b2000_o3_partial_climatology():
    """Load B2000 all-year O3 partial-column climatology by day of year."""
    cache = CACHE_DIR / "B2000_O3_partial_60_90N_30_70hPa_climatology_doy.nc"
    if cache.exists():
        return xr.open_dataarray(cache).load()
    path = B2000_ROOT / "partial_O3" / "B2000WCN_partial_O3_all_ranges.nc"
    if not path.exists():
        log_message(f"missing B2000 partial O3 climatology source {path}")
        return None
    try:
        with xr.open_dataset(path, decode_times=False) as ds:
            da = ds[O3_VAR]
            date = np.asarray(ds["date"].values, dtype=np.int64)
            doy = date_to_doy(date)
            da = da.assign_coords(doy=("time", doy)).groupby("doy").mean("time", skipna=True).load()
        da.name = "B2000_O3_partial_60_90N_30_70hPa_climatology"
        da.attrs.update({"units": "DU"})
        da.astype(np.float32).to_netcdf(cache)
        return da
    except Exception as exc:
        log_message(f"failed B2000 partial O3 climatology: {exc}")
        return None


def load_bwcn_ep100_reference(year: str | int, wave: str, window, plev_hpa: float = 100, lat_range: tuple = LAT_EP) -> float:
    """Compute BWCN same-year EP100 reference from the long BWCN EP-flux file.

    The available BWCN reference product is omega-corrected
    ``EPflux_daily_ubar_wcorr``. Current hindcast EP100 uses
    ``EPflux_daily_ubar`` for no-omega-consistent ensemble comparison, so this
    reference anomaly should be interpreted as an approximate pathway error.
    """
    path = BWCN_ROOT / "EPflux_daily_ubar_wcorr" / wave / f"EPFLUX_{wave}_24yr_time_plev_lat.nc"
    if not path.exists():
        log_message(f"BWCN EP100 reference missing {path}")
        return np.nan
    year_i = int(year)
    try:
        with xr.open_dataset(path, decode_times=False) as ds:
            time = np.asarray(ds["time"].values, dtype=float)
            ref_year = (np.floor(time).astype(int) // 365) + 1
            ref_doy = (np.floor(time).astype(int) % 365) + 1
            if isinstance(window[0], (tuple, list)):
                mask = (
                    (ref_year == year_i)
                    & (ref_doy >= mmdd_to_doy(*window[0]))
                    & (ref_doy <= mmdd_to_doy(*window[1]))
                )
            else:
                start = init_doy_for_case(f"{year_i:04d}-01") + int(window[0])
                end = init_doy_for_case(f"{year_i:04d}-01") + int(window[1])
                mask = (ref_year == year_i) & (ref_doy >= start) & (ref_doy <= end)
            if mask.sum() == 0:
                return np.nan
            da = -ds["ep2"].sel(plev=float(plev_hpa) * 100.0, method="nearest").isel(time=mask)
            val = coslat_weighted_mean(da, lat_range).mean("time", skipna=True).load()
        return float(val.values)
    except Exception as exc:
        log_message(f"BWCN year {year_i:04d}: failed EP100 reference {wave} {window}: {exc}")
        return np.nan


def weighted_pattern_corr(a, b) -> float:
    """Cos-lat weighted spatial pattern correlation over shared lat-lon grid."""
    if a is None or b is None:
        return np.nan
    a, b = xr.align(select_latband(a, LAT_Z300), select_latband(b, LAT_Z300), join="inner")
    w = np.cos(np.deg2rad(a["lat"])).clip(0, 1)
    aa = a - a.weighted(w).mean(("lat", "lon"), skipna=True)
    bb = b - b.weighted(w).mean(("lat", "lon"), skipna=True)
    num = (aa * bb * w).sum(skipna=True)
    den = np.sqrt((aa * aa * w).sum(skipna=True) * (bb * bb * w).sum(skipna=True))
    return float((num / den).values)


def weighted_projection(a, target) -> float:
    """Project a stationary anomaly onto a target pattern using cos-lat weights."""
    if a is None or target is None:
        return np.nan
    a, target = xr.align(select_latband(a, LAT_Z300), select_latband(target, LAT_Z300), join="inner")
    a = compute_z300_stationary_anomaly(a)
    target = compute_z300_stationary_anomaly(target)
    w = np.cos(np.deg2rad(a["lat"])).clip(0, 1)
    num = (a * target * w).sum(skipna=True)
    den = (target * target * w).sum(skipna=True)
    return float((num / den).values)


def z300_wave_amplitude(z300_stationary_anomaly, k: int) -> float:
    """Cos-lat mean Fourier amplitude of a Z300 stationary anomaly.

    Parameters
    ----------
    z300_stationary_anomaly:
        Lat-lon DataArray with zonal mean removed.
    k:
        Zonal wavenumber.

    Returns
    -------
    float
        Wave-k amplitude in meters.
    """
    if z300_stationary_anomaly is None:
        return np.nan
    sub = select_latband(z300_stationary_anomaly, LAT_Z300)
    arr = np.asarray(sub.transpose("lat", "lon").values, dtype=float)
    lon = np.deg2rad(sub["lon"].values)
    a = np.nanmean(arr * np.cos(k * lon)[None, :], axis=1) * 2.0
    b = np.nanmean(arr * np.sin(k * lon)[None, :], axis=1) * 2.0
    amp = np.sqrt(a * a + b * b)
    w = np.cos(np.deg2rad(sub["lat"].values)).clip(0, 1)
    return float(np.nansum(amp * w) / np.nansum(w))


def pointwise_member_correlation(field_member_lat_lon, metric_member: pd.Series):
    """Compute pointwise member correlation between a lat-lon field and metric.

    Parameters
    ----------
    field_member_lat_lon:
        DataArray with member, lat, lon dimensions.
    metric_member:
        pandas Series indexed by member id.

    Returns
    -------
    xarray.Dataset
        Variables R and p on lat-lon grid.
    """
    da = field_member_lat_lon
    if da is None or "member" not in da.dims:
        return None
    da = _assign_member_short(da).swap_dims({"member": "member_short"})
    common = [m for m in da["member_short"].values if m in metric_member.index]
    if len(common) < 3:
        return None
    da = da.sel(member_short=common)
    x = np.asarray([metric_member.loc[m] for m in common], dtype=float)
    y = da.values
    x_anom = x - np.nanmean(x)
    y_anom = y - np.nanmean(y, axis=0)
    cov = np.nanmean(x_anom[:, None, None] * y_anom, axis=0)
    r = cov / (np.nanstd(x) * np.nanstd(y, axis=0))
    n = np.isfinite(y).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        tval = r * np.sqrt((n - 2) / (1 - r * r))
    # Avoid importing scipy.stats.t here repeatedly; p is approximated by pearsonr per point where possible.
    p = np.full_like(r, np.nan, dtype=float)
    from scipy.stats import t as student_t
    p = 2 * student_t.sf(np.abs(tval), np.maximum(n - 2, 1))
    return xr.Dataset(
        {
            "R": (("lat", "lon"), r.astype(np.float32)),
            "p": (("lat", "lon"), p.astype(np.float32)),
        },
        coords={"lat": da["lat"], "lon": da["lon"]},
    )


def bootstrap_mean_ci(x, n_boot: int = 5000, ci: int = 95) -> dict:
    """Bootstrap mean confidence interval for a one-dimensional sample.

    Returns
    -------
    dict
        mean, lo, hi, n.
    """
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": np.nan, "lo": np.nan, "hi": np.nan, "n": 0}
    rng = np.random.default_rng(42)
    draws = rng.choice(arr, size=(int(n_boot), arr.size), replace=True).mean(axis=1)
    alpha = (100 - ci) / 2
    return {
        "mean": float(np.mean(arr)),
        "lo": float(np.percentile(draws, alpha)),
        "hi": float(np.percentile(draws, 100 - alpha)),
        "n": int(arr.size),
    }


def o3_ma_min(da, date) -> pd.DataFrame:
    """Compute March-April minimum O3 for reference or member O3 data."""
    if da is None:
        return pd.DataFrame()
    mask = date_mask_from_mmdd_or_leadday(date, (3, 1), (4, 30))
    sub = da.isel(lead_time=mask)
    if "member" in sub.dims:
        vals = sub.min("lead_time", skipna=True)
        return pd.DataFrame({"member": [member_short_id(v) for v in vals["member"].values], "O3_MA_min": vals.values.astype(float)})
    return pd.DataFrame({"member": ["reference"], "O3_MA_min": [float(sub.min(skipna=True))]})


def case_source_table(case: str, source_key: str = "primary") -> pd.DataFrame:
    """Assemble core member metrics for one case: EP100 waves, O3 RMSE, U means, and FWD."""
    meta = parse_case_name(case)
    window = source_windows_for_case(case)[source_key]
    target = target_window_for_case(case)
    ep = compute_all_ep100(case, window)
    o3, date = load_hindcast_o3(case)
    ref, ref_date = load_bwcn_reference_o3(meta["year"])
    if meta["year"] == "0003":
        log_message("0003 uses BWCN .002 reference year 0003; special .007 branch caveat applies.", "case_caveats.log")
    rmse = compute_o3_rmse(o3, ref, target[0], target[1])
    out = ep.merge(rmse, on="member", how="outer") if not ep.empty else rmse
    for plev in [10, 50]:
        u, udate = compute_u60(case, plev)
        if u is not None:
            mask = date_mask_for_case_window(case, udate, target[0], target[1])
            metric = u.isel(lead_time=mask).mean("lead_time", skipna=True)
            udf = pd.DataFrame({"member": [member_short_id(v) for v in metric["member"].values], f"U60N{plev}_mean": metric.values.astype(float)})
            out = out.merge(udf, on="member", how="outer")
            if plev == 50:
                fwd = compute_fwd_from_u60n50(u)
                out = out.merge(fwd, on="member", how="outer")
    out["case"] = case
    out["year"] = meta["year"]
    out["init_month"] = meta["init_month"]
    out["config"] = meta["config"]
    out["perturbation"] = meta["perturbation"]
    out["source_window"] = window_to_label(window)
    out["target_window"] = window_to_label(target)
    return out


def paired_int_nocoupl_cases(inventory: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Return INT/NOCOUPL small-perturbation pairs by year and init month."""
    if inventory is None:
        inventory = discover_hindcast_cases()
    small = inventory.loc[inventory["perturbation"] == "small_temperature"].copy()
    rows = []
    for (year, init), sub in small.groupby(["year", "init_month"]):
        int_cases = sub.loc[sub["config"] == "INT", "case"].tolist()
        nc_cases = sub.loc[sub["config"] == "NOCOUPL", "case"].tolist()
        if int_cases and nc_cases:
            rows.append({"year": year, "init_month": init, "INT_case": int_cases[0], "NOCOUPL_case": nc_cases[0]})
        elif year == "0003" and int_cases:
            log_message(f"{year}-{init}: INT-only special branch; no forced NOCOUPL pair.", "case_caveats.log")
    return pd.DataFrame(rows)


def write_figure_guide() -> Path:
    """Write EXTENTION_ANALYSIS_FIGURE_GUIDE.md from the figure manifest."""
    manifest = TAB_DIR / "figure_manifest.csv"
    out = OUT_ROOT / "EXTENTION_ANALYSIS_FIGURE_GUIDE.md"
    if not manifest.exists():
        out.write_text("# Extention Analysis Figure Guide\n\nNo figures have been registered yet.\n")
        return out
    df = pd.read_csv(manifest).sort_values(["notebook_source", "figure"])
    lines = [
        "# Extention Analysis Figure Guide",
        "",
        "This file is generated from `outputs/tables/figure_manifest.csv`.",
        "",
    ]
    for _, row in df.iterrows():
        lines.extend([
            f"## {row['figure']}",
            "",
            f"- notebook source: `{row['notebook_source']}`",
            f"- scientific question: {row['scientific_question']}",
            f"- variables and windows: {row['variables_and_windows']}",
            f"- short interpretation: {row['short_interpretation']}",
            f"- caveat: {row['caveat']}",
            f"- supporting csv table: `{row['supporting_csv_table']}`",
            f"- png: `{row['png']}`",
            f"- pdf: `{row['pdf']}`",
            "",
        ])
    out.write_text("\n".join(lines))
    return out


def empty_figure(message: str, title: str = "No data"):
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    ax.set_title(title)
    return fig
