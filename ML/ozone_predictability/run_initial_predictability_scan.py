#!/usr/bin/env python3
"""Initial physically constrained ML scan for longrun ozone-loss predictability.

The script trains on B2000WCN001002 and evaluates out-of-fold lead-time skill
from 1 January to 28 February. It also applies the fitted B2000WCN models to
BWCN so that individual years such as BWCN0008 can be inspected.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import brier_score_loss, mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_ROOT = REPO_ROOT / "ML" / "outputs" / "initial_predictability_scan"

MONTH_LENGTHS = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
MONTH_START_DOY0 = np.concatenate([[0], np.cumsum(MONTH_LENGTHS)[:-1]])

TARGET_START_DOY = 60
CUTOFF_DOY_MIN = 1
CUTOFF_DOY_MAX = 59
SHORT_WINDOW_DAYS = 28
LONG_WINDOW_DAYS = 90
EVENT_FRACTION = 0.25
RANDOM_STATE = 42

PREDICTOR_SETS: Mapping[str, List[str]] = OrderedDict(
    [
        (
            "DYN",
            [
                "NAM50_short_mean",
                "NAM50_short_trend",
                "EP2_100hPa_40_80N_long_mean",
            ],
        ),
        ("O3_MEMORY", ["O3_short_mean", "O3_short_trend"]),
        (
            "DYN_PLUS_O3",
            [
                "NAM50_short_mean",
                "NAM50_short_trend",
                "EP2_100hPa_40_80N_long_mean",
                "O3_short_mean",
                "O3_short_trend",
            ],
        ),
    ]
)


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    o3_file: Path
    nam_file: Path
    ep_file: Path


DATASETS: Mapping[str, DatasetConfig] = {
    "B2000WCN001002": DatasetConfig(
        name="B2000WCN001002",
        o3_file=Path(
            "/mnt/soclim0/public_data/weiji/B2000WCN001002_timefixed/partial_O3/"
            "B2000WCN_partial_O3_all_ranges.nc"
        ),
        nam_file=Path(
            "/mnt/soclim0/public_data/weiji/B2000WCN001002_timefixed/NAM/"
            "B2000WCN001002_Vertical_NAM.nc"
        ),
        ep_file=Path(
            "/mnt/soclim0/public_data/weiji/B2000WCN001002_timefixed/EPflux_daily/"
            "all_waves/EPFLUX_all_waves_210yr_time_plev_lat.nc"
        ),
    ),
    "BWCN": DatasetConfig(
        name="BWCN",
        o3_file=Path(
            "/mnt/soclim0/public_data/weiji/BWCN/partial_O3/"
            "BWCN_partial_O3_all_ranges.nc"
        ),
        nam_file=Path("/mnt/soclim0/public_data/weiji/BWCN/NAM/BWCN_Vertical_NAM.nc"),
        ep_file=Path(
            "/mnt/soclim0/public_data/weiji/BWCN/EPflux_daily/all_waves/"
            "EPFLUX_all_waves_24yr_time_plev_lat.nc"
        ),
    ),
}


def ensure_repo_output(path: Path) -> Path:
    resolved = path.resolve()
    repo = REPO_ROOT.resolve()
    if repo not in resolved.parents and resolved != repo:
        raise ValueError(f"Output path must stay inside {repo}: {resolved}")
    return resolved


def require_files(configs: Iterable[DatasetConfig]) -> None:
    missing: List[Path] = []
    for cfg in configs:
        for path in [cfg.o3_file, cfg.nam_file, cfg.ep_file]:
            if not path.exists():
                missing.append(path)
    if missing:
        joined = "\n".join(str(p) for p in missing)
        raise FileNotFoundError(f"Missing required input files:\n{joined}")


def doy_from_month_day(month: np.ndarray, day: np.ndarray) -> np.ndarray:
    return MONTH_START_DOY0[month.astype(int) - 1] + day.astype(int)


def month_day_from_doy(doy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    doy0 = doy.astype(int) - 1
    month = np.searchsorted(np.cumsum(MONTH_LENGTHS), doy0, side="right") + 1
    day = doy0 - MONTH_START_DOY0[month - 1] + 1
    return month.astype(int), day.astype(int)


def month_day_label(month: Sequence[int], day: Sequence[int]) -> List[str]:
    return [f"{int(m):02d}-{int(d):02d}" for m, d in zip(month, day)]


def frame_from_numeric_time(n_time: int, year_start: int = 1) -> pd.DataFrame:
    idx = np.arange(n_time, dtype=int)
    year = idx // 365 + year_start
    doy = idx % 365 + 1
    month, day = month_day_from_doy(doy)
    return pd.DataFrame(
        {
            "year": year.astype(int),
            "doy": doy.astype(int),
            "month": month,
            "day": day,
            "month_day": month_day_label(month, day),
            "abs_day": ((year - 1) * 365 + doy).astype(int),
        }
    )


def frame_from_cam_date(date_values: np.ndarray) -> pd.DataFrame:
    date = date_values.astype(int)
    year = date // 10000
    month = (date % 10000) // 100
    day = date % 100
    doy = doy_from_month_day(month, day)
    return pd.DataFrame(
        {
            "year": year.astype(int),
            "doy": doy.astype(int),
            "month": month.astype(int),
            "day": day.astype(int),
            "month_day": month_day_label(month, day),
            "abs_day": ((year - 1) * 365 + doy).astype(int),
        }
    )


def detect_data_var(ds: xr.Dataset, candidates: Sequence[str]) -> str:
    for name in candidates:
        if name in ds.data_vars:
            return name
    data_vars = [
        name
        for name, da in ds.data_vars.items()
        if da.ndim >= 1 and np.issubdtype(da.dtype, np.number)
    ]
    if not data_vars:
        raise ValueError(f"No numeric data variable found in {list(ds.data_vars)}")
    return data_vars[0]


def select_pressure(da: xr.DataArray, coord_name: str, target_hpa: float) -> xr.DataArray:
    coord = da[coord_name]
    values = np.asarray(coord.values, dtype=float)
    target = target_hpa * 100.0 if np.nanmax(values) > 2000.0 else target_hpa
    nearest = float(values[np.nanargmin(np.abs(values - target))])
    return da.sel({coord_name: nearest})


def load_o3_daily(cfg: DatasetConfig) -> pd.DataFrame:
    print(f"[LOAD] {cfg.name}: O3 partial column")
    with xr.open_dataset(cfg.o3_file, decode_times=False) as ds:
        var = detect_data_var(ds, ["O3_partial_60_90N_30_70hPa"])
        values = ds[var].astype("float64").load().values
        if "date" in ds:
            frame = frame_from_cam_date(ds["date"].load().values)
        else:
            frame = frame_from_numeric_time(values.shape[0])

    frame["O3_30_70_60_90N_DU"] = values
    frame["O3_30_70_60_90N_DU_rm5"] = (
        pd.Series(values).rolling(5, center=True, min_periods=3).mean().to_numpy()
    )
    clim = frame.groupby("month_day")["O3_30_70_60_90N_DU_rm5"].transform("mean")
    frame["O3_climatology_rm5"] = clim
    frame["O3_anom_rm5"] = frame["O3_30_70_60_90N_DU_rm5"] - clim
    return frame


def load_nam50_daily(cfg: DatasetConfig) -> pd.DataFrame:
    print(f"[LOAD] {cfg.name}: NAM 50 hPa")
    with xr.open_dataset(cfg.nam_file, decode_times=False) as ds:
        var = detect_data_var(ds, ["NAM_Vertical", "NAM", "nam_vertical", "nam"])
        da = ds[var]
        lev_name = "lev" if "lev" in da.coords else list(da.dims)[-1]
        values = select_pressure(da, lev_name, 50.0).astype("float64").load().values
    frame = frame_from_numeric_time(values.shape[0])
    frame["NAM50"] = values
    return frame


def load_ep2_100hpa_daily(cfg: DatasetConfig) -> pd.DataFrame:
    print(f"[LOAD] {cfg.name}: EP2 100 hPa 40-80N")
    with xr.open_dataset(cfg.ep_file, decode_times=False) as ds:
        var = detect_data_var(ds, ["ep2", "EP2", "Fz", "EP_flux_z"])
        da = ds[var]
        da100 = select_pressure(da, "plev", 100.0)
        if "lat" in da100.dims:
            lat = da100["lat"]
            sub = da100.sel(lat=slice(40, 80))
            if sub.sizes.get("lat", 0) == 0:
                sub = da100.sel(lat=slice(80, 40))
            weights = np.cos(np.deg2rad(sub["lat"]))
            da100 = sub.weighted(weights).mean("lat")
        values = da100.astype("float64").load().values
    frame = frame_from_numeric_time(values.shape[0])
    frame["EP2_100hPa_40_80N"] = values
    return frame


def build_target_table(o3_df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for year, group in o3_df.groupby("year"):
        ma = group[group["month"].isin([3, 4])].copy()
        if ma["O3_30_70_60_90N_DU_rm5"].notna().sum() < 50:
            continue
        min_idx = ma["O3_30_70_60_90N_DU_rm5"].idxmin()
        rows.append(
            {
                "dataset": dataset_name,
                "year": int(year),
                "MA_O3_anom": float(ma["O3_anom_rm5"].mean()),
                "MA_O3_loss": float(-ma["O3_anom_rm5"].mean()),
                "MA_O3_min_DU": float(ma["O3_30_70_60_90N_DU_rm5"].min()),
                "MA_O3_min_doy": int(ma.loc[min_idx, "doy"]),
            }
        )
    target = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    if target.empty:
        raise ValueError(f"No target years built for {dataset_name}")
    n_event = max(1, int(math.floor(EVENT_FRACTION * len(target))))
    target["low_o3_rank"] = target["MA_O3_min_DU"].rank(method="first", ascending=True)
    target["Low25_min_label"] = (target["low_o3_rank"] <= n_event).astype(int)
    target["event_fraction"] = EVENT_FRACTION
    target["n_event"] = n_event
    return target


def linear_trend(values: np.ndarray, x: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(x)
    if mask.sum() < 5:
        return np.nan
    x0 = x[mask].astype(float)
    y0 = values[mask].astype(float)
    x0 = x0 - x0.mean()
    return float(np.polyfit(x0, y0, 1)[0])


def window_mean(df: pd.DataFrame, value_col: str, start_abs: int, end_abs: int, min_count: int) -> float:
    sub = df[(df["abs_day"] >= start_abs) & (df["abs_day"] <= end_abs)]
    values = sub[value_col].astype(float)
    if values.notna().sum() < min_count:
        return np.nan
    return float(values.mean())


def window_trend(df: pd.DataFrame, value_col: str, start_abs: int, end_abs: int, min_count: int) -> float:
    sub = df[(df["abs_day"] >= start_abs) & (df["abs_day"] <= end_abs)]
    if sub[value_col].notna().sum() < min_count:
        return np.nan
    return linear_trend(sub[value_col].to_numpy(), sub["abs_day"].to_numpy())


def cutoff_label(cutoff_doy: int) -> str:
    month, day = month_day_from_doy(np.array([cutoff_doy]))
    return f"{int(month[0]):02d}-{int(day[0]):02d}"


def build_features_for_cutoff(
    cutoff_doy: int,
    target: pd.DataFrame,
    o3_df: pd.DataFrame,
    nam_df: pd.DataFrame,
    ep_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    label = cutoff_label(cutoff_doy)
    lead_days = TARGET_START_DOY - cutoff_doy
    for _, target_row in target.iterrows():
        year = int(target_row["year"])
        cutoff_abs = (year - 1) * 365 + cutoff_doy
        short_start = cutoff_abs - SHORT_WINDOW_DAYS + 1
        long_start = cutoff_abs - LONG_WINDOW_DAYS + 1
        row = {
            "dataset": target_row["dataset"],
            "year": year,
            "cutoff_doy": int(cutoff_doy),
            "cutoff_label": label,
            "lead_days_to_Mar1": int(lead_days),
            "short_window_start_abs": int(short_start),
            "short_window_end_abs": int(cutoff_abs),
            "long_window_start_abs": int(long_start),
            "long_window_end_abs": int(cutoff_abs),
            "O3_short_mean": window_mean(
                o3_df, "O3_anom_rm5", short_start, cutoff_abs, min_count=14
            ),
            "O3_short_trend": window_trend(
                o3_df, "O3_anom_rm5", short_start, cutoff_abs, min_count=14
            ),
            "NAM50_short_mean": window_mean(
                nam_df, "NAM50", short_start, cutoff_abs, min_count=14
            ),
            "NAM50_short_trend": window_trend(
                nam_df, "NAM50", short_start, cutoff_abs, min_count=14
            ),
            "EP2_100hPa_40_80N_long_mean": window_mean(
                ep_df, "EP2_100hPa_40_80N", long_start, cutoff_abs, min_count=45
            ),
            "MA_O3_loss": float(target_row["MA_O3_loss"]),
            "MA_O3_min_DU": float(target_row["MA_O3_min_DU"]),
            "MA_O3_min_doy": int(target_row["MA_O3_min_doy"]),
            "low_o3_rank": float(target_row["low_o3_rank"]),
            "Low25_min_label": int(target_row["Low25_min_label"]),
        }
        rows.append(row)
    features = pd.DataFrame(rows)
    all_features = sorted({f for group in PREDICTOR_SETS.values() for f in group})
    return features.dropna(subset=all_features + ["MA_O3_loss", "Low25_min_label"]).reset_index(drop=True)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    clim_mse = float(mean_squared_error(y_true, np.full_like(y_true, np.mean(y_true), dtype=float)))
    skill = float(1.0 - mse / clim_mse) if clim_mse > 0 else np.nan
    if len(y_true) >= 3 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        corr, pval = pearsonr(y_true, y_pred)
    else:
        corr, pval = np.nan, np.nan
    return {
        "reg_rmse": rmse,
        "reg_mse_skill_vs_climatology": skill,
        "reg_correlation": float(corr),
        "reg_correlation_pvalue": float(pval),
    }


def top_fraction_alarm(y_true: np.ndarray, score: np.ndarray, fraction: float = EVENT_FRACTION) -> Dict[str, float]:
    valid = np.isfinite(score)
    y = y_true[valid].astype(int)
    s = score[valid].astype(float)
    n = len(y)
    if n == 0 or len(np.unique(y)) < 2:
        return {
            "top25_hit_rate": np.nan,
            "top25_false_alarm_rate": np.nan,
            "top25_precision": np.nan,
            "top25_tp": np.nan,
            "top25_fp": np.nan,
            "top25_tn": np.nan,
            "top25_fn": np.nan,
        }
    n_alarm = max(1, int(math.floor(fraction * n)))
    alarm = np.zeros(n, dtype=int)
    alarm[np.argsort(-s)[:n_alarm]] = 1
    tp = int(((alarm == 1) & (y == 1)).sum())
    fp = int(((alarm == 1) & (y == 0)).sum())
    tn = int(((alarm == 0) & (y == 0)).sum())
    fn = int(((alarm == 0) & (y == 1)).sum())
    hit = tp / (tp + fn) if (tp + fn) else np.nan
    far = fp / (fp + tn) if (fp + tn) else np.nan
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    return {
        "top25_hit_rate": float(hit),
        "top25_false_alarm_rate": float(far),
        "top25_precision": float(precision),
        "top25_tp": tp,
        "top25_fp": fp,
        "top25_tn": tn,
        "top25_fn": fn,
    }


def classification_metrics(y_true: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    valid = np.isfinite(prob)
    y = y_true[valid].astype(int)
    p = prob[valid].astype(float)
    if len(y) == 0 or len(np.unique(y)) < 2:
        out = {
            "cls_roc_auc": np.nan,
            "cls_brier_score": np.nan,
            "cls_brier_skill_vs_climatology": np.nan,
        }
    else:
        brier = float(brier_score_loss(y, p))
        q = float(np.mean(y))
        brier_clim = q * (1.0 - q)
        out = {
            "cls_roc_auc": float(roc_auc_score(y, p)),
            "cls_brier_score": brier,
            "cls_brier_skill_vs_climatology": float(1.0 - brier / brier_clim)
            if brier_clim > 0
            else np.nan,
        }
    out.update(top_fraction_alarm(y_true, prob))
    return out


def make_regressor(model_family: str):
    if model_family == "linear":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("reg", RidgeCV(alphas=np.logspace(-4, 4, 41))),
            ]
        )
    if model_family == "random_forest":
        return RandomForestRegressor(
            n_estimators=120,
            max_depth=4,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=2,
        )
    raise ValueError(f"Unknown model family: {model_family}")


def make_classifier(model_family: str, y: np.ndarray):
    if model_family == "linear":
        folds = min(5, int(np.bincount(y).min()))
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegressionCV(
                        Cs=np.logspace(-3, 3, 13),
                        cv=max(3, folds),
                        penalty="l2",
                        solver="lbfgs",
                        max_iter=5000,
                        scoring="neg_brier_score",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
    if model_family == "random_forest":
        return RandomForestClassifier(
            n_estimators=120,
            max_depth=4,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=2,
        )
    raise ValueError(f"Unknown model family: {model_family}")


def evaluate_one_model(
    cutoff_df: pd.DataFrame,
    features: Sequence[str],
    model_family: str,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    sub = cutoff_df.dropna(subset=list(features) + ["MA_O3_loss", "Low25_min_label"]).copy()
    x = sub[list(features)].to_numpy(dtype=float)
    y_reg = sub["MA_O3_loss"].to_numpy(dtype=float)
    y_cls = sub["Low25_min_label"].to_numpy(dtype=int)
    cv_reg = KFold(n_splits=min(5, len(sub)), shuffle=True, random_state=RANDOM_STATE)
    reg_pred = cross_val_predict(make_regressor(model_family), x, y_reg, cv=cv_reg)
    metrics = regression_metrics(y_reg, reg_pred)

    prob = np.full(len(sub), np.nan, dtype=float)
    counts = np.bincount(y_cls, minlength=2)
    if counts.min() >= 3:
        cv_cls = StratifiedKFold(
            n_splits=min(5, int(counts.min())),
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        prob = cross_val_predict(
            make_classifier(model_family, y_cls),
            x,
            y_cls,
            cv=cv_cls,
            method="predict_proba",
        )[:, 1]
    metrics.update(classification_metrics(y_cls, prob))
    metrics["n_samples"] = int(len(sub))
    metrics["n_events"] = int(y_cls.sum())
    return metrics, reg_pred, prob


def fit_predict_external(
    train_df: pd.DataFrame,
    external_df: pd.DataFrame,
    features: Sequence[str],
    model_family: str,
) -> Tuple[np.ndarray, np.ndarray]:
    train = train_df.dropna(subset=list(features) + ["MA_O3_loss", "Low25_min_label"]).copy()
    ext = external_df.dropna(subset=list(features)).copy()
    x_train = train[list(features)].to_numpy(dtype=float)
    x_ext = ext[list(features)].to_numpy(dtype=float)
    y_reg = train["MA_O3_loss"].to_numpy(dtype=float)
    y_cls = train["Low25_min_label"].to_numpy(dtype=int)
    reg = make_regressor(model_family)
    reg.fit(x_train, y_reg)
    reg_pred = reg.predict(x_ext)
    prob = np.full(len(ext), np.nan, dtype=float)
    if np.bincount(y_cls, minlength=2).min() >= 3:
        clf = make_classifier(model_family, y_cls)
        clf.fit(x_train, y_cls)
        prob = clf.predict_proba(x_ext)[:, 1]
    return reg_pred, prob


def add_probability_ranks(pred: pd.DataFrame) -> pd.DataFrame:
    out = pred.copy()
    group_cols = ["dataset", "cutoff_doy", "predictor_set", "model_family"]
    out["prob_rank_desc"] = np.nan
    out["prob_top25_alarm"] = False
    for _, idx in out.groupby(group_cols).groups.items():
        sub = out.loc[idx]
        valid = sub["pred_low25_probability"].notna()
        valid_idx = sub[valid].index
        if len(valid_idx) == 0:
            continue
        ranks = sub.loc[valid_idx, "pred_low25_probability"].rank(method="first", ascending=False)
        n_alarm = max(1, int(math.floor(EVENT_FRACTION * len(valid_idx))))
        out.loc[valid_idx, "prob_rank_desc"] = ranks
        out.loc[valid_idx, "prob_top25_alarm"] = ranks <= n_alarm
    return out


def is_usable(row: pd.Series) -> bool:
    checks = [
        row.get("cls_roc_auc", np.nan) >= 0.75,
        row.get("cls_brier_skill_vs_climatology", np.nan) > 0.0,
        row.get("top25_hit_rate", np.nan) >= 0.60,
        row.get("top25_false_alarm_rate", np.nan) <= 0.35,
        row.get("reg_correlation", np.nan) >= 0.50,
        row.get("reg_mse_skill_vs_climatology", np.nan) > 0.0,
    ]
    return bool(all(checks))


def sustained_earliest_lead(sub: pd.DataFrame, streak: int = 7) -> float:
    ordered = sub.sort_values("lead_days_to_Mar1", ascending=False).reset_index(drop=True)
    usable = ordered["usable_predictability"].to_numpy(dtype=bool)
    leads = ordered["lead_days_to_Mar1"].to_numpy(dtype=int)
    for i in range(len(ordered)):
        if usable[i : i + streak].size == streak and usable[i : i + streak].all():
            return float(leads[i])
    return np.nan


def summarize_predictability(summary: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for (model_family, predictor_set), sub in summary.groupby(["model_family", "predictor_set"]):
        usable = sub[sub["usable_predictability"]].sort_values("lead_days_to_Mar1", ascending=False)
        rows.append(
            {
                "model_family": model_family,
                "predictor_set": predictor_set,
                "earliest_single_usable_lead_days": float(usable["lead_days_to_Mar1"].max())
                if not usable.empty
                else np.nan,
                "earliest_7day_stable_usable_lead_days": sustained_earliest_lead(sub, streak=7),
                "best_auc": float(sub["cls_roc_auc"].max()),
                "best_reg_correlation": float(sub["reg_correlation"].max()),
                "best_reg_mse_skill": float(sub["reg_mse_skill_vs_climatology"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["earliest_7day_stable_usable_lead_days", "earliest_single_usable_lead_days"],
        ascending=False,
        na_position="last",
    )


def plot_leadtime(summary: pd.DataFrame, metric: str, ylabel: str, outfile: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for (model, pset), sub in summary.groupby(["model_family", "predictor_set"]):
        sub = sub.sort_values("lead_days_to_Mar1")
        ax.plot(
            sub["lead_days_to_Mar1"],
            sub[metric],
            marker="o",
            markersize=3,
            linewidth=1.3,
            label=f"{model} / {pset}",
        )
    ax.set_xlabel("Lead days before 1 March")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(outfile, dpi=180)
    plt.close(fig)


def plot_bwcn0008_trace(trace: pd.DataFrame, outfile: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for (model, pset), sub in trace.groupby(["model_family", "predictor_set"]):
        sub = sub.sort_values("lead_days_to_Mar1")
        ax.plot(
            sub["lead_days_to_Mar1"],
            sub["pred_low25_probability"],
            marker="o",
            markersize=3,
            linewidth=1.3,
            label=f"{model} / {pset}",
        )
    ax.set_xlabel("Lead days before 1 March")
    ax.set_ylabel("Predicted low-O3 event probability")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(outfile, dpi=180)
    plt.close(fig)


def format_float(value: float, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def markdown_table(df: pd.DataFrame, columns: Optional[Sequence[str]] = None) -> str:
    """Render a small DataFrame as Markdown without optional dependencies."""
    if columns is not None:
        frame = df.loc[:, list(columns)].copy()
    else:
        frame = df.copy()
    if frame.empty:
        return "_No rows._"

    def cell(value) -> str:
        if isinstance(value, float):
            return format_float(value)
        if isinstance(value, (np.floating,)):
            return format_float(float(value))
        return str(value)

    headers = list(frame.columns)
    rows = [[cell(v) for v in row] for row in frame.to_numpy()]
    widths = [
        max(len(str(header)), *(len(row[i]) for row in rows))
        for i, header in enumerate(headers)
    ]
    header_line = "| " + " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line] + body)


def write_report(
    out_root: Path,
    summary: pd.DataFrame,
    windows: pd.DataFrame,
    external_summary: pd.DataFrame,
    bwcn0008: pd.DataFrame,
) -> None:
    lines: List[str] = []
    lines.append("# Initial Longrun Ozone-Loss Predictability Scan")
    lines.append("")
    lines.append("## Operational Definition")
    lines.append("")
    lines.append(
        "A severe ozone-loss event is the lowest quartile of March-April minimum "
        "5-day-smoothed 60-90N, 30-70 hPa partial ozone. Predictability is called "
        "usable when B2000WCN out-of-fold skill passes the thresholds documented "
        "in `ML/README.md`."
    )
    lines.append("")
    lines.append("## Earliest Usable Lead")
    lines.append("")
    lines.append(markdown_table(windows))
    lines.append("")
    best = windows.dropna(subset=["earliest_7day_stable_usable_lead_days"]).head(1)
    if not best.empty:
        row = best.iloc[0]
        lines.append(
            "Best stable window: "
            f"{row.model_family} / {row.predictor_set}, "
            f"{format_float(row.earliest_7day_stable_usable_lead_days, 0)} days "
            "before 1 March."
        )
    lines.append("")
    lines.append("## BWCN External Check")
    lines.append("")
    if not external_summary.empty:
        keep_cols = [
            "model_family",
            "predictor_set",
            "best_auc",
            "best_reg_correlation",
            "best_top25_hit_rate",
            "best_top25_false_alarm_rate",
        ]
        lines.append(markdown_table(external_summary, keep_cols))
    lines.append("")
    lines.append("## BWCN0008 Trace")
    lines.append("")
    if bwcn0008.empty:
        lines.append("BWCN year 8 was not present after feature filtering.")
    else:
        alarms = bwcn0008[bwcn0008["prob_top25_alarm"]].copy()
        if alarms.empty:
            lines.append("No model placed BWCN0008 inside its top-quartile alarm set.")
        else:
            grouped = (
                alarms.groupby(["model_family", "predictor_set"])["lead_days_to_Mar1"]
                .max()
                .reset_index(name="earliest_top25_alarm_lead_days")
                .sort_values("earliest_top25_alarm_lead_days", ascending=False)
            )
            lines.append(markdown_table(grouped))
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    for name in [
        "b2000wcn_cv_skill_summary.csv",
        "b2000wcn_cv_predictions.csv",
        "bwcn_external_skill_summary.csv",
        "bwcn_external_predictions.csv",
        "bwcn0008_trace.csv",
        "predictability_windows.csv",
        "leadtime_auc.png",
        "leadtime_reg_correlation.png",
        "bwcn0008_probability_trace.png",
    ]:
        lines.append(f"- `{name}`")
    (out_root / "initial_findings.md").write_text("\n".join(lines) + "\n")


def load_dataset_bundle(cfg: DatasetConfig) -> Dict[str, pd.DataFrame]:
    o3 = load_o3_daily(cfg)
    nam = load_nam50_daily(cfg)
    ep = load_ep2_100hpa_daily(cfg)
    target = build_target_table(o3, cfg.name)
    print(
        f"[TARGET] {cfg.name}: years={len(target)}, "
        f"events={int(target['Low25_min_label'].sum())}"
    )
    return {"o3": o3, "nam": nam, "ep": ep, "target": target}


def external_group_summary(pred: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for (model, pset), sub in pred.groupby(["model_family", "predictor_set"]):
        metric_rows = []
        for _, cut in sub.groupby("cutoff_doy"):
            y = cut["Low25_min_label"].to_numpy(dtype=int)
            prob = cut["pred_low25_probability"].to_numpy(dtype=float)
            reg = regression_metrics(cut["MA_O3_loss"].to_numpy(dtype=float), cut["pred_MA_O3_loss"].to_numpy(dtype=float))
            cls = classification_metrics(y, prob)
            metric_rows.append({**reg, **cls})
        metric_df = pd.DataFrame(metric_rows)
        rows.append(
            {
                "model_family": model,
                "predictor_set": pset,
                "best_auc": float(metric_df["cls_roc_auc"].max()),
                "best_reg_correlation": float(metric_df["reg_correlation"].max()),
                "best_top25_hit_rate": float(metric_df["top25_hit_rate"].max()),
                "best_top25_false_alarm_rate": float(metric_df["top25_false_alarm_rate"].min()),
            }
        )
    return pd.DataFrame(rows)


def run_scan(args: argparse.Namespace) -> None:
    out_root = ensure_repo_output(Path(args.output_root))
    out_root.mkdir(parents=True, exist_ok=True)
    require_files([DATASETS["B2000WCN001002"], DATASETS["BWCN"]])

    metadata = {
        "train_dataset": "B2000WCN001002",
        "external_dataset": "BWCN",
        "target_start_doy": TARGET_START_DOY,
        "cutoff_doy_min": CUTOFF_DOY_MIN,
        "cutoff_doy_max": CUTOFF_DOY_MAX,
        "cutoff_step": args.cutoff_step,
        "short_window_days": SHORT_WINDOW_DAYS,
        "long_window_days": LONG_WINDOW_DAYS,
        "event_fraction": EVENT_FRACTION,
        "predictor_sets": PREDICTOR_SETS,
        "models": args.models,
    }
    (out_root / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

    train = load_dataset_bundle(DATASETS["B2000WCN001002"])
    bwcn = load_dataset_bundle(DATASETS["BWCN"])

    cutoff_doys = list(range(CUTOFF_DOY_MIN, CUTOFF_DOY_MAX + 1, args.cutoff_step))
    if args.max_cutoffs is not None:
        cutoff_doys = cutoff_doys[: args.max_cutoffs]

    summary_rows: List[Dict[str, float]] = []
    cv_pred_rows: List[pd.DataFrame] = []
    external_pred_rows: List[pd.DataFrame] = []
    feature_rows_train: List[pd.DataFrame] = []
    feature_rows_bwcn: List[pd.DataFrame] = []

    for cutoff_doy in cutoff_doys:
        label = cutoff_label(cutoff_doy)
        print(f"[SCAN] cutoff={label} lead={TARGET_START_DOY - cutoff_doy} days")
        train_features = build_features_for_cutoff(
            cutoff_doy, train["target"], train["o3"], train["nam"], train["ep"]
        )
        bwcn_features = build_features_for_cutoff(
            cutoff_doy, bwcn["target"], bwcn["o3"], bwcn["nam"], bwcn["ep"]
        )
        feature_rows_train.append(train_features)
        feature_rows_bwcn.append(bwcn_features)

        for predictor_set, features in PREDICTOR_SETS.items():
            for model_family in args.models:
                metrics, reg_pred, prob = evaluate_one_model(train_features, features, model_family)
                summary_rows.append(
                    {
                        "dataset": "B2000WCN001002",
                        "cutoff_doy": cutoff_doy,
                        "cutoff_label": label,
                        "lead_days_to_Mar1": TARGET_START_DOY - cutoff_doy,
                        "predictor_set": predictor_set,
                        "model_family": model_family,
                        "features": ",".join(features),
                        **metrics,
                    }
                )

                sub_train = train_features.dropna(
                    subset=list(features) + ["MA_O3_loss", "Low25_min_label"]
                ).copy()
                cv_pred_rows.append(
                    sub_train[
                        [
                            "dataset",
                            "year",
                            "cutoff_doy",
                            "cutoff_label",
                            "lead_days_to_Mar1",
                            "MA_O3_loss",
                            "MA_O3_min_DU",
                            "MA_O3_min_doy",
                            "low_o3_rank",
                            "Low25_min_label",
                        ]
                    ].assign(
                        predictor_set=predictor_set,
                        model_family=model_family,
                        pred_MA_O3_loss=reg_pred,
                        pred_low25_probability=prob,
                    )
                )

                ext_sub = bwcn_features.dropna(subset=list(features)).copy()
                ext_reg, ext_prob = fit_predict_external(
                    train_features, ext_sub, features, model_family
                )
                external_pred_rows.append(
                    ext_sub[
                        [
                            "dataset",
                            "year",
                            "cutoff_doy",
                            "cutoff_label",
                            "lead_days_to_Mar1",
                            "MA_O3_loss",
                            "MA_O3_min_DU",
                            "MA_O3_min_doy",
                            "low_o3_rank",
                            "Low25_min_label",
                        ]
                    ].assign(
                        predictor_set=predictor_set,
                        model_family=model_family,
                        pred_MA_O3_loss=ext_reg,
                        pred_low25_probability=ext_prob,
                    )
                )

    summary = pd.DataFrame(summary_rows)
    summary["usable_predictability"] = summary.apply(is_usable, axis=1)
    windows = summarize_predictability(summary)

    cv_pred = add_probability_ranks(pd.concat(cv_pred_rows, ignore_index=True))
    external_pred = add_probability_ranks(pd.concat(external_pred_rows, ignore_index=True))
    bwcn_external_summary = external_group_summary(external_pred)
    bwcn0008 = external_pred[external_pred["year"].astype(int) == 8].copy()

    pd.concat(feature_rows_train, ignore_index=True).to_csv(
        out_root / "b2000wcn_features_by_cutoff.csv", index=False
    )
    pd.concat(feature_rows_bwcn, ignore_index=True).to_csv(
        out_root / "bwcn_features_by_cutoff.csv", index=False
    )
    summary.to_csv(out_root / "b2000wcn_cv_skill_summary.csv", index=False)
    cv_pred.to_csv(out_root / "b2000wcn_cv_predictions.csv", index=False)
    external_pred.to_csv(out_root / "bwcn_external_predictions.csv", index=False)
    bwcn_external_summary.to_csv(out_root / "bwcn_external_skill_summary.csv", index=False)
    bwcn0008.to_csv(out_root / "bwcn0008_trace.csv", index=False)
    windows.to_csv(out_root / "predictability_windows.csv", index=False)

    plot_leadtime(summary, "cls_roc_auc", "Low-O3 event AUC", out_root / "leadtime_auc.png")
    plot_leadtime(
        summary,
        "reg_correlation",
        "Regression correlation",
        out_root / "leadtime_reg_correlation.png",
    )
    if not bwcn0008.empty:
        plot_bwcn0008_trace(bwcn0008, out_root / "bwcn0008_probability_trace.png")
    write_report(out_root, summary, windows, bwcn_external_summary, bwcn0008)
    print(f"[DONE] outputs: {out_root}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--cutoff-step", type=int, default=1)
    parser.add_argument("--max-cutoffs", type=int, default=None)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["linear", "random_forest"],
        default=["linear", "random_forest"],
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    run_scan(parse_args())
