#!/usr/bin/env python3
"""Time-boxed backfill for baseline RH/WIND max/min columns.

Purpose
- Add missing columns to existing monthly baseline CSVs:
  - rh_pct_max, rh_pct_min
  - wind_mps_max, wind_mps_min

Design
- Resumable: skip files already containing all target columns (non-empty)
- Timeboxed: stop gracefully after TIME_BUDGET_SEC (default 5h)
- Source of truth: each month's cached inst nc file under
  SEAS5/baseline/<YYYY>/<MM>/_cache_nc/initYYYY-MM-01_inst.nc
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import xarray as xr


# =========================
# Config (env)
# =========================
BASELINE_ROOT = Path(os.getenv("BASELINE_ROOT", "SEAS5/baseline"))
YEAR_MIN = int(os.getenv("BASELINE_YEAR_MIN", "2000"))
YEAR_MAX = int(os.getenv("BASELINE_YEAR_MAX", "2025"))
INIT_MONTHS = os.getenv("INIT_MONTHS", "1-12").strip()

TIME_BUDGET_SEC = int(os.getenv("TIME_BUDGET_SEC", str(5 * 3600)))
STOP_GRACE_SEC = int(os.getenv("STOP_GRACE_SEC", "120"))
DEBUG = os.getenv("DEBUG", "0").strip() in ("1", "true", "True", "yes", "YES")

TARGET_COLS = ["rh_pct_max", "rh_pct_min", "wind_mps_max", "wind_mps_min"]


# =========================
# Logging/time helpers
# =========================
t0 = time.time()


def log(msg: str):
    print(msg, flush=True)


def dlog(msg: str):
    if DEBUG:
        log(msg)


def time_left_sec() -> float:
    return TIME_BUDGET_SEC - (time.time() - t0)


def should_stop_now() -> bool:
    return time_left_sec() <= STOP_GRACE_SEC


# =========================
# Shared physics/helpers
# =========================
def rh_from_t_td_c(t_c: pd.Series, td_c: pd.Series) -> pd.Series:
    a, b = 17.625, 243.04
    es = 6.1094 * np.exp(a * t_c / (b + t_c))
    e = 6.1094 * np.exp(a * td_c / (b + td_c))
    rh = 100.0 * (e / es)
    return rh.clip(0.0, 100.0)


def pick_var(ds: xr.Dataset, candidates: List[str]) -> str:
    for c in candidates:
        if c in ds.data_vars:
            return c
    raise KeyError(f"None of {candidates} found in ds vars={list(ds.data_vars)}")


def ensure_valid_time(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "valid_time" in out.columns:
        out["valid_time"] = pd.to_datetime(out["valid_time"], utc=True)
        return out
    if ("forecast_reference_time" in out.columns) and ("forecast_period" in out.columns):
        fp = out["forecast_period"]
        if np.issubdtype(fp.dtype, np.timedelta64):
            delta = fp
        else:
            delta = pd.to_timedelta(pd.to_numeric(fp, errors="coerce"), unit="h")
        frt = pd.to_datetime(out["forecast_reference_time"], utc=True)
        out["valid_time"] = frt + delta
        return out
    if ("time" in out.columns) and ("step" in out.columns):
        step = out["step"]
        if np.issubdtype(step.dtype, np.timedelta64):
            delta = step
        else:
            delta = pd.to_timedelta(pd.to_numeric(step, errors="coerce"), unit="h")
        out["valid_time"] = pd.to_datetime(out["time"], utc=True) + delta
        return out
    raise KeyError(f"cannot build valid_time, columns={out.columns.tolist()}")


def get_init_time(df: pd.DataFrame) -> pd.Timestamp:
    if "time" in df.columns:
        return pd.to_datetime(df["time"].iloc[0], utc=True)
    if "forecast_reference_time" in df.columns:
        return pd.to_datetime(df["forecast_reference_time"].iloc[0], utc=True)
    raise KeyError("cannot find init time column (time or forecast_reference_time)")


# =========================
# Task helpers
# =========================
def parse_init_months(s: str) -> List[int]:
    s = s.strip()
    if "," in s:
        out = [int(x.strip()) for x in s.split(",") if x.strip()]
        return sorted(set(out))
    if "-" in s:
        a, b = [int(x.strip()) for x in s.split("-", 1)]
        if a > b:
            a, b = b, a
        return list(range(a, b + 1))
    return [int(s)]


def plan_tasks() -> List[Tuple[int, int]]:
    months = parse_init_months(INIT_MONTHS)
    return [(y, m) for y in range(YEAR_MIN, YEAR_MAX + 1) for m in months]


def month_paths(y: int, m: int) -> Tuple[Path, Path]:
    month_dir = BASELINE_ROOT / f"{y:04d}" / f"{m:02d}"
    csv_fp = month_dir / f"init{y:04d}-{m:02d}-01.csv"
    inst_nc = month_dir / "_cache_nc" / f"init{y:04d}-{m:02d}-01_inst.nc"
    return csv_fp, inst_nc


def needs_backfill(df: pd.DataFrame) -> bool:
    for c in TARGET_COLS:
        if c not in df.columns:
            return True
        if df[c].notna().sum() == 0:
            return True
    return False


def extrema_from_inst_nc(inst_nc: Path) -> pd.DataFrame:
    ds = xr.open_dataset(inst_nc)
    t2m = pick_var(ds, ["t2m", "2m_temperature"])
    d2m = pick_var(ds, ["d2m", "2m_dewpoint_temperature"])
    u10 = pick_var(ds, ["u10", "10m_u_component_of_wind"])
    v10 = pick_var(ds, ["v10", "10m_v_component_of_wind"])

    df6 = ds[[t2m, d2m, u10, v10]].to_dataframe().reset_index()
    df6 = ensure_valid_time(df6)
    init_time = get_init_time(df6)

    df6["lead_day"] = ((df6["valid_time"] - init_time).dt.total_seconds() // 86400).astype(int)
    df6["t2m_C"] = df6[t2m] - 273.15
    df6["d2m_C"] = df6[d2m] - 273.15
    df6["rh_pct"] = rh_from_t_td_c(df6["t2m_C"], df6["d2m_C"])
    df6["wind_mps"] = np.sqrt(df6[u10] ** 2 + df6[v10] ** 2)

    group_cols = ["latitude", "longitude", "lead_day"]
    if "number" in df6.columns:
        grp_member = ["number"] + group_cols
        member_daily = (
            df6.groupby(grp_member, as_index=False)
            .agg(
                rh_pct_max=("rh_pct", "max"),
                rh_pct_min=("rh_pct", "min"),
                wind_mps_max=("wind_mps", "max"),
                wind_mps_min=("wind_mps", "min"),
            )
        )
        daily = (
            member_daily.groupby(group_cols, as_index=False)
            .agg(
                rh_pct_max=("rh_pct_max", "mean"),
                rh_pct_min=("rh_pct_min", "mean"),
                wind_mps_max=("wind_mps_max", "mean"),
                wind_mps_min=("wind_mps_min", "mean"),
            )
        )
    else:
        daily = (
            df6.groupby(group_cols, as_index=False)
            .agg(
                rh_pct_max=("rh_pct", "max"),
                rh_pct_min=("rh_pct", "min"),
                wind_mps_max=("wind_mps", "max"),
                wind_mps_min=("wind_mps", "min"),
            )
        )

    return daily


def run_one(y: int, m: int) -> str:
    csv_fp, inst_nc = month_paths(y, m)
    if not csv_fp.exists():
        return "missing_csv"

    df = pd.read_csv(csv_fp)
    if df.empty:
        return "empty_csv"
    if not needs_backfill(df):
        return "skip_done"
    if not inst_nc.exists():
        return "missing_cache"

    ext = extrema_from_inst_nc(inst_nc)

    on_cols = ["latitude", "longitude", "lead_day"]
    merged = df.merge(ext, on=on_cols, how="left", suffixes=("", "_new"))

    for c in TARGET_COLS:
        if c in df.columns:
            merged[c] = merged[c].where(merged[c].notna(), merged.get(f"{c}_new"))
            if f"{c}_new" in merged.columns:
                merged = merged.drop(columns=[f"{c}_new"])
        else:
            merged[c] = merged.get(f"{c}_new")
            if f"{c}_new" in merged.columns:
                merged = merged.drop(columns=[f"{c}_new"])

    # keep legacy style column ordering by appending new cols at end only when absent
    ordered_cols = list(df.columns)
    for c in TARGET_COLS:
        if c not in ordered_cols:
            ordered_cols.append(c)
    merged = merged[ordered_cols]

    # basic integrity check
    if len(merged) != len(df):
        raise ValueError(f"row count changed unexpectedly: {csv_fp} {len(df)}->{len(merged)}")

    for c in TARGET_COLS:
        if merged[c].notna().sum() == 0:
            raise ValueError(f"{csv_fp}: backfill failed, all null in {c}")

    merged.to_csv(csv_fp, index=False)
    return "updated"


def main() -> int:
    if not BASELINE_ROOT.exists():
        log(f"[INFO] baseline root not found: {BASELINE_ROOT}")
        return 0

    tasks = plan_tasks()
    log(f"[PLAN] tasks={len(tasks)} years={YEAR_MIN}-{YEAR_MAX} months={INIT_MONTHS}")

    stats = {
        "updated": 0,
        "skip_done": 0,
        "missing_csv": 0,
        "missing_cache": 0,
        "empty_csv": 0,
        "error": 0,
    }

    for (y, m) in tasks:
        if should_stop_now():
            log("[STOP] time budget reached, graceful exit")
            break
        try:
            st = run_one(y, m)
            stats[st] = stats.get(st, 0) + 1
            dlog(f"[TASK] {y}-{m:02d}: {st}")
        except Exception as e:
            stats["error"] += 1
            log(f"[ERROR] {y}-{m:02d}: {e}")

    log("[SUMMARY] " + " ".join([f"{k}={v}" for k, v in stats.items()]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
