#!/usr/bin/env python3
"""Time-boxed SEAS5 latest-forecast updater.

Dataset
- Uses CDS dataset: seasonal-original-single-levels (ECMWF system 51).

Purpose
- Build/update monthly latest forecast files under SEAS5/latest.
- Catch up all missing init months from the latest existing file up to current month.

Output
- SEAS5/latest/initYYYY-MM-01.csv
- SEAS5/latest/_cache_nc/initYYYY-MM-01_{inst,tp}.nc
"""

from __future__ import annotations

import calendar
import os
import re
import time
from datetime import date
from pathlib import Path
from typing import List, Optional, Tuple

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr

TIME_BUDGET_SEC = int(os.getenv("TIME_BUDGET_SEC", str(5 * 3600)))
STOP_GRACE_SEC = int(os.getenv("STOP_GRACE_SEC", "120"))
HORIZON_AHEAD_MONTHS = int(os.getenv("HORIZON_AHEAD_MONTHS", "4"))
RETRY_MAX = int(os.getenv("RETRY_MAX", "4"))
DEBUG = os.getenv("DEBUG", "0").strip() in ("1", "true", "True", "YES", "yes")

LATEST_ROOT = Path(os.getenv("LATEST_ROOT", "SEAS5/latest"))
POINTS_FILE = Path(os.getenv("POINTS_FILE", "points.csv"))
GRID_DEG = float(os.getenv("GRID_DEG", "1.0"))

# fallback points (Taiwan 1-degree grid cell centers)
TAIWAN_LAT_MIN = float(os.getenv("TAIWAN_LAT_MIN", "22"))
TAIWAN_LAT_MAX = float(os.getenv("TAIWAN_LAT_MAX", "25"))
TAIWAN_LON_MIN = float(os.getenv("TAIWAN_LON_MIN", "120"))
TAIWAN_LON_MAX = float(os.getenv("TAIWAN_LON_MAX", "122"))

START_YEAR = os.getenv("START_YEAR", "").strip()
START_MONTH = os.getenv("START_MONTH", "").strip()

DATASET = "seasonal-original-single-levels"
ORIGINATING_CENTRE = "ecmwf"
SYSTEM = "51"
INST_VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]
PRECIP_VARIABLE = ["total_precipitation"]

INST_STEP_HOURS = 6
PRECIP_STEP_HOURS = 24


t0 = time.time()


def log(msg: str) -> None:
    print(msg, flush=True)


def dlog(msg: str) -> None:
    if DEBUG:
        log(msg)


def time_left_sec() -> float:
    return TIME_BUDGET_SEC - (time.time() - t0)


def should_stop_now() -> bool:
    return time_left_sec() <= STOP_GRACE_SEC


def add_months(y: int, m: int, offset: int) -> Tuple[int, int]:
    mm = (m - 1) + offset
    y2 = y + (mm // 12)
    m2 = (mm % 12) + 1
    return y2, m2


def last_day_of_month(y: int, m: int) -> int:
    return calendar.monthrange(y, m)[1]


def horizon_end_date(init_y: int, init_m: int, ahead_months: int) -> date:
    end_y, end_m = add_months(init_y, init_m, ahead_months)
    return date(end_y, end_m, last_day_of_month(end_y, end_m))


def lead_hours_inst(init_date: date, end_date: date) -> List[str]:
    days = (end_date - init_date).days
    last_h = days * 24 + 18
    return [str(h) for h in range(0, last_h + INST_STEP_HOURS, INST_STEP_HOURS)]


def lead_hours_tp(init_date: date, end_date: date) -> List[str]:
    days = (end_date - init_date).days
    last_h = (days + 1) * 24
    return [str(h) for h in range(0, last_h + 1, PRECIP_STEP_HOURS)]


def write_cdsapirc_from_env() -> None:
    url = os.getenv("CDSAPI_URL", "https://cds.climate.copernicus.eu/api").strip()
    if url.endswith("/api/v2"):
        url = url.replace("/api/v2", "/api")
    key = os.getenv("CDSAPI_KEY", "").strip()
    if not key:
        raise RuntimeError("Missing CDSAPI_KEY")

    text = f"url: {url}\nkey: {key}\n"
    p = Path.home() / ".cdsapirc"
    p.write_text(text, encoding="utf-8")


def build_default_points() -> List[Tuple[float, float]]:
    lat_edges = np.arange(TAIWAN_LAT_MIN, TAIWAN_LAT_MAX, GRID_DEG)
    lon_edges = np.arange(TAIWAN_LON_MIN, TAIWAN_LON_MAX, GRID_DEG)
    lats = [float(x + GRID_DEG / 2.0) for x in lat_edges]
    lons = [float(x + GRID_DEG / 2.0) for x in lon_edges]
    return [(lat, lon) for lat in lats for lon in lons]


def load_points() -> List[Tuple[float, float]]:
    if POINTS_FILE.exists():
        df = pd.read_csv(POINTS_FILE)
        lat_col = "lat" if "lat" in df.columns else ("latitude" if "latitude" in df.columns else None)
        lon_col = "lon" if "lon" in df.columns else ("longitude" if "longitude" in df.columns else None)
        if lat_col is None or lon_col is None:
            raise ValueError(f"{POINTS_FILE} must have lat/lon columns")
        return [(float(r[lat_col]), float(r[lon_col])) for _, r in df.iterrows()]

    pts = build_default_points()
    log(f"[POINTS] using default points n={len(pts)}")
    return pts


def area_from_points(points: List[Tuple[float, float]]) -> List[float]:
    if not points:
        raise ValueError("points is empty")
    half = GRID_DEG / 2.0
    lats = [lat for lat, _ in points]
    lons = [lon for _, lon in points]
    return [max(lats) + half, min(lons) - half, min(lats) - half, max(lons) + half]


def build_request(year: int, month: int, leadtime_hours: List[str], variables: List[str], area: List[float]) -> dict:
    return {
        "originating_centre": ORIGINATING_CENTRE,
        "system": SYSTEM,
        "variable": variables,
        "year": [f"{year:04d}"],
        "month": [f"{month:02d}"],
        "day": ["01"],
        "leadtime_hour": leadtime_hours,
        "area": area,
        "data_format": "netcdf",
    }


def is_not_ready_error(err: Exception) -> bool:
    msg = str(err).lower()
    hints = [
        "no data is available",
        "not available",
        "invalid combination",
        "not yet",
        "does not exist",
    ]
    return any(h in msg for h in hints)


def retrieve_with_retry(client: "cdsapi.Client", request: dict, target: Path) -> bool:
    last_err: Optional[Exception] = None
    for i in range(1, RETRY_MAX + 1):
        if should_stop_now():
            raise TimeoutError("time budget reached")
        try:
            client.retrieve(DATASET, request, str(target))
            if target.exists() and target.stat().st_size > 0:
                return True
            raise RuntimeError("downloaded file missing/empty")
        except Exception as e:
            if is_not_ready_error(e):
                log(f"[INFO] {target.name} not published yet: {e}")
                return False
            last_err = e
            log(f"[WARN] retrieve {target.name} failed attempt {i}/{RETRY_MAX}: {e}")
            if i < RETRY_MAX:
                time.sleep(min(10 * (2 ** (i - 1)), 180))
    raise RuntimeError(f"download failed: {last_err}")


def pick_var(ds: xr.Dataset, candidates: List[str]) -> str:
    for c in candidates:
        if c in ds.data_vars:
            return c
    raise KeyError(f"None of {candidates} found")


def ensure_valid_time(df: pd.DataFrame) -> pd.DataFrame:
    if "valid_time" in df.columns:
        df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True)
        return df
    if ("forecast_reference_time" in df.columns) and ("forecast_period" in df.columns):
        frt = pd.to_datetime(df["forecast_reference_time"], utc=True)
        fp = df["forecast_period"]
        if not np.issubdtype(fp.dtype, np.timedelta64):
            fp = pd.to_timedelta(fp)
        df["valid_time"] = frt + fp
        return df
    if ("time" in df.columns) and ("step" in df.columns):
        t0_ = pd.to_datetime(df["time"], utc=True)
        st = df["step"]
        if not np.issubdtype(st.dtype, np.timedelta64):
            st = pd.to_timedelta(st)
        df["valid_time"] = t0_ + st
        return df
    raise KeyError("cannot build valid_time")


def get_init_time(df: pd.DataFrame) -> pd.Timestamp:
    if "time" in df.columns:
        return pd.to_datetime(df["time"].iloc[0], utc=True)
    if "forecast_reference_time" in df.columns:
        return pd.to_datetime(df["forecast_reference_time"].iloc[0], utc=True)
    raise KeyError("cannot find init time")


def rh_from_t_td_c(t_c: pd.Series, td_c: pd.Series) -> pd.Series:
    a, b = 17.625, 243.04
    es = 6.1094 * np.exp(a * t_c / (b + t_c))
    e = 6.1094 * np.exp(a * td_c / (b + td_c))
    return (100.0 * (e / es)).clip(0.0, 100.0)


def to_daily(inst_nc: Path, tp_nc: Path) -> pd.DataFrame:
    ds_inst = xr.open_dataset(inst_nc)
    ds_tp = xr.open_dataset(tp_nc)

    t2m = pick_var(ds_inst, ["t2m", "2m_temperature"])
    d2m = pick_var(ds_inst, ["d2m", "2m_dewpoint_temperature"])
    u10 = pick_var(ds_inst, ["u10", "10m_u_component_of_wind"])
    v10 = pick_var(ds_inst, ["v10", "10m_v_component_of_wind"])
    tpv = pick_var(ds_tp, ["tp", "total_precipitation"])

    df6 = ds_inst[[t2m, d2m, u10, v10]].to_dataframe().reset_index()
    df6 = ensure_valid_time(df6)
    init_time = get_init_time(df6)
    df6["lead_day"] = ((df6["valid_time"] - init_time).dt.total_seconds() // 86400).astype(int)
    df6["t2m_C"] = df6[t2m] - 273.15
    df6["d2m_C"] = df6[d2m] - 273.15
    df6["rh_pct"] = rh_from_t_td_c(df6["t2m_C"], df6["d2m_C"])
    df6["wind_mps"] = np.sqrt(df6[u10] ** 2 + df6[v10] ** 2)

    base_cols = ["number", "latitude", "longitude", "lead_day"] if "number" in df6.columns else ["latitude", "longitude", "lead_day"]
    df6_member = df6.groupby(base_cols, as_index=False).agg(
        t2m_C=("t2m_C", "mean"),
        t2m_C_max=("t2m_C", "max"),
        t2m_C_min=("t2m_C", "min"),
        rh_pct=("rh_pct", "mean"),
        rh_pct_max=("rh_pct", "max"),
        rh_pct_min=("rh_pct", "min"),
        wind_mps=("wind_mps", "mean"),
        wind_mps_max=("wind_mps", "max"),
        wind_mps_min=("wind_mps", "min"),
    )

    if "number" in df6_member.columns:
        df6_daily = df6_member.groupby(["latitude", "longitude", "lead_day"], as_index=False).agg(
            t2m_C=("t2m_C", "mean"),
            t2m_C_max=("t2m_C_max", "mean"),
            t2m_C_min=("t2m_C_min", "mean"),
            rh_pct=("rh_pct", "mean"),
            rh_pct_max=("rh_pct_max", "mean"),
            rh_pct_min=("rh_pct_min", "mean"),
            wind_mps=("wind_mps", "mean"),
            wind_mps_max=("wind_mps_max", "mean"),
            wind_mps_min=("wind_mps_min", "mean"),
            ens_n=("number", "nunique"),
        )
    else:
        df6_daily = df6_member.copy()
        df6_daily["ens_n"] = 1

    dfp = ds_tp[[tpv]].to_dataframe().reset_index()
    dfp = ensure_valid_time(dfp)
    dfp["tp_mm_cum"] = dfp[tpv] * 1000.0
    if "number" in dfp.columns:
        dfp = dfp.sort_values(["number", "latitude", "longitude", "valid_time"])
        dfp["tp_mm_24h"] = dfp.groupby(["number", "latitude", "longitude"])["tp_mm_cum"].diff()
        dfp_member = dfp.dropna(subset=["tp_mm_24h"]).copy()
        dfp_member["lead_day"] = (((dfp_member["valid_time"] - init_time).dt.total_seconds() // 86400).astype(int) - 1)
        dfp_daily = dfp_member.groupby(["latitude", "longitude", "lead_day"], as_index=False).agg(tp_mm_mean=("tp_mm_24h", "mean"))
    else:
        dfp = dfp.sort_values(["latitude", "longitude", "valid_time"])
        dfp["tp_mm_24h"] = dfp.groupby(["latitude", "longitude"])["tp_mm_cum"].diff()
        dfp = dfp.dropna(subset=["tp_mm_24h"]).copy()
        dfp["lead_day"] = (((dfp["valid_time"] - init_time).dt.total_seconds() // 86400).astype(int) - 1)
        dfp_daily = dfp.groupby(["latitude", "longitude", "lead_day"], as_index=False).agg(tp_mm_mean=("tp_mm_24h", "sum"))

    out = df6_daily.merge(dfp_daily, on=["latitude", "longitude", "lead_day"], how="inner")
    return out.sort_values(["latitude", "longitude", "lead_day"]).reset_index(drop=True)


def list_existing_months(root: Path) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    pat = re.compile(r"init(\d{4})-(\d{2})-01\.csv$")
    if not root.exists():
        return out
    for fp in root.glob("init????-??-01.csv"):
        m = pat.search(fp.name)
        if m:
            out.append((int(m.group(1)), int(m.group(2))))
    return sorted(set(out))


def month_iter(start_y: int, start_m: int, end_y: int, end_m: int) -> List[Tuple[int, int]]:
    vals: List[Tuple[int, int]] = []
    y, m = start_y, start_m
    while (y, m) <= (end_y, end_m):
        vals.append((y, m))
        y, m = add_months(y, m, 1)
    return vals


def resolve_targets(today: date) -> List[Tuple[int, int]]:
    if START_YEAR and START_MONTH:
        start = (int(START_YEAR), int(START_MONTH))
    else:
        existing = list_existing_months(LATEST_ROOT)
        if existing:
            start = add_months(existing[-1][0], existing[-1][1], 1)
        else:
            start = (today.year, 1)

    end = (today.year, today.month)
    return month_iter(start[0], start[1], end[0], end[1])


def build_paths(y: int, m: int) -> Tuple[Path, Path, Path]:
    cache_dir = LATEST_ROOT / "_cache_nc"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return (
        LATEST_ROOT / f"init{y:04d}-{m:02d}-01.csv",
        cache_dir / f"init{y:04d}-{m:02d}-01_inst.nc",
        cache_dir / f"init{y:04d}-{m:02d}-01_tp.nc",
    )


def run_one(client: "cdsapi.Client", points: List[Tuple[float, float]], y: int, m: int) -> str:
    out_csv, inst_nc, tp_nc = build_paths(y, m)
    if out_csv.exists() and out_csv.stat().st_size > 0:
        return "skip_exists"

    init_dt = date(y, m, 1)
    end_dt = horizon_end_date(y, m, HORIZON_AHEAD_MONTHS)
    area = area_from_points(points)

    req_inst = build_request(y, m, lead_hours_inst(init_dt, end_dt), INST_VARIABLES, area)
    req_tp = build_request(y, m, lead_hours_tp(init_dt, end_dt), PRECIP_VARIABLE, area)

    ready_inst = retrieve_with_retry(client, req_inst, inst_nc)
    ready_tp = retrieve_with_retry(client, req_tp, tp_nc)
    if not (ready_inst and ready_tp):
        if inst_nc.exists() and inst_nc.stat().st_size == 0:
            inst_nc.unlink(missing_ok=True)
        if tp_nc.exists() and tp_nc.stat().st_size == 0:
            tp_nc.unlink(missing_ok=True)
        return "not_ready"

    df = to_daily(inst_nc, tp_nc)
    df = df[(df["lead_day"] >= 0) & (df["lead_day"] <= (end_dt - init_dt).days)].copy()
    df["init_date"] = pd.Timestamp(init_dt).date()
    df["valid_date"] = (pd.Timestamp(init_dt) + pd.to_timedelta(df["lead_day"], unit="D")).dt.date

    out_cols = [
        "latitude", "longitude", "init_date", "valid_date", "lead_day", "ens_n",
        "t2m_C", "t2m_C_max", "t2m_C_min", "rh_pct", "wind_mps", "tp_mm_mean",
        "rh_pct_max", "rh_pct_min", "wind_mps_max", "wind_mps_min",
    ]
    df[out_cols].to_csv(out_csv, index=False)
    return "updated"


def main() -> int:
    write_cdsapirc_from_env()
    client = cdsapi.Client()
    points = load_points()

    today = date.today()
    targets = resolve_targets(today)
    if not targets:
        log("[INFO] no target months to update")
        return 0

    stats = {"updated": 0, "skip_exists": 0, "not_ready": 0, "error": 0}
    log(f"[PLAN] targets={len(targets)} first={targets[0]} last={targets[-1]}")

    for y, m in targets:
        if should_stop_now():
            log("[STOP] time budget reached")
            break
        try:
            st = run_one(client, points, y, m)
            stats[st] = stats.get(st, 0) + 1
            log(f"[TASK] {y}-{m:02d}: {st}")
        except Exception as e:
            stats["error"] += 1
            log(f"[WARN] {y}-{m:02d}: {e}")

    log("[SUMMARY] " + " ".join([f"{k}={v}" for k, v in stats.items()]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
