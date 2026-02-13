#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Time-boxed SEAS5 monthly baseline builder (hindcast).

Key behavior
- Resumable: skips outputs that already exist (CSV present & non-empty)
- Timeboxed: stops gracefully after TIME_BUDGET_SEC (default 5 hours)
- Safe exit: exits with code 0 even if unfinished, so GitHub Actions can commit/push partial progress

Outputs (repo-relative)
  ./SEAS5/baseline/<YYYY>/<MM>/lat{LAT:.3f}_lon{LON:.3f}_initYYYY-MM-01.csv
"""

from __future__ import annotations

import os
import time
import calendar
from datetime import date
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import xarray as xr

import cdsapi


# =========================
# Time budget
# =========================
TIME_BUDGET_SEC = int(os.getenv("TIME_BUDGET_SEC", str(5 * 3600)))  # 5 hours default
STOP_GRACE_SEC = int(os.getenv("STOP_GRACE_SEC", "120"))           # keep some buffer


# =========================
# User settings (via env)
# =========================
BASELINE_YEAR_MIN = int(os.getenv("BASELINE_YEAR_MIN", "2020"))
BASELINE_YEAR_MAX = int(os.getenv("BASELINE_YEAR_MAX", "2025"))

# Which init months to build. Examples: "1-12", "1,2,3", "2"
INIT_MONTHS = os.getenv("INIT_MONTHS", "1-12").strip()

# Lead horizon: how many months ahead (excluding the init month)
# Example: 4 -> Jan init covers Jan..May
HORIZON_AHEAD_MONTHS = int(os.getenv("HORIZON_AHEAD_MONTHS", "4"))

# Keep all members for baseline? Default: no (store ensemble mean daily)
KEEP_BASELINE_MEMBERS = os.getenv("KEEP_BASELINE_MEMBERS", "0").strip() in ("1", "true", "True", "YES", "yes")

# Points: if points.csv exists, it overrides DEFAULT_POINTS
DEFAULT_POINTS: List[Tuple[float, float]] = [(23.0, 120.0)]  # (lat, lon)
POINTS_FILE = Path(os.getenv("POINTS_FILE", "points.csv"))

# Output root
OUT_ROOT = Path(os.getenv("OUT_ROOT", "SEAS5")) / "baseline"

# Logging
DEBUG = os.getenv("DEBUG", "0").strip() in ("1", "true", "True", "YES", "yes")

# SEAS5 CDS dataset params
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

RETRY_MAX = int(os.getenv("RETRY_MAX", "5"))


# =========================
# Logging helpers
# =========================
def log(msg: str):
    print(msg, flush=True)

def dlog(msg: str):
    if DEBUG:
        log(msg)


# =========================
# Time helpers
# =========================
t0 = time.time()

def time_left_sec() -> float:
    return TIME_BUDGET_SEC - (time.time() - t0)

def should_stop_now() -> bool:
    return time_left_sec() <= STOP_GRACE_SEC


# =========================
# Date helpers
# =========================
def last_day_of_month(y: int, m: int) -> int:
    return calendar.monthrange(y, m)[1]

def add_months(y: int, m: int, offset: int) -> Tuple[int, int]:
    mm = (m - 1) + offset
    y2 = y + (mm // 12)
    m2 = (mm % 12) + 1
    return y2, m2

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


# =========================
# CDS credentials (.cdsapirc)
# =========================
def write_cdsapirc_from_env() -> None:
    """
    Write ~/.cdsapirc in your proven format:
      url: https://cds.climate.copernicus.eu/api
      key: <token>

    Your secret might be /api/v2; normalize to /api.
    """
    url = os.getenv("CDSAPI_URL", "https://cds.climate.copernicus.eu/api").strip()
    if url.endswith("/api/v2"):
        url = url.replace("/api/v2", "/api")
    key = os.getenv("CDSAPI_KEY", "").strip()
    if not key:
        raise RuntimeError("Missing env CDSAPI_KEY (set it in GitHub Secrets).")

    text = f"url: {url}\nkey: {key}\n"
    candidates = [Path("/root/.cdsapirc"), Path.home() / ".cdsapirc"]
    last_err = None
    for p in candidates:
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
            dlog(f"[cdsapirc] written: {p}")
            return
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to write .cdsapirc: {last_err}")


# =========================
# Request + download
# =========================
def build_request(year: int, init_month: int, leadtime_hours: List[str], variables: List[str], area: List[float]) -> dict:
    return {
        "originating_centre": ORIGINATING_CENTRE,
        "system": SYSTEM,
        "variable": variables,
        "year": [f"{year:04d}"],
        "month": [f"{init_month:02d}"],
        "day": ["01"],
        "leadtime_hour": leadtime_hours,
        "area": area,  # [N, W, S, E]
        "data_format": "netcdf",
    }

def retrieve_with_retry(client: "cdsapi.Client", dataset: str, request: dict, target: Path) -> None:
    if target.exists() and target.stat().st_size > 0:
        dlog(f"[skip nc] exists: {target.name}")
        return

    last_err: Optional[Exception] = None
    for i in range(1, RETRY_MAX + 1):
        if should_stop_now():
            raise TimeoutError("Time budget almost exceeded; stop before starting a new download.")

        try:
            dlog(f"[cds] retrieve {i}/{RETRY_MAX}: {target.name}")
            client.retrieve(dataset, request, str(target))
            if (not target.exists()) or target.stat().st_size == 0:
                raise RuntimeError("downloaded file missing or empty")
            return
        except Exception as e:
            last_err = e
            log(f"[WARN] download failed (attempt {i}/{RETRY_MAX}): {e}")
            if i < RETRY_MAX:
                time.sleep(min(10.0 * (2 ** (i - 1)), 300.0))
    raise RuntimeError(f"download failed after {RETRY_MAX} attempts: {last_err}")


# =========================
# xarray -> daily table (lead_day)
# =========================
def ensure_valid_time(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if "valid_time" in df.columns:
        df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True)
        return df

    if ("forecast_reference_time" in df.columns) and ("forecast_period" in df.columns):
        fp = df["forecast_period"]
        if not np.issubdtype(fp.dtype, np.timedelta64):
            fp = pd.to_timedelta(fp)
        frt = pd.to_datetime(df["forecast_reference_time"], utc=True)
        df["valid_time"] = frt + fp
        return df

    if ("time" in df.columns) and ("step" in df.columns):
        st = df["step"]
        if not np.issubdtype(st.dtype, np.timedelta64):
            st = pd.to_timedelta(st)
        t0_ = pd.to_datetime(df["time"], utc=True)
        df["valid_time"] = t0_ + st
        return df

    raise KeyError(f"{label}: cannot build valid_time, columns={df.columns.tolist()}")

def get_init_time(df: pd.DataFrame) -> pd.Timestamp:
    if "time" in df.columns:
        return pd.to_datetime(df["time"].iloc[0], utc=True)
    if "forecast_reference_time" in df.columns:
        return pd.to_datetime(df["forecast_reference_time"].iloc[0], utc=True)
    raise KeyError("cannot find init time column (time or forecast_reference_time)")

def rh_from_t_td_c(t_c: pd.Series, td_c: pd.Series) -> pd.Series:
    a = 17.625
    b = 243.04
    es = 6.1094 * np.exp(a * t_c / (b + t_c))
    e  = 6.1094 * np.exp(a * td_c / (b + td_c))
    rh = 100.0 * (e / es)
    return rh.clip(0.0, 100.0)

def pick_var(ds: xr.Dataset, candidates: List[str]) -> str:
    for c in candidates:
        if c in ds.data_vars:
            return c
    raise KeyError(f"None of {candidates} found. Vars={list(ds.data_vars)}")

def to_daily_for_point(inst_nc: Path, tp_nc: Path, *, keep_members: bool) -> pd.DataFrame:
    ds_inst = xr.open_dataset(inst_nc)
    ds_tp = xr.open_dataset(tp_nc)

    t2m = pick_var(ds_inst, ["t2m", "2m_temperature"])
    d2m = pick_var(ds_inst, ["d2m", "2m_dewpoint_temperature"])
    u10 = pick_var(ds_inst, ["u10", "10m_u_component_of_wind"])
    v10 = pick_var(ds_inst, ["v10", "10m_v_component_of_wind"])
    tpv = pick_var(ds_tp, ["tp", "total_precipitation"])

    df6 = ds_inst[[t2m, d2m, u10, v10]].to_dataframe().reset_index()
    df6 = ensure_valid_time(df6, "inst")
    init_time = get_init_time(df6)

    df6["lead_day"] = ((df6["valid_time"] - init_time).dt.total_seconds() // 86400).astype(int)
    df6["t2m_C"] = df6[t2m] - 273.15
    df6["d2m_C"] = df6[d2m] - 273.15
    df6["rh_pct"] = rh_from_t_td_c(df6["t2m_C"], df6["d2m_C"])
    df6["wind_mps"] = np.sqrt(df6[u10] ** 2 + df6[v10] ** 2)

    group_cols = ["number", "latitude", "longitude", "lead_day"] if "number" in df6.columns else ["latitude", "longitude", "lead_day"]
    df6_member = (
        df6.groupby(group_cols, as_index=False)
           .agg(
               t2m_C_mean=("t2m_C", "mean"),
               t2m_C_max=("t2m_C", "max"),
               t2m_C_min=("t2m_C", "min"),
               rh_pct_mean=("rh_pct", "mean"),
               wind_mps_mean=("wind_mps", "mean"),
               n6=("t2m_C", "count"),
           )
    )

    if keep_members and ("number" in df6_member.columns):
        df6_daily = df6_member.copy()
        df6_daily["ens_n"] = df6_daily.groupby(["latitude", "longitude", "lead_day"])["number"].transform("nunique")
    else:
        if "number" in df6_member.columns:
            df6_daily = (
                df6_member.groupby(["latitude", "longitude", "lead_day"], as_index=False)
                          .agg(
                              t2m_C_mean=("t2m_C_mean", "mean"),
                              t2m_C_max=("t2m_C_max", "mean"),
                              t2m_C_min=("t2m_C_min", "mean"),
                              rh_pct_mean=("rh_pct_mean", "mean"),
                              wind_mps_mean=("wind_mps_mean", "mean"),
                              ens_n=("number", "nunique"),
                              n6=("n6", "sum"),
                          )
            )
        else:
            df6_daily = df6_member.copy()
            df6_daily["ens_n"] = 1

    dfp = ds_tp[[tpv]].to_dataframe().reset_index()
    dfp = ensure_valid_time(dfp, "tp")
    init_time_p = get_init_time(dfp)
    if init_time_p != init_time and DEBUG:
        log(f"[WARN] init_time mismatch inst={init_time} tp={init_time_p}")

    dfp["lead_day_end"] = ((dfp["valid_time"] - init_time).dt.total_seconds() // 86400).astype(int)
    dfp["tp_mm_cum"] = dfp[tpv] * 1000.0

    if "number" in dfp.columns:
        dfp = dfp.sort_values(["number", "latitude", "longitude", "valid_time"])
        dfp["tp_mm_24h"] = dfp.groupby(["number", "latitude", "longitude"])["tp_mm_cum"].diff()
    else:
        dfp = dfp.sort_values(["latitude", "longitude", "valid_time"])
        dfp["tp_mm_24h"] = dfp.groupby(["latitude", "longitude"])["tp_mm_cum"].diff()

    dfp["tp_mm_24h"] = dfp["tp_mm_24h"].clip(lower=0.0)
    dfp = dfp.dropna(subset=["tp_mm_24h"]).copy()
    dfp["lead_day"] = (dfp["lead_day_end"] - 1).astype(int)

    if keep_members and ("number" in dfp.columns):
        dfp_daily = dfp.groupby(["number", "latitude", "longitude", "lead_day"], as_index=False).agg(tp_mm_sum=("tp_mm_24h", "sum"))
    else:
        if "number" in dfp.columns:
            dfp_member = dfp.groupby(["number", "latitude", "longitude", "lead_day"], as_index=False).agg(tp_mm_sum=("tp_mm_24h", "sum"))
            dfp_daily = dfp_member.groupby(["latitude", "longitude", "lead_day"], as_index=False).agg(tp_mm_sum=("tp_mm_sum", "mean"))
        else:
            dfp_daily = dfp.groupby(["latitude", "longitude", "lead_day"], as_index=False).agg(tp_mm_sum=("tp_mm_24h", "sum"))

    on_cols = ["latitude", "longitude", "lead_day"]
    if keep_members and ("number" in df6_daily.columns) and ("number" in dfp_daily.columns):
        on_cols = ["number"] + on_cols

    df_daily = pd.merge(df6_daily, dfp_daily, on=on_cols, how="left")
    df_daily["tp_mm_sum"] = df_daily["tp_mm_sum"].fillna(0.0)

    sort_cols = ["lead_day"]
    if "number" in df_daily.columns:
        sort_cols = ["number"] + sort_cols
    return df_daily.sort_values(sort_cols).reset_index(drop=True)


# =========================
# IO helpers
# =========================
def parse_init_months(s: str) -> List[int]:
    s = s.strip()
    if "," in s:
        out = []
        for x in s.split(","):
            x = x.strip()
            if x:
                out.append(int(x))
        return sorted(set(out))
    if "-" in s:
        a, b = s.split("-", 1)
        a = int(a.strip())
        b = int(b.strip())
        if a > b:
            a, b = b, a
        return list(range(a, b + 1))
    return [int(s)]

def load_points() -> List[Tuple[float, float]]:
    if POINTS_FILE.exists():
        df = pd.read_csv(POINTS_FILE)
        lat_col = "lat" if "lat" in df.columns else ("latitude" if "latitude" in df.columns else None)
        lon_col = "lon" if "lon" in df.columns else ("longitude" if "longitude" in df.columns else None)
        if lat_col is None or lon_col is None:
            raise ValueError(f"{POINTS_FILE} must have columns lat/lon (or latitude/longitude)")
        pts = [(float(r[lat_col]), float(r[lon_col])) for _, r in df.iterrows()]
        if not pts:
            raise ValueError(f"{POINTS_FILE} is empty.")
        return pts
    return DEFAULT_POINTS

def point_tag(lat: float, lon: float) -> str:
    return f"lat{lat:.3f}_lon{lon:.3f}"

def build_paths(lat: float, lon: float, y: int, m: int) -> Tuple[Path, Path, Path]:
    out_dir = OUT_ROOT / f"{y:04d}" / f"{m:02d}"
    cache_dir = out_dir / "_cache_nc"
    cache_dir.mkdir(parents=True, exist_ok=True)

    pfx = point_tag(lat, lon)
    out_csv = out_dir / f"{pfx}_init{y:04d}-{m:02d}-01.csv"
    inst_nc = cache_dir / f"{pfx}_init{y:04d}-{m:02d}-01_inst.nc"
    tp_nc = cache_dir / f"{pfx}_init{y:04d}-{m:02d}-01_tp.nc"
    return out_csv, inst_nc, tp_nc

def save_daily_csv(df_daily: pd.DataFrame, out_csv: Path, *, init_date: date, end_date: date, keep_members: bool):
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df_daily["init_date"] = pd.Timestamp(init_date)
    df_daily["valid_date"] = (pd.Timestamp(init_date) + pd.to_timedelta(df_daily["lead_day"], unit="D")).dt.date

    expected_n = (end_date - init_date).days + 1
    df_daily = df_daily[(df_daily["lead_day"] >= 0) & (df_daily["lead_day"] < expected_n)].copy()

    if keep_members and ("number" in df_daily.columns):
        cols = [
            "latitude", "longitude", "number",
            "init_date", "valid_date", "lead_day",
            "t2m_C_mean", "t2m_C_max", "t2m_C_min",
            "rh_pct_mean", "wind_mps_mean",
            "tp_mm_sum", "ens_n", "n6",
        ]
        cols = [c for c in cols if c in df_daily.columns]
        out = df_daily[cols].copy()
    else:
        out = pd.DataFrame({
            "latitude": df_daily["latitude"],
            "longitude": df_daily["longitude"],
            "init_date": df_daily["init_date"],
            "valid_date": df_daily["valid_date"],
            "lead_day": df_daily["lead_day"],
            "ens_n": df_daily.get("ens_n", 1),
            "t2m_C": df_daily["t2m_C_mean"],
            "t2m_C_max": df_daily["t2m_C_max"],
            "t2m_C_min": df_daily["t2m_C_min"],
            "rh_pct": df_daily["rh_pct_mean"],
            "wind_mps": df_daily["wind_mps_mean"],
            "tp_mm_mean": df_daily["tp_mm_sum"],
        })

    out.to_csv(out_csv, index=False)
    log(f"[OK] saved: {out_csv} rows={len(out)}")


# =========================
# Work planning
# =========================
def plan_tasks(year_min: int, year_max: int, init_months: List[int]) -> List[Tuple[int, int]]:
    tasks = []
    for y in range(year_min, year_max + 1):
        for m in init_months:
            tasks.append((y, m))
    return tasks

def task_is_done(lat: float, lon: float, y: int, m: int) -> bool:
    out_csv, _, _ = build_paths(lat, lon, y, m)
    return out_csv.exists() and out_csv.stat().st_size > 0


# =========================
# Main
# =========================
def run_one_init(client: "cdsapi.Client", *, lat: float, lon: float, y: int, m: int):
    init_date = date(y, m, 1)
    end_date = horizon_end_date(y, m, HORIZON_AHEAD_MONTHS)

    out_csv, inst_nc, tp_nc = build_paths(lat, lon, y, m)
    if out_csv.exists() and out_csv.stat().st_size > 0:
        dlog(f"[skip] {out_csv}")
        return

    if should_stop_now():
        raise TimeoutError("Time budget almost exceeded; stop before starting a new init-month task.")

    area = [lat, lon, lat, lon]
    inst_leads = lead_hours_inst(init_date, end_date)
    tp_leads = lead_hours_tp(init_date, end_date)

    req_inst = build_request(y, m, inst_leads, INST_VARIABLES, area)
    req_tp = build_request(y, m, tp_leads, PRECIP_VARIABLE, area)

    log(f"[DL] {point_tag(lat,lon)} init={y}-{m:02d}-01 end={end_date.isoformat()} members={'ALL' if KEEP_BASELINE_MEMBERS else 'MEAN'} left={int(time_left_sec())}s")

    retrieve_with_retry(client, DATASET, req_inst, inst_nc)
    retrieve_with_retry(client, DATASET, req_tp, tp_nc)

    df_daily = to_daily_for_point(inst_nc, tp_nc, keep_members=KEEP_BASELINE_MEMBERS)

    # Optional debug: check 6-hour sampling counts
    if DEBUG and "n6" in df_daily.columns:
        chk = df_daily.groupby("lead_day")["n6"].agg(["min", "max"])
        weird = chk[(chk["min"] < 4) | (chk["max"] > 4)]
        if len(weird) > 0:
            log(f"[WARN] n6 not always 4 at init={y}-{m:02d}-01 (first 10):")
            log(str(weird.head(10)))

    save_daily_csv(df_daily, out_csv, init_date=init_date, end_date=end_date, keep_members=KEEP_BASELINE_MEMBERS)


def main():
    if BASELINE_YEAR_MIN > BASELINE_YEAR_MAX:
        raise ValueError("BASELINE_YEAR_MIN must be <= BASELINE_YEAR_MAX")

    init_months = parse_init_months(INIT_MONTHS)
    points = load_points()

    write_cdsapirc_from_env()
    client = cdsapi.Client()

    # Build a global task list; we will stop when time budget is nearly exceeded
    tasks = plan_tasks(BASELINE_YEAR_MIN, BASELINE_YEAR_MAX, init_months)

    # Simple progress tracking file (optional)
    prog_file = OUT_ROOT / "_progress.txt"
    prog_file.parent.mkdir(parents=True, exist_ok=True)

    total = len(tasks) * len(points)
    done0 = 0
    for lat, lon in points:
        for (y, m) in tasks:
            if task_is_done(lat, lon, y, m):
                done0 += 1

    log(f"[RUN] years={BASELINE_YEAR_MIN}..{BASELINE_YEAR_MAX} months={init_months} horizon_ahead={HORIZON_AHEAD_MONTHS} keep_members={KEEP_BASELINE_MEMBERS}")
    log(f"[RUN] time_budget={TIME_BUDGET_SEC}s grace={STOP_GRACE_SEC}s")
    log(f"[RUN] already_done={done0}/{total}  out={OUT_ROOT.resolve()}")

    done = done0
    try:
        for lat, lon in points:
            for (y, m) in tasks:
                if task_is_done(lat, lon, y, m):
                    continue

                if should_stop_now():
                    raise TimeoutError("Reached time budget.")

                run_one_init(client, lat=lat, lon=lon, y=y, m=m)
                done += 1

                # Update progress file occasionally
                if done % 5 == 0 or done == total:
                    prog_file.write_text(
                        f"done={done}/{total}\nleft_sec={int(time_left_sec())}\n",
                        encoding="utf-8"
                    )

    except TimeoutError as e:
        log(f"[TIMEBOX] stop: {e}")
        prog_file.write_text(
            f"done={done}/{total}\nleft_sec={int(time_left_sec())}\nstatus=timeboxed\n",
            encoding="utf-8"
        )
        # Exit 0 so Actions continues to commit/push partial results
        return

    prog_file.write_text(
        f"done={done}/{total}\nleft_sec={int(time_left_sec())}\nstatus=complete\n",
        encoding="utf-8"
    )
    log("[DONE] baseline complete")


if __name__ == "__main__":
    main()
