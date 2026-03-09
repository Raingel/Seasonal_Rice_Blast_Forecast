#!/usr/bin/env python3
"""Generate SEAS5 anomaly validation report against Open-Meteo ERA5.

Method choices (fixed by design)
- Point matching: nearest available SEAS5 grid point to target lat/lon.
- Climatology: day-of-year ±7-day moving window, baseline years 2000-2020.
- Variables:
  - Temperature / RH / Wind: difference anomaly and z-score proxy metrics.
  - Precipitation: dual-track anomaly (difference + ratio to climatology).
- Output: overwrite one report per year.
"""

from __future__ import annotations

import json
import math
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TARGET_LAT = float(os.getenv("TARGET_LAT", "23.0"))
TARGET_LON = float(os.getenv("TARGET_LON", "120.0"))
TZ_NAME = os.getenv("TZ_NAME", "Asia/Taipei")
REPORT_YEAR_RAW = os.getenv("REPORT_YEAR", "").strip()
REPORT_YEAR = int(REPORT_YEAR_RAW) if REPORT_YEAR_RAW else date.today().year
CLIM_YEAR_START = int(os.getenv("CLIM_YEAR_START", "2000"))
CLIM_YEAR_END = int(os.getenv("CLIM_YEAR_END", "2020"))
SEAS5_LATEST_DIR = Path(os.getenv("SEAS5_LATEST_DIR", "SEAS5/latest"))
SEAS5_BASELINE_DIR = Path(os.getenv("SEAS5_BASELINE_DIR", "SEAS5/baseline"))
OUT_ROOT = Path(os.getenv("VALIDATION_OUT_ROOT", "validation/anomaly_reports"))

ERA_DAILY_VARS = [
    "temperature_2m_mean",
    "wind_speed_10m_mean",
    "relative_humidity_2m_mean",
    "precipitation_sum",
]

# Open-Meteo wind is km/h; SEAS5 is m/s.
KMH_TO_MPS = 1.0 / 3.6
EPS = 1e-6


@dataclass
class PointInfo:
    lat: float
    lon: float
    dist_deg: float


def log(msg: str) -> None:
    print(msg, flush=True)


def open_meteo_archive(lat: float, lon: float, start_d: str, end_d: str) -> pd.DataFrame:
    params = {
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "start_date": start_d,
        "end_date": end_d,
        "daily": ",".join(ERA_DAILY_VARS),
        "models": "era5_seamless",
        "timezone": TZ_NAME,
    }
    url = "https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=120) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    daily = payload.get("daily", {})
    if not daily or "time" not in daily:
        raise RuntimeError(f"open-meteo missing daily payload: {url}")

    df = pd.DataFrame({
        "valid_date": pd.to_datetime(daily["time"]).date,
        "t2m_C": pd.to_numeric(daily.get("temperature_2m_mean", []), errors="coerce"),
        "wind_mps": pd.to_numeric(daily.get("wind_speed_10m_mean", []), errors="coerce") * KMH_TO_MPS,
        "rh_pct": pd.to_numeric(daily.get("relative_humidity_2m_mean", []), errors="coerce"),
        "tp_mm_mean": pd.to_numeric(daily.get("precipitation_sum", []), errors="coerce"),
    })
    return df


def list_latest_files(year: int) -> List[Path]:
    if not SEAS5_LATEST_DIR.exists():
        return []
    return sorted(SEAS5_LATEST_DIR.glob(f"init{year:04d}-??-01.csv"))


def list_baseline_files(y0: int, y1: int) -> List[Path]:
    out: List[Path] = []
    for y in range(y0, y1 + 1):
        out.extend(sorted((SEAS5_BASELINE_DIR / f"{y:04d}").glob("??/init*.csv")))
    return out


def find_nearest_point(files: Iterable[Path], target_lat: float, target_lon: float) -> PointInfo:
    pts = set()
    for fp in files:
        try:
            d = pd.read_csv(fp, usecols=["latitude", "longitude"])
        except Exception:
            continue
        d = d.drop_duplicates()
        for _, r in d.iterrows():
            pts.add((float(r["latitude"]), float(r["longitude"])))
        if len(pts) >= 20:
            break
    if not pts:
        raise RuntimeError("No SEAS5 points found to match target location")

    best = None
    for lat, lon in pts:
        dist = math.hypot(lat - target_lat, lon - target_lon)
        if best is None or dist < best[2]:
            best = (lat, lon, dist)
    assert best is not None
    return PointInfo(*best)


def load_seas5_latest_year(year: int, p: PointInfo) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for fp in list_latest_files(year):
        try:
            d = pd.read_csv(fp, usecols=["latitude", "longitude", "init_date", "valid_date", "lead_day", "t2m_C", "rh_pct", "wind_mps", "tp_mm_mean"])
        except Exception:
            continue
        d = d[(np.isclose(d["latitude"], p.lat)) & (np.isclose(d["longitude"], p.lon))].copy()
        if d.empty:
            continue
        d["init_date"] = pd.to_datetime(d["init_date"]).dt.date
        d["valid_date"] = pd.to_datetime(d["valid_date"]).dt.date
        parts.append(d)
    if not parts:
        return pd.DataFrame(columns=["init_date", "valid_date", "lead_day", "t2m_C", "rh_pct", "wind_mps", "tp_mm_mean"])
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["init_date", "valid_date", "lead_day"]).reset_index(drop=True)


def load_seas5_baseline_clim_source(p: PointInfo) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for fp in list_baseline_files(CLIM_YEAR_START, CLIM_YEAR_END):
        try:
            d = pd.read_csv(fp, usecols=["latitude", "longitude", "valid_date", "t2m_C", "rh_pct", "wind_mps", "tp_mm_mean"])
        except Exception:
            continue
        d = d[(np.isclose(d["latitude"], p.lat)) & (np.isclose(d["longitude"], p.lon))].copy()
        if d.empty:
            continue
        d["valid_date"] = pd.to_datetime(d["valid_date"]).dt.date
        parts.append(d)
    if not parts:
        return pd.DataFrame(columns=["valid_date", "t2m_C", "rh_pct", "wind_mps", "tp_mm_mean"])

    out = pd.concat(parts, ignore_index=True)
    # Avoid overweighting duplicated valid_date from overlapping init windows.
    out = out.groupby("valid_date", as_index=False).agg(
        t2m_C=("t2m_C", "mean"),
        rh_pct=("rh_pct", "mean"),
        wind_mps=("wind_mps", "mean"),
        tp_mm_mean=("tp_mm_mean", "mean"),
    )
    return out.sort_values("valid_date").reset_index(drop=True)


def doy_366(d: pd.Series) -> pd.Series:
    return pd.to_datetime(d).dt.dayofyear.astype(int)


def circular_dist(a: np.ndarray, b: int, period: int = 366) -> np.ndarray:
    delta = np.abs(a - b)
    return np.minimum(delta, period - delta)


def build_climatology(df: pd.DataFrame, value_cols: List[str], window: int = 7) -> pd.DataFrame:
    base = df.copy()
    base["doy"] = doy_366(base["valid_date"])
    by_doy = []
    all_doy = base["doy"].to_numpy(dtype=int)
    for d in range(1, 367):
        mask = circular_dist(all_doy, d, period=366) <= window
        row = {"doy": d}
        for c in value_cols:
            vals = pd.to_numeric(base.loc[mask, c], errors="coerce").to_numpy(dtype=float)
            row[f"{c}_clim"] = float(np.nanmean(vals)) if vals.size else np.nan
        by_doy.append(row)
    return pd.DataFrame(by_doy)


def attach_anomaly(df: pd.DataFrame, clim: pd.DataFrame, value_cols: List[str], prefix: str) -> pd.DataFrame:
    out = df.copy()
    out["doy"] = doy_366(out["valid_date"])
    out = out.merge(clim, on="doy", how="left")
    for c in value_cols:
        out[f"{prefix}_{c}_anom"] = out[c] - out[f"{c}_clim"]
        if c == "tp_mm_mean":
            out[f"{prefix}_{c}_ratio_anom"] = (out[c] / (out[f"{c}_clim"] + EPS)) - 1.0
    return out


def compute_metrics(df: pd.DataFrame, var: str) -> Dict[str, float]:
    a = pd.to_numeric(df[f"seas5_{var}_anom"], errors="coerce")
    b = pd.to_numeric(df[f"era5_{var}_anom"], errors="coerce")
    m = pd.DataFrame({"a": a, "b": b}).dropna()
    if m.empty:
        return {"n": 0}
    err = m["a"] - m["b"]
    out = {
        "n": int(len(m)),
        "bias": float(err.mean()),
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "corr": float(m["a"].corr(m["b"])),
        "sign_hit": float((np.sign(m["a"]) == np.sign(m["b"]).astype(int)).mean()),
    }
    if var == "tp_mm_mean":
        ra = pd.to_numeric(df["seas5_tp_mm_mean_ratio_anom"], errors="coerce")
        rb = pd.to_numeric(df["era5_tp_mm_mean_ratio_anom"], errors="coerce")
        mr = pd.DataFrame({"a": ra, "b": rb}).dropna()
        out["ratio_mae"] = float((mr["a"] - mr["b"]).abs().mean()) if not mr.empty else np.nan
    return out


def make_line_plot(df_day: pd.DataFrame, var: str, out_png: Path, title: str) -> None:
    plt.figure(figsize=(11, 4.8))
    x = pd.to_datetime(df_day["valid_date"])
    plt.plot(x, df_day[f"seas5_{var}_anom"], label="SEAS5 anomaly", lw=1.7)
    plt.plot(x, df_day[f"era5_{var}_anom"], label="ERA5 anomaly", lw=1.7)
    plt.plot(x, df_day[f"seas5_{var}_anom"] - df_day[f"era5_{var}_anom"], label="Anomaly error", lw=1.2, ls="--")
    plt.axhline(0, color="gray", lw=0.8)
    plt.title(title)
    plt.xlabel("Valid date")
    plt.ylabel("Anomaly")
    plt.legend(loc="best")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=140)
    plt.close()


def make_value_clim_plot(df_day: pd.DataFrame, var: str, out_png: Path, title: str) -> None:
    plt.figure(figsize=(11, 4.8))
    x = pd.to_datetime(df_day["valid_date"])
    plt.plot(x, df_day[f"seas5_{var}"], label="SEAS5 value", lw=1.5)
    plt.plot(x, df_day[f"era5_{var}"], label="ERA5 value", lw=1.5)
    plt.plot(x, df_day[f"{var}_clim"], label="SEAS5 climatology", lw=1.0, ls=":")
    plt.plot(x, df_day[f"era5_{var}_clim"], label="ERA5 climatology", lw=1.0, ls=':')
    plt.title(title)
    plt.xlabel("Valid date")
    plt.ylabel("Value")
    plt.legend(loc="best", ncol=2)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=140)
    plt.close()


def to_html_table(metrics: Dict[str, Dict[str, float]]) -> str:
    rows = []
    for var, m in metrics.items():
        rows.append(
            "<tr>"
            f"<td>{var}</td><td>{m.get('n', 0)}</td>"
            f"<td>{m.get('bias', float('nan')):.3f}</td>"
            f"<td>{m.get('mae', float('nan')):.3f}</td>"
            f"<td>{m.get('rmse', float('nan')):.3f}</td>"
            f"<td>{m.get('corr', float('nan')):.3f}</td>"
            f"<td>{m.get('sign_hit', float('nan')):.3f}</td>"
            f"<td>{m.get('ratio_mae', float('nan')):.3f}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def write_report(year: int, point: PointInfo, metrics: Dict[str, Dict[str, float]], note: str = "") -> None:
    out_dir = OUT_ROOT / f"{year:04d}"
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    rows = to_html_table(metrics)
    html = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>SEAS5 anomaly validation {year}</title>
<style>body{{font-family:Arial,sans-serif;max-width:1200px;margin:20px auto;padding:0 12px}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:6px 10px}}</style>
</head><body>
<h1>SEAS5 anomaly validation report ({year})</h1>
<p>Target point: requested ({TARGET_LAT:.3f}, {TARGET_LON:.3f}), matched SEAS5 grid ({point.lat:.3f}, {point.lon:.3f}), distance={point.dist_deg:.3f} deg.</p>
<p>Method: nearest-grid matching; climatology {CLIM_YEAR_START}-{CLIM_YEAR_END} with day-of-year ±7-day window; precipitation dual-track anomaly (difference + ratio).</p>
<p>Timezone alignment: {TZ_NAME}.</p>
<p>{note}</p>
<h2>Metrics</h2>
<table><tr><th>Variable</th><th>N</th><th>Bias</th><th>MAE</th><th>RMSE</th><th>Corr</th><th>Sign hit</th><th>Precip ratio MAE</th></tr>
{rows}
</table>
<h2>Figures</h2>
<ul>
<li><img src='figures/t2m_C_anomaly.png' width='1000'></li>
<li><img src='figures/t2m_C_value_vs_clim.png' width='1000'></li>
<li><img src='figures/rh_pct_anomaly.png' width='1000'></li>
<li><img src='figures/rh_pct_value_vs_clim.png' width='1000'></li>
<li><img src='figures/wind_mps_anomaly.png' width='1000'></li>
<li><img src='figures/wind_mps_value_vs_clim.png' width='1000'></li>
<li><img src='figures/tp_mm_mean_anomaly.png' width='1000'></li>
<li><img src='figures/tp_mm_mean_value_vs_clim.png' width='1000'></li>
</ul>
</body></html>"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> int:
    out_dir = OUT_ROOT / f"{REPORT_YEAR:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    latest_files = list_latest_files(REPORT_YEAR)
    baseline_files = list_baseline_files(CLIM_YEAR_START, CLIM_YEAR_END)
    if not baseline_files:
        raise RuntimeError("No SEAS5 baseline files found for climatology")

    match_ref_files = latest_files if latest_files else baseline_files
    point = find_nearest_point(match_ref_files, TARGET_LAT, TARGET_LON)
    log(f"[POINT] requested=({TARGET_LAT},{TARGET_LON}) matched=({point.lat},{point.lon})")

    seas5_latest = load_seas5_latest_year(REPORT_YEAR, point)
    if seas5_latest.empty:
        write_report(REPORT_YEAR, point, {v: {"n": 0} for v in ["t2m_C", "rh_pct", "wind_mps", "tp_mm_mean"]}, note="No SEAS5 latest data for current year yet.")
        log("[INFO] no latest data yet; wrote placeholder report")
        return 0

    seas5_clim_src = load_seas5_baseline_clim_source(point)
    if seas5_clim_src.empty:
        raise RuntimeError("No baseline climatology source rows after point filter")

    era_clim_src = open_meteo_archive(TARGET_LAT, TARGET_LON, f"{CLIM_YEAR_START}-01-01", f"{CLIM_YEAR_END}-12-31")
    era_y_src = open_meteo_archive(TARGET_LAT, TARGET_LON, f"{REPORT_YEAR}-01-01", date.today().isoformat())

    value_cols = ["t2m_C", "rh_pct", "wind_mps", "tp_mm_mean"]
    seas5_clim = build_climatology(seas5_clim_src, value_cols=value_cols, window=7)
    era_clim = build_climatology(era_clim_src, value_cols=value_cols, window=7)

    s = attach_anomaly(seas5_latest, seas5_clim, value_cols, prefix="seas5")
    e = attach_anomaly(era_y_src, era_clim, value_cols, prefix="era5")

    # Merge by valid_date; ground truth availability controls final rows.
    merged = s.merge(
        e[["valid_date", *value_cols, *[f"{c}_clim" for c in value_cols], *[f"era5_{c}_anom" for c in value_cols], "era5_tp_mm_mean_ratio_anom"]],
        on="valid_date",
        how="inner",
        suffixes=("", "_era"),
    )
    if merged.empty:
        write_report(REPORT_YEAR, point, {v: {"n": 0} for v in value_cols}, note="No overlap yet between SEAS5 valid dates and ERA5 ground truth dates.")
        log("[INFO] no overlapping dates; wrote placeholder report")
        return 0

    # Save tables
    tab_dir = out_dir / "tables"
    tab_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(tab_dir / "daily_pairs.csv", index=False)

    # Daily aggregate for line charts (x-axis day)
    daily = merged.groupby("valid_date", as_index=False).agg(
        seas5_t2m_C_anom=("seas5_t2m_C_anom", "mean"),
        era5_t2m_C_anom=("era5_t2m_C_anom", "mean"),
        seas5_rh_pct_anom=("seas5_rh_pct_anom", "mean"),
        era5_rh_pct_anom=("era5_rh_pct_anom", "mean"),
        seas5_wind_mps_anom=("seas5_wind_mps_anom", "mean"),
        era5_wind_mps_anom=("era5_wind_mps_anom", "mean"),
        seas5_tp_mm_mean_anom=("seas5_tp_mm_mean_anom", "mean"),
        era5_tp_mm_mean_anom=("era5_tp_mm_mean_anom", "mean"),
        seas5_t2m_C=("t2m_C", "mean"),
        era5_t2m_C=("t2m_C_era", "mean"),
        seas5_rh_pct=("rh_pct", "mean"),
        era5_rh_pct=("rh_pct_era", "mean"),
        seas5_wind_mps=("wind_mps", "mean"),
        era5_wind_mps=("wind_mps_era", "mean"),
        seas5_tp_mm_mean=("tp_mm_mean", "mean"),
        era5_tp_mm_mean=("tp_mm_mean_era", "mean"),
        t2m_C_clim=("t2m_C_clim", "mean"),
        rh_pct_clim=("rh_pct_clim", "mean"),
        wind_mps_clim=("wind_mps_clim", "mean"),
        tp_mm_mean_clim=("tp_mm_mean_clim", "mean"),
        era5_t2m_C_clim=("t2m_C_clim_era", "mean"),
        era5_rh_pct_clim=("rh_pct_clim_era", "mean"),
        era5_wind_mps_clim=("wind_mps_clim_era", "mean"),
        era5_tp_mm_mean_clim=("tp_mm_mean_clim_era", "mean"),
    )

    fig_dir = out_dir / "figures"
    make_line_plot(daily, "t2m_C", fig_dir / "t2m_C_anomaly.png", f"{REPORT_YEAR} Temperature anomaly (daily)")
    make_value_clim_plot(daily, "t2m_C", fig_dir / "t2m_C_value_vs_clim.png", f"{REPORT_YEAR} Temperature value vs climatology")
    make_line_plot(daily, "rh_pct", fig_dir / "rh_pct_anomaly.png", f"{REPORT_YEAR} RH anomaly (daily)")
    make_value_clim_plot(daily, "rh_pct", fig_dir / "rh_pct_value_vs_clim.png", f"{REPORT_YEAR} RH value vs climatology")
    make_line_plot(daily, "wind_mps", fig_dir / "wind_mps_anomaly.png", f"{REPORT_YEAR} Wind anomaly (daily)")
    make_value_clim_plot(daily, "wind_mps", fig_dir / "wind_mps_value_vs_clim.png", f"{REPORT_YEAR} Wind value vs climatology")
    make_line_plot(daily, "tp_mm_mean", fig_dir / "tp_mm_mean_anomaly.png", f"{REPORT_YEAR} Precip anomaly (daily)")
    make_value_clim_plot(daily, "tp_mm_mean", fig_dir / "tp_mm_mean_value_vs_clim.png", f"{REPORT_YEAR} Precip value vs climatology")

    metrics = {v: compute_metrics(merged, v) for v in value_cols}
    write_report(REPORT_YEAR, point, metrics, note=f"Rows with SEAS5-ERA5 overlap: {len(merged)}")

    log(f"[OK] report written: {(out_dir / 'index.html').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
