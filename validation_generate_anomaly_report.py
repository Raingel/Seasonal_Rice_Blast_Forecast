#!/usr/bin/env python3
"""Generate SEAS5 anomaly report using SEAS5 internal climatology only.

Method choices (fixed by design)
- Point matching: nearest available SEAS5 grid point to target lat/lon.
- Climatology source: historical SEAS5 baseline years 2000-2025.
- Validation target: SEAS5 latest forecast initializations.
- Climatology method: day-of-year ±7-day moving window.
- Variables:
  - Temperature / RH / Wind: difference anomaly metrics.
  - Precipitation: dual-track anomaly (difference + ratio to climatology).
- Output: one report per initialization month.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import date, datetime
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
REPORT_YEAR_DEFAULT = date.today().year
CLIM_YEAR_START = int(os.getenv("CLIM_YEAR_START", "2000"))
CLIM_YEAR_END = int(os.getenv("CLIM_YEAR_END", "2025"))
SEAS5_LATEST_DIR = Path(os.getenv("SEAS5_LATEST_DIR", "SEAS5/latest"))
SEAS5_BASELINE_DIR = Path(os.getenv("SEAS5_BASELINE_DIR", "SEAS5/baseline"))
OUT_ROOT = Path(os.getenv("VALIDATION_OUT_ROOT", "validation/anomaly_reports"))
INIT_MONTHS_RAW = os.getenv("INIT_MONTHS", "").strip()
DEBUG = os.getenv("DEBUG", "0").strip() in ("1", "true", "True", "yes", "YES")
EPS = 1e-6
EXPECTED_FIG_NAMES = [
    "t2m_C_anomaly.png",
    "t2m_C_value_vs_clim.png",
    "rh_pct_anomaly.png",
    "rh_pct_value_vs_clim.png",
    "wind_mps_anomaly.png",
    "wind_mps_value_vs_clim.png",
    "tp_mm_mean_anomaly.png",
    "tp_mm_mean_value_vs_clim.png",
]


@dataclass
class PointInfo:
    lat: float
    lon: float
    dist_deg: float


def log(msg: str) -> None:
    print(msg, flush=True)


def dlog(msg: str) -> None:
    if DEBUG:
        log(msg)


def list_latest_files(year: int) -> List[Path]:
    if not SEAS5_LATEST_DIR.exists():
        return []
    return sorted(SEAS5_LATEST_DIR.glob(f"init{year:04d}-??-01.csv"))


def list_latest_years() -> List[int]:
    if not SEAS5_LATEST_DIR.exists():
        return []
    years = set()
    for fp in SEAS5_LATEST_DIR.glob("init????-??-01.csv"):
        name = fp.name
        try:
            y = int(name[4:8])
            years.add(y)
        except Exception:
            continue
    return sorted(years)


def parse_init_date_from_filename(fp: Path) -> date | None:
    try:
        return datetime.strptime(fp.stem.replace("init", ""), "%Y-%m-%d").date()
    except ValueError:
        return None


def resolve_target_inits() -> List[date]:
    available = sorted(
        [d for d in (parse_init_date_from_filename(fp) for fp in SEAS5_LATEST_DIR.glob("init????-??-01.csv")) if d is not None]
    )
    if not available:
        return []

    if REPORT_YEAR_RAW:
        report_year = int(REPORT_YEAR_RAW)
        selected = [d for d in available if d.year == report_year]
        if INIT_MONTHS_RAW:
            mm = {int(x) for x in INIT_MONTHS_RAW.split(",") if x.strip()}
            selected = [d for d in selected if d.month in mm]
        return selected

    # Auto mode: process latest init only (for scheduled runs).
    return [available[-1]]


def list_baseline_files(y0: int, y1: int) -> List[Path]:
    out: List[Path] = []
    for y in range(y0, y1 + 1):
        out.extend(sorted((SEAS5_BASELINE_DIR / f"{y:04d}").glob("??/init*.csv")))
    return out


def collect_points(files: Iterable[Path]) -> List[Tuple[float, float]]:
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
    return sorted(pts)


def find_nearest_point(points: Iterable[Tuple[float, float]], target_lat: float, target_lon: float) -> PointInfo:
    pts = list(points)
    if not pts:
        raise RuntimeError("No SEAS5 points found to match target location")

    best = None
    for lat, lon in pts:
        dist = math.hypot(lat - target_lat, lon - target_lon)
        if best is None or dist < best[2]:
            best = (lat, lon, dist)
    assert best is not None
    return PointInfo(*best)


def find_nearest_common_point(
    pts_a: Iterable[Tuple[float, float]],
    pts_b: Iterable[Tuple[float, float]],
    target_lat: float,
    target_lon: float,
) -> PointInfo | None:
    set_a = set(pts_a)
    set_b = set(pts_b)
    common = sorted(set_a & set_b)
    if not common:
        return None
    return find_nearest_point(common, target_lat, target_lon)


def load_seas5_latest_inits(inits: List[date], p: PointInfo) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    init_set = {d.isoformat() for d in inits}
    for fp in SEAS5_LATEST_DIR.glob("init????-??-01.csv"):
        try:
            d = pd.read_csv(fp, usecols=["latitude", "longitude", "init_date", "valid_date", "lead_day", "t2m_C", "rh_pct", "wind_mps", "tp_mm_mean"])
        except Exception:
            continue
        if d.empty or str(d.iloc[0].get("init_date", "")) not in init_set:
            continue
        d = d[(np.isclose(d["latitude"], p.lat, atol=1e-4)) & (np.isclose(d["longitude"], p.lon, atol=1e-4))].copy()
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
        d = d[(np.isclose(d["latitude"], p.lat, atol=1e-4)) & (np.isclose(d["longitude"], p.lon, atol=1e-4))].copy()
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
    a = pd.to_numeric(df[f"seas5_{var}_anom"], errors="coerce").dropna()
    if a.empty:
        return {"n": 0}
    out = {
        "n": int(len(a)),
        "anom_mean": float(a.mean()),
        "anom_abs_mean": float(a.abs().mean()),
        "anom_std": float(a.std(ddof=0)),
        "anom_pos_frac": float((a > 0).mean()),
    }
    if var == "tp_mm_mean":
        ra = pd.to_numeric(df["seas5_tp_mm_mean_ratio_anom"], errors="coerce").dropna()
        out["ratio_anom_mean"] = float(ra.mean()) if not ra.empty else np.nan
        out["ratio_anom_abs_mean"] = float(ra.abs().mean()) if not ra.empty else np.nan
    return out


def make_line_plot(df_day: pd.DataFrame, var: str, out_png: Path, title: str) -> None:
    plt.figure(figsize=(11, 4.8))
    x = pd.to_datetime(df_day["valid_date"])
    plt.plot(x, df_day[f"seas5_{var}_anom"], label="SEAS5 anomaly", lw=1.7)
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
    plt.plot(x, df_day[f"{var}_clim"], label="SEAS5 climatology", lw=1.0, ls=":")
    plt.title(title)
    plt.xlabel("Valid date")
    plt.ylabel("Value")
    plt.legend(loc="best", ncol=2)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=140)
    plt.close()


def make_placeholder_figure(out_png: Path, title: str, message: str) -> None:
    plt.figure(figsize=(11, 4.8))
    plt.title(title)
    plt.axis("off")
    plt.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=140)
    plt.close()


def ensure_placeholder_figures(fig_dir: Path, reason: str) -> None:
    for name in EXPECTED_FIG_NAMES:
        fp = fig_dir / name
        if not fp.exists() or fp.stat().st_size == 0:
            make_placeholder_figure(
                fp,
                title=name.replace("_", " ").replace(".png", ""),
                message=f"Figure unavailable: {reason}",
            )


def to_html_table(metrics: Dict[str, Dict[str, float]]) -> str:
    rows = []
    for var, m in metrics.items():
        rows.append(
            "<tr>"
            f"<td>{var}</td><td>{m.get('n', 0)}</td>"
            f"<td>{m.get('anom_mean', float('nan')):.3f}</td>"
            f"<td>{m.get('anom_abs_mean', float('nan')):.3f}</td>"
            f"<td>{m.get('anom_std', float('nan')):.3f}</td>"
            f"<td>{m.get('anom_pos_frac', float('nan')):.3f}</td>"
            f"<td>{m.get('ratio_anom_mean', float('nan')):.3f}</td>"
            f"<td>{m.get('ratio_anom_abs_mean', float('nan')):.3f}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def write_report(
    init_date: date,
    latest_point: PointInfo,
    baseline_point: PointInfo,
    metrics: Dict[str, Dict[str, float]],
    note: str = "",
) -> None:
    out_dir = OUT_ROOT / init_date.isoformat()
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    ensure_placeholder_figures(fig_dir, reason=note or "not generated")

    rows = to_html_table(metrics)
    html = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>SEAS5 anomaly report {init_date}</title>
<style>body{{font-family:Arial,sans-serif;max-width:1200px;margin:20px auto;padding:0 12px}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:6px 10px}}</style>
</head><body>
<h1>SEAS5 anomaly report (init {init_date})</h1>
<p>Target point requested: ({TARGET_LAT:.3f}, {TARGET_LON:.3f}).</p>
<p>Latest forecast grid: ({latest_point.lat:.3f}, {latest_point.lon:.3f}), distance={latest_point.dist_deg:.3f} deg.</p>
<p>Baseline climatology grid: ({baseline_point.lat:.3f}, {baseline_point.lon:.3f}), distance={baseline_point.dist_deg:.3f} deg.</p>
<p>Method: nearest-grid matching; SEAS5 climatology {CLIM_YEAR_START}-{CLIM_YEAR_END} with day-of-year ±7-day window; precipitation dual-track anomaly (difference + ratio).</p>
<p>{note}</p>
<h2>Metrics</h2>
<table><tr><th>Variable</th><th>N</th><th>Anom mean</th><th>Anom abs mean</th><th>Anom std</th><th>Positive frac</th><th>Precip ratio mean</th><th>Precip ratio abs mean</th></tr>
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
    target_inits = resolve_target_inits()
    if not target_inits:
        raise RuntimeError("No SEAS5 latest files found for requested initialization month(s)")

    latest_files = [SEAS5_LATEST_DIR / f"init{d.isoformat()}.csv" for d in target_inits]
    baseline_files = list_baseline_files(CLIM_YEAR_START, CLIM_YEAR_END)
    if not baseline_files:
        raise RuntimeError("No SEAS5 baseline files found for climatology")

    latest_points = collect_points(latest_files)
    baseline_points = collect_points(baseline_files)
    dlog(f"[DEBUG] latest_files={len(latest_files)} latest_points={len(latest_points)}")
    dlog(f"[DEBUG] baseline_files={len(baseline_files)} baseline_points={len(baseline_points)}")

    common_point = find_nearest_common_point(latest_points, baseline_points, TARGET_LAT, TARGET_LON)
    if common_point is not None:
        latest_point = common_point
        baseline_point = common_point
        log(
            f"[POINT] requested=({TARGET_LAT},{TARGET_LON}) matched_common=({common_point.lat},{common_point.lon}) "
            f"dist={common_point.dist_deg:.3f}"
        )
    else:
        latest_point = find_nearest_point(latest_points, TARGET_LAT, TARGET_LON)
        baseline_point = find_nearest_point(baseline_points, TARGET_LAT, TARGET_LON)
        log(
            f"[POINT] requested=({TARGET_LAT},{TARGET_LON}) no-common-point; "
            f"latest=({latest_point.lat},{latest_point.lon}) dist={latest_point.dist_deg:.3f}; "
            f"baseline=({baseline_point.lat},{baseline_point.lon}) dist={baseline_point.dist_deg:.3f}"
        )

    seas5_clim_src = load_seas5_baseline_clim_source(baseline_point)
    if seas5_clim_src.empty:
        # Extra diagnostics for fast troubleshooting in Actions logs.
        sample = baseline_points[:10]
        raise RuntimeError(
            "No baseline climatology source rows after point filter; "
            f"point=({baseline_point.lat},{baseline_point.lon}) baseline_points={len(baseline_points)} sample={sample}"
        )

    value_cols = ["t2m_C", "rh_pct", "wind_mps", "tp_mm_mean"]
    seas5_clim = build_climatology(seas5_clim_src, value_cols=value_cols, window=7)
    for init_date in target_inits:
        out_dir = OUT_ROOT / init_date.isoformat()
        out_dir.mkdir(parents=True, exist_ok=True)
        seas5_latest = load_seas5_latest_inits([init_date], latest_point)
        if seas5_latest.empty:
            write_report(
                init_date,
                latest_point,
                baseline_point,
                {v: {"n": 0} for v in value_cols},
                note="No SEAS5 latest data for selected init after point filtering.",
            )
            continue

        merged = attach_anomaly(seas5_latest, seas5_clim, value_cols, prefix="seas5")
        tab_dir = out_dir / "tables"
        tab_dir.mkdir(parents=True, exist_ok=True)
        merged.to_csv(tab_dir / "daily_pairs.csv", index=False)

        daily = merged.groupby("valid_date", as_index=False).agg(
            seas5_t2m_C_anom=("seas5_t2m_C_anom", "mean"),
            seas5_rh_pct_anom=("seas5_rh_pct_anom", "mean"),
            seas5_wind_mps_anom=("seas5_wind_mps_anom", "mean"),
            seas5_tp_mm_mean_anom=("seas5_tp_mm_mean_anom", "mean"),
            seas5_t2m_C=("t2m_C", "mean"),
            seas5_rh_pct=("rh_pct", "mean"),
            seas5_wind_mps=("wind_mps", "mean"),
            seas5_tp_mm_mean=("tp_mm_mean", "mean"),
            t2m_C_clim=("t2m_C_clim", "mean"),
            rh_pct_clim=("rh_pct_clim", "mean"),
            wind_mps_clim=("wind_mps_clim", "mean"),
            tp_mm_mean_clim=("tp_mm_mean_clim", "mean"),
        )

        fig_dir = out_dir / "figures"
        make_line_plot(daily, "t2m_C", fig_dir / "t2m_C_anomaly.png", f"{init_date} Temperature anomaly (daily)")
        make_value_clim_plot(daily, "t2m_C", fig_dir / "t2m_C_value_vs_clim.png", f"{init_date} Temperature value vs climatology")
        make_line_plot(daily, "rh_pct", fig_dir / "rh_pct_anomaly.png", f"{init_date} RH anomaly (daily)")
        make_value_clim_plot(daily, "rh_pct", fig_dir / "rh_pct_value_vs_clim.png", f"{init_date} RH value vs climatology")
        make_line_plot(daily, "wind_mps", fig_dir / "wind_mps_anomaly.png", f"{init_date} Wind anomaly (daily)")
        make_value_clim_plot(daily, "wind_mps", fig_dir / "wind_mps_value_vs_clim.png", f"{init_date} Wind value vs climatology")
        make_line_plot(daily, "tp_mm_mean", fig_dir / "tp_mm_mean_anomaly.png", f"{init_date} Precip anomaly (daily)")
        make_value_clim_plot(daily, "tp_mm_mean", fig_dir / "tp_mm_mean_value_vs_clim.png", f"{init_date} Precip value vs climatology")

        metrics = {v: compute_metrics(merged, v) for v in value_cols}
        write_report(init_date, latest_point, baseline_point, metrics, note=f"Rows in selected init: {len(merged)}")
        log(f"[OK] report written: {(out_dir / 'index.html').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
