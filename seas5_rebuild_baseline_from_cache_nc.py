#!/usr/bin/env python3
"""One-off baseline rebuild from existing local cache NetCDF files.

Important safety note
- This path is only safe when the cache NetCDF itself is already on the native
  half-degree cell-center grid.
- If the cache coordinates are legacy integer-grid outputs, rebuilding from
  cache would preserve the wrong grid and should be refused.

Goal
- Rebuild baseline CSVs from cache only when the cache itself is native.
- Prefer monthly cache pair:
    _cache_nc/initYYYY-MM-01_inst.nc + initYYYY-MM-01_tp.nc
- Fallback to legacy cache pairs:
    _cache_nc/*_initYYYY-MM-01_inst.nc + *_tp.nc
"""

from __future__ import annotations

import argparse
import csv
import sys
import types
from datetime import date
from pathlib import Path
from typing import List, Tuple

import h5py
import pandas as pd

if 'cdsapi' not in sys.modules:
    sys.modules['cdsapi'] = types.SimpleNamespace(Client=object)

from seas5_build_monthly_baseline_timeboxed import (
    horizon_end_date,
    save_daily_csv,
    to_daily_for_month,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start-year", type=int, default=2000)
    p.add_argument("--end-year", type=int, default=2025)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def month_dir(y: int, m: int) -> Path:
    return Path("SEAS5") / "baseline" / f"{y:04d}" / f"{m:02d}"


def out_csv_path(y: int, m: int) -> Path:
    return month_dir(y, m) / f"init{y:04d}-{m:02d}-01.csv"


def csv_is_native_half_grid(csv_path: Path) -> bool:
    if (not csv_path.exists()) or csv_path.stat().st_size == 0:
        return False
    lats = set()
    lons = set()
    with csv_path.open("r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            lats.add(float(r["latitude"]))
            lons.add(float(r["longitude"]))
    if not lats or not lons:
        return False
    lat_frac = sorted({round(v - int(v), 6) for v in lats})
    lon_frac = sorted({round(v - int(v), 6) for v in lons})
    return lat_frac == [0.5] and lon_frac == [0.5]


def nc_is_native_half_grid(nc_path: Path) -> bool:
    if (not nc_path.exists()) or nc_path.stat().st_size == 0:
        return False
    with h5py.File(nc_path, "r") as f:
        if ("latitude" not in f) or ("longitude" not in f):
            return False
        lats = [float(v) for v in f["latitude"][...].tolist()]
        lons = [float(v) for v in f["longitude"][...].tolist()]
    if not lats or not lons:
        return False
    lat_frac = sorted({round(v - int(v), 6) for v in lats})
    lon_frac = sorted({round(v - int(v), 6) for v in lons})
    return lat_frac == [0.5] and lon_frac == [0.5]


def find_cache_pairs(y: int, m: int) -> List[Tuple[Path, Path]]:
    cdir = month_dir(y, m) / "_cache_nc"
    if not cdir.exists():
        return []

    monthly_inst = cdir / f"init{y:04d}-{m:02d}-01_inst.nc"
    monthly_tp = cdir / f"init{y:04d}-{m:02d}-01_tp.nc"
    if monthly_inst.exists() and monthly_tp.exists():
        return [(monthly_inst, monthly_tp)]

    out: List[Tuple[Path, Path]] = []
    for inst in sorted(cdir.glob(f"*_init{y:04d}-{m:02d}-01_inst.nc")):
        tp = Path(str(inst).replace("_inst.nc", "_tp.nc"))
        if tp.exists():
            out.append((inst, tp))
    return out


def merge_daily_frames(frames: List[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    if len(frames) == 1:
        return frames[0]

    all_df = pd.concat(frames, ignore_index=True)
    agg = all_df.groupby(["latitude", "longitude", "lead_day"], as_index=False).agg(
        t2m_C_mean=("t2m_C_mean", "mean"),
        t2m_C_max=("t2m_C_max", "mean"),
        t2m_C_min=("t2m_C_min", "mean"),
        rh_pct_mean=("rh_pct_mean", "mean"),
        wind_mps_mean=("wind_mps_mean", "mean"),
        tp_mm_sum=("tp_mm_sum", "mean"),
        ens_n=("ens_n", "max"),
        n6=("n6", "max"),
    )
    return agg.sort_values(["latitude", "longitude", "lead_day"]).reset_index(drop=True)


def rebuild_one(y: int, m: int, overwrite: bool) -> str:
    out_csv = out_csv_path(y, m)
    if out_csv.exists() and out_csv.stat().st_size > 0 and not overwrite:
        if csv_is_native_half_grid(out_csv):
            return "skip_native"

    pairs = find_cache_pairs(y, m)
    if not pairs:
        return "missing_cache"

    native_pairs = [
        (inst_nc, tp_nc)
        for inst_nc, tp_nc in pairs
        if nc_is_native_half_grid(inst_nc) and nc_is_native_half_grid(tp_nc)
    ]
    if not native_pairs:
        return "cache_non_native"

    frames: List[pd.DataFrame] = []
    for inst_nc, tp_nc in native_pairs:
        try:
            d = to_daily_for_month(inst_nc, tp_nc, keep_members=False)
        except Exception:
            continue
        if not d.empty:
            frames.append(d)

    if not frames:
        return "cache_unreadable"

    merged = merge_daily_frames(frames)
    if merged.empty:
        return "cache_empty"

    init_date = date(y, m, 1)
    end_date = horizon_end_date(y, m, 4)
    save_daily_csv(merged, out_csv, init_date=init_date, end_date=end_date, keep_members=False)
    return "rebuilt"


def main() -> int:
    args = parse_args()
    stats = {
        "rebuilt": 0,
        "skip_native": 0,
        "missing_cache": 0,
        "cache_non_native": 0,
        "cache_unreadable": 0,
        "cache_empty": 0,
    }

    for y in range(args.start_year, args.end_year + 1):
        for m in range(1, 13):
            st = rebuild_one(y, m, overwrite=args.overwrite)
            stats[st] = stats.get(st, 0) + 1
            print(f"[{st}] {y:04d}-{m:02d}", flush=True)

    print(f"[SUMMARY] {stats}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
