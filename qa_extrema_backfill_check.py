#!/usr/bin/env python3
"""Quick QA for baseline extrema backfill outputs using stdlib only."""

import csv
import glob
import math

TARGET = ["rh_pct_max", "rh_pct_min", "wind_mps_max", "wind_mps_min"]


def f(x):
    try:
        return float(x)
    except Exception:
        return math.nan


files = sorted(glob.glob("SEAS5/baseline/*/*/init????-??-01.csv"))
with_cols = 0
rows = bad_order = bad_range = neg_wind = null_ext = 0
for fp in files:
    with open(fp, encoding="utf-8") as fh:
        dr = csv.DictReader(fh)
        cols = dr.fieldnames or []
        if not all(c in cols for c in TARGET):
            continue
        with_cols += 1
        for r in dr:
            rows += 1
            rhmin, rhmax = f(r["rh_pct_min"]), f(r["rh_pct_max"])
            wmin, wmax = f(r["wind_mps_min"]), f(r["wind_mps_max"])
            rh, w = f(r.get("rh_pct", "")), f(r.get("wind_mps", ""))
            if any(math.isnan(v) for v in [rhmin, rhmax, wmin, wmax]):
                null_ext += 1
                continue
            if rhmin > rhmax or wmin > wmax:
                bad_order += 1
            if not (0 <= rhmin <= 100 and 0 <= rhmax <= 100):
                bad_range += 1
            if min(wmin, wmax) < 0:
                neg_wind += 1
            if not math.isnan(rh) and not (rhmin - 1e-6 <= rh <= rhmax + 1e-6):
                bad_order += 1
            if not math.isnan(w) and not (wmin - 1e-6 <= w <= wmax + 1e-6):
                bad_order += 1

print(f"files_total={len(files)}")
print(f"files_with_extrema_cols={with_cols}")
print(f"rows_checked={rows}")
print(f"null_extrema_rows={null_ext}")
print(f"bad_order_rows={bad_order}")
print(f"bad_rh_range_rows={bad_range}")
print(f"negative_wind_rows={neg_wind}")
