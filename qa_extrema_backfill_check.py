#!/usr/bin/env python3
"""QA checks for baseline extrema backfill outputs (stdlib only).

Checks
- Column coverage across monthly files.
- Value sanity: RH range, wind non-negative, min<=mean<=max.
- Continuity: per-point lead_day sequence should be contiguous from min..max.
- Cross-point consistency: neighbor-point (1-degree apart) t2m correlation by lead_day.

Exit code
- Returns non-zero if any threshold is exceeded.
"""

from __future__ import annotations

import csv
import glob
import math
import os
from collections import defaultdict
from itertools import combinations

TARGET = ["rh_pct_max", "rh_pct_min", "wind_mps_max", "wind_mps_min"]
REQ_CONT = ["latitude", "longitude", "lead_day"]

MAX_MISSING_COL_FILES = int(os.getenv("MAX_MISSING_COL_FILES", "0"))
MAX_NULL_EXTREMA_ROWS = int(os.getenv("MAX_NULL_EXTREMA_ROWS", "0"))
MAX_BAD_ORDER_ROWS = int(os.getenv("MAX_BAD_ORDER_ROWS", "0"))
MAX_BAD_RH_RANGE_ROWS = int(os.getenv("MAX_BAD_RH_RANGE_ROWS", "0"))
MAX_NEG_WIND_ROWS = int(os.getenv("MAX_NEG_WIND_ROWS", "0"))
MAX_CONTINUITY_GAP_GROUPS = int(os.getenv("MAX_CONTINUITY_GAP_GROUPS", "0"))
MIN_NEIGHBOR_PAIR_CORR = float(os.getenv("MIN_NEIGHBOR_PAIR_CORR", "0.3"))
MIN_NEIGHBOR_PAIR_COUNT = int(os.getenv("MIN_NEIGHBOR_PAIR_COUNT", "10"))
SAMPLE_PAIR_LIMIT = int(os.getenv("SAMPLE_PAIR_LIMIT", "40"))


def f(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return math.nan


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return math.nan
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return math.nan
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sxx * syy)


def point_key(row: dict) -> tuple[float, float]:
    return (round(f(row["latitude"]), 3), round(f(row["longitude"]), 3))


files = sorted(glob.glob("SEAS5/baseline/*/*/init????-??-01.csv"))
files_with_extrema = 0
missing_col_files = 0
rows_checked = 0
null_extrema_rows = 0
bad_order_rows = 0
bad_rh_range_rows = 0
negative_wind_rows = 0
continuity_gap_groups = 0
neighbor_corr_vals: list[float] = []

for fp in files:
    with open(fp, encoding="utf-8") as fh:
        dr = csv.DictReader(fh)
        cols = dr.fieldnames or []
        if not all(c in cols for c in TARGET):
            missing_col_files += 1
            continue
        files_with_extrema += 1

        if not all(c in cols for c in REQ_CONT):
            # Cannot perform continuity/correlation without keys.
            continue

        leads_by_point: dict[tuple[float, float], set[int]] = defaultdict(set)
        t2m_by_point_lead: dict[tuple[float, float], dict[int, float]] = defaultdict(dict)

        for r in dr:
            rows_checked += 1
            p = point_key(r)
            ld = int(f(r["lead_day"])) if not math.isnan(f(r["lead_day"])) else None
            if ld is not None:
                leads_by_point[p].add(ld)

            rhmin, rhmax = f(r["rh_pct_min"]), f(r["rh_pct_max"])
            wmin, wmax = f(r["wind_mps_min"]), f(r["wind_mps_max"])
            rh, w = f(r.get("rh_pct", "")), f(r.get("wind_mps", ""))

            if any(math.isnan(v) for v in [rhmin, rhmax, wmin, wmax]):
                null_extrema_rows += 1
            else:
                if rhmin > rhmax or wmin > wmax:
                    bad_order_rows += 1
                if not (0 <= rhmin <= 100 and 0 <= rhmax <= 100):
                    bad_rh_range_rows += 1
                if min(wmin, wmax) < 0:
                    negative_wind_rows += 1
                if not math.isnan(rh) and not (rhmin - 1e-6 <= rh <= rhmax + 1e-6):
                    bad_order_rows += 1
                if not math.isnan(w) and not (wmin - 1e-6 <= w <= wmax + 1e-6):
                    bad_order_rows += 1

            t2m = f(r.get("t2m_C", ""))
            if ld is not None and not math.isnan(t2m):
                t2m_by_point_lead[p][ld] = t2m

        # continuity
        for p, leads in leads_by_point.items():
            if not leads:
                continue
            mn, mx = min(leads), max(leads)
            if len(leads) != (mx - mn + 1):
                continuity_gap_groups += 1

        # cross-point neighbor correlation using t2m by lead_day
        points = list(t2m_by_point_lead.keys())
        # Sample stable subset for speed in very dense grids.
        if len(points) > SAMPLE_PAIR_LIMIT:
            points = points[:SAMPLE_PAIR_LIMIT]
        for a, b in combinations(points, 2):
            # neighbors: 1 degree apart along either axis, same other axis
            dlat = abs(a[0] - b[0])
            dlon = abs(a[1] - b[1])
            is_neighbor = (abs(dlat - 1.0) < 1e-6 and dlon < 1e-6) or (abs(dlon - 1.0) < 1e-6 and dlat < 1e-6)
            if not is_neighbor:
                continue
            la = t2m_by_point_lead[a]
            lb = t2m_by_point_lead[b]
            common = sorted(set(la.keys()) & set(lb.keys()))
            if len(common) < 20:
                continue
            xa = [la[k] for k in common]
            xb = [lb[k] for k in common]
            c = pearson(xa, xb)
            if not math.isnan(c):
                neighbor_corr_vals.append(c)

neighbor_pair_count = len(neighbor_corr_vals)
neighbor_pair_corr_min = min(neighbor_corr_vals) if neighbor_corr_vals else math.nan
neighbor_pair_corr_med = sorted(neighbor_corr_vals)[neighbor_pair_count // 2] if neighbor_corr_vals else math.nan

print(f"files_total={len(files)}")
print(f"files_with_extrema_cols={files_with_extrema}")
print(f"missing_extrema_col_files={missing_col_files}")
print(f"rows_checked={rows_checked}")
print(f"null_extrema_rows={null_extrema_rows}")
print(f"bad_order_rows={bad_order_rows}")
print(f"bad_rh_range_rows={bad_rh_range_rows}")
print(f"negative_wind_rows={negative_wind_rows}")
print(f"continuity_gap_groups={continuity_gap_groups}")
print(f"neighbor_pair_count={neighbor_pair_count}")
print(f"neighbor_pair_corr_min={neighbor_pair_corr_min}")
print(f"neighbor_pair_corr_median={neighbor_pair_corr_med}")

errors = []
if missing_col_files > MAX_MISSING_COL_FILES:
    errors.append(f"missing_extrema_col_files={missing_col_files} > {MAX_MISSING_COL_FILES}")
if null_extrema_rows > MAX_NULL_EXTREMA_ROWS:
    errors.append(f"null_extrema_rows={null_extrema_rows} > {MAX_NULL_EXTREMA_ROWS}")
if bad_order_rows > MAX_BAD_ORDER_ROWS:
    errors.append(f"bad_order_rows={bad_order_rows} > {MAX_BAD_ORDER_ROWS}")
if bad_rh_range_rows > MAX_BAD_RH_RANGE_ROWS:
    errors.append(f"bad_rh_range_rows={bad_rh_range_rows} > {MAX_BAD_RH_RANGE_ROWS}")
if negative_wind_rows > MAX_NEG_WIND_ROWS:
    errors.append(f"negative_wind_rows={negative_wind_rows} > {MAX_NEG_WIND_ROWS}")
if continuity_gap_groups > MAX_CONTINUITY_GAP_GROUPS:
    errors.append(f"continuity_gap_groups={continuity_gap_groups} > {MAX_CONTINUITY_GAP_GROUPS}")
if neighbor_pair_count >= MIN_NEIGHBOR_PAIR_COUNT and (math.isnan(neighbor_pair_corr_min) or neighbor_pair_corr_min < MIN_NEIGHBOR_PAIR_CORR):
    errors.append(
        f"neighbor_pair_corr_min={neighbor_pair_corr_min} < MIN_NEIGHBOR_PAIR_CORR={MIN_NEIGHBOR_PAIR_CORR}"
    )

if errors:
    print("QA_STATUS=FAIL")
    for e in errors:
        print(f"QA_ERROR: {e}")
    raise SystemExit(1)

print("QA_STATUS=PASS")
