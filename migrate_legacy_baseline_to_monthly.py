#!/usr/bin/env python3
"""One-time migration tool for SEAS5 baseline outputs.

What it does
1) Discover legacy per-point files:
   SEAS5/baseline/<YYYY>/<MM>/lat*_lon*_initYYYY-MM-01.csv
2) Validate data completeness for each month.
3) Merge them into one monthly file:
   SEAS5/baseline/<YYYY>/<MM>/initYYYY-MM-01.csv
4) Remove legacy per-point files only after successful validation/write.

The script is idempotent: if no legacy files are found, it exits cleanly.
"""

from __future__ import annotations

import argparse
import calendar
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

LEGACY_GLOB = "lat*_lon*_init????-??-01.csv"


@dataclass
class MonthStats:
    year: int
    month: int
    legacy_files: int
    points: int
    expected_days: int
    rows: int


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="SEAS5/baseline", help="Baseline root directory")
    p.add_argument("--horizon-ahead-months", type=int, default=4)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()




def collapse_duplicate_rows(df: pd.DataFrame, key_cols: List[str]) -> pd.DataFrame:
    """
    Legacy per-point files can overlap in coverage (same lat/lon/lead_day).
    Collapse duplicates deterministically so migration can proceed.

    Strategy
    - group by key columns
    - numeric columns: mean
    - non-numeric columns: first non-null value
    """
    if not df.duplicated(key_cols).any():
        return df

    value_cols = [c for c in df.columns if c not in key_cols]
    numeric_cols = [c for c in value_cols if pd.api.types.is_numeric_dtype(df[c])]
    other_cols = [c for c in value_cols if c not in numeric_cols]

    agg = {c: "mean" for c in numeric_cols}
    for c in other_cols:
        agg[c] = lambda x: x.dropna().iloc[0] if not x.dropna().empty else pd.NA

    collapsed = df.groupby(key_cols, as_index=False).agg(agg)
    return collapsed

def validate_month(df: pd.DataFrame, y: int, m: int, horizon_ahead_months: int) -> Tuple[int, int]:
    required_cols = {
        "latitude",
        "longitude",
        "init_date",
        "valid_date",
        "lead_day",
    }
    miss = required_cols - set(df.columns)
    if miss:
        raise ValueError(f"{y}-{m:02d}: missing required columns: {sorted(miss)}")

    init_str = f"{y:04d}-{m:02d}-01"
    bad_init = df["init_date"].astype(str) != init_str
    if bad_init.any():
        n_bad = int(bad_init.sum())
        raise ValueError(f"{y}-{m:02d}: found {n_bad} rows with unexpected init_date")

    end_date = horizon_end_date(y, m, horizon_ahead_months)
    expected_days = (end_date - date(y, m, 1)).days + 1

    lead = pd.to_numeric(df["lead_day"], errors="coerce")
    if lead.isna().any():
        raise ValueError(f"{y}-{m:02d}: lead_day contains non-numeric values")
    lead = lead.astype(int)

    if (lead < 0).any() or (lead >= expected_days).any():
        bad = int(((lead < 0) | (lead >= expected_days)).sum())
        raise ValueError(f"{y}-{m:02d}: found {bad} rows with out-of-range lead_day")

    point_cols = ["latitude", "longitude"]
    key_cols = point_cols + ["lead_day"]
    if "number" in df.columns:
        key_cols = point_cols + ["number", "lead_day"]

    dup = df.duplicated(key_cols)
    if dup.any():
        raise ValueError(
            f"{y}-{m:02d}: duplicated rows still present after collapse; "
            f"keys={key_cols}"
        )

    points = (
        df[point_cols]
        .drop_duplicates()
        .sort_values(point_cols)
        .reset_index(drop=True)
    )
    expected_leads = set(range(expected_days))

    if "number" in df.columns:
        grp_cols = ["latitude", "longitude", "number"]
    else:
        grp_cols = ["latitude", "longitude"]

    for keys, g in df.groupby(grp_cols, dropna=False):
        got = set(pd.to_numeric(g["lead_day"], errors="coerce").astype(int).tolist())
        if got != expected_leads:
            missing = sorted(expected_leads - got)
            extra = sorted(got - expected_leads)
            raise ValueError(
                f"{y}-{m:02d}: incomplete lead_day for {keys}; "
                f"missing={missing[:10]} extra={extra[:10]}"
            )

    return len(points), expected_days


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        print(f"[INFO] root not found: {root}")
        return 0

    ym_to_files: Dict[Tuple[int, int], List[Path]] = {}
    for f in root.glob("*/*/" + LEGACY_GLOB):
        try:
            y = int(f.parent.parent.name)
            m = int(f.parent.name)
        except ValueError:
            continue
        ym_to_files.setdefault((y, m), []).append(f)

    if not ym_to_files:
        print("[INFO] no legacy per-point files found; nothing to migrate")
        return 0

    stats: List[MonthStats] = []

    for (y, m) in sorted(ym_to_files):
        files = sorted(ym_to_files[(y, m)])
        month_dir = root / f"{y:04d}" / f"{m:02d}"
        monthly_csv = month_dir / f"init{y:04d}-{m:02d}-01.csv"

        print(f"[CHECK] {y}-{m:02d}: legacy_files={len(files)}")
        frames = []
        for f in files:
            df = pd.read_csv(f)
            if df.empty:
                raise ValueError(f"{y}-{m:02d}: empty legacy file: {f}")
            frames.append(df)

        merged = pd.concat(frames, ignore_index=True)

        key_cols = ["latitude", "longitude", "lead_day"]
        if "number" in merged.columns:
            key_cols = ["latitude", "longitude", "number", "lead_day"]

        n_raw = len(merged)
        merged = collapse_duplicate_rows(merged, key_cols=key_cols)
        n_after = len(merged)
        if n_after < n_raw:
            print(f"[INFO] {y}-{m:02d}: deduplicated rows {n_raw} -> {n_after}")

        points, expected_days = validate_month(
            merged,
            y=y,
            m=m,
            horizon_ahead_months=args.horizon_ahead_months,
        )

        sort_cols = ["latitude", "longitude", "lead_day"]
        if "number" in merged.columns:
            sort_cols = ["latitude", "longitude", "number", "lead_day"]
        merged = merged.sort_values(sort_cols).reset_index(drop=True)

        stats.append(
            MonthStats(
                year=y,
                month=m,
                legacy_files=len(files),
                points=points,
                expected_days=expected_days,
                rows=len(merged),
            )
        )

        if args.dry_run:
            continue

        merged.to_csv(monthly_csv, index=False)
        for f in files:
            f.unlink()

        print(f"[OK] migrated {y}-{m:02d} -> {monthly_csv} rows={len(merged)}")

    report = root / "_migration_report.txt"
    lines = [
        "legacy_to_monthly_migration_report",
        f"dry_run={int(args.dry_run)}",
        f"months={len(stats)}",
        "",
    ]
    for s in stats:
        lines.append(
            f"{s.year:04d}-{s.month:02d} legacy_files={s.legacy_files} "
            f"points={s.points} expected_days={s.expected_days} rows={s.rows}"
        )

    if not args.dry_run:
        report.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[OK] wrote report: {report}")
    else:
        print("\n".join(lines))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
