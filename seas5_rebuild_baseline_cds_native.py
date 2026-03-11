#!/usr/bin/env python3
"""One-off/resumable baseline rebuild runner to restore CDS-native grid for SEAS5 baseline.

This script is intentionally conservative:
1) PLAN mode (default): inspect current baseline and produce risk/report files only.
2) EXECUTE mode: archive current baseline once, rebuild with timeboxed downloader,
   then verify rebuilt grid fractions. Safe for repeated resume runs.

Why:
- Current audit shows latest aligns with CDS native (.5 grid) but baseline is legacy (.0 grid).
- We want a careful, auditable one-time rebuild before changing downstream logic again.
"""

from __future__ import annotations

import argparse
import calendar
import csv
import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


REPO_BASELINE_DIR = Path("SEAS5/baseline")
REPORT_DIR = Path("validation/grid_audit")
STATE_FILE = Path("SEAS5") / "_baseline_rebuild_state.json"


@dataclass
class MonthGridCheck:
    year: int
    month: int
    exists: bool
    n_points: int
    lat_frac_set: List[float]
    lon_frac_set: List[float]
    is_native_half_grid: bool


@dataclass
class RebuildReport:
    mode: str
    baseline_dir: str
    start_year: int
    end_year: int
    total_months: int
    months_present: int
    months_half_grid: int
    months_non_half_grid: int
    checks: List[MonthGridCheck]
    archived_from: str | None
    archived_to: str | None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start-year", type=int, default=2000)
    p.add_argument("--end-year", type=int, default=2025)
    p.add_argument("--mode", choices=["plan", "execute"], default="plan")
    p.add_argument("--archive-root", default="SEAS5")
    p.add_argument("--time-budget-sec", type=int, default=18000)
    p.add_argument("--stop-grace-sec", type=int, default=180)
    p.add_argument("--debug", default="0")
    return p.parse_args()


def month_iter(y0: int, y1: int):
    for y in range(y0, y1 + 1):
        for m in range(1, 13):
            yield y, m


def _frac_set(vals: List[float]) -> List[float]:
    out = sorted({round(v - int(v), 6) for v in vals})
    return out


def check_month_grid(csv_path: Path, y: int, m: int) -> MonthGridCheck:
    if not csv_path.exists():
        return MonthGridCheck(y, m, False, 0, [], [], False)

    pts = set()
    with csv_path.open("r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            pts.add((float(r["latitude"]), float(r["longitude"])))

    if not pts:
        return MonthGridCheck(y, m, True, 0, [], [], False)

    lats = [p[0] for p in pts]
    lons = [p[1] for p in pts]
    lat_frac = _frac_set(lats)
    lon_frac = _frac_set(lons)
    is_half = (lat_frac == [0.5]) and (lon_frac == [0.5])
    return MonthGridCheck(y, m, True, len(pts), lat_frac, lon_frac, is_half)


def inspect_baseline_grid(root: Path, y0: int, y1: int) -> RebuildReport:
    checks: List[MonthGridCheck] = []
    for y, m in month_iter(y0, y1):
        csv_path = root / f"{y:04d}" / f"{m:02d}" / f"init{y:04d}-{m:02d}-01.csv"
        checks.append(check_month_grid(csv_path, y, m))

    total = len(checks)
    present = sum(1 for c in checks if c.exists)
    half = sum(1 for c in checks if c.exists and c.is_native_half_grid)
    non_half = sum(1 for c in checks if c.exists and not c.is_native_half_grid)
    return RebuildReport(
        mode="plan",
        baseline_dir=str(root),
        start_year=y0,
        end_year=y1,
        total_months=total,
        months_present=present,
        months_half_grid=half,
        months_non_half_grid=non_half,
        checks=checks,
        archived_from=None,
        archived_to=None,
    )


def write_report(report: RebuildReport, suffix: str) -> Tuple[Path, Path]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    json_fp = REPORT_DIR / f"baseline_rebuild_{suffix}_{ts}.json"
    md_fp = REPORT_DIR / f"baseline_rebuild_{suffix}_{ts}.md"

    payload = {
        **{k: v for k, v in asdict(report).items() if k != "checks"},
        "checks": [asdict(c) for c in report.checks],
    }
    json_fp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        f"# Baseline Rebuild Report ({suffix})",
        "",
        f"- mode: {report.mode}",
        f"- baseline_dir: `{report.baseline_dir}`",
        f"- year_range: {report.start_year}-{report.end_year}",
        f"- months_present: {report.months_present}/{report.total_months}",
        f"- months_half_grid: {report.months_half_grid}",
        f"- months_non_half_grid: {report.months_non_half_grid}",
    ]
    if report.archived_to:
        lines.append(f"- archived_to: `{report.archived_to}`")
    lines.append("")
    lines.append("## Non-half-grid months")
    bad = [c for c in report.checks if c.exists and not c.is_native_half_grid]
    if not bad:
        lines.append("- none")
    else:
        for c in bad[:200]:
            lines.append(
                f"- {c.year:04d}-{c.month:02d}: n_points={c.n_points}, "
                f"lat_frac={c.lat_frac_set}, lon_frac={c.lon_frac_set}"
            )
    md_fp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_fp, md_fp


def ensure_cds_env() -> None:
    if not os.getenv("CDSAPI_KEY", "").strip():
        raise RuntimeError("Missing CDSAPI_KEY in environment")


def run_rebuild_pass(args: argparse.Namespace) -> None:
    env = os.environ.copy()
    env.update(
        {
            "BASELINE_YEAR_MIN": str(args.start_year),
            "BASELINE_YEAR_MAX": str(args.end_year),
            "INIT_MONTHS": "1-12",
            "HORIZON_AHEAD_MONTHS": "4",
            "OUT_ROOT": "SEAS5",
            "TIME_BUDGET_SEC": str(args.time_budget_sec),
            "STOP_GRACE_SEC": str(args.stop_grace_sec),
            "DEBUG": str(args.debug),
        }
    )
    cmd = ["python", "seas5_build_monthly_baseline_timeboxed.py"]
    print(f"[RUN] rebuild pass years={args.start_year}-{args.end_year}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, env=env, check=True)


def load_state() -> Dict[str, object]:
    if not STATE_FILE.exists():
        return {}
    return json.loads(STATE_FILE.read_text(encoding="utf-8"))


def save_state(state: Dict[str, object]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def execute_rebuild(args: argparse.Namespace) -> RebuildReport:
    ensure_cds_env()

    state = load_state()
    archived_to = None
    if not state.get("archive_done", False):
        archive_root = Path(args.archive_root)
        archive_root.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        archived_to = archive_root / f"baseline_legacy_backup_{ts}"
        if REPO_BASELINE_DIR.exists():
            print(f"[ARCHIVE] move {REPO_BASELINE_DIR} -> {archived_to}", flush=True)
            shutil.move(str(REPO_BASELINE_DIR), str(archived_to))
        REPO_BASELINE_DIR.mkdir(parents=True, exist_ok=True)
        state = {
            "archive_done": True,
            "archived_to": str(archived_to),
            "start_year": args.start_year,
            "end_year": args.end_year,
            "created_at_utc": datetime.utcnow().isoformat() + "Z",
        }
        save_state(state)
    else:
        archived_to = Path(str(state.get("archived_to", ""))) if state.get("archived_to") else None

    run_rebuild_pass(args)

    rep = inspect_baseline_grid(REPO_BASELINE_DIR, args.start_year, args.end_year)
    rep.mode = "execute"
    rep.archived_from = str(REPO_BASELINE_DIR)
    rep.archived_to = str(archived_to) if archived_to else None
    return rep


def main() -> int:
    args = parse_args()

    if args.mode == "plan":
        rep = inspect_baseline_grid(REPO_BASELINE_DIR, args.start_year, args.end_year)
        j, m = write_report(rep, "plan")
        print(f"[OK] plan report: {j}")
        print(f"[OK] plan report: {m}")
        return 0

    rep = execute_rebuild(args)
    j, m = write_report(rep, "execute")
    print(f"[OK] execute report: {j}")
    print(f"[OK] execute report: {m}")

    if rep.months_non_half_grid > 0:
        print("[WARN] rebuild finished but non-half-grid months still exist", flush=True)
        return 2

    print("[OK] rebuild completed with CDS-native half-grid for all present months", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
