#!/usr/bin/env python3
"""Audit SEAS5 grid consistency between baseline/latest CSVs and fresh CDS samples.

Usage examples
- Local check only (no download):
  python seas5_grid_sampling_audit.py --sample latest:2026-01 --sample baseline:2025-01

- Include fresh CDS downloads for each sample:
  CDSAPI_URL=... CDSAPI_KEY=... python seas5_grid_sampling_audit.py \
    --sample latest:2026-01 --sample baseline:2025-01 --download
"""

from __future__ import annotations

import argparse
import calendar
import json
import os
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import csv
import math


DATASET = "seasonal-original-single-levels"
ORIGINATING_CENTRE = "ecmwf"
SYSTEM = "51"


@dataclass
class GridSummary:
    n_points: int
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    unique_lats: int
    unique_lons: int
    lat_step_guess: float | None
    lon_step_guess: float | None
    lat_fraction_set: List[float]
    lon_fraction_set: List[float]
    sample_points: List[Tuple[float, float]]


@dataclass
class SampleResult:
    sample: str
    csv_path: str
    csv_grid: GridSummary
    downloaded: bool
    cds_nc_path: str | None
    cds_grid: GridSummary | None
    point_intersection_count: int | None
    point_intersection_ratio_vs_csv: float | None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--sample",
        action="append",
        default=[],
        help="Sample token in form latest:YYYY-MM or baseline:YYYY-MM; can be repeated.",
    )
    p.add_argument("--download", action="store_true", help="Retrieve one small CDS sample per token and compare grids.")
    p.add_argument("--horizon-ahead-months", type=int, default=4)
    p.add_argument("--grid-deg", type=float, default=1.0)
    p.add_argument("--out-dir", default="validation/grid_audit")
    return p.parse_args()


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
    return [str(h) for h in range(0, last_h + 6, 6)]


def sample_to_csv_path(sample: str) -> Path:
    kind, ym = sample.split(":", 1)
    year_s, month_s = ym.split("-", 1)
    y = int(year_s)
    m = int(month_s)
    if kind == "latest":
        return Path("SEAS5/latest") / f"init{y:04d}-{m:02d}-01.csv"
    if kind == "baseline":
        return Path("SEAS5/baseline") / f"{y:04d}" / f"{m:02d}" / f"init{y:04d}-{m:02d}-01.csv"
    raise ValueError(f"Unknown sample kind: {kind}")


def read_points_from_csv(csv_path: Path) -> List[Tuple[float, float]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    pts = set()
    with csv_path.open("r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            pts.add((float(r["latitude"]), float(r["longitude"])))
    if not pts:
        raise RuntimeError(f"CSV empty: {csv_path}")
    return sorted(pts)


def _step_guess(vals: Sequence[float]) -> float | None:
    if len(vals) < 2:
        return None
    arr = sorted(float(v) for v in vals)
    diffs = [arr[i + 1] - arr[i] for i in range(len(arr) - 1)]
    diffs = [d for d in diffs if abs(d) > 1e-9]
    if not diffs:
        return None
    return round(min(diffs), 6)


def summarize_points(points: List[Tuple[float, float]]) -> GridSummary:
    lats = sorted({p[0] for p in points})
    lons = sorted({p[1] for p in points})
    lat_frac = sorted({round(v - math.floor(v), 6) for v in lats})
    lon_frac = sorted({round(v - math.floor(v), 6) for v in lons})
    return GridSummary(
        n_points=len(points),
        lat_min=float(min(lats)),
        lat_max=float(max(lats)),
        lon_min=float(min(lons)),
        lon_max=float(max(lons)),
        unique_lats=len(lats),
        unique_lons=len(lons),
        lat_step_guess=_step_guess(lats),
        lon_step_guess=_step_guess(lons),
        lat_fraction_set=lat_frac,
        lon_fraction_set=lon_frac,
        sample_points=points[:12],
    )


def write_cdsapirc_from_env() -> None:
    url = os.getenv("CDSAPI_URL", "https://cds.climate.copernicus.eu/api").strip()
    if url.endswith("/api/v2"):
        url = url.replace("/api/v2", "/api")
    key = os.getenv("CDSAPI_KEY", "").strip()
    if not key:
        raise RuntimeError("Missing CDSAPI_KEY")
    Path.home().joinpath(".cdsapirc").write_text(f"url: {url}\nkey: {key}\n", encoding="utf-8")


def area_from_points(points: List[Tuple[float, float]], grid_deg: float) -> List[float]:
    half = grid_deg / 2.0
    lats = [p[0] for p in points]
    lons = [p[1] for p in points]
    return [max(lats) + half, min(lons) - half, min(lats) - half, max(lons) + half]


def download_grid_sample(sample: str, csv_points: List[Tuple[float, float]], out_dir: Path, horizon_ahead_months: int, grid_deg: float) -> Tuple[Path, List[Tuple[float, float]]]:
    import cdsapi
    import xarray as xr

    kind, ym = sample.split(":", 1)
    y, m = map(int, ym.split("-", 1))
    init_d = date(y, m, 1)
    end_d = horizon_end_date(y, m, horizon_ahead_months)
    req = {
        "originating_centre": ORIGINATING_CENTRE,
        "system": SYSTEM,
        "variable": ["2m_temperature"],
        "year": [f"{y:04d}"],
        "month": [f"{m:02d}"],
        "day": ["01"],
        "leadtime_hour": lead_hours_inst(init_d, end_d),
        "area": area_from_points(csv_points, grid_deg),
        "data_format": "netcdf",
    }

    nc_path = out_dir / f"cds_{kind}_{y:04d}-{m:02d}.nc"
    client = cdsapi.Client()
    client.retrieve(DATASET, req, str(nc_path))

    ds = xr.open_dataset(nc_path)
    if "latitude" not in ds.coords or "longitude" not in ds.coords:
        raise RuntimeError(f"Downloaded NC missing latitude/longitude coords: {nc_path}")
    lats = [float(x) for x in ds["latitude"].values.tolist()]
    lons = [float(x) for x in ds["longitude"].values.tolist()]
    ds.close()

    pts = sorted({(lat, lon) for lat in lats for lon in lons})
    return nc_path, pts


def main() -> int:
    args = parse_args()
    samples = args.sample or ["latest:2026-01", "baseline:2025-01", "baseline:2011-06"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.download:
        write_cdsapirc_from_env()

    results: List[SampleResult] = []
    for sample in samples:
        csv_path = sample_to_csv_path(sample)
        csv_pts = read_points_from_csv(csv_path)
        csv_grid = summarize_points(csv_pts)

        downloaded = False
        cds_nc_path = None
        cds_grid = None
        intersection_count = None
        ratio = None

        if args.download:
            nc_path, cds_pts = download_grid_sample(
                sample=sample,
                csv_points=csv_pts,
                out_dir=out_dir,
                horizon_ahead_months=args.horizon_ahead_months,
                grid_deg=args.grid_deg,
            )
            downloaded = True
            cds_nc_path = str(nc_path)
            cds_grid = summarize_points(cds_pts)

            csv_set = {(round(a, 4), round(b, 4)) for a, b in csv_pts}
            cds_set = {(round(a, 4), round(b, 4)) for a, b in cds_pts}
            inter = csv_set & cds_set
            intersection_count = len(inter)
            ratio = float(len(inter) / max(len(csv_set), 1))

        results.append(
            SampleResult(
                sample=sample,
                csv_path=str(csv_path),
                csv_grid=csv_grid,
                downloaded=downloaded,
                cds_nc_path=cds_nc_path,
                cds_grid=cds_grid,
                point_intersection_count=intersection_count,
                point_intersection_ratio_vs_csv=ratio,
            )
        )

    # cross-sample quick comparison
    cross = {}
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            a = {(round(x, 4), round(y, 4)) for x, y in read_points_from_csv(Path(results[i].csv_path))}
            b = {(round(x, 4), round(y, 4)) for x, y in read_points_from_csv(Path(results[j].csv_path))}
            inter = len(a & b)
            cross[f"{results[i].sample}__vs__{results[j].sample}"] = {
                "intersection": inter,
                "ratio_vs_left": float(inter / max(len(a), 1)),
                "ratio_vs_right": float(inter / max(len(b), 1)),
            }

    payload = {
        "dataset": DATASET,
        "originating_centre": ORIGINATING_CENTRE,
        "system": SYSTEM,
        "samples": [
            {
                **{k: v for k, v in asdict(r).items() if k not in {"csv_grid", "cds_grid"}},
                "csv_grid": asdict(r.csv_grid),
                "cds_grid": asdict(r.cds_grid) if r.cds_grid else None,
            }
            for r in results
        ],
        "cross_sample_csv_overlap": cross,
    }

    out_json = out_dir / "grid_audit_report.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    out_md = out_dir / "grid_audit_report.md"
    lines = ["# SEAS5 Grid Audit Report", ""]
    for r in results:
        lines.append(f"## {r.sample}")
        lines.append(f"- csv: `{r.csv_path}`")
        lines.append(f"- csv n_points={r.csv_grid.n_points} lat_frac={r.csv_grid.lat_fraction_set} lon_frac={r.csv_grid.lon_fraction_set}")
        if r.downloaded and r.cds_grid:
            lines.append(f"- cds nc: `{r.cds_nc_path}`")
            lines.append(f"- cds n_points={r.cds_grid.n_points} lat_frac={r.cds_grid.lat_fraction_set} lon_frac={r.cds_grid.lon_fraction_set}")
            lines.append(
                f"- overlap vs csv: {r.point_intersection_count}/{r.csv_grid.n_points} "
                f"({(r.point_intersection_ratio_vs_csv or 0.0):.3f})"
            )
        lines.append("")

    lines.append("## Cross-sample CSV overlap")
    for k, v in cross.items():
        lines.append(f"- {k}: intersection={v['intersection']}, ratio_vs_left={v['ratio_vs_left']:.3f}, ratio_vs_right={v['ratio_vs_right']:.3f}")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {out_json}")
    print(f"[OK] wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
