#!/usr/bin/env python3
"""Summarize SEAS5 grid evidence across CSV outputs, local caches, and CDS samples.

Purpose
- Show which datasets/files currently use legacy integer-grid coordinates.
- Show which raw/native samples use half-degree cell-center coordinates.
- Produce a compact JSON/Markdown report for QC and rebuild planning.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import h5py


OUT_DIR = Path("validation") / "grid_audit"


@dataclass
class GridSummary:
    path: str
    source_kind: str
    n_points: int
    lat_values: List[float]
    lon_values: List[float]
    lat_fraction_set: List[float]
    lon_fraction_set: List[float]
    is_half_degree_centered: bool
    is_integer_degree_centered: bool


def frac_set(vals: Sequence[float]) -> List[float]:
    return sorted({round(float(v) - math.floor(float(v)), 6) for v in vals})


def grid_summary(path: Path, points: Iterable[Tuple[float, float]], source_kind: str) -> GridSummary:
    pts = sorted({(float(lat), float(lon)) for lat, lon in points})
    lats = sorted({lat for lat, _ in pts})
    lons = sorted({lon for _, lon in pts})
    lat_frac = frac_set(lats)
    lon_frac = frac_set(lons)
    return GridSummary(
        path=str(path),
        source_kind=source_kind,
        n_points=len(pts),
        lat_values=lats,
        lon_values=lons,
        lat_fraction_set=lat_frac,
        lon_fraction_set=lon_frac,
        is_half_degree_centered=(lat_frac == [0.5] and lon_frac == [0.5]),
        is_integer_degree_centered=(lat_frac == [0.0] and lon_frac == [0.0]),
    )


def read_csv_summary(path: Path, source_kind: str) -> GridSummary:
    with path.open("r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        pts = {(float(r["latitude"]), float(r["longitude"])) for r in rd}
    return grid_summary(path, pts, source_kind)


def read_nc_summary(path: Path, source_kind: str) -> GridSummary:
    with h5py.File(path, "r") as f:
        lats = [float(v) for v in f["latitude"][...].tolist()]
        lons = [float(v) for v in f["longitude"][...].tolist()]
    pts = {(lat, lon) for lat in lats for lon in lons}
    return grid_summary(path, pts, source_kind)


def summarize_group(label: str, summaries: List[GridSummary]) -> dict:
    frac_counter = Counter((tuple(s.lat_fraction_set), tuple(s.lon_fraction_set)) for s in summaries)
    point_counter = Counter(s.n_points for s in summaries)
    examples = {}
    for s in summaries:
        key = (tuple(s.lat_fraction_set), tuple(s.lon_fraction_set))
        examples.setdefault(key, s.path)
    return {
        "label": label,
        "files": len(summaries),
        "fraction_patterns": [
            {
                "lat_fraction_set": list(latf),
                "lon_fraction_set": list(lonf),
                "count": count,
                "example_path": examples[(latf, lonf)],
            }
            for (latf, lonf), count in sorted(frac_counter.items(), key=lambda x: (-x[1], x[0]))
        ],
        "point_count_patterns": [
            {"n_points": n_points, "count": count}
            for n_points, count in sorted(point_counter.items())
        ],
        "half_degree_centered_files": sum(1 for s in summaries if s.is_half_degree_centered),
        "integer_degree_centered_files": sum(1 for s in summaries if s.is_integer_degree_centered),
        "sample_files": [asdict(s) for s in summaries[:3]],
    }


def build_report() -> dict:
    latest_csv = [read_csv_summary(p, "latest_csv") for p in sorted(Path("SEAS5/latest").glob("init*.csv"))]
    baseline_csv = [read_csv_summary(p, "baseline_csv") for p in sorted(Path("SEAS5/baseline").glob("*/*/init*.csv"))]
    latest_cache = [
        read_nc_summary(p, "latest_cache_nc")
        for p in sorted((Path("SEAS5/latest") / "_cache_nc").glob("init*_inst.nc"))
    ]
    baseline_monthly_cache = [
        read_nc_summary(p, "baseline_monthly_cache_nc")
        for p in sorted(Path("SEAS5/baseline").glob("*/*/_cache_nc/init*_inst.nc"))
    ]
    baseline_legacy_cache = [
        read_nc_summary(p, "baseline_legacy_cache_nc")
        for p in sorted(Path("SEAS5/baseline").glob("*/*/_cache_nc/lat*_lon*_init*_inst.nc"))
    ]
    cds_samples = [
        read_nc_summary(p, "cds_sample_nc")
        for p in sorted((Path("validation") / "grid_audit").glob("cds_*.nc"))
    ]

    groups = [
        summarize_group("latest_csv", latest_csv),
        summarize_group("baseline_csv", baseline_csv),
        summarize_group("latest_cache_nc", latest_cache),
        summarize_group("baseline_monthly_cache_nc", baseline_monthly_cache),
        summarize_group("baseline_legacy_cache_nc", baseline_legacy_cache),
        summarize_group("cds_sample_nc", cds_samples),
    ]

    baseline_is_native = bool(baseline_csv) and all(s.is_half_degree_centered for s in baseline_csv)
    baseline_cache_is_native = bool(baseline_monthly_cache) and all(s.is_half_degree_centered for s in baseline_monthly_cache)
    legacy_cache_present = any(True for _ in baseline_legacy_cache)

    if baseline_is_native and baseline_cache_is_native and (not legacy_cache_present):
        reason = (
            "Fresh CDS samples, current latest raw cache, rebuilt baseline CSVs, and rebuilt baseline monthly cache "
            "are all consistently on 1-degree cell centers at .5."
        )
        action_items = [
            "Keep latest workflow on half-degree cell-center coordinates (.5).",
            "Baseline has already been rebuilt to the native half-degree cell-center grid.",
            "Do not use legacy integer-grid cache files as a rebuild source if they reappear in future runs.",
        ]
    else:
        reason = (
            "Fresh CDS samples and current latest raw cache are consistently on 1-degree cell centers at .5, "
            "while some baseline outputs or caches still indicate legacy integer-grid processing."
        )
        action_items = [
            "Keep latest workflow on half-degree cell-center coordinates (.5).",
            "Do not use baseline cache_nc rebuild to recover native grid when the cache itself is integer-grid.",
            "Rebuild baseline from CDS/native source for months still stored on integer-grid coordinates.",
        ]

    return {
        "native_grid_recommendation": {
            "recommended_label": "half_degree_cell_center",
            "recommended_fraction_set": {"latitude": [0.5], "longitude": [0.5]},
            "reason": reason,
        },
        "groups": groups,
        "action_items": action_items,
    }


def write_report(report: dict) -> Tuple[Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "native_grid_qc_report.json"
    md_path = OUT_DIR / "native_grid_qc_report.md"

    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# SEAS5 Native Grid QC Report",
        "",
        "## Recommendation",
        f"- recommended native grid: `{report['native_grid_recommendation']['recommended_label']}`",
        (
            "- expected fractions: "
            f"lat={report['native_grid_recommendation']['recommended_fraction_set']['latitude']} "
            f"lon={report['native_grid_recommendation']['recommended_fraction_set']['longitude']}"
        ),
        f"- reason: {report['native_grid_recommendation']['reason']}",
        "",
    ]

    for group in report["groups"]:
        lines.append(f"## {group['label']}")
        lines.append(f"- files={group['files']}")
        lines.append(f"- half_degree_centered_files={group['half_degree_centered_files']}")
        lines.append(f"- integer_degree_centered_files={group['integer_degree_centered_files']}")
        for item in group["fraction_patterns"]:
            lines.append(
                f"- fractions lat={item['lat_fraction_set']} lon={item['lon_fraction_set']}: "
                f"count={item['count']} example=`{item['example_path']}`"
            )
        for item in group["point_count_patterns"]:
            lines.append(f"- n_points={item['n_points']}: count={item['count']}")
        lines.append("")

    lines.append("## Action Items")
    for item in report["action_items"]:
        lines.append(f"- {item}")
    lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def main() -> int:
    report = build_report()
    json_path, md_path = write_report(report)
    print(f"[OK] wrote {json_path}")
    print(f"[OK] wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
