# SEAS5 Grid Audit Report

## latest:2026-01
- csv: `SEAS5/latest/init2026-01-01.csv`
- csv n_points=6 lat_frac=[0.5] lon_frac=[0.5]
- cds nc: `validation/grid_audit/cds_latest_2026-01.nc`
- cds n_points=6 lat_frac=[0.5] lon_frac=[0.5]
- overlap vs csv: 6/6 (1.000)

## baseline:2025-01
- csv: `SEAS5/baseline/2025/01/init2025-01-01.csv`
- csv n_points=9 lat_frac=[0.0] lon_frac=[0.0]
- cds nc: `validation/grid_audit/cds_baseline_2025-01.nc`
- cds n_points=16 lat_frac=[0.5] lon_frac=[0.5]
- overlap vs csv: 0/9 (0.000)

## baseline:2011-06
- csv: `SEAS5/baseline/2011/06/init2011-06-01.csv`
- csv n_points=12 lat_frac=[0.0] lon_frac=[0.0]
- cds nc: `validation/grid_audit/cds_baseline_2011-06.nc`
- cds n_points=20 lat_frac=[0.5] lon_frac=[0.5]
- overlap vs csv: 0/12 (0.000)

## Cross-sample CSV overlap
- latest:2026-01__vs__baseline:2025-01: intersection=0, ratio_vs_left=0.000, ratio_vs_right=0.000
- latest:2026-01__vs__baseline:2011-06: intersection=0, ratio_vs_left=0.000, ratio_vs_right=0.000
- baseline:2025-01__vs__baseline:2011-06: intersection=9, ratio_vs_left=1.000, ratio_vs_right=0.750
