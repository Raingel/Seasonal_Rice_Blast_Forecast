# SEAS5 Native Grid QC Report

## Recommendation
- recommended native grid: `half_degree_cell_center`
- expected fractions: lat=[0.5] lon=[0.5]
- reason: Fresh CDS samples and current latest raw cache are consistently on 1-degree cell centers at .5, while baseline CSVs and baseline caches are legacy integer-grid outputs.

## latest_csv
- files=3
- half_degree_centered_files=3
- integer_degree_centered_files=0
- fractions lat=[0.5] lon=[0.5]: count=3 example=`SEAS5\latest\init2026-01-01.csv`
- n_points=6: count=3

## baseline_csv
- files=312
- half_degree_centered_files=0
- integer_degree_centered_files=312
- fractions lat=[0.0] lon=[0.0]: count=312 example=`SEAS5\baseline\2000\01\init2000-01-01.csv`
- n_points=9: count=68
- n_points=11: count=4
- n_points=12: count=240

## latest_cache_nc
- files=3
- half_degree_centered_files=3
- integer_degree_centered_files=0
- fractions lat=[0.5] lon=[0.5]: count=3 example=`SEAS5\latest\_cache_nc\init2026-01-01_inst.nc`
- n_points=6: count=3

## baseline_monthly_cache_nc
- files=240
- half_degree_centered_files=0
- integer_degree_centered_files=240
- fractions lat=[0.0] lon=[0.0]: count=240 example=`SEAS5\baseline\2000\01\_cache_nc\init2000-01-01_inst.nc`
- n_points=12: count=240

## baseline_legacy_cache_nc
- files=292
- half_degree_centered_files=0
- integer_degree_centered_files=292
- fractions lat=[0.0] lon=[0.0]: count=292 example=`SEAS5\baseline\2020\01\_cache_nc\lat22.500_lon120.500_init2020-01-01_inst.nc`
- n_points=4: count=292

## cds_sample_nc
- files=3
- half_degree_centered_files=3
- integer_degree_centered_files=0
- fractions lat=[0.5] lon=[0.5]: count=3 example=`validation\grid_audit\cds_baseline_2011-06.nc`
- n_points=6: count=1
- n_points=16: count=1
- n_points=20: count=1

## Action Items
- Keep latest workflow on half-degree cell-center coordinates (.5).
- Do not use baseline cache_nc rebuild to recover native grid when the cache itself is integer-grid.
- Rebuild baseline from CDS/native source for months still stored on integer-grid coordinates.
