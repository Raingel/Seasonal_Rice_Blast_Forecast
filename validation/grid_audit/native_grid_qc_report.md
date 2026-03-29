# SEAS5 Native Grid QC Report

## Recommendation
- recommended native grid: `half_degree_cell_center`
- expected fractions: lat=[0.5] lon=[0.5]
- reason: Fresh CDS samples, current latest raw cache, rebuilt baseline CSVs, and rebuilt baseline monthly cache are all consistently on 1-degree cell centers at .5.

## latest_csv
- files=3
- half_degree_centered_files=3
- integer_degree_centered_files=0
- fractions lat=[0.5] lon=[0.5]: count=3 example=`SEAS5\latest\init2026-01-01.csv`
- n_points=6: count=3

## baseline_csv
- files=312
- half_degree_centered_files=312
- integer_degree_centered_files=0
- fractions lat=[0.5] lon=[0.5]: count=312 example=`SEAS5\baseline\2000\01\init2000-01-01.csv`
- n_points=6: count=312

## latest_cache_nc
- files=3
- half_degree_centered_files=3
- integer_degree_centered_files=0
- fractions lat=[0.5] lon=[0.5]: count=3 example=`SEAS5\latest\_cache_nc\init2026-01-01_inst.nc`
- n_points=6: count=3

## baseline_monthly_cache_nc
- files=312
- half_degree_centered_files=312
- integer_degree_centered_files=0
- fractions lat=[0.5] lon=[0.5]: count=312 example=`SEAS5\baseline\2000\01\_cache_nc\init2000-01-01_inst.nc`
- n_points=6: count=312

## baseline_legacy_cache_nc
- files=0
- half_degree_centered_files=0
- integer_degree_centered_files=0

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
- Baseline has already been rebuilt to the native half-degree cell-center grid.
- Do not use legacy integer-grid cache files as a rebuild source if they reappear in future runs.
