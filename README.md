# Seasonal Rice Blast Forecast（稻熱病季節預報資料庫）

This repository stores and auto-updates ECMWF SEAS5 seasonal forecast CSV data used for rice blast risk workflows in Taiwan.  
本 repository 主要保存與自動更新 ECMWF SEAS5 季節預報 CSV，供台灣稻熱病風險分析流程使用。

## What is kept active（目前啟用）

- **Active GitHub Action:** `.github/workflows/seas5_latest_update.yml`
- Purpose:
  - Fetch newest SEAS5 monthly forecast (with catch-up support).
  - Update `SEAS5/latest/*.csv`.
  - Regenerate anomaly validation report under `validation/anomaly_reports/`.

## What is archived/inactive（已封存停用）

To keep maintenance simple and avoid accidental one-off jobs, all non-essential workflows were moved to:

- `.github/workflows/bak/`

GitHub Actions only executes workflow files directly under `.github/workflows/`, so files in `bak/` are inactive.

## Update behavior（更新方式）

- Scheduled monthly around ECMWF release timing (plus retry).
- Additional quarterly health-check trigger to ensure the seasonal pipeline remains healthy.
- Manual dispatch is still available for catch-up runs.

## Repository structure（主要結構）

- `SEAS5/latest/`: latest seasonal forecast CSVs.
- `validation/anomaly_reports/`: generated anomaly reports.
- `seas5_update_latest_timeboxed.py`: latest forecast update script.
- `.github/workflows/seas5_latest_update.yml`: the only active automation workflow.
- `.github/workflows/bak/`: archived inactive workflows.

## Maintenance notes（維運備註）

- Keep the update path single and simple: **latest update only**.
- If future one-off rebuild/backfill is needed, copy a workflow back from `bak/` to `.github/workflows/` temporarily.
- Avoid adding heavy multi-path automation unless clearly needed, to reduce merge complexity.
