# 最小上傳檔案清單（把既有模型移植到本 repo）

## 先講結論
如果你要在這個 repo 用**現有 SEAS5 歷史資料**做跨數十年的每日風險預報，最少只要補齊「模型本體」與（可選）「最新季預報檔」。

### 必要（最少）
1. `4_model/final/final_model.pt`
2. `4_model/final/final_model_config.json`
3. `3_data_packed/norm_params.json`

> 這三個檔案對應你舊程式 Cell 4/5 的 `MODEL_PT_PATH`、`MODEL_CFG_PATH`、`NORM_PARAMS_JSON`。

### 視需求（可選）
4. **最新預報單檔**（如果你要畫/算「latest forecast」那條黑線）
   - 例如放在 `SEAS5/latest/initYYYY-MM-01.csv`。
5. `points.csv`（若你要自訂格點）
   - 若不提供，本 repo 的資料建置流程會用台灣預設格點中心（lat 22.5/23.5/24.5 × lon 120.5/121.5）。

## 為什麼 baseline 不用再上傳
本 repo 已有大量 baseline 月初始化檔，格式為：

- `SEAS5/baseline/<YYYY>/<MM>/initYYYY-MM-01.csv`

這正是 repo 內建的輸出結構；可直接作為你後續「整併成每年/每日序列再送模型」的來源資料。

## 你的推論程式需要的欄位（從 baseline 轉特徵）
你舊 code 的 `BASELINE_MAP` 對 baseline CSV 的主要依賴欄位是：

- `valid_date`
- `t2m_C`, `t2m_C_max`, `t2m_C_min`
- `rh_pct`（若要 max/min，需有 `rh_pct_max`, `rh_pct_min`）
- `wind_mps`（若要 max/min，需有 `wind_mps_max`, `wind_mps_min`）
- `tp_mm_mean`（或 fallback `tp_mm`）

此外，為了分年/分點整理，通常也會用到：
- `latitude`, `longitude`, `init_date`, `lead_day`。

## 建議的最小移植步驟
1. 先把上面 3 個必要檔放到 repo 對應路徑。
2. 寫一個小轉接器，把 `SEAS5/baseline/*/*/init*.csv` 依 `(lat, lon, year)` 聚合成你舊程式吃的 yearly daily dataframe。
3. 套用你現有 `LOOKBACK_DAYS=30`、`GAP_DAYS=3`、z-score 與 GRUAttn 推論。
4. 若要 latest 曲線，再另外上傳 1 個 latest 檔即可。

## 上傳前快速檢查
- `final_model_config.json` 內 `time_steps` 是否等於 `LOOKBACK_DAYS - GAP_DAYS + 1`。
- `selected_feature_cols`（或 `feature_cols`）是否都能在 `norm_params.json` 找到對應 base feature。
- baseline/ latest 欄位名稱是否和你的 `BASELINE_MAP` / `LATEST_MAP` 一致。

## 如果你的模型用了 rh/wind 的 max/min（你現在這個情況）
是，代表目前 baseline 需要補欄位。

本 repo 已新增：
- `backfill_baseline_extrema_timeboxed.py`：從每月 `_cache_nc/initYYYY-MM-01_inst.nc` 回填
  `rh_pct_max/rh_pct_min/wind_mps_max/wind_mps_min` 到既有 baseline CSV。
- `.github/workflows/backfill-baseline-extrema.yml`：每 4 小時自動跑一次，
  time-box 5 小時，可中斷續跑，直到全部月份補齊。

這樣你不需要一次重抓全部資料，也不會因為單次工作時間限制而中斷整體進度。
