#!/usr/bin/env python3
"""Bulk rebuild SEAS5 baseline from CDS-native data by init month.

This script reduces request count by retrieving all target years for one init month
in a single CDS request, then splitting the result back into per-year monthly CSVs.
"""

from __future__ import annotations

import argparse
import calendar
import os
import time
from datetime import date
from pathlib import Path
from typing import List, Tuple

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr

from seas5_build_monthly_baseline_timeboxed import (
    INST_VARIABLES,
    PRECIP_VARIABLE,
    ORIGINATING_CENTRE,
    SYSTEM,
    DATASET,
    GRID_DEG,
    TAIWAN_LAT_MIN,
    TAIWAN_LAT_MAX,
    TAIWAN_LON_MIN,
    TAIWAN_LON_MAX,
    build_request,
    ensure_valid_time,
    pick_var,
    rh_from_t_td_c,
    write_cdsapirc_from_env,
)

CACHE_ROOT = Path(os.getenv('BULK_CACHE_ROOT', 'SEAS5/_bulk_native_cache'))
OUT_ROOT = Path(os.getenv('OUT_ROOT', 'SEAS5')) / 'baseline'
RETRY_MAX = int(os.getenv('RETRY_MAX', '4'))
DEBUG = os.getenv('DEBUG', '0').strip() in ('1', 'true', 'True', 'YES', 'yes')


def log(msg: str) -> None:
    print(msg, flush=True)


def dlog(msg: str) -> None:
    if DEBUG:
        log(msg)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--start-year', type=int, default=2000)
    p.add_argument('--end-year', type=int, default=2025)
    p.add_argument('--init-months', default='1-12')
    p.add_argument('--overwrite', action='store_true')
    return p.parse_args()


def parse_init_months(s: str) -> List[int]:
    s = s.strip()
    if ',' in s:
        return sorted({int(x.strip()) for x in s.split(',') if x.strip()})
    if '-' in s:
        a, b = s.split('-', 1)
        a, b = int(a.strip()), int(b.strip())
        if a > b:
            a, b = b, a
        return list(range(a, b + 1))
    return [int(s)]


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


def max_lead_hours_inst(years: List[int], month: int, ahead_months: int) -> List[str]:
    last_h = 0
    for y in years:
        init_d = date(y, month, 1)
        end_d = horizon_end_date(y, month, ahead_months)
        days = (end_d - init_d).days
        last_h = max(last_h, days * 24 + 18)
    return [str(h) for h in range(0, last_h + 6, 6)]


def max_lead_hours_tp(years: List[int], month: int, ahead_months: int) -> List[str]:
    last_h = 0
    for y in years:
        init_d = date(y, month, 1)
        end_d = horizon_end_date(y, month, ahead_months)
        days = (end_d - init_d).days
        last_h = max(last_h, (days + 1) * 24)
    return [str(h) for h in range(0, last_h + 1, 24)]


def area_from_default_points() -> List[float]:
    return [TAIWAN_LAT_MAX, TAIWAN_LON_MIN, TAIWAN_LAT_MIN, TAIWAN_LON_MAX]


def retrieve_with_retry(client: cdsapi.Client, request: dict, target: Path) -> None:
    last_err = None
    for i in range(1, RETRY_MAX + 1):
        try:
            client.retrieve(DATASET, request, str(target))
            if target.exists() and target.stat().st_size > 0:
                return
            raise RuntimeError('downloaded file missing/empty')
        except Exception as e:
            last_err = e
            log(f'[WARN] retrieve {target.name} failed attempt {i}/{RETRY_MAX}: {e}')
            if i < RETRY_MAX:
                time.sleep(min(20 * (2 ** (i - 1)), 180))
    raise RuntimeError(f'download failed: {last_err}')


def open_df_with_init(ds: xr.Dataset, vars_: List[str]) -> pd.DataFrame:
    df = ds[vars_].to_dataframe().reset_index()
    df = ensure_valid_time(df, 'bulk')
    if 'forecast_reference_time' in df.columns:
        df['init_time'] = pd.to_datetime(df['forecast_reference_time'], utc=True)
    elif 'time' in df.columns:
        df['init_time'] = pd.to_datetime(df['time'], utc=True)
    else:
        raise KeyError('cannot find init time column')
    df['init_date'] = df['init_time'].dt.date
    df['lead_day'] = ((df['valid_time'] - df['init_time']).dt.total_seconds() // 86400).astype(int)
    return df


def build_daily_frames(inst_nc: Path, tp_nc: Path) -> pd.DataFrame:
    ds_inst = xr.open_dataset(inst_nc)
    ds_tp = xr.open_dataset(tp_nc)

    t2m = pick_var(ds_inst, ['t2m', '2m_temperature'])
    d2m = pick_var(ds_inst, ['d2m', '2m_dewpoint_temperature'])
    u10 = pick_var(ds_inst, ['u10', '10m_u_component_of_wind'])
    v10 = pick_var(ds_inst, ['v10', '10m_v_component_of_wind'])
    tpv = pick_var(ds_tp, ['tp', 'total_precipitation'])

    df6 = open_df_with_init(ds_inst, [t2m, d2m, u10, v10])
    df6['t2m_C'] = df6[t2m] - 273.15
    df6['d2m_C'] = df6[d2m] - 273.15
    df6['rh_pct'] = rh_from_t_td_c(df6['t2m_C'], df6['d2m_C'])
    df6['wind_mps'] = np.sqrt(df6[u10] ** 2 + df6[v10] ** 2)

    base_cols = ['init_date', 'number', 'latitude', 'longitude', 'lead_day'] if 'number' in df6.columns else ['init_date', 'latitude', 'longitude', 'lead_day']
    df6_member = df6.groupby(base_cols, as_index=False).agg(
        t2m_C=('t2m_C', 'mean'),
        t2m_C_max=('t2m_C', 'max'),
        t2m_C_min=('t2m_C', 'min'),
        rh_pct=('rh_pct', 'mean'),
        rh_pct_max=('rh_pct', 'max'),
        rh_pct_min=('rh_pct', 'min'),
        wind_mps=('wind_mps', 'mean'),
        wind_mps_max=('wind_mps', 'max'),
        wind_mps_min=('wind_mps', 'min'),
    )
    if 'number' in df6_member.columns:
        df6_daily = df6_member.groupby(['init_date', 'latitude', 'longitude', 'lead_day'], as_index=False).agg(
            t2m_C=('t2m_C', 'mean'),
            t2m_C_max=('t2m_C_max', 'mean'),
            t2m_C_min=('t2m_C_min', 'mean'),
            rh_pct=('rh_pct', 'mean'),
            rh_pct_max=('rh_pct_max', 'mean'),
            rh_pct_min=('rh_pct_min', 'mean'),
            wind_mps=('wind_mps', 'mean'),
            wind_mps_max=('wind_mps_max', 'mean'),
            wind_mps_min=('wind_mps_min', 'mean'),
            ens_n=('number', 'nunique'),
        )
    else:
        df6_daily = df6_member.copy()
        df6_daily['ens_n'] = 1

    dfp = open_df_with_init(ds_tp, [tpv])
    dfp['tp_mm_cum'] = dfp[tpv] * 1000.0
    if 'number' in dfp.columns:
        dfp = dfp.sort_values(['init_date', 'number', 'latitude', 'longitude', 'valid_time'])
        dfp['tp_mm_24h'] = dfp.groupby(['init_date', 'number', 'latitude', 'longitude'])['tp_mm_cum'].diff()
        dfp_member = dfp.dropna(subset=['tp_mm_24h']).copy()
        dfp_member['lead_day'] = dfp_member['lead_day'] - 1
        dfp_member_daily = dfp_member.groupby(['init_date', 'number', 'latitude', 'longitude', 'lead_day'], as_index=False).agg(tp_mm_mean=('tp_mm_24h', 'sum'))
        dfp_daily = dfp_member_daily.groupby(['init_date', 'latitude', 'longitude', 'lead_day'], as_index=False).agg(tp_mm_mean=('tp_mm_mean', 'mean'))
    else:
        dfp = dfp.sort_values(['init_date', 'latitude', 'longitude', 'valid_time'])
        dfp['tp_mm_24h'] = dfp.groupby(['init_date', 'latitude', 'longitude'])['tp_mm_cum'].diff()
        dfp = dfp.dropna(subset=['tp_mm_24h']).copy()
        dfp['lead_day'] = dfp['lead_day'] - 1
        dfp_daily = dfp.groupby(['init_date', 'latitude', 'longitude', 'lead_day'], as_index=False).agg(tp_mm_mean=('tp_mm_24h', 'sum'))

    out = df6_daily.merge(dfp_daily, on=['init_date', 'latitude', 'longitude', 'lead_day'], how='inner')
    out = out.sort_values(['init_date', 'latitude', 'longitude', 'lead_day']).reset_index(drop=True)
    ds_inst.close()
    ds_tp.close()
    return out


def save_one_init(df: pd.DataFrame, init_d: date, overwrite: bool) -> str:
    y = init_d.year
    m = init_d.month
    out_dir = OUT_ROOT / f'{y:04d}' / f'{m:02d}'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f'init{y:04d}-{m:02d}-01.csv'
    if out_csv.exists() and out_csv.stat().st_size > 0 and (not overwrite):
        return 'skip_exists'
    end_d = horizon_end_date(y, m, 4)
    max_lead = (end_d - init_d).days
    d = df[(df['lead_day'] >= 0) & (df['lead_day'] <= max_lead)].copy()
    d['valid_date'] = (pd.Timestamp(init_d) + pd.to_timedelta(d['lead_day'], unit='D')).dt.date
    d['init_date'] = init_d
    cols = [
        'latitude', 'longitude', 'init_date', 'valid_date', 'lead_day', 'ens_n',
        't2m_C', 't2m_C_max', 't2m_C_min', 'rh_pct', 'wind_mps', 'tp_mm_mean',
        'rh_pct_max', 'rh_pct_min', 'wind_mps_max', 'wind_mps_min',
    ]
    d[cols].to_csv(out_csv, index=False)
    return 'written'


def process_month(client: cdsapi.Client, years: List[int], month: int, overwrite: bool) -> None:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    inst_nc = CACHE_ROOT / f'month{month:02d}_{years[0]}_{years[-1]}_inst.nc'
    tp_nc = CACHE_ROOT / f'month{month:02d}_{years[0]}_{years[-1]}_tp.nc'
    area = area_from_default_points()
    req_inst = build_request(years[0], month, max_lead_hours_inst(years, month, 4), INST_VARIABLES, area)
    req_tp = build_request(years[0], month, max_lead_hours_tp(years, month, 4), PRECIP_VARIABLE, area)
    req_inst['year'] = [f'{y:04d}' for y in years]
    req_tp['year'] = [f'{y:04d}' for y in years]
    log(f'[MONTH] {month:02d} years={years[0]}-{years[-1]} n_years={len(years)}')
    if not (inst_nc.exists() and inst_nc.stat().st_size > 0):
        retrieve_with_retry(client, req_inst, inst_nc)
    else:
        log(f'[CACHE] using {inst_nc.name}')
    if not (tp_nc.exists() and tp_nc.stat().st_size > 0):
        retrieve_with_retry(client, req_tp, tp_nc)
    else:
        log(f'[CACHE] using {tp_nc.name}')

    daily = build_daily_frames(inst_nc, tp_nc)
    stats = {'written': 0, 'skip_exists': 0}
    for init_date_value, g in daily.groupby('init_date', sort=True):
        st = save_one_init(g.copy(), init_date_value, overwrite=overwrite)
        stats[st] = stats.get(st, 0) + 1
    log(f'[MONTH_DONE] {month:02d} {stats}')


def main() -> int:
    args = parse_args()
    years = list(range(args.start_year, args.end_year + 1))
    months = parse_init_months(args.init_months)
    write_cdsapirc_from_env()
    client = cdsapi.Client()
    for month in months:
        process_month(client, years, month, overwrite=args.overwrite)
    log('[DONE] bulk baseline rebuild finished')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
