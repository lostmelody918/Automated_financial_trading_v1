import optuna
import pandas as pd
import numpy as np
from hft_data import fetch_hft_data
from hft_main import run_hft_vectorized
import os

def objective(trial, df_is):
    params = {
        'risk_pct': trial.suggest_float('risk_pct', 0.05, 0.2),
        'base_delta': trial.suggest_int('base_delta', 50, 200),
        'gamma': trial.suggest_int('gamma', 1000, 8000),
        'vega_sens': trial.suggest_int('vega_sens', 10, 150),
        'use_vwap': trial.suggest_categorical('use_vwap', [True, False]),
        'tsl_atr': trial.suggest_float('tsl_atr', 5.0, 15.0),
        'bbw_p': trial.suggest_float('bbw_p', 0.1, 0.35),
        'hma_slope_min': trial.suggest_float('hma_slope_min', 0.0001, 0.002),
        'sl_pct': trial.suggest_float('sl_pct', -0.9, -0.6),
        'slippage': 0.03,
        'fee': 0.0005
    }
    
    ret, _ = run_hft_vectorized(df_is, params)
    return ret

def run_multi_year_cv():
    print("🚀 Fetching multi-year data for Deep Cross-Validation...")
    df = fetch_hft_data("^TWII", period="max", interval="1h")
    if df.empty: return

    # Focus on 2023-2026 range
    df = df[df['date'].dt.year >= 2023].copy()
    print(f"Total bars from 2023: {len(df)}")

    bars_per_day = 5
    is_len = 120 * bars_per_day
    oos_len = 60 * bars_per_day
    step = 90 * bars_per_day

    results = []
    
    for start_idx in range(0, len(df) - (is_len + oos_len), step):
        df_is = df.iloc[start_idx : start_idx + is_len]
        df_oos = df.iloc[start_idx + is_len : start_idx + is_len + oos_len]
        
        is_start = df_is['date'].iloc[0].date()
        oos_end = df_oos['date'].iloc[-1].date()
        print(f"\n--- Testing Window: {is_start} to {oos_end} ---")
        
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, df_is), n_trials=100)
        
        best_p = study.best_params
        best_p['slippage'] = 0.03
        best_p['fee'] = 0.0005
        
        ret_oos, _ = run_hft_vectorized(df_oos, best_p)
        print(f"IS Best: {study.best_value*100:.2f}% | OOS Result: {ret_oos*100:.2f}%")
        
        results.append({
            'is_start': is_start,
            'oos_end': oos_end,
            'is_ret': study.best_value,
            'oos_ret': ret_oos,
            'params': best_p
        })

    df_results = pd.DataFrame(results)
    print("\n📊 Multi-Year Stability Report:")
    print(df_results[['is_start', 'oos_end', 'is_ret', 'oos_ret']])
    
    avg_oos = df_results['oos_ret'].mean()
    print(f"\nAverage OOS Return: {avg_oos*100:.2f}%")
    
    if avg_oos >= 8.0:
        print("🎯 TARGET REACHED: HFT Model with Vega & VWAP is robust!")
    else:
        print("💡 Target not reached. Refining search space...")

if __name__ == '__main__':
    run_multi_year_cv()
