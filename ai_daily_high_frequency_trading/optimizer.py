import optuna
import pandas as pd
import numpy as np
from hft_data import fetch_hft_data
from hft_main import run_hft_vectorized
import os

def objective(trial, df_is):
    account_risk_pct = trial.suggest_float('account_risk_pct', 0.01, 0.05)
    sl_pct = trial.suggest_float('sl_pct', -0.9, -0.4)
    risk_pct = min(account_risk_pct / abs(sl_pct), 0.15)

    params = {
        'risk_pct': risk_pct,
        'base_delta': trial.suggest_int('base_delta', 50, 200),
        'gamma': trial.suggest_int('gamma', 1000, 8000),
        'vega_sens': trial.suggest_int('vega_sens', 10, 150),
        'use_vwap': trial.suggest_categorical('use_vwap', [True, False]),
        'tsl_atr': trial.suggest_float('tsl_atr', 5.0, 15.0),
        'bbw_z': trial.suggest_float('bbw_z', -2.5, -0.5),
        'hma_slope_min': trial.suggest_float('hma_slope_min', 0.0001, 0.002),
        'sl_pct': sl_pct,
        'slippage': 0.03,
        'fee': 0.0005,
        'max_trade_capital': 1000000.0
    }

    ret, df_result = run_hft_vectorized(df_is, params)

    curve_array = df_result['capital'].values
    if len(curve_array) < 2:
        return -999.0

    # Prevent division by zero
    curve_array_safe = np.where(curve_array <= 0, 1e-9, curve_array)
    returns = np.diff(curve_array_safe) / curve_array_safe[:-1]

    # Clean up any remaining inf/nan
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

    annual_factor = 252 * 5
    annual_ret = ret * (annual_factor / max(len(returns), 1))

    running_max = np.maximum.accumulate(curve_array_safe)
    drawdowns = (running_max - curve_array_safe) / running_max
    max_dd = np.max(drawdowns)

    total_trades = np.count_nonzero(returns)

    # Require at least 20 trades in the IS window (~6 months) to avoid overfitting to corner cases
    if max_dd > 0.60 or annual_ret <= 0 or total_trades < 20:
        raise optuna.exceptions.TrialPruned()

    volatility = np.std(returns) * np.sqrt(annual_factor)
    sharpe = annual_ret / (volatility + 1e-9)

    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0:
        downside_vol = np.std(downside_returns) * np.sqrt(annual_factor)
    else:
        downside_vol = 1e-9

    sortino = annual_ret / (downside_vol + 1e-9)
    calmar = annual_ret / (max_dd + 1e-9)

    score = (0.01 * sharpe) + (0.4 * sortino) + (0.59 * calmar)

    if np.isnan(score) or np.isinf(score):
        return -999.0

    return score

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
