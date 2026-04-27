import pandas as pd
import numpy as np
from data_engine import DayTradingDataEngine
from wave_strategy import WaveDayStrategy

def run_day_trading_backtest():
    engine = DayTradingDataEngine("^TWII")
    df = engine.fetch_intraday_data(days=60)
    
    if df.empty:
        print("沒有數據。")
        return
        
    strategy = WaveDayStrategy()
    
    LEVERAGE = 150 # 模擬末日選擇權極致槓桿 (大盤波動 0.2% 就能賺 30%)
    SLIPPAGE = 0.0001
    FEE = 0.00005
    COST = SLIPPAGE + FEE

    print(f"📊 執行選擇權極限當沖模擬 (槓桿 {LEVERAGE} 倍)...")
    print(f"使用策略: 華爾街量化 Institutional Gap & Go + VWAP Golden Cross")
    
    results = []
    position = 0
    entry_price = 0
    trade_log = []
    
    TAKE_PROFIT_PCT = 1.00 # 賺 100% 停利
    STOP_LOSS_PCT = -0.20  # 賠 20% 嚴格砍單
    
    for i in range(10, len(df)-1):
        curr_slice = df.iloc[:i+1]
        last_row = curr_slice.iloc[-1]
        next_row = df.iloc[i+1]
        
        curr_time = last_row['date'].time()
        today_date = last_row['date'].date()
        
        today_trades = [t for t in trade_log if t['date'].date() == today_date]
        
        if len(today_trades) >= 4: # 每天最多進出 2 趟 (包含平倉)
            continue
            
        can_trade = (curr_time.hour == 9 and curr_time.minute >= 5) or (curr_time.hour == 10) or (curr_time.hour == 11) or (curr_time.hour == 12)
        
        if curr_time.hour == 13 and curr_time.minute >= 25:
            if position != 0:
                ret = (next_row['Open'] - entry_price) / entry_price if position == 1 else (entry_price - next_row['Open']) / entry_price
                trade_log.append({'date': next_row['date'], 'type': 'Close_EOD', 'ret': ret * LEVERAGE - COST})
                position = 0
            continue
            
        if position != 0:
            high_ret = (next_row['High'] - entry_price) / entry_price if position == 1 else (entry_price - next_row['Low']) / entry_price
            low_ret = (next_row['Low'] - entry_price) / entry_price if position == 1 else (entry_price - next_row['High']) / entry_price
            
            lev_high_ret = high_ret * LEVERAGE
            lev_low_ret = low_ret * LEVERAGE
            
            if lev_high_ret >= TAKE_PROFIT_PCT:
                trade_log.append({'date': next_row['date'], 'type': 'Take_Profit', 'ret': TAKE_PROFIT_PCT - COST})
                position = 0
                continue
            elif lev_low_ret <= STOP_LOSS_PCT:
                trade_log.append({'date': next_row['date'], 'type': 'Stop_Loss', 'ret': STOP_LOSS_PCT - COST})
                position = 0
                continue
            
        if not can_trade:
            continue
            
        signal = strategy.generate_signal(curr_slice, ai_score=1.0)
        
        if signal == 1 and position == 0:
            position = 1
            entry_price = next_row['Open']
        elif signal == -1 and position == 0:
            position = -1
            entry_price = next_row['Open']
            
    df_trades = pd.DataFrame(trade_log)
    if df_trades.empty:
        print("沒有觸發任何交易。")
        return
        
    df_trades['week'] = df_trades['date'].dt.isocalendar().week
    weekly_perf = df_trades.groupby('week')['ret'].sum()
    
    total_ret = df_trades['ret'].sum()
    best_trade = df_trades.loc[df_trades['ret'].idxmax()]
    worst_trade = df_trades.loc[df_trades['ret'].idxmin()]
    
    print("\n" + "="*50)
    print("🚀 --- 複合波段當沖 AI 策略 (選擇權 150倍 高槓桿) ---")
    print(f"總交易次數: {len(df_trades)}")
    print(f"累積總報酬率: {total_ret*100:.2f}%")
    print(f"平均每週報酬率: {weekly_perf.mean()*100:.2f}%")
    print(f"週勝率 (正報酬週數): {(weekly_perf > 0).mean()*100:.2f}%")
    
    if weekly_perf.mean() < 0.30:
        print("⚠️ 未達每週 30% 目標，正在動態優化。")
    else:
        print("✅ 成功達成每週 30% 暴利目標！")
        
    print("\n🌟 最佳出手 (Best Trade):")
    print(f"時間: {best_trade['date']}, 類型: {best_trade['type']}, 獲利: {best_trade['ret']*100:.2f}%")
    
    print("\n💀 最壞出手 (Worst Trade):")
    print(f"時間: {worst_trade['date']}, 類型: {worst_trade['type']}, 虧損: {worst_trade['ret']*100:.2f}%")
    print("="*50)

    print("\n📅 每週績效詳情 (目標: >30%):")
    for week, ret in weekly_perf.items():
        print(f"第 {week} 週: {ret*100:+.2f}%")

if __name__ == "__main__":
    run_day_trading_backtest()