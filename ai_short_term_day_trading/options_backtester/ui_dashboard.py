import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import sys
from datetime import datetime, date

# 確保可以匯入 core.simulator
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from core.simulator import OptionsSimulator

st.set_page_config(page_title="AI 選擇權回測視覺化儀表板", layout="wide")

st.title("📈 AI 選擇權回測視覺化儀表板")
st.markdown("執行貼合盤中的真實交易邏輯 (`live_option_simulator_v2.py`)，並在歷史委託簿資料上驗證 AI 決策、動能防護網 (Safety Nets) 以及動態停損/停利 (BSM Bounds) 的效果。")

# 側邊欄設定
st.sidebar.header("⚙️ 回測設定")
# 改用 date_input 選擇日期
selected_date = st.sidebar.date_input("選擇回測日期", value=date(2026, 6, 11))
target_date_str = selected_date.strftime("%Y-%m-%d")

if st.sidebar.button("🚀 開始回測"):
    with st.spinner(f"正在執行 {target_date_str} 的高頻回測，請稍候..."):
        try:
            # 初始化並執行回測
            sim = OptionsSimulator(target_date_str)
            trade_log, df_intraday = sim.run_simulation()
            
            st.success("✅ 回測模擬完成！")
            
            if not df_intraday.empty:
                # 繪製台指期 (TXF) 走勢圖與進出場標記
                st.subheader(f"📊 {target_date_str} 台指期 (TXF) 盤中走勢與交易觸發點")
                
                fig = go.Figure()
                
                # 台指期走勢線
                fig.add_trace(go.Scatter(
                    x=df_intraday['time'],
                    y=df_intraday['Close'],
                    mode='lines',
                    name='台指期 (TXF) 收盤價',
                    line=dict(color='gray', width=1)
                ))
                
                # 標記進出場點
                if trade_log:
                    buy_times = []
                    buy_prices = []
                    sell_times = []
                    sell_prices = []
                    
                    for trade in trade_log:
                        # 將字串轉換為 datetime.time 格式
                        entry_t = pd.to_datetime(f"{target_date_str} {trade['entry_time']}").time()
                        
                        # 在 K 線資料中找到最接近進場時間的點來畫圖
                        idx_match = df_intraday[df_intraday['time'] >= entry_t].first_valid_index()
                        if idx_match is not None:
                            txf_price = df_intraday.loc[idx_match, 'Close']
                            if trade['type'] == 'Call':
                                buy_times.append(entry_t)
                                buy_prices.append(txf_price)
                            else:
                                sell_times.append(entry_t)
                                sell_prices.append(txf_price)
                                
                    if buy_times:
                        fig.add_trace(go.Scatter(
                            x=buy_times, y=buy_prices,
                            mode='markers', name='買進買權 (Buy Call)',
                            marker=dict(symbol='triangle-up', color='#00FF00', size=14, line=dict(width=2, color='white'))
                        ))
                    if sell_times:
                        fig.add_trace(go.Scatter(
                            x=sell_times, y=sell_prices,
                            mode='markers', name='買進賣權 (Buy Put)',
                            marker=dict(symbol='triangle-down', color='#FF0000', size=14, line=dict(width=2, color='white'))
                        ))
                
                fig.update_layout(
                    height=500, 
                    xaxis_title="盤中時間", 
                    yaxis_title="指數價格", 
                    template='plotly_dark',
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)
                
            # 顯示交易紀錄
            st.subheader("📝 交易明細紀錄 (Trade Log)")
            if trade_log:
                df_trades = pd.DataFrame(trade_log)
                
                # 欄位重新命名以符合繁體中文
                df_trades_display = df_trades.rename(columns={
                    'entry_time': '進場時間',
                    'exit_time': '出場時間',
                    'symbol': '合約代碼',
                    'type': '方向',
                    'entry_price': '進場價',
                    'exit_price': '出場價',
                    'pnl': '損益 (NT$)'
                })
                
                st.dataframe(df_trades_display.style.format({
                    '進場價': '{:.2f}',
                    '出場價': '{:.2f}',
                    '損益 (NT$)': '{:,.0f}'
                }))
                
                # 計算 KPI
                total_pnl = sum(t['pnl'] for t in trade_log)
                win_rate = sum(1 for t in trade_log if t['pnl'] > 0) / len(trade_log) * 100
                
                col1, col2, col3 = st.columns(3)
                col1.metric("總交易筆數", len(trade_log))
                col2.metric("勝率", f"{win_rate:.1f}%")
                col3.metric("總損益 (PnL)", f"NT$ {total_pnl:,.0f}")
                
            else:
                st.info("💡 今日無任何交易訊號觸發。")
                
            # 顯示 AI 特徵值參照表
            st.markdown("---")
            st.subheader("🧠 AI 模型特徵值監控 (Feature Reference)")
            st.markdown("下方顯示當日計算的所有技術特徵 (如 `vwap_bias`, `rsi_fast`, `macd` 等)，這些資料會即時餵給 Transformer 模型進行決策。您可以觀察在交易觸發時間點附近，特徵值的變化情形。")
            
            if not df_intraday.empty:
                # 把不需要的欄位隱藏，專注在特徵
                cols_to_hide = ['date', 'date_only', 'time_ms', 'date_str']
                display_cols = [c for c in df_intraday.columns if c not in cols_to_hide]
                
                # 重新排列：把時間放在第一欄
                if 'time' in display_cols:
                    display_cols.remove('time')
                    display_cols = ['time'] + display_cols
                    
                st.dataframe(df_intraday[display_cols])
            
        except Exception as e:
            st.error(f"❌ 執行回測時發生錯誤: {str(e)}")
