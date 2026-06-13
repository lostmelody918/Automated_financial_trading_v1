import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
from datetime import datetime, date, time

# 確保可以匯入 core.simulator
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from core.simulator import OptionsSimulator

st.set_page_config(page_title="AI 選擇權回測視覺化儀表板", layout="wide")

st.title("📈 AI 選擇權回測視覺化儀表板")
st.markdown("執行貼合盤中的真實交易邏輯 (`live_option_simulator_v2.py`)，並在歷史委託簿資料上驗證 AI 決策、動能防護網 (Safety Nets) 以及動態停損/停利 (BSM Bounds) 的效果。")

# --- 側邊欄設定 ---
st.sidebar.header("⚙️ 回測設定")
selected_date = st.sidebar.date_input("選擇回測日期", value=date(2026, 6, 11))
target_date_str = selected_date.strftime("%Y-%m-%d")

# Session State 初始化
if "run_complete" not in st.session_state:
    st.session_state.run_complete = False

if st.sidebar.button("🚀 開始載入與回測"):
    with st.spinner(f"正在執行 {target_date_str} 的高頻回測與資料載入，請稍候..."):
        try:
            # 初始化並執行回測
            sim = OptionsSimulator(target_date_str)
            trade_log, df_intraday, top_contracts, df_ticks = sim.run_simulation()
            
            # 儲存至 session state 以便後續互動
            st.session_state.trade_log = trade_log
            st.session_state.df_intraday = df_intraday
            st.session_state.top_contracts = top_contracts
            st.session_state.df_ticks = df_ticks
            st.session_state.run_complete = True
            
            st.success("✅ 回測模擬與資料載入完成！")
        except Exception as e:
            st.error(f"❌ 執行回測時發生錯誤: {str(e)}")

# --- 互動式 UI 區塊 ---
if st.session_state.run_complete:
    df_intraday = st.session_state.df_intraday
    trade_log = st.session_state.trade_log
    top_contracts = st.session_state.top_contracts
    df_ticks = st.session_state.df_ticks

    if not df_intraday.empty:
        # 1. 繪製台指期與特徵值疊加圖
        st.markdown("---")
        st.subheader(f"📊 {target_date_str} 台指期 (TXF) 與特徵值對比圖")
        
        # 特徵選擇器
        cols_to_hide = ['date', 'time', 'date_only', 'time_ms', 'date_str', 'Open', 'High', 'Low', 'Close']
        available_features = [c for c in df_intraday.columns if c not in cols_to_hide]
        selected_features = st.multiselect(
            "選擇要疊加顯示的特徵值 (顯示於右側 Y 軸)", 
            options=available_features,
            default=['vwap_bias', 'rsi_fast'] if 'vwap_bias' in available_features else []
        )
        
        # 使用雙 Y 軸
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # 台指期走勢線
        fig.add_trace(go.Scatter(
            x=df_intraday['time'],
            y=df_intraday['Close'],
            mode='lines',
            name='台指期 (TXF) 收盤價',
            line=dict(color='gray', width=2)
        ), secondary_y=False)
        
        # 加入選定的特徵
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        for i, feat in enumerate(selected_features):
            fig.add_trace(go.Scatter(
                x=df_intraday['time'],
                y=df_intraday[feat],
                mode='lines',
                name=feat,
                line=dict(width=1, dash='dot', color=colors[i % len(colors)])
            ), secondary_y=True)
            
        # 標記實際回測的進出場點
        if trade_log:
            buy_times = []
            buy_prices = []
            sell_times = []
            sell_prices = []
            
            for trade in trade_log:
                entry_t = pd.to_datetime(f"{target_date_str} {trade['entry_time']}").time()
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
                    mode='markers', name='策略買權 (Buy Call)',
                    marker=dict(symbol='triangle-up', color='#00FF00', size=14, line=dict(width=2, color='white'))
                ), secondary_y=False)
            if sell_times:
                fig.add_trace(go.Scatter(
                    x=sell_times, y=sell_prices,
                    mode='markers', name='策略賣權 (Buy Put)',
                    marker=dict(symbol='triangle-down', color='#FF0000', size=14, line=dict(width=2, color='white'))
                ), secondary_y=False)
        
        fig.update_layout(
            height=600, 
            xaxis_title="盤中時間", 
            template='plotly_dark',
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_yaxes(title_text="指數價格", secondary_y=False)
        fig.update_yaxes(title_text="特徵數值", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # 2. 顯示交易紀錄
        st.markdown("---")
        st.subheader("📝 實際策略交易明細紀錄 (Trade Log)")
        if trade_log:
            df_trades = pd.DataFrame(trade_log)
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
            
            total_pnl = sum(t['pnl'] for t in trade_log)
            win_rate = sum(1 for t in trade_log if t['pnl'] > 0) / len(trade_log) * 100
            
            col1, col2, col3 = st.columns(3)
            col1.metric("策略總交易筆數", len(trade_log))
            col2.metric("策略勝率", f"{win_rate:.1f}%")
            col3.metric("策略總損益 (PnL)", f"NT$ {total_pnl:,.0f}")
        else:
            st.info("💡 今日策略無任何交易訊號觸發。")
            
        # 3. 顯示當日 Top 4 合約
        st.markdown("---")
        st.subheader("🏆 當日主力選擇權合約 (Top 4 Volume)")
        st.markdown("系統自動篩選出今日多方與空方總交易量最高的前四名合約。")
        col_c, col_p = st.columns(2)
        with col_c:
            st.write("📈 **多方 (Call) 前 4 名**")
            for c in top_contracts.get('calls', []):
                st.code(c)
        with col_p:
            st.write("📉 **空方 (Put) 前 4 名**")
            for p in top_contracts.get('puts', []):
                st.code(p)
                
        # 4. 互動式自訂交易模擬
        st.markdown("---")
        st.subheader("🕹️ 自訂交易模擬與特徵觀察 (Interactive Manual Simulator)")
        st.markdown("您可以從上方主力合約中選擇一個，並自行設定買入與賣出時間，觀察 PnL 變化以及在該時刻的 AI 特徵狀態。")
        
        all_options = top_contracts.get('calls', []) + top_contracts.get('puts', [])
        if all_options and not df_ticks.empty:
            sel_contract = st.selectbox("選擇要模擬的選擇權合約", options=all_options)
            
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                manual_entry = st.time_input("設定進場時間", value=time(9, 0))
            with col_t2:
                manual_exit = st.time_input("設定出場時間", value=time(9, 30))
                
            if manual_entry >= manual_exit:
                st.warning("⚠️ 進場時間必須早於出場時間！")
            else:
                # 篩選該合約的 tick
                df_sym_ticks = df_ticks[df_ticks['symbol'] == sel_contract].copy()
                
                if df_sym_ticks.empty:
                    st.warning(f"⚠️ 找不到合約 {sel_contract} 的 Tick 資料。")
                else:
                    df_sym_ticks['time_dt'] = pd.to_datetime(df_sym_ticks['time'])
                    df_sym_ticks['time_only'] = df_sym_ticks['time_dt'].dt.time
                    
                    # 尋找進出場價格 (使用買賣權報價模擬：買進看 Ask，賣出看 Bid)
                    entry_mask = df_sym_ticks[df_sym_ticks['time_only'] >= manual_entry]
                    exit_mask = df_sym_ticks[df_sym_ticks['time_only'] >= manual_exit]
                    
                    if entry_mask.empty or exit_mask.empty:
                        st.warning("⚠️ 所選時間範圍內無交易資料。")
                    else:
                        entry_row = entry_mask.iloc[0]
                        exit_row = exit_mask.iloc[0]
                        
                        m_entry_price = entry_row['ask_price'] if entry_row['ask_price'] > 0 else entry_row['price']
                        m_exit_price = exit_row['bid_price'] if exit_row['bid_price'] > 0 else exit_row['price']
                        
                        m_pnl = (m_exit_price - m_entry_price) * 50
                        
                        # 畫出合約走勢與模擬進出場
                        fig_opt = go.Figure()
                        fig_opt.add_trace(go.Scatter(
                            x=df_sym_ticks['time'],
                            y=df_sym_ticks['price'],
                            mode='lines',
                            name=f'{sel_contract} 成交價',
                            line=dict(color='#17BECF')
                        ))
                        # 進場點
                        fig_opt.add_trace(go.Scatter(
                            x=[entry_row['time']], y=[m_entry_price],
                            mode='markers', name='手動買進',
                            marker=dict(symbol='star', color='#00FF00', size=16, line=dict(width=1, color='white'))
                        ))
                        # 出場點
                        fig_opt.add_trace(go.Scatter(
                            x=[exit_row['time']], y=[m_exit_price],
                            mode='markers', name='手動賣出',
                            marker=dict(symbol='x', color='#FF0000', size=14, line=dict(width=1, color='white'))
                        ))
                        fig_opt.update_layout(height=400, template='plotly_dark', title=f"{sel_contract} 合約走勢與手動模擬", hovermode="x unified")
                        st.plotly_chart(fig_opt, use_container_width=True)
                        
                        st.success(f"**模擬結果**：買進價 {m_entry_price:.1f} ➡️ 賣出價 {m_exit_price:.1f} │ 模擬損益：NT$ {m_pnl:,.0f}")
                        
                        # 特徵對比
                        st.markdown("#### 🔍 進出場時的 AI 特徵狀態對比")
                        
                        # 在 intraday 中找到最接近的 K 線
                        idx_in = df_intraday[df_intraday['time'] >= manual_entry].first_valid_index()
                        idx_out = df_intraday[df_intraday['time'] >= manual_exit].first_valid_index()
                        
                        if idx_in is not None and idx_out is not None:
                            feat_in = df_intraday.loc[idx_in]
                            feat_out = df_intraday.loc[idx_out]
                            
                            compare_df = pd.DataFrame({
                                '特徵名稱': available_features,
                                '進場時數值': [feat_in[c] for c in available_features],
                                '出場時數值': [feat_out[c] for c in available_features]
                            })
                            # 計算差異
                            compare_df['變化量'] = compare_df['出場時數值'] - compare_df['進場時數值']
                            st.dataframe(compare_df.style.format({'進場時數值': '{:.4f}', '出場時數值': '{:.4f}', '變化量': '{:+.4f}'}), use_container_width=True)
                        else:
                            st.info("無法找到對應時間的 K 線特徵。")

        # 5. 顯示完整 AI 特徵值參照表
        st.markdown("---")
        st.subheader("🧠 完整 AI 模型特徵值監控 (All Feature Reference)")
        display_cols = ['time'] + available_features
        st.dataframe(df_intraday[display_cols], use_container_width=True)
