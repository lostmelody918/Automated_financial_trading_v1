import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time
from data_engine import DayTradingDataEngine

# ==========================================
# 頁面基礎設定
# ==========================================
st.set_page_config(page_title="AI 全特徵實盤監控", layout="wide", page_icon="👁️", initial_sidebar_state="expanded")
st.title("👁️ 台指期 AI 全特徵動態監控站 (Omni-Feature)")

# ==========================================
# 初始化 DataEngine (快取)
# ==========================================
@st.cache_resource
def get_data_engine():
    try:
        return DayTradingDataEngine()
    except Exception as e:
        st.error(f"無法連線至 Shioaji API: {e}")
        return None

engine = get_data_engine()

# ==========================================
# 定義所有可用的特徵 (對應 data_engine)
# ==========================================
ALL_FEATURES = [
    'slope_vwap', 'macd_hist', 'vol_surge_ratio', 'atr', 'rsi', 
    'vwap_bias', 'pv_divergence', 'is_squeeze', 'bb_width', 
    'macd', 'signal', 'h_pc', 'l_pc', 'price_roc'
]

# ==========================================
# 側邊欄：控制面板
# ==========================================
with st.sidebar:
    st.header("⚙️ 視覺化控制面板")
    
    st.markdown("### 🔄 更新設定")
    col1, col2 = st.columns([1, 1])
    with col1:
        auto_refresh = st.checkbox("自動更新", value=True, help="啟用後將自動刷新資料")
    with col2:
        if st.button("🚀 立即更新", use_container_width=True):
            st.rerun()
            
    if auto_refresh:
        refresh_interval = st.number_input("更新頻率 (秒)", min_value=5, max_value=300, value=30, step=5, help="設定最短 5 秒的自動更新頻率")
    else:
        refresh_interval = 30 # fallback

    st.markdown("---")
    st.markdown("### 📊 顯示設定")
    display_bars = st.slider("圖表顯示 K 線數量", min_value=50, max_value=300, value=150, step=10)
    table_rows = st.slider("資料表顯示筆數", min_value=10, max_value=50, value=15, step=5)

    st.markdown("---")
    st.markdown("### 📈 自訂副圖指標 (支援疊圖)")
    st.caption("選擇要在 K 線下方顯示的特徵曲線 (可複選以疊圖)：")
    sub1_feat = st.multiselect("副圖一", ALL_FEATURES, default=['slope_vwap'])
    sub2_feat = st.multiselect("副圖二", ALL_FEATURES, default=['macd_hist'])
    sub3_feat = st.multiselect("副圖三", ALL_FEATURES, default=['vol_surge_ratio'])

    st.markdown("---")
    with st.expander("💡 專業特徵解讀指南 (展開查看全部)", expanded=False):
        st.markdown("""
        **動能與趨勢類**
        - **`slope_vwap` (VWAP 斜率)**: 衡量均價線的變動率。正值代表多頭資金推升，負值代表空頭壓制，斜率陡峭反映動能強勁。
        - **`macd_hist` (MACD 柱狀圖)**: MACD 與信號線之差。柱狀圖轉正(紅)視為多頭動能增強，轉負(綠)為空頭增強，捕捉翻轉極佳。
        - **`macd` & `signal`**: MACD 快線與慢線，趨勢判斷的基礎核心。
        - **`price_roc` (價格變動率)**: 衡量價格變化的加速度，捕捉趨勢爆發力。

        **波動與極端偏離類**
        - **`vwap_bias` (VWAP 乖離率)**: 現價與 VWAP 的距離。過大正乖離易引發獲利了結，過大負乖離易引發空單回補的反彈。
        - **`rsi` (相對強弱指標)**: 評估超買(>70)或超賣(<30)。在當沖中用於尋找極端偏離的逆勢反轉點。
        - **`atr` (真實波動幅度)**: 衡量市場波動率。數值急升代表「快市」或趨勢發動；低迷代表盤整。
        
        **型態與籌碼狀態類**
        - **`vol_surge_ratio` (爆量比例)**: 當前量與均量比值。>1.5 或 >1.8 通常暗示主力介入或停損盤引發的極端情緒，為變盤前兆。
        - **`pv_divergence` (價量背離)**: 價格創高但量縮(-1 頂背離)，或價格破底但量縮(1 底背離)，為極具價值的反轉預警。
        - **`is_squeeze` (布林擠壓)**: 布林寬度縮窄至歷史低點 (狀態值 1)。「靜如處子，動如脫兔」，擠壓後的突破往往伴隨大行情。
        - **`bb_width` (布林通道寬度)**: 反映市場波動收斂與發散的循環週期。
        - **`h_pc` / `l_pc`**: 當前價格距離近期高/低點的相對位置，反映潛在壓力與支撐。
        """)

# ==========================================
# 核心繪圖函數 (動態特徵渲染)
# ==========================================
def plot_dynamic_dashboard(df, feat1, feat2, feat3):
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.52, 0.16, 0.16, 0.16]
    )

    # 1. 主圖：K線 + VWAP + 擠壓視覺化
    fig.add_trace(go.Candlestick(
        x=df['date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='K線', increasing_line_color='#ef5350', decreasing_line_color='#26a69a'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df['date'], y=df['vwap'], line=dict(color='#ff9800', width=2), name='VWAP'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['bb_upper'], line=dict(color='rgba(200,200,200,0.4)', width=1, dash='dot'), name='BB_Up'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['bb_lower'], line=dict(color='rgba(200,200,200,0.4)', width=1, dash='dot'), name='BB_Low'), row=1, col=1)

    # 動態加入副圖的輔助函數
    def add_feature_trace(fig, feat_names, row_idx):
        if not feat_names:
            return
            
        colors_palette = ["#42a5f5", "#ab47bc", "#ffca28", "#66bb6a", "#ff7043"]

        for i, feat_name in enumerate(feat_names):
            if feat_name not in df.columns:
                continue

            # 若只有單一指標且屬於柱狀圖類，用紅綠色；否則為了疊圖清晰，轉為統一顏色的折線圖或柱圖
            is_single = len(feat_names) == 1
            trace_color = colors_palette[i % len(colors_palette)]

            if feat_name in ['macd_hist', 'slope_vwap', 'vwap_bias', 'pv_divergence', 'price_roc', 'macd']:
                if is_single:
                    colors = ['#ef5350' if val > 0 else '#26a69a' for val in df[feat_name]]
                    fig.add_trace(go.Bar(x=df['date'], y=df[feat_name], marker_color=colors, name=feat_name), row=row_idx, col=1)
                else:
                    # 疊圖時為了避免互相遮擋，改用折線圖
                    fig.add_trace(go.Scatter(x=df['date'], y=df[feat_name], line=dict(width=2, color=trace_color), name=feat_name), row=row_idx, col=1)
                fig.add_hline(y=0, line_width=1, line_color="gray", line_dash="dash", row=row_idx, col=1)
            
            # 針對狀態類使用面積圖
            elif feat_name == 'is_squeeze':
                fig.add_trace(go.Scatter(x=df['date'], y=df[feat_name], fill='tozeroy', fillcolor='rgba(255,235,59,0.3)', line=dict(color='#fbc02d'), name=feat_name), row=row_idx, col=1)
            
            # 其他使用折線圖
            else:
                line_color = "#42a5f5" if is_single else trace_color
                fig.add_trace(go.Scatter(x=df['date'], y=df[feat_name], line=dict(width=2, color=line_color), name=feat_name), row=row_idx, col=1)
                
                # 特定指標的警戒線
                if feat_name == 'rsi':
                    fig.add_hline(y=70, line_dash="dash", line_color="#ef5350", row=row_idx, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="#26a69a", row=row_idx, col=1)
                elif feat_name == 'vol_surge_ratio':
                    fig.add_hline(y=1.8, line_dash="dash", line_color="#ff9800", row=row_idx, col=1)

    # 繪製使用者選擇的三個副圖
    add_feature_trace(fig, feat1, 2)
    add_feature_trace(fig, feat2, 3)
    add_feature_trace(fig, feat3, 4)

    fig.update_layout(
        height=850, 
        margin=dict(l=40, r=40, t=30, b=20), 
        showlegend=True, 
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode="x unified",
        uirevision='constant' # 核心關鍵：保持重新渲染時的縮放比例與位置
    )
    # 加入 autorange=True 與 fixedrange=False，確保未來加入履約價選擇權時，切換合約能動態調整 X/Y 軸價格區間，避免比例失真
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)', autorange=True, fixedrange=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)', autorange=True, fixedrange=False)
    fig.update_yaxes(title_text="價格", row=1, col=1)
    fig.update_yaxes(title_text=", ".join(feat1) if feat1 else "", row=2, col=1)
    fig.update_yaxes(title_text=", ".join(feat2) if feat2 else "", row=3, col=1)
    fig.update_yaxes(title_text=", ".join(feat3) if feat3 else "", row=4, col=1)
    return fig

# ==========================================
# 主程式邏輯 (資料更新與渲染)
# ==========================================
if engine is None:
    st.warning("請確認 Shioaji 權限與憑證。")
else:
    top_placeholder = st.empty()
    content_placeholder = st.empty()

    def update_dashboard():
        try:
            with st.spinner('📡 正在從 Shioaji 與期交所拉取最新資料...'):
                df = engine.fetch_intraday_data(days=5)
                
            if df is None or df.empty:
                st.warning("⚠️ 等待盤中資料或資料獲取失敗...")
                return

            df_display = df.tail(display_bars).copy()
            last_row = df_display.iloc[-1]
            
            # 抓取大盤現貨指數 (TAIEX) 以供比對
            try:
                tse_snap = engine.api.snapshots([engine.api.Contracts.Indexs.TSE.TSE001])[0]
                spot_price = tse_snap.close
                basis = last_row['Close'] - spot_price
                spot_str = f"{spot_price:,.0f} (價差 {basis:+.0f})"
            except:
                spot_str = "載入中..."

            # --- 1. 渲染頂部狀態列 ---
            with top_placeholder.container():
                st.markdown(f"### 📍 標的: TXF (台指期) | 🕒 最新資料時間: **{last_row['date'].strftime('%Y-%m-%d %H:%M:%S')}**")
                
                # 計算漲跌
                prev_close = df_display.iloc[-2]['Close'] if len(df_display) > 1 else last_row['Close']
                diff = last_row['Close'] - prev_close
                diff_str = f"+{diff:.0f}" if diff > 0 else f"{diff:.0f}"
                
                c1, c1_spot, c2, c3, c4, c5 = st.columns(6)
                c1.metric("期貨指數 (TXF)", f"{last_row['Close']:,.0f}", diff_str)
                c1_spot.metric("大盤現貨 (TAIEX)", spot_str, help="您上網查到的通常是這個大盤加權指數。期貨與現貨之間會有正逆價差。")
                c2.metric("VWAP", f"{last_row['vwap']:,.0f}")
                c3.metric("RSI (動能)", f"{last_row['rsi']:.1f}")
                c4.metric("ATR (波動)", f"{last_row['atr']:.1f}")
                
                vol_status = f"{last_row['vol_surge_ratio']:.1f}x"
                if last_row['is_squeeze'] == 1: vol_status += " (擠壓中 ⚠️)"
                c5.metric("爆量/狀態", vol_status)

            # --- 2. 渲染內容區塊 (圖表與數據切換) ---
            with content_placeholder.container():
                tab1, tab2, tab3 = st.tabs(["📈 即時動態圖表", "🧩 特徵數據矩陣", "📚 歷史績效與複盤"])
                
                with tab1:
                    fig = plot_dynamic_dashboard(df_display, sub1_feat, sub2_feat, sub3_feat)
                    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
                
                with tab2:
                    st.markdown(f"### 📋 全特徵即時數據 (最新 {table_rows} 根 K 線)")
                    st.caption("觀察所有 AI 判斷依據。支援欄位點擊排序與全螢幕檢視。")
                    
                    # 整理要顯示的欄位
                    available_features = [f for f in ALL_FEATURES if f in df_display.columns]
                    display_cols = ['date', 'Close', 'Volume'] + available_features
                    df_table = df_display[display_cols].tail(table_rows).sort_values('date', ascending=False)
                    
                    # 將索引重置，並將時間轉為字串
                    df_table['date'] = df_table['date'].dt.strftime('%H:%M:%S')
                    df_table.set_index('date', inplace=True)
                    
                    # 判斷要上色的欄位是否存在
                    grad_cols_1 = [c for c in ['slope_vwap', 'macd_hist', 'vwap_bias', 'price_roc'] if c in df_table.columns]
                    grad_cols_2 = [c for c in ['vol_surge_ratio', 'atr', 'bb_width'] if c in df_table.columns]
                    
                    styled_df = df_table.style.background_gradient(cmap='RdYlGn_r', subset=grad_cols_1) \
                                              .background_gradient(cmap='Oranges', subset=grad_cols_2) \
                                              .format("{:.3f}")
                    
                    st.dataframe(styled_df, use_container_width=True, height=600)
                
                with tab3:
                    st.markdown("### 📚 歷史績效與複盤")
                    import glob
                    import os
                    import json
                    
                    base_path = "data_learn"
                    if not os.path.exists(base_path):
                        base_path = "../data_learn"
                        if not os.path.exists(base_path):
                            base_path = "F:/Gemini_CLI_Application/finance_v2/data_learn"
                    
                    report_files = glob.glob(os.path.join(base_path, "daily_trade_report_*.csv"))
                    if not report_files:
                        st.info("尚未找到任何歷史交易紀錄 (daily_trade_report_*.csv)。")
                    else:
                        dfs = []
                        for f in report_files:
                            try:
                                dfs.append(pd.read_csv(f))
                            except Exception:
                                pass
                        
                        if dfs:
                            df_hist = pd.concat(dfs, ignore_index=True)
                            if 'pnl' in df_hist.columns:
                                total_pnl = df_hist['pnl'].sum()
                                max_dd = df_hist['pnl'].cumsum().cummax() - df_hist['pnl'].cumsum()
                                mdd = max_dd.max() if not max_dd.empty else 0
                                
                                gross_profit = df_hist[df_hist['pnl'] > 0]['pnl'].sum()
                                gross_loss = abs(df_hist[df_hist['pnl'] < 0]['pnl'].sum())
                                profit_factor = (gross_profit / gross_loss) if gross_loss != 0 else float('inf')
                                
                                avg_win = df_hist[df_hist['pnl'] > 0]['pnl'].mean()
                                avg_loss = abs(df_hist[df_hist['pnl'] < 0]['pnl'].mean())
                                risk_reward = (avg_win / avg_loss) if avg_loss != 0 else float('inf')
                                
                                win_rate = len(df_hist[df_hist['pnl'] > 0]) / len(df_hist) if len(df_hist) > 0 else 0
                                
                                hc1, hc2, hc3, hc4 = st.columns(4)
                                hc1.metric("Maximum Drawdown (MDD)", f"{mdd:,.0f}")
                                hc2.metric("Profit Factor", f"{profit_factor:.2f}")
                                hc3.metric("Risk-Reward Ratio", f"{risk_reward:.2f}")
                                hc4.metric("Win Rate", f"{win_rate*100:.1f}%")
                                
                                st.markdown("#### 📅 勝率熱力圖 (星期 vs 小時)")
                                if 'entry_time' in df_hist.columns:
                                    df_hist['entry_time'] = pd.to_datetime(df_hist['entry_time'], errors='coerce')
                                    df_hist_valid = df_hist.dropna(subset=['entry_time']).copy()
                                    if not df_hist_valid.empty:
                                        df_hist_valid['hour'] = df_hist_valid['entry_time'].dt.hour
                                        df_hist_valid['day_of_week'] = df_hist_valid['entry_time'].dt.dayofweek
                                        df_hist_valid['is_win'] = (df_hist_valid['pnl'] > 0).astype(int)
                                        
                                        pivot = df_hist_valid.pivot_table(index='day_of_week', columns='hour', values='is_win', aggfunc='mean')
                                        day_names = {0:'一', 1:'二', 2:'三', 3:'四', 4:'五', 5:'六', 6:'日'}
                                        pivot.index = pivot.index.map(day_names)
                                        
                                        fig_hm = go.Figure(data=go.Heatmap(
                                            z=pivot.values,
                                            x=pivot.columns,
                                            y=pivot.index,
                                            colorscale='RdYlGn',
                                            zmin=0, zmax=1
                                        ))
                                        fig_hm.update_layout(title='各時段勝率分布', xaxis_title='小時', yaxis_title='星期')
                                        st.plotly_chart(fig_hm, use_container_width=True)
                                
                                st.markdown("#### 🔍 SHAP 虧損深度分析")
                                col_btn1, col_btn2 = st.columns(2)
                                
                                do_shap = False
                                target_dir = None
                                
                                if col_btn1.button("執行多單虧損 SHAP 分析"):
                                    do_shap = True
                                    target_dir = 'long'
                                if col_btn2.button("執行空單虧損 SHAP 分析"):
                                    do_shap = True
                                    target_dir = 'short'
                                    
                                if do_shap:
                                    import torch
                                    import shap
                                    import matplotlib.pyplot as plt
                                    import sys
                                    
                                    curr_dir = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
                                    if curr_dir not in sys.path:
                                        sys.path.append(curr_dir)
                                        
                                    try:
                                        from composite_ai import CompositeDayTradingAI
                                    except ImportError:
                                        sys.path.append("F:/Gemini_CLI_Application/finance_v2/ai_short_term_day_trading")
                                        from composite_ai import CompositeDayTradingAI
                                    
                                    with st.spinner(f"正在執行 {target_dir} SHAP 分析..."):
                                        dir_series = df_hist['direction'].fillna('').str.lower()
                                        if target_dir == 'long':
                                            mask_dir = dir_series.str.contains('long') | dir_series.str.contains('call')
                                        else:
                                            mask_dir = dir_series.str.contains('short') | dir_series.str.contains('put')
                                            
                                        df_loss = df_hist[(df_hist['pnl'] < 0) & mask_dir].copy()
                                        
                                        if len(df_loss) == 0:
                                            st.warning(f"沒有找到符合條件的 {target_dir} 虧損紀錄。")
                                        else:
                                            try:
                                                model_dir = "saved_models"
                                                if not os.path.exists(model_dir):
                                                    model_dir = "../saved_models"
                                                    if not os.path.exists(model_dir):
                                                        model_dir = "F:/Gemini_CLI_Application/finance_v2/ai_short_term_day_trading/saved_models"
                                                
                                                with open(os.path.join(model_dir, "norm_params.json"), "r", encoding='utf-8') as f:
                                                    norm_params = json.load(f)
                                                    
                                                feature_cols = [c for c in norm_params['feature_cols'] if c in norm_params['mean']]
                                                
                                                model_files = glob.glob(os.path.join(model_dir, "trading_model_*.pth"))
                                                if not model_files:
                                                    st.error("找不到任何模型檔 (.pth)")
                                                else:
                                                    latest_model_path = max(model_files, key=os.path.getctime)
                                                    
                                                    # 讀取對應的 metadata 來取得 hyperparameter
                                                    meta_path = latest_model_path.replace('.pth', '_metadata.json')
                                                    window_size = 40
                                                    d_model = 256
                                                    nhead = 16
                                                    num_layers = 4
                                                    
                                                    if os.path.exists(meta_path):
                                                        with open(meta_path, 'r', encoding='utf-8') as f:
                                                            meta = json.load(f)
                                                            if "experiment_info" in meta and "hyperparameters" in meta["experiment_info"]:
                                                                hp = meta["experiment_info"]["hyperparameters"]
                                                                window_size = hp.get('window_size', 40)
                                                                d_model = hp.get('d_model', 256)
                                                                nhead = hp.get('nhead', 16)
                                                                num_layers = hp.get('num_layers', 4)
                                                    
                                                    ai_model = CompositeDayTradingAI(input_dim=len(feature_cols), d_model=d_model, nhead=nhead, num_layers=num_layers)
                                                    checkpoint = torch.load(latest_model_path, map_location='cpu', weights_only=True)
                                                    ai_model.load_state_dict(checkpoint['model_state_dict'])
                                                    ai_model.eval()
                                                    
                                                    df_loss = df_loss.head(100)
                                                    X_list = []
                                                    for idx, row in df_loss.iterrows():
                                                        row_feat = []
                                                        for col in feature_cols:
                                                            feat_name = f"feat_{col}"
                                                            val = float(row[feat_name]) if feat_name in row and pd.notnull(row[feat_name]) else 0.0
                                                            row_feat.append(val)
                                                        X_list.append([row_feat] * window_size)
                                                    
                                                    X_tensor = torch.tensor(X_list, dtype=torch.float32)
                                                    background = torch.zeros((1, window_size, len(feature_cols)), dtype=torch.float32)
                                                    
                                                    explainer = shap.GradientExplainer(ai_model, background)
                                                    shap_values = explainer.shap_values(X_tensor)
                                                    
                                                    if isinstance(shap_values, list):
                                                        shap_vals = shap_values[0]
                                                    else:
                                                        shap_vals = shap_values
                                                        
                                                    shap_vals_last_step = shap_vals[:, -1, :]
                                                    X_tensor_last_step = X_tensor[:, -1, :].numpy()
                                                    
                                                    fig_shap = plt.figure()
                                                    shap.summary_plot(shap_vals_last_step, X_tensor_last_step, feature_names=feature_cols, show=False)
                                                    st.pyplot(fig_shap)
                                                    plt.close(fig_shap)
                                                    
                                            except Exception as e:
                                                st.error(f"SHAP 分析過程發生錯誤: {e}")

        except Exception as e:
            st.error(f"❌ 資料更新發生錯誤: {str(e)}")
            st.exception(e)

    # 執行更新
    update_dashboard()

    # 自動更新邏輯
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()
