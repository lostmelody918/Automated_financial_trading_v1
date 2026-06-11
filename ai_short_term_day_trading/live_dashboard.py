import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time
import os
import json
from data_engine import DayTradingDataEngine

# streamlit run ai_short_term_day_trading/live_dashboard.py
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

@st.cache_data(ttl=3600) # 每小時快取一次
def load_chips_cache():
    chips_file = os.path.join(os.path.dirname(__file__), "chips_cache.json")
    if os.path.exists(chips_file):
        try:
            with open(chips_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return pd.DataFrame(data)
        except Exception as e:
            st.error(f"載入籌碼快照失敗: {e}")
    return pd.DataFrame()

# ==========================================
# 定義所有可用的特徵 (對應 data_engine)
# ==========================================
ALL_FEATURES = [
    'slope_vwap', 'macd_hist', 'vol_surge_ratio', 'atr', 'rsi',
    'vwap_bias', 'pv_divergence', 'is_squeeze', 'bb_width',
    'macd', 'signal', 'h_pc', 'l_pc', 'price_roc',
    'rsi_fast', 'body_length', 'upper_shadow', 'lower_shadow',
    'momentum_explosion', 'gap_amplitude', 'intraday_trend',
    'gap_filled', 'spot_futures_proxy', 'slope_ma20',
    'dist_from_ma20', 'pullback_from_high', 'bounce_from_low',
    'foreign_net_oi', 'dealer_net_oi', 'pc_ratio',
    'dealer_relative_momentum', 'dealer_extreme_score'
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

    if st.button("📥 重新抓取籌碼資料", help="從期交所/證交所更新最新籌碼快照"):
        with st.spinner("正在更新籌碼..."):
            from fetch_daily_chips import main as fetch_chips
            fetch_chips()
            st.cache_data.clear()
            st.success("籌碼資料已更新！")
            st.rerun()

    st.markdown("---")
    st.markdown("### 📊 顯示設定")
    display_bars = st.slider("圖表顯示 K 線數量", min_value=50, max_value=300, value=150, step=10)
    table_rows = st.slider("資料表顯示筆數", min_value=10, max_value=50, value=15, step=5)

    st.markdown("---")
    st.markdown("### 📈 自訂副圖指標 (支援動態增減與疊圖)")
    num_subplots = st.slider("選擇副圖數量", min_value=0, max_value=5, value=3, step=1)

    subplot_features_list = []
    default_feats = [['slope_vwap'], ['macd_hist'], ['vol_surge_ratio'], ['dist_from_ma20'], ['dealer_relative_momentum']]
    for i in range(num_subplots):
        default_val = default_feats[i] if i < len(default_feats) else []
        valid_default = [f for f in default_val if f in ALL_FEATURES]
        feats = st.multiselect(f"副圖 {i+1}", ALL_FEATURES, default=valid_default, key=f"sub_{i}")
        subplot_features_list.append(feats)

    st.markdown("---")
    with st.expander("💡 專業特徵解讀指南 (展開查看全部)", expanded=False):
        st.markdown("""
        **動能與趨勢類**
        - **`slope_vwap` / `slope_ma20` (均線斜率)**: 衡量趨勢速度。`slope_ma20` > 2 通常暗示單邊多頭趨勢(軋空/緩漲)，<-2 暗示空頭趨勢。
        - **`macd_hist` / `macd`**: MACD 與信號線差距。柱狀圖轉正為多頭動能增強，捕捉翻轉極佳。
        - **`price_roc` / `intraday_trend`**: 價格變動率與日內開盤以來的絕對趨勢，捕捉爆發力。

        **波動、乖離與均值回歸類**
        - **`vwap_bias` (VWAP 乖離率)**: 現價與均價的距離。過大正負值易引發反彈或回檔。
        - **`dist_from_ma20` (MA20 乖離)**: 價格與 20 MA 之間的距離，判斷極端拉回深度。
        - **`pullback_from_high` / `bounce_from_low`**: 衡量價格從近期高點的回撤幅度或低點反彈幅度。
        - **`rsi` / `rsi_fast`**: 評估超買(>70)或超賣(<30)。`rsi_fast` 更敏感，常用於捕捉微觀 V 轉與淺回檔。
        - **`atr`**: 真實波動幅度。急升代表趨勢發動或快市。

        **型態與特殊狀態類**
        - **`vol_surge_ratio` (爆量比例)**: >1.5 或 >1.8 暗示主力介入或停損盤爆發。
        - **`pv_divergence` (價量背離)**: 價格創高但量縮(-1)或破底量縮(1)，高價值反轉預警。
        - **`is_squeeze` (布林擠壓)**: 波動收斂至低點(1)，常為大行情前兆。
        - **`momentum_explosion` / `gap_amplitude`**: 實體 K 線爆發與跳空缺口幅度。
        - **`body_length` / `upper_shadow` / `lower_shadow`**: K 線實體與上下影線長度解析。

        **籌碼類 (法人動向)**
        - **`foreign_net_oi` / `dealer_net_oi`**: 外資與自營商期貨未平倉量。
        - **`dealer_relative_momentum`**: 自營商相對於外資的動能強弱(抓法人背離)。
        - **`dealer_extreme_score`**: 自營商異常大動作轉折預警 (-3 ~ +3)。
        - **`pc_ratio`**: 選擇權 P/C Ratio (判斷大戶莊家多空心態)。
        """)

# ==========================================
# 核心繪圖函數 (動態特徵渲染)
# ==========================================
def plot_dynamic_dashboard(df, subplot_features_list):
    df_plot = df.copy()
    # 確保按照時間嚴格排序，避免圖表時間軸亂跳
    if 'date' in df_plot.columns:
        df_plot = df_plot.sort_values('date', ascending=True).reset_index(drop=True)
        
    # 將 datetime 轉為字串，強制 Plotly 視為類別變數 (Category)，藉此消除盤後與假日的巨大空白時間軸
    df_plot['date_str'] = df_plot['date'].dt.strftime('%m/%d %H:%M')

    num_subs = len(subplot_features_list)
    rows = num_subs + 1

    # Calculate row heights dynamically
    main_height = 0.55 if num_subs > 0 else 1.0
    sub_height = 0.45 / num_subs if num_subs > 0 else 0
    row_heights = [main_height] + [sub_height] * num_subs if num_subs > 0 else [1.0]

    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights
    )

    # 1. 主圖：K線 + VWAP + 擠壓視覺化
    fig.add_trace(go.Candlestick(
        x=df_plot['date_str'], open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'],
        name='K線', increasing_line_color='#ef5350', decreasing_line_color='#26a69a'
    ), row=1, col=1)

    # 強制設定 x 軸為 category 並且按照 array 順序排列，防止 Plotly 自己亂排序字串
    fig.update_xaxes(type='category', categoryorder='array', categoryarray=df_plot['date_str'])

    fig.add_trace(go.Scatter(x=df_plot['date_str'], y=df_plot['vwap'], line=dict(color='#ff9800', width=2), name='VWAP'), row=1, col=1)
    if 'bb_upper' in df_plot.columns and 'bb_lower' in df_plot.columns:
        fig.add_trace(go.Scatter(x=df_plot['date_str'], y=df_plot['bb_upper'], line=dict(color='rgba(200,200,200,0.4)', width=1, dash='dot'), name='BB_Up'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_plot['date_str'], y=df_plot['bb_lower'], line=dict(color='rgba(200,200,200,0.4)', width=1, dash='dot'), name='BB_Low'), row=1, col=1)

    # 動態加入副圖的輔助函數
    def add_feature_trace(fig, feat_names, row_idx):
        if not feat_names:
            return

        colors_palette = ["#42a5f5", "#ab47bc", "#ffca28", "#66bb6a", "#ff7043"]

        # 定義哪些指標該用什麼圖表類型
        BAR_FEATURES = ['macd_hist', 'slope_vwap', 'vwap_bias', 'pv_divergence', 'price_roc', 'macd', 'dist_from_ma20', 'pullback_from_high', 'bounce_from_low', 'dealer_relative_momentum', 'dealer_extreme_score', 'intraday_trend', 'gap_amplitude']
        AREA_FEATURES = ['is_squeeze', 'momentum_explosion', 'gap_filled']

        for i, feat_name in enumerate(feat_names):
            if feat_name not in df_plot.columns:
                continue

            is_single = len(feat_names) == 1
            trace_color = colors_palette[i % len(colors_palette)]

            if feat_name in BAR_FEATURES:
                if is_single:
                    colors = ['#ef5350' if val > 0 else '#26a69a' for val in df_plot[feat_name]]
                    fig.add_trace(go.Bar(x=df_plot['date_str'], y=df_plot[feat_name], marker_color=colors, name=feat_name), row=row_idx, col=1)
                else:
                    fig.add_trace(go.Scatter(x=df_plot['date_str'], y=df_plot[feat_name], line=dict(width=2, color=trace_color), name=feat_name), row=row_idx, col=1)
                fig.add_hline(y=0, line_width=1, line_color="gray", line_dash="dash", row=row_idx, col=1)

            elif feat_name in AREA_FEATURES:
                fig.add_trace(go.Scatter(x=df_plot['date_str'], y=df_plot[feat_name], fill='tozeroy', fillcolor='rgba(255,235,59,0.3)', line=dict(color='#fbc02d', width=1), name=feat_name), row=row_idx, col=1)

            else:
                line_color = "#42a5f5" if is_single else trace_color
                fig.add_trace(go.Scatter(x=df_plot['date_str'], y=df_plot[feat_name], line=dict(width=2, color=line_color), name=feat_name), row=row_idx, col=1)

                if feat_name in ['rsi', 'rsi_fast']:
                    fig.add_hline(y=70, line_dash="dash", line_color="#ef5350", row=row_idx, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="#26a69a", row=row_idx, col=1)
                elif feat_name == 'vol_surge_ratio':
                    fig.add_hline(y=1.8, line_dash="dash", line_color="#ff9800", row=row_idx, col=1)

    # 繪製使用者選擇的副圖
    for i, feats in enumerate(subplot_features_list):
        add_feature_trace(fig, feats, i + 2)

    total_height = 450 + 150 * num_subs if num_subs > 0 else 600

    fig.update_layout(
        height=total_height,
        margin=dict(l=40, r=40, t=30, b=20),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode="x unified",
        spikedistance=-1,
        hoverdistance=-1,
        uirevision='constant'
    )

    # 確保開啟網格，依賴 hovermode='x unified' 自動產生貫穿全部圖表的垂直游標
    fig.update_xaxes(
        type='category', # 強制設定為分類軸，徹底消滅時間斷層
        nticks=10,       # 避免文字擠在一起
        showgrid=True, 
        gridwidth=1, 
        gridcolor='rgba(128,128,128,0.2)', 
        autorange=True, 
        fixedrange=False,
        # 強制開啟跨圖表十字游標
        showspikes=True, 
        spikemode="across", 
        spikesnap="cursor", 
        spikethickness=1, 
        spikedash="solid", 
        spikecolor="#888888"
    )
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)', autorange=True, fixedrange=False)

    fig.update_yaxes(title_text="價格", row=1, col=1)
    for i, feats in enumerate(subplot_features_list):
        fig.update_yaxes(title_text=", ".join(feats) if feats else "", row=i+2, col=1)

    return fig

# ==========================================
# 主程式邏輯 (資料更新與渲染)
# ==========================================
if engine is None:
    st.warning("請確認 Shioaji 權限與憑證。")
else:
    time_filter_placeholder = st.sidebar.empty()
    top_placeholder = st.empty()
    content_placeholder = st.empty()

    def update_dashboard():
        try:
            import os
            import json
            
            with st.spinner('📡 正在從 Shioaji 與期交所拉取最新資料...'):
                df_intraday = engine.fetch_intraday_data(days=2)

            if df_intraday is None or df_intraday.empty:
                st.warning("⚠️ 等待盤中資料或資料獲取失敗...")
                return

            # --- 載入籌碼快照並與 K 線資料融合 ---
            df_chips_daily = load_chips_cache()
                    
            if not df_chips_daily.empty and hasattr(engine, 'integrate_institutional_chips'):
                df = engine.integrate_institutional_chips(df_intraday, df_chips_daily)
            else:
                df = df_intraday

            # --- 0. 側邊欄：歷史區間過濾 (複盤用) ---
            with time_filter_placeholder.container():
                st.markdown("---")
                st.markdown("### 🕒 歷史區間過濾 (複盤用)")
                st.caption("啟用後將暫停自動跟隨最新 K 線，改為顯示指定時間段。")
                enable_time_filter = st.checkbox("啟用特定時間區間過濾", value=False)

                sel_time_range = None
                sel_date = None
                if enable_time_filter:
                    # 取出所有有資料的日期
                    unique_dates = sorted(df['date_only'].unique())
                    sel_date = st.selectbox("選擇複盤日期", unique_dates, index=len(unique_dates)-1)

                    df_date = df[df['date_only'] == sel_date]
                    if not df_date.empty:
                        min_t = df_date['time'].min()
                        max_t = df_date['time'].max()
                        sel_time_range = st.slider(
                            "選擇時間範圍",
                            min_value=min_t,
                            max_value=max_t,
                            value=(min_t, max_t),
                            format="HH:mm"
                        )
                    else:
                        st.warning("該日無資料")

            # 根據過濾器切出顯示資料
            if enable_time_filter and sel_time_range and sel_date:
                df_display = df[(df['date_only'] == sel_date) &
                                (df['time'] >= sel_time_range[0]) &
                                (df['time'] <= sel_time_range[1])].copy()
                if df_display.empty:
                    st.warning("所選時間範圍內無資料。")
                    return
            else:
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
                    fig = plot_dynamic_dashboard(df_display, subplot_features_list)
                    st.plotly_chart(fig, use_container_width=True, theme="streamlit")

                    st.markdown("---")
                    col_export1, col_export2 = st.columns([1, 2])

                    with col_export1:
                        if st.button("💾 手動匯出今日全日複盤資料", key="btn_manual_export", help="包含今天所有的 K 線特徵與全日互動圖表", use_container_width=True):
                            st.session_state.trigger_export = True

                    with col_export2:
                        from datetime import time as dt_time
                        is_after_market = datetime.now().time() >= dt_time(13, 45)

                        if "auto_exported_today" not in st.session_state:
                            st.session_state.auto_exported_today = False

                        if is_after_market and not st.session_state.auto_exported_today:
                            st.info("💡 盤後時段：系統正進行自動複盤資料匯出...")
                            st.session_state.trigger_export = True
                            st.session_state.auto_exported_today = True
                        elif is_after_market:
                            st.success("✅ 今日自動複盤資料已儲存完畢。")

                    if st.session_state.get('trigger_export', False):
                        import os
                        export_dir = "data_learn/daily_reports"
                        os.makedirs(export_dir, exist_ok=True)
                        today_str = datetime.now().strftime("%Y%m%d")

                        # 過濾今日數據
                        df_today = df[df['date'].dt.date == datetime.now().date()].copy()
                        if df_today.empty:
                            df_today = df.tail(300).copy() # fallback 取最後 300 根 (約一天)

                        csv_path = os.path.join(export_dir, f"EOD_Features_{today_str}.csv")
                        html_path = os.path.join(export_dir, f"EOD_Chart_{today_str}.html")

                        # 儲存 CSV
                        df_today.to_csv(csv_path, index=False, encoding='utf-8-sig')

                        # 產生全日圖表並儲存 HTML
                        fig_export = plot_dynamic_dashboard(df_today, subplot_features_list)
                        fig_export.update_layout(title=f"台指期 {today_str} 全日複盤特徵圖")
                        fig_export.write_html(html_path)

                        st.success(f"✅ 成功匯出全日複盤資料至:\n- {csv_path}\n- {html_path}")
                        st.session_state.trigger_export = False

                with tab2:
                    st.markdown(f"### 📋 全特徵即時數據 (最新 {table_rows} 根 K 線)")
                    st.markdown("💡 **使用提示**：本表利用熱力圖(Gradient)自動標示各特徵的高低點。紅色/深橘色通常代表極端值(過熱/超買)，綠色代表低值(超賣/冷卻)，幫助您一眼看出當前的特徵群聚狀態。支援欄位點擊排序與全螢幕檢視。")

                    # 整理要顯示的欄位
                    available_features = [f for f in ALL_FEATURES if f in df_display.columns]
                    display_cols = ['date', 'Close', 'Volume'] + available_features
                    df_table = df_display[display_cols].tail(table_rows).sort_values('date', ascending=False)

                    # 將索引重置，並將時間轉為字串
                    df_table['date'] = df_table['date'].dt.strftime('%H:%M:%S')
                    df_table.set_index('date', inplace=True)

                    # 將所有數值欄位都加上顏色漸層 (除了價格與成交量以保持畫面乾淨)
                    numeric_cols = df_table.select_dtypes(include=np.number).columns.tolist()
                    grad_cols = [c for c in numeric_cols if c not in ['Close', 'Volume', 'is_squeeze', 'momentum_explosion', 'gap_filled']]

                    styled_df = df_table.style.background_gradient(cmap='RdYlGn_r', subset=grad_cols).format("{:.3f}")

                    st.dataframe(styled_df, use_container_width=True, height=600)

                with tab3:
                    st.markdown("### 📚 歷史績效與複盤")
                    st.markdown("""
                    💡 **虧損分析使用提示**：
                    1. **勝率熱力圖**：找出您在哪個星期/時段勝率最高，藉此避開絞肉機時段。
                    2. **特徵比較矩陣**：比較「獲利單」與「虧損單」在進場當下的平均特徵。若某個特徵差距極大（請看「差異比例」），代表該特徵是區分勝敗的關鍵指標！
                    3. **深度虧損診斷**：自動掃描逆勢與高風險操作。
                    4. **SHAP 分析**：AI 黑盒子解密！它會告訴您是哪幾個特徵「促使 AI 做出虧損的決定」，幫助您修改策略防護網。
                    """)

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

                                st.markdown("#### 🤖 AI 策略建議 (Gemini)")
                                if st.button("🧠 請求 AI 歷史複盤建議"):
                                    with st.spinner("正在請 Gemini AI 進行診斷..."):
                                        try:
                                            import google.generativeai as genai
                                            from dotenv import load_dotenv
                                            load_dotenv()
                                            
                                            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GMINI_API")
                                            if not api_key:
                                                st.error("找不到 GEMINI_API_KEY 或 GMINI_API，請確認 .env 檔案設定。")
                                            else:
                                                genai.configure(api_key=api_key)
                                                model = genai.GenerativeModel('gemini-1.5-pro')
                                                
                                                # 準備傳送給 AI 的簡化數據
                                                summary_data = f"總交易次數: {len(df_hist)}, 勝率: {win_rate*100:.1f}%, 總盈虧: {total_pnl:,.0f}, MDD: {mdd:,.0f}, 獲利因子: {profit_factor:.2f}"
                                                if 'entry_time' in df_hist.columns and 'pnl' in df_hist.columns:
                                                    recent_trades = df_hist[['entry_time', 'pnl']].tail(10).to_string(index=False)
                                                else:
                                                    recent_trades = "無最近交易明細"
                                                    
                                                prompt = f"你是一個專業的量化交易教練。以下是我的當沖策略近期表現：\n{summary_data}\n\n最近10筆交易：\n{recent_trades}\n\n請根據以上數據，給我3到5點具體且簡短的改善建議或策略調整方向。"
                                                
                                                response = model.generate_content(prompt)
                                                st.success("✅ AI 診斷完成")
                                                st.write(response.text)
                                        except ImportError:
                                            st.error("請先安裝套件： pip install google-generativeai python-dotenv")
                                        except Exception as e:
                                            st.error(f"呼叫 AI 發生錯誤: {e}")

                                st.markdown("#### 🔬 進階特徵與虧損分析 (Advanced Feature & Loss Analysis)")
                                df_win = df_hist[df_hist['pnl'] > 0].copy()
                                df_loss = df_hist[df_hist['pnl'] < 0].copy()

                                # 將 feat_ 開頭的欄位提取出來比較
                                feature_cols = [c for c in df_hist.columns if c.startswith('feat_')]
                                if feature_cols:
                                    st.markdown("**特徵平均值比較 (Win vs Loss)**")

                                    # 計算平均值 (強制轉換為數值，避免字串欄位導致 TypeError)
                                    df_win_num = df_win[feature_cols].apply(pd.to_numeric, errors='coerce')
                                    df_loss_num = df_loss[feature_cols].apply(pd.to_numeric, errors='coerce')

                                    win_means = df_win_num.mean() if not df_win_num.empty else pd.Series(dtype=float)
                                    loss_means = df_loss_num.mean() if not df_loss_num.empty else pd.Series(dtype=float)

                                    df_compare = pd.DataFrame({
                                        '獲利單 (Win Avg)': win_means,
                                        '虧損單 (Loss Avg)': loss_means
                                    })

                                    # 整理名稱，去掉 feat_ 前綴
                                    df_compare.index = df_compare.index.str.replace('feat_', '')

                                    # 新增差異與關鍵程度分析
                                    df_compare['差距 (Diff)'] = df_compare['獲利單 (Win Avg)'] - df_compare['虧損單 (Loss Avg)']
                                    df_compare['差異比例 (%)'] = (df_compare['差距 (Diff)'].abs() / (df_compare['虧損單 (Loss Avg)'].abs() + 1e-9)) * 100

                                    # 依據差異比例排序，讓最關鍵的特徵浮到最上面
                                    df_compare = df_compare.sort_values(by='差異比例 (%)', ascending=False)

                                    # 特徵敏感度標籤
                                    df_compare['指標關鍵度'] = np.where(df_compare['差異比例 (%)'] > 100, '⭐⭐⭐ 極高',
                                                                 np.where(df_compare['差異比例 (%)'] > 50, '⭐⭐ 高',
                                                                 np.where(df_compare['差異比例 (%)'] > 20, '⭐ 中', '低')))

                                    # 利用背景色高亮差異大的特徵
                                    styled_compare = df_compare.style.background_gradient(cmap='bwr', subset=['差距 (Diff)']).format({
                                        "獲利單 (Win Avg)": "{:.4f}",
                                        "虧損單 (Loss Avg)": "{:.4f}",
                                        "差距 (Diff)": "{:.4f}",
                                        "差異比例 (%)": "{:.1f}%"
                                    })

                                    st.dataframe(styled_compare, use_container_width=True, height=400)

                                    st.markdown("**🚨 深度虧損診斷 (Deep Loss Diagnostics)**")
                                    if not df_loss.empty:
                                        c1, c2, c3 = st.columns(3)

                                        # 逆勢交易分析: 做多但 MACD < 0，做空但 MACD > 0
                                        if 'direction' in df_loss.columns and 'feat_macd_hist' in df_loss.columns:
                                            dir_lower = df_loss['direction'].fillna('').str.lower()
                                            is_long = dir_lower.str.contains('long') | dir_lower.str.contains('call')
                                            is_short = dir_lower.str.contains('short') | dir_lower.str.contains('put')

                                            counter_trend = sum((is_long & (df_loss['feat_macd_hist'] < 0)) | (is_short & (df_loss['feat_macd_hist'] > 0)))
                                            c1.metric("逆勢 MACD 虧損筆數", f"{counter_trend} / {len(df_loss)}", delta="-高風險", delta_color="inverse")

                                        # 高波動分析: ATR 大於全體中位數
                                        if 'feat_atr' in df_loss.columns and 'feat_atr' in df_hist.columns:
                                            median_atr = df_hist['feat_atr'].median()
                                            high_vol = sum(df_loss['feat_atr'] > median_atr)
                                            c2.metric("高波動 (ATR超中位數) 虧損筆數", f"{high_vol} / {len(df_loss)}")

                                        # 爆量虧損分析: vol_surge_ratio > 1.5
                                        if 'feat_vol_surge_ratio' in df_loss.columns:
                                            surge_loss = sum(df_loss['feat_vol_surge_ratio'] > 1.5)
                                            c3.metric("爆量進場 (Surge>1.5) 虧損筆數", f"{surge_loss} / {len(df_loss)}")
                                else:
                                    st.info("歷史紀錄中無特徵資料 (feat_*)，請累積更多新版模擬器交易紀錄以啟用此功能。")

                                st.markdown("---")
                                st.markdown("#### 🔍 SHAP 虧損深度歸因分析")
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
