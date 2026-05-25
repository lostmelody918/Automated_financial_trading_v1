import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_fetcher import DataFetcher
from database import DatabaseManager
from bt_setup import run_backtest, SmaCross, MacdStrategy, ForeignBuyStrategy, RsiStrategy
from analysis import KLineAnalyzer, KDAnalyzer, SentimentAnalyzer

st.set_page_config(page_title="Taiwan Financial Trading System", layout="wide")

st.title("📈 台灣金融市場分析與回測系統 (Taiwan Financial Market Analysis)")

# Sidebar for inputs
with st.sidebar:
    st.header("設定 (Settings)")
    target_id = st.text_input("股票代號或大盤 (Stock ID / Index)", value="TAIEX")
    st.caption("提示: 輸入 TAIEX 可抓取台股大盤資料")
    start_date = st.date_input("開始日期 (Start Date)", value=date(2020, 1, 1))
    end_date = st.date_input("結束日期 (End Date)", value=date.today())
    
    st.markdown("---")
    if st.button("📥 抓取最新資料 (Fetch Data)", use_container_width=True):
        with st.spinner(f"正在抓取 {target_id} 的資料..."):
            fetcher = DataFetcher()
            fetcher.fetch_stock_daily(target_id, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            st.success(f"✅ {target_id} 資料抓取完成！")

    st.markdown("---")
    st.header("回測設定 (Backtest Settings)")
    strategy_name = st.selectbox(
        "選擇策略 (Strategy)",
        ("SmaCross (雙均線)", "MacdStrategy (MACD交叉)", "ForeignBuyStrategy (外資連續買超)", "RsiStrategy (RSI 強弱指標)")
    )
    initial_cash = st.number_input("初始資金 (Initial Cash)", value=1000000.0, step=100000.0)
    run_bt_btn = st.button("🚀 執行回測 (Run Backtest)", use_container_width=True)

# Main Content Area - Tabs for Pagination
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📊 歷史數據", 
    "🕯️ KD指標分析", 
    "🧬 產業相關性",
    "💰 基本面/營業比重",
    "🎟️ 籌碼面分析",
    "📰 新聞輿情",
    "📈 回測結果", 
    "🤖 AI 預測"
])

# Load data for display
db = DatabaseManager()
try:
    df_raw = db.load_dataframe(f"stock_{target_id}_daily")
    if not df_raw.empty:
        df_raw['date'] = pd.to_datetime(df_raw['date'])
        df = df_raw.sort_values('date', ascending=False)
    else:
        df = None
except Exception:
    df = None

with tab1:
    st.subheader(f"{target_id} 歷史資料預覽")
    if df is not None:
        st.dataframe(df.head(100), use_container_width=True)
    else:
        st.info(f"尚未找到 {target_id} 的本地資料，請先在左側點擊「抓取最新資料」。")

with tab2:
    st.subheader(f"🕯️ {target_id} 技術指標與 KD 線型態分析")
    if df_raw is not None and not df_raw.empty:
        # 1. 準備數據
        analysis_df = df_raw.sort_values('date').copy()
        analysis_df = KLineAnalyzer.add_indicators(analysis_df)
        kd_patterns = KDAnalyzer.analyze_patterns(analysis_df)
        
        # 取最近 150 根 K 線顯示
        plot_df = analysis_df.tail(150)
        
        # 2. 顯示 KD 型態分析結果
        col_lt, col_rt = st.columns([0.7, 0.3])
        with col_rt:
            st.info("🕒 **KD 型態辨識報告**")
            if kd_patterns:
                for p in kd_patterns:
                    st.write(p)
            else:
                st.write("目前無明顯 KD 指標訊號。")
            
            with st.expander("📚 查看 KD 技術原理"):
                st.markdown(KDAnalyzer.get_principles())
            
        with col_lt:
            # 3. 使用 Plotly 繪圖 (三層圖：K線、成交量、KD)
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                               vertical_spacing=0.03, subplot_titles=(f'{target_id} K線與均線', '成交量', 'KD 指標'), 
                               row_width=[0.2, 0.2, 0.6])

            # Row 1: Candlestick & MA
            fig.add_trace(go.Candlestick(
                x=plot_df['date'], open=plot_df['Open'], high=plot_df['High'],
                low=plot_df['Low'], close=plot_df['Close'], name='K線'
            ), row=1, col=1)
            fig.add_trace(go.Scatter(x=plot_df['date'], y=plot_df['MA5'], name='MA5', line=dict(width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=plot_df['date'], y=plot_df['MA20'], name='MA20', line=dict(width=1)), row=1, col=1)

            # Row 2: Volume
            colors = ['red' if row['Close'] >= row['Open'] else 'green' for _, row in plot_df.iterrows()]
            fig.add_trace(go.Bar(x=plot_df['date'], y=plot_df['Volume'], name='成交量', marker_color=colors), row=2, col=1)

            # Row 3: KD
            fig.add_trace(go.Scatter(x=plot_df['date'], y=plot_df['K'], name='K 值 (快)', line=dict(color='blue', width=1.5)), row=3, col=1)
            fig.add_trace(go.Scatter(x=plot_df['date'], y=plot_df['D'], name='D 值 (慢)', line=dict(color='orange', width=1.5)), row=3, col=1)
            # 加 20/80 基準線
            fig.add_shape(type="line", x0=plot_df['date'].iloc[0], y0=80, x1=plot_df['date'].iloc[-1], y1=80, 
                          line=dict(color="red", width=1, dash="dash"), row=3, col=1)
            fig.add_shape(type="line", x0=plot_df['date'].iloc[0], y0=20, x1=plot_df['date'].iloc[-1], y1=20, 
                          line=dict(color="green", width=1, dash="dash"), row=3, col=1)

            fig.update_layout(height=800, xaxis_rangeslider_visible=False, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("尚未找到本地資料。")

with tab3:
    st.subheader("🧬 產業相關性分析 (Correlation Analysis)")
    industry_list = db.load_dataframe("taiwan_stock_info")
    if not industry_list.empty:
        selected_industry = st.selectbox("選擇產業類別", sorted(industry_list['industry_category'].unique()))
        # 取得該產業股票，並確保當前選擇的股票也在清單內
        stocks_in_ind = industry_list[industry_list['industry_category'] == selected_industry]['stock_id'].tolist()[:8]
        if target_id not in stocks_in_ind and target_id != "TAIEX":
            stocks_in_ind = [target_id] + stocks_in_ind
        
        if st.button("執行相關性比較"):
            with st.spinner("正在計算..."):
                fetcher = DataFetcher()
                df_corr_raw = fetcher.fetch_industry_prices(stocks_in_ind, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
                if not df_corr_raw.empty:
                    corr = df_corr_raw.corr()
                    
                    # 1. 熱圖
                    fig_corr = go.Figure(data=go.Heatmap(
                        z=corr.values, x=corr.index, y=corr.columns,
                        colorscale='RdBu', zmin=-1, zmax=1
                    ))
                    fig_corr.update_layout(title=f"{selected_industry} 與 {target_id} 相關性熱圖", height=500)
                    st.plotly_chart(fig_corr, use_container_width=True)
                    
                    # 2. 數值表格
                    st.write("### 相關性數值矩陣")
                    st.dataframe(corr.style.background_gradient(cmap='RdBu', axis=None).format("{:.2f}"), use_container_width=True)
                    
                    st.info("""
                    💡 **如何看相關性表格？**
                    - 表格中的數值代表兩檔股票走勢的相似度。
                    - **紅色/正值越大**: 走勢越一致（適合產業趨勢分析）。
                    - **藍色/負值越大**: 走勢相反。
                    """)
                else:
                    st.error("無法取得該產業資料。")

with tab4:
    st.subheader("💰 股票基本面與獲利來源")
    if target_id != "TAIEX":
        fetch_btn = st.button("🔍 抓取最新獲利來源數據")
        
        # 嘗試從資料庫讀取現有資料
        df_fin_db = db.load_dataframe(f"stock_{target_id}_financials")
        
        if fetch_btn:
            with st.spinner("正在抓取深度資料..."):
                fetcher = DataFetcher()
                df_rev = fetcher.fetch_revenue_breakdown(target_id)
                fin_data = fetcher.fetch_stock_financials(target_id, start_date.strftime("%Y-%m-%d"))
                df_fin = fin_data.get("financials")
                df_month = fetcher.fetch_monthly_revenue(target_id, start_date.strftime("%Y-%m-%d"))
        else:
            df_fin = df_fin_db
            df_rev = pd.DataFrame() # 營業比重目前不進資料庫
            df_month = pd.DataFrame()

        if not df_fin.empty or not df_rev.empty:
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                st.write("### 營收與利潤趨勢 (季報)")
                if not df_fin.empty:
                    df_fin['date'] = pd.to_datetime(df_fin['date'])
                    df_pivot = df_fin.pivot(index='date', columns='type', values='value')
                    # 統一名稱以便顯示
                    if 'IncomeAfterTaxes' in df_pivot.columns:
                        df_pivot.rename(columns={'IncomeAfterTaxes': 'NetIncome'}, inplace=True)
                    
                    plot_types = [c for c in ['Revenue', 'GrossProfit', 'NetIncome'] if c in df_pivot.columns]
                    st.line_chart(df_pivot[plot_types])
                else:
                    st.warning("暫無最新財報趨勢資料。")
                
                st.write("### 月營收趨勢")
                if not df_month.empty:
                    df_month['date'] = pd.to_datetime(df_month['date'])
                    st.bar_chart(df_month.set_index('date')['revenue'])
                else:
                    st.info("點擊「抓取」按鈕以獲取月營收資料。")
            
            with col_f2:
                st.write("### 營業比重 (真實獲利來源)")
                if not df_rev.empty:
                    # 嘗試抓取 Shioaji 的營業比重欄位 (通常是 'revenue_name' 與 'revenue_percentage')
                    label_col = 'revenue_name' if 'revenue_name' in df_rev.columns else (df_rev.columns[2] if len(df_rev.columns) > 2 else "")
                    val_col = 'revenue_percentage' if 'revenue_percentage' in df_rev.columns else (df_rev.columns[3] if len(df_rev.columns) > 3 else "")
                    
                    if label_col and val_col and label_col in df_rev.columns:
                        fig_pie = go.Figure(data=[go.Pie(labels=df_rev[label_col], values=df_rev[val_col], hole=.3)])
                        st.plotly_chart(fig_pie, use_container_width=True)
                        st.dataframe(df_rev[[label_col, val_col]], use_container_width=True)
                    else:
                        st.warning("無法自動解析營業比重格式，顯示原始資料：")
                        st.dataframe(df_rev)
                else:
                    st.error("⚠️ 無法自動獲取該股營業比重資料。")
                    st.info(f"""
                    由於數據源限制，建議手動查看：
                    - [Yahoo 股市 - {target_id} 公司資料](https://tw.stock.yahoo.com/quote/{target_id}/profile)
                    - [鉅亨網 - {target_id} 公司簡介](https://invest.cnyes.com/twstock/TWS/{target_id}/company/profile)
                    """)
            
            st.write("### 損益表詳細數據")
            if not df_fin.empty:
                st.dataframe(df_fin.head(20), use_container_width=True)
        else:
            st.info("請點擊「抓取最新獲利來源數據」以獲取資料。")
    else:
        st.info("大盤指數無基本面資料。")

with tab5:
    st.subheader("🎟️ 籌碼面分析 (Chip Analysis)")
    if target_id != "TAIEX":
        if st.button("📊 抓取籌碼面數據"):
            fetcher = DataFetcher()
            chip_data = fetcher.fetch_stock_chips(target_id, start_date.strftime("%Y-%m-%d"))
            df_margin = chip_data.get("margin")
            
            if not df_margin.empty:
                st.write("### 融資融券變化")
                fig_margin = make_subplots(specs=[[{"secondary_y": True}]])
                fig_margin.add_trace(go.Bar(x=df_margin['date'], y=df_margin['MarginPurchaseBuy'], name='融資買進'), secondary_y=False)
                fig_margin.add_trace(go.Scatter(x=df_margin['date'], y=df_margin['MarginPurchaseLimit'], name='融資餘額', line=dict(color='red')), secondary_y=True)
                st.plotly_chart(fig_margin, use_container_width=True)
                
                st.info("💡 **籌碼冷知識**：融資餘額飆高通常代表散戶進場，若股價不漲反跌，需注意『資增價跌』的殺融資風險。")
            else:
                st.warning("查無籌碼面資料。")
    else:
        st.info("大盤指數籌碼通常參考整體三大法人買賣超。")

with tab6:
    st.subheader(f"📰 {target_id} 相關新聞與輿情分析")
    if target_id != "TAIEX":
        if st.button("🔍 抓取並分析最新新聞"):
            with st.spinner("正在獲取新聞..."):
                fetcher = DataFetcher()
                news_df = fetcher.fetch_stock_news(target_id, (date.today() - pd.Timedelta(days=30)).strftime("%Y-%m-%d"))
                if not news_df.empty:
                    avg_sentiment, analyzed_df = SentimentAnalyzer.analyze_sentiment(news_df)
                    
                    col_s1, col_s2 = st.columns([0.3, 0.7])
                    with col_s1:
                        st.metric("平均輿情分數", f"{avg_sentiment:.2f}")
                        if avg_sentiment > 0.1:
                            st.success("🚀 偏向正面 (Bullish)")
                        elif avg_sentiment < -0.1:
                            st.error("📉 偏向負面 (Bearish)")
                        else:
                            st.warning("😐 中立 (Neutral)")
                        
                        st.info("""
                        💡 **情緒計算邏輯**：
                        對新聞標題進行關鍵詞匹配。
                        - **正面**: 獲利、成長、創新高、買超...
                        - **負面**: 虧損、衰退、看淡、賣超...
                        """)
                    
                    with col_s2:
                        # 顯示情緒分布
                        sentiment_counts = analyzed_df['sentiment_score'].value_counts().sort_index()
                        fig_sent = go.Figure(data=[go.Bar(x=sentiment_counts.index, y=sentiment_counts.values, marker_color='indianred')])
                        fig_sent.update_layout(title="新聞情緒得分分布", xaxis_title="得分", yaxis_title="新聞篇數", template="plotly_white")
                        st.plotly_chart(fig_sent, use_container_width=True)

                    st.write("### 最新新聞清單 (近 30 天)")
                    st.dataframe(analyzed_df[['date', 'title', 'source', 'sentiment_score', 'link']], use_container_width=True)
                else:
                    st.warning("目前查無相關新聞。")
    else:
        st.info("大盤指數新聞功能暫未整合。")

# Handle backtest execution
if run_bt_btn:
    strategies_map = {
        "SmaCross (雙均線)": SmaCross,
        "MacdStrategy (MACD交叉)": MacdStrategy,
        "ForeignBuyStrategy (外資連續買超)": ForeignBuyStrategy,
        "RsiStrategy (RSI 強弱指標)": RsiStrategy
    }
    
    selected_strat = strategies_map[strategy_name]
    
    with st.spinner("正在執行回測與繪圖..."):
        results = run_backtest(target_id, strategy=selected_strat, cash=initial_cash, plot=True)
        
        if "error" in results:
            st.error(results["error"])
        else:
            with tab7:
                st.subheader("回測績效指標 (Performance Metrics)")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("初始資金", f"${results['initial_value']:,.2f}")
                col2.metric("最終資產", f"${results['final_value']:,.2f}", f"{results['return_pct']:.2f}%")
                
                sharpe = results['sharpe']
                sharpe_str = f"{sharpe:.2f}" if isinstance(sharpe, float) else "N/A"
                col3.metric("夏普值 (Sharpe)", sharpe_str)
                
                dd = results['max_drawdown']
                dd_str = f"-{dd:.2f}%" if isinstance(dd, float) else "N/A"
                col4.metric("最大回落 (Max Drawdown)", dd_str)

                st.markdown("---")
                
                # 互動式回測圖表 (使用 Plotly)
                st.subheader("📈 互動式回測圖表 (支援縮放)")
                bt_df = df_raw.sort_values('date')
                fig_bt = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                      vertical_spacing=0.03, subplot_titles=('價格與策略區間', '成交量'), 
                                      row_width=[0.3, 0.7])
                
                fig_bt.add_trace(go.Candlestick(
                    x=bt_df['date'], open=bt_df['Open'], high=bt_df['High'],
                    low=bt_df['Low'], close=bt_df['Close'], name='價格'
                ), row=1, col=1)
                
                fig_bt.add_trace(go.Bar(x=bt_df['date'], y=bt_df['Volume'], name='成交量'), row=2, col=1)
                
                fig_bt.update_layout(height=600, xaxis_rangeslider_visible=True, template="plotly_white")
                st.plotly_chart(fig_bt, use_container_width=True)

                st.markdown("---")
                col5, col6, col7 = st.columns(3)
                col5.metric("總交易次數", f"{results['total_trades']}")
                col6.metric("勝率", f"{results['win_rate']:.2f}%")
                col7.metric("SQN", f"{results['sqn']:.2f}")
                
                st.success("回測執行成功！")
                
                if results.get("plot_file"):
                    with st.expander("查看靜態原始分析圖 (Matplotlib)"):
                        st.image(results["plot_file"])

with tab8:
    st.subheader("🤖 AI 股價預測 (線性回歸模型)")
    if df is not None and not df.empty:
        # Prepare data for prediction
        predict_df = df_raw.sort_values('date').copy()
        predict_df['Next_Close'] = predict_df['Close'].shift(-1)
        
        # Features: Use past 5 days of Close prices
        for i in range(1, 6):
            predict_df[f'Close_Lag_{i}'] = predict_df['Close'].shift(i)
        
        predict_df.dropna(inplace=True)
        
        features = [f'Close_Lag_{i}' for i in range(1, 6)]
        X = predict_df[features]
        y = predict_df['Next_Close']
        
        # Train on all but the last 30 days, test on the last 30 days
        split_idx = int(len(predict_df) * 0.9)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Display Results
        res_df = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred}, index=predict_df.index[split_idx:])
        st.line_chart(res_df)
        
        # Predict Tomorrow
        latest_features = predict_df['Close'].tail(5).values[::-1].reshape(1, -1)
        tomorrow_pred = model.predict(latest_features)[0]
        
        st.write(f"### 預測下一交易日收盤價: **{tomorrow_pred:.2f}**")
        
        current_price = predict_df['Close'].iloc[-1]
        diff = tomorrow_pred - current_price
        st.write(f"當前收盤價: {current_price:.2f} | 預測漲跌: {diff:+.2f} ({diff/current_price*100:+.2f}%)")
        
    else:
        st.info("尚未找到本地資料。")
