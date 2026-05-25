import sys
import traceback
import pandas as pd
from datetime import datetime, timedelta
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QComboBox, QPushButton, QDateEdit,
                             QFormLayout, QGroupBox, QMessageBox, QTextEdit, QCheckBox, QTabWidget)
from PyQt6.QtCore import QDate, Qt, QTimer
from PyQt6.QtGui import QPixmap

from bt_setup import run_backtest, SmaCross, MacdStrategy, ForeignBuyStrategy, RsiStrategy
from data_fetcher import DataFetcher
from database import DatabaseManager

from analysis import KLineAnalyzer, KDAnalyzer, MomentumAnalyzer, SentimentAnalyzer
from get_new import AdvancedSentimentAnalyzer

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("全球金融市場分析監控系統 (Taiwan & US Stocks)")
        self.setGeometry(100, 100, 1500, 950)

        self.db = DatabaseManager()
        self.fetcher = DataFetcher()
        self.stock_info_df = self.load_stock_info()

        # 即時監控定時器
        self.rt_timer = QTimer()
        self.rt_timer.timeout.connect(self.update_realtime)

        # 策略描述內容
        self.strat_info = {
            "SmaCross (雙均線)": "【雙均線策略】\n快線(10)突破慢線(30)買入，跌破則平倉賣出。",
            "MacdStrategy (MACD)": "【MACD 策略】\nMACD線突破訊號線買入，低於訊號線時平倉。",
            "ForeignBuyStrategy (外資)": "【外資策略】\n外資連續兩日買超則跟進，外資一轉賣即平倉。",
            "RsiStrategy (RSI)": "【RSI 策略】\nRSI < 30 (超賣) 買入，RSI > 70 (超熱) 賣出。"
        }

        # 主佈局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # --- 左側：控制面板 ---
        sidebar = QWidget()
        sidebar.setFixedWidth(350)
        sidebar_layout = QVBoxLayout(sidebar)
        main_layout.addWidget(sidebar)

        # 0. 市場與接口選擇
        api_group = QGroupBox("0. 市場與接口設定")
        api_form = QFormLayout()

        self.market_combo = QComboBox()
        self.market_combo.addItems(["台股 (Taiwan)", "美股 (US)"])
        self.market_combo.currentTextChanged.connect(self.on_market_change)

        self.api_source_combo = QComboBox()
        self.api_source_combo.addItems(["預設 (Auto)", "Shioaji", "FinMind", "yfinance"])
        self.api_source_combo.setCurrentText("Shioaji")

        api_form.addRow("選擇市場:", self.market_combo)
        api_form.addRow("API 來源:", self.api_source_combo)
        api_group.setLayout(api_form)
        sidebar_layout.addWidget(api_group)

        # 1. 產業分類與股票選擇
        set_group = QGroupBox("1. 產業與股票設定")
        set_form = QFormLayout()

        self.industry_combo = QComboBox()
        self.update_industry_list()
        self.industry_combo.currentTextChanged.connect(self.filter_stocks_by_industry)

        self.stock_id_input = QComboBox()
        self.stock_id_input.setEditable(True)
        self.filter_stocks_by_industry("全部")

        self.start_date_input = QDateEdit(QDate(2020, 1, 1))
        self.start_date_input.setCalendarPopup(True)
        self.end_date_input = QDateEdit(QDate.currentDate())
        self.end_date_input.setCalendarPopup(True)

        self.cash_input = QLineEdit("1000000")
        self.auto_update_cb = QCheckBox("自動更新回測資料")

        set_form.addRow("產業分類:", self.industry_combo)
        set_form.addRow("股票代號:", self.stock_id_input)
        set_form.addRow("開始日期:", self.start_date_input)
        set_form.addRow("結束日期:", self.end_date_input)
        set_form.addRow("初始資金:", self.cash_input)
        set_form.addRow(self.auto_update_cb)
        set_group.setLayout(set_form)
        sidebar_layout.addWidget(set_group)

        # 2. 回測與即時監控
        mode_group = QGroupBox("2. 模式切換")
        mode_layout = QVBoxLayout()

        # 即時監控區塊
        rt_layout = QHBoxLayout()
        self.interval_combo = QComboBox()
        self.interval_combo.addItems(["5s", "10s", "30s", "1m", "5m", "10m", "15m", "30m", "1h", "3h", "今天(靜態)"])
        self.interval_combo.setCurrentText("1m")

        self.btn_rt_toggle = QPushButton("🔴 開啟即時監控")
        self.btn_rt_toggle.setCheckable(True)
        self.btn_rt_toggle.clicked.connect(self.toggle_realtime)

        rt_layout.addWidget(QLabel("頻率:"))
        rt_layout.addWidget(self.interval_combo)
        rt_layout.addWidget(self.btn_rt_toggle)
        mode_layout.addLayout(rt_layout)

        self.btn_fetch = QPushButton("📥 手動更新歷史資料")
        self.btn_fetch.clicked.connect(self.do_fetch)
        mode_layout.addWidget(self.btn_fetch)

        self.btn_run = QPushButton("🚀 執行策略回測")
        self.btn_run.setStyleSheet("background-color: #2e7d32; color: white; height: 40px; font-weight: bold;")
        self.btn_run.clicked.connect(self.do_backtest)

        self.strat_combo = QComboBox()
        self.strat_combo.addItems(list(self.strat_info.keys()))
        self.strat_combo.currentIndexChanged.connect(self.update_desc)

        mode_layout.addWidget(QLabel("選擇策略:"))
        mode_layout.addWidget(self.strat_combo)
        mode_layout.addWidget(self.btn_run)
        mode_group.setLayout(mode_layout)
        sidebar_layout.addWidget(mode_group)

        # 3. 說明
        desc_group = QGroupBox("3. 策略說明")
        desc_layout = QVBoxLayout()
        self.desc_box = QTextEdit()
        self.desc_box.setReadOnly(True)
        self.desc_box.setFixedHeight(80)
        self.desc_box.setStyleSheet("background-color: #f9f9f9; color: #333;")
        desc_layout.addWidget(self.desc_box)
        desc_group.setLayout(desc_layout)
        sidebar_layout.addWidget(desc_group)

        self.status_lbl = QLabel("狀態: ⚠️ 待命")
        sidebar_layout.addWidget(self.status_lbl)
        sidebar_layout.addStretch()

        # --- 右側：結果展示 ---
        display_layout = QVBoxLayout()
        main_layout.addLayout(display_layout)

        self.tabs = QTabWidget()
        display_layout.addWidget(self.tabs)

        # Tab 1: 績效與圖表
        self.bt_tab = QWidget()
        bt_layout = QVBoxLayout(self.bt_tab)

        metrics_group = QGroupBox("回測績效指標")
        self.metrics_layout = QVBoxLayout()
        row1 = QHBoxLayout(); self.lbl_final = QPushButton("最終資產: --"); self.lbl_ret = QPushButton("總報酬率: --"); row1.addWidget(self.lbl_final); row1.addWidget(self.lbl_ret)
        row2 = QHBoxLayout(); self.lbl_sharpe = QPushButton("夏普值: --"); self.lbl_dd = QPushButton("最大回落: --"); row2.addWidget(self.lbl_sharpe); row2.addWidget(self.lbl_dd)
        row3 = QHBoxLayout(); self.lbl_trades = QPushButton("總交易次數: --"); self.lbl_winrate = QPushButton("勝率: --"); self.lbl_sqn = QPushButton("SQN: --"); row3.addWidget(self.lbl_trades); row3.addWidget(self.lbl_winrate); row3.addWidget(self.lbl_sqn)
        self.metrics_layout.addLayout(row1); self.metrics_layout.addLayout(row2); self.metrics_layout.addLayout(row3)
        metrics_group.setLayout(self.metrics_layout)
        bt_layout.addWidget(metrics_group)

        self.img_label = QLabel("點擊「執行回測」以顯示圖表")
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_label.setScaledContents(True)
        bt_layout.addWidget(self.img_label, stretch=1)
        self.tabs.addTab(self.bt_tab, "回測分析")

        # Tab 2: 即時監控數據
        self.rt_tab = QWidget()
        rt_tab_layout = QVBoxLayout(self.rt_tab)
        self.rt_log = QTextEdit(); self.rt_log.setReadOnly(True); self.rt_log.append("即時監控日誌...")
        rt_tab_layout.addWidget(self.rt_log)
        self.tabs.addTab(self.rt_tab, "即時監控")

        # Tab 3: 產業相關性
        self.corr_tab = QWidget()
        corr_layout = QVBoxLayout(self.corr_tab)
        self.btn_run_corr = QPushButton("📊 計算產業相關性 (取前10檔)"); self.btn_run_corr.clicked.connect(self.do_correlation); corr_layout.addWidget(self.btn_run_corr)
        self.corr_box = QTextEdit(); self.corr_box.setReadOnly(True); self.corr_box.setFontFamily("Courier New"); corr_layout.addWidget(self.corr_box)
        self.tabs.addTab(self.corr_tab, "產業相關性")

        # Tab 4: 股票分析與基本面
        self.basic_tab = QWidget()
        basic_layout = QVBoxLayout(self.basic_tab)
        btn_layout = QHBoxLayout(); self.btn_fetch_fin = QPushButton("💰 抓取損益表/籌碼 (台股限定)"); self.btn_fetch_fin.clicked.connect(self.do_fetch_financials); btn_layout.addWidget(self.btn_fetch_fin); basic_layout.addLayout(btn_layout)
        self.fin_table = QTextEdit(); self.fin_table.setReadOnly(True); basic_layout.addWidget(self.fin_table)
        self.tabs.addTab(self.basic_tab, "深度資料")

        # Tab 5: 技術指標與動能分析
        self.kline_tab = QWidget()
        kline_layout = QVBoxLayout(self.kline_tab)
        self.btn_analyze_k = QPushButton("🕯️ 執行技術與動能辨識"); self.btn_analyze_k.clicked.connect(self.do_kline_analysis); kline_layout.addWidget(self.btn_analyze_k)
        self.kline_box = QTextEdit(); self.kline_box.setReadOnly(True); kline_layout.addWidget(self.kline_box)
        self.tabs.addTab(self.kline_tab, "技術/動能分析")

        # Tab 6: 新聞情緒分析
        self.sentiment_tab = QWidget()
        sentiment_layout = QVBoxLayout(self.sentiment_tab)
        self.btn_analyze_sentiment = QPushButton("📰 執行新聞情緒分析"); self.btn_analyze_sentiment.clicked.connect(self.do_sentiment_analysis); sentiment_layout.addWidget(self.btn_analyze_sentiment)
        
        # 情緒文字與圖表佈局
        sent_split_layout = QHBoxLayout()
        self.sentiment_box = QTextEdit(); self.sentiment_box.setReadOnly(True)
        self.sentiment_box.setFixedWidth(400)
        self.sentiment_img_label = QLabel("點擊「執行新聞情緒分析」以顯示趨勢圖")
        self.sentiment_img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sentiment_img_label.setScaledContents(True)
        
        sent_split_layout.addWidget(self.sentiment_box)
        sent_split_layout.addWidget(self.sentiment_img_label, stretch=1)
        sentiment_layout.addLayout(sent_split_layout)
        
        self.tabs.addTab(self.sentiment_tab, "市場情緒")

        self.update_desc()

    def on_market_change(self, market):
        """當市場切換時更新 UI"""
        if "美股" in market:
            self.industry_combo.setEnabled(False)
            self.stock_id_input.clear()
            # 新增美股大盤指數與熱門股
            self.stock_id_input.addItems([
                "^GSPC (S&P 500)", "^IXIC (Nasdaq)", "^DJI (Dow Jones)",
                "AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMD", "META", "AMZN", "NFLX"
            ])
            self.api_source_combo.setCurrentText("Shioaji")
        else:
            self.industry_combo.setEnabled(True)
            self.update_industry_list()
            self.filter_stocks_by_industry(self.industry_combo.currentText())
            self.api_source_combo.setCurrentText("Shioaji")

    def update_industry_list(self):
        self.industry_combo.clear()
        industries = ["全部"] + sorted(self.stock_info_df['industry_category'].unique().tolist()) if not self.stock_info_df.empty else ["全部"]
        self.industry_combo.addItems(industries)

    def do_kline_analysis(self):
        sid = self.stock_id_input.currentText().split(" ")[0]
        market = self.market_combo.currentText()
        is_us = "美股" in market

        self.status_lbl.setText(f"狀態: ⏳ 正在分析 {sid} 技術指標...")
        QApplication.processEvents()

        table_name = f"us_stock_{sid}_daily" if is_us else f"stock_{sid}_daily"
        df = self.db.load_dataframe(table_name)

        if df.empty:
            QMessageBox.warning(self, "警告", f"找不到 {sid} 本地資料，請先抓取資料。")
            return

        df = df.sort_values('date')
        df = KLineAnalyzer.add_indicators(df)
        patterns = KDAnalyzer.analyze_patterns(df)
        mom_res = MomentumAnalyzer.analyze_momentum(df)

        report = f"--- {sid} 深度技術分析報告 ({datetime.now().strftime('%Y-%m-%d')}) ---\n\n"
        report += "【價格動能指標 (Momentum)】\n"
        report += f"- 1日動能: {mom_res['m1']:.2f}%\n"
        report += f"- 3日動能: {mom_res['m3']:.2f}%\n"
        report += f"- 5日動能: {mom_res['m5']:.2f}%\n"
        report += f"- 15日動能: {mom_res['m15']:.2f}%\n"
        report += f"- 30日動能: {mom_res['m30']:.2f}%\n"
        if mom_res['status']:
            report += "\n".join([f"  {s}" for s in mom_res['status']]) + "\n"
        report += "\n【成交量分析 (Volume)】\n"
        latest = df.iloc[-1]
        report += f"- 當日成交量: {latest['Volume']:,.0f}\n"
        if len(df) > 5:
            avg_vol = df['Volume'].tail(5).mean()
            vol_ratio = (latest['Volume'] / avg_vol) * 100
            report += f"- 5日均量: {avg_vol:,.0f}\n"
            report += f"- 量能比: {vol_ratio:.2f}% (相對於5日均量)\n"
        report += "\n【KD 訊號】:\n"
        if patterns:
            report += "\n".join([f"- {p}" for p in patterns])
        else:
            report += "目前無明顯 KD 指標訊號。"
        report += "\n\n" + KDAnalyzer.get_principles()
        self.kline_box.setText(report)
        self.status_lbl.setText("狀態: ✅ 分析完成")

    def do_fetch(self):
        sid = self.stock_id_input.currentText().split(" ")[0]
        market = self.market_combo.currentText()
        is_us = "美股" in market
        start = self.start_date_input.date().toString("yyyy-MM-dd")
        end = self.end_date_input.date().toString("yyyy-MM-dd")
        self.status_lbl.setText(f"狀態: ⏳ 正在抓取 {sid} ({market})...")
        QApplication.processEvents()
        try:
            if is_us:
                self.fetcher.fetch_us_stock_daily(sid, start, end)
            else:
                self.fetcher.fetch_stock_daily(sid, start, end)
            self.status_lbl.setText(f"狀態: ✅ {sid} 資料就緒")
            QMessageBox.information(self, "成功", f"{sid} ({market}) 歷史資料更新完成")
        except Exception as e:
            QMessageBox.critical(self, "錯誤", str(e))

    def do_correlation(self):
        industry = self.industry_combo.currentText()
        if industry == "全部":
            QMessageBox.warning(self, "警告", "請先選擇一個特定的產業分類。")
            return
        selected_sid = self.stock_id_input.currentText().split(" ")[0]
        self.status_lbl.setText(f"狀態: ⏳ 正在計算 {selected_sid} 產業相關性...")
        QApplication.processEvents()

        # 獲取產業代表龍頭股
        leaders = self.fetcher.get_industry_leaders(industry)
        industry_stocks = self.stock_info_df[self.stock_info_df['industry_category'] == industry]['stock_id'].tolist()
        
        # 組合比較清單：選中股 + 龍頭股 + 剩餘補位 (上限 10 檔)
        stocks_to_compare = [selected_sid]
        for s in leaders:
            if s != selected_sid and s not in stocks_to_compare:
                stocks_to_compare.append(s)
        
        for s in industry_stocks:
            if len(stocks_to_compare) >= 10: break
            if s != selected_sid and s not in stocks_to_compare:
                stocks_to_compare.append(s)

        start = self.start_date_input.date().toString("yyyy-MM-dd")
        end = self.end_date_input.date().toString("yyyy-MM-dd")
        df_all = self.fetcher.fetch_industry_prices(stocks_to_compare, start, end)
        if df_all.empty:
            self.corr_box.setText("無法獲取資料。"); return
        # 確保選中股票確實在結果中
        if selected_sid not in df_all.columns:
             self.corr_box.setText(f"警告: 無法獲取選定股票 {selected_sid} 的價格資料，無法進行比較。"); return

        corr_matrix = df_all.corr().round(2)
        self.corr_box.setText(f"【{selected_sid} vs {industry} 龍頭】產業相關性矩陣\n\n" + corr_matrix.to_string())
        self.status_lbl.setText("狀態: ✅ 矩陣已生成")

    def do_fetch_financials(self):
        sid = self.stock_id_input.currentText().split(" ")[0]
        market = self.market_combo.currentText()
        is_us = "美股" in market

        start = self.start_date_input.date().toString("yyyy-MM-dd")
        self.status_lbl.setText(f"狀態: ⏳ 正在抓取 {sid} 深度資料...")
        QApplication.processEvents()

        if is_us:
            info = self.fetcher.fetch_us_stock_info(sid)
            fins = self.fetcher.fetch_us_stock_financials(sid)
            if info:
                report = f"==== {sid} ({info['名稱']}) 美股深度報告 ====\n\n"
                report += f"【基本面摘要】\n"
                report += f"- 產業: {info['產業']} | 板塊: {info['板塊']}\n"
                report += f"- 市值: ${info['市值']} | PE: {info['本益比 (PE)']}\n"
                report += f"- 股息率: {info['股息率']} | 52週: {info['52週低點']} - {info['52週高點']}\n\n"

                if fins:
                    report += "【近期年度損益表 (Income Statement)】\n"
                    report += fins['income'].head(10).to_string() + "\n\n"
                    report += "【近期年度資產負債 (Balance Sheet)】\n"
                    report += fins['balance'].head(10).to_string() + "\n\n"

                report += f"【公司簡介】\n{info['公司簡介']}\n"
                self.fin_table.setText(report)
            else:
                self.fin_table.setText(f"無法獲取 {sid} 的美股資訊。")
        else:
            # 台股邏輯
            # 確保抓取足夠近的資料以獲取最新損益表 (預設抓取最近兩年)
            recent_start = (datetime.now() - timedelta(days=365*2)).strftime("%Y-%m-%d")
            actual_start = start if start < recent_start else recent_start
            
            data = self.fetcher.fetch_stock_financials(sid, actual_start)
            chips = self.fetcher.fetch_stock_chips(sid, start)
            df_rev = self.fetcher.fetch_revenue_breakdown(sid)

            # 獲取技術指標與動能
            df_ta = self.db.load_dataframe(f"stock_{sid}_daily")
            # 如果資料庫為空，或資料筆數不足 35 筆 (無法計算 30 日動能)
            if df_ta.empty or len(df_ta) < 35:
                end = self.end_date_input.date().toString("yyyy-MM-dd")
                # 強制至少抓取半年 (約 180 天) 前的資料以確保動能指標和均線能正確計算
                force_start = (datetime.now() - pd.Timedelta(days=180)).strftime("%Y-%m-%d")
                actual_ta_start = start if start < force_start else force_start
                df_ta = self.fetcher.fetch_stock_daily(sid, actual_ta_start, end)

            mom_report = "無資料"
            if not df_ta.empty:
                df_ta = df_ta.sort_values('date')
                df_ta = KLineAnalyzer.add_indicators(df_ta)
                mom_res = MomentumAnalyzer.analyze_momentum(df_ta)
                mom_report = f"- 1日: {mom_res['m1']:.2f}% | 5日: {mom_res['m5']:.2f}% | 30日: {mom_res['m30']:.2f}%\n"
                if mom_res['status']: mom_report += "  " + " ".join(mom_res['status'])

            # 獲取新聞情緒
            df_news = self.fetcher.fetch_stock_news(sid, start)
            sent_report = "無資料"
            if not df_news.empty:
                avg_score, _ = SentimentAnalyzer.analyze_sentiment(df_news.to_dict('records'))
                summary = "偏向看多" if avg_score > 0.2 else ("偏向看空" if avg_score < -0.2 else "中性")
                sent_report = f"評分: {avg_score:.2f} ({summary})"

            report = f"==== {sid} 深度綜合分析報告 ({datetime.now().strftime('%Y-%m-%d')}) ====\n\n"

            report += "[1. 營業比重 (Revenue Breakdown)]\n"
            if not df_rev.empty:
                for _, row in df_rev.iterrows():
                    t_str = f"[{row.get('type', '')}] " if 'type' in row and pd.notna(row.get('type')) else ""
                    report += f"- {t_str}{row.get('name', '未知')}: {row.get('percentage', 0.0)}%\n"
            else:
                report += "  (暫無資料)\n"

            report += "\n[2. 籌碼面 (Chip Analysis)]\n"
            df_margin = chips.get("margin")
            if df_margin is not None and not df_margin.empty:
                latest_m = df_margin.iloc[-1]
                report += f"- 融資餘額: {latest_m.get('MarginPurchaseTodayBalance', 0):,}\n"
                report += f"- 融券餘額: {latest_m.get('ShortSaleTodayBalance', 0):,}\n"

            df_holdings = chips.get("holdings")
            if df_holdings is not None and not df_holdings.empty:
                # 取得最近一週的大戶持股 (400張以上與1000張以上)
                latest_h = df_holdings.sort_values('date').iloc[-1]
                report += f"- 大戶持股比例 (最新): {latest_h.get('holding_shares_percentage', 'N/A')}%\n"

            report += f"\n[3. 技術動能 (Technical Momentum)]\n{mom_report}\n"
            report += f"\n[4. 市場情緒 (Market Sentiment)]\n{sent_report}\n"

            df_fin = data.get("financials", pd.DataFrame())
            if not df_fin.empty:
                report += "\n[5. 近期損益摘要 (由新到舊)]\n"
                # 只顯示最近 8 筆
                report += df_fin.head(15).to_string()

            self.fin_table.setText(report)

            self.fin_table.setText(report)

        self.status_lbl.setText(f"狀態: ✅ {sid} 資料分析完成")

    def load_stock_info(self):
        try:
            df = self.db.load_dataframe("taiwan_stock_info")
            
            # 檢查是否需要重新抓取 (如果資料為空，或者產業分類看起來只有數字，或者缺少指數大盤)
            need_refresh = False
            if df.empty:
                need_refresh = True
            else:
                # 抽樣檢查產業分類是否包含中文描述
                sample = df['industry_category'].dropna().unique().tolist()
                # 如果前 5 個樣本都是純數字或長度 <= 2，則認為需要更新
                if sample and all(str(s).isdigit() or len(str(s)) <= 2 for s in sample[:5]):
                    need_refresh = True
                
                # 檢查是否包含指數 (00 指數/大盤)
                if not any("00 指數" in str(s) for s in sample):
                    need_refresh = True
            
            if need_refresh:
                print("Refreshing stock and index info...")
                df = self.fetcher.fetch_stock_info()
            
            # 確保欄位名稱一致性，避免 KeyError: 'industry_category'
            if not df.empty:
                if 'industry' in df.columns and 'industry_category' not in df.columns:
                    df.rename(columns={'industry': 'industry_category'}, inplace=True)
                elif 'industry_category' not in df.columns:
                    # 如果連 industry 都沒有，建立一個空的
                    df['industry_category'] = '未知'
            return df
        except Exception as e:
            print(f"Error loading stock info: {e}")
            return pd.DataFrame()

    def filter_stocks_by_industry(self, industry):
        self.stock_id_input.clear()
        if self.stock_info_df.empty:
            self.stock_id_input.addItems(["TAIEX", "2330", "2317"]); return
        filtered = self.stock_info_df if industry == "全部" else self.stock_info_df[self.stock_info_df['industry_category'] == industry]
        items = filtered.apply(lambda x: f"{x['stock_id']} {x['stock_name']}", axis=1).tolist()
        self.stock_id_input.addItems(items)

    def toggle_realtime(self):
        if self.btn_rt_toggle.isChecked():
            self.btn_rt_toggle.setText("🟢 監控中"); self.btn_rt_toggle.setStyleSheet("background-color: #388e3c; color: white;")
            interval_map = {"5s": 5000, "10s": 10000, "1m": 60000}; ms = interval_map.get(self.interval_combo.currentText(), 60000)
            self.rt_timer.start(ms)
        else:
            self.btn_rt_toggle.setText("🔴 開啟即時監控"); self.btn_rt_toggle.setStyleSheet(""); self.rt_timer.stop()

    def update_realtime(self):
        sid = self.stock_id_input.currentText().split(" ")[0]
        tick = self.fetcher.fetch_realtime_tick(sid)
        if not tick.empty: self.rt_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {sid} 價: {tick['Close'].iloc[0]}")

    def update_desc(self):
        self.desc_box.setText(self.strat_info.get(self.strat_combo.currentText(), ""))

    def fmt(self, val, suffix=""):
        return f"{val:.2f}{suffix}" if isinstance(val, (int, float)) else "N/A"

    def do_backtest(self):
        try:
            sid = self.stock_id_input.currentText().split(" ")[0]
            if self.auto_update_cb.isChecked(): self.do_fetch()
            strats = {"SmaCross (雙均線)": SmaCross, "MacdStrategy (MACD)": MacdStrategy, "ForeignBuyStrategy (外資)": ForeignBuyStrategy, "RsiStrategy (RSI)": RsiStrategy}
            self.status_lbl.setText("狀態: ⏳ 運算中...")
            QApplication.processEvents()
            initial_cash = float(self.cash_input.text()) if self.cash_input.text() else 1000000.0
            results = run_backtest(sid, strategy=strats[self.strat_combo.currentText()], cash=initial_cash, plot=True)
            if "error" in results: QMessageBox.warning(self, "警告", results["error"]); return
            self.lbl_final.setText(f"最終資產: ${results['final_value']:,.0f}"); self.lbl_ret.setText(f"總報酬率: {results['return_pct']:.2f}%")
            if results.get("plot_file"): self.img_label.setPixmap(QPixmap(results["plot_file"]))
            self.status_lbl.setText("狀態: ✅ 回測完成")
        except Exception as e: QMessageBox.critical(self, "錯誤", str(e))

    def do_sentiment_analysis(self):
        full_text = self.stock_id_input.currentText()
        sid = full_text.split(" ")[0]
        # 嘗試取得名稱，以便在 ID 搜不到時備用
        sname = full_text.split(" ")[1] if " " in full_text else ""
        
        market = self.market_combo.currentText()
        is_us = "美股" in market

        self.status_lbl.setText(f"狀態: ⏳ 正在分析 {sid} 市場情緒...")
        QApplication.processEvents()

        try:
            if is_us:
                # 美股目前沿用原本的簡單分析
                news_list = self.fetcher.fetch_us_stock_news(sid)
                if not news_list:
                    self.sentiment_box.setText(f"目前沒有關於 {sid} 的近期新聞資料。")
                    return
                avg_score, results = SentimentAnalyzer.analyze_sentiment(news_list, is_us=is_us)

                report = f"--- {sid} 市場情緒分析報告 ({datetime.now().strftime('%Y-%m-%d')}) ---\n\n"
                report += f"【綜合情緒評分】: {avg_score:.2f} \n"
                summary = "偏向看多" if avg_score > 0.2 else ("偏向看空" if avg_score < -0.2 else "中性/觀望")
                report += f"【情緒導向】: {summary}\n"
                report += "="*40 + "\n\n"

                for res in results:
                    report += f"[{res['date']}] {res['label']} (分: {res['score']})\n"
                    report += f"標題: {res['title']}\n"
                    if res['link']: report += f"連結: {res['link']}\n"
                    report += "-"*20 + "\n"

                self.sentiment_box.setText(report)
            else:
                # 台股使用整合進來的 AdvancedSentimentAnalyzer
                analyzer = AdvancedSentimentAnalyzer()
                df_news = analyzer.fetch_and_analyze(sid)
                
                # 如果用代號找不到，試試用名稱
                if df_news.empty and sname:
                    df_news = analyzer.fetch_and_analyze(sname)

                if df_news.empty:
                    self.sentiment_box.setText(f"目前沒有關於 {sid} {sname} 的近期新聞資料。\n(建議手動至 Google 搜尋確認)")
                    return

                avg_score = df_news['情緒分數'].mean()
                weighted_avg = df_news['加權分數'].mean()

                report = f"--- {sid} {sname} 深度新聞情緒分析 ({datetime.now().strftime('%Y-%m-%d')}) ---\n\n"
                report += f"📈 樣本新聞數：{len(df_news)} 則\n"
                report += f"🌡️ 平均情緒分數：{round(avg_score, 4)}\n"
                report += f"⏳ 加權情緒分數：{round(weighted_avg, 4)} (已計算時間衰減)\n"
                
                summary = "🔥 極度樂觀" if avg_score > 0.2 else ("❄️ 偏向保守" if avg_score < -0.2 else "⚖️ 盤整中性")
                report += f"📢 市場傾向：{summary}\n"
                report += "="*40 + "\n\n"

                # 偵測跳空點警告
                jumps = df_news[df_news['不連續性'].abs() > 0.4].tail(3)
                if not jumps.empty:
                    report += "⚠️ 【偵測到情緒劇烈跳動 (Jumps)】\n"
                    for _, row in jumps.iterrows():
                        report += f"[{row['時間']}] 跳空度: {row['不連續性']:.2f} -> {row['標題'][:30]}...\n"
                    report += "-"*40 + "\n\n"

                report += "📝 【近期重要新聞評分列表】\n"
                display_df = df_news.sort_values('date_obj', ascending=False).head(15)
                for _, row in display_df.iterrows():
                    report += f"[{row['時間']}] {row['情緒評價']} (分: {row['情緒分數']:.2f})\n"
                    report += f"標題: {row['標題']}\n"
                    report += "-"*20 + "\n"

                self.sentiment_box.setText(report)
                
                # 產出圖片並顯示
                import os
                import matplotlib.pyplot as plt
                analyzer.plot_trends(df_news, sid)
                img_path = f"ai_research/{sid}_advanced_sentiment.png"
                if os.path.exists(img_path):
                    self.sentiment_img_label.setPixmap(QPixmap(img_path))
                
            self.status_lbl.setText("狀態: ✅ 情緒分析完成")

        except Exception as e:
            QMessageBox.critical(self, "錯誤", str(e))

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow(); window.show()
    app.exec()
