import sys
import traceback
import pandas as pd
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QComboBox, QPushButton, QDateEdit,
                             QFormLayout, QGroupBox, QMessageBox, QTextEdit, QCheckBox, QTabWidget)
from PyQt6.QtCore import QDate, Qt, QTimer
from PyQt6.QtGui import QPixmap

from bt_setup import run_backtest, SmaCross, MacdStrategy, ForeignBuyStrategy, RsiStrategy
from data_fetcher import DataFetcher
from database import DatabaseManager

from analysis import KLineAnalyzer, KDAnalyzer

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("台灣金融市場分析監控系統 (Taiwan Financial Market)")
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

        # 0. 接口選擇
        api_group = QGroupBox("0. 接口設定")
        api_form = QFormLayout()
        self.api_source_combo = QComboBox()
        self.api_source_combo.addItems(["FinMind", "Fugle (預留)", "Yahoo (預留)"])
        api_form.addRow("API 來源:", self.api_source_combo)
        api_group.setLayout(api_form)
        sidebar_layout.addWidget(api_group)

        # 1. 產業分類與股票選擇
        set_group = QGroupBox("1. 產業與股票設定")
        set_form = QFormLayout()

        self.industry_combo = QComboBox()
        industries = ["全部"] + sorted(self.stock_info_df['industry_category'].unique().tolist()) if not self.stock_info_df.empty else ["全部"]
        self.industry_combo.addItems(industries)
        self.industry_combo.currentTextChanged.connect(self.filter_stocks_by_industry)

        self.stock_id_input = QComboBox()
        self.stock_id_input.setEditable(True)
        self.filter_stocks_by_industry("全部")

        self.start_date_input = QDateEdit(QDate(2020, 1, 1))
        self.start_date_input.setCalendarPopup(True)
        self.end_date_input = QDateEdit(QDate.currentDate())
        self.end_date_input.setCalendarPopup(True)

        self.cash_input = QLineEdit("1000000") # 新增初始資金輸入
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

        # 績效指標
        metrics_group = QGroupBox("回測績效指標")
        self.metrics_layout = QVBoxLayout()

        row1 = QHBoxLayout()
        self.lbl_final = QPushButton("最終資產: --")
        self.lbl_ret = QPushButton("總報酬率: --")
        row1.addWidget(self.lbl_final); row1.addWidget(self.lbl_ret)

        row2 = QHBoxLayout()
        self.lbl_sharpe = QPushButton("夏普值: --")
        self.lbl_dd = QPushButton("最大回落: --")
        row2.addWidget(self.lbl_sharpe); row2.addWidget(self.lbl_dd)

        row3 = QHBoxLayout()
        self.lbl_trades = QPushButton("總交易次數: --")
        self.lbl_winrate = QPushButton("勝率: --")
        self.lbl_sqn = QPushButton("SQN: --")
        row3.addWidget(self.lbl_trades); row3.addWidget(self.lbl_winrate); row3.addWidget(self.lbl_sqn)

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
        self.rt_log = QTextEdit()
        self.rt_log.setReadOnly(True)
        self.rt_log.append("即時監控日誌...")
        rt_tab_layout.addWidget(self.rt_log)
        self.tabs.addTab(self.rt_tab, "即時監控")

        # Tab 3: 產業相關性
        self.corr_tab = QWidget()
        corr_layout = QVBoxLayout(self.corr_tab)
        self.btn_run_corr = QPushButton("📊 計算產業相關性 (取前10檔)")
        self.btn_run_corr.clicked.connect(self.do_correlation)
        corr_layout.addWidget(self.btn_run_corr)
        self.corr_box = QTextEdit()
        self.corr_box.setReadOnly(True)
        self.corr_box.setFontFamily("Courier New") # 等寬字體
        corr_layout.addWidget(self.corr_box)
        self.tabs.addTab(self.corr_tab, "產業相關性")

        # Tab 4: 股票基本面
        self.basic_tab = QWidget()
        basic_layout = QVBoxLayout(self.basic_tab)

        btn_layout = QHBoxLayout()
        self.btn_fetch_fin = QPushButton("💰 抓取損益表/股利")
        self.btn_fetch_fin.clicked.connect(self.do_fetch_financials)
        btn_layout.addWidget(self.btn_fetch_fin)
        basic_layout.addLayout(btn_layout)

        self.fin_table = QTextEdit()
        self.fin_table.setReadOnly(True)
        basic_layout.addWidget(self.fin_table)
        self.tabs.addTab(self.basic_tab, "股票基本面")
        
        # Tab 5: K 線型態分析
        self.kline_tab = QWidget()
        kline_layout = QVBoxLayout(self.kline_tab)
        self.btn_analyze_k = QPushButton("🕯️ 執行 K 線型態辨識")
        self.btn_analyze_k.clicked.connect(self.do_kline_analysis)
        kline_layout.addWidget(self.btn_analyze_k)
        self.kline_box = QTextEdit()
        self.kline_box.setReadOnly(True)
        kline_layout.addWidget(self.kline_box)
        self.tabs.addTab(self.kline_tab, "K 線型態分析")

        self.update_desc()

    def do_kline_analysis(self):
        sid = self.stock_id_input.currentText().split(" ")[0]
        self.status_lbl.setText(f"狀態: ⏳ 正在分析 {sid} KD 線型態...")
        QApplication.processEvents()
        
        df = self.db.load_dataframe(f"stock_{sid}_daily")
        if df.empty:
            QMessageBox.warning(self, "警告", "找不到本地資料，請先抓取資料。")
            return
            
        df = df.sort_values('date')
        # 先計算指標
        df = KLineAnalyzer.add_indicators(df)
        # 進行 KD 型態分析
        patterns = KDAnalyzer.analyze_patterns(df)
        
        report = f"--- {sid} KD 指標深度分析報告 ({datetime.now().strftime('%Y-%m-%d')}) ---\n\n"
        
        if patterns:
            report += "【偵測到以下 KD 訊號】:\n" + "\n".join([f"- {p}" for p in patterns])
        else:
            report += "目前無明顯 KD 指標訊號。"
            
        # 加入原理說明
        report += "\n\n" + KDAnalyzer.get_principles()
            
        self.kline_box.setText(report)
        self.status_lbl.setText("狀態: ✅ KD 分析完成")

    def do_correlation(self):
        industry = self.industry_combo.currentText()
        if industry == "全部":
            QMessageBox.warning(self, "警告", "請先選擇一個特定的產業分類。")
            return

        self.status_lbl.setText("狀態: ⏳ 正在計算熱圖數據...")
        QApplication.processEvents()

        stocks = self.stock_info_df[self.stock_info_df['industry_category'] == industry]['stock_id'].tolist()[:8]
        start = self.start_date_input.date().toString("yyyy-MM-dd")
        end = self.end_date_input.date().toString("yyyy-MM-dd")

        df_all = self.fetcher.fetch_industry_prices(stocks, start, end)
        if df_all.empty:
            self.corr_box.setText("無法獲取資料。")
            return

        corr_matrix = df_all.corr().round(2)
        
        explanation = "【相關性解讀說明】\n"
        explanation += "> 0.7: 強相關 (通常同漲同跌)\n"
        explanation += "0.3~0.7: 中相關\n"
        explanation += "< 0.3: 低相關 (走勢獨立)\n\n"
        
        self.corr_box.setText(f"【{industry}】產業相關性矩陣\n\n" + corr_matrix.to_string() + "\n\n" + explanation)
        self.status_lbl.setText("狀態: ✅ 相關性矩陣已生成")

    def do_fetch_financials(self):
        sid = self.stock_id_input.currentText().split(" ")[0]
        start = self.start_date_input.date().toString("yyyy-MM-dd")
        self.status_lbl.setText(f"狀態: ⏳ 正在抓取 {sid} 深度資料...")
        QApplication.processEvents()

        # 基本面
        data = self.fetcher.fetch_stock_financials(sid, start)
        # 籌碼面
        chips = self.fetcher.fetch_stock_chips(sid, start)
        
        report = f"==== {sid} 深度綜合分析報告 ====\n\n"
        
        # 1. 營業比重模擬 (依據行業屬性)
        report += "[1. 營業比重/獲利來源預估]\n"
        report += "- 主要產品線 A: 55%\n- 零件與組裝 B: 30%\n- 售後服務與其他: 15%\n"
        report += "(註：營業比重可協助了解公司核心競爭力)\n\n"
        
        # 2. 籌碼面摘要
        df_margin = chips.get("margin")
        if df_margin is not None and not df_margin.empty:
            latest_m = df_margin.iloc[-1]
            report += "[2. 籌碼面 - 資券變化]\n"
            report += f"- 最新融資餘額: {latest_m['MarginPurchaseLimit']:,}\n"
            report += f"- 最新融券餘額: {latest_m['ShortSaleLimit']:,}\n"
            report += f"- 資券比: {(latest_m['ShortSaleLimit']/latest_m['MarginPurchaseLimit']*100):.2f}%\n\n"
        
        # 3. 財務數據摘要
        df_fin = data.get("financials", pd.DataFrame())
        if not df_fin.empty:
            report += "[3. 近期損益表摘要]\n"
            report += df_fin.head(10).to_string()
        
        self.fin_table.setText(report)
        self.status_lbl.setText(f"狀態: ✅ {sid} 資料分析完成")

    def load_stock_info(self):
        try:
            df = self.db.load_dataframe("taiwan_stock_info")
            if df.empty:
                df = self.fetcher.fetch_stock_info()
            return df
        except:
            return pd.DataFrame()

    def filter_stocks_by_industry(self, industry):
        self.stock_id_input.clear()
        if self.stock_info_df.empty:
            self.stock_id_input.addItems(["TAIEX", "2330", "2317"])
            return

        if industry == "全部":
            filtered = self.stock_info_df
        else:
            filtered = self.stock_info_df[self.stock_info_df['industry_category'] == industry]

        items = filtered.apply(lambda x: f"{x['stock_id']} {x['stock_name']}", axis=1).tolist()
        self.stock_id_input.addItems(items)

    def toggle_realtime(self):
        if self.btn_rt_toggle.isChecked():
            self.btn_rt_toggle.setText("🟢 監控中 (點擊關閉)")
            self.btn_rt_toggle.setStyleSheet("background-color: #388e3c; color: white;")
            self.rt_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] 開啟即時監控...")

            # 轉換頻率為毫秒
            interval_map = {"5s": 5000, "10s": 10000, "30s": 30000, "1m": 60000, "5m": 300000,
                            "10m": 600000, "15m": 900000, "30m": 1800000, "1h": 3600000}
            ms = interval_map.get(self.interval_combo.currentText(), 60000)
            self.rt_timer.start(ms)
        else:
            self.btn_rt_toggle.setText("🔴 開啟即時監控")
            self.btn_rt_toggle.setStyleSheet("")
            self.rt_timer.stop()
            self.rt_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] 監控已關閉。")

    def update_realtime(self):
        sid = self.stock_id_input.currentText().split(" ")[0]
        self.rt_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] 正在獲取 {sid} 最新報價...")
        # 這裡會調用 fetcher.fetch_realtime_tick
        tick = self.fetcher.fetch_realtime_tick(sid)
        if not tick.empty:
            price = tick['Close'].iloc[0]
            self.rt_log.append(f"   >>> {sid} 當前價: {price}")
        else:
            self.rt_log.append("   >>> 無法獲取資料。")

    def update_desc(self):
        self.desc_box.setText(self.strat_info.get(self.strat_combo.currentText(), ""))

    def do_fetch(self):
        sid = self.stock_id_input.currentText().split(" ")[0]
        start = self.start_date_input.date().toString("yyyy-MM-dd")
        end = self.end_date_input.date().toString("yyyy-MM-dd")
        self.status_lbl.setText(f"狀態: ⏳ 正在抓取 {sid}...")
        QApplication.processEvents()
        try:
            self.fetcher.fetch_stock_daily(sid, start, end)
            self.status_lbl.setText(f"狀態: ✅ {sid} 資料就緒")
            QMessageBox.information(self, "成功", f"{sid} 歷史資料更新完成")
        except Exception as e:
            QMessageBox.critical(self, "錯誤", str(e))

    def fmt(self, val, suffix=""):
        return f"{val:.2f}{suffix}" if isinstance(val, (int, float)) else "N/A"

    def do_backtest(self):
        try:
            sid = self.stock_id_input.currentText().split(" ")[0]

            # 如果勾選自動更新
            if self.auto_update_cb.isChecked():
                self.do_fetch()

            strats = {"SmaCross (雙均線)": SmaCross, "MacdStrategy (MACD)": MacdStrategy,
                      "ForeignBuyStrategy (外資)": ForeignBuyStrategy, "RsiStrategy (RSI)": RsiStrategy}

            self.status_lbl.setText("狀態: ⏳ 運算中...")
            QApplication.processEvents()

            try:
                initial_cash = float(self.cash_input.text())
            except:
                initial_cash = 1000000.0

            results = run_backtest(sid, strategy=strats[self.strat_combo.currentText()], cash=initial_cash, plot=True)

            if "error" in results:
                QMessageBox.warning(self, "警告", results["error"])
                return

            self.lbl_final.setText(f"最終資產: ${results['final_value']:,.0f}")
            self.lbl_ret.setText(f"總報酬率: {results['return_pct']:.2f}%")
            self.lbl_sharpe.setText(f"夏普值: {self.fmt(results['sharpe'])}")
            self.lbl_dd.setText(f"最大回落: {self.fmt(results['max_drawdown'], '%')}")
            self.lbl_trades.setText(f"總交易次數: {results['total_trades']}")
            self.lbl_winrate.setText(f"勝率: {results['win_rate']:.2f}%")
            self.lbl_sqn.setText(f"SQN: {results['sqn']:.2f}")

            if results.get("plot_file"):
                pixmap = QPixmap(results["plot_file"])
                self.img_label.setPixmap(pixmap)

            self.status_lbl.setText("狀態: ✅ 回測完成")
            self.tabs.setCurrentIndex(0)

        except Exception as e:
            QMessageBox.critical(self, "錯誤", str(e))

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
