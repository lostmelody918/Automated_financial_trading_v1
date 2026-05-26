import re

file_path = r"F:\Gemini_CLI_Application\finance_v2\ai_short_term_day_trading\live_option_simulator.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Add matplotlib imports
if "import matplotlib.pyplot as plt" not in content:
    content = re.sub(
        r"import pandas as pd",
        "import pandas as pd\nimport matplotlib.pyplot as plt\nimport matplotlib.dates as mdates",
        content
    )

# 2. Add generate_eod_report function before run_live_simulator
report_func = """
def generate_eod_report(daily_trades, df, best_contract, current_capital, today_date):
    print("📊 正在產出今日選擇權損益報告與交易圖表...")
    
    # 建立保存目錄
    report_dir = os.path.join(os.path.dirname(__file__), "daily_reports")
    os.makedirs(report_dir, exist_ok=True)
    
    report_filename = os.path.join(report_dir, f"EOD_Report_{today_date.strftime('%Y%m%d')}.png")
    
    # 準備繪圖資料
    # 我們只取今天的資料
    today_df = df[df['date'].dt.date == today_date].copy()
    if today_df.empty:
        print("⚠️ 無法產出報告：缺少今日 K 線資料。")
        return
        
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.3)
    
    # 1. 價格與出手時機圖
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(today_df['date'], today_df['Close'], label='Close Price', color='black', linewidth=1.5)
    ax1.set_title(f"[{today_date}] {best_contract.symbol} Intraday Price & Trading Timing", fontsize=16, fontweight='bold')
    ax1.set_ylabel("Price")
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # 標記進出場點
    total_pnl = 0
    trade_details = []
    
    for i, trade in enumerate(daily_trades):
        total_pnl += trade['pnl']
        # 買進標記
        marker_color = 'green' if '多' in trade['direction'] or 'Call' in trade['direction'] else 'red'
        marker_symbol = '^' if '多' in trade['direction'] or 'Call' in trade['direction'] else 'v'
        
        ax1.scatter(trade['entry_time'], trade['entry_price'], color=marker_color, marker=marker_symbol, s=150, zorder=5, label=f"Entry {i+1}" if i==0 else "")
        
        # 賣出標記
        exit_color = 'blue' if trade['pnl'] > 0 else 'orange'
        ax1.scatter(trade['exit_time'], trade['exit_price'], color=exit_color, marker='x', s=150, zorder=5, label=f"Exit {i+1}" if i==0 else "")
        
        # 連接線
        ax1.plot([trade['entry_time'], trade['exit_time']], [trade['entry_price'], trade['exit_price']], color='gray', linestyle=':', linewidth=1)
        
        trade_details.append(
            f"Trade {i+1}: {trade['direction']} | Entry: {trade['entry_time'].strftime('%H:%M:%S')} @ {trade['entry_price']:.2f} | "
            f"Exit: {trade['exit_time'].strftime('%H:%M:%S')} @ {trade['exit_price']:.2f} | "
            f"PnL: NT$ {trade['pnl']:,.0f}"
        )

    ax1.legend(loc='best')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    # 2. 量能/籌碼圖 (這裡用成交量 Volume 代表)
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    # 分辨紅黑K量
    colors = ['red' if c >= o else 'green' for c, o in zip(today_df['Close'], today_df['Open'])]
    ax2.bar(today_df['date'], today_df['Volume'], color=colors, alpha=0.7, width=0.001)
    ax2.set_title("Volume (Trading Activity)", fontsize=14)
    ax2.set_ylabel("Volume")
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # 3. 損益文字報告區塊
    ax3 = fig.add_subplot(gs[2])
    ax3.axis('off')
    
    summary_text = (
        f"📅 Date: {today_date}\\n"
        f"💰 Total Daily PnL: NT$ {total_pnl:,.0f}\\n"
        f"🏦 Current Capital: NT$ {current_capital:,.0f}\\n"
        f"📊 Number of Trades: {len(daily_trades)}\\n\\n"
        f"--- Trade Details ---\\n"
    )
    
    for detail in trade_details:
        summary_text += detail + "\\n"
        
    if not daily_trades:
        summary_text += "No trades executed today.\\n"
        
    ax3.text(0.01, 0.95, summary_text, fontsize=12, family='monospace', verticalalignment='top', 
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='gray', alpha=0.8))

    plt.tight_layout()
    plt.savefig(report_filename, dpi=150)
    plt.close()
    print(f"✅ 今日損益報告已匯出至: {report_filename}")

def is_market_open"""

if "def generate_eod_report" not in content:
    content = content.replace("def is_market_open", report_func)

# 3. Add state variables for report tracking inside run_live_simulator
if "daily_trades = []" not in content:
    content = re.sub(
        r"trailing_stop_price = 0.0\s*print\(\"📡",
        "trailing_stop_price = 0.0\n\n    daily_trades = []\n    eod_report_generated_date = None\n    entry_time = None\n\n    print(\"📡",
        content
    )

# 4. Modify event loop variables
if "today_date = now.date()" not in content:
    content = content.replace("current_time = now.time()", "current_time = now.time()\n        today_date = now.date()")

# 5. Handle EOD regardless of position
eod_guard = """        # A. 非交易時間守衛
        if not is_market_open(current_time):
            if position != 0:
                print(f"[{time_str}] ⚠️ 異常警訊：非交易時間仍持有虛擬部位，執行系統強制清倉。")
                position = 0
                num_contracts = 0
                trade_capital_used = 0
            print(f"[{time_str}] 💤 目前為非交易時段，系統休眠中。等待日盤開盤 (08:45)...")
            
            # 確保非交易時間重置隔日產報表狀態
            if current_time < datetime_time(8, 45):
                pass # 這裡不需要處理，但可以保持空狀態
                
            time.sleep(60)
            continue"""
# Just replace not is_market_open block
# Actually, the best place to intercept EOD is exactly here:
eod_interceptor = """        # B. 檢查與執行尾盤產出報告 (無持倉時)
        if is_eod_closing_time(current_time) and position == 0:
            if eod_report_generated_date != today_date:
                try:
                    df, best_contract = engine.fetch_active_option_intraday_data(days=1)
                    generate_eod_report(daily_trades, df, best_contract, current_capital, today_date)
                except Exception as e:
                    print(f"[{time_str}] ⚠️ 產出尾盤報告失敗: {e}")
                eod_report_generated_date = today_date
                daily_trades.clear()
                print("💤 已完成本日尾盤結算，暫停盤中交易，等待明日開盤...")
                time.sleep(300)
                continue"""

if "# B. 檢查與執行尾盤產出報告" not in content:
    content = content.replace(
        "        try:\n            # B. 抓取盤中即時資料", 
        eod_interceptor + "\n\n        try:\n            # B. 抓取盤中即時資料"
    )

# 6. Record trades when position is opened
entry_tracker = """                    entry_price = current_price
                    entry_time = now"""
if "entry_time = now" not in content:
    content = content.replace("entry_price = current_price", entry_tracker)

# 7. Record trades and generate report when position is closed
exit_logic = """                # 執行虛擬平倉結算
                if exit_reason:
                    current_capital += current_pnl
                    last_trade_win = current_pnl > 0
                    
                    # 紀錄這筆交易
                    daily_trades.append({
                        'entry_time': entry_time,
                        'exit_time': now,
                        'direction': 'Buy Call' if position == 1 else 'Buy Put',
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'contracts': num_contracts,
                        'pnl': current_pnl,
                        'return': current_ret
                    })

                    print("" + "="*50)"""

if "daily_trades.append({" not in content:
    content = content.replace("""                # 執行虛擬平倉結算
                if exit_reason:
                    current_capital += current_pnl
                    last_trade_win = current_pnl > 0

                    print("" + "="*50)""", exit_logic)

# 8. Report generation on EOD exit
eod_report_exit = """                    # 若為尾盤強制平倉，則直接讓系統休息至收盤
                    if "EOD" in exit_reason:
                        if eod_report_generated_date != today_date:
                            generate_eod_report(daily_trades, df, best_contract, current_capital, today_date)
                            eod_report_generated_date = today_date
                            daily_trades.clear()
                        print("💤 已完成本日尾盤結算，暫停盤中交易，等待明日開盤...")
                        time.sleep(300)"""

if "generate_eod_report(" not in eod_report_exit: # Need to check the string replacement accurately
    # Let's use regex for safety
    pass

content = re.sub(
    r"# 若為尾盤強制平倉，則直接讓系統休息至收盤\s+if \"EOD\" in exit_reason:\s+print\(\"💤 已完成本日尾盤結算，暫停盤中交易，等待明日開盤\.\.\.\"\)\s+time\.sleep\(300\)",
    """# 若為尾盤強制平倉，則直接讓系統休息至收盤
                    if "EOD" in exit_reason:
                        if eod_report_generated_date != today_date:
                            generate_eod_report(daily_trades, df, best_contract, current_capital, today_date)
                            eod_report_generated_date = today_date
                            daily_trades.clear()
                        print("💤 已完成本日尾盤結算，暫停盤中交易，等待明日開盤...")
                        time.sleep(300)""",
    content
)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)
print("Patch applied successfully.")
