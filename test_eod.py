import sys
import traceback
sys.path.insert(0, 'F:\\Gemini_CLI_Application\\finance_v2\\ai_short_term_day_trading')
try:
    from live_option_simulator_v2 import generate_eod_report
    trade_log = [{'entry_time': '2023-01-01', 'exit_time': '2023-01-01 10:00:00', 'symbol': 'TXO', 'direction': 'Call', 'entry_price': 100, 'exit_price': 150, 'pnl': 5000, 'ret': 0.5, 'reason': 'test', 'feature1': 1.5, 'feature2': 2.3}]
    print("calling generate_eod_report...")
    generate_eod_report(trade_log, 100000, 105000, out_dir="test_data_learn")
    print("call completed.")
except BaseException as e:
    print("Caught exception:")
    traceback.print_exc()
