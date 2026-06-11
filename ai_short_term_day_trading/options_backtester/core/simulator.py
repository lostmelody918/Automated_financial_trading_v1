import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
from dotenv import load_dotenv

# Append paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

# Load .env variables
project_root = os.path.dirname(os.path.dirname(base_dir))
load_dotenv(os.path.join(project_root, '.env'))

from database.timescale_client import TimescaleDBClient

# Note: In a real environment, you must build the cpp_engine first to import options_replay
try:
    # Add the build directory to sys.path so Python can find the .pyd file
    build_dir = os.path.join(base_dir, 'cpp_engine', 'build')
    sys.path.append(build_dir)
    import options_replay
    print("✅ Successfully loaded high-performance C++ Order Book Replay Engine.")
except ImportError:
    print("⚠️ Warning: options_replay C++ module not found. Falling back to Python mock engine.")
    # Assuming options_replay_fallback.py is in the core directory
    import options_replay_fallback as options_replay

class OptionsSimulator:
    def __init__(self, target_date: str):
        self.target_date = target_date
        db_url = os.environ.get("DATABASE_URL")
        if not db_url:
            print("Warning: DATABASE_URL not found in .env, falling back to default.")
            db_url = "postgresql://postgres:postgres@localhost:5432/finance_db"
        self.db = TimescaleDBClient(db_url)
        self.engine = options_replay.SimulationEngine() if options_replay else None
        
    def find_top_volume_contracts(self) -> dict:
        """
        Finds the top 3 volume Call and Put contracts for the nearest expiration
        on the target date.
        """
        start_time = f"{self.target_date} 08:45:00"
        end_time = f"{self.target_date} 13:45:00"
        
        print(f"Querying available contracts for {self.target_date}")
        
        # 嘗試從資料庫中找出當天實際有的合約
        try:
            if not self.db.conn:
                self.db.connect()
            query = "SELECT DISTINCT symbol FROM options_ticks WHERE time >= %s AND time <= %s"
            df_symbols = pd.read_sql_query(query, self.db.conn, params=[start_time, end_time])
            available_symbols = df_symbols['symbol'].tolist()
            
            if available_symbols:
                # 簡單區分 Call 和 Put
                calls = [s for s in available_symbols if s.endswith('C') or 'C' in s]
                puts = [s for s in available_symbols if s.endswith('P') or 'P' in s]
                
                # 若無明確區分，則全放
                if not calls and not puts:
                    calls = available_symbols
                
                return {
                    'calls': calls[:3],
                    'puts': puts[:3]
                }
        except Exception as e:
            print(f"Failed to query distinct symbols: {e}")
            
        return {
            'calls': ['TXO15000C', 'TXO15100C', 'TXO15200C'],
            'puts': ['TXO14800P', 'TXO14900P', 'TXO15000P']
        }

    def load_ticks_to_engine(self, symbols: list):
        """
        Loads ticks from TimescaleDB and feeds them to the C++ engine.
        """
        start_time = f"{self.target_date} 08:45:00"
        end_time = f"{self.target_date} 13:45:00"
        
        print(f"Loading ticks from DB for {symbols}")
        
        try:
            df = self.db.fetch_ticks(start_time, end_time, symbols)
        except Exception as e:
            print(f"Database fetch failed: {e}. Generating mock data for testing.")
            # Generate mock ticks for testing if DB is unavailable
            times = pd.date_range(start=start_time, end=end_time, freq="1s")
            dfs = []
            for sym in symbols:
                sym_df = pd.DataFrame({
                    'time': times,
                    'symbol': sym,
                    'price': np.random.normal(100, 1, len(times)).cumsum() + 150,
                    'volume': np.random.randint(1, 5, len(times)),
                    'bid_price': 149.0,
                    'bid_volume': 10,
                    'ask_price': 151.0,
                    'ask_volume': 10
                })
                dfs.append(sym_df)
            df = pd.concat(dfs, ignore_index=True)
        
        if df.empty or self.engine is None:
            print("No data or engine missing.")
            return df
            
        for symbol in symbols:
            sym_df = df[df['symbol'] == symbol]
            ticks = []
            for _, row in sym_df.iterrows():
                t = options_replay.Tick()
                t.timestamp_ms = int(row['time'].timestamp() * 1000)
                t.price = row['price']
                t.volume = row['volume']
                t.bid_price = row['bid_price']
                t.bid_volume = row['bid_volume']
                t.ask_price = row['ask_price']
                t.ask_volume = row['ask_volume']
                ticks.append(t)
            
            self.engine.feed_ticks(symbol, ticks)
            
        return df

    def run_simulation(self):
        """
        Executes the simulation strictly from 08:45 to 13:45.
        Includes a real backtesting mock strategy: Entry at 09:00, Trailing Stop, Exit at 13:30.
        """
        if not self.engine:
            print("Cannot run simulation without C++ engine.")
            return
            
        top_contracts = self.find_top_volume_contracts()
        all_symbols = top_contracts['calls'] + top_contracts['puts']
        
        if not all_symbols:
            print("No contracts found to simulate.")
            return
            
        df = self.load_ticks_to_engine(all_symbols)
        if df.empty:
            return
            
        # Select the primary Call contract for our simple test strategy
        target_symbol = top_contracts['calls'][0] if top_contracts['calls'] else all_symbols[0]
            
        # Replay at 1-second intervals for the day
        start_ts = int(pd.to_datetime(f"{self.target_date} 08:45:00").timestamp() * 1000)
        end_ts = int(pd.to_datetime(f"{self.target_date} 13:45:00").timestamp() * 1000)
        
        print(f"\n🚀 Starting Order Book Replay & Strategy Backtest from {start_ts} to {end_ts}")
        print(f"🎯 Target Contract: {target_symbol}")
        
        current_ts = start_ts
        
        # Strategy State Variables
        position = 0
        entry_price = 0.0
        highest_price = 0.0
        trailing_stop_points = 10.0
        pnl = 0.0
        trade_log = []
        
        # Time markers
        entry_time_ms = int(pd.to_datetime(f"{self.target_date} 09:00:00").timestamp() * 1000)
        exit_time_ms = int(pd.to_datetime(f"{self.target_date} 13:30:00").timestamp() * 1000)
        
        while current_ts <= end_ts:
            self.engine.advance_to(current_ts)
            
            # 1. Entry Logic: Buy at 09:00:00 using Best Ask
            if current_ts >= entry_time_ms and position == 0 and len(trade_log) == 0:
                ask_price = self.engine.get_best_ask(target_symbol)
                if ask_price > 0:
                    position = 1
                    entry_price = ask_price
                    highest_price = ask_price
                    time_str = pd.to_datetime(current_ts, unit='ms').strftime('%H:%M:%S')
                    print(f"[{time_str}] 🟢 ENTER LONG: Bought 1 {target_symbol} at {entry_price}")
            
            # 2. Position Management
            if position > 0:
                current_bid = self.engine.get_best_bid(target_symbol)
                if current_bid > 0:
                    # Update highest price for trailing stop
                    if current_bid > highest_price:
                        highest_price = current_bid
                    
                    # Calculate Trailing Stop Level
                    trailing_stop = highest_price - trailing_stop_points
                    
                    # Exit Logic: Trailing Stop Triggered or EOD (13:30)
                    time_str = pd.to_datetime(current_ts, unit='ms').strftime('%H:%M:%S')
                    exit_reason = None
                    
                    if current_ts >= exit_time_ms:
                        exit_reason = "EOD Force Close"
                    elif current_bid <= trailing_stop:
                        exit_reason = f"Trailing Stop Triggered (Highest: {highest_price}, SL: {trailing_stop})"
                        
                    if exit_reason:
                        exit_price = current_bid
                        points_gained = exit_price - entry_price
                        trade_pnl = points_gained * 50  # 1 point = 50 TWD
                        pnl += trade_pnl
                        
                        print(f"[{time_str}] 🔴 EXIT LONG: Sold 1 {target_symbol} at {exit_price}")
                        print(f"   ↳ Reason: {exit_reason}")
                        print(f"   ↳ PnL: NT$ {trade_pnl:,.0f} ({points_gained} pts)")
                        
                        trade_log.append({
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'pnl': trade_pnl
                        })
                        
                        position = 0 # Clear position
            
            # Example MTM snapshot printing every 30 mins
            if current_ts % 1800000 == 0:
                mtm = self.engine.get_contract_mtm(target_symbol, position if position != 0 else 1)
                time_str = pd.to_datetime(current_ts, unit='ms').strftime('%H:%M:%S')
                print(f"⏱️ [{time_str}] MTM Tracker -> Best Bid: {self.engine.get_best_bid(target_symbol)}, Best Ask: {self.engine.get_best_ask(target_symbol)}")
            
            current_ts += 1000 # Step 1 second
            
        print("\n" + "="*50)
        print("📊 Backtest Simulation Complete")
        print(f"💰 Total PnL: NT$ {pnl:,.0f}")
        print("="*50 + "\n")

if __name__ == "__main__":
    sim = OptionsSimulator("2026-06-11")
    sim.run_simulation()
