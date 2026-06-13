import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import sys


# streamlit run F:\Gemini_CLI_Application\finance_v2\ai_short_term_day_trading\options_backtester\ui_dashboard.py

# Add path so we can import simulator
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from core.simulator import OptionsSimulator

st.set_page_config(page_title="AI Options Backtester", layout="wide")

st.title("📈 AI Options Backtester Visualizer")
st.markdown("Run live trading logic (`live_option_simulator_v2.py`) on historical order book data to verify AI decisions, Safety Nets, and dynamic Stop-Loss/Take-Profit bounds.")

# Sidebar settings
st.sidebar.header("Backtest Settings")
target_date = st.sidebar.text_input("Target Date (YYYY-MM-DD)", value="2026-06-11")

if st.sidebar.button("🚀 Run Backtest"):
    with st.spinner(f"Running full simulation for {target_date}..."):
        try:
            # Initialize and run simulator
            sim = OptionsSimulator(target_date)
            trade_log, df_intraday = sim.run_simulation()

            st.success("Simulation Complete!")

            if not df_intraday.empty:
                # Plotly Chart for the Index (TXF)
                st.subheader(f"TXF Intraday Chart with Trade Execution ({target_date})")

                fig = go.Figure()

                # Add Index line
                fig.add_trace(go.Scatter(
                    x=df_intraday['time'],
                    y=df_intraday['Close'],
                    mode='lines',
                    name='TXF Close',
                    line=dict(color='gray', width=1)
                ))

                # Overlay Trade Entries
                if trade_log:
                    buy_times = []
                    buy_prices = []
                    sell_times = []
                    sell_prices = []

                    for trade in trade_log:
                        entry_t = pd.to_datetime(f"{target_date} {trade['entry_time']}").time()
                        # Find closest TXF price at entry time for plotting
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
                            mode='markers', name='Buy Call',
                            marker=dict(symbol='triangle-up', color='green', size=12)
                        ))
                    if sell_times:
                        fig.add_trace(go.Scatter(
                            x=sell_times, y=sell_prices,
                            mode='markers', name='Buy Put',
                            marker=dict(symbol='triangle-down', color='red', size=12)
                        ))

                fig.update_layout(height=500, xaxis_title="Time", yaxis_title="TXF Price", template='plotly_dark')
                st.plotly_chart(fig, use_container_width=True)

            # Display Trade Log
            st.subheader("Trade Log")
            if trade_log:
                df_trades = pd.DataFrame(trade_log)
                st.dataframe(df_trades.style.format({
                    'entry_price': '{:.2f}',
                    'exit_price': '{:.2f}',
                    'pnl': 'NT$ {:,.0f}'
                }))

                total_pnl = sum(t['pnl'] for t in trade_log)
                win_rate = sum(1 for t in trade_log if t['pnl'] > 0) / len(trade_log) * 100

                col1, col2, col3 = st.columns(3)
                col1.metric("Total Trades", len(trade_log))
                col2.metric("Win Rate", f"{win_rate:.1f}%")
                col3.metric("Total PnL", f"NT$ {total_pnl:,.0f}")

            else:
                st.info("No trades executed on this date.")

        except Exception as e:
            st.error(f"Error running simulation: {str(e)}")
