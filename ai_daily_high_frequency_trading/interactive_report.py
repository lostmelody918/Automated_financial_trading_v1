import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

def create_interactive_report():
    out_dir = os.path.dirname(__file__)
    csv_path = os.path.join(out_dir, "reports", "full_backtest_data.csv")
    
    if not os.path.exists(csv_path):
        print(f"Error: Data file not found at {csv_path}. Please run hft_main.py first.")
        return
    
    print("Loading data for report generation...")
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    
    print("Generating interactive plots...")
    # Create subplots
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, subplot_titles=('Price & Signals', 'Capital Curve', 'Position', 'Indicators'),
                        row_heights=[0.5, 0.2, 0.1, 0.2])

    # Candlestick
    fig.add_trace(go.Candlestick(x=df['date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
    
    # BB & VWAP
    fig.add_trace(go.Scatter(x=df['date'], y=df['bb_up'], line=dict(color='rgba(173,216,230,0.5)', width=1), name='BB Up'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['bb_dn'], line=dict(color='rgba(173,216,230,0.5)', width=1), name='BB Dn', fill='tonexty'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['vwap'], line=dict(color='orange', width=2), name='VWAP'), row=1, col=1)
    
    # Signals
    longs = df[df['long_signal'] == True]
    shorts = df[df['short_signal'] == True]
    fig.add_trace(go.Scatter(x=longs['date'], y=longs['Low'] * 0.99, mode='markers', marker=dict(symbol='triangle-up', color='green', size=12), name='Long Signal'), row=1, col=1)
    fig.add_trace(go.Scatter(x=shorts['date'], y=shorts['High'] * 1.01, mode='markers', marker=dict(symbol='triangle-down', color='red', size=12), name='Short Signal'), row=1, col=1)

    # Capital
    fig.add_trace(go.Scatter(x=df['date'], y=df['capital'], line=dict(color='#00ff00', width=2), name='Capital'), row=2, col=1)
    
    # Position
    fig.add_trace(go.Scatter(x=df['date'], y=df['position'], line=dict(color='purple', width=2, shape='hv'), name='Position'), row=3, col=1)
    
    # Indicators
    fig.add_trace(go.Scatter(x=df['date'], y=df['bbw'], line=dict(color='gray', width=1.5), name='BBW'), row=4, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['adx'], line=dict(color='magenta', width=1.5), name='ADX'), row=4, col=1)
    
    fig.update_layout(title='HFT Options Buyer Strategy - Interactive Report', height=1000, template='plotly_dark')
    fig.update_xaxes(rangeslider_visible=False)
    
    html_path = os.path.join(out_dir, "reports", "interactive_report.html")
    fig.write_html(html_path)
    print(f"✅ Interactive HTML report generated at {html_path}")

if __name__ == '__main__':
    create_interactive_report()
