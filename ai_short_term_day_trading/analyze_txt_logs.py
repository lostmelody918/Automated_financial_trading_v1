import os
import glob

plots_dir = os.path.join(os.path.dirname(__file__), 'data_learn', 'trade_plots')
txt_files = glob.glob(os.path.join(plots_dir, '*.txt'))

losses = []
wins = []

for file in txt_files:
    is_win = 'WIN' in os.path.basename(file)
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    data = {}
    for line in lines:
        if ':' in line:
            parts = line.strip().split(':', 1)
            key = parts[0].strip()
            val = parts[1].strip()
            
            try:
                # Try to convert to float
                if '%' in val:
                    val = float(val.replace('%', '')) / 100
                else:
                    val = float(val)
            except ValueError:
                pass
            
            data[key] = val
            
    if is_win:
        wins.append(data)
    else:
        losses.append(data)

def print_avg(feature, wins, losses):
    win_vals = [w[feature] for w in wins if feature in w and isinstance(w[feature], float)]
    loss_vals = [l[feature] for l in losses if feature in l and isinstance(l[feature], float)]
    
    avg_win = sum(win_vals)/len(win_vals) if win_vals else 0
    avg_loss = sum(loss_vals)/len(loss_vals) if loss_vals else 0
    print(f"{feature:15s} | Win: {avg_win:8.4f} | Loss: {avg_loss:8.4f}")

print(f"Total Wins: {len(wins)}, Total Losses: {len(losses)}")
print("\n--- Feature Analysis ---")
features_to_analyze = ['Return', 'prob_up', 'prob_down', 'atr', 'macd_hist', 'rsi', 'vwap_bias', 'vix_ret_1d', 'n225_ret', 'ixic_ret_1d']

for feat in features_to_analyze:
    print_avg(feat, wins, losses)

# Further deep analysis on Losses
print("\n--- Deep Loss Analysis ---")
high_prob_loss = sum(1 for l in losses if l.get('prob_up', 0) > 0.6 or l.get('prob_down', 0) > 0.6)
print(f"High AI Confidence Losses (>0.6): {high_prob_loss} / {len(losses)}")

counter_trend = sum(1 for l in losses if (l.get('signal') == 1 and l.get('macd_hist', 0) < 0) or (l.get('signal') == -1 and l.get('macd_hist', 0) > 0))
print(f"Counter Trend MACD Losses: {counter_trend} / {len(losses)}")

high_volatility = sum(1 for l in losses if l.get('atr', 0) > 100)
print(f"High Volatility (ATR>100) Losses: {high_volatility} / {len(losses)}")

print("\n--- Recommendation for Strategy Update ---")
print("1. Strategy MUST capture large swings: AI should enter when volume surges (we will add Volume and Momentum checks in Simulator).")
print("2. Dynamic TP: Instead of fixed 0.8%, trailing stop based on MACD turning or Volume spiking.")
print("3. Hard SL: Keep a firm SL, e.g., -0.6% to prevent blowups, but trail TP aggressively.")
