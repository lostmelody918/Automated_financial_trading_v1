import backtrader as bt
import pandas as pd
import numpy as np
from database import DatabaseManager
from datetime import datetime
import matplotlib
matplotlib.use('Agg') # 安全模式：不開啟任何視窗，專注於輸出圖片
import mplfinance as mpf

class PandasData(bt.feeds.PandasData):
    lines = ('foreign', 'trust', 'dealer',)
    params = (
        ('datetime', None), ('open', 'Open'), ('high', 'High'),
        ('low', 'Low'), ('close', 'Close'), ('volume', 'Volume'),
        ('openinterest', -1), ('foreign', 'Foreign'),
        ('trust', 'Trust'), ('dealer', 'Dealer'),
    )

class TaiwanStockCommission(bt.CommInfoBase):
    params = (('commission', 0.001425), ('tax', 0.003), ('stocklike', True), ('commtype', bt.CommInfoBase.COMM_PERC))
    def _getcommission(self, size, price, pseudoexec):
        comm = size * price * self.p.commission
        if size < 0: comm += abs(size) * price * self.p.tax
        return comm

class SmaCross(bt.Strategy):
    params = dict(pfast=10, pslow=30)
    def __init__(self):
        self.crossover = bt.ind.CrossOver(bt.ind.SMA(period=self.p.pfast), bt.ind.SMA(period=self.p.pslow))
    def next(self):
        if not self.position and self.crossover > 0: self.buy()
        elif self.crossover < 0: self.close()

class MacdStrategy(bt.Strategy):
    params = dict(macd1=12, macd2=26, macdsig=9)
    def __init__(self):
        self.macd = bt.ind.MACD(period_me1=self.p.macd1, period_me2=self.p.macd2, period_signal=self.p.macdsig)
        self.crossover = bt.ind.CrossOver(self.macd.macd, self.macd.signal)
    def next(self):
        if not self.position and self.crossover > 0: self.buy()
        elif self.crossover < 0: self.close()

class ForeignBuyStrategy(bt.Strategy):
    def __init__(self): self.foreign = self.data.foreign
    def next(self):
        if len(self) < 2: return
        if not self.position and self.foreign[0] > 0 and self.foreign[-1] > 0: self.buy()
        elif self.position and self.foreign[0] < 0: self.close()

class RsiStrategy(bt.Strategy):
    params = dict(period=14, low=30, high=70)
    def __init__(self): self.rsi = bt.ind.RSI(period=self.p.period)
    def next(self):
        if not self.position and self.rsi < self.p.low: self.buy()
        elif self.position and self.rsi > self.p.high: self.close()

def run_backtest(stock_id: str, strategy=SmaCross, cash: float = 1000000.0, plot: bool = False):
    db = DatabaseManager()
    df = db.load_dataframe(f"stock_{stock_id}_daily")
    if df.empty or len(df) < 50: return {"error": f"找不到 {stock_id} 足夠的歷史資料 (需大於 50 筆)。"}
    
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    df = df.ffill().fillna(0) # 徹底清除空值
    
    cerebro = bt.Cerebro()
    cerebro.adddata(PandasData(dataname=df))
    cerebro.addstrategy(strategy)
    cerebro.broker.setcash(cash)
    cerebro.broker.addcommissioninfo(TaiwanStockCommission())
    cerebro.addsizer(bt.sizers.SizerFix, stake=1000)
    
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade')
    cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')

    results = cerebro.run()
    strat = results[0]
    
    final_val = cerebro.broker.getvalue()
    ret_pct = (final_val - cash) / cash * 100
    sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', None)
    max_dd = strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', None)
    
    # New metrics
    trade_analysis = strat.analyzers.trade.get_analysis()
    total_trades = trade_analysis.total.total if 'total' in trade_analysis else 0
    win_rate = (trade_analysis.won.total / total_trades * 100) if total_trades > 0 else 0
    sqn = strat.analyzers.sqn.get_analysis().get('sqn', 0)

    timerets = pd.Series(strat.analyzers.timereturn.get_analysis())
    std_dev = timerets.std() * (252**0.5) * 100 if not timerets.empty else None
    
    beta = "N/A"
    try:
        taiex_df = db.load_dataframe("stock_TAIEX_daily")
        if not taiex_df.empty and stock_id != "TAIEX":
            taiex_df['date'] = pd.to_datetime(taiex_df['date'])
            taiex_df.set_index('date', inplace=True)
            taiex_rets = taiex_df['Close'].pct_change()
            combined = pd.concat([timerets, taiex_rets], axis=1).dropna()
            if len(combined) > 5:
                beta = np.cov(combined.iloc[:, 0], combined.iloc[:, 1])[0, 1] / np.var(combined.iloc[:, 1])
    except: pass

    plot_file = None
    plot_error = None
    if plot:
        try:
            # 不再使用 .tail(200)，直接使用所有資料以對齊開始結束日期
            plot_df = df.copy() 
            add_plots = []
            if strategy == SmaCross:
                add_plots.append(mpf.make_addplot(plot_df['Close'].rolling(10).mean(), color='blue'))
                add_plots.append(mpf.make_addplot(plot_df['Close'].rolling(30).mean(), color='orange'))
            elif strategy == MacdStrategy:
                exp1 = plot_df['Close'].ewm(span=12).mean()
                exp2 = plot_df['Close'].ewm(span=26).mean()
                macd = exp1 - exp2
                signal = macd.ewm(span=9).mean()
                add_plots.append(mpf.make_addplot(macd, panel=1, color='fuchsia', ylabel='MACD'))
                add_plots.append(mpf.make_addplot(signal, panel=1, color='b'))
            elif strategy == RsiStrategy:
                delta = plot_df['Close'].diff()
                up, down = delta.copy(), delta.copy()
                up[up < 0] = 0; down[down > 0] = 0
                rsi = 100.0 - (100.0 / (1.0 + up.rolling(14).mean() / (down.abs().rolling(14).mean() + 1e-9)))
                add_plots.append(mpf.make_addplot(rsi, panel=1, color='blue', ylabel='RSI'))
            elif strategy == ForeignBuyStrategy:
                add_plots.append(mpf.make_addplot(plot_df['Foreign'], type='bar', panel=1, color='purple', alpha=0.5))

            mc = mpf.make_marketcolors(up='red', down='green', inherit=True)
            s = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', y_on_right=False)
            
            output_filename = f"backtest_plot_current.png"
            mpf.plot(plot_df, type='candle', volume=True, addplot=add_plots, style=s, 
                     savefig=dict(fname=output_filename, dpi=100, bbox_inches='tight'), 
                     figsize=(10, 6))
            plot_file = output_filename
        except Exception as e:
            plot_error = str(e)

    return {
        "final_value": final_val, "return_pct": ret_pct,
        "sharpe": sharpe if sharpe else "N/A", "max_drawdown": max_dd if max_dd else "N/A",
        "std_dev": std_dev if std_dev else "N/A", "beta": beta, 
        "total_trades": total_trades, "win_rate": win_rate, "sqn": sqn,
        "plot_file": plot_file, "plot_error": plot_error
    }
