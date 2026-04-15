import backtrader as bt
import pandas as pd
from database import DatabaseManager
from datetime import datetime

class PandasData(bt.feeds.PandasData):
    """
    Custom Data Feed for Backtrader to consume Pandas DataFrames with FinMind data format
    and institutional investor data.
    """
    
    # Add custom lines for chip data
    lines = ('foreign', 'trust', 'dealer',)
    
    params = (
        ('datetime', None),
        ('open', 'Open'),
        ('high', 'High'),
        ('low', 'Low'),
        ('close', 'Close'),
        ('volume', 'Volume'),
        ('openinterest', -1),
        ('foreign', 'Foreign'), # Match the new columns from data_fetcher
        ('trust', 'Trust'),
        ('dealer', 'Dealer'),
    )

class TaiwanStockCommission(bt.CommInfoBase):
    """
    Custom commission scheme for Taiwan Stock Market.
    - Commission: 0.1425% (usually discounted, but using standard here)
    - Tax: 0.3% on sell only (stock transaction tax)
    """
    params = (
        ('commission', 0.001425),
        ('tax', 0.003),
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_PERC),
    )

    def _getcommission(self, size, price, pseudoexec):
        """
        Calculate commission. If selling, add transaction tax.
        """
        comm = size * price * self.p.commission
        if size < 0: # Selling
            tax = abs(size) * price * self.p.tax
            comm += tax
        return comm

class SmaCross(bt.Strategy):
    """
    A simple Moving Average Crossover strategy for testing.
    Buys when fast SMA crosses over slow SMA, sells when it crosses under.
    """
    params = dict(
        pfast=10,  # period for the fast moving average
        pslow=30   # period for the slow moving average
    )

    def __init__(self):
        sma1 = bt.ind.SMA(period=self.p.pfast)  # fast moving average
        sma2 = bt.ind.SMA(period=self.p.pslow)  # slow moving average
        self.crossover = bt.ind.CrossOver(sma1, sma2)  # crossover signal

    def next(self):
        if not self.position:  # not in the market
            if self.crossover > 0:  # if fast crosses slow to the upside
                self.buy()  # enter long
        elif self.crossover < 0:  # in the market & cross to the downside
            self.close()  # close long position

class MacdStrategy(bt.Strategy):
    """
    MACD Strategy:
    Buy when MACD line crosses above Signal line.
    Sell when MACD line crosses below Signal line.
    """
    params = dict(
        macd1=12,
        macd2=26,
        macdsig=9
    )

    def __init__(self):
        self.macd = bt.ind.MACD(period_me1=self.p.macd1, period_me2=self.p.macd2, period_signal=self.p.macdsig)
        self.crossover = bt.ind.CrossOver(self.macd.macd, self.macd.signal)

    def next(self):
        if not self.position:
            if self.crossover > 0:
                self.buy()
        elif self.crossover < 0:
            self.close()

class ForeignBuyStrategy(bt.Strategy):
    """
    Chip (Institutional) Strategy:
    Buy when Foreign Investors net buy is positive for 2 consecutive days.
    Sell when Foreign Investors net sell is negative.
    """
    def __init__(self):
        # We use data.foreign (which is the net buy from our PandasData)
        self.foreign_net_buy = self.data.foreign
    
    def next(self):
        if len(self) < 2:
            return # Need at least 2 days of data
            
        if not self.position:
            # Buy if foreign bought today and yesterday
            if self.foreign_net_buy[0] > 0 and self.foreign_net_buy[-1] > 0:
                self.buy()
        else:
            # Sell if foreign sells
            if self.foreign_net_buy[0] < 0:
                self.close()

def run_backtest(stock_id: str, strategy=SmaCross, cash: float = 1000000.0, plot: bool = False):
    """
    Runs a backtest for a specific stock using data from the local database.
    """
    db = DatabaseManager()
    table_name = f"stock_{stock_id}_daily"
    
    print(f"Loading data for {stock_id} from database...")
    df = db.load_dataframe(table_name)
    
    if df.empty:
        print(f"No data found for {stock_id} in the database. Please fetch it first.")
        return
        
    # Data preprocessing
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    
    print(f"Data loaded: {len(df)} rows. Starting backtest...")
    
    # Initialize Cerebro engine
    cerebro = bt.Cerebro()

    # Add data feed
    data = PandasData(dataname=df)
    cerebro.adddata(data)

    # Add strategy
    cerebro.addstrategy(strategy)

    # Set broker settings (initial cash)
    cerebro.broker.setcash(cash)

    # Add Taiwan stock commission and tax
    comminfo = TaiwanStockCommission()
    cerebro.broker.addcommissioninfo(comminfo)
    
    # Add sizers (how many shares to buy)
    cerebro.addsizer(bt.sizers.SizerFix, stake=1000)

    # Analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

    # Run Cerebro
    initial_value = cerebro.broker.getvalue()
    print(f'Starting Portfolio Value: {initial_value:.2f}')
    
    results = cerebro.run()
    strat = results[0]
    
    final_value = cerebro.broker.getvalue()
    print(f'Final Portfolio Value: {final_value:.2f}')
    print(f'Return: {(final_value - initial_value) / initial_value * 100:.2f}%')
    
    print('Sharpe Ratio:', strat.analyzers.sharpe.get_analysis().get('sharperatio', 'N/A'))
    print('Max Drawdown:', strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 'N/A'))
    
    if plot:
        # Save plot to an HTML file to avoid matplotlib blocking UI and errors
        try:
            import matplotlib
            matplotlib.use('Agg') # Use non-interactive backend
            fig = cerebro.plot(style='candlestick')[0][0]
            fig.savefig(f"backtest_{stock_id}_plot.png", dpi=300)
            print(f"Plot saved to backtest_{stock_id}_plot.png")
        except Exception as e:
            print(f"Plotting failed: {e}")

if __name__ == '__main__':
    run_backtest('2330')
