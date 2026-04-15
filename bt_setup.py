import backtrader as bt
import pandas as pd
from database import DatabaseManager
from datetime import datetime

class PandasData(bt.feeds.PandasData):
    """
    Custom Data Feed for Backtrader to consume Pandas DataFrames with FinMind data format.
    FinMind column names: date, stock_id, Trading_Volume (Volume), Trading_money (Turnover),
    open (Open), max (High), min (Low), close (Close), spread (Change), Trading_turnover (Transactions)
    """
    
    params = (
        ('datetime', None),
        ('open', 'Open'),
        ('high', 'High'),
        ('low', 'Low'),
        ('close', 'Close'),
        ('volume', 'Volume'),
        ('openinterest', -1), # Not available in standard stock data
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

def run_backtest(stock_id: str, strategy=SmaCross, cash: float = 1000000.0):
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
    # Taiwan stocks are traded in lots of 1000 shares (or odd lots). Let's use 1000 for simplicity.
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
    
    # Plot results
    # cerebro.plot(style='candlestick')

if __name__ == '__main__':
    # You would normally run data_fetcher.py first to populate the DB
    run_backtest('2330')
