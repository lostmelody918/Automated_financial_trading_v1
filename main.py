import argparse
from data_fetcher import DataFetcher
from bt_setup import run_backtest, SmaCross

def fetch_data(args):
    """Handler for the fetch command."""
    fetcher = DataFetcher()
    fetcher.fetch_stock_daily(args.stock_id, args.start_date, args.end_date)
    print(f"Data fetching for {args.stock_id} complete.")

def run_strategy(args):
    """Handler for the backtest command."""
    print(f"Running backtest for {args.stock_id}...")
    
    # Currently only SmaCross is implemented
    if args.strategy == 'smacross':
        run_backtest(args.stock_id, strategy=SmaCross, cash=args.cash)
    else:
        print(f"Strategy {args.strategy} not recognized.")

def main():
    parser = argparse.ArgumentParser(description="Taiwan Financial Market Analysis & Trading System")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Fetch Command
    fetch_parser = subparsers.add_parser("fetch", help="Fetch stock data from FinMind")
    fetch_parser.add_argument("stock_id", type=str, help="Stock ID (e.g., 2330)")
    fetch_parser.add_argument("start_date", type=str, help="Start date (YYYY-MM-DD)")
    fetch_parser.add_argument("--end_date", type=str, default=None, help="End date (YYYY-MM-DD) - Optional")

    # Backtest Command
    backtest_parser = subparsers.add_parser("backtest", help="Run a strategy backtest")
    backtest_parser.add_argument("stock_id", type=str, help="Stock ID (e.g., 2330)")
    backtest_parser.add_argument("--strategy", type=str, default="smacross", help="Strategy to run (default: smacross)")
    backtest_parser.add_argument("--cash", type=float, default=1000000.0, help="Initial cash for the portfolio")

    args = parser.parse_args()

    if args.command == "fetch":
        fetch_data(args)
    elif args.command == "backtest":
        run_strategy(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()