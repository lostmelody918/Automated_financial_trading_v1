import os
from dotenv import load_dotenv
import shioaji as sj
import pandas as pd
from datetime import datetime

load_dotenv()
api = sj.Shioaji()
api.login(
    api_key=os.environ.get('SHIOAJI_API_KEY'),
    secret_key=os.environ.get('SHIOAJI_SECRET_KEY')
)

contract = api.Contracts.Futures.TXF.TXFR1
ticks = api.ticks(contract, datetime.today().strftime('%Y-%m-%d'))
df = pd.DataFrame({**ticks})
print(df.columns)
print(df.head())
