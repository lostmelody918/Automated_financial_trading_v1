import os
import shioaji as sj
from dotenv import load_dotenv

load_dotenv('.env')
api = sj.Shioaji()
api.login(os.getenv('SHIOAJI_API_KEY'), os.getenv('SHIOAJI_SECRET_KEY'), contracts_timeout=10000)

for cat in ['TXO', 'TX1', 'TX2', 'TX4', 'TX5']:
    if hasattr(api.Contracts.Options, cat):
        contracts = list(getattr(api.Contracts.Options, cat))
        dates = set([getattr(c, 'delivery_date', 'N/A') for c in contracts])
        print(f'{cat}: {len(contracts)} contracts, Dates: {dates}')









