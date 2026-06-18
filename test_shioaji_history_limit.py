import os
import shioaji as sj
from datetime import datetime, timedelta
from dotenv import load_dotenv

env_path = r"F:\Gemini_CLI_Application\finance_v2\.env"
load_dotenv(dotenv_path=env_path)

api_key = os.environ.get('SHIOAJI_API_KEY', '')
secret_key = os.environ.get('SHIOAJI_SECRET_KEY', '')

api = sj.Shioaji()
api.login(api_key, secret_key)

# Try to fetch 1 day from 1 year ago
target_date = datetime.now() - timedelta(days=365)
start_str = target_date.strftime("%Y-%m-%d")
end_str = (target_date + timedelta(days=1)).strftime("%Y-%m-%d")

print(f"Fetching for {start_str} to {end_str}")
try:
    contract = api.Contracts.Futures.TXF.TXFR1
    kbars = api.kbars(contract, start=start_str, end=end_str)
    if kbars and len(kbars.ts) > 0:
        print(f"✅ Success! Got {len(kbars.ts)} bars.")
    else:
        print("❌ No data for this period.")
except Exception as e:
    print(f"❌ Error: {e}")

# Try to fetch 1 day from 2 years ago
target_date = datetime.now() - timedelta(days=730)
start_str = target_date.strftime("%Y-%m-%d")
end_str = (target_date + timedelta(days=1)).strftime("%Y-%m-%d")

print(f"Fetching for {start_str} to {end_str}")
try:
    kbars = api.kbars(contract, start=start_str, end=end_str)
    if kbars and len(kbars.ts) > 0:
        print(f"✅ Success! Got {len(kbars.ts)} bars.")
    else:
        print("❌ No data for this period.")
except Exception as e:
    print(f"❌ Error: {e}")
