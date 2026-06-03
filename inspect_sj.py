
import shioaji as sj

api = sj.Shioaji()
print(f"Has api.quote: {hasattr(api, 'quote')}")
if hasattr(api, 'quote'):
    print(f"Dir api.quote: {dir(api.quote)}")

print(f"Dir api: {dir(api)}")
