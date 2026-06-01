import requests
headers = {'User-Agent': 'Mozilla/5.0'}
r = requests.get('https://www.taifex.com.tw/cht/3/futContractsDate', headers=headers)
with open('taifex_test.html', 'w', encoding='utf-8') as f:
    f.write(r.text)
print("Saved to taifex_test.html")
