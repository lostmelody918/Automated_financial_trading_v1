import os
import shioaji as sj
from database import DatabaseManager
from datetime import datetime, timedelta
import pandas as pd
import requests
import urllib3
import re
from dotenv import load_dotenv

# Fallbacks for data Shioaji does not provide
from FinMind.data import DataLoader
import yfinance as yf

load_dotenv()

class DataFetcher:
    """金融數據抓取器 - 使用 Shioaji 提升即時性"""
    
    def __init__(self, api_token: str = None):
        self.api = sj.Shioaji()
        api_key = os.environ.get('SHIOAJI_API_KEY', '')
        secret_key = os.environ.get('SHIOAJI_SECRET_KEY', '')
        
        # 只在有 key 的情況下嘗試登入
        if api_key and secret_key:
            try:
                # 使用 contracts_timeout 確保合約下載完成，為自動化交易做準備
                self.api.login(api_key, secret_key, contracts_timeout=10000)
            except Exception as e:
                print(f"Shioaji login failed: {e}")
        else:
            print("Warning: SHIOAJI_API_KEY or SHIOAJI_SECRET_KEY not found in environment.")
            
        self.db = DatabaseManager()
        self.dl = DataLoader() # FinMind fallback
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    def fetch_stock_intraday(self, stock_id: str, start_date: str, end_date: str = None):
        """Fetch 1-min kbars for intraday analysis (自動化交易/高頻當沖使用)"""
        if not end_date: end_date = datetime.today().strftime('%Y-%m-%d')
        try:
            try:
                contract = self.api.Contracts.Stocks[stock_id]
            except KeyError:
                print(f"Contract not found for stock {stock_id}")
                return pd.DataFrame()
                
            kbars = self.api.kbars(
                contract=contract, 
                start=start_date, 
                end=end_date
            )
            df = pd.DataFrame({**kbars})
            if df.empty: return pd.DataFrame()
            
            df.ts = pd.to_datetime(df.ts)
            self.db.save_dataframe(df, f"stock_{stock_id}_intraday")
            return df
        except Exception as e:
            print(f"Error in fetch_stock_intraday: {e}")
            return pd.DataFrame()

    def fetch_stock_daily(self, stock_id: str, start_date: str, end_date: str = None):
        if not end_date: end_date = datetime.today().strftime('%Y-%m-%d')
        try:
            try:
                contract = self.api.Contracts.Stocks[stock_id]
            except KeyError:
                print(f"Contract not found for stock {stock_id}")
                return pd.DataFrame()
                
            kbars = self.api.kbars(
                contract=contract, 
                start=start_date, 
                end=end_date
            )
            df = pd.DataFrame({**kbars})
            if df.empty: return pd.DataFrame()
            
            df.ts = pd.to_datetime(df.ts)
            df['date'] = df.ts.dt.strftime('%Y-%m-%d')
            df = df.groupby('date').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).reset_index()
            
            # 嘗試用 FinMind 抓取法人買賣超 (Shioaji 無歷史法人資料)
            try:
                df_inst = self.dl.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=start_date, end_date=end_date)
                if df_inst is not None and not df_inst.empty:
                    df_inst['net_buy'] = df_inst['buy'] - df_inst['sell']
                    name_map = {'Foreign_Investor': 'Foreign', 'Investment_Trust': 'Trust', 'Dealer_self': 'Dealer'}
                    df_inst['name'] = df_inst['name'].map(name_map).fillna(df_inst['name'])
                    df_inst_agg = df_inst.groupby(['date', 'name'])['net_buy'].sum().reset_index()
                    df_inst_pivot = df_inst_agg.pivot(index='date', columns='name', values='net_buy').reset_index().fillna(0)
                    df = pd.merge(df, df_inst_pivot, on='date', how='left').fillna(0)
                else:
                    for col in ['Foreign', 'Trust', 'Dealer']: df[col] = 0
            except:
                for col in ['Foreign', 'Trust', 'Dealer']: df[col] = 0

            self.db.save_dataframe(df, f"stock_{stock_id}_daily")
            return df
        except Exception as e:
            print(f"Error in fetch_stock_daily: {e}")
            return pd.DataFrame()

    def get_txo_contract(self, delivery_month: str, strike_price: float, option_right: str):
        """獲取選擇權合約"""
        try:
            txo_contracts = self.api.Contracts.Options.TXO
            for c in txo_contracts:
                if str(c.delivery_month).strip() == delivery_month and c.strike_price == strike_price:
                    val = str(c.option_right).lower()
                    target_right = option_right.lower()
                    if (target_right in ('call', 'c') and ('call' in val or val == 'c')) or \
                       (target_right in ('put', 'p') and ('put' in val or val == 'p')):
                        return c
        except Exception as e:
            print(f"Error getting TXO contract: {e}")
        return None

    def fetch_options_intraday(self, delivery_month: str, strike_price: float, option_right: str, start_date: str, end_date: str = None):
        """抓取選擇權1分鐘K線"""
        if not end_date: end_date = datetime.today().strftime('%Y-%m-%d')
        contract = self.get_txo_contract(delivery_month, strike_price, option_right)
        if not contract: return pd.DataFrame()
            
        try:
            kbars = self.api.kbars(contract=contract, start=start_date, end=end_date)
            df = pd.DataFrame({**kbars})
            if df.empty: return pd.DataFrame()
            df.ts = pd.to_datetime(df.ts)
            self.db.save_dataframe(df, f"option_TXO_{delivery_month}_{int(strike_price)}_{option_right}_intraday")
            return df
        except: return pd.DataFrame()

    def fetch_us_stock_daily(self, symbol: str, start_date: str, end_date: str = None):
        if not end_date: end_date = datetime.today().strftime('%Y-%m-%d')
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            if df.empty: return pd.DataFrame()
            df.reset_index(inplace=True)
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
            df.rename(columns={'Date': 'date'}, inplace=True)
            df = df[['date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            self.db.save_dataframe(df, f"us_stock_{symbol}_daily")
            return df
        except Exception as e:
            print(f"Error in fetch_us_stock_daily: {e}")
            return pd.DataFrame()

    def fetch_revenue_breakdown(self, stock_id: str):
        print(f"Fetching comprehensive revenue breakdown for {stock_id}...")
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                url = f"https://www.cnyes.com/twstock/{stock_id}/company/profile"
                page.goto(url, wait_until="domcontentloaded", timeout=30000)
                
                try:
                    page.locator(".business-composition.fixed-tab").first.wait_for(timeout=5000)
                    page.locator(".business-composition.fixed-tab").first.scroll_into_view_if_needed()
                    page.mouse.wheel(0, 500)
                    page.wait_for_selector('.table-content .cell .name', timeout=5000)
                except: pass
                
                scraped_data = page.evaluate('''() => {
                    const sections = document.querySelectorAll('.business-composition.fixed-tab');
                    let all_data = [];
                    sections.forEach(section => {
                        const yearEl = section.querySelector('.date-container .year');
                        const year = yearEl ? yearEl.innerText.replace('年', '') : "N/A";
                        const title = section.querySelector('.top-container div')?.innerText || "未知";
                        const type = title.includes('地區') ? "銷售地區" : "產品結構";
                        
                        const rows = section.querySelectorAll('.table-content .cell');
                        rows.forEach(row => {
                            const name = row.querySelector('.name')?.innerText;
                            let percentVal = 0.0;
                            const percentEl = row.querySelector('.percent');
                            if (percentEl) {
                                percentVal = parseFloat(percentEl.innerText.replace('%', ''));
                            }
                            if (name && !isNaN(percentVal) && !name.includes('合計') && !name.includes('淨額')) {
                                all_data.push({ year: year, type: type, name: name, percentage: percentVal });
                            }
                        });
                    });
                    return all_data;
                }''')
                browser.close()
                if scraped_data: return pd.DataFrame(scraped_data)
        except Exception as e:
            print(f"Playwright scraping failed: {e}")

        # 靜態資料備援
        static_data = {
            "2330": [{"year": "2023", "type": "產品結構", "name": "積體電路", "percentage": 88.0}],
            "2317": [{"year": "2023", "type": "產品結構", "name": "通訊終端", "percentage": 30.0}]
        }
        if stock_id in static_data: return pd.DataFrame(static_data[stock_id])
        return pd.DataFrame()

    def fetch_stock_chips(self, stock_id: str, start_date: str):
        end_date = datetime.today().strftime('%Y-%m-%d')
        res = {"margin": pd.DataFrame(), "holdings": pd.DataFrame()}
        try:
            df_m = self.dl.taiwan_stock_margin_purchase_short_sale(stock_id=stock_id, start_date=start_date, end_date=end_date)
            if df_m is not None and not df_m.empty: res["margin"] = df_m
        except: pass
        
        try:
            df_h = self.dl.taiwan_stock_holding_shares_per(stock_id=stock_id, start_date=start_date, end_date=end_date)
            if df_h is not None and isinstance(df_h, pd.DataFrame) and not df_h.empty: res["holdings"] = df_h
        except: pass
        return res

    def fetch_industry_prices(self, stock_ids: list, start_date: str, end_date: str):
        combined = {}
        for sid in stock_ids:
            try:
                try:
                    contract = self.api.Contracts.Stocks[sid]
                except KeyError:
                    continue
                kbars = self.api.kbars(contract=contract, start=start_date, end=end_date)
                df = pd.DataFrame({**kbars})
                if not df.empty:
                    df.ts = pd.to_datetime(df.ts)
                    df['date'] = df.ts.dt.strftime('%Y-%m-%d')
                    df_daily = df.groupby('date').agg({'Close': 'last'}).reset_index()
                    combined[sid] = df_daily.set_index('date')['Close'].astype(float)
            except: continue
        return pd.DataFrame(combined)

    def fetch_stock_news(self, stock_id: str, start_date: str):
        try:
            df = self.dl.taiwan_stock_news(stock_id=stock_id, start_date=start_date, end_date=datetime.today().strftime('%Y-%m-%d'))
            return df if df is not None else pd.DataFrame()
        except: return pd.DataFrame()

    def fetch_stock_info(self):
        # 台灣股市產業代碼對照表 (整合上市與上櫃分類)
        INDUSTRY_MAP = {
            "00": "指數/大盤", "01": "水泥工業", "02": "食品工業", "03": "塑膠工業", "04": "紡織纖維",
            "05": "電機機械", "06": "電器電纜", "07": "化學工業", "08": "玻璃陶瓷",
            "09": "造紙工業", "10": "鋼鐵工業", "11": "橡膠工業", "12": "汽車工業",
            "13": "電子工業", "14": "建材營造", "15": "航運業", "16": "觀光事業",
            "17": "金融保險", "18": "貿易百貨", "19": "綜合", "20": "其他",
            "21": "化學生技", "22": "生技醫療", "23": "油電燃氣", "24": "半導體業",
            "25": "電腦及週邊設備業", "26": "光電業", "27": "通信網路業",
            "28": "電子零組件業", "29": "電子通路業", "30": "資訊服務業",
            "31": "其他電子業", "32": "文化創意業", "33": "農業科技業", "34": "電子商務",
            "35": "綠能環保", "36": "數位雲端", "37": "運動休閒", "38": "居家生活",
            "80": "管理股票", "91": "臺灣存託憑證", "97": "認購售權證", "99": "其他"
        }
        try:
            data = []
            # 1. 抓取股票 (包含大盤指數與各類股)
            contracts = self.api.Contracts.Stocks
            for category in contracts:
                for contract in category:
                    cat_val = str(contract.category).strip()
                    # 處理代碼補零與對應
                    cat_code = cat_val.zfill(2) if cat_val.isdigit() else cat_val
                    industry_name = INDUSTRY_MAP.get(cat_code, contract.category if contract.category else "其他")
                    data.append({
                        'stock_id': contract.code,
                        'stock_name': contract.name,
                        'industry_category': f"{cat_code} {industry_name}"
                    })
            
            df = pd.DataFrame(data)
            self.db.save_dataframe(df, "taiwan_stock_info")
            return df
        except Exception as e:
            print(f"Failed to fetch stock info: {e}")
            return pd.DataFrame()

    def fetch_us_stock_info(self, symbol: str):
        try:
            t = yf.Ticker(symbol); info = t.info
            return {"名稱": info.get("longName"), "產業": info.get("industry"), "市值": info.get("marketCap"), "公司簡介": info.get("longBusinessSummary")}
        except: return None

    def fetch_us_stock_news(self, symbol: str):
        try:
            t = yf.Ticker(symbol); news = t.news
            return [{'title': n.get('title'), 'date': datetime.fromtimestamp(n.get('providerPublishTime')).strftime('%Y-%m-%d')} for n in news]
        except: return []

    def get_industry_leaders(self, industry_code: str):
        """獲取各產業的大宗代表股 (權值股/龍頭股)"""
        LEADERS = {
            "24": ["2330", "2454", "2303", "3711", "2379", "3034"], 
            "25": ["2382", "2301", "2357", "3231", "2356", "2324"], 
            "17": ["2881", "2882", "2886", "2891", "5880", "2884"], 
            "15": ["2603", "2609", "2615", "2610", "2618"],         
            "28": ["2317", "2308", "3008", "2474", "2354"],         
            "01": ["1101", "1102"],                                 
            "02": ["1216", "1210"],                                 
            "03": ["1301", "1303", "1326", "6505"],                 
            "14": ["2542", "2520"],                                 
            "26": ["2409", "3481", "2448"],                         
            "27": ["2412", "4904", "3045"],                         
        }
        code = industry_code.split(" ")[0]
        return LEADERS.get(code, [])

    def fetch_stock_financials(self, stock_id: str, start_date: str):
        try:
            df = self.dl.taiwan_stock_financial_statement(stock_id=stock_id, start_date=start_date)
            if df is not None and not df.empty:
                items = ['Revenue', 'GrossProfit', 'OperatingIncome', 'NetIncome', 'EPS']
                df_filtered = df[df['type'].isin(items)]
                df_filtered = df_filtered.sort_values('date', ascending=False)
                self.db.save_dataframe(df_filtered, f"stock_{stock_id}_financials")
                return {"financials": df_filtered}
        except: pass
        return {"financials": pd.DataFrame()}

    def fetch_monthly_revenue(self, stock_id: str, start_date: str):
        try:
            df = self.dl.taiwan_stock_month_revenue(stock_id=stock_id, start_date=start_date)
            return df if df is not None else pd.DataFrame()
        except: return pd.DataFrame()

    def fetch_us_stock_financials(self, symbol: str):
        try:
            t = yf.Ticker(symbol)
            return {"income": t.quarterly_financials.T, "balance": t.quarterly_balance_sheet.T}
        except: return {"income": pd.DataFrame(), "balance": pd.DataFrame()}

    def fetch_realtime_tick(self, stock_id: str, is_us: bool = False):
        try:
            if is_us:
                stock_us = getattr(self.api.Contracts, 'StockUS', {})
                contract = stock_us.get(stock_id) if isinstance(stock_us, dict) else (stock_us[stock_id] if hasattr(stock_us, '__getitem__') else getattr(stock_us, 'get', lambda x: None)(stock_id))
                if not contract: return pd.DataFrame()
            else:
                try: contract = self.api.Contracts.Stocks[stock_id]
                except: return pd.DataFrame()
            
            snapshots = self.api.snapshots([contract])
            if snapshots:
                snapshot = snapshots[0]
                return pd.DataFrame({'Close': [snapshot.close], 'date': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]})
            return pd.DataFrame()
        except: return pd.DataFrame()
