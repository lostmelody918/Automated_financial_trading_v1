import pandas as pd
import pandas_ta as ta
import numpy as np

class KDAnalyzer:
    """提供 KD 指標 (Stochastic Oscillator) 計算、型態辨識與原理說明"""
    
    @staticmethod
    def calculate_kd(df: pd.DataFrame, k_period=9, d_period=3):
        """計算 KD 指標"""
        df = df.copy()
        # pandas-ta 的 stoch 會回傳 k, d, j
        kd = ta.stoch(df["High"], df["Low"], df["Close"], k=k_period, d=d_period)
        if kd is not None:
            df = pd.concat([df, kd], axis=1)
            # 重新命名以便識別 (預設為 STOCHk_9_3_3 等)
            df.rename(columns={kd.columns[0]: 'K', kd.columns[1]: 'D'}, inplace=True)
        return df

    @staticmethod
    def analyze_patterns(df: pd.DataFrame):
        """偵測 KD 型態與背離"""
        if 'K' not in df.columns or 'D' not in df.columns:
            return [], "尚未計算 KD 資料"

        latest_k = df['K'].iloc[-1]
        latest_d = df['D'].iloc[-1]
        prev_k = df['K'].iloc[-2]
        prev_d = df['D'].iloc[-2]
        
        patterns = []
        
        # 1. 超買超賣偵測
        if latest_k > 80: patterns.append("⚠️ KD 高檔超買 (Overbought): 市場情緒過熱，留意拉回風險。")
        elif latest_k < 20: patterns.append("✅ KD 低檔超賣 (Oversold): 市場情緒過冷，可能醞釀反彈。")
        
        # 2. 交叉偵測
        if prev_k < prev_d and latest_k > latest_d:
            loc = "低檔" if latest_k < 30 else ("高檔" if latest_k > 70 else "中軸")
            patterns.append(f"🚀 KD 黃金交叉 ({loc}): K 線由下往上穿越 D 線，為看多訊號。")
        elif prev_k > prev_d and latest_k < latest_d:
            loc = "高檔" if latest_k > 70 else ("低檔" if latest_k < 30 else "中軸")
            patterns.append(f"📉 KD 死亡交叉 ({loc}): K 線由上往下跌破 D 線，為看空訊號。")
            
        # 3. 簡單背離偵測 (價格創高但 KD 未創高)
        # 這裡僅實作基礎邏輯
        recent_price_max = df['Close'].tail(20).max()
        recent_kd_max = df['K'].tail(20).max()
        if df['Close'].iloc[-1] >= recent_price_max * 0.98 and latest_k < recent_kd_max * 0.9:
            patterns.append("🚨 疑似高檔背離: 價格接近高點但 KD 弱勢，警惕轉折。")

        return patterns

    @staticmethod
    def get_principles():
        """提供 KD 指標的技術原理說明"""
        return """
### 🕯️ KD 指標 (隨機指標) 技術原理
**1. 核心公式:**
- **未成熟隨機值 (RSV)** = [(今日收盤價 - 最近n日最低價) / (最近n日最高價 - 最近n日最低價)] × 100
- **K 值 (快線)** = (2/3 × 前一日K值) + (1/3 × 當日RSV)
- **D 值 (慢線)** = (2/3 × 前一日D值) + (1/3 × 當日K值)

**2. 研判準則:**
- **數值區間**: 0 ~ 100 之間波動。
- **80 以上**: 定義為「超買區」，代表多方強勢，但也暗示漲幅過大，隨時可能獲利回吐。
- **20 以下**: 定義為「超賣區」，代表空方強勢，但也暗示跌幅已深，可能跌深反彈。
- **黃金交叉**: 當 K 值由下往上突破 D 值，通常被視為買進訊號 (低檔 20 附近交叉最具參考價值)。
- **死亡交叉**: 當 K 值由上往下跌破 D 值，通常被視為賣出訊號 (高檔 80 附近交叉最具參考價值)。

**3. 背離現象:**
- 當股價創新高，但 KD 指標卻未創新高時，稱為「頂背離」，是強烈的趨勢反轉預警訊號。
        """

class KLineAnalyzer:
    """舊有的 K 線型態分析 (保留基礎 MA 指標)"""
    @staticmethod
    def add_indicators(df: pd.DataFrame):
        df = df.copy()
        df["MA5"] = ta.sma(df["Close"], length=5)
        df["MA20"] = ta.sma(df["Close"], length=20)
        df["MA60"] = ta.sma(df["Close"], length=60)
        df["RSI"] = ta.rsi(df["Close"], length=14)
        
        # 加入 ATR 與 波動率 (ATR / Close)
        if "High" in df.columns and "Low" in df.columns:
            atr = ta.atr(df["High"], df["Low"], df["Close"], length=14)
            if atr is not None:
                df["ATR"] = atr
                df["Volatility"] = (df["ATR"] / df["Close"]) * 100
            else:
                df["ATR"] = 0.0
                df["Volatility"] = 0.0
        
        # 加入 KD
        df = KDAnalyzer.calculate_kd(df)
        
        # 加入 1/3/5/15/30 日價格動能 (Momentum)
        # 動能計算公式: (當前價格 - n日前價格) / n日前價格 * 100
        def safe_momentum(period):
            p = min(period, max(1, len(df) - 1))
            if p > 0:
                return df["Close"].pct_change(periods=p) * 100
            return pd.Series([0.0] * len(df), index=df.index)

        df["Momentum_1"] = safe_momentum(1)
        df["Momentum_3"] = safe_momentum(3)
        df["Momentum_5"] = safe_momentum(5)
        df["Momentum_15"] = safe_momentum(15)
        df["Momentum_30"] = safe_momentum(30)
        
        return df

class MomentumAnalyzer:
    """提供價格動能指標分析"""
    
    @staticmethod
    def analyze_momentum(df: pd.DataFrame):
        """分析 1/3/5/15/30 日動能狀態"""
        if "Momentum_1" not in df.columns:
            df = KLineAnalyzer.add_indicators(df)
            
        latest = df.iloc[-1]
        def safe_get(col):
            val = latest[col]
            return val if pd.notna(val) else 0.0

        m1, m3, m5 = safe_get("Momentum_1"), safe_get("Momentum_3"), safe_get("Momentum_5")
        m15, m30 = safe_get("Momentum_15"), safe_get("Momentum_30")
        
        status = []
        # 短期動能
        if m1 > 0 and m3 > 0 and m5 > 0:
            status.append("🔥 短期強勁上升動能: 1/3/5 日動能皆為正值。")
        elif m1 < 0 and m3 < 0 and m5 < 0:
            status.append("❄️ 短期強勁下跌動能: 1/3/5 日動能皆為負值。")
            
        # 中期動能
        if m15 > 0 and m30 > 0:
            status.append("📈 中期趨勢向上: 15/30 日動能保持正成長。")
        elif m15 < 0 and m30 < 0:
            status.append("📉 中期趨勢向下: 15/30 日動能轉負。")
        
        if m1 > 5: status.append("⚡ 極短線暴漲: 單日漲幅超過 5%。")
        if m1 < -5: status.append("🆘 極短線重挫: 單日跌幅超過 5%。")
        
        return {
            "m1": m1, "m3": m3, "m5": m5, "m15": m15, "m30": m30,
            "status": status
        }

class SentimentAnalyzer:
    """進階新聞情緒分析 - 支援中英文權重評分"""
    
    # 權重設定: 2為強烈訊號, 1為一般訊號
    POS_MAP = {
        # 中文強烈看多
        '創新高': 2, '噴發': 2, '大漲': 2, '強勢': 2, '優於預期': 2, '轉盈': 2, '買超': 1,
        '成長': 1, '看好': 1, '營收增加': 1, '突破': 1, '看多': 1, '獲利': 1, '目標價上調': 2,
        # 英文強烈看多
        'bullish': 2, 'soar': 2, 'surge': 2, 'beat': 2, 'outperform': 2, 'buy': 1,
        'growth': 1, 'positive': 1, 'breakout': 1, 'upgrade': 2, 'gain': 1
    }
    
    NEG_MAP = {
        # 中文強烈看空
        '虧損擴大': 2, '重挫': 2, '大跌': 2, '低於預期': 2, '看淡': 1, '賣超': 1,
        '衰退': 1, '跌破': 1, '調降': 2, '看空': 1, '壓力': 1, '獲利回吐': 1,
        # 英文強烈看空
        'bearish': 2, 'plummet': 2, 'crash': 2, 'miss': 2, 'underperform': 2, 'sell': 1,
        'decline': 1, 'negative': 1, 'breakdown': 1, 'downgrade': 2, 'loss': 1
    }

    @staticmethod
    def analyze_sentiment(news_list: list, is_us: bool = False):
        """分析新聞情緒並回傳評分後的結果"""
        if not news_list:
            return 0, []
        
        processed_news = []
        total_score = 0
        
        for news in news_list:
            title = news.get('title', '').lower()
            score = 0
            
            # 匹配正向詞
            for word, weight in SentimentAnalyzer.POS_MAP.items():
                if word.lower() in title:
                    score += weight
            
            # 匹配負向詞
            for word, weight in SentimentAnalyzer.NEG_MAP.items():
                if word.lower() in title:
                    score -= weight
            
            total_score += score
            
            # 判斷標籤
            label = "🟢 看多" if score > 0 else ("🔴 看空" if score < 0 else "⚪ 中性")
            
            processed_news.append({
                'date': news.get('date', news.get('publisher_time', 'N/A')),
                'title': news.get('title', '無標題'),
                'score': score,
                'label': label,
                'link': news.get('link', '')
            })
            
        avg_score = total_score / len(news_list)
        return avg_score, processed_news
