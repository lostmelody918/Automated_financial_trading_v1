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
        # 加入 KD
        df = KDAnalyzer.calculate_kd(df)
        return df

class SentimentAnalyzer:
    """提供新聞情緒分析功能"""
    
    @staticmethod
    def analyze_sentiment(news_df: pd.DataFrame):
        """簡單的新聞標題情緒分析"""
        if news_df.empty:
            return 0, pd.DataFrame()
        
        pos_keywords = ['獲利', '成長', '創新高', '看好', '買超', '營收增加', '突破', '轉盈', '優於預期', '大漲', '噴發', '看多']
        neg_keywords = ['虧損', '衰退', '看淡', '賣超', '營收減少', '跌破', '虧損擴大', '調降', '低於預期', '重挫', '大跌', '看空']
        
        scores = []
        for title in news_df['title']:
            score = 0
            for k in pos_keywords:
                if k in title: score += 1
            for k in neg_keywords:
                if k in title: score -= 1
            scores.append(score)
            
        news_df = news_df.copy()
        news_df['sentiment_score'] = scores
        avg_score = sum(scores) / len(scores) if scores else 0
        
        return avg_score, news_df
