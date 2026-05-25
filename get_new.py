import requests
import pandas as pd
import jieba
import re
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 1. 設置環境與自定義詞庫
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

pos_words = {"利多", "大漲", "噴發", "創高", "優於預期", "買超", "噴出", "成長", "展望佳", "營收翻倍", "法說會"}
neg_words = {"利空", "大跌", "重摔", "修正", "不如預期", "賣超", "疲軟", "衰退", "砍單", "保守", "跳空"}
for w in pos_words | neg_words:
    jieba.add_word(w)

class AdvancedSentimentAnalyzer:
    def __init__(self, lambda_decay=0.03):
        self.lambda_decay = lambda_decay
        self.headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    def get_basic_score(self, text):
        words = jieba.lcut(text)
        pos = sum(1 for w in words if w in pos_words)
        neg = sum(1 for w in words if w in neg_words)
        return (pos - neg) / (pos + neg) if (pos + neg) > 0 else 0

    def fetch_and_analyze(self, stock_id="2330"):
        url = f"https://api.cnyes.com/media/api/v1/search?q={stock_id}&limit=50"
        try:
            res = requests.get(url, headers=self.headers)
            res_json = res.json()
            
            # 安全地獲取 items 與 data，防止回傳的是空陣列 [] 等意外格式
            items_container = res_json.get('items', {})
            if isinstance(items_container, list):
                items = items_container
            elif isinstance(items_container, dict):
                items = items_container.get('data', [])
            else:
                items = []

            if not items:
                print(f"No news items found for {stock_id}")
                return pd.DataFrame()

            raw_data = []
            now = datetime.now()
            for item in items:
                title = item.get('title', '')
                summary = re.sub(r'<[^>]+>', '', item.get('summary', ''))
                
                # 處理發布時間，若缺失則跳過
                pub_ts = item.get('publishAt')
                if not pub_ts:
                    continue
                pub_date = pd.to_datetime(pub_ts, unit='s')

                # 計算分數與時間權重
                score = self.get_basic_score(title + summary)
                delta_hours = (now - pub_date).total_seconds() / 3600
                weight = np.exp(-self.lambda_decay * delta_hours)

                raw_data.append({
                    "時間": pub_date.strftime('%Y-%m-%d %H:%M'),
                    "date_obj": pub_date,
                    "標題": title,
                    "情緒分數": round(score, 3),
                    "加權分數": round(score * weight, 3),
                    "情緒評價": "利多" if score > 0 else ("利空" if score < 0 else "中性")
                })

            if not raw_data:
                return pd.DataFrame()

            df = pd.DataFrame(raw_data).sort_values('date_obj')
            # 計算情緒不連續性
            df['ma_sentiment'] = df['情緒分數'].rolling(window=5, min_periods=1).mean()
            df['不連續性'] = df['情緒分數'] - df['ma_sentiment']
            return df
        except Exception as e:
            print(f"抓取失敗: {e}")
            return pd.DataFrame()

    def plot_trends(self, df, stock_id):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        # 子圖 1
        ax1.plot(df['date_obj'], df['情緒分數'], label='原始情緒', color='#bdc3c7', alpha=0.5, marker='o')
        ax1.plot(df['date_obj'], df['加權分數'], label='加權情緒 (考慮衰減)', color='#1a73e8', linewidth=2)
        ax1.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax1.set_title(f'{stock_id} 情緒趨勢分析', fontsize=16)
        ax1.legend()
        # 子圖 2
        ax2.bar(df['date_obj'], df['不連續性'], label='情緒跳空 (Discontinuity)',
               color=np.where(df['不連續性'] >= 0, '#27ae60', '#e74c3c'), alpha=0.7)
        ax2.axhline(0, color='black', alpha=0.3)
        ax2.set_title('情緒跳空檢測 (Jumps)', fontsize=14)
        ax2.legend()
        plt.tight_layout()
        plt.savefig(f"ai_research/{stock_id}_advanced_sentiment.png")
        plt.show()

# --- 執行 ---
if __name__ == "__main__":
    analyzer = AdvancedSentimentAnalyzer()
    target_stock = "2330"
    df = analyzer.fetch_and_analyze(target_stock)

    if not df.empty:
        # 1. 基礎分析結果輸出 (依你的需求保留並更新)
        avg_score = df['情緒分數'].mean()
        weighted_avg = df['加權分數'].mean()

        print("\n" + "="*50)
        print(f"📊 分析結果 - {target_stock}")
        print(f"📈 樣本新聞數：{len(df)} 則")
        print(f"🌡️ 平均情緒溫度：{round(avg_score, 4)}")
        print(f"⏳ 加權情緒溫度：{round(weighted_avg, 4)} (已計算時間衰減)")
        print(f"📢 市場傾向：{'🔥 極度樂觀' if avg_score > 0.2 else ('❄️ 偏向保守' if avg_score < -0.2 else '⚖️ 盤整中性')}")
        print("="*50)

        # 2. 顯示近期新聞評分列表 (前 10 則)
        print("\n📝 近期新聞評分列表：")
        # 依時間降序顯示最近的新聞
        display_df = df.sort_values('date_obj', ascending=False).head(10)
        print(display_df[['時間', '標題', '情緒分數', '情緒評價']])

        # 3. 偵測跳空點警告
        jumps = df[df['不連續性'].abs() > 0.4].tail(2)
        if not jumps.empty:
            print("\n⚠️ 偵測到情緒劇烈跳動！")
            for _, row in jumps.iterrows():
                print(f"[{row['時間']}] 跳空度: {row['不連續性']:.2f} -> {row['標題'][:25]}...")

        # 4. 繪製並儲存趨勢圖
        analyzer.plot_trends(df, target_stock)
    else:
        print("未抓取到資料。")