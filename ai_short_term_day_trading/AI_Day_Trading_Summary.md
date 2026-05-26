# AI Short-Term Day Trading (選擇權當沖 AI 交易模組)

本模組 (`ai_short_term_day_trading`) 是一個針對「台指選擇權」的短線當沖 (Day Trading) 所設計的完整 AI 交易與回測框架。系統內建資料擷取、模型訓練、多重策略整合、回測模擬以及盤中即時模擬交易等功能。

## 核心系統架構

1. **資料引擎 (`data_engine.py`)**
   - 負責串接 Shioaji API，擷取盤中即時 K 線以及歷史 K 線資料。
   - 計算技術指標 (如 ATR, MA, RSI, MACD 等) 並產出給 AI 的特徵值。

2. **AI 模型 (`composite_ai.py`)**
   - 實作了 `CompositeDayTradingAI`，採用深度學習架構 (如 CNN / Transformer)，以連續的時間視窗 (Window Size) 特徵為輸入，預測未來的漲跌平機率。

3. **策略工廠 (`strategy_factory.py`)**
   - 負責將 AI 輸出的機率 (勝率) 結合傳統技術指標 (如均線趨勢、突破等)，產出最終的做多 (Buy Call) 或做空 (Buy Put) 訊號。

4. **模型管理與訓練 (`train_model.py` / `model_manager.py`)**
   - 提供 AI 模型的訓練流程、權重儲存、正規化參數 (norm_params) 的存檔，以及在實盤/回測時自動載入最新版本的模型。

5. **回測系統 (`advanced_simulator(Backtest).py`)**
   - 用於對歷史資料進行深度的回測，評估模型與策略在過去的績效。

6. **即時模擬交易 (`live_option_simulator.py`)**
   - 獨立的盤中實時監控器 (Live Trader)。
   - 在交易時段 (08:45 ~ 13:45) 每隔數秒輪詢即時盤況，將最新資料餵入 AI 產生預測，並根據嚴格的風控邏輯執行虛擬進出場。

7. **分析工具 (`analyze_features.py`, `analyze_losing_trades.py`, `analyze_txt_logs.py`)**
   - 輔助工具集，用於解析交易日誌、檢討虧損交易原因、以及特徵重要性分析，協助後續優化。

## 近期優化項目：`live_option_simulator.py`

為了讓盤中模擬更貼近真實市場並適應高風險高報酬的選擇權當沖特性，已針對 `live_option_simulator.py` 進行以下參數優化：

- **資金與滑價真實化**：
  - `INITIAL_CAPITAL` 提升至 120,000，因應高波動性與高價權利金。
  - `FEE_SLIPPAGE_PER_CONTRACT` 調整為 100 (約 1.5~2 個 tick 的滑價與手續費)，更符合真實市場快市時的滑價耗損。
- **風險容忍與停利停損放寬** (高風險高報酬設定)：
  - `TAKE_PROFIT_PCT`: 調整為 `2.50` (250%) 硬停利，保留獲利暴衝空間。
  - `STOP_LOSS_PCT`: 調整為 `-0.60` (60%) 硬停損，避免在極端洗盤時過早被掃出場。
  - `TRAILING_START_PCT`: 上調至 `0.40` (40%) 時才啟動追蹤停損。
  - `TRAILING_ATR_MULTIPLIER`: 調整為 `3.0` 倍 ATR，給予波動更大之緩衝空間。
- **輪詢頻率**：
  - `POLL_INTERVAL_SECONDS` 縮短為 `30` 秒，提高 AI 在盤中的反應速度。

這套框架在經過這些優化後，具備了更彈性的空間去捕捉選擇權倍數的獲利機會，並能承受實戰中難以避免的滑價與較大震幅洗盤。