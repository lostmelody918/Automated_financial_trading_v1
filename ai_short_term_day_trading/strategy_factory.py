import numpy as np

class StrategyFactory:
    """模組化策略工廠：AI 與機構級量化策略融合決策矩陣 (整合增強版)"""
    @staticmethod
    def get_strategy(strategy_name="composite"):
        return CompositeOptionsStrategy()

class CompositeOptionsStrategy:
    def __init__(self):
        self.name = "Omni-Strategy AI Factory Plus"

    def generate_signal(self, df_slice, ai_score=None, last_win=False):
        signal = self._generate_signal_internal(df_slice, ai_score, last_win)
        
        if signal == 0 or df_slice.empty:
            return signal

        # 取得最後一根 K 線的特徵
        last_row = df_slice.iloc[-1]
        obv_bias_3m = last_row.get('3m_obv_bias', 0.0)
        slope_ma20_15m = last_row.get('15m_slope_ma20', 0.0)
        volume_delta_3m = last_row.get('3m_volume_delta', 0.0)

        # ==================================================
        # 🛡️ 寬鬆防護網：v22 機器學習特徵過濾 (Aggressive ML Filters)
        # 應使用者要求「對波段積極，不為一點虧損放棄更高利益」
        # 僅剔除【絕對不可能賺錢】的極端惡劣進場點
        # ==================================================
        if signal > 0:
            # 放寬：只有當籌碼完全沒殺(OBV>0.2)、且趨勢強烈向下(MA20<-1.5)、且爆量下跌(Volume>20)才擋
            if obv_bias_3m > 0.2 and slope_ma20_15m < -1.5 and volume_delta_3m > 20.0:
                print(f"🛑 [ML特徵防護] 極端劣勢，拒絕做多！OBV({obv_bias_3m:.2f}), 趨勢({slope_ma20_15m:.2f}), 量能({volume_delta_3m:.1f})")
                return 0
        elif signal < 0:
            # 放寬：只有當籌碼完全沒漲(OBV<-0.2)、且趨勢強烈向上(MA20>1.5)、且爆量上漲(Volume>20)才擋
            if obv_bias_3m < -0.2 and slope_ma20_15m > 1.5 and volume_delta_3m > 20.0:
                print(f"🛑 [ML特徵防護] 極端劣勢，拒絕做空！OBV({obv_bias_3m:.2f}), 趨勢({slope_ma20_15m:.2f}), 量能({volume_delta_3m:.1f})")
                return 0

        return signal

    def _generate_signal_internal(self, df_slice, ai_score=None, last_win=False):
        if ai_score is None or len(ai_score) != 5: return 0

        # 若已經是機率分佈 (總和約為 1.0)，則跳過 Softmax，避免過度扁平化 (Squashing)
        if np.isclose(np.sum(ai_score), 1.0):
            probs = np.array(ai_score)
        else:
            # 將原始分數 (Logits) 轉換為機率 (Softmax) 以避免高於 1 的信心值
            exp_scores = np.exp(ai_score - np.max(ai_score))
            probs = exp_scores / exp_scores.sum()

        # 計算 AI 信心水準 (機率最高的一項)
        ai_confidence = np.max(probs)

        # AI 基礎訊號轉換 (5分類: 0=強空, 1=中空, 2=盤整, 3=中多, 4=強多)
        pred_class = np.argmax(probs)
        # mapping 給予強度 (這會影響後面過濾的信心度)
        mapping = {0: -2, 1: -1, 2: 0, 3: 1, 4: 2}
        base_signal = mapping.get(pred_class, 0)

        # [Sniper Mode] 極限狙擊手模式：若 AI 信心偏低 (< 0.65)，直接放棄交易，不讓傳統量化策略接管
        if ai_confidence < 0.65:
            base_signal = 0

        if df_slice.empty: return base_signal

        # 取得當前與前一根 K 線狀態
        last_row = df_slice.iloc[-1]
        prev_row = df_slice.iloc[-2] if len(df_slice) > 1 else last_row

        # ==================================================
        # 📊 提取全環境特徵 (Feature Extraction)
        # ==================================================
        close_price = last_row.get('Close', 1.0) # 避免除以零
        open_price = last_row.get('Open', close_price)
        slope_ma20 = last_row.get('slope_ma20', 0.0)
        vwap_bias = last_row.get('vwap_bias', 0.0)
        rsi_fast = last_row.get('rsi_fast', 50.0)
        rsi_fast_prev = prev_row.get('rsi_fast', 50.0)
        vol_surge = last_row.get('vol_surge_ratio', 1.0)
        momentum_explosion = last_row.get('momentum_explosion', 0)

        # 新增美股與日股特徵
        us_tw_gap_divergence = last_row.get('us_tw_gap_divergence', 0.0)
        nikkei_premarket_momentum = last_row.get('nikkei_premarket_momentum', 0.0)

        minutes_of_day = 0
        if 'time' in last_row:
            current_time = last_row['time']
            minutes_of_day = current_time.hour * 60 + current_time.minute

        date_str = ""
        if 'date_only' in last_row:
            date_str = last_row['date_only'].strftime('%Y-%m-%d') if hasattr(last_row['date_only'], 'strftime') else str(last_row['date_only'])

        # 簡化版 FOMC 日期
        fomc_dates = ['2026-01-28', '2026-03-18', '2026-04-29', '2026-06-17', '2026-07-29', '2026-09-16', '2026-11-04', '2026-12-16']
        is_fomc_day = date_str in fomc_dates

        # 結算與時間特徵
        settlement_type = last_row.get('settlement_type', 0)
        dte = last_row.get('dte', 1.0)
        is_settlement_day = (last_row.get('is_settlement_day', 0) == 1) or (dte < 0.5)

        # 籌碼與巨觀特徵
        foreign_z = last_row.get('foreign_net_oi_zscore', 0)
        dealer_mom = last_row.get('dealer_relative_momentum', 0)
        pc_ratio = last_row.get('pc_ratio', 1.0)

        # 波動率與布林特徵
        atr = last_row.get('atr', 20.0)
        prev_atr = prev_row.get('atr', 20.0)
        atr_expansion = (atr - prev_atr) / (prev_atr + 1e-9)
        is_squeeze = last_row.get('is_squeeze', 0)
        prev_squeeze = prev_row.get('is_squeeze', 0)
        bb_upper = last_row.get('bb_upper', float('inf'))
        bb_lower = last_row.get('bb_lower', 0)

        # 動能特徵
        macd_hist = last_row.get('macd_hist', 0)
        prev_macd_hist = prev_row.get('macd_hist', 0)

        # 微觀 K 線結構
        lower_shadow = last_row.get('lower_shadow', 0)
        upper_shadow = last_row.get('upper_shadow', 0)
        body_length = max(last_row.get('body_length', 1.0), 0.1)
        is_pin_bar_bottom = (lower_shadow > body_length * 2.0)
        is_pin_bar_top = (upper_shadow > body_length * 2.0)

        # 動態閥值計算 (Dynamic Thresholds based on ATR) - 來自學術論文降低風險的啟發
        # 在高波動率環境下，容忍更大的乖離；在低波動環境，乖離閾值縮小。
        # [優化] 適度緊縮乖離容忍度，增加防護力
        dynamic_bias_limit = max(0.0025, (atr / close_price) * 1.0)
        extreme_bias_limit = max(0.0040, (atr / close_price) * 1.7)

        # 從 v21 回測分析中提取的機器學習「賠錢基因」過濾器
        obv_bias_3m = last_row.get('3m_obv_bias', 0.0)
        slope_ma20_15m = last_row.get('15m_slope_ma20', 0.0)
        volume_delta_3m = last_row.get('3m_volume_delta', 0.0)
        rsi_fast_15m = last_row.get('15m_rsi_fast', 50.0)
        macd_15m = last_row.get('15m_macd', 0.0)
        dist_from_ma20 = last_row.get('dist_from_ma20', 0.0)
        volume_delta = last_row.get('volume_delta', 0.0)
        pv_divergence = last_row.get('pv_divergence', 0)
        obv_bias = last_row.get('obv_bias', 0.0)
        obv_slope_1m = last_row.get('1m_obv_slope', 0.0)
        macd_hist_30m = last_row.get('30m_macd_hist', 0.0)
        macd_hist_15m = last_row.get('15m_macd_hist', 0.0)
        vol_surge_1m = last_row.get('1m_vol_surge_ratio', vol_surge)
        slope_vwap_30m = last_row.get('30m_slope_vwap', 0.0)
        nasdaq_prev_ret = last_row.get('nasdaq_prev_ret', 0.0)
        lower_shadow_1m = last_row.get('1m_lower_shadow', 0.0)
        cvd_bias = last_row.get('cvd_bias', 0.0)

        # ==================================================
        # ⚡ 第零層：特種部隊 - 波動獵手 (Volatility Sniper) [絕對最高優先級]
        # ==================================================
        if vol_surge > 1.5 or atr_expansion > 0.15:
            # 1. V轉抄底/摸頭 (V-Reversal)
            if rsi_fast_prev < 15 and rsi_fast > rsi_fast_prev and macd_hist > prev_macd_hist and macd_hist < -1.5:
                print(f"⚡ [波動獵手] V轉抄底！RSI (<15) + MACD收斂，強制吃 Gamma Call！")
                return 10
            elif rsi_fast_prev > 85 and rsi_fast < rsi_fast_prev and macd_hist < prev_macd_hist and macd_hist > 1.5:
                print(f"⚡ [波動獵手] V轉摸頭！RSI (>85) + MACD收斂，強制吃 Gamma Put！")
                return -10
            
            # 2. 嘎空/殺多突破 (Breakout)
            if prev_squeeze == 1 or is_squeeze == 1:
                if close_price > bb_upper:
                    print(f"⚡ [波動獵手] 嘎空突破！狹幅震盪後突破上軌，強制追擊 Call！")
                    return 10
                elif close_price < bb_lower:
                    print(f"⚡ [波動獵手] 殺多跌破！狹幅震盪後跌破下軌，強制追擊 Put！")
                    return -10

        # ==================================================
        # 🛡️ 第一層：絕對防禦網 (Safety Nets)
        # ==================================================
        if base_signal != 0:
            # 0. FOMC 日防護 (波動率風險控制)
            if is_fomc_day:
                if vol_surge < 2.0 or abs(slope_ma20) < 2.0:
                    print(f"🛡️ [FOMC保護] 利率決策日震盪劇烈，動能不足 (Vol={vol_surge:.2f}) 沒收訊號。")
                    base_signal = 0

            # 1. 結算日防護 (Gamma/Theta 雙刃劍)
            if is_settlement_day:
                if settlement_type == 2: # 期貨月大結算
                    if vol_surge < 1.7 or (abs(base_signal) < 2 and abs(slope_ma20) < 1.5):
                        print(f"🛡️ [月結算保護] 動能不足以對抗 Theta (Vol={vol_surge:.2f})，沒收訊號。")
                        base_signal = 0
                elif settlement_type == 1 or dte < 0.5: # 週結算 / 0DTE
                    # [優化] 當有明確強烈趨勢 (abs(slope_ma20) >= 2.5) 時，不因乖離過大而沒收訊號
                    if abs(vwap_bias) > dynamic_bias_limit * 1.5 and abs(slope_ma20) < 2.5:
                        print(f"🛡️ [週結算/0DTE保護] 動態乖離過大防追高殺低 (Bias={vwap_bias:.5f}, 極限={dynamic_bias_limit*1.5:.5f})，沒收訊號。")
                        base_signal = 0

            # 2. 動態動能耗竭防護 (Dynamic Momentum Exhaustion)
            if base_signal > 0:
                if vwap_bias > extreme_bias_limit * 1.3 or (vwap_bias > dynamic_bias_limit * 1.15 and rsi_fast > 80):
                    print(f"🛡️ [過熱網] 極端多頭乖離，降級訊號以防反轉。")
                    base_signal = max(1, base_signal - 1)
            elif base_signal < 0:
                if vwap_bias < -extreme_bias_limit * 1.3 or (vwap_bias < -dynamic_bias_limit * 1.15 and rsi_fast < 20):
                    print(f"🛡️ [過冷網] 極端空頭乖離，降級訊號以防反彈。")
                    base_signal = min(-1, base_signal + 1)
                    
            # 3. 大盤環境趨勢配合 (Intraday Trend Alignment)
            intraday_trend = last_row.get('intraday_trend', 0.0)
            if base_signal > 0 and intraday_trend < -0.005:
                if rsi_fast < 25 and vwap_bias < -dynamic_bias_limit:
                    print(f"🛡️ [大盤防護] 強空趨勢中，但極度超賣，允許AI搶反彈。")
                else:
                    print(f"🛡️ [大盤防護] 強空趨勢盤中，沒收多頭訊號以防逆勢。")
                    base_signal = 0
            elif base_signal > 0 and intraday_trend < 0.0 and atr > 30:
                # 震盪偏弱且高波動，提高多單進場門檻
                if vol_surge < 2.0 or base_signal < 2:
                    print(f"🛡️ [大盤防護] 震盪偏弱且高波動 (ATR={atr:.1f})，提高做多門檻，動能不足沒收訊號。")
                    base_signal = 0
            elif base_signal < 0 and intraday_trend > 0.005:
                if rsi_fast > 75 and vwap_bias > dynamic_bias_limit:
                    print(f"🛡️ [大盤防護] 強多趨勢中，但極度超買，允許AI搶回檔。")
                else:
                    print(f"🛡️ [大盤防護] 強多趨勢盤中，沒收空頭訊號以防逆勢。")
                    base_signal = 0

            # 4. 高波動死水防護 (High Volatility Dead Water Filter)
            if base_signal != 0 and atr > 40.0:
                vol_surge_ratio_1m = last_row.get('1m_vol_surge_ratio', vol_surge)
                if vol_surge_ratio_1m < 1.0:
                    print(f"🛡️ [高波動死水防護] ATR({atr:.1f})極高，但微觀量能枯竭({vol_surge_ratio_1m:.2f})，沒收訊號防盤整耗損。")
                    base_signal = 0

            # 5. [聰明過濾] 極低波動死水盤防護 (Low ATR Chop Filter)
            if base_signal != 0 and atr < 12.0:
                # 波動太小，選擇權買方會被 Theta 吃乾抹淨，且扣掉滑價必賠
                print(f"🛡️ [死水防護] 真實波動率太低(ATR={atr:.1f})，選擇權買方無利可圖，沒收訊號。")
                base_signal = 0

            # 6. [聰明過濾] 天價權利金防護 (Extreme IV Premium Filter)
            if base_signal != 0 and vol_surge > 3.5 and abs(slope_ma20) < 2.0:
                # 爆大量但均線還沒跟上 (代表可能是亂流或洗盤)，這時進場的 IV 膨脹太嚴重
                print(f"🛡️ [昂貴權利金防護] 爆量(Vol={vol_surge:.1f})但趨勢不明確，此時進場滑價與 IV 極高，沒收訊號。")
                base_signal = 0

            # ==================================================
            # 🛡️ 深度大數據防護網 (Deep Data Insights Guard)
            # ==================================================
            
            # 7. [大數據提煉] 巨觀逆勢防護 (Macro MACD Guard)
            if base_signal > 0 and macd_hist_30m > 5.0:
                print(f"🛡️ [巨觀防護] 30m_MACD({macd_hist_30m:.1f})過高，高檔回踩做多風險極大，沒收訊號。")
                base_signal = 0
            elif base_signal < 0 and macd_hist_30m < -5.0:
                print(f"🛡️ [巨觀防護] 30m_MACD({macd_hist_30m:.1f})過低，低檔追空容易被嘎，沒收訊號。")
                base_signal = 0

            # 8. [大數據提煉] 微觀主力動能防護 (Micro OBV Slope Guard)
            if base_signal > 0 and obv_slope_1m < -0.1:
                print(f"🛡️ [微觀防護] 做多但 1m_OBV_Slope({obv_slope_1m:.3f}) 嚴重向下，主力仍在倒貨，拒絕接刀！")
                base_signal = 0
            elif base_signal < 0 and obv_slope_1m > 0.1:
                print(f"🛡️ [微觀防護] 做空但 1m_OBV_Slope({obv_slope_1m:.3f}) 嚴重向上，主力仍在吃貨，拒絕摸頭！")
                base_signal = 0

            # 9. [大數據提煉] 爆量高潮冷卻防護 (Volume Climax Cooldown)
            if base_signal != 0 and vol_surge_1m > 3.0:
                print(f"🛡️ [滑價防護] 當下1分鐘爆出天量({vol_surge_1m:.1f}x)，此時滑價最嚴重，強制等待下一根K線冷卻！")
                base_signal = 0

            # 10. [大數據提煉] 大勢 VWAP 共振防護 (30m VWAP Guard)
            # VWAP 斜率過大代表單邊趨勢，絕對不准逆勢
            if base_signal > 0 and slope_vwap_30m < -1.0:
                print(f"🛡️ [大勢防護] 30m VWAP 向下({slope_vwap_30m:.1f})，絕對禁止逆勢做多！")
                base_signal = 0
            elif base_signal < 0 and slope_vwap_30m > 1.0:
                print(f"🛡️ [大勢防護] 30m VWAP 向上({slope_vwap_30m:.1f})，絕對禁止逆勢做空！")
                base_signal = 0

            # 11. [多時區共振防護] (30m MACD vs 15m MACD)
            if base_signal > 0 and macd_hist_30m > -1.0 and macd_hist_15m < 0:
                print(f"🛡️ [多時區防護] 30m未見深底，且15m動能仍向下，拒絕做多接刀！")
                base_signal = 0

            # 12. [開盤騙線防護] (美股大漲台股拉高倒貨)
            if base_signal > 0 and nasdaq_prev_ret > 0.005 and current_time and current_time.hour < 10:
                print(f"🛡️ [騙線防護] 昨晚美股大漲(>0.5%)，早盤極易開高走低，沒收多頭訊號！")
                base_signal = 0

            # 13. [第二落點防護] (量能冷卻與下影線確認)
            if base_signal > 0 and vol_surge_1m > 1.5 and lower_shadow_1m < 5.0:
                print(f"🛡️ [第二落點防護] 正在爆量且未見1分鐘長下影線(={lower_shadow_1m:.1f})，洗盤尚未結束，暫不接刀！")
                base_signal = 0


        # ==================================================
        # 🤖 第一點五層：機器學習六大波段模態 (ML Six Archetypes)
        # 執行條件：當 AI 給出盤整(0) 時，由大數據提煉的 6 種必殺波段前兆優先掃描
        # ==================================================
        if base_signal == 0:
            
            # --- 📈 做多波段的三大模態 (Long Archetypes) ---
            
            # 1. 極度超賣 V 轉 (Deep Over-sold Reversal)
            # 邏輯: 15m MACD極度負值, 微觀1m/3m出現收下影線或背離
            if macd_15m < -15.0 and rsi_fast < 20 and (is_pin_bar_bottom or momentum_explosion == 1):
                print(f"🔥 [ML模態1: 極度超賣V轉] 15m_MACD({macd_15m:.1f}) + 微觀耗竭，做多！")
                return 5
                
            # 2. 無聲吸籌 / 牛旗 (Stealth Accumulation)
            # 邏輯: 15m趨勢向上, 量能萎縮, 但OBV異常偏高
            if slope_ma20_15m > 1.0 and volume_delta_3m < -10.0 and obv_bias_3m > 0.3:
                if abs(vwap_bias) < dynamic_bias_limit: # 在均線附近
                    print(f"🐂 [ML模態2: 牛旗無聲吸籌] 趨勢向上({slope_ma20_15m:.1f}) + 量縮 + OBV背離高({obv_bias_3m:.2f})，做多！")
                    return 3
                    
            # 3. 爆量吸收 / 嘎空突破 (High-Volume Absorption)
            # 邏輯: 爆量殺跌但收長下影線 (量大但跌不下去)
            if vol_surge > 2.5 and is_pin_bar_bottom and close_price > open_price and rsi_fast < 45:
                print(f"🧲 [ML模態3: 爆量吸收] 爆量下殺被買盤全數吸收 (PinBar)，嘎空做多！")
                return 4

            # --- 📉 做空波段的三大模態 (Short Archetypes) ---
            
            # 4. 拋物線高潮摸頭 (Parabolic Climax Top)
            # 邏輯: 波動率極大, MACD極度超買, 微觀收上影線
            if (atr_expansion > 0.2 or atr > 25.0) and macd_15m > 15.0 and rsi_fast > 80 and is_pin_bar_top:
                print(f"💥 [ML模態4: 拋物線高潮] 波動率爆發({atr:.1f}) + 超買 + 避雷針，做空！")
                return -5
                
            # 5. 逃命波誘多 / 熊旗 (Failed Bounce)
            # 邏輯: 15m趨勢向下, 反彈到阻力, 量縮, OBV偏弱
            if slope_ma20_15m < -1.0 and vwap_bias > 0 and volume_delta_3m < -10.0 and obv_bias_3m < -0.3:
                print(f"🐻 [ML模態5: 熊旗逃命波] 趨勢向下({slope_ma20_15m:.1f}) + 阻力量縮 + OBV弱({obv_bias_3m:.2f})，做空！")
                return -3
                
            # 6. 鈍刀子割肉 / 陰跌 (The Slow Bleed)
            # 邏輯: 趨勢持續向下, 量能死水, OBV穩步向下
            if slope_ma20_15m < -1.0 and slope_ma20 < -1.0 and vol_surge < 1.0 and obv_bias_3m < -0.2:
                print(f"🩸 [ML模態6: 陰跌緩殺] 雙趨勢向下 + 死水量能 + OBV弱({obv_bias_3m:.2f})，順勢做空！")
                return -2

        # ==================================================
        # ⚔️ 第二層：十大經典量化兵器庫 (Omni-Strategy Arsenal)
        # 執行條件：當 AI 給出盤整(0)，或需強制覆寫訊號時觸發
        # ==================================================
        if base_signal == 0:

            # ==================================================
            # [優先級 1] 終極反轉層 (Signal Level ±10)
            # ==================================================
            
            # 🌟 策略 0.1: 順勢爆量吸籌做多策略 (Volume Absorption Buy-the-Dip) - 2年黃金破譯版
            # 邏輯: 尋找急跌時的「假摔」 + 恐懼吸收(volume_delta < 0, cvd_bias < 0) + 賣壓背離(pv_divergence >= 0)
            if dist_from_ma20 < 0 and rsi_fast < 35 and vol_surge > 1.5 and volume_delta < 0 and cvd_bias < 0 and pv_divergence >= 0:
                print(f"🌟 [恐懼吸收做多] 跌破均線 + 恐慌賣壓({volume_delta:.1f})被吸收 + CVD底({cvd_bias:.2f}) + 量價背離({pv_divergence})，絕佳起漲點！")
                return 10

            # 🧨 策略 1: 終極 V 轉狙擊 (Level 10) - 逆勢吃 Gamma 爆發
            if (rsi_fast < 15 and vwap_bias < -dynamic_bias_limit * 1.5 and (momentum_explosion == 1 or is_pin_bar_bottom)):
                print(f"🔥 [V轉做多] 恐慌超賣底 (RSI<15) + 籌碼換手 (PinBar={is_pin_bar_bottom})")
                return 10

            if (rsi_fast > 85 and vwap_bias > dynamic_bias_limit * 1.5 and (momentum_explosion == 1 or is_pin_bar_top)):
                print(f"🔥 [V轉做空] 瘋狂超買頂 (RSI>85) + 避雷針出現 (PinBar={is_pin_bar_top})")
                return -10

            # ==================================================
            # [優先級 2] 動能耗竭層 (Signal Level ±5)
            # ==================================================
            
            # 📉 策略 0.2: 動能衰竭反轉做空策略 (Momentum Exhaustion Fade) - 2年黃金破譯版
            # 邏輯: 尋找創高時的「量縮背離」 + 散戶追高(volume_delta > 0) + 量價極度背離(pv_divergence < 0)
            if rsi_fast > 65 and dist_from_ma20 > 0.0015 and vol_surge < 0.95 and volume_delta > 0 and pv_divergence < 0:
                print(f"📉 [動能衰竭做空] RSI極高 + 散戶追高({volume_delta:.1f})但量能背離({pv_divergence})，動能耗竭試空！")
                return -5

            # 🕵️ 策略 0.3: 籌碼背離隱蔽吃貨策略 (OBV & PV Divergence Stealth Accumulation)
            # 邏輯: 融合 OBV Z-Score 與微觀量價背離，專抓大戶偷偷吃貨與出貨
            if pv_divergence == 1 and obv_bias > 1.5 and dist_from_ma20 < 0:
                print(f"🕵️ [隱蔽吃貨做多] 價跌量縮(背離={pv_divergence}) + OBV強勢背離({obv_bias:.2f})，大戶逢低暗中吸籌！")
                return 5

            if pv_divergence == -1 and obv_bias < -1.5 and dist_from_ma20 > 0:
                print(f"🕵️ [隱蔽出貨做空] 價漲量縮(背離={pv_divergence}) + OBV弱勢背離({obv_bias:.2f})，大戶拉高暗中倒貨！")
                return -5

            # ==================================================
            # [優先級 3] 突破與籌碼極值層 (Signal Level ±3 ~ ±4)
            # ==================================================
            
            # 🩸 策略 2: 莊家投降極值反轉 P/C Capitulation
            if pc_ratio < 0.75 and rsi_fast < 20 and vol_surge > 2.0:
                print(f"🩸 [莊家投降] P/C 極度悲觀 ({pc_ratio:.2f}) 且爆量殺跌，極值反彈 Call！")
                return 3

            if pc_ratio > 1.25 and rsi_fast > 80 and vol_surge > 2.0:
                print(f"🩸 [莊家投降] P/C 極度樂觀 ({pc_ratio:.2f}) 且爆量急拉，極值反轉 Put！")
                return -3

            # 🕵️ 策略 3: 法人籌碼背離軋空 Institutional Divergence
            if slope_ma20 < -1.5 and vwap_bias < -0.0010 and rsi_fast < 40:
                if foreign_z > 1.2 and dealer_mom > 0:
                    print(f"🕵️ [法人背離] 散戶殺跌但法人吃貨 (Foreign Z={foreign_z:.2f})，軋空 Call！")
                    return 3

            if slope_ma20 > 1.5 and vwap_bias > 0.0010 and rsi_fast > 60:
                if foreign_z < -1.2 and dealer_mom < 0:
                    print(f"🕵️ [法人背離] 散戶追高但法人倒貨 (Foreign Z={foreign_z:.2f})，殺多 Put！")
                    return -3

            # 🌪️ 策略 4: 高基期 ATR 擴張突破 Gamma Breakout
            if atr_expansion > 0.15 and is_squeeze == 0:
                if slope_ma20 > 2.0 and close_price > open_price:
                    print(f"🌪️ [ATR擴張] 波動率膨脹 (擴張率 {atr_expansion:.1%})，追擊 Gamma 爆發 Call！")
                    return 3
                elif slope_ma20 < -2.0 and close_price < open_price:
                    print(f"🌪️ [ATR擴張] 波動率膨脹 (擴張率 {atr_expansion:.1%})，追擊 Gamma 爆發 Put！")
                    return -3

            # 🚀 策略 5: 布林擠壓突破 Squeeze Breakout
            if prev_squeeze == 1 and vol_surge > 1.5:
                if close_price > bb_upper and rsi_fast > 60:
                    print(f"🚀 [擠壓突破] 帶量突破布林上軌，波動率擴張！(Vol_Surge={vol_surge:.1f})")
                    return 3
                elif close_price < bb_lower and rsi_fast < 40:
                    print(f"🚀 [擠壓跌破] 帶量跌破布林下軌，波動率擴張！(Vol_Surge={vol_surge:.1f})")
                    return -3

            # ==================================================
            # [優先級 4] 動能輔助層 (Signal Level ±2)
            # ==================================================
            
            # ⚡ 策略 7: MACD 動能穿越 MACD Cross
            if macd_hist > 0 and prev_macd_hist <= 0 and slope_ma20 > 1.0 and rsi_fast < 65:
                print(f"⚡ [MACD穿越] 動能柱由負翻正，多方啟動！")
                return 2

            if macd_hist < 0 and prev_macd_hist >= 0 and slope_ma20 < -1.0 and rsi_fast > 35:
                print(f"⚡ [MACD穿越] 動能柱由正翻負，空方啟動！")
                return -2

            # --------------------------------------------------
            # 🐢 策略 8 & 9 (已於 Sniper Mode 移除): 弱勢趨勢跟隨容易被雙巴，予以剔除
            # --------------------------------------------------

        return base_signal
