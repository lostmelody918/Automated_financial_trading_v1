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
        if ai_score is None or len(ai_score) != 7: return 0

        # 若已經是機率分佈 (總和約為 1.0)，則跳過 Softmax，避免過度扁平化 (Squashing)
        if np.isclose(np.sum(ai_score), 1.0):
            probs = np.array(ai_score)
        else:
            # 將原始分數 (Logits) 轉換為機率 (Softmax) 以避免高於 1 的信心值
            exp_scores = np.exp(ai_score - np.max(ai_score))
            probs = exp_scores / exp_scores.sum()

        # 計算 AI 信心水準 (機率最高的一項)
        ai_confidence = np.max(probs)

        # AI 基礎訊號轉換 (-3 到 3)
        pred_class = np.argmax(probs)
        mapping = {0: -3, 1: -2, 2: -1, 3: 0, 4: 1, 5: 2, 6: 3}
        base_signal = mapping.get(pred_class, 0)

        # 若 AI 信心極低 (< 35%)，強制降級基礎訊號，交由量化策略接管
        if ai_confidence < 0.35 and base_signal != 0:
            base_signal = np.sign(base_signal) * 1

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
        dynamic_bias_limit = max(0.0025, (atr / close_price) * 1.2) 
        extreme_bias_limit = max(0.0040, (atr / close_price) * 2.0)

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
                    if vol_surge < 1.8 or (abs(base_signal) < 2 and abs(slope_ma20) < 3.5):
                        print(f"🛡️ [月結算保護] 動能不足以對抗 Theta (Vol={vol_surge:.2f})，沒收訊號。")
                        base_signal = 0
                elif settlement_type == 1 or dte < 0.5: # 週結算 / 0DTE
                    if abs(vwap_bias) > dynamic_bias_limit:
                        print(f"🛡️ [週結算/0DTE保護] 動態乖離過大防追高殺低 (Bias={vwap_bias:.5f}, 極限={dynamic_bias_limit:.5f})，沒收訊號。")
                        base_signal = 0

            # 2. 動態動能耗竭防護 (Dynamic Momentum Exhaustion)
            if base_signal > 0:
                if vwap_bias > extreme_bias_limit or (vwap_bias > dynamic_bias_limit and rsi_fast > 85):
                    print(f"🛡️ [過熱網] 極端多頭乖離，降級訊號以防反轉。")
                    base_signal = max(1, base_signal - 1)
            elif base_signal < 0:
                if vwap_bias < -extreme_bias_limit or (vwap_bias < -dynamic_bias_limit and rsi_fast < 15):
                    print(f"🛡️ [過冷網] 極端空頭乖離，降級訊號以防反彈。")
                    base_signal = min(-1, base_signal + 1)


        # ==================================================
        # ⚔️ 第二層：十大經典量化兵器庫 (Omni-Strategy Arsenal)
        # 執行條件：當 AI 給出盤整(0)，或需強制覆寫訊號時觸發
        # ==================================================
        if base_signal == 0:

            # --------------------------------------------------
            # 🧨 策略 1: 終極 V 轉狙擊 (Level 10) - 逆勢吃 Gamma 爆發
            # --------------------------------------------------
            if (rsi_fast < 25 and vwap_bias < -dynamic_bias_limit and (momentum_explosion == 1 or is_pin_bar_bottom)):
                print(f"🔥 [V轉做多] 恐慌超賣底 + 籌碼換手 (PinBar={is_pin_bar_bottom})")
                return 10

            if (rsi_fast > 75 and vwap_bias > dynamic_bias_limit and (momentum_explosion == 1 or is_pin_bar_top)):
                print(f"🔥 [V轉做空] 瘋狂超買頂 + 避雷針出現 (PinBar={is_pin_bar_top})")
                return -10

            # --------------------------------------------------
            # 🧲 策略 1.5 (新增): 流動性吸收/微結構耗竭 (Microstructure Absorption) (Level 4)
            # 爆大量但價格卻收長下影線/上影線，代表大單吃貨或出貨 (Limit Order Absorption)
            # --------------------------------------------------
            if vol_surge > 2.5 and is_pin_bar_bottom and close_price > open_price and rsi_fast < 40:
                print(f"🧲 [流動性吸收] 爆量下殺被買盤全數吸收，微結構反轉 Call！")
                return 4
            
            if vol_surge > 2.5 and is_pin_bar_top and close_price < open_price and rsi_fast > 60:
                print(f"🧲 [流動性吸收] 爆量上漲被賣盤全數吸收，微結構反轉 Put！")
                return -4

            # --------------------------------------------------
            # 🩸 策略 2: 莊家投降極值反轉 P/C Capitulation (Level 3)
            # --------------------------------------------------
            if pc_ratio < 0.75 and rsi_fast < 20 and vol_surge > 2.0:
                print(f"🩸 [莊家投降] P/C 極度悲觀 ({pc_ratio:.2f}) 且爆量殺跌，極值反彈 Call！")
                return 3

            if pc_ratio > 1.25 and rsi_fast > 80 and vol_surge > 2.0:
                print(f"🩸 [莊家投降] P/C 極度樂觀 ({pc_ratio:.2f}) 且爆量急拉，極值反轉 Put！")
                return -3

            # --------------------------------------------------
            # 🕵️ 策略 3: 法人籌碼背離軋空 Institutional Divergence (Level 3)
            # --------------------------------------------------
            if slope_ma20 < -1.5 and vwap_bias < -0.0010 and rsi_fast < 40:
                if foreign_z > 1.2 and dealer_mom > 0:
                    print(f"🕵️ [法人背離] 散戶殺跌但法人吃貨 (Foreign Z={foreign_z:.2f})，軋空 Call！")
                    return 3

            if slope_ma20 > 1.5 and vwap_bias > 0.0010 and rsi_fast > 60:
                if foreign_z < -1.2 and dealer_mom < 0:
                    print(f"🕵️ [法人背離] 散戶追高但法人倒貨 (Foreign Z={foreign_z:.2f})，殺多 Put！")
                    return -3

            # --------------------------------------------------
            # 🌪️ 策略 4: 高基期 ATR 擴張突破 Gamma Breakout (Level 3)
            # --------------------------------------------------
            if atr_expansion > 0.15 and is_squeeze == 0:
                if slope_ma20 > 2.0 and close_price > open_price:
                    print(f"🌪️ [ATR擴張] 波動率膨脹 (擴張率 {atr_expansion:.1%})，追擊 Gamma 爆發 Call！")
                    return 3
                elif slope_ma20 < -2.0 and close_price < open_price:
                    print(f"🌪️ [ATR擴張] 波動率膨脹 (擴張率 {atr_expansion:.1%})，追擊 Gamma 爆發 Put！")
                    return -3

            # --------------------------------------------------
            # 🚀 策略 5: 布林擠壓突破 Squeeze Breakout (Level 3)
            # --------------------------------------------------
            if prev_squeeze == 1 and vol_surge > 1.5:
                if close_price > bb_upper and rsi_fast > 60:
                    print(f"🚀 [擠壓突破] 帶量突破布林上軌，波動率擴張！(Vol_Surge={vol_surge:.1f})")
                    return 3
                elif close_price < bb_lower and rsi_fast < 40:
                    print(f"🚀 [擠壓跌破] 帶量跌破布林下軌，波動率擴張！(Vol_Surge={vol_surge:.1f})")
                    return -3

            # --------------------------------------------------
            # 🌊 策略 6: VWAP 防守反擊 VWAP Bounce (Level 2)
            # --------------------------------------------------
            if slope_ma20 > 1.5 and abs(vwap_bias) < 0.0008:
                if rsi_fast < 50 and close_price > open_price:
                    print(f"🌊 [VWAP反擊] 多頭回測均價線有撐，順勢買入 Call！")
                    return 2

            if slope_ma20 < -1.5 and abs(vwap_bias) < 0.0008:
                if rsi_fast > 50 and close_price < open_price:
                    print(f"🌊 [VWAP反擊] 空頭反彈均價線遇壓，順勢買入 Put！")
                    return -2

            # --------------------------------------------------
            # ⚡ 策略 7: MACD 動能穿越 MACD Cross (Level 2)
            # --------------------------------------------------
            if macd_hist > 0 and prev_macd_hist <= 0 and slope_ma20 > 1.0 and rsi_fast < 65:
                print(f"⚡ [MACD穿越] 動能柱由負翻正，多方啟動！")
                return 2

            if macd_hist < 0 and prev_macd_hist >= 0 and slope_ma20 < -1.0 and rsi_fast > 35:
                print(f"⚡ [MACD穿越] 動能柱由正翻負，空方啟動！")
                return -2

            # --------------------------------------------------
            # 🐢 策略 8: 無量緩漲/跌軋空 Squeeze Grind (Level 1)
            # --------------------------------------------------
            if 1.0 < slope_ma20 < 2.5 and 0.0005 < vwap_bias < 0.0020 and 45 < rsi_fast < 75:
                if last_row.get('Low', 0) >= prev_row.get('Low', 0):
                    print(f"🐢 [緩漲軋空] 無量緩步墊高，順勢輕倉 Call: slope={slope_ma20:.2f}")
                    return 1

            if -2.5 < slope_ma20 < -1.0 and -0.0020 < vwap_bias < -0.0005 and 25 < rsi_fast < 55:
                if last_row.get('High', 0) <= prev_row.get('High', 0):
                    print(f"🐢 [緩跌殺多] 無量緩步破底，順勢輕倉 Put: slope={slope_ma20:.2f}")
                    return -1

            # --------------------------------------------------
            # 🎯 策略 9: 淺回檔順勢承接 Micro Pullback (Level 1)
            # --------------------------------------------------
            if slope_ma20 > 2.5 and -0.0005 < vwap_bias < dynamic_bias_limit / 2 and rsi_fast < 45:
                print(f"🎯 [微觀回檔] 強多勢極短線拉回，承接 Call: RSI={rsi_fast:.1f}")
                return 1

            if slope_ma20 < -2.5 and -dynamic_bias_limit / 2 < vwap_bias < 0.0005 and rsi_fast > 55:
                print(f"🎯 [微觀回檔] 強空勢極短線反彈，承接 Put: RSI={rsi_fast:.1f}")
                return -1

        return base_signal
