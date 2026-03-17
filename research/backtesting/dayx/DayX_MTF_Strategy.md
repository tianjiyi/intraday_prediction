# DayX Multi-Timeframe Strategy

## Overview
DayX is an intraday mean-reversion + trend-dip strategy for QQQ, optimized via Optuna walk-forward optimization. It combines 3-minute entry signals with a 15-minute regime filter to achieve a Sharpe of 2.55 (full period 2016-2026) and 3.05 (recent 2020-2026) with max drawdown under 6%.

**Instrument**: QQQ (Nasdaq 100 ETF)
**Entry Timeframe**: 3 minutes
**Regime Filter Timeframe**: 15 minutes
**Trading Hours**: RTH only (09:30-16:00 ET)
**Entry Window**: 09:57 - 14:37 ET

---

## Strategy Architecture

```
15Min Bars ──► CCI(20) + VWAP ──► Regime State (Bullish / Bearish / Neutral)
                                         │
                                         ▼ filter
3Min Bars ──► Indicators ──► Signal Detection ──► Entry ──► Position Management ──► Exit
              (CCI, BB,       (buy_dip,                      (stop, T1 partial,
               VWAP, ATR,      sell_rip,                      T2 full, trailing,
               patterns)       trend_dip)                     EOD flatten)
```

---

## 15-Minute Regime Filter (Medium Level)

Before any 3Min entry signal is taken, check the most recent 15Min bar:

| Regime | Condition (15Min) | Allowed 3Min Signals |
|--------|-------------------|---------------------|
| **Bullish** | CCI(20) > 0 AND Close > VWAP | buy_dip, trend_dip |
| **Bearish** | CCI(20) < 0 AND Close < VWAP | sell_rip |
| **Neutral** | Neither condition met | All signals pass |

The 15Min CCI uses period 20 (the `cci_fast` param from 5Min config). VWAP is session-reset (cumulative typical_price * volume / cumulative volume).

---

## Entry Signals (3-Minute)

### 1. buy_dip (Mean Reversion Long)
Catches exhausted selling near Bollinger Band support with a reversal candle.

**Conditions (ALL must be true):**
1. **BB Zone**: Close within 1.15% of BB Lower band
   - `(close - bb_lower) / close * 100 <= 1.15`
2. **Oversold**: CCI(18) < -100
3. **Exhaustion**: 8+ consecutive lower closes
4. **Reversal Pattern**: Hammer OR Bullish Engulfing candle
5. **Below VWAP**: Close < VWAP (confirming dip, not trend)
6. **Time Window**: Between 09:57 and 14:37 ET
7. **Regime Filter**: 15Min must be Bullish (CCI > 0 AND close > VWAP)

### 2. sell_rip (Mean Reversion Short)
Catches exhausted buying near Bollinger Band resistance with a reversal candle.

**Conditions (ALL must be true):**
1. **BB Zone**: Close within 1.15% of BB Upper band
   - `(bb_upper - close) / close * 100 <= 1.15`
2. **Overbought**: CCI(18) > 100
3. **Exhaustion**: 8+ consecutive higher closes
4. **Reversal Pattern**: Inverted Hammer OR Bearish Engulfing candle
5. **Above VWAP**: Close > VWAP (confirming rip, not trend)
6. **Time Window**: Between 09:57 and 14:37 ET
7. **Regime Filter**: 15Min must be Bearish (CCI < 0 AND close < VWAP)

### 3. trend_dip (Trend Continuation Long)
Buy pullbacks to VWAP in established uptrends.

**Conditions (ALL must be true):**
1. **VWAP Rising**: VWAP slope > 0 (VWAP now > VWAP 6 bars ago)
2. **Trend Day**: >= 42.5% of session bars have closed above VWAP
3. **Near VWAP**: Close within 0.29% of VWAP
   - `abs((close - vwap) / vwap * 100) <= 0.29`
4. **CCI Neutral**: CCI(18) between -96 and +98 (healthy pullback, not panic)
5. **Trigger Candle**: Hammer OR Bullish Engulfing OR Green candle (close > open)
6. **Time Window**: Between 09:57 and 14:37 ET
7. **Regime Filter**: 15Min must be Bullish (CCI > 0 AND close > VWAP)

---

## Indicator Specifications

### CCI (Commodity Channel Index)
- **Period**: 18 (3Min), 20 (15Min regime)
- **Formula**: `CCI = (TP - SMA(TP, n)) / (0.015 * MeanDeviation(TP, n))`
- **TP** = (High + Low + Close) / 3

### Bollinger Bands
- **Period**: 25, **Std Dev**: 1.717
- **Mid** = SMA(Close, 25)
- **Upper** = Mid + 1.717 * StdDev(Close, 25)
- **Lower** = Mid - 1.717 * StdDev(Close, 25)

### VWAP (Volume Weighted Average Price)
- **Session-reset**: Resets at 09:30 ET each day
- **Formula**: Cumulative(TP * Volume) / Cumulative(Volume)
- **Slope**: VWAP - VWAP[6] (change over 6 bars = 18 minutes on 3Min)
- **% Above**: Running % of session bars where Close > VWAP

### ATR (Average True Range)
- **Period**: 18
- **Method**: Wilder's EMA (alpha = 1/period)
- **TR** = max(High-Low, |High-PrevClose|, |Low-PrevClose|)

### Candlestick Patterns
- **Hammer**: Lower shadow >= 2x body, upper shadow <= 0.5x body, body > 0
- **Inverted Hammer**: Upper shadow >= 2x body, lower shadow <= 0.5x body, body > 0
- **Bullish Engulfing**: Current green candle body fully engulfs prior red candle body
- **Bearish Engulfing**: Current red candle body fully engulfs prior green candle body

### Exhaustion
- Count consecutive closes in same direction
- **Up**: N bars where close > close[1] (N >= 8)
- **Down**: N bars where close < close[1] (N >= 8)

---

## Position Management

### Entry Sizing
- Full equity allocation: `shares = floor(equity / price)`
- One position at a time (max_positions = 1)
- Slippage: $0.01/share

### Stop Loss
- **Distance**: ATR * 2.913 from entry price
- Long: `stop = entry - ATR * 2.913`
- Short: `stop = entry + ATR * 2.913`

### Target 1 (Partial Exit)
- **R-Multiple**: 0.556
- Long: `T1 = entry + ATR * 2.913 * 0.556`
- Exit 31.3% of position at T1
- Move stop to breakeven after T1 hit

### Target 2 (Full Exit)
- **R-Multiple**: 2.447
- Long: `T2 = entry + ATR * 2.913 * 2.447`
- Exit remaining position at T2

### Progressive Trailing Stop
After T1 hit, trail stop upward in steps:
- **Step size**: 0.839R
- Ratchet: For every 0.839R price moves in favor, trail stop up by 0.839R
- Formula: `trail_R = (floor(best_R / 0.839) - 1) * 0.839`
- New stop (long): `entry + trail_R * risk`
- Only ratchets up, never down

### EOD Flatten
- Close all positions at 15:55 ET at market price
- No new entries after 15:30 ET

### Exit Priority Order (each bar)
1. EOD Flatten (15:55)
2. Stop Loss
3. Target 2 (full exit)
4. Target 1 (partial exit + trail activation)

---

## Optimized Parameters (3Min - Optuna Trial #709)

| Parameter | Value | Description |
|-----------|-------|-------------|
| cci_fast | 18 | CCI period |
| bb_period | 25 | Bollinger Band period |
| bb_std | 1.717 | Bollinger Band std multiplier |
| atr_period | 18 | ATR period |
| bb_zone_pct | 1.15 | % distance from BB for entry zone |
| exhaustion_lookback | 8 | Consecutive closes for exhaustion |
| cci_neutral_lo | -96 | CCI neutral zone low (trend_dip) |
| cci_neutral_hi | 98 | CCI neutral zone high (trend_dip) |
| stop_atr_mult | 2.913 | Stop distance in ATR multiples |
| target1_r | 0.556 | T1 as fraction of risk |
| target2_r | 2.447 | T2 as fraction of risk |
| partial_exit_pct | 0.313 | % of position to exit at T1 |
| trail_progressive_step | 0.839 | Trailing stop step in R |
| trend_dip_vwap_pct | 0.292 | Max % distance from VWAP for trend_dip |
| trend_dip_above_pct | 0.425 | Min % of session bars above VWAP |
| entry_earliest | 09:57 | Earliest entry time |
| entry_latest | 14:37 | Latest entry time |

---

## Performance (2016-2026, with 15Min Medium Regime Filter)

| Metric | Full Period | Recent (2020+) |
|--------|------------|----------------|
| Sharpe | 2.55 | 3.05 |
| Max Drawdown | -5.8% | -3.7% |
| Trades | 2,469 | 1,404 |
| Win Rate | 70.1% | 71.4% |
| Avg Win | $396 | — |
| Avg Loss | -$590 | — |
| Total P&L | $248,724 | $170,530 |
