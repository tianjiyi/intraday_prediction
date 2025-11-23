# Trading Rules & Constitution (AI-Enforced)

> **IMPORTANT**: These rules are designed to enforce discipline and protect capital. **The LLM will check your trades against these rules.**

---

## 1. Daily Mindset & Context Analysis (The "First 30 Minutes" Ritual)

**You MUST define the battlefield before fighting.**

### A. Key Levels Check (The "Daily Import")

_Before Market Open, identify and draw these lines:_

1. **QQQ 5-Day SMA**: The daily bull/bear line in sand.
   - _Above 5-SMA_: Bullish bias.
   - _Below 5-SMA_: Defensive bias (Caution on Sell Puts).

2. **Support/Resistance**: Major daily support zones.

**Rule**: Do not initiate new trades _directly_ into these levels. Wait for a reaction.

### B. Volatility Discipline

- **High Volatility**: When VIX spikes or price actions are erratic, **reduce frequency**.
- **Logic**: Human reaction time cannot beat algorithms. "Doing nothing is better than doing the wrong thing."

### C. Market Type Identification: Trending vs. Swing

Observe the first 30-60 mins to classify the day:

#### 🟢 Type A: Trending Day (Trend Day)

**Signs:**
- Price consistently making Higher Highs/Lows (or Lower Lows/Highs).
- Candles are "stacking" with little overlap.
- Price holds the 8/21 EMA and stays on one side of VWAP.

**Action**: Trend Following. Buy pullbacks to EMA/VWAP. Breakouts work well. **DO NOT FADE.**

**Kronos Signal**: usually shows strong prob (>70% or <30%).

#### 🟡 Type B: Swing/Range Day (Choppy Day)

**Signs:**
- Price is stuck inside Yesterday's Range or Pre-Market Range.
- Candles have long wicks and lots of overlap.
- VWAP is flat/horizontal; price oscillates around it.

**Action**: Mean Reversion (Fade). Buy at Support, Sell at Resistance. **DO NOT CHASE BREAKOUTS** (they will fail).

**Kronos Signal**: usually hovers around 45-55% or flips rapidly.

---

## 2. Intraday Entry Rules (1-30 min timeframes)

1. **Time Window**: Trade between 9:45 AM - 3:30 PM ET only.
2. **Avoid First 15 Mins**: Too much noise, wait for range to form.
3. **Avoid Last 30 Mins**: Gamma hedging can cause random volatility.

4. **Kronos Validation**:
   - **Long**: P(up) > 65%.
   - **Short**: P(up) < 35%.
   - **Range Day Exception**: If identifying a "Swing Day", can trade reversals at extremes even if P(up) is neutral (50%), relying on Support/Resistance.

5. **Trend Filter**:
   - **Long**: Price > SMA 21 (intraday) AND Price > VWAP.
   - **Short**: Price < SMA 21 (intraday) AND Price < VWAP.

6. **The "5-Day SMA" Rule**:
   - If Price < Daily 5-SMA: No Aggressive Longs (Caution on Sell Puts).
   - If Price > Daily 5-SMA: No Aggressive Shorts.

---

## 3. Weekly Sell Put Strategy (The Engine)

_This is the primary income generator. Strict adherence to parameters is required._

### A. Execution Timing (The Friday Rotation)

1. **Entry/Exit Day**: **Friday**.
   - _Close_: Let existing options expire worthless or close them.
   - _Open_: Enter new positions for the next 1-2 weeks.

2. **Logic**: Capitalize on theta decay over the weekend and establish the next week's range.

### B. Entry Criteria (Must Meet ALL)

1. **Trend Filter**: **Weekly** 5 EMA > 21 EMA (Golden Cross).
   - _Rule_: **NO Naked Puts** if Weekly Trend is Bearish (Dead Cross). Only Spreads or Cash.

2. **Selection**: QQQ (primary) or High Liquidity Tech.

3. **Option Parameters**:
   - **Delta**: `0.05` to `0.15` (High Probability Zone).
   - **DTE**: `1` to `2` Weeks.
   - **Yield Target**: Annualized Return `> 30%`.
   - _Rule_: If volatility is too low to meet the 30% target with these settings, **DO NOT FORCE IT**.

4. **Strike Selection**:
   - MUST be below **Weekly 5-SMA**.
   - OR below **Major Daily Support**.
   - _Safety_: Always leave a buffer; never sell ATM.

### C. Standard Exit (The Happy Path)

1. **Expiration**: Let expire worthless to capture full premium (preferred).
2. **Early Take Profit**: If >50% profit is achieved quickly (e.g., within 1-2 days), close to release buying power.

### D. Defense Mechanisms (The "Oh Sh*t" Plan)

- **Trigger**: Option Delta hits `0.5` (ATM) OR Price crashes through Support.
- **Action**:
  1. Check support strength.
  2. **Roll Down & Out**: Roll to a lower strike and later date.
  3. **Constraint**: Do not roll indefinitely (max ~2 months out). Do not increase risk exposure just to chase premium.

### E. Stop Loss / Circuit Breaker

- **Account Drawdown**: If total account drawdown hits **3% - 5%**:
  - **ACTION**: Mandatory Risk Reduction.
  - Consider **Deep Roll** or **Close Loss**. Do not "hope" it comes back.

### F. Repair Tactics (Advanced)

- **Crash Scenario**: If market freefalls, buy **Long Calls** (utilizing high IV/oversold bounce) to hedge delta.
- **Stalled Rebound**: If rebound stalls, **Sell Calls** (convert to Strangle/Covered Call) to recover losses.
- **Goal**: Minimize drawdown and close flat/small loss. Do not try to be a hero.

---

## 4. Exit Rules (Intraday)

1. **Hard Stop Loss**: Set immediately upon entry. Max 1% loss of total account per trade.

2. **Profit Taking**: Scale out at 2:1 and 3:1 R/R.

3. **Time Stop**: Close all intraday speculation by 3:45 PM ET. No overnight holds for 0DTE/Day trades.

4. **Friday Clearing**: Close all short-term risky positions by Friday 3:00 PM ET.

---

## 5. Portfolio Balance & Sizing

1. **Portfolio Delta**: Aim for **Neutral to Slightly Positive**.
   - _Monitor_: SPX Beta-Weighted Delta.
2. **Leverage**: Max 2x Notional.
3. **Position Size**: Max 2% risk per trade idea.
4. **Tranching**: Enter positions in 2-3 tranches (Scale in), never "All-In" at once.

---

## 6. Risk Management

1. **Daily Circuit Breaker**: Stop trading if down 3% in a single day. Walk away.
2. **Weekly Circuit Breaker**: Reduce size by 50% if down 5% for the week.
3. **Cool Down**: After 3 consecutive losing trades, take a mandatory 1-hour break.
4. **Drawdown Protocol**: If total drawdown > 10%, stop trading, review all "Pain Logs".

---

## 7. Emotional Discipline

1. **No Revenge**: Never increase size to "make back" a loss.
2. **No FOMO**: If you missed the entry at the level, let it go. Don't chase.
3. **Perfect Setup Only**: If Kronos says 60% (weak signal) and chart looks messy, SIT ON HANDS.
4. **Journaling**: Log the "Context" (Trend vs Range) for every trade.

---

## 8. Market Conditions to Avoid

1. **High Impact News**: FOMC, CPI, NFP, Major Tech Earnings.
2. **Low Volume**: Holidays, lunch hour (12:00-1:00 PM ET) on slow days.
3. **The "Chop Zone"**: If Kronos P(up) is consistently 45-55%, STAY FLAT.

---

## 9. AI Copilot Instructions (System Prompt)

_When the user asks for advice, verify against these logic gates:_

### Intraday Trading Checks

1. **Context Check**: "Based on the 5-min chart and VWAP, is today a Trending Day or Range Day?" (Ask user if data is missing).

2. **The 5-Day Filter**: "Where is the price relative to the Daily 5-SMA? Does this trade align with that bias?"

3. **Kronos Confirmation**: "Is P(up) strong enough (>65%/<35%) to justify this entry?"

4. **Rule Enforcement**:
   - If the user is trying to trade counter-trend on a "Trending Day", **STOP THEM**.
   - If they are chasing a breakout on a "Range Day", **STOP THEM**.

### Weekly Sell Put Checks

1. **Trend Check**: "Is QQQ Weekly 5/21 EMA bullish?" (If No -> STOP Naked Puts).

2. **Friday Check**: "Is today Friday? If yes, look for Rotation (Close old/Open new)."

3. **Parameter Check**: "Is the proposed Delta between 0.05-0.15? Is the Strike below the Weekly 5-SMA?"

4. **Risk Check**: "If the user is asking about a losing position, reference the '3-5% Drawdown Rule' and suggest Rolling or Hedging."

---

## Quick Reference Table

| Condition | Day Type | Kronos Signal | Action |
|-----------|----------|---------------|--------|
| Price > 5-SMA + P(up) > 70% + Stacking Candles | Trend Day | Strong Bullish | BUY pullbacks, NO fading |
| Price < 5-SMA + P(up) < 30% + Stacking Candles | Trend Day | Strong Bearish | SELL rallies, NO fading |
| Price in Range + P(up) 45-55% + Wicky Candles | Range Day | Neutral | FADE extremes, NO breakouts |
| P(up) 45-55% + Unclear Structure | Chop | Avoid | SIT ON HANDS |

---

*Last Updated: 2025-11-22*
*These rules are AI-enforced via the Kronos prediction system*
