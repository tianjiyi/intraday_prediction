# Trading Rules & Constitution (AI-Enforced)

> **IMPORTANT**: These rules are designed to enforce discipline and protect capital.
> The LLM will check your trades against these rules.

---

## 1. Daily Mindset & Context Analysis (The "First 30 Minutes" Ritual)

**You MUST define the battlefield before fighting.**

### A. Key Levels Check (The "Daily Import")

Before Market Open, identify and draw these lines:

- **5-Day SMA**: The bull/bear line in sand. (Above = Bullish Bias, Below = Bearish Bias).
- **PDH / PDL**: Previous Day High and Low.
- **PMH / PML**: Pre-Market High and Low.
- **Big Levels**: Major daily support/resistance zones.

**Rule**: Do not initiate new trades directly into these levels. Wait for a reaction (Bounce or Clean Break & Retest).

### B. Market Type Identification: Trending vs. Swing

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

## 2. Position Sizing Rules

1. **Maximum Risk**: Never risk more than 2% of portfolio on a single trade idea.
2. **Tranching**: Enter positions in 2-3 tranches (Scale in), never "All-In" at once.
3. **Leverage Limits**: Maximum 2x notional leverage. Prefer Cash-Secured for Puts.

---

## 3. Entry Rules

### Intraday Trading (1-30 min timeframes)

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

### Weekly/Swing Trading (Sell Put Strategy)

1. **Entry Days**: Monday-Wednesday preferred.
2. **No Friday Entries**: Do not open new Premium Selling positions on Fridays (Gamma risk).
3. **Trend Alignment**: Weekly 5/21 EMA must be bullish for Naked Puts.
4. **Strike Selection**: Below the Daily 5-SMA or major Support.

---

## 4. Exit Rules

1. **Hard Stop Loss**: Set immediately upon entry. Max 1% loss of total account per trade.

2. **Profit Taking**:
   - **Intraday**: Scale out at 2:1 and 3:1 R/R.
   - **Sell Put**: Close at 50% profit if achieved quickly (<3 days).

3. **Time Stop**: Close all intraday speculation by 3:45 PM ET. No overnight holds for 0DTE/Day trades.

4. **Friday Clearing**: Close all short-term risky positions by Friday 3:00 PM ET.

---

## 5. Risk Management

1. **Daily Circuit Breaker**: Stop trading if down 3% in a single day. Walk away.
2. **Weekly Circuit Breaker**: Reduce size by 50% if down 5% for the week.
3. **Cool Down**: After 3 consecutive losing trades, take a mandatory 1-hour break.
4. **Drawdown Protocol**: If total drawdown > 10%, stop trading, review all "Pain Logs".

---

## 6. Emotional Discipline

1. **No Revenge**: Never increase size to "make back" a loss.
2. **No FOMO**: If you missed the entry at the level, let it go. Don't chase.
3. **Perfect Setup Only**: If Kronos says 60% (weak signal) and chart looks messy, SIT ON HANDS.
4. **Journaling**: Log the "Context" (Trend vs Range) for every trade.

---

## 7. Market Conditions to Avoid

1. **High Impact News**: FOMC, CPI, NFP, Major Tech Earnings.
2. **Low Volume**: Holidays, lunch hour (12:00-1:00 PM ET) on slow days.
3. **The "Chop Zone"**: If Kronos P(up) is consistently 45-55%, STAY FLAT.

---

## 8. AI Copilot Instructions (System Prompt)

When the user asks for advice, verify against these logic gates:

1. **Context Check**: "Based on the 5-min chart and VWAP, is today a Trending Day or Range Day?" (Ask user if data is missing).

2. **The 5-Day Filter**: "Where is the price relative to the Daily 5-SMA? Does this trade align with that bias?"

3. **Kronos Confirmation**: "Is P(up) strong enough (>65%/<35%) to justify this entry?"

4. **Rule Enforcement**:
   - If the user is trying to trade counter-trend on a "Trending Day", **STOP THEM**.
   - If they are chasing a breakout on a "Range Day", **STOP THEM**.

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
