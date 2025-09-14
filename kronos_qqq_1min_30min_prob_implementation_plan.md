# Implementation Plan — QQQ 1-min → 30-min Probabilistic Paths (Kronos)

## reference
- https://github.com/shiyu-coder/Kronos
- https://shiyu-coder.github.io/Kronos-demo/

## note
- for cuda please use 12.8 this version supports RTX5090
- also setup python virtual env for me

## 0) Goal & Scope
- **Goal:** From the most recent **3 trading days** of **1-minute QQQ OHLCV**, generate **N sampled future paths** for the next **30 minutes (30 bars)** using Kronos’ autoregressive sampling, and compute:
  - `P(close[t+30] > close[t])` — 30-min up probability  
  - Expected 30-min return (mean over paths)  
  - Per-step percentile bands (e.g., 10/50/90%)  
  - *(Optional)* Prob. of reverting to intraday VWAP / mid-band
- **MVP Output:** Print metrics to console + write JSON/CSV of sampled paths & summary.  
  *(Fine-tuning & web UI can follow later.)*

---

## 1) Assumptions & Design Choices
- **Data:** Alpaca 1-min bars for `QQQ` (RTH by default; configurable for extended hours).
- **Lookback:** Keep **3 trading days** for features; feed only the **last `L=480..512` bars** to Kronos if the model has a max context.
- **Horizon:** 30 bars (30 minutes).
- **Sampling:** Start with `N=100` paths (tune by latency).
- **Sampling controls:** `temperature (T)=1.0`, `top_p=0.9`.
- **Time:** Use **UTC** consistently; ensure minute alignment & gap-free series.

---

## 2) Environment Setup
```bash
git clone https://github.com/shiyu-coder/Kronos
cd Kronos
pip install -r requirements.txt
# optional
pip install -e .
pip install alpaca-py pandas numpy matplotlib pytz
```
- Download Kronos weights per repo instructions (or let the loader pull from HF).
- Set Alpaca keys: `ALPACA_KEY_ID`, `ALPACA_SECRET_KEY`.

---

## 3) Data Ingestion (Alpaca)
- Pull **last 3 trading days** of 1-min OHLCV for QQQ.  
- Sort by timestamp (UTC).  
- *(Optional)* Filter to **RTH (09:30–16:00 ET)**.  
- Ensure no gaps (forward-fill close with zero volume if you prefer; be consistent).

---

## 4) Preprocessing for Kronos
- Build a strictly ordered, gap-free DataFrame: columns `open, high, low, close, volume`.
- Trim to **last `L` bars** (e.g., `L=480`) for Kronos input.  
- Keep the full 3-day window separately if you need VWAP/other MR features.

---

## 5) Model Load
- Start with a **light checkpoint** (e.g., `kronos-mini` or `kronos-small`) for quick MVP latency.
- Instantiate predictor/tokenizer once and reuse.

---

## 6) Monte-Carlo Forecast (N Paths)
- For each sample:
  1. Condition on the last `L` bars.
  2. **Autoregressively sample** 30 future 1-min bars with `T`, `top_p`.
- Aggregate closes into array **`[N, 30]`** (keep highs/lows too if you need barrier/“touch” probabilities).

---

## 7) Metrics & Outputs
- **Primary:**
  - `p_up_30m = mean( paths[:, -1] > close_now )`
  - `exp_ret_30m = mean( paths[:, -1] / close_now - 1.0 )`
  - Per-step percentiles: p10/p50/p90 for uncertainty band.
- **Mean-reversion helpers (optional):**
  - `p_revert_VWAP_30m = fraction of paths that touch >= today_VWAP within 30 bars`
  - `p_retrace_midband` (e.g., SMA/Bollinger middle)
  - Drawdown distribution for stop sizing
- **Artifacts:**
  - `pred_summary_<SYMBOL>_<ts>.json`
  - `paths_<SYMBOL>_<ts>.csv` (N×30 closes; optionally highs/lows)
  - *(Optional)* quick PNG plot for sanity check

---

## 8) Minimal CLI Script (skeleton)

> Have Claude wire the real Kronos import & predictor calls and replace the stubbed sampler.

```python
# cli_kronos_prob_qqq.py
import os, json, datetime as dt
import numpy as np, pandas as pd

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# TODO: replace with actual Kronos imports per repo
# from kronos import KronosPredictor

SYMBOL = "QQQ"
LOOKBACK_BARS = 480
HORIZON = 30
N_SAMPLES = 100
TEMP = 1.0
TOP_P = 0.9

def fetch_alpaca_1m(symbol: str, days: int = 3) -> pd.DataFrame:
    client = StockHistoricalDataClient(os.getenv("ALPACA_KEY_ID"), os.getenv("ALPACA_SECRET_KEY"))
    end = dt.datetime.utcnow()
    start = end - dt.timedelta(days=7)          # buffer to cover 3 trading days
    req = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Minute,
        start=start, end=end, adjustment=None
    )
    bars = client.get_stock_bars(req).df
    df = bars.xs(symbol).rename(columns={
        "open":"open","high":"high","low":"low","close":"close","volume":"volume"
    }).sort_index()
    return df

def prepare_kronos_input(df: pd.DataFrame) -> pd.DataFrame:
    return df.tail(LOOKBACK_BARS)[["open","high","low","close","volume"]].copy()

def monte_carlo_forecast_kronos(kronos, context_df: pd.DataFrame,
                                horizon=HORIZON, n=N_SAMPLES, T=TEMP, top_p=TOP_P):
    # TODO: Replace with repo's actual sampling API (temperature/top_p/sample_count)
    # returns np.ndarray shape [n, horizon] of sampled close paths
    paths = []
    for _ in range(n):
        # sampled_df = kronos.sample_future(context_df, horizon=horizon, temperature=T, top_p=top_p)
        # close_path = sampled_df["close"].to_numpy()
        close_path = np.full(horizon, context_df["close"].iloc[-1])  # stub
        paths.append(close_path)
    return np.stack(paths, axis=0)

def summarize(paths: np.ndarray, close_now: float):
    last_close = paths[:, -1]
    p_up = float(np.mean(last_close > close_now))
    exp_ret = float(np.mean(last_close / close_now - 1.0))
    pct10 = np.percentile(paths, 10, axis=0).tolist()
    pct50 = np.percentile(paths, 50, axis=0).tolist()
    pct90 = np.percentile(paths, 90, axis=0).tolist()
    return {
        "p_up_30m": p_up,
        "exp_ret_30m": exp_ret,
        "percentiles": {"p10": pct10, "p50": pct50, "p90": pct90}
    }

def main():
    df = fetch_alpaca_1m(SYMBOL, days=3)
    ctx = prepare_kronos_input(df)
    close_now = float(ctx["close"].iloc[-1])

    # kronos = KronosPredictor.load("kronos-mini-or-small")
    # paths = monte_carlo_forecast_kronos(kronos, ctx)
    paths = np.random.normal(loc=close_now, scale=close_now*0.001, size=(N_SAMPLES, HORIZON))  # temp stub

    summary = summarize(paths, close_now)

    ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    pd.DataFrame(paths).to_csv(f"paths_{SYMBOL}_{ts}.csv", index=False)
    with open(f"pred_summary_{SYMBOL}_{ts}.json","w") as f:
        json.dump({
            "symbol": SYMBOL,
            "now_close": close_now,
            "n_samples": N_SAMPLES,
            "temp": TEMP,
            "top_p": TOP_P,
            "horizon_min": HORIZON,
            "summary": summary
        }, f, indent=2)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
```

---

## 9) Validation & Sanity Checks (MVP)
- **Shapes:** `paths.shape == (N, 30)`; monotonic timestamps; no NaNs.
- **Sampling behavior:** `T→0` or small `top_p` → narrower bands; larger values → wider bands.
- **Latency:** Target end-to-end < 1–2s on your rig for `N≈50–100` (tune `N`).

---

## 10) MR-Aligned Extra Metrics (Optional)
- `p_hit_VWAP_30m`: share of paths where max(high) ≥ current-session VWAP within 30 min.  
- `p_retrace_midband`: share touching SMA/Bollinger middle.  
- `drawdown_pXX`: e.g., 90th-pct drawdown to size stops.

---

## 11) Backtest Harness (Next)
- Rolling 1-min walk-forward:
  - At each minute, run forecast → log `p_up_30m`.
  - **Toy rule:** go long if `p_up_30m ≥ 0.60` **and** MR deviation (e.g., z-score) is high; exit after 30 min or stop.
  - Record P&L **net of fees/slippage**.
- **Calibration:** reliability diagram for `p_up_30m`; isotonic/Platt calibration if needed.

---

## 12) Micro-Tuning Plan (Later)
- Gather 2–6 weeks of 1-min QQQ; fine-tune Kronos with small LR & early stopping.  
- Re-run backtests; compare A/B vs vanilla checkpoint; watch for overfit.

---

## 13) Realtime Mode (Later)
- Minute scheduler / Alpaca streaming:
  - Append latest bar → trim to `L` → run forecast → emit JSON (Redis/pubsub or file).
- Fail-safe: on inference timeout or data gaps → **skip** this bar.

---

## 14) Web Demo (Later)
- **FastAPI** endpoint `/forecast` returns summary JSON + percentiles.  
- **Front-end (Plotly):** mean path + uncertainty band; display `p_up_30m`.

---

## Deliverables for Claude (Phase-1 “simple output”)
1. `cli_kronos_prob_qqq.py` with **real Kronos API** wired (replace stubs).  
2. `config.yaml` (symbol, horizon, N, T, top_p, lookback L, RTH flag).  
3. `README.md` with:
   - Setup (env vars, weights)
   - Run command
   - Sample outputs (JSON/CSV)
4. *(Optional)* `plot_paths.py` for quick visualization.

---

## Edge Cases & Guardrails
- **Open warm-up:** If you don’t yet have `L` bars (just after open), warm up or wait.  
- **Event risk:** Quiet mode around CPI/FOMC; degrade signals or skip.  
- **Context limits:** Respect model max context; raise `L` only if supported.  
- **Extended hours:** Config switch; MR behavior differs outside RTH.
