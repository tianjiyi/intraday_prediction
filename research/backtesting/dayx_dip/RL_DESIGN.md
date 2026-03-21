# DayX Dip RL — Reinforcement Learning Entry Agent for Intraday Dip Buy

## Context
The rule-based `dayx_dip` strategy (BB lower + RSI + CCI + exhaustion + pivot candle) works but fires too rarely — only ~24 trades/year on 3Min QQQ (about 1 trade every 10 trading days). The user wants **at least 1 entry per day**. Rather than loosening hard thresholds (which risks overfitting), we train an RL agent to learn which "close enough" indicator states are worth entering. Exits stay rule-based (BB upper cross / stop loss / EOD flatten). The agent only controls entry timing.

## Design Decisions (from brainstorming)
- **Role**: Entry-only agent — replaces the `signal_bear_exhaustion_dip` boolean, keeps all exit logic
- **Reward**: R-multiple based — realized `(exit_price - entry_price) / risk` per trade
- **Framework**: Stable-Baselines3 + Gymnasium
- **Algorithm**: PPO (handles continuous observations well, stable training)
- **Data split**: Train 2020–2022, Val 2023 (checkpoint selection), Test 2024 (untouched holdout)

## Architecture

### Gymnasium Environment: `DipBuyEnv`

Wraps the existing dayx_dip data pipeline into a `gymnasium.Env`. One episode = one trading day. The agent sees indicator values each bar and decides whether to enter long.

### Observation Space (11 features, normalized in features.py)

| # | Feature | Formula | Range |
|---|---------|---------|-------|
| 0 | `bb_position` | `(close - bb_lower) / (bb_upper - bb_lower)` | ~[0, 1], can exceed |
| 1 | `rsi_norm` | `rsi14 / 100` | [0, 1] |
| 2 | `cci_norm` | `clip(cci14 / 200, -1, 1)` | [-1, 1] |
| 3 | `vwap_dist` | `clip((close - vwap) / atr, -3, 3) / 3` | [-1, 1] |
| 4 | `atr_pct` | `clip(atr / close * 100, 0, 5) / 5` | [0, 1] |
| 5 | `exhaust_down` | `min(exhaustion_down, 10) / 10` | [0, 1] |
| 6 | `hammer` | `float(hammer)` | {0, 1} |
| 7 | `bull_engulfing` | `float(bullish_engulfing)` | {0, 1} |
| 8 | `time_of_day` | `clip(minutes_since_930 / 390, 0, 1)` | [0, 1] |
| 9 | `has_position` | `float(in_position)` | {0, 1} |
| 10 | `rel_volume` | `clip(volume / volume_sma_20, 0, 5) / 5` | [0, 1] |

**Note**: Features 8-10 are computed in `features.py` at observation build time, not by `indicators.py`. The `observation_space` is declared as `Box(low=-1.5, high=1.5, shape=(11,))` to accommodate minor overshoot from clipping boundaries.

### Action Space

`Discrete(2)`:
- `0` = **Hold** (do nothing)
- `1` = **Buy** (enter long if not already in position)

### Guardrails (hard rules the agent cannot override)
- Cannot enter if already in a position
- Cannot enter after `no_new_entries_after` (14:00 default)
- Cannot enter before `entry_earliest` (10:00 default)
- Stop loss is always set at `entry_price - atr * stop_atr_mult`
- **Invalid buy penalty**: action=1 when guardrails block → reward **-0.05** (deters buy-spamming)
- Max **3 trades per day** — after 3 entries, further buys are invalid

### Reward Function: Cost-Adjusted R-Multiple

The reward is the **cost-adjusted R-multiple** of each trade, where `R = ATR * stop_atr_mult` at entry, and **round-trip costs** are deducted.

**Round-trip costs** = `2 * slippage_per_share + 2 * commission_per_share` (entry + exit). The backtester's `slippage_per_share` ($0.01 default) and `commission_per_share` ($0.00 default) are each one-way, so the full round-trip is $0.02 with defaults.

| Event | Reward | Rationale |
|-------|--------|-----------|
| Each bar (no position) | 0 | No cost for waiting |
| Each bar (in position) | **gamma * phi(s') - phi(s)** | Potential-based shaping (see below) |
| **Invalid buy** (guardrail blocked) | **-0.05** | Penalizes buy-spamming |
| **Trade exit** | **(exit_price - entry_price - round_trip_costs) / R** | Trade quality after costs, in risk units |

**Potential-based reward shaping** (Ng et al., 1999):
- Potential function: `phi(s) = 0.1 * unrealized_R` when in position, `phi(s) = 0` otherwise
- Per-bar shaping: `gamma * phi(s_{t+1}) - phi(s_t)` where gamma=0.99
- This is guaranteed not to change the optimal policy (policy invariance theorem)
- Over an entire hold, total shaping sums to approximately `0.1 * final_unrealized_R - 0.1 * initial_unrealized_R`, which is bounded and small relative to the terminal reward
- On exit bar: shaping = `gamma * 0 - phi(s_t) = -0.1 * last_unrealized_R` (cancels out the accumulated shaping)

Examples with ATR=0.50, stop_mult=2.0 (R=$1.00), round-trip costs=$0.02:
- BB upper cross at +$2.50 above entry: reward = **(2.50 - 0.02) / 1.00 = +2.48**
- Stop loss hit: reward = **(-1.00 - 0.02) / 1.00 = -1.02**
- EOD flatten at +$0.30: reward = **(0.30 - 0.02) / 1.00 = +0.28**
- EOD flatten at -$0.40: reward = **(-0.40 - 0.02) / 1.00 = -0.42**

**Why cost-adjusted R-multiple?**
1. Aligns reward with actual P&L including round-trip execution costs
2. Prevents farming small-R trades that are negative after costs
3. Normalizes across volatility regimes
4. No categorical mislabeling (EOD trades get their true R, positive or negative)
5. Eliminates the no-entry penalty that created "always buy" degenerate policy

### Exit Logic (rule-based, inside env.step())

**Pessimistic path resolution** for OHLC bar ambiguity:

When both `low <= stop_price` AND `close >= bb_upper` on the same bar, we cannot know the intrabar sequence. The conservative assumption is **stop was hit first**.

Priority:
1. **Stop loss**: `low <= stop_price` → exit at stop_price (checked FIRST — pessimistic)
2. **BB upper cross**: `close >= bb_upper` → exit at close price (only if stop wasn't hit)
3. **EOD flatten** at 15:55 → exit at close price

This avoids overstating reward on bars where both conditions trigger. A trade that "survived" the stop and closed above BB upper is rare on 1-3Min bars, but when it happens, the pessimistic resolution prevents inflated backtest results.

### Episode Structure

- **Reset**: start of a trading day (random during training, sequential for eval)
- **Step**: advance 1 bar
- **Done**: when market hits last RTH bar or all trades exhausted for the day
- Multiple entries per day allowed (up to `max_trades_per_day = 3`)

### Data Split (clean 3-way)

| Split | Period | Purpose |
|-------|--------|---------|
| **Train** | 2020-01-01 → 2022-12-31 | PPO learning (random episode sampling) |
| **Val** | 2023-01-01 → 2023-12-31 | EvalCallback checkpoint selection |
| **Test** | 2024-01-01 → 2025-01-01 | Final evaluation (never seen during training) |

The val set is ONLY used for `EvalCallback` to select best checkpoint. The test set is ONLY used once for final comparison.

## Folder Structure

```
research/backtesting/dayx_dip_rl/
├── __init__.py
├── config.py           # RLConfig dataclass (DayXDipConfig + RL hyperparams)
├── env.py              # DipBuyEnv(gymnasium.Env)
├── features.py         # build_observation(row, state) → np.ndarray
├── train.py            # SB3 PPO training with EvalCallback
├── evaluate.py         # Backtest trained model, collect trades + metrics
├── compare.py          # Side-by-side: RL agent vs rule-based dayx_dip
└── run.py              # CLI: train / evaluate / compare
```

## Reused from `dayx_dip/`

| Module | What we reuse | Note |
|--------|---------------|------|
| `dayx_dip/data_loader.py` | `load_data()`, `fetch_bars_cached()`, `filter_rth()`, `add_session_markers()` | |
| `dayx_dip/indicators.py` | `compute_all()` — produces 8 indicator columns: `rsi14`, `cci14`, `bb_mid/upper/lower`, `vwap`, `atr`, `exhaustion_down`, `hammer`, `bullish_engulfing`, `bb_upper_cross` | Obs features `time_of_day`, `has_position`, `rel_volume` are derived at runtime in `features.py`, NOT from `compute_all()` |
| `dayx_dip/strategy.py` | `Trade` dataclass only | Exit logic reimplemented in env with pessimistic path resolution |
| `dayx_dip/metrics.py` | `compute_metrics()`, `hourly_pnl()`, `print_summary()` | For rule-based comparison only |

## Implementation Steps

### Step 1: `config.py`
```python
@dataclass
class RLConfig:
    dip_config: DayXDipConfig = field(default_factory=DayXDipConfig)

    # RL hyperparams
    algorithm: str = "PPO"
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    total_timesteps: int = 1_000_000

    # Reward
    reward_invalid_buy: float = -0.05
    max_trades_per_day: int = 3

    # Data split
    train_start: str = "2020-01-01"
    train_end: str = "2022-12-31"
    val_start: str = "2023-01-01"
    val_end: str = "2023-12-31"
    test_start: str = "2024-01-01"
    test_end: str = "2025-01-01"

    timeframe: str = "3Min"
    symbol: str = "QQQ"
```

### Step 2: `features.py`
- `build_observation(row, has_position, volume_sma) → np.ndarray[11]`
- All features clipped in code (not relying on Box bounds)
- NaN → 0 for early warmup bars

### Step 3: `env.py` — DipBuyEnv
- `__init__`: load data, compute indicators, group by day
- `reset()`: pick a day, return first bar obs
- `step(action)`:
  1. If in position → check exits (stop_loss → bb_upper_cross → eod_flatten, pessimistic) → emit cost-adjusted R-multiple reward
  2. If action=1 and guardrails block → reward -0.05
  3. If action=1 and guardrails pass → enter (set stop at entry - ATR * mult)
  4. Advance bar, build next obs
  5. Return (obs, reward, terminated, truncated, info)
- `observation_space`: `Box(low=-1.5, high=1.5, shape=(11,))`
- `action_space`: `Discrete(2)`

### Step 4: `train.py`
- Train PPO on train split, EvalCallback on val split
- Checkpoints saved to `data/backtests/dayx_dip_rl/models/`
- TensorBoard logs to `data/backtests/dayx_dip_rl/logs/`

### Step 5: `evaluate.py`
- Load best model from val checkpoint
- Run on TEST split (2024) deterministically
- Collect Trade objects, compute metrics, plot equity

### Step 6: `compare.py`
- Side-by-side: RL vs rule-based on test split
- Metrics table + equity curve overlay

### Step 7: `run.py`
```bash
python research/backtesting/dayx_dip_rl/run.py train --timesteps 1000000
python research/backtesting/dayx_dip_rl/run.py evaluate --split test
python research/backtesting/dayx_dip_rl/run.py compare --plot
```

## Dependencies
```
stable-baselines3>=2.3.0
gymnasium>=0.29.0
tensorboard>=2.15.0
shimmy>=1.3.0
```

## Verification

1. **No "always buy" convergence**: check that trained agent's entry rate < 50% of eligible bars
2. **Entry frequency**: average >= 1.0 entries/day on train and test (matching stated goal)
3. **R-multiple distribution**: mean R > 0 on val set (agent learned positive expectancy after costs)
4. **Cost impact**: verify that cost-adjusted R < raw R for every trade (costs are being deducted)
5. **Exit distribution**: track bb_upper_cross vs stop_loss vs eod_flatten ratios
6. **Pessimistic resolution**: on bars where low <= stop AND close >= bb_upper, verify stop_loss is the exit reason (not bb_upper_cross)
7. **No data leak**: test set metrics computed ONCE after final model selection on val checkpoint
8. **Regime test**: report metrics separately for 2020 (COVID), 2021 (bull), 2022 (bear)
9. **Invalid buy rate**: should decrease over training (agent learns when guardrails apply)
10. **Shaping sanity**: verify total shaping reward (0.01 * unrealized_R) is small relative to exit rewards
11. **Reproducibility**: seeds for numpy, torch, gymnasium
