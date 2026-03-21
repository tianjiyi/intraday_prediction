"""
DipBuyEnv — Gymnasium environment for RL-based intraday dip buy entry.

The agent controls only entry timing. Exits are rule-based:
  1. BB upper cross (close >= bb_upper)
  2. Stop loss (low <= stop_price)
  3. EOD flatten at 15:55

Reward = cost-adjusted R-multiple: (exit_price - entry_price - costs) / risk
Pessimistic path resolution: stop always wins on ambiguous bars.
"""

import logging
from typing import Optional

import gymnasium as gym
import numpy as np
import pandas as pd

from .config import RLConfig
from .features import build_observation

logger = logging.getLogger(__name__)


class DipBuyEnv(gym.Env):
    """
    Gymnasium environment for DayX Dip RL agent.

    Episode = one trading day.
    Observation = 11 normalized features.
    Action = Discrete(2): 0=hold, 1=buy.
    Reward = R-multiple on trade exit, -0.05 for invalid buy.
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: RLConfig, mode: str = "train", seed: int = 42):
        super().__init__()

        self.cfg = cfg
        self.mode = mode
        self._seed = seed
        self._rng = np.random.RandomState(seed)

        # Curriculum: forced random entries that decay over training
        # Only active during training; eval/test always use agent's policy
        self.exploration_rate = 0.3 if mode == "train" else 0.0
        self._total_steps = 0

        # Observation and action spaces
        self.observation_space = gym.spaces.Box(
            low=-1.5, high=1.5, shape=(11,), dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(2)

        # Load and prepare data
        self._load_data()

        # Episode state
        self._day_idx = 0
        self._bar_idx = 0
        self._position: Optional[dict] = None  # {entry_price, stop_price, risk}
        self._trades_today = 0
        self._day_bars: Optional[pd.DataFrame] = None
        self._volume_sma: Optional[pd.Series] = None

    def _load_data(self):
        """Load data, compute indicators, split into daily episodes."""
        import sys
        from pathlib import Path
        _root = str(Path(__file__).resolve().parents[3])
        if _root not in sys.path:
            sys.path.insert(0, _root)
        import types
        sys.modules.setdefault("research.backtesting", types.ModuleType("research.backtesting"))
        sys.modules["research.backtesting"].__path__ = [
            str(Path(_root) / "research" / "backtesting")
        ]

        from research.backtesting.dayx_dip.config import DayXDipConfig
        from research.backtesting.dayx_dip.data_loader import fetch_bars_cached, filter_rth, add_session_markers
        from research.backtesting.dayx_dip.indicators import compute_all

        # Determine date range based on mode
        if self.mode == "train":
            start, end = self.cfg.train_start, self.cfg.train_end
        elif self.mode == "val":
            start, end = self.cfg.val_start, self.cfg.val_end
        else:  # test
            start, end = self.cfg.test_start, self.cfg.test_end

        data_cfg = DayXDipConfig(
            symbol=self.cfg.symbol,
            timeframe=self.cfg.timeframe,
            start_date=start,
            end_date=end,
        )

        df = fetch_bars_cached(data_cfg)
        df = filter_rth(df, data_cfg)
        df = add_session_markers(df)
        df = compute_all(df, self.cfg.dip_config)

        # Rolling volume SMA for relative volume feature
        df["volume_sma_20"] = df["volume"].rolling(20, min_periods=1).mean()

        # Group by trading day
        self._daily_groups = {
            date: group for date, group in df.groupby("session_date")
            if len(group) >= 20  # skip short days
        }
        self._dates = sorted(self._daily_groups.keys())

        if not self._dates:
            raise ValueError(f"No valid trading days found for {self.mode} ({start} → {end})")

        logger.info(f"DipBuyEnv [{self.mode}]: {len(self._dates)} trading days loaded")

    def reset(self, seed=None, options=None):
        """Start a new episode (trading day)."""
        super().reset(seed=seed)

        if self.mode == "train":
            self._day_idx = self._rng.randint(0, len(self._dates))
        else:
            # Sequential for val/test
            self._day_idx = (self._day_idx + 1) % len(self._dates)

        date = self._dates[self._day_idx]
        self._day_bars = self._daily_groups[date]
        self._bar_idx = 0
        self._position = None
        self._trades_today = 0

        obs = self._get_obs()
        info = {"date": str(date), "bars": len(self._day_bars)}
        return obs, info

    def step(self, action: int):
        """
        Advance one bar.

        Returns: (obs, reward, terminated, truncated, info)
        """
        reward = 0.0
        info = {}
        row = self._day_bars.iloc[self._bar_idx]
        timestamp = self._day_bars.index[self._bar_idx]

        dcfg = self.cfg.dip_config
        flatten_h, flatten_m = map(int, dcfg.eod_flatten_time.split(":"))
        is_eod = (
            timestamp.hour > flatten_h or
            (timestamp.hour == flatten_h and timestamp.minute >= flatten_m)
        )

        # --- Check exits if in position ---
        if self._position is not None:
            exit_price, exit_reason = self._check_exits(row, is_eod)
            if exit_price is not None:
                # Cost-adjusted R-multiple (round-trip: entry + exit costs)
                rt_costs = 2 * (dcfg.commission_per_share + dcfg.slippage_per_share)
                r_mult = (exit_price - self._position["entry_price"] - rt_costs) / self._position["risk"]
                # Potential-based shaping on exit: gamma * phi(s') - phi(s)
                # phi(s') = 0 (no position after exit), phi(s) = 0.1 * prev_unrealized_R
                prev_phi = 0.1 * self._position.get("last_unrealized_r", 0)
                shaping = self.cfg.gamma * 0 - prev_phi
                reward = r_mult + shaping
                info["trade_exit"] = exit_reason
                info["r_multiple"] = round(r_mult, 3)
                info["shaping"] = round(shaping, 4)
                info["entry_price"] = self._position["entry_price"]
                info["exit_price"] = exit_price
                self._position = None
            else:
                # Potential-based shaping: gamma * phi(s') - phi(s)
                # phi(s) = 0.1 * unrealized_R
                unrealized_r = (row["close"] - self._position["entry_price"]) / self._position["risk"]
                prev_phi = 0.1 * self._position.get("last_unrealized_r", 0)
                curr_phi = 0.1 * unrealized_r
                shaping = self.cfg.gamma * curr_phi - prev_phi
                reward += shaping
                self._position["last_unrealized_r"] = unrealized_r

        # --- Curriculum: force random entries early in training ---
        if (action == 0 and self._position is None
                and self.exploration_rate > 0
                and self._rng.random() < self.exploration_rate):
            action = 1  # override hold → buy

        # --- Opportunity cost: small penalty for sitting flat during entry window ---
        if self._position is None and action == 0:
            can_enter, _ = self._can_enter(timestamp)
            if can_enter:
                reward += self.cfg.reward_flat_per_bar  # -0.001 default

        # --- Handle entry action ---
        if action == 1 and self._position is None:
            can_enter, block_reason = self._can_enter(timestamp)
            if can_enter:
                atr_val = row.get("atr", 0)
                if atr_val > 0:
                    risk = atr_val * dcfg.stop_atr_mult
                    self._position = {
                        "entry_price": row["close"],
                        "stop_price": row["close"] - risk,
                        "risk": risk,
                        "last_unrealized_r": 0.0,
                    }
                    self._trades_today += 1
                    info["trade_entry"] = True
                    info["entry_price"] = row["close"]
                    info["stop_price"] = self._position["stop_price"]
            else:
                # Invalid buy — penalize spam
                reward += self.cfg.reward_invalid_buy
                info["invalid_buy"] = block_reason

        # --- Track total steps for curriculum decay ---
        self._total_steps += 1

        # --- Advance to next bar ---
        self._bar_idx += 1
        terminated = self._bar_idx >= len(self._day_bars)

        # If EOD and still in position, force flatten
        if terminated and self._position is not None:
            last_row = self._day_bars.iloc[-1]
            rt_costs = 2 * (dcfg.commission_per_share + dcfg.slippage_per_share)
            r_mult = (last_row["close"] - self._position["entry_price"] - rt_costs) / self._position["risk"]
            prev_phi = 0.1 * self._position.get("last_unrealized_r", 0)
            shaping = self.cfg.gamma * 0 - prev_phi
            reward += r_mult + shaping
            info["trade_exit"] = "eod_flatten_forced"
            info["r_multiple"] = round(r_mult, 3)
            self._position = None

        obs = self._get_obs() if not terminated else np.zeros(11, dtype=np.float32)
        return obs, reward, terminated, False, info

    def _check_exits(self, row: pd.Series, is_eod: bool):
        """
        Check exits with PESSIMISTIC path resolution for OHLC bars.

        When both stop_loss and bb_upper_cross could trigger on the same bar,
        we assume stop was hit first (conservative). This avoids overstating
        reward on ambiguous intrabar paths.

        Priority:
          1. Stop loss (low <= stop) — always checked first (pessimistic)
          2. BB upper cross (close >= bb_upper) — only if stop wasn't hit
          3. EOD flatten

        Returns (exit_price, reason) or (None, None).
        """
        close = row["close"]
        low = row["low"]
        bb_upper = row.get("bb_upper", float("inf"))
        stop = self._position["stop_price"]

        # 1. Stop loss — pessimistic: if stop was breached, assume it hit first
        if low <= stop:
            return stop, "stop_loss"

        # 2. BB upper cross — only reachable if stop wasn't hit
        if close >= bb_upper:
            return close, "bb_upper_cross"

        # 3. EOD flatten
        if is_eod:
            return close, "eod_flatten"

        return None, None

    def _can_enter(self, timestamp) -> tuple[bool, str]:
        """Check guardrails. Returns (can_enter, block_reason)."""
        dcfg = self.cfg.dip_config

        if self._position is not None:
            return False, "already_in_position"

        if self._trades_today >= self.cfg.max_trades_per_day:
            return False, "max_trades_reached"

        early_h, early_m = map(int, dcfg.entry_earliest.split(":"))
        if timestamp.hour < early_h or (timestamp.hour == early_h and timestamp.minute < early_m):
            return False, "before_entry_earliest"

        cutoff_h, cutoff_m = map(int, dcfg.no_new_entries_after.split(":"))
        if timestamp.hour > cutoff_h or (timestamp.hour == cutoff_h and timestamp.minute > cutoff_m):
            return False, "after_entry_cutoff"

        return True, ""

    def _get_obs(self) -> np.ndarray:
        """Build observation vector for current bar."""
        if self._bar_idx >= len(self._day_bars):
            return np.zeros(11, dtype=np.float32)

        row = self._day_bars.iloc[self._bar_idx]
        volume_sma = row.get("volume_sma_20", 1.0)
        has_pos = self._position is not None

        return build_observation(row, has_pos, volume_sma)
