"""
SignalFilterEnv — Gymnasium environment for RL signal accept/reject.

The agent only acts when the DayX strategy fires a signal.
Each step = one signal decision (accept or reject).
Episode = one trading day's worth of signals.
Reward = realized cost-adjusted R-multiple for accepted trades, 0 for rejected.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import pandas as pd

from .config import RLFilterConfig
from .features import build_signal_observation, N_FEATURES

logger = logging.getLogger(__name__)

_root = str(Path(__file__).resolve().parents[3])
if _root not in sys.path:
    sys.path.insert(0, _root)


class SignalFilterEnv(gym.Env):
    """
    RL environment for signal filtering.

    Episode = one trading day.
    Step = one signal event (agent decides accept/reject).
    Reward = R-multiple of the resulting trade (accept) or 0 (reject).
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: RLFilterConfig, mode: str = "train", seed: int = 42):
        super().__init__()

        self.cfg = cfg
        self.mode = mode
        self._seed = seed
        self._rng = np.random.RandomState(seed)

        self.observation_space = gym.spaces.Box(
            low=-1.5, high=1.5, shape=(N_FEATURES,), dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(2)  # 0=reject, 1=accept

        self._load_data()

        # Episode state
        self._day_idx = 0
        self._signal_idx = 0
        self._day_signals: Optional[pd.DataFrame] = None
        self._day_df: Optional[pd.DataFrame] = None

    def _load_data(self):
        """Load data, compute indicators, generate signals, group by day."""
        import types
        sys.modules.setdefault("research.backtesting", types.ModuleType("research.backtesting"))
        sys.modules["research.backtesting"].__path__ = [
            str(Path(_root) / "research" / "backtesting")
        ]

        from research.backtesting.dayx.data_loader import (
            fetch_bars_cached, filter_rth, add_session_markers, compute_opening_range,
        )
        from research.backtesting.dayx.indicators import compute_all
        from research.backtesting.dayx.signals import generate_signals

        dcfg = self.cfg.dayx_config

        # Set date range based on mode
        if self.mode == "train":
            start, end = self.cfg.train_start, self.cfg.train_end
        elif self.mode == "val":
            start, end = self.cfg.val_start, self.cfg.val_end
        else:
            start, end = self.cfg.test_start, self.cfg.test_end

        # Load full cached dataset and slice by date (avoids re-fetching)
        from dataclasses import replace
        data_cfg = replace(dcfg, start_date="2016-01-01", end_date="2025-12-31")

        df = fetch_bars_cached(data_cfg)
        df = df.loc[start:end]
        df = filter_rth(df, data_cfg)
        df = add_session_markers(df, data_cfg)
        df = compute_opening_range(df, data_cfg)
        df = compute_all(df, data_cfg)

        # Rolling volume SMA for relative volume feature
        df["volume_sma_20"] = df["volume"].rolling(20, min_periods=1).mean()

        df = generate_signals(df, data_cfg)

        # Group by trading day
        self._daily_data = {}
        self._daily_signals = {}

        for date, group in df.groupby("session_date"):
            signals = group[group["signal_any"]].copy()
            if len(group) >= 20:  # skip short days
                self._daily_data[date] = group
                self._daily_signals[date] = signals

        self._dates = sorted(self._daily_data.keys())

        total_signals = sum(len(s) for s in self._daily_signals.values())
        days_with_signals = sum(1 for s in self._daily_signals.values() if len(s) > 0)
        logger.info(
            f"SignalFilterEnv [{self.mode}]: {len(self._dates)} days, "
            f"{total_signals} total signals, "
            f"{days_with_signals} days with signals, "
            f"{total_signals / max(len(self._dates), 1):.1f} signals/day avg"
        )

        if not self._dates:
            raise ValueError(f"No valid trading days for {self.mode} ({start} → {end})")

    def reset(self, seed=None, options=None):
        """Start a new episode (trading day)."""
        super().reset(seed=seed)

        if self.mode == "train":
            self._day_idx = self._rng.randint(0, len(self._dates))
        else:
            self._day_idx = (self._day_idx + 1) % len(self._dates)

        date = self._dates[self._day_idx]
        self._day_df = self._daily_data[date]
        self._day_signals = self._daily_signals[date]
        self._signal_idx = 0

        obs = self._get_obs()
        info = {
            "date": str(date),
            "n_signals": len(self._day_signals),
        }
        return obs, info

    def step(self, action: int):
        """
        Process one signal decision with counterfactual reward.

        Always simulates the trade to get the R-multiple.
        action=1 (accept): reward = +R-multiple
        action=0 (reject): reward = -R-multiple (counterfactual penalty/reward)

        This makes rejecting a winner costly and rejecting a loser rewarding,
        so the agent only rejects signals it believes will lose.
        """
        reward = 0.0
        info = {}

        if self._signal_idx >= len(self._day_signals):
            obs = np.zeros(N_FEATURES, dtype=np.float32)
            return obs, 0.0, True, False, info

        signal_row = self._day_signals.iloc[self._signal_idx]
        signal_name = signal_row["signal_name"]
        signal_dir = signal_row["signal_direction"]
        signal_ts = self._day_signals.index[self._signal_idx]

        info["signal_name"] = signal_name
        info["signal_direction"] = signal_dir
        info["signal_time"] = str(signal_ts)

        # Always simulate the trade (for both accept and reject)
        r_mult, trade_info = self._simulate_trade(signal_ts, signal_row)
        info.update(trade_info)
        info["counterfactual_r"] = round(r_mult, 4)

        # Clip R-multiple to remove tail fear
        r_mult = np.clip(r_mult, -self.cfg.r_clip, self.cfg.r_clip)

        if action == 1:
            reward = r_mult
            info["action"] = "accept"
        else:
            # Asymmetric counterfactual + fixed reject penalty
            reward = -r_mult * self.cfg.reject_reward_scale - self.cfg.reject_penalty
            info["action"] = "reject"

        # Advance to next signal
        self._signal_idx += 1
        terminated = self._signal_idx >= len(self._day_signals)

        obs = self._get_obs() if not terminated else np.zeros(N_FEATURES, dtype=np.float32)
        return obs, reward, terminated, False, info

    def _simulate_trade(self, entry_ts, signal_row) -> tuple[float, dict]:
        """
        Simulate a trade from signal entry to exit using PositionManager logic.

        Returns (r_multiple, info_dict).
        """
        from research.backtesting.dayx.strategy import PositionManager

        dcfg = self.cfg.dayx_config
        pm = PositionManager(dcfg)

        # Enter the trade
        atr_val = signal_row.get("atr", 0)
        if atr_val <= 0:
            return 0.0, {"trade_skip": "no_atr"}

        entered = pm.try_enter(
            timestamp=entry_ts,
            price=signal_row["close"],
            direction=signal_row["signal_direction"],
            strategy=signal_row["signal_name"],
            atr_val=atr_val,
            vwap_price=signal_row.get("vwap", 0.0),
            bb_upper=signal_row.get("bb_upper", 0.0),
            bb_lower=signal_row.get("bb_lower", 0.0),
        )

        if not entered:
            return 0.0, {"trade_skip": "entry_failed"}

        entry_price = signal_row["close"]
        risk = abs(entry_price - pm.position.trade.stop_price)
        if risk <= 0:
            return 0.0, {"trade_skip": "zero_risk"}

        # Walk forward through remaining bars of the day
        flatten_h, flatten_m = map(int, dcfg.eod_flatten_time.split(":"))
        day_df = self._day_df

        # Find bars after entry
        entry_loc = day_df.index.get_loc(entry_ts)
        remaining_bars = day_df.iloc[entry_loc + 1:]

        for ts, row in remaining_bars.iterrows():
            is_eod = (
                ts.hour > flatten_h or
                (ts.hour == flatten_h and ts.minute >= flatten_m)
            )
            closed = pm.check_exits(
                timestamp=ts,
                high=row["high"],
                low=row["low"],
                close=row["close"],
                is_eod=is_eod,
            )
            if closed:
                break

        # If still open at end of data, force close
        if pm.has_position:
            last_ts = day_df.index[-1]
            last_row = day_df.iloc[-1]
            pm._close_position(last_ts, last_row["close"], "backtest_end")

        trade = pm.closed_trades[0]
        # Cost-adjusted R-multiple
        rt_costs = 2 * (dcfg.commission_per_share + dcfg.slippage_per_share) * trade.size
        r_mult = (trade.pnl - rt_costs) / (risk * trade.size) if risk > 0 else 0.0

        trade_info = {
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "exit_reason": trade.exit_reason,
            "r_multiple": round(r_mult, 4),
            "pnl": round(trade.pnl, 2),
            "strategy": trade.strategy,
        }
        return r_mult, trade_info

    def _get_obs(self) -> np.ndarray:
        """Build observation for current signal."""
        if self._signal_idx >= len(self._day_signals):
            return np.zeros(N_FEATURES, dtype=np.float32)

        row = self._day_signals.iloc[self._signal_idx]
        signal_name = row["signal_name"]
        volume_sma = row.get("volume_sma_20", 1.0)

        return build_signal_observation(row, signal_name, volume_sma)
