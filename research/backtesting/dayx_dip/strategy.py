"""
Position management for DayX Dip: long-only, exit at BB upper cross or stop loss.
No partial exits, no fixed R targets.
"""

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

from .config import DayXDipConfig


@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    stop_price: float
    size: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""
    pnl: float = 0.0
    max_adverse: float = 0.0
    max_favorable: float = 0.0


class PositionManager:
    """Long-only position manager with BB upper cross exit."""

    def __init__(self, cfg: DayXDipConfig):
        self.cfg = cfg
        self.position: Optional[Trade] = None
        self.closed_trades: list[Trade] = []
        self.equity = cfg.initial_capital
        self.equity_curve: list[tuple[pd.Timestamp, float]] = []

    @property
    def has_position(self) -> bool:
        return self.position is not None

    def try_enter(self, timestamp: pd.Timestamp, price: float, atr_val: float) -> bool:
        """Attempt to open a long position. Returns True if entered."""
        if self.has_position:
            return False

        risk = atr_val * self.cfg.stop_atr_mult
        stop = price - risk
        if risk <= 0:
            return False

        size = int(self.equity * self.cfg.position_size_pct / price)
        if size <= 0:
            return False

        self.position = Trade(
            entry_time=timestamp,
            entry_price=price,
            stop_price=stop,
            size=size,
        )
        return True

    def check_exits(self, timestamp: pd.Timestamp, high: float, low: float,
                    close: float, bb_upper: float, is_eod: bool) -> bool:
        """
        Check exits in priority order:
          1. EOD flatten
          2. BB upper cross (close >= bb_upper) — profit take
          3. Stop loss (low <= stop_price)
        Returns True if position was closed.
        """
        if not self.has_position:
            return False

        trade = self.position

        # Track MAE/MFE (long only)
        trade.max_adverse = max(trade.max_adverse, trade.entry_price - low)
        trade.max_favorable = max(trade.max_favorable, high - trade.entry_price)

        # 1. EOD flatten
        if is_eod:
            self._close(timestamp, close, "eod_flatten")
            return True

        # 2. BB upper cross
        if close >= bb_upper:
            self._close(timestamp, close, "bb_upper_cross")
            return True

        # 3. Stop loss
        if low <= trade.stop_price:
            self._close(timestamp, trade.stop_price, "stop_loss")
            return True

        return False

    def _close(self, timestamp: pd.Timestamp, price: float, reason: str):
        trade = self.position
        pnl = (price - trade.entry_price) * trade.size
        pnl -= trade.size * (self.cfg.commission_per_share + self.cfg.slippage_per_share)
        trade.pnl = pnl
        trade.exit_time = timestamp
        trade.exit_price = price
        trade.exit_reason = reason
        self.equity += pnl
        self.closed_trades.append(trade)
        self.position = None

    def record_equity(self, timestamp: pd.Timestamp):
        self.equity_curve.append((timestamp, self.equity))
