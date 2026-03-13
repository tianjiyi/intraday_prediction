"""
Position management: entries, exits, partial scaling, EOD flatten.
"""

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

from .config import DayXConfig


@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    direction: str              # 'long' or 'short'
    strategy: str               # strategy name
    stop_price: float
    target1_price: float
    target2_price: float
    size: float                 # number of shares (full position)
    remaining_size: float = 0.0 # shares still open
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""
    pnl: float = 0.0
    partial_exits: list = field(default_factory=list)
    max_adverse: float = 0.0    # max adverse excursion (MAE)
    max_favorable: float = 0.0  # max favorable excursion (MFE)

    def __post_init__(self):
        if self.remaining_size == 0.0:
            self.remaining_size = self.size


@dataclass
class Position:
    """Active position state."""
    trade: Trade
    target1_hit: bool = False
    best_r: float = 0.0  # Best R-multiple reached (for progressive trailing)


class PositionManager:
    """Manages entries, exits, and position tracking."""

    def __init__(self, cfg: DayXConfig):
        self.cfg = cfg
        self.position: Optional[Position] = None
        self.closed_trades: list[Trade] = []
        self.equity = cfg.initial_capital
        self.equity_curve: list[tuple[pd.Timestamp, float]] = []

    @property
    def has_position(self) -> bool:
        return self.position is not None

    def try_enter(self, timestamp: pd.Timestamp, price: float,
                  direction: str, strategy: str, atr_val: float,
                  vwap_price: float = 0.0, bb_upper: float = 0.0,
                  bb_lower: float = 0.0) -> bool:
        """Attempt to open a new position. Returns True if entered."""
        if self.has_position:
            return False

        # Strategy-specific stop and target logic
        if self.cfg.use_vwap_target and strategy in ("buy_dip", "sell_rip"):
            # Mean reversion: wider stop, target VWAP
            stop_mult = self.cfg.mr_stop_atr_mult
            risk = atr_val * stop_mult
            if direction == "long":
                stop = price - risk
                t1 = vwap_price if vwap_price > price else price + risk * self.cfg.target1_r
                t2 = price + risk * self.cfg.target2_r
            else:
                stop = price + risk
                t1 = vwap_price if vwap_price < price else price - risk * self.cfg.target1_r
                t2 = price - risk * self.cfg.target2_r
        elif self.cfg.use_vwap_target and strategy == "trend_dip":
            # Trend dip: tighter stop, target BB_upper
            stop_mult = self.cfg.trend_dip_stop_atr_mult
            risk = atr_val * stop_mult
            stop = price - risk  # always long
            t1 = bb_upper if bb_upper > price else price + risk * self.cfg.target1_r
            t2 = price + risk * self.cfg.target2_r
        else:
            # Default: fixed R multiples
            risk = atr_val * self.cfg.stop_atr_mult
            if direction == "long":
                stop = price - risk
                t1 = price + risk * self.cfg.target1_r
                t2 = price + risk * self.cfg.target2_r
            else:
                stop = price + risk
                t1 = price - risk * self.cfg.target1_r
                t2 = price - risk * self.cfg.target2_r

        # Position sizing
        risk_per_share = abs(price - stop)
        if risk_per_share <= 0:
            return False
        capital_at_risk = self.equity * self.cfg.position_size_pct
        size = int(capital_at_risk / price)
        if size <= 0:
            return False

        trade = Trade(
            entry_time=timestamp,
            entry_price=price,
            direction=direction,
            strategy=strategy,
            stop_price=stop,
            target1_price=t1,
            target2_price=t2,
            size=size,
        )
        self.position = Position(trade=trade)
        return True

    def check_exits(self, timestamp: pd.Timestamp, high: float, low: float,
                    close: float, is_eod: bool) -> bool:
        """
        Check exit conditions on current bar. Returns True if fully closed.
        Order: EOD flatten → stop-loss → target2 → target1
        """
        if not self.has_position:
            return False

        pos = self.position
        trade = pos.trade

        # Track MAE/MFE
        if trade.direction == "long":
            adverse = trade.entry_price - low
            favorable = high - trade.entry_price
        else:
            adverse = high - trade.entry_price
            favorable = trade.entry_price - low
        trade.max_adverse = max(trade.max_adverse, adverse)
        trade.max_favorable = max(trade.max_favorable, favorable)

        # --- EOD flatten ---
        if is_eod:
            self._close_position(timestamp, close, "eod_flatten")
            return True

        # --- Stop-loss ---
        if trade.direction == "long" and low <= trade.stop_price:
            self._close_position(timestamp, trade.stop_price, "stop_loss")
            return True
        if trade.direction == "short" and high >= trade.stop_price:
            self._close_position(timestamp, trade.stop_price, "stop_loss")
            return True

        # --- Target 2 (full exit) ---
        if trade.direction == "long" and high >= trade.target2_price:
            self._close_position(timestamp, trade.target2_price, "target2")
            return True
        if trade.direction == "short" and low <= trade.target2_price:
            self._close_position(timestamp, trade.target2_price, "target2")
            return True

        # --- Target 1 (partial exit) ---
        if not pos.target1_hit:
            if trade.direction == "long" and high >= trade.target1_price:
                self._partial_exit(timestamp, trade.target1_price, "target1")
                pos.target1_hit = True
                self._update_trail(pos, trade)
            elif trade.direction == "short" and low <= trade.target1_price:
                self._partial_exit(timestamp, trade.target1_price, "target1")
                pos.target1_hit = True
                self._update_trail(pos, trade)

        # --- Progressive trailing: ratchet stop as price moves favorably ---
        if pos.target1_hit and self.cfg.trail_mode == "progressive":
            risk = abs(trade.target1_price - trade.entry_price) / self.cfg.target1_r
            if risk > 0:
                if trade.direction == "long":
                    current_r = (high - trade.entry_price) / risk
                else:
                    current_r = (trade.entry_price - low) / risk
                if current_r > pos.best_r:
                    pos.best_r = current_r
                    step = self.cfg.trail_progressive_step
                    trail_r = (int(pos.best_r / step) - 1) * step
                    if trail_r > 0:
                        if trade.direction == "long":
                            new_stop = trade.entry_price + trail_r * risk
                        else:
                            new_stop = trade.entry_price - trail_r * risk
                        # Only ratchet up, never down
                        if trade.direction == "long":
                            trade.stop_price = max(trade.stop_price, new_stop)
                        else:
                            trade.stop_price = min(trade.stop_price, new_stop)

        return False

    def _update_trail(self, pos: Position, trade: Trade):
        """Set trail stop after target1 hit based on trail_mode."""
        if not self.cfg.trail_after_target1:
            return
        if self.cfg.trail_mode == "breakeven":
            trade.stop_price = trade.entry_price
        elif self.cfg.trail_mode == "progressive":
            # Start at breakeven, progressive logic in check_exits will ratchet up
            trade.stop_price = trade.entry_price
            pos.best_r = self.cfg.target1_r

    def _partial_exit(self, timestamp: pd.Timestamp, price: float, reason: str):
        """Exit a portion of the position."""
        trade = self.position.trade
        exit_shares = int(trade.remaining_size * self.cfg.partial_exit_pct)
        if exit_shares <= 0:
            exit_shares = 1

        if trade.direction == "long":
            pnl = (price - trade.entry_price) * exit_shares
        else:
            pnl = (trade.entry_price - price) * exit_shares

        pnl -= exit_shares * (self.cfg.commission_per_share + self.cfg.slippage_per_share)
        trade.partial_exits.append({
            "time": timestamp, "price": price, "shares": exit_shares,
            "reason": reason, "pnl": pnl,
        })
        trade.remaining_size -= exit_shares
        trade.pnl += pnl
        self.equity += pnl

    def _close_position(self, timestamp: pd.Timestamp, price: float, reason: str):
        """Fully close the remaining position."""
        trade = self.position.trade
        remaining = trade.remaining_size

        if trade.direction == "long":
            pnl = (price - trade.entry_price) * remaining
        else:
            pnl = (trade.entry_price - price) * remaining

        pnl -= remaining * (self.cfg.commission_per_share + self.cfg.slippage_per_share)
        trade.pnl += pnl
        trade.remaining_size = 0
        trade.exit_time = timestamp
        trade.exit_price = price
        trade.exit_reason = reason
        self.equity += pnl

        self.closed_trades.append(trade)
        self.position = None

    def record_equity(self, timestamp: pd.Timestamp):
        """Snapshot equity for the curve."""
        self.equity_curve.append((timestamp, self.equity))
