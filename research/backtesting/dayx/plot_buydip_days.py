"""Plot best/worst buy_dip days with candlesticks, BB bands, RSI, and entry/exit markers."""

import sys
import types
import logging
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.WARNING)

_root = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, _root)
sys.modules.setdefault("research.backtesting", types.ModuleType("research.backtesting"))
sys.modules["research.backtesting"].__path__ = [
    str(Path(_root) / "research" / "backtesting")
]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from research.backtesting.dayx.config import DayXConfig
from research.backtesting.dayx.data_loader import (
    fetch_bars_cached, filter_rth, add_session_markers, compute_opening_range,
)
from research.backtesting.dayx.indicators import compute_all
from research.backtesting.dayx.signals import generate_signals
from research.backtesting.dayx.simulator import run_simulation

_out_dir = Path(_root) / "data" / "backtests" / "dayx"

FIXED = dict(
    strategies=["buy_dip"],
    use_vwap_target=False,
    trail_mode="progressive",
    time_filter=True,
    require_exhaustion=False,
    require_cci20_alignment=False,
)

TRIAL396 = dict(
    rsi_period=10, bb_lookback_bars=9, bb_period=30, bb_std=2.762,
    cci_fast=18, cci_neutral_hi=95, cci_neutral_lo=-77,
    entry_earliest="09:44", entry_latest="13:49",
    partial_exit_pct=0.350, stop_atr_mult=2.187,
    target1_r=1.972, target2_r=3.077,
    trail_progressive_step=0.316,
    trend_dip_above_pct=0.707, trend_dip_vwap_pct=0.484,
    atr_period=20,
)


def plot_day(day_df, day_trades, date_str, total_pnl, out_dir):
    """Plot a single day's chart with BB bands, RSI, entries/exits."""
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(18, 12), height_ratios=[4, 1, 1.2],
        gridspec_kw={"hspace": 0.08},
    )

    times = day_df.index
    opens = day_df["open"].values
    highs = day_df["high"].values
    lows = day_df["low"].values
    closes = day_df["close"].values

    colors = ["#26a69a" if c >= o else "#ef5350" for o, c in zip(opens, closes)]

    # Candle wicks
    for i, t in enumerate(times):
        ax1.plot([t, t], [lows[i], highs[i]], color=colors[i], linewidth=0.8)

    # Candle bodies
    bar_width = pd.Timedelta(seconds=40)
    for i, t in enumerate(times):
        body_low = min(opens[i], closes[i])
        body_high = max(opens[i], closes[i])
        body_h = max(body_high - body_low, 0.01)
        rect = Rectangle(
            (t - bar_width / 2, body_low), bar_width, body_h,
            facecolor=colors[i], edgecolor=colors[i], linewidth=0.5,
        )
        ax1.add_patch(rect)

    # Bollinger Bands
    if "bb_upper" in day_df.columns:
        ax1.plot(times, day_df["bb_upper"], color="#FF9800", linewidth=0.8, alpha=0.6, label="BB Upper")
        ax1.plot(times, day_df["bb_lower"], color="#FF9800", linewidth=0.8, alpha=0.6, label="BB Lower")
        ax1.fill_between(times, day_df["bb_lower"], day_df["bb_upper"],
                         color="#FF9800", alpha=0.05)
        ax1.plot(times, day_df["bb_mid"], color="#FF9800", linewidth=0.5, alpha=0.3, linestyle="--")

    # VWAP
    if "vwap" in day_df.columns:
        ax1.plot(times, day_df["vwap"], color="#2196F3", linewidth=1.2, alpha=0.7, label="VWAP")

    # Signal markers (all buy_dip signals that fired)
    signal_bars = day_df[day_df["signal_any"].fillna(False).astype(bool)]
    for idx in signal_bars.index:
        ax1.axvline(x=idx, color="#9C27B0", alpha=0.15, linewidth=1)

    # Trades: entry/exit markers
    tc_map = {"W": "#4CAF50", "L": "#F44336"}
    for t in day_trades:
        win = "W" if t.pnl > 0 else "L"
        tc = tc_map[win]

        # Entry triangle
        ax1.scatter(t.entry_time, t.entry_price, marker="^", color=tc,
                    s=150, zorder=5, edgecolors="black", linewidth=0.8)
        # Exit square
        ax1.scatter(t.exit_time, t.exit_price, marker="s", color=tc,
                    s=100, zorder=5, edgecolors="black", linewidth=0.8)
        # Connecting line
        ax1.plot([t.entry_time, t.exit_time], [t.entry_price, t.exit_price],
                 color=tc, linewidth=1.5, alpha=0.6, linestyle="--")
        # Stop level
        ax1.axhline(y=t.stop_price, xmin=0, xmax=1, color="#F44336",
                     alpha=0.2, linewidth=0.5, linestyle=":")

        pnl_str = f"${t.pnl:+,.0f}\n{t.exit_reason}"
        ax1.annotate(
            pnl_str, xy=(t.exit_time, t.exit_price),
            xytext=(10, 15 if t.pnl > 0 else -25),
            textcoords="offset points", fontsize=8, fontweight="bold", color=tc,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=tc, alpha=0.9),
        )

    # Volume bars
    ax2.bar(times, day_df["volume"], width=bar_width,
            color=colors, alpha=0.6)
    ax2.set_ylabel("Vol", fontsize=9)
    ax2.grid(True, alpha=0.2)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    # RSI subplot
    if "rsi" in day_df.columns:
        ax3.plot(times, day_df["rsi"], color="#9C27B0", linewidth=1.2)
        ax3.axhline(y=30, color="#4CAF50", linewidth=0.8, linestyle="--", alpha=0.6)
        ax3.axhline(y=70, color="#F44336", linewidth=0.8, linestyle="--", alpha=0.6)
        ax3.fill_between(times, 30, day_df["rsi"],
                         where=day_df["rsi"] < 30, color="#4CAF50", alpha=0.15)
        ax3.fill_between(times, 70, day_df["rsi"],
                         where=day_df["rsi"] > 70, color="#F44336", alpha=0.15)
        ax3.set_ylabel("RSI(7)", fontsize=9)
        ax3.set_ylim(10, 90)
        ax3.grid(True, alpha=0.2)

        # Mark RSI cross above 30
        rsi_cross = day_df["rsi_cross_above_30"].fillna(False)
        cross_bars = day_df[rsi_cross]
        for idx in cross_bars.index:
            ax3.scatter(idx, 30, marker="^", color="#4CAF50", s=60, zorder=5)

    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax3.set_xlabel("Time (ET)", fontsize=10)

    # Title
    tag = "WINNER" if total_pnl > 0 else "LOSER"
    tag_color = "#4CAF50" if total_pnl > 0 else "#F44336"
    n_trades = len(day_trades)
    wins = sum(1 for t in day_trades if t.pnl > 0)
    losses = n_trades - wins
    n_signals = len(signal_bars)

    ax1.set_title(
        f"QQQ {date_str}  —  [{tag}] Day PnL: ${total_pnl:+,.2f}  |  "
        f"Signals: {n_signals}  |  Trades: {wins}W/{losses}L",
        fontsize=13, fontweight="bold", color=tag_color,
    )
    ax1.set_ylabel("Price", fontsize=10)
    ax1.grid(True, alpha=0.2)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    legend_elements = [
        Line2D([0], [0], color="#FF9800", linewidth=1, label="BB Bands"),
        Line2D([0], [0], color="#2196F3", linewidth=1, label="VWAP"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#4CAF50",
               markersize=10, label="Entry (win)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#F44336",
               markersize=10, label="Entry (loss)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#4CAF50",
               markersize=8, label="Exit (win)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#F44336",
               markersize=8, label="Exit (loss)"),
    ]
    ax1.legend(handles=legend_elements, loc="upper left", fontsize=8, ncol=3)

    plt.tight_layout()
    fname = f"buydip_{date_str}.png"
    plt.savefig(str(out_dir / fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}  (PnL: ${total_pnl:+,.0f}, {wins}W/{losses}L)")


def main():
    merged = {**FIXED, **TRIAL396}
    cfg = DayXConfig(
        symbol="QQQ", timeframe="1Min",
        start_date="2024-01-01", end_date="2026-03-15",
        **merged,
    )

    print("Loading data...")
    df = fetch_bars_cached(cfg)
    df = filter_rth(df, cfg)
    df = add_session_markers(df, cfg)
    df = compute_opening_range(df, cfg)
    df = compute_all(df, cfg)
    df = generate_signals(df, cfg)

    print("Running simulation...")
    trades, eq = run_simulation(df, cfg)

    # Group trades by date
    trades_by_date = defaultdict(list)
    for t in trades:
        trades_by_date[t.entry_time.strftime("%Y-%m-%d")].append(t)

    day_pnl = {d: sum(t.pnl for t in ts) for d, ts in trades_by_date.items()}

    best_days = sorted(day_pnl.items(), key=lambda x: -x[1])[:3]
    worst_days = sorted(day_pnl.items(), key=lambda x: x[1])[:3]
    target_days = best_days + worst_days

    print(f"\nPlotting {len(target_days)} days (3 best, 3 worst)...")
    _out_dir.mkdir(parents=True, exist_ok=True)

    for date_str, total_pnl in target_days:
        day_df = df[df.index.strftime("%Y-%m-%d") == date_str].copy()
        if len(day_df) == 0:
            continue
        day_trades = trades_by_date.get(date_str, [])
        plot_day(day_df, day_trades, date_str, total_pnl, _out_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
