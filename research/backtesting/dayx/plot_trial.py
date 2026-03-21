"""Plot equity curve for a specific Optuna trial."""

import sys
import types
from pathlib import Path

_root = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, _root)
sys.modules.setdefault("research.backtesting", types.ModuleType("research.backtesting"))
sys.modules["research.backtesting"].__path__ = [str(Path(_root) / "research" / "backtesting")]

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from research.backtesting.dayx.config import DayXConfig
from research.backtesting.dayx.data_loader import (
    fetch_bars_cached, filter_rth, add_session_markers, compute_opening_range,
)
from research.backtesting.dayx.indicators import compute_all
from research.backtesting.dayx.signals import generate_signals
from research.backtesting.dayx.simulator import run_simulation
from research.backtesting.dayx.metrics import (
    compute_metrics, per_strategy_breakdown, print_summary, hourly_pnl, weekday_pnl,
)

FIXED = dict(
    strategies=["buy_dip", "sell_rip", "trend_dip"],
    use_vwap_target=False,
    trail_mode="progressive",
    time_filter=True,
    require_exhaustion=False,
    require_cci20_alignment=False,
)

_out_dir = Path(_root) / "data" / "backtests" / "dayx"


def run_trial(params: dict, start: str = "2016-01-01", end: str = "2025-12-31",
              label: str = "trial"):
    merged = {**FIXED, **params}  # params override FIXED
    cfg = DayXConfig(
        symbol="QQQ", timeframe="1Min",
        start_date=start, end_date=end,
        **merged,
    )

    df = fetch_bars_cached(cfg)
    df = filter_rth(df, cfg)
    df = add_session_markers(df, cfg)
    df = compute_opening_range(df, cfg)
    df = compute_all(df, cfg)
    df = generate_signals(df, cfg)

    signal_counts = df["signal_name"].value_counts()
    print(f"Signal counts:\n{signal_counts.to_string()}\n")

    trades, eq = run_simulation(df, cfg)
    overall = compute_metrics(trades, eq, cfg.initial_capital)
    per_strat = per_strategy_breakdown(trades, eq, cfg.initial_capital)
    print_summary(overall, per_strat, hourly_pnl(trades), weekday_pnl(trades))

    # Per-year breakdown
    print("\n  Per-year P&L:")
    for year in range(int(start[:4]), int(end[:4]) + 1):
        yt = [t for t in trades if t.entry_time.year == year]
        if yt:
            pnl = sum(t.pnl for t in yt)
            wr = sum(1 for t in yt if t.pnl > 0) / len(yt) * 100
            print(f"    {year}: ${pnl:>9,.0f}  ({len(yt)} trades, {wr:.0f}% WR)")

    # Equity curve
    eq_df = pd.DataFrame(eq, columns=["timestamp", "equity"]).set_index("timestamp")
    daily = eq_df["equity"].resample("D").last().dropna()

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(daily.index, daily.values, linewidth=1.5, color="#2196F3")
    ax.axhline(y=100000, color="gray", linestyle="--", alpha=0.5)

    sharpe = overall.get("sharpe", 0)
    pf = overall.get("profit_factor", 0)
    pnl = overall.get("total_pnl", 0)
    tc = overall.get("trade_count", 0)
    wr = overall.get("win_rate", 0)
    ax.set_title(
        f"{label} | Sharpe={sharpe:.2f}, PF={pf:.2f}, "
        f"PnL=${pnl:,.0f}, Trades={tc}, WR={wr:.0f}%",
        fontsize=12, fontweight="bold",
    )
    ax.set_ylabel("Equity ($)")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    plt.tight_layout()

    out_path = _out_dir / f"{label.lower().replace(' ', '_')}_equity.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"\nEquity curve saved to {out_path}")
    return trades, eq, overall


TRIAL14_PARAMS = dict(
    rsi_period=7,
    bb_lookback_bars=1,
    bb_period=30,
    bb_std=2.667,
    cci_fast=8,
    cci_neutral_hi=100,
    cci_neutral_lo=-99,
    entry_earliest="09:58",
    entry_latest="13:58",
    partial_exit_pct=0.481,
    stop_atr_mult=2.905,
    target1_r=1.987,
    target2_r=3.985,
    trail_progressive_step=0.420,
    trend_dip_above_pct=0.484,
    trend_dip_vwap_pct=0.103,
    atr_period=14,
)

def compare_trials(trials: dict, start="2016-01-01", end="2025-12-31"):
    """Run multiple trials and overlay equity curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    results = {}
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]

    for i, (label, params) in enumerate(trials.items()):
        print(f"\n{'='*60}")
        print(f"Running: {label}")
        print(f"{'='*60}")
        trades, eq, overall = run_trial(params, start=start, end=end, label=label)
        results[label] = (trades, eq, overall, colors[i % len(colors)])

    # Overlay plot
    fig, ax = plt.subplots(figsize=(16, 7))
    for label, (trades, eq, overall, color) in results.items():
        eq_df = pd.DataFrame(eq, columns=["timestamp", "equity"]).set_index("timestamp")
        daily = eq_df["equity"].resample("D").last().dropna()
        s = overall.get("sharpe", 0)
        pf = overall.get("profit_factor", 0)
        pnl = overall.get("total_pnl", 0)
        tc = overall.get("trade_count", 0)
        ax.plot(daily.index, daily.values, linewidth=1.5, color=color,
                label=f"{label} (S={s:.2f} PF={pf:.2f} ${pnl:,.0f} n={tc})")

    ax.axhline(y=100000, color="gray", linestyle="--", alpha=0.4)
    ax.set_title("Trial Comparison — Equity Curves", fontsize=13, fontweight="bold")
    ax.set_ylabel("Equity ($)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    plt.tight_layout()

    out = _out_dir / "trial_comparison.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    print(f"\nComparison saved to {out}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--strategies", nargs="+", default=None)
    parser.add_argument("--label", default="Trial 14")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    if args.compare:
        TRIAL94 = dict(
            rsi_period=10, bb_lookback_bars=6, bb_period=27, bb_std=2.907,
            cci_fast=15, cci_neutral_hi=95, cci_neutral_lo=-94,
            entry_earliest="09:51", entry_latest="13:41",
            partial_exit_pct=0.318, stop_atr_mult=2.803,
            target1_r=1.921, target2_r=3.888,
            trail_progressive_step=0.486,
            trend_dip_above_pct=0.419, trend_dip_vwap_pct=0.427,
            atr_period=19,
        )
        compare_trials({
            "Trial 14 (RSI7, LB=1)": TRIAL14_PARAMS,
            "Trial 94 (RSI10, LB=6)": TRIAL94,
        }, start=args.start, end=args.end)
    else:
        params = dict(TRIAL14_PARAMS)
        if args.strategies:
            params["strategies"] = args.strategies
        run_trial(params, start=args.start, end=args.end, label=args.label)
