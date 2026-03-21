"""
CLI entry point for DayX Dip backtest.

Usage:
    python research/backtesting/dayx_dip/run_backtest.py --symbol QQQ --timeframe 1Min --start 2024-01-01 --end 2025-01-01
    python research/backtesting/dayx_dip/run_backtest.py --symbol QQQ --timeframe 3Min --start 2024-01-01 --end 2025-01-01 --save-trades
"""

import argparse
import logging
import sys
from pathlib import Path

if __name__ == "__main__":
    _root = str(Path(__file__).resolve().parents[3])
    sys.path.insert(0, _root)
    import types
    sys.modules.setdefault("research.backtesting", types.ModuleType("research.backtesting"))
    sys.modules["research.backtesting"].__path__ = [
        str(Path(_root) / "research" / "backtesting")
    ]

from research.backtesting.dayx_dip.config import DayXDipConfig
from research.backtesting.dayx_dip.data_loader import load_data
from research.backtesting.dayx_dip.indicators import compute_all
from research.backtesting.dayx_dip.signals import bear_exhaustion_dip
from research.backtesting.dayx_dip.simulator import run_simulation
from research.backtesting.dayx_dip.metrics import compute_metrics, hourly_pnl, weekday_pnl, print_summary

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="DayX Dip Intraday Strategy Backtest")
    parser.add_argument("--symbol", default="QQQ")
    parser.add_argument("--timeframe", default="1Min")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2025-01-01")
    parser.add_argument("--capital", type=float, default=100_000)
    parser.add_argument("--rsi-oversold", type=float, default=35.0)
    parser.add_argument("--cci-oversold", type=float, default=-100.0)
    parser.add_argument("--exhaustion-lookback", type=int, default=3)
    parser.add_argument("--stop-atr-mult", type=float, default=2.0)
    parser.add_argument("--save-trades", action="store_true")
    parser.add_argument("--plot", action="store_true", help="Plot equity curve")
    args = parser.parse_args()

    cfg = DayXDipConfig(
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        rsi_oversold=args.rsi_oversold,
        cci_oversold=args.cci_oversold,
        exhaustion_lookback=args.exhaustion_lookback,
        stop_atr_mult=args.stop_atr_mult,
    )

    logger.info(f"DayX Dip Backtest: {cfg.symbol} {cfg.start_date} → {cfg.end_date} ({cfg.timeframe})")

    # 1. Load data
    logger.info("Loading data...")
    df = load_data(cfg)
    logger.info(f"Loaded {len(df)} bars ({df.index[0]} → {df.index[-1]})")

    # 2. Compute indicators
    logger.info("Computing indicators...")
    df = compute_all(df, cfg)

    # 3. Generate signals
    logger.info("Generating signals...")
    df = bear_exhaustion_dip(df, cfg)
    signal_count = df["signal_bear_exhaustion_dip"].sum()
    logger.info(f"Signals: {signal_count} bear_exhaustion_dip entries")

    # 4. Run simulation
    logger.info("Running simulation...")
    trades, equity_curve = run_simulation(df, cfg)

    # 5. Verify: all trades long-only with correct exit reasons
    assert all(t.exit_reason in ("bb_upper_cross", "stop_loss", "eod_flatten", "backtest_end")
               for t in trades), "Unexpected exit reason found!"

    # 6. Metrics
    overall = compute_metrics(trades, equity_curve, cfg.initial_capital)
    by_hour = hourly_pnl(trades)
    by_day = weekday_pnl(trades)
    print_summary(overall, by_hour, by_day)

    # 7. Optionally save trade log
    if args.save_trades and trades:
        import pandas as pd
        out_dir = Path(__file__).resolve().parents[3] / "data" / "backtests" / "dayx_dip"
        out_dir.mkdir(parents=True, exist_ok=True)
        trade_file = out_dir / f"trades_{cfg.symbol}_{cfg.timeframe}_{cfg.start_date}_{cfg.end_date}.csv"
        records = []
        for t in trades:
            records.append({
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "stop_price": t.stop_price,
                "size": t.size,
                "pnl": t.pnl,
                "exit_reason": t.exit_reason,
                "hold_min": (t.exit_time - t.entry_time).total_seconds() / 60 if t.exit_time else 0,
                "mae": t.max_adverse,
                "mfe": t.max_favorable,
            })
        pd.DataFrame(records).to_csv(trade_file, index=False)
        logger.info(f"Trade log saved to {trade_file}")

    # 8. Plot equity curve
    if args.plot:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import pandas as pd

        eq_df = pd.DataFrame(equity_curve, columns=["timestamp", "equity"])
        eq_df.set_index("timestamp", inplace=True)

        # Buy-and-hold benchmark
        first_close = df["close"].iloc[0]
        last_close = df["close"].iloc[-1]
        bh_return = (last_close / first_close - 1) * cfg.initial_capital
        bh_curve = (df["close"] / first_close) * cfg.initial_capital

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                        gridspec_kw={"height_ratios": [3, 1]})
        fig.suptitle(
            f"DayX Dip — {cfg.symbol} {cfg.timeframe}  "
            f"({cfg.start_date} → {cfg.end_date})",
            fontsize=13, fontweight="bold",
        )

        # --- Equity curve ---
        ax1.plot(eq_df.index, eq_df["equity"], color="#2196F3", linewidth=1.5,
                 label=f"Strategy (${overall['total_pnl']:+,.0f})")
        ax1.plot(bh_curve.index, bh_curve.values, color="#9E9E9E", linewidth=1,
                 linestyle="--", label=f"Buy & Hold (${bh_return:+,.0f})", alpha=0.7)

        # Mark trade entries and exits
        for t in trades:
            color = "#4CAF50" if t.pnl > 0 else "#F44336"
            ax1.axvline(t.entry_time, color=color, alpha=0.15, linewidth=0.5)

        ax1.axhline(cfg.initial_capital, color="#666", linewidth=0.8, linestyle=":")
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.legend(loc="upper left", fontsize=9)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax1.grid(True, alpha=0.3)

        # --- Drawdown ---
        running_max = eq_df["equity"].cummax()
        drawdown = (eq_df["equity"] - running_max) / running_max * 100
        ax2.fill_between(drawdown.index, drawdown.values, 0, color="#F44336", alpha=0.4)
        ax2.plot(drawdown.index, drawdown.values, color="#F44336", linewidth=0.8)
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_xlabel("Date")
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
        ax2.grid(True, alpha=0.3)

        # Stats box
        stats = (
            f"Trades: {overall['trade_count']}  |  "
            f"Win Rate: {overall['win_rate']}%  |  "
            f"Sharpe: {overall['sharpe']}  |  "
            f"Max DD: {overall['max_drawdown_pct']}%  |  "
            f"PF: {overall['profit_factor']}"
        )
        fig.text(0.5, 0.01, stats, ha="center", fontsize=9, color="#444")

        plt.tight_layout(rect=[0, 0.03, 1, 1])

        out_dir = Path(__file__).resolve().parents[3] / "data" / "backtests" / "dayx_dip"
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_file = out_dir / f"equity_{cfg.symbol}_{cfg.timeframe}_{cfg.start_date}_{cfg.end_date}.png"
        plt.savefig(plot_file, dpi=150, bbox_inches="tight")
        logger.info(f"Equity curve saved to {plot_file}")
        plt.show()


if __name__ == "__main__":
    main()
