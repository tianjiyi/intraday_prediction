"""
CLI entry point for Day_X backtest.

Usage:
    python -m research.backtesting.dayx.run_backtest --symbol QQQ --start 2025-01-01 --end 2025-12-31

    # Or from the project root:
    python research/backtesting/dayx/run_backtest.py --symbol QQQ
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow running as script from project root
if __name__ == "__main__":
    _root = str(Path(__file__).resolve().parents[3])
    sys.path.insert(0, _root)
    # Prevent parent backtesting __init__.py from importing broken Kronos deps
    import types
    sys.modules.setdefault("research.backtesting", types.ModuleType("research.backtesting"))
    sys.modules["research.backtesting"].__path__ = [
        str(Path(_root) / "research" / "backtesting")
    ]

from research.backtesting.dayx.config import DayXConfig
from research.backtesting.dayx.data_loader import load_data
from research.backtesting.dayx.indicators import compute_all
from research.backtesting.dayx.signals import generate_signals
from research.backtesting.dayx.simulator import run_simulation
from research.backtesting.dayx.metrics import (
    compute_metrics, per_strategy_breakdown,
    hourly_pnl, weekday_pnl, print_summary,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Day_X Intraday Strategy Backtest")
    parser.add_argument("--symbol", default="QQQ", help="Ticker symbol")
    parser.add_argument("--start", default="2025-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--timeframe", default="5Min", help="Bar timeframe")
    parser.add_argument("--capital", type=float, default=100_000, help="Initial capital")
    parser.add_argument("--no-exhaustion", action="store_true", help="Disable exhaustion filter")
    parser.add_argument("--no-cci20", action="store_true", help="Disable CCI20 alignment filter")
    parser.add_argument("--save-trades", action="store_true", help="Save trade log to CSV")
    parser.add_argument("--strategies", nargs="+", default=None,
                        help="Strategies to enable (e.g. long_trend short_trend long_dip)")
    args = parser.parse_args()

    # Build config
    cfg = DayXConfig(
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end,
        timeframe=args.timeframe,
        initial_capital=args.capital,
        require_exhaustion=not args.no_exhaustion,
        require_cci20_alignment=not args.no_cci20,
    )
    if args.strategies:
        cfg.strategies = args.strategies

    logger.info(f"Day_X Backtest: {cfg.symbol} {cfg.start_date} → {cfg.end_date} ({cfg.timeframe})")

    # 1. Load data
    logger.info("Loading data from Alpaca...")
    df = load_data(cfg)
    logger.info(f"Loaded {len(df)} bars ({df.index[0]} → {df.index[-1]})")

    # 2. Compute indicators
    logger.info("Computing indicators...")
    df = compute_all(df, cfg)

    # 3. Generate signals
    logger.info("Generating signals...")
    df = generate_signals(df, cfg)
    signal_counts = df["signal_name"].value_counts()
    logger.info(f"Signals generated:\n{signal_counts.to_string()}")

    # 4. Run simulation
    logger.info("Running simulation...")
    trades, equity_curve = run_simulation(df, cfg)

    # 5. Compute & print metrics
    overall = compute_metrics(trades, equity_curve, cfg.initial_capital)
    per_strat = per_strategy_breakdown(trades, equity_curve, cfg.initial_capital)
    by_hour = hourly_pnl(trades)
    by_day = weekday_pnl(trades)

    print_summary(overall, per_strat, by_hour, by_day)

    # 6. Optionally save trade log
    if args.save_trades and trades:
        out_dir = Path(__file__).resolve().parents[3] / "data" / "backtests" / "dayx"
        out_dir.mkdir(parents=True, exist_ok=True)
        trade_file = out_dir / f"trades_{cfg.symbol}_{cfg.start_date}_{cfg.end_date}.csv"

        import pandas as pd
        records = []
        for t in trades:
            records.append({
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "strategy": t.strategy,
                "direction": t.direction,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "size": t.size,
                "pnl": t.pnl,
                "exit_reason": t.exit_reason,
                "hold_min": (t.exit_time - t.entry_time).total_seconds() / 60 if t.exit_time else 0,
                "mae": t.max_adverse,
                "mfe": t.max_favorable,
            })
        pd.DataFrame(records).to_csv(trade_file, index=False)
        logger.info(f"Trade log saved to {trade_file}")


if __name__ == "__main__":
    main()
