"""Side-by-side comparison: RL agent vs rule-based dayx_dip."""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RLConfig
from .evaluate import evaluate

logger = logging.getLogger(__name__)

_root = Path(__file__).resolve().parents[3]


def run_rule_based(cfg: RLConfig, split: str = "test"):
    """Run the rule-based dayx_dip strategy on the same data split."""
    from research.backtesting.dayx_dip.config import DayXDipConfig
    from research.backtesting.dayx_dip.data_loader import load_data
    from research.backtesting.dayx_dip.indicators import compute_all
    from research.backtesting.dayx_dip.signals import bear_exhaustion_dip
    from research.backtesting.dayx_dip.simulator import run_simulation
    from research.backtesting.dayx_dip.metrics import compute_metrics

    if split == "train":
        start, end = cfg.train_start, cfg.train_end
    elif split == "val":
        start, end = cfg.val_start, cfg.val_end
    else:
        start, end = cfg.test_start, cfg.test_end

    dcfg = DayXDipConfig(
        symbol=cfg.symbol,
        timeframe=cfg.timeframe,
        start_date=start,
        end_date=end,
    )

    df = load_data(dcfg)
    df = compute_all(df, dcfg)
    df = bear_exhaustion_dip(df, dcfg)
    trades, eq = run_simulation(df, dcfg)
    metrics = compute_metrics(trades, eq, dcfg.initial_capital)

    return trades, eq, metrics


def compare(cfg: RLConfig, model_path: str = None, split: str = "test",
            plot: bool = False):
    """Compare RL agent vs rule-based on the same split."""
    print(f"\n{'='*70}")
    print(f"COMPARISON: RL vs Rule-Based — {split.upper()} SPLIT")
    print(f"{'='*70}")

    # Rule-based
    print("\n--- Rule-Based DayX Dip ---")
    rb_trades, rb_eq, rb_metrics = run_rule_based(cfg, split)

    # RL agent
    print("\n--- RL Agent ---")
    rl_trades, rl_daily = evaluate(cfg, model_path, split)

    # Side-by-side table
    rl_r_multiples = [t["r_multiple"] for t in rl_trades] if rl_trades else []
    rl_wins = sum(1 for r in rl_r_multiples if r > 0)

    print(f"\n{'METRIC':<25} {'RULE-BASED':>15} {'RL AGENT':>15}")
    print("-" * 55)
    print(f"{'Trade count':<25} {rb_metrics.get('trade_count', 0):>15} {len(rl_trades):>15}")
    print(f"{'Win rate':<25} {rb_metrics.get('win_rate', 0):>14.1f}% {(rl_wins/len(rl_trades)*100 if rl_trades else 0):>14.1f}%")
    print(f"{'Avg entries/day':<25} {'—':>15} {np.mean([d['entries'] for d in rl_daily]) if rl_daily else 0:>15.2f}")
    print(f"{'Total P&L':<25} ${rb_metrics.get('total_pnl', 0):>14,.0f} {'—':>15}")
    print(f"{'Total R':<25} {'—':>15} {sum(rl_r_multiples):>15.2f}")
    print(f"{'Mean R-multiple':<25} {'—':>15} {np.mean(rl_r_multiples) if rl_r_multiples else 0:>15.3f}")
    print(f"{'Sharpe':<25} {rb_metrics.get('sharpe', 0):>15.2f} {'—':>15}")
    print(f"{'Max DD':<25} {rb_metrics.get('max_drawdown_pct', 0):>14.2f}% {'—':>15}")
    print(f"{'Profit Factor':<25} {rb_metrics.get('profit_factor', 0):>15.2f} {'—':>15}")

    # Exit reason comparison
    if rl_trades:
        print(f"\n{'EXIT REASONS':<25} {'RULE-BASED':>15} {'RL AGENT':>15}")
        print("-" * 55)
        all_reasons = set()
        rb_reasons = rb_metrics.get("exit_reasons", {})
        rl_reasons = {}
        for t in rl_trades:
            rl_reasons[t["exit_reason"]] = rl_reasons.get(t["exit_reason"], 0) + 1
        all_reasons.update(rb_reasons.keys(), rl_reasons.keys())
        for reason in sorted(all_reasons):
            print(f"  {reason:<23} {rb_reasons.get(reason, 0):>15} {rl_reasons.get(reason, 0):>15}")

    print(f"\n{'='*70}")
