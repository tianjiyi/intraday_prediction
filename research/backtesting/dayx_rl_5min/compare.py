"""Side-by-side comparison: RL-filtered vs unfiltered DayX strategy.

Uses the REAL sequential backtest (run_simulation) for both paths,
ensuring apples-to-apples comparison with position gating.
"""

import logging
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from .config import RLFilterConfig
from .features import build_signal_observation

logger = logging.getLogger(__name__)

_root = Path(__file__).resolve().parents[3]
_out_dir = _root / "data" / "backtests" / "dayx_rl_5min"


def _prepare_data(cfg: RLFilterConfig, split: str = "test"):
    """Load and prepare data for a given split. Returns (df, data_cfg)."""
    from research.backtesting.dayx.data_loader import (
        fetch_bars_cached, filter_rth, add_session_markers, compute_opening_range,
    )
    from research.backtesting.dayx.indicators import compute_all
    from research.backtesting.dayx.signals import generate_signals
    from dataclasses import replace

    dcfg = cfg.dayx_config
    if split == "train":
        start, end = cfg.train_start, cfg.train_end
    elif split == "val":
        start, end = cfg.val_start, cfg.val_end
    else:
        start, end = cfg.test_start, cfg.test_end

    data_cfg = replace(dcfg, start_date=start, end_date=end)
    df = fetch_bars_cached(data_cfg)
    df = filter_rth(df, data_cfg)
    df = add_session_markers(df, data_cfg)
    df = compute_opening_range(df, data_cfg)
    df = compute_all(df, data_cfg)
    df["volume_sma_20"] = df["volume"].rolling(20, min_periods=1).mean()
    df = generate_signals(df, data_cfg)
    return df, data_cfg


def run_filtered_simulation(cfg: RLFilterConfig, model_path: str = None,
                            split: str = "test"):
    """
    Run RL-filtered backtest using the REAL sequential simulator.

    1. Load data and generate all signals
    2. For each signal bar, ask the RL model: accept or reject?
    3. Set signal_any=False on rejected signals
    4. Run the standard run_simulation() on the modified DataFrame
    """
    from research.backtesting.dayx.simulator import run_simulation
    from research.backtesting.dayx.metrics import compute_metrics

    if model_path is None:
        model_path = str(_out_dir / "models" / "final_model")

    model = PPO.load(model_path)
    df, data_cfg = _prepare_data(cfg, split)

    # Find all signal bars and classify with RL model
    signal_mask = df["signal_any"].fillna(False).astype(bool)
    signal_rows = df[signal_mask]

    total_signals = len(signal_rows)
    accepted = 0
    rejected = 0

    for idx in signal_rows.index:
        row = df.loc[idx]
        signal_name = row["signal_name"]
        volume_sma = row.get("volume_sma_20", 1.0)
        obs = build_signal_observation(row, signal_name, volume_sma)

        action, _ = model.predict(obs, deterministic=True)
        if int(action) == 0:  # reject
            df.at[idx, "signal_any"] = False
            rejected += 1
        else:
            accepted += 1

    print(f"  RL Filter: {accepted}/{total_signals} signals accepted "
          f"({accepted/max(total_signals,1)*100:.0f}%), "
          f"{rejected} rejected ({rejected/max(total_signals,1)*100:.0f}%)")

    # Run real sequential backtest with filtered signals
    trades, eq = run_simulation(df, data_cfg)
    metrics = compute_metrics(trades, eq, data_cfg.initial_capital)

    return trades, eq, metrics, {"accepted": accepted, "rejected": rejected,
                                  "total_signals": total_signals}


def run_unfiltered(cfg: RLFilterConfig, split: str = "test"):
    """Run unfiltered DayX strategy (accept all signals) on the same split."""
    from research.backtesting.dayx.simulator import run_simulation
    from research.backtesting.dayx.metrics import compute_metrics

    df, data_cfg = _prepare_data(cfg, split)
    trades, eq = run_simulation(df, data_cfg)
    metrics = compute_metrics(trades, eq, data_cfg.initial_capital)
    return trades, eq, metrics


def compare(cfg: RLFilterConfig, model_path: str = None, split: str = "test"):
    """Compare RL-filtered vs unfiltered strategy using real sequential backtest."""
    print(f"\n{'='*70}")
    print(f"COMPARISON: RL-Filtered vs Unfiltered — {split.upper()} SPLIT")
    print(f"(Both use identical sequential backtest with position gating)")
    print(f"{'='*70}")

    # Unfiltered baseline
    print("\n--- Unfiltered (all signals accepted) ---")
    uf_trades, _, uf_metrics = run_unfiltered(cfg, split)

    # RL-filtered (real sequential backtest)
    print("\n--- RL-Filtered (sequential backtest) ---")
    rl_trades, _, rl_metrics, filter_stats = run_filtered_simulation(
        cfg, model_path, split
    )

    # Side-by-side comparison with identical metrics
    print(f"\n{'METRIC':<25} {'UNFILTERED':>15} {'RL-FILTERED':>15} {'DELTA':>12}")
    print("-" * 67)

    def _fmt(label, uf_val, rl_val, fmt_str, delta_fmt):
        uf_s = fmt_str.format(uf_val)
        rl_s = fmt_str.format(rl_val)
        d = rl_val - uf_val
        d_s = delta_fmt.format(d)
        print(f"{label:<25} {uf_s} {rl_s} {d_s}")

    uf_tc = uf_metrics.get("trade_count", 0)
    rl_tc = rl_metrics.get("trade_count", 0)
    uf_pnl = uf_metrics.get("total_pnl", 0)
    rl_pnl = rl_metrics.get("total_pnl", 0)

    _fmt("Trade count", uf_tc, rl_tc, "{:>15}", "{:>+12.0f}")
    _fmt("Win rate", uf_metrics.get("win_rate", 0), rl_metrics.get("win_rate", 0),
         "{:>14.1f}%", "{:>+11.1f}pp")
    _fmt("Total P&L", uf_pnl, rl_pnl, "${:>14,.0f}", "${:>+11,.0f}")
    _fmt("Avg P&L/trade", uf_pnl / max(uf_tc, 1), rl_pnl / max(rl_tc, 1),
         "${:>14,.2f}", "${:>+11,.2f}")
    _fmt("Sharpe", uf_metrics.get("sharpe", 0), rl_metrics.get("sharpe", 0),
         "{:>15.2f}", "{:>+12.2f}")
    _fmt("Profit Factor", uf_metrics.get("profit_factor", 0),
         rl_metrics.get("profit_factor", 0), "{:>15.2f}", "{:>+12.2f}")
    _fmt("Max DD", uf_metrics.get("max_drawdown_pct", 0),
         rl_metrics.get("max_drawdown_pct", 0), "{:>14.1f}%", "{:>+11.1f}pp")
    _fmt("Avg Hold (min)", uf_metrics.get("avg_hold_min", 0),
         rl_metrics.get("avg_hold_min", 0), "{:>15.1f}", "{:>+12.1f}")

    # Filter stats
    print(f"\n{'Filter Stats':<25}")
    print(f"{'  Total signals':<25} {filter_stats['total_signals']:>15}")
    print(f"{'  RL accepted':<25} {filter_stats['accepted']:>15} "
          f"({filter_stats['accepted']/max(filter_stats['total_signals'],1)*100:.0f}%)")
    print(f"{'  RL rejected':<25} {filter_stats['rejected']:>15} "
          f"({filter_stats['rejected']/max(filter_stats['total_signals'],1)*100:.0f}%)")
    print(f"{'  Trades executed':<25} {rl_metrics.get('trade_count', 0):>15} "
          f"(after position gating)")

    # Per-strategy breakdown for both
    print(f"\n{'Per-Strategy Breakdown':<25}")
    print(f"  {'Strategy':<15} {'UF Trades':>10} {'UF WR%':>8} "
          f"{'RL Trades':>10} {'RL WR%':>8}")
    print("  " + "-" * 55)

    uf_by_strat = {}
    for t in uf_trades:
        s = t.strategy
        uf_by_strat.setdefault(s, []).append(t)

    rl_by_strat = {}
    for t in rl_trades:
        s = t.strategy
        rl_by_strat.setdefault(s, []).append(t)

    all_strats = sorted(set(list(uf_by_strat.keys()) + list(rl_by_strat.keys())))
    for strat in all_strats:
        uf_st = uf_by_strat.get(strat, [])
        rl_st = rl_by_strat.get(strat, [])
        uf_wr = sum(1 for t in uf_st if t.pnl > 0) / max(len(uf_st), 1) * 100
        rl_wr = sum(1 for t in rl_st if t.pnl > 0) / max(len(rl_st), 1) * 100
        print(f"  {strat:<15} {len(uf_st):>10} {uf_wr:>7.0f}% "
              f"{len(rl_st):>10} {rl_wr:>7.0f}%")

    print(f"\n{'='*70}")
