"""
Optuna hyperparameter optimization for Day_X strategy.
Walk-forward validation across rolling 5-year train / 1-year validate windows.

Usage:
    python research/backtesting/dayx/optimize.py --timeframes 5Min --n-trials 500
    python research/backtesting/dayx/optimize.py --timeframes 1Min 3Min 5Min --n-trials 500 --final-backtest
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from dataclasses import asdict

if __name__ == "__main__":
    _root = str(Path(__file__).resolve().parents[3])
    sys.path.insert(0, _root)
    import types
    sys.modules.setdefault("research.backtesting", types.ModuleType("research.backtesting"))
    sys.modules["research.backtesting"].__path__ = [
        str(Path(_root) / "research" / "backtesting")
    ]

import numpy as np
import pandas as pd
import optuna

from research.backtesting.dayx.config import DayXConfig
from research.backtesting.dayx.data_loader import (
    fetch_bars_cached, filter_rth, add_session_markers, compute_opening_range,
)
from research.backtesting.dayx.indicators import compute_all
from research.backtesting.dayx.signals import generate_signals
from research.backtesting.dayx.simulator import run_simulation
from research.backtesting.dayx.metrics import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_root_path = Path(__file__).resolve().parents[3]
_out_dir = _root_path / "data" / "backtests" / "dayx"

# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

SEARCH_SPACE = {
    # Signal params
    "rsi_period": ("int", 7, 21),
    "bb_lookback_bars": ("int", 1, 10),
    "trend_dip_vwap_pct": ("float", 0.1, 0.5),
    "trend_dip_above_pct": ("float", 0.4, 0.8),
    "cci_neutral_lo": ("int", -100, 0),
    "cci_neutral_hi": ("int", 0, 100),
    # Risk params
    "stop_atr_mult": ("float", 1.0, 3.0),
    "target1_r": ("float", 0.5, 2.0),
    "target2_r": ("float", 1.5, 4.0),
    "partial_exit_pct": ("float", 0.3, 0.7),
    "trail_progressive_step": ("float", 0.25, 1.0),
    # Indicator params
    "cci_fast": ("int", 8, 20),
    "bb_period": ("int", 10, 30),
    "bb_std": ("float", 1.5, 3.0),
    "atr_period": ("int", 10, 20),
    # Time params (minutes from midnight; converted to HH:MM for config)
    "entry_earliest_min": ("int", 570, 630),   # 09:30 to 10:30
    "entry_latest_min": ("int", 780, 900),     # 13:00 to 15:00
}

FIXED_PARAMS = {
    "strategies": ["buy_dip", "sell_rip", "trend_dip"],
    "use_vwap_target": False,
    "trail_mode": "progressive",
    "time_filter": True,
    "require_exhaustion": False,
    "require_cci20_alignment": False,
}


def _min_to_time(m: int) -> str:
    return f"{m // 60:02d}:{m % 60:02d}"


# ---------------------------------------------------------------------------
# Walk-forward windows
# ---------------------------------------------------------------------------

def _walk_forward_windows(start_year: int = 2016, end_year: int = 2025,
                          train_years: int = 5, val_years: int = 1):
    windows = []
    for y in range(start_year, end_year - train_years - val_years + 2):
        train_start = f"{y}-01-01"
        train_end = f"{y + train_years - 1}-12-31"
        val_start = f"{y + train_years}-01-01"
        val_end = f"{y + train_years + val_years - 1}-12-31"
        windows.append((train_start, train_end, val_start, val_end))
    return windows


# ---------------------------------------------------------------------------
# Backtest on a date slice
# ---------------------------------------------------------------------------

def _run_backtest_on_slice(df_raw: pd.DataFrame, cfg: DayXConfig,
                           date_start: str, date_end: str) -> dict:
    """Slice raw data, apply full pipeline, return metrics dict."""
    df = df_raw.loc[date_start:date_end].copy()
    if len(df) < 200:
        return {"sharpe": 0.0, "trade_count": 0, "max_drawdown_pct": 0.0}

    df = filter_rth(df, cfg)
    df = add_session_markers(df, cfg)
    df = compute_opening_range(df, cfg)
    df = compute_all(df, cfg)
    df = generate_signals(df, cfg)
    trades, eq = run_simulation(df, cfg)

    if len(trades) < 10:
        return {"sharpe": 0.0, "trade_count": len(trades), "max_drawdown_pct": 0.0}

    metrics = compute_metrics(trades, eq, cfg.initial_capital)
    return metrics


def _composite_score(sharpe: float, trade_count: int, max_dd_pct: float) -> float:
    """
    Composite objective that penalizes tail risk and low trade count.
    score = sharpe * sqrt(trade_count / 100) - 0.1 * abs(max_drawdown)
    """
    volume_adj = np.sqrt(max(trade_count, 0) / 100.0)
    dd_penalty = 0.1 * abs(max_dd_pct)
    return sharpe * volume_adj - dd_penalty


# ---------------------------------------------------------------------------
# Objective function factory
# ---------------------------------------------------------------------------

def create_objective(df_raw: pd.DataFrame, symbol: str, timeframe: str,
                     windows: list):
    """Return an Optuna objective using composite score:
    score = sharpe * sqrt(trade_count / 100) - 0.1 * |max_drawdown%|
    Penalizes tail risk and low trade count (overfitting to rare signals)."""

    def objective(trial: optuna.Trial) -> float:
        # Suggest params
        params = {}
        for name, (ptype, lo, hi) in SEARCH_SPACE.items():
            if ptype == "int":
                params[name] = trial.suggest_int(name, lo, hi)
            else:
                params[name] = trial.suggest_float(name, lo, hi)

        # Convert time params
        params["entry_earliest"] = _min_to_time(params.pop("entry_earliest_min"))
        params["entry_latest"] = _min_to_time(params.pop("entry_latest_min"))

        # Constraint: target2 must be > target1
        if params["target2_r"] <= params["target1_r"]:
            raise optuna.TrialPruned()

        cfg = DayXConfig(
            symbol=symbol, timeframe=timeframe,
            **FIXED_PARAMS, **params,
        )

        val_scores = []
        for i, (train_start, train_end, val_start, val_end) in enumerate(windows):
            # Prune if params can't be profitable on training data
            # (skip pruning for first 20 trials to let TPE warm up)
            train_m = _run_backtest_on_slice(df_raw, cfg, train_start, train_end)
            if trial.number >= 20 and train_m.get("sharpe", 0.0) < 0:
                raise optuna.TrialPruned()

            val_m = _run_backtest_on_slice(df_raw, cfg, val_start, val_end)
            score = _composite_score(
                val_m.get("sharpe", 0.0),
                val_m.get("trade_count", 0),
                val_m.get("max_drawdown_pct", 0.0),
            )
            val_scores.append(score)

            # Report intermediate value for Optuna's pruner
            trial.report(np.mean(val_scores), i)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(val_scores))

    return objective


# ---------------------------------------------------------------------------
# Run optimization
# ---------------------------------------------------------------------------

def run_optimization(symbol: str, timeframe: str, n_trials: int = 500,
                     n_jobs: int = 1, storage: str | None = None) -> optuna.Study:
    # Load raw data for full range
    cfg_data = DayXConfig(symbol=symbol, timeframe=timeframe,
                          start_date="2016-01-01", end_date="2025-12-31")
    df_raw = fetch_bars_cached(cfg_data)
    logger.info(f"Loaded {len(df_raw)} raw bars for {symbol} {timeframe}")

    windows = _walk_forward_windows()
    logger.info(f"Walk-forward windows: {len(windows)}")
    for ts, te, vs, ve in windows:
        logger.info(f"  Train {ts} → {te}  |  Val {vs} → {ve}")

    # SQLite storage for resumability
    if storage is None:
        db_path = _out_dir / "optuna.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{db_path}"

    study_name = f"dayx_v3_{symbol}_{timeframe}"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
    )

    objective = create_objective(df_raw, symbol, timeframe, windows)
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)

    return study


# ---------------------------------------------------------------------------
# Results output
# ---------------------------------------------------------------------------

def print_results(study: optuna.Study, symbol: str, timeframe: str):
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print(f"\n{'='*70}")
        print(f"NO COMPLETED TRIALS for {symbol} {timeframe}")
        print(f"  All {len(study.trials)} trials were pruned. Try more trials.")
        print(f"{'='*70}")
        return

    best = study.best_trial
    print(f"\n{'='*70}")
    print(f"BEST RESULT: {symbol} {timeframe}")
    print(f"  Composite score: {best.value:.4f}  (sharpe * sqrt(n/100) - 0.1*|maxDD|)")
    print(f"  Trial #{best.number} of {len(study.trials)}")
    print(f"  Params:")
    for k, v in sorted(best.params.items()):
        if k.endswith("_min"):
            print(f"    {k.replace('_min', '')}: {_min_to_time(v)} ({v})")
        else:
            print(f"    {k}: {v}")

    # Save best config as JSON
    _out_dir.mkdir(parents=True, exist_ok=True)
    params = dict(best.params)
    params["entry_earliest"] = _min_to_time(params.pop("entry_earliest_min"))
    params["entry_latest"] = _min_to_time(params.pop("entry_latest_min"))
    full_config = {**FIXED_PARAMS, **params, "symbol": symbol, "timeframe": timeframe}
    # Convert list to make JSON serializable
    full_config["strategies"] = list(full_config["strategies"])

    config_file = _out_dir / f"best_config_{symbol}_{timeframe}.json"
    with open(config_file, "w") as f:
        json.dump(full_config, f, indent=2)
    print(f"  Config saved to {config_file}")
    print(f"{'='*70}")


def run_final_backtest(study: optuna.Study, symbol: str, timeframe: str):
    """Run full-period backtest with best params and print metrics."""
    params = dict(study.best_trial.params)
    params["entry_earliest"] = _min_to_time(params.pop("entry_earliest_min"))
    params["entry_latest"] = _min_to_time(params.pop("entry_latest_min"))

    cfg = DayXConfig(
        symbol=symbol, timeframe=timeframe,
        start_date="2016-01-01", end_date="2025-12-31",
        **FIXED_PARAMS, **params,
    )

    cfg_data = DayXConfig(symbol=symbol, timeframe=timeframe,
                          start_date="2016-01-01", end_date="2025-12-31")
    df_raw = fetch_bars_cached(cfg_data)
    df = filter_rth(df_raw, cfg)
    df = add_session_markers(df, cfg)
    df = compute_opening_range(df, cfg)
    df = compute_all(df, cfg)
    df = generate_signals(df, cfg)
    trades, eq = run_simulation(df, cfg)
    metrics = compute_metrics(trades, eq, cfg.initial_capital)

    print(f"\n--- Final Backtest: {symbol} {timeframe} (2016-2025) ---")
    print(f"  Trades: {metrics['trade_count']}")
    print(f"  Win Rate: {metrics['win_rate']:.1f}%")
    print(f"  Total P&L: ${metrics['total_pnl']:,.0f}")
    print(f"  Return: {metrics['total_return_pct']:.1f}%")
    print(f"  Sharpe: {metrics['sharpe']:.2f}")
    print(f"  Max Drawdown: {metrics['max_drawdown_pct']:.1f}%")
    print(f"  Profit Factor: {metrics['profit_factor']:.2f}")

    # Per-year breakdown
    print(f"\n  Per-year P&L:")
    for year in range(2016, 2026):
        yt = [t for t in trades if t.entry_time.year == year]
        if yt:
            pnl = sum(t.pnl for t in yt)
            wr = sum(1 for t in yt if t.pnl > 0) / len(yt) * 100
            print(f"    {year}: ${pnl:>9,.0f}  ({len(yt)} trades, {wr:.0f}% WR)")

    # Per-strategy breakdown
    print(f"\n  Per-strategy:")
    for strat in sorted(set(t.strategy for t in trades)):
        st = [t for t in trades if t.strategy == strat]
        pnl = sum(t.pnl for t in st)
        wr = sum(1 for t in st if t.pnl > 0) / len(st) * 100
        print(f"    {strat:15s}: n={len(st):4d}  wr={wr:5.1f}%  pnl=${pnl:>9,.0f}")

    return trades, eq, metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Day_X Optuna Hyperparameter Optimization")
    parser.add_argument("--symbol", default="QQQ")
    parser.add_argument("--timeframes", nargs="+", default=["5Min"])
    parser.add_argument("--n-trials", type=int, default=500)
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Parallel trials (default 1; try 4-6 for multi-core)")
    parser.add_argument("--final-backtest", action="store_true",
                        help="Run final backtest with best params after optimization")
    args = parser.parse_args()

    all_studies = {}
    for tf in args.timeframes:
        logger.info(f"\n{'='*70}")
        logger.info(f"Optimizing {args.symbol} {tf} ({args.n_trials} trials, {args.n_jobs} jobs)")
        logger.info(f"{'='*70}")

        study = run_optimization(args.symbol, tf, args.n_trials, n_jobs=args.n_jobs)
        print_results(study, args.symbol, tf)
        all_studies[tf] = study

    # Cross-timeframe comparison
    if len(all_studies) > 1:
        print(f"\n{'='*70}")
        print(f"CROSS-TIMEFRAME COMPARISON")
        print(f"{'='*70}")
        print(f"{'Timeframe':<10} {'Best Score':>12} {'Trials':>8} {'Completed':>10}")
        print("-" * 45)
        for tf, study in all_studies.items():
            completed = [t for t in study.trials
                        if t.state == optuna.trial.TrialState.COMPLETE]
            best_val = study.best_value if completed else float("nan")
            print(f"{tf:<10} {best_val:>12.4f} {len(study.trials):>8} {len(completed):>10}")

    # Final backtest with best params
    if args.final_backtest:
        for tf, study in all_studies.items():
            completed = [t for t in study.trials
                        if t.state == optuna.trial.TrialState.COMPLETE]
            if not completed:
                continue
            run_final_backtest(study, args.symbol, tf)


if __name__ == "__main__":
    main()
