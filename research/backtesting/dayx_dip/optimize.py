"""
Optuna hyperparameter optimization for DayX Dip strategy.
Walk-forward: 1-year train / 3-month validate windows.

Usage:
    python research/backtesting/dayx_dip/optimize.py --symbol QQQ --timeframe 1Min --n-trials 300
    python research/backtesting/dayx_dip/optimize.py --symbol QQQ --timeframe 3Min --n-trials 300
"""

import sys
import json
import logging
import argparse
from pathlib import Path

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

from research.backtesting.dayx_dip.config import DayXDipConfig
from research.backtesting.dayx_dip.data_loader import (
    fetch_bars_cached, filter_rth, add_session_markers,
)
from research.backtesting.dayx_dip.indicators import compute_all
from research.backtesting.dayx_dip.signals import bear_exhaustion_dip
from research.backtesting.dayx_dip.simulator import run_simulation
from research.backtesting.dayx_dip.metrics import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_root_path = Path(__file__).resolve().parents[3]
_out_dir = _root_path / "data" / "backtests" / "dayx_dip"


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

SEARCH_SPACE = {
    # Indicator params
    "bb_period": ("int", 10, 30),
    "bb_std": ("float", 1.5, 3.0),
    "rsi_period": ("int", 10, 20),
    "rsi_oversold": ("float", 25.0, 45.0),
    "cci_period": ("int", 8, 20),
    "cci_oversold": ("float", -150.0, -75.0),
    # Signal params
    "exhaustion_lookback": ("int", 2, 6),
    "be_bb_zone_pct": ("float", 0.1, 1.5),
    # Risk params
    "atr_period": ("int", 10, 20),
    "stop_atr_mult": ("float", 1.5, 3.5),
    # Time params (minutes from midnight)
    "entry_earliest_min": ("int", 570, 660),   # 09:30 to 11:00
    "no_new_entries_after_min": ("int", 750, 900),  # 12:30 to 15:00
}


def _min_to_time(m: int) -> str:
    return f"{m // 60:02d}:{m % 60:02d}"


# ---------------------------------------------------------------------------
# Walk-forward windows: 1-year train / 3-month validate
# ---------------------------------------------------------------------------

def _walk_forward_windows(start_year: int = 2020, end_year: int = 2025):
    """Generate quarterly walk-forward windows."""
    windows = []
    quarters = [(1, 3), (4, 6), (7, 9), (10, 12)]
    quarter_months = {1: "01-01", 4: "04-01", 7: "07-01", 10: "10-01"}
    quarter_ends = {3: "03-31", 6: "06-30", 9: "09-30", 12: "12-31"}

    for year in range(start_year, end_year):
        for q_start, q_end in quarters:
            # Train: 1 year before this quarter
            train_start_year = year - 1
            train_start_q = q_start
            train_start = f"{train_start_year}-{quarter_months[train_start_q]}"
            train_end_year = year
            train_end = f"{train_end_year}-{quarter_months[q_start]}"
            # Actually: train for 1 year ending at quarter start
            train_start_dt = f"{year - 1}-{quarter_months[q_start]}"
            val_start = f"{year}-{quarter_months[q_start]}"
            val_end = f"{year}-{quarter_ends[q_end]}"
            # Skip if val end would be in future
            import datetime
            try:
                ve = datetime.date.fromisoformat(val_end)
                if ve > datetime.date(end_year, 12, 31):
                    continue
            except ValueError:
                continue
            windows.append((train_start_dt, val_start, val_start, val_end))

    return windows


# ---------------------------------------------------------------------------
# Backtest helper
# ---------------------------------------------------------------------------

def _run_slice(df_raw: pd.DataFrame, cfg: DayXDipConfig,
               date_start: str, date_end: str) -> dict:
    """Run pipeline on a date slice and return metrics."""
    df = df_raw.loc[date_start:date_end].copy()
    if len(df) < 100:
        return {"sharpe": 0.0, "trade_count": 0, "max_drawdown_pct": 0.0}

    df = filter_rth(df, cfg)
    df = add_session_markers(df)
    df = compute_all(df, cfg)
    df = bear_exhaustion_dip(df, cfg)
    trades, eq = run_simulation(df, cfg)

    if len(trades) < 5:
        return {"sharpe": 0.0, "trade_count": len(trades), "max_drawdown_pct": 0.0}

    return compute_metrics(trades, eq, cfg.initial_capital)


def _composite_score(sharpe: float, trade_count: int, max_dd_pct: float) -> float:
    """score = sharpe * sqrt(trade_count / 100) - 0.1 * |max_drawdown%|"""
    volume_adj = np.sqrt(max(trade_count, 0) / 100.0)
    dd_penalty = 0.1 * abs(max_dd_pct)
    return sharpe * volume_adj - dd_penalty


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def create_objective(df_raw: pd.DataFrame, symbol: str, timeframe: str, windows: list):
    def objective(trial: optuna.Trial) -> float:
        params = {}
        for name, (ptype, lo, hi) in SEARCH_SPACE.items():
            if ptype == "int":
                params[name] = trial.suggest_int(name, lo, hi)
            else:
                params[name] = trial.suggest_float(name, lo, hi)

        params["entry_earliest"] = _min_to_time(params.pop("entry_earliest_min"))
        params["no_new_entries_after"] = _min_to_time(params.pop("no_new_entries_after_min"))

        cfg = DayXDipConfig(symbol=symbol, timeframe=timeframe, **params)

        val_scores = []
        for i, (train_start, train_end, val_start, val_end) in enumerate(windows):
            # Prune if unprofitable on train (after warmup)
            if trial.number >= 20:
                train_m = _run_slice(df_raw, cfg, train_start, train_end)
                if train_m.get("sharpe", 0.0) < 0:
                    raise optuna.TrialPruned()

            val_m = _run_slice(df_raw, cfg, val_start, val_end)
            score = _composite_score(
                val_m.get("sharpe", 0.0),
                val_m.get("trade_count", 0),
                val_m.get("max_drawdown_pct", 0.0),
            )
            val_scores.append(score)
            trial.report(np.mean(val_scores), i)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(val_scores))

    return objective


# ---------------------------------------------------------------------------
# Run optimization
# ---------------------------------------------------------------------------

def run_optimization(symbol: str, timeframe: str, n_trials: int = 300,
                     n_jobs: int = 1, storage: str | None = None) -> optuna.Study:
    cfg_data = DayXDipConfig(
        symbol=symbol, timeframe=timeframe,
        start_date="2019-01-01", end_date="2025-12-31",
    )
    df_raw = fetch_bars_cached(cfg_data)
    logger.info(f"Loaded {len(df_raw)} raw bars for {symbol} {timeframe}")

    windows = _walk_forward_windows()
    logger.info(f"Walk-forward windows: {len(windows)}")
    for ts, te, vs, ve in windows:
        logger.info(f"  Train {ts} → {te}  |  Val {vs} → {ve}")

    if storage is None:
        db_path = _out_dir / "optuna.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{db_path}"

    study_name = f"dayx_dip_{symbol}_{timeframe}"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
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
        print(f"\nNO COMPLETED TRIALS for {symbol} {timeframe}")
        return

    best = study.best_trial
    print(f"\n{'='*70}")
    print(f"BEST RESULT: {symbol} {timeframe}")
    print(f"  Composite score: {best.value:.4f}")
    print(f"  Trial #{best.number} of {len(study.trials)}")
    print(f"  Params:")
    for k, v in sorted(best.params.items()):
        if k.endswith("_min"):
            print(f"    {k.replace('_min', '')}: {_min_to_time(v)} ({v})")
        else:
            print(f"    {k}: {v}")

    _out_dir.mkdir(parents=True, exist_ok=True)
    params = dict(best.params)
    params["entry_earliest"] = _min_to_time(params.pop("entry_earliest_min"))
    params["no_new_entries_after"] = _min_to_time(params.pop("no_new_entries_after_min"))
    full_config = {**params, "symbol": symbol, "timeframe": timeframe}

    config_file = _out_dir / f"best_config_{symbol}_{timeframe}.json"
    with open(config_file, "w") as f:
        json.dump(full_config, f, indent=2)
    print(f"  Config saved to {config_file}")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DayX Dip Optuna Optimization")
    parser.add_argument("--symbol", default="QQQ")
    parser.add_argument("--timeframe", default="1Min")
    parser.add_argument("--n-trials", type=int, default=300)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    logger.info(f"Optimizing {args.symbol} {args.timeframe} ({args.n_trials} trials)")
    study = run_optimization(args.symbol, args.timeframe, args.n_trials, args.n_jobs)
    print_results(study, args.symbol, args.timeframe)


if __name__ == "__main__":
    main()
