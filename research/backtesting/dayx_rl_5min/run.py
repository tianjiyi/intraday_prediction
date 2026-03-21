"""
CLI entry point for DayX RL Signal Filter.

Usage:
    python research/backtesting/dayx_rl_5min/run.py optimize-1min --n-trials 500
    python research/backtesting/dayx_rl_5min/run.py train --timesteps 500000
    python research/backtesting/dayx_rl_5min/run.py evaluate --split test
    python research/backtesting/dayx_rl_5min/run.py compare --split test
    python research/backtesting/dayx_rl_5min/run.py ml-compare --split test
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="DayX RL Signal Filter")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Optimize 1Min
    opt_p = subparsers.add_parser("optimize-1min")
    opt_p.add_argument("--n-trials", type=int, default=500)
    opt_p.add_argument("--symbol", default="QQQ")

    # Train
    train_p = subparsers.add_parser("train")
    train_p.add_argument("--timesteps", type=int, default=500_000)
    train_p.add_argument("--lr", type=float, default=3e-4)
    train_p.add_argument("--seed", type=int, default=42)
    train_p.add_argument("--config", default=None, help="Path to DayX config JSON")

    # Evaluate
    eval_p = subparsers.add_parser("evaluate")
    eval_p.add_argument("--model", default=None, help="Path to model zip")
    eval_p.add_argument("--split", default="test", choices=["train", "val", "test"])
    eval_p.add_argument("--config", default=None)

    # Compare
    comp_p = subparsers.add_parser("compare")
    comp_p.add_argument("--model", default=None)
    comp_p.add_argument("--split", default="test", choices=["train", "val", "test"])
    comp_p.add_argument("--config", default=None)

    # ML Compare (4-way: Unfiltered vs RL vs XGBoost vs LogReg)
    ml_p = subparsers.add_parser("ml-compare")
    ml_p.add_argument("--model", default=None, help="Path to RL model zip")
    ml_p.add_argument("--split", default="test", choices=["train", "val", "test"])
    ml_p.add_argument("--config", default=None)

    args = parser.parse_args()

    if args.command == "optimize-1min":
        from research.backtesting.dayx.optimize import run_optimization, print_results, run_final_backtest
        study = run_optimization(args.symbol, "1Min", args.n_trials)
        print_results(study, args.symbol, "1Min")
        run_final_backtest(study, args.symbol, "1Min")

    elif args.command == "train":
        from research.backtesting.dayx_rl_5min.config import RLFilterConfig, load_dayx_config
        dayx_cfg = load_dayx_config(args.config)
        cfg = RLFilterConfig(
            dayx_config=dayx_cfg,
            total_timesteps=args.timesteps,
            learning_rate=args.lr,
        )
        from research.backtesting.dayx_rl_5min.train import train
        train(cfg, seed=args.seed)

    elif args.command == "evaluate":
        from research.backtesting.dayx_rl_5min.config import RLFilterConfig, load_dayx_config
        dayx_cfg = load_dayx_config(args.config)
        cfg = RLFilterConfig(dayx_config=dayx_cfg)
        from research.backtesting.dayx_rl_5min.evaluate import evaluate
        evaluate(cfg, model_path=args.model, split=args.split)

    elif args.command == "compare":
        from research.backtesting.dayx_rl_5min.config import RLFilterConfig, load_dayx_config
        dayx_cfg = load_dayx_config(args.config)
        cfg = RLFilterConfig(dayx_config=dayx_cfg)
        from research.backtesting.dayx_rl_5min.compare import compare
        compare(cfg, model_path=args.model, split=args.split)

    elif args.command == "ml-compare":
        from research.backtesting.dayx_rl_5min.config import RLFilterConfig, load_dayx_config
        dayx_cfg = load_dayx_config(args.config)
        cfg = RLFilterConfig(dayx_config=dayx_cfg)
        from research.backtesting.dayx_rl_5min.ml_classifier import compare_all
        compare_all(cfg, rl_model_path=args.model, split=args.split)


if __name__ == "__main__":
    main()
