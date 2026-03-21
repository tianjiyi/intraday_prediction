"""
CLI entry point for DayX Dip RL agent.

Usage:
    python research/backtesting/dayx_dip_rl/run.py train --timesteps 1000000
    python research/backtesting/dayx_dip_rl/run.py evaluate --split test
    python research/backtesting/dayx_dip_rl/run.py compare --split test
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
    parser = argparse.ArgumentParser(description="DayX Dip RL Agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train
    train_p = subparsers.add_parser("train")
    train_p.add_argument("--symbol", default="QQQ")
    train_p.add_argument("--timeframe", default="3Min")
    train_p.add_argument("--timesteps", type=int, default=1_000_000)
    train_p.add_argument("--lr", type=float, default=3e-4)
    train_p.add_argument("--seed", type=int, default=42)

    # Evaluate
    eval_p = subparsers.add_parser("evaluate")
    eval_p.add_argument("--model", default=None, help="Path to model zip")
    eval_p.add_argument("--split", default="test", choices=["train", "val", "test"])

    # Compare
    comp_p = subparsers.add_parser("compare")
    comp_p.add_argument("--model", default=None)
    comp_p.add_argument("--split", default="test", choices=["train", "val", "test"])

    args = parser.parse_args()

    from research.backtesting.dayx_dip_rl.config import RLConfig

    if args.command == "train":
        cfg = RLConfig(
            symbol=args.symbol,
            timeframe=args.timeframe,
            total_timesteps=args.timesteps,
            learning_rate=args.lr,
        )
        from research.backtesting.dayx_dip_rl.train import train
        train(cfg, seed=args.seed)

    elif args.command == "evaluate":
        cfg = RLConfig()
        from research.backtesting.dayx_dip_rl.evaluate import evaluate
        evaluate(cfg, model_path=args.model, split=args.split)

    elif args.command == "compare":
        cfg = RLConfig()
        from research.backtesting.dayx_dip_rl.compare import compare
        compare(cfg, model_path=args.model, split=args.split)


if __name__ == "__main__":
    main()
