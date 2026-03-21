"""RL configuration — extends DayXDipConfig with RL hyperparameters."""

from dataclasses import dataclass, field
import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parents[3])
if _root not in sys.path:
    sys.path.insert(0, _root)

from research.backtesting.dayx_dip.config import DayXDipConfig


@dataclass
class RLConfig:
    # Base strategy config (indicator params, stop logic, session times)
    dip_config: DayXDipConfig = field(default_factory=DayXDipConfig)

    # RL hyperparams
    algorithm: str = "PPO"
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    total_timesteps: int = 1_000_000

    # Reward shaping
    reward_invalid_buy: float = -0.05   # penalty for buy when guardrails block
    reward_flat_per_bar: float = -0.003  # opportunity cost for sitting flat in entry window
    max_trades_per_day: int = 3

    # Clean 3-way data split
    train_start: str = "2020-01-01"
    train_end: str = "2022-12-31"
    val_start: str = "2023-01-01"
    val_end: str = "2023-12-31"
    test_start: str = "2024-01-01"
    test_end: str = "2025-01-01"

    # Env
    timeframe: str = "3Min"
    symbol: str = "QQQ"
