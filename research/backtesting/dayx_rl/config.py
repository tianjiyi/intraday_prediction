"""RL signal filter configuration — DayXConfig + RL hyperparameters."""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

_root = str(Path(__file__).resolve().parents[3])
if _root not in sys.path:
    sys.path.insert(0, _root)

from research.backtesting.dayx.config import DayXConfig


def load_dayx_config(config_path: str = None) -> DayXConfig:
    """Load DayXConfig from Optuna best_config JSON."""
    if config_path is None:
        config_path = str(
            Path(__file__).resolve().parents[3]
            / "data" / "backtests" / "dayx" / "best_config_QQQ_1Min.json"
        )
    with open(config_path) as f:
        params = json.load(f)
    # Filter to only valid DayXConfig fields
    valid_fields = {f.name for f in DayXConfig.__dataclass_fields__.values()}
    filtered = {k: v for k, v in params.items() if k in valid_fields}
    return DayXConfig(**filtered)


@dataclass
class RLFilterConfig:
    # Base strategy config (loaded from Optuna best config)
    dayx_config: DayXConfig = field(default_factory=DayXConfig)

    # RL hyperparams
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    ent_coef: float = 0.15
    total_timesteps: int = 500_000
    reject_reward_scale: float = 0.5  # Asymmetric: reject reward = -R * scale
    r_clip: float = 2.0              # Clip R-multiples to [-r_clip, +r_clip]
    reject_penalty: float = 0.05     # Fixed cost per rejection

    # Clean 3-way data split
    train_start: str = "2016-01-01"
    train_end: str = "2022-12-31"
    val_start: str = "2023-01-01"
    val_end: str = "2023-12-31"
    test_start: str = "2024-01-01"
    test_end: str = "2025-01-01"
