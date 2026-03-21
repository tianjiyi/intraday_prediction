"""PPO training loop with EvalCallback for checkpoint selection."""

import logging
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback

from .config import RLConfig
from .env import DipBuyEnv

logger = logging.getLogger(__name__)

_root = Path(__file__).resolve().parents[3]
_out_dir = _root / "data" / "backtests" / "dayx_dip_rl"


class CurriculumCallback(BaseCallback):
    """Decay exploration_rate from initial value to 0 over decay_steps."""

    def __init__(self, decay_steps: int = 500_000, verbose: int = 0):
        super().__init__(verbose)
        self.decay_steps = decay_steps

    def _on_step(self) -> bool:
        progress = min(self.num_timesteps / self.decay_steps, 1.0)
        new_rate = 0.3 * (1.0 - progress)
        # Update the underlying env inside DummyVecEnv
        for env in self.training_env.envs:
            env.exploration_rate = new_rate
        if self.num_timesteps % 50_000 == 0:
            logger.info(f"Curriculum: exploration_rate = {new_rate:.3f} at step {self.num_timesteps}")
        return True


def train(cfg: RLConfig, seed: int = 42):
    """Train PPO agent on train split, checkpoint on val split."""
    model_dir = _out_dir / "models"
    log_dir = _out_dir / "logs"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create environments
    train_env = DummyVecEnv([lambda: DipBuyEnv(cfg, mode="train", seed=seed)])
    val_env = DummyVecEnv([lambda: DipBuyEnv(cfg, mode="val", seed=seed + 1)])

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=cfg.learning_rate,
        n_steps=4096,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        ent_coef=0.05,
        verbose=1,
        seed=seed,
        device="cpu",
        policy_kwargs=dict(net_arch=[128, 128]),
        tensorboard_log=str(log_dir),
    )

    eval_cb = EvalCallback(
        val_env,
        eval_freq=10_000,
        n_eval_episodes=50,
        best_model_save_path=str(model_dir),
        log_path=str(log_dir),
        deterministic=True,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=str(model_dir / "checkpoints"),
        name_prefix="ppo_dip",
    )

    curriculum_cb = CurriculumCallback(decay_steps=500_000)

    logger.info(f"Training PPO for {cfg.total_timesteps} timesteps...")
    logger.info(f"  Train: {cfg.train_start} → {cfg.train_end}")
    logger.info(f"  Val:   {cfg.val_start} → {cfg.val_end}")
    logger.info(f"  Curriculum: exploration 0.3 → 0.0 over 500K steps")

    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=[eval_cb, checkpoint_cb, curriculum_cb],
        progress_bar=True,
    )

    final_path = model_dir / "final_model"
    model.save(str(final_path))
    logger.info(f"Final model saved to {final_path}")

    return model
