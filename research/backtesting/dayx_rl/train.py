"""PPO training loop for signal filter RL agent."""

import logging
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from .config import RLFilterConfig
from .env import SignalFilterEnv

logger = logging.getLogger(__name__)

_root = Path(__file__).resolve().parents[3]
_out_dir = _root / "data" / "backtests" / "dayx_rl"


def train(cfg: RLFilterConfig, seed: int = 42):
    """Train PPO agent to filter signals, checkpoint on val split."""
    model_dir = _out_dir / "models"
    log_dir = _out_dir / "logs"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    train_env = DummyVecEnv([lambda: SignalFilterEnv(cfg, mode="train", seed=seed)])
    val_env = DummyVecEnv([lambda: SignalFilterEnv(cfg, mode="val", seed=seed + 1)])

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        ent_coef=cfg.ent_coef,
        verbose=1,
        seed=seed,
        device="cpu",
        policy_kwargs=dict(net_arch=[128, 128]),
        tensorboard_log=str(log_dir),
    )

    eval_cb = EvalCallback(
        val_env,
        eval_freq=5_000,
        n_eval_episodes=50,
        best_model_save_path=str(model_dir),
        log_path=str(log_dir),
        deterministic=True,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=25_000,
        save_path=str(model_dir / "checkpoints"),
        name_prefix="ppo_filter",
    )

    logger.info(f"Training PPO signal filter for {cfg.total_timesteps} timesteps...")
    logger.info(f"  Train: {cfg.train_start} → {cfg.train_end}")
    logger.info(f"  Val:   {cfg.val_start} → {cfg.val_end}")

    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=[eval_cb, checkpoint_cb],
        progress_bar=True,
    )

    final_path = model_dir / "final_model"
    model.save(str(final_path))
    logger.info(f"Final model saved to {final_path}")

    return model
