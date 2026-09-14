"""
Hierarchical DRL training loop for XL-MIMO subarray + beam selection.

Ties together:
    - XLMIMOEnv (environment)
    - HighLevelDQNAgent (subarray selection)
    - LowLevelDQNAgent (per-subarray beam selection)
    - MetricsTracker (per-episode logging)

Training loop (per episode):
    1. env.reset() -> obs
    2. Loop until truncated:
        a. HL picks subarray mask from obs
        b. LL picks beam indices from (obs, mask)
        c. env.step({mask, beams}) -> reward, next_obs
        d. Store transitions in BOTH agents' replay buffers
        e. Train HL and LL (Double DQN, one step each)
    3. Log episode metrics

Usage (from CLI):
    python -m src.training.loop --config configs/default.yaml
    python -m src.training.loop --config configs/default.yaml --experiment configs/experiments/exp1_subarray.yaml

Usage (from Python):
    from src.training.loop import train_hdrl
    results = train_hdrl(cfg)
"""

from __future__ import annotations

import os
import time
import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.utils.config import ExperimentConfig
from src.utils.metrics import (
    MetricsTracker, TensorBoardWriter, save_run_summary,
    plot_training_curves, print_run_summary,
)
from src.environment.xl_mimo_env import XLMIMOEnv
from src.agents.high_level import HighLevelDQNAgent
from src.agents.low_level import LowLevelDQNAgent


# ============================================================================
# TRAINING RESULTS
# ============================================================================

@dataclass
class TrainingResults:
    """Summary of a completed training run."""

    experiment_name: str
    num_episodes: int
    total_wall_time_s: float
    final_avg_reward: float
    final_avg_rate: float
    final_avg_beams: float
    final_avg_subarrays: float
    initial_avg_reward: float
    initial_avg_rate: float

    checkpoint_path: str = ""
    metrics_csv: str = ""
    summary_json: str = ""
    plot_dir: str = ""

    def summary(self) -> str:
        return "\n".join([
            "=" * 60,
            f"Training Results: {self.experiment_name}",
            "=" * 60,
            f"  Episodes:         {self.num_episodes}",
            f"  Wall time:        {self.total_wall_time_s:.1f} s",
            f"  Initial reward:   {self.initial_avg_reward:+.4f}",
            f"  Final reward:     {self.final_avg_reward:+.4f}",
            f"  Reward delta:     {self.final_avg_reward - self.initial_avg_reward:+.4f}",
            f"  Initial rate:     {self.initial_avg_rate:.3f} b/s/Hz",
            f"  Final rate:       {self.final_avg_rate:.3f} b/s/Hz",
            f"  Final beams:      {self.final_avg_beams:.2f}",
            f"  Final subarrays:  {self.final_avg_subarrays:.2f}",
            "=" * 60,
        ])


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_hdrl(
    cfg: ExperimentConfig,
    verbose: bool = True,
    enable_tensorboard: bool = False,
) -> TrainingResults:
    """
    Train the full HDRL system on the given config.

    Args:
        cfg: ExperimentConfig with system, rl, training sections
        verbose: print per-episode progress
        enable_tensorboard: write TB event files (optional)

    Returns:
        TrainingResults with paths and final metrics
    """
    t_start = time.time()

    # ------------------------------------------------------------------
    # SETUP
    # ------------------------------------------------------------------
    if verbose:
        print("=" * 60)
        print(f"Training HDRL: {cfg.experiment_name}")
        print("=" * 60)
        print(f"  Antennas:     {cfg.system.num_antennas}")
        print(f"  Subarrays:    {cfg.system.num_subarrays}")
        print(f"  Users:        {cfg.system.num_users}")
        print(f"  Device:       {cfg.rl.device}")
        print(f"  Episodes:     {cfg.training.num_episodes}")
        print(f"  Steps/ep:     {cfg.training.max_steps_per_episode}")
        print("=" * 60)

    # Build env
    env = XLMIMOEnv(config=cfg)
    # Resolve observation dimension explicitly (Pylance-safe).
    # env.observation_space.shape is typed as Optional[tuple[int, ...]]
    # by Gymnasium stubs. Extract and validate before use.
    obs_shape = env.observation_space.shape
    assert obs_shape is not None, "observation_space.shape is unexpectedly None"
    obs_dim: int = int(obs_shape[0])
    

    # Build agents
    hl_agent = HighLevelDQNAgent(
        obs_dim=obs_dim,
        num_subarrays=env.num_subarrays,
        learning_rate=cfg.rl.high_level_learning_rate,
        gamma=cfg.rl.high_level_gamma,
        hidden_dim=cfg.rl.high_level_hidden_dim,
        buffer_size=cfg.rl.high_level_replay_buffer_size,
        batch_size=cfg.rl.high_level_batch_size,
        target_update_freq=cfg.rl.high_level_target_update_freq,
        epsilon_start=cfg.rl.high_level_epsilon_start,
        epsilon_end=cfg.rl.high_level_epsilon_end,
        epsilon_decay=cfg.rl.high_level_epsilon_decay,
        device=cfg.rl.device,
        seed=cfg.system.seed,
    )

    ll_agent = LowLevelDQNAgent(
        obs_dim=obs_dim,
        num_subarrays=env.num_subarrays,
        num_beams=env.num_beams,
        learning_rate=cfg.rl.low_level_learning_rate,
        gamma=cfg.rl.low_level_gamma,
        hidden_dim=cfg.rl.low_level_hidden_dim,
        buffer_size=cfg.rl.low_level_replay_buffer_size,
        batch_size=cfg.rl.low_level_batch_size,
        target_update_freq=cfg.rl.low_level_target_update_freq,
        epsilon_start=cfg.rl.low_level_epsilon_start,
        epsilon_end=cfg.rl.low_level_epsilon_end,
        epsilon_decay=cfg.rl.low_level_epsilon_decay,
        device=cfg.rl.device,
        seed=cfg.system.seed + 1,
    )

    # Metrics
    tracker = MetricsTracker(
        window_size=min(50, max(5, cfg.training.num_episodes // 5)),
        verbose=verbose,
    )
    tb_writer = TensorBoardWriter(
        log_dir=cfg.training.log_dir,
        enabled=enable_tensorboard,
    )

    # Output directories
    checkpoint_dir = cfg.training.checkpoint_dir
    # Route all outputs to runs/<experiment_name>/
    run_root = os.path.join("runs", cfg.experiment_name)
    metrics_csv_path = os.path.join(run_root, "metrics.csv")
    summary_json_path = os.path.join(run_root, "summary.json")
    plot_dir = os.path.join(run_root, "plots")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(run_root, exist_ok=True)

    # ------------------------------------------------------------------
    # TRAINING LOOP
    # ------------------------------------------------------------------
    if verbose:
        print("\nStarting training...\n")

    for episode in range(cfg.training.num_episodes):
        tracker.start_episode(episode)

        obs, _ = env.reset(seed=cfg.system.seed + episode)

        hl_losses = []
        ll_losses = []
        last_info: dict = {}

        for step in range(cfg.training.max_steps_per_episode):
            # ---- High-level action: subarray mask ----
            hl_action = hl_agent.select_action(obs, evaluate=False)

            # ---- Low-level action: beam indices ----
            ll_action = ll_agent.select_action(obs, hl_action, evaluate=False)

            # ---- Combine and step env ----
            env_action = {
                "subarray_mask": hl_action,
                "beam_indices": ll_action,
            }
            next_obs, reward, terminated, truncated, info = env.step(env_action)
            last_info = info
            tracker.add_step(reward, info)     # <-- NEW: log per-step metrics

            # ---- Next HL action (for bootstrapping LL's next_mask) ----
            next_hl_action = hl_agent.select_action(next_obs, evaluate=True)

            done = bool(terminated or truncated)

            # ---- Store transitions in both agents ----
            hl_agent.store(obs, hl_action, reward, next_obs, done)
            ll_agent.store(
                obs, hl_action, ll_action, reward,
                next_obs, next_hl_action, done,
            )

            # ---- Train both agents ----
            hl_loss = hl_agent.train_step()
            ll_loss = ll_agent.train_step()
            if hl_loss is not None:
                hl_losses.append(hl_loss)
            if ll_loss is not None:
                ll_losses.append(ll_loss)

            # ---- Advance ----
            obs = next_obs
            if done:
                break

        # ---- End episode ----
        mean_hl_loss = float(np.mean(hl_losses)) if hl_losses else 0.0
        mean_ll_loss = float(np.mean(ll_losses)) if ll_losses else 0.0

        em = tracker.end_episode(
            high_level_loss=mean_hl_loss,
            low_level_loss=mean_ll_loss,
            epsilon=hl_agent.epsilon,
        )
        tb_writer.add_episode(em)

        # ---- Periodic checkpoint ----
        if ((episode + 1) % cfg.training.save_frequency == 0
                or episode == cfg.training.num_episodes - 1):
            ckpt_base = os.path.join(
                checkpoint_dir,
                f"{cfg.experiment_name}_ep{episode + 1:04d}.pt",
            )
            hl_agent.save(ckpt_base + ".hl")
            ll_agent.save(ckpt_base + ".ll")
            if verbose:
                print(f"    [checkpoint saved: ep {episode + 1}]")

    # ------------------------------------------------------------------
    # SAVE OUTPUTS
    # ------------------------------------------------------------------
    if verbose:
        print("\n" + "=" * 60)
        print("Saving outputs...")

    tracker.save_csv(metrics_csv_path)
    save_run_summary(
        tracker, summary_json_path,
        extra={
            "experiment_name": cfg.experiment_name,
            "num_antennas": cfg.system.num_antennas,
            "num_subarrays": cfg.system.num_subarrays,
            "num_users": cfg.system.num_users,
        },
    )

    plot_paths = plot_training_curves(
        tracker, plot_dir,
        window=max(5, cfg.training.num_episodes // 20),
        prefix="",
    )

    tb_writer.flush()
    tb_writer.close()

    if verbose:
        print(f"  metrics:  {metrics_csv_path}")
        print(f"  summary:  {summary_json_path}")
        print(f"  plots:    {plot_dir}  ({len(plot_paths)} PNGs)")

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------
    total_time = time.time() - t_start

    eps = tracker.episodes
    n = max(1, len(eps))
    k = max(1, n // 10)
    tail = eps[-k:]
    head = eps[:k]

    results = TrainingResults(
        experiment_name=cfg.experiment_name,
        num_episodes=len(eps),
        total_wall_time_s=total_time,
        final_avg_reward=float(np.mean([e.avg_reward for e in tail])),
        final_avg_rate=float(np.mean([e.sum_rate for e in tail])),
        final_avg_beams=float(np.mean([e.num_active_beams for e in tail])),
        final_avg_subarrays=float(np.mean([e.num_active_subarrays for e in tail])),
        initial_avg_reward=float(np.mean([e.avg_reward for e in head])),
        initial_avg_rate=float(np.mean([e.sum_rate for e in head])),
        checkpoint_path=checkpoint_dir,
        metrics_csv=metrics_csv_path,
        summary_json=summary_json_path,
        plot_dir=plot_dir,
    )

    if verbose:
        print("\n" + results.summary())
        print_run_summary(tracker)

    env.close()
    return results


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train HDRL for XL-MIMO subarray + beam selection"
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to default config YAML",
    )
    parser.add_argument(
        "--experiment", type=str, default=None,
        help="Path to experiment override YAML (optional)",
    )
    parser.add_argument(
        "--episodes", type=int, default=None,
        help="Override num_episodes (for quick tests)",
    )
    parser.add_argument(
        "--tensorboard", action="store_true",
        help="Enable TensorBoard event logging",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Load config
    cfg = ExperimentConfig.from_yaml(
        default_path=args.config,
        experiment_path=args.experiment,
    )

    # Optional CLI override for quick testing
    if args.episodes is not None:
        cfg.training.num_episodes = int(args.episodes)
        cfg.training.save_frequency = max(1, cfg.training.num_episodes // 5)
        cfg.training.log_frequency = max(1, cfg.training.num_episodes // 10)
        print(f"[override] num_episodes -> {cfg.training.num_episodes}")

    # Train
    results = train_hdrl(
        cfg,
        verbose=True,
        enable_tensorboard=args.tensorboard,
    )

    print("\n" + results.summary()) 