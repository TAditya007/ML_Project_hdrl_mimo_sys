"""
Metrics tracking, logging, and visualization for HDRL training.

Provides:
    - MetricsTracker:  accumulate per-episode metrics, running averages
    - EpisodeLogger:   write per-episode CSV (report-ready)
    - TensorBoardWriter:  optional live training curves
    - plot_training_curves():  PNGs from logged metrics
    - save_run_summary():  aggregated JSON summary

Design:
    - CSV is the source of truth — easy to inspect, no framework lock-in
    - TensorBoard is optional (only used if available)
    - Plots are saved as PNG for direct report inclusion
"""

from __future__ import annotations

import os
import json
import csv
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Sequence

import numpy as np


# ============================================================================
# METRICS TRACKER
# ============================================================================

@dataclass
class EpisodeMetrics:
    """Metrics for a single training episode."""
    
    episode: int
    total_reward: float = 0.0
    avg_reward: float = 0.0
    sum_rate: float = 0.0
    num_active_beams: float = 0.0
    num_active_subarrays: float = 0.0
    interference_power: float = 0.0
    
    # Loss / training signal (optional)
    high_level_loss: float = 0.0
    low_level_loss: float = 0.0
    
    # Timing
    wall_time_s: float = 0.0
    steps: int = 0
    
    # Exploration
    epsilon: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MetricsTracker:
    """
    Accumulates metrics during a training run.
    
    Usage:
        tracker = MetricsTracker()
        for episode in range(N):
            tracker.start_episode(episode)
            for step in env:
                tracker.add_step(reward, info)
            tracker.end_episode(loss_hl=..., loss_ll=..., epsilon=...)
        
        df = tracker.to_dataframe()   # optional (needs pandas)
        tracker.save_csv("runs/exp1/metrics.csv")
    """
    
    def __init__(self,
                 window_size: int = 50,
                 verbose: bool = True):
        """
        Args:
            window_size: rolling average window for smoothed metrics
            verbose: print episode lines during training
        """
        self.window_size = window_size
        self.verbose = verbose
        
        # All completed episodes
        self.episodes: List[EpisodeMetrics] = []
        
        # Rolling window buffers
        self._reward_window: deque[float] = deque(maxlen=window_size)
        self._rate_window: deque[float] = deque(maxlen=window_size)
        
        # Current episode (in progress)
        self._current: Optional[EpisodeMetrics] = None
        self._episode_start_time: float = 0.0
        self._episode_rewards: List[float] = []
        self._episode_rates: List[float] = []
        self._episode_beams: List[int] = []
        self._episode_subarrays: List[int] = []
        self._episode_interference: List[float] = []
    
    # ------------------------------------------------------------------------
    # EPISODE LIFECYCLE
    # ------------------------------------------------------------------------
    
    def start_episode(self, episode: int) -> None:
        """Begin a new episode."""
        self._current = EpisodeMetrics(episode=episode)
        self._episode_start_time = time.time()
        self._episode_rewards = []
        self._episode_rates = []
        self._episode_beams = []
        self._episode_subarrays = []
        self._episode_interference = []
    
    def add_step(self, reward: float, info: Dict[str, Any]) -> None:
        """Record a step within the episode."""
        self._episode_rewards.append(float(reward))
        if "sum_rate" in info:
            self._episode_rates.append(float(info["sum_rate"]))
        if "num_active_beams" in info:
            self._episode_beams.append(int(info["num_active_beams"]))
        if "num_active_subarrays" in info:
            self._episode_subarrays.append(int(info["num_active_subarrays"]))
        if "interference_power" in info:
            self._episode_interference.append(float(info["interference_power"]))
    
    def end_episode(self,
                    high_level_loss: float = 0.0,
                    low_level_loss: float = 0.0,
                    epsilon: float = 0.0) -> EpisodeMetrics:
        """Finalize the current episode."""
        if self._current is None:
            raise RuntimeError("No episode in progress. Call start_episode first.")
        
        em = self._current
        em.total_reward = float(np.sum(self._episode_rewards))
        em.avg_reward = float(np.mean(self._episode_rewards)) if self._episode_rewards else 0.0
        em.sum_rate = float(np.mean(self._episode_rates)) if self._episode_rates else 0.0
        em.num_active_beams = float(np.mean(self._episode_beams)) if self._episode_beams else 0.0
        em.num_active_subarrays = float(np.mean(self._episode_subarrays)) if self._episode_subarrays else 0.0
        em.interference_power = float(np.mean(self._episode_interference)) if self._episode_interference else 0.0
        em.high_level_loss = float(high_level_loss)
        em.low_level_loss = float(low_level_loss)
        em.epsilon = float(epsilon)
        em.steps = len(self._episode_rewards)
        em.wall_time_s = time.time() - self._episode_start_time
        
        self.episodes.append(em)
        self._reward_window.append(em.avg_reward)
        self._rate_window.append(em.sum_rate)
        self._current = None
        
        if self.verbose:
            print(
                f"  Ep {em.episode:4d} | "
                f"reward={em.avg_reward:+.4f} | "
                f"rate={em.sum_rate:6.2f} | "
                f"beams={em.num_active_beams:.1f} | "
                f"subs={em.num_active_subarrays:.1f} | "
                f"ε={em.epsilon:.3f} | "
                f"HL_loss={em.high_level_loss:.4f} | "
                f"LL_loss={em.low_level_loss:.4f} | "
                f"{em.wall_time_s:.1f}s"
            )
        
        return em
    
    # ------------------------------------------------------------------------
    # ROLLING STATS
    # ------------------------------------------------------------------------
    
    def rolling_avg_reward(self) -> float:
        """Rolling average of episode rewards."""
        if not self._reward_window:
            return 0.0
        return float(np.mean(self._reward_window))
    
    def rolling_avg_rate(self) -> float:
        """Rolling average of episode rates."""
        if not self._rate_window:
            return 0.0
        return float(np.mean(self._rate_window))
    
    def last_n_episodes(self, n: int) -> List[EpisodeMetrics]:
        """Last n completed episodes."""
        return self.episodes[-n:] if n > 0 else []
    
    # ------------------------------------------------------------------------
    # EXPORT
    # ------------------------------------------------------------------------
    
    def to_dict(self) -> Dict[str, Any]:
        """Full tracked data as a plain dict."""
        return {
            "num_episodes": len(self.episodes),
            "window_size": self.window_size,
            "rolling_avg_reward": self.rolling_avg_reward(),
            "rolling_avg_rate": self.rolling_avg_rate(),
            "episodes": [e.to_dict() for e in self.episodes],
        }
    
    def save_csv(self, path: str) -> None:
        """Save per-episode metrics as CSV."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if not self.episodes:
            return
        
        fieldnames = list(self.episodes[0].to_dict().keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for ep in self.episodes:
                writer.writerow(ep.to_dict())
    
    def save_json(self, path: str) -> None:
        """Save full metrics (including rolling stats) as JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# ============================================================================
# TENSORBOARD WRITER (OPTIONAL)
# ============================================================================

class TensorBoardWriter:
    """
    Optional TensorBoard integration.
    
    Gracefully degrades if `torch.utils.tensorboard` or `tensorboardX` is
    not installed — all methods become no-ops.
    """
    
    def __init__(self, log_dir: str, enabled: bool = True):
        self.log_dir = log_dir
        self.enabled = enabled
        self._writer: Optional[Any] = None
        
        if not enabled:
            return
        
        # Try torch.utils.tensorboard first (comes with PyTorch)
        try:
            from torch.utils.tensorboard import SummaryWriter
            os.makedirs(log_dir, exist_ok=True)
            self._writer = SummaryWriter(log_dir=log_dir)
            return
        except Exception:
            pass
        
        # Fallback: tensorboardX
        try:
            from tensorboardX import SummaryWriter  # type: ignore
            os.makedirs(log_dir, exist_ok=True)
            self._writer = SummaryWriter(log_dir=log_dir)
            return
        except Exception:
            print("[TensorBoardWriter] TensorBoard not available — logging disabled")
            self.enabled = False
    
    def add_scalar(self, tag: str, value: float, step: int) -> None:
        if self._writer is not None:
            self._writer.add_scalar(tag, value, step)
    
    def add_episode(self, em: EpisodeMetrics) -> None:
        if self._writer is None:
            return
        self._writer.add_scalar("reward/avg", em.avg_reward, em.episode)
        self._writer.add_scalar("reward/total", em.total_reward, em.episode)
        self._writer.add_scalar("rate/sum_rate", em.sum_rate, em.episode)
        self._writer.add_scalar("beams/num_active", em.num_active_beams, em.episode)
        self._writer.add_scalar("subarrays/num_active", em.num_active_subarrays, em.episode)
        self._writer.add_scalar("interference/power", em.interference_power, em.episode)
        self._writer.add_scalar("loss/high_level", em.high_level_loss, em.episode)
        self._writer.add_scalar("loss/low_level", em.low_level_loss, em.episode)
        self._writer.add_scalar("exploration/epsilon", em.epsilon, em.episode)
        self._writer.add_scalar("timing/wall_time_s", em.wall_time_s, em.episode)
    
    def flush(self) -> None:
        if self._writer is not None:
            self._writer.flush()
    
    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None


# ============================================================================
# PLOTTING
# ============================================================================

def plot_training_curves(metrics: MetricsTracker,
                          save_dir: str,
                          window: int = 20,
                          prefix: str = "") -> List[str]:
    """
    Generate training curve PNGs from tracked metrics.
    
    Produces:
        - reward.png:      avg reward per episode + smoothed
        - rate.png:        sum-rate per episode + smoothed
        - actions.png:     active beams/subarrays per episode
        - losses.png:      HL/LL loss curves
        - epsilon.png:     exploration schedule
    
    Args:
        metrics: MetricsTracker with completed episodes
        save_dir: Directory to save PNGs
        window: Smoothing window for plots
        prefix: Optional filename prefix
    
    Returns:
        List of PNG paths created
    """
    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot_training_curves] matplotlib not available — skipping")
        return []
    
    os.makedirs(save_dir, exist_ok=True)
    
    if not metrics.episodes:
        print("[plot_training_curves] No episodes to plot")
        return []
    
    eps = np.array([e.episode for e in metrics.episodes])
    rewards = np.array([e.avg_reward for e in metrics.episodes])
    rates = np.array([e.sum_rate for e in metrics.episodes])
    beams = np.array([e.num_active_beams for e in metrics.episodes])
    subs = np.array([e.num_active_subarrays for e in metrics.episodes])
    hl_loss = np.array([e.high_level_loss for e in metrics.episodes])
    ll_loss = np.array([e.low_level_loss for e in metrics.episodes])
    epsilons = np.array([e.epsilon for e in metrics.episodes])
    
    def _smooth(x: np.ndarray, w: int) -> np.ndarray:
        if w <= 1 or len(x) < w:
            return x
        kernel = np.ones(w) / w
        return np.convolve(x, kernel, mode="valid")
    
    def _smoothed_x(x: np.ndarray, w: int) -> np.ndarray:
        if w <= 1 or len(x) < w:
            return x
        # center-aligned
        offset = w // 2
        return x[offset:offset + len(x) - w + 1]
    
    paths: List[str] = []
    
    # ---------- Reward ----------
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(eps, rewards, alpha=0.35, label="raw")
    if len(rewards) >= window:
        ax.plot(_smoothed_x(eps, window), _smooth(rewards, window),
                linewidth=2, label=f"smoothed (w={window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg reward per step")
    ax.set_title("Training reward")
    ax.grid(True, alpha=0.3)
    ax.legend()
    p = os.path.join(save_dir, f"{prefix}reward.png")
    plt.tight_layout(); plt.savefig(p); plt.close(fig)
    paths.append(p)
    
    # ---------- Sum-rate ----------
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(eps, rates, alpha=0.35, label="raw")
    if len(rates) >= window:
        ax.plot(_smoothed_x(eps, window), _smooth(rates, window),
                linewidth=2, label=f"smoothed (w={window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Sum-rate (b/s/Hz)")
    ax.set_title("Sum-rate per episode")
    ax.grid(True, alpha=0.3)
    ax.legend()
    p = os.path.join(save_dir, f"{prefix}rate.png")
    plt.tight_layout(); plt.savefig(p); plt.close(fig)
    paths.append(p)
    
    # ---------- Actions ----------
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(eps, beams, alpha=0.6, label="active beams")
    ax.plot(eps, subs, alpha=0.6, label="active subarrays")
    if len(beams) >= window:
        ax.plot(_smoothed_x(eps, window), _smooth(beams, window),
                linewidth=2, label=f"beams (smoothed)")
        ax.plot(_smoothed_x(eps, window), _smooth(subs, window),
                linewidth=2, label=f"subarrays (smoothed)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Count")
    ax.set_title("Active resources per episode")
    ax.grid(True, alpha=0.3)
    ax.legend()
    p = os.path.join(save_dir, f"{prefix}actions.png")
    plt.tight_layout(); plt.savefig(p); plt.close(fig)
    paths.append(p)
    
    # ---------- Losses ----------
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(eps, hl_loss, alpha=0.6, label="High-level loss")
    ax.plot(eps, ll_loss, alpha=0.6, label="Low-level loss")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Loss")
    ax.set_title("Agent losses")
    ax.grid(True, alpha=0.3)
    ax.legend()
    p = os.path.join(save_dir, f"{prefix}losses.png")
    plt.tight_layout(); plt.savefig(p); plt.close(fig)
    paths.append(p)
    
    # ---------- Epsilon ----------
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(eps, epsilons, linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("ε")
    ax.set_title("Exploration schedule")
    ax.grid(True, alpha=0.3)
    p = os.path.join(save_dir, f"{prefix}epsilon.png")
    plt.tight_layout(); plt.savefig(p); plt.close(fig)
    paths.append(p)
    
    return paths


# ============================================================================
# RUN SUMMARY
# ============================================================================

def save_run_summary(metrics: MetricsTracker,
                      path: str,
                      extra: Optional[Dict[str, Any]] = None) -> None:
    """
    Save a JSON summary of the training run.
    
    Useful for report tables and for cross-run comparison.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    
    eps = metrics.episodes
    if not eps:
        summary: Dict[str, Any] = {"num_episodes": 0}
    else:
        n = len(eps)
        k = max(1, n // 10)   # last 10%
        tail = eps[-k:]
        head = eps[:k]
        
        summary = {
            "num_episodes": n,
            "final_window_avg_reward": float(np.mean([e.avg_reward for e in tail])),
            "final_window_avg_rate": float(np.mean([e.sum_rate for e in tail])),
            "final_window_avg_beams": float(np.mean([e.num_active_beams for e in tail])),
            "final_window_avg_subarrays": float(np.mean([e.num_active_subarrays for e in tail])),
            "initial_window_avg_reward": float(np.mean([e.avg_reward for e in head])),
            "initial_window_avg_rate": float(np.mean([e.sum_rate for e in head])),
            "rolling_avg_reward": metrics.rolling_avg_reward(),
            "rolling_avg_rate": metrics.rolling_avg_rate(),
            "total_wall_time_s": float(np.sum([e.wall_time_s for e in eps])),
        }
    
    if extra is not None:
        summary["extra"] = extra
    
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)


def print_run_summary(metrics: MetricsTracker) -> None:
    """Print a human-readable run summary."""
    eps = metrics.episodes
    print("=" * 60)
    print("Training Run Summary")
    print("=" * 60)
    if not eps:
        print("  No episodes completed.")
        return
    
    n = len(eps)
    k = max(1, n // 10)
    head = eps[:k]
    tail = eps[-k:]
    
    print(f"  Episodes:          {n}")
    print(f"  Total wall time:   {sum(e.wall_time_s for e in eps):.1f} s")
    print()
    print(f"  Initial (first {k}):")
    print(f"    avg reward:      {np.mean([e.avg_reward for e in head]):+.4f}")
    print(f"    avg rate:        {np.mean([e.sum_rate for e in head]):6.3f} b/s/Hz")
    print(f"    avg beams:       {np.mean([e.num_active_beams for e in head]):5.2f}")
    print(f"    avg subarrays:   {np.mean([e.num_active_subarrays for e in head]):5.2f}")
    print()
    print(f"  Final (last {k}):")
    print(f"    avg reward:      {np.mean([e.avg_reward for e in tail]):+.4f}")
    print(f"    avg rate:        {np.mean([e.sum_rate for e in tail]):6.3f} b/s/Hz")
    print(f"    avg beams:       {np.mean([e.num_active_beams for e in tail]):5.2f}")
    print(f"    avg subarrays:   {np.mean([e.num_active_subarrays for e in tail]):5.2f}")
    print("=" * 60)


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Metrics Module Test")
    print("=" * 60)
    
    # ------------------------------------------------------------------
    # Test 1: Basic tracker lifecycle
    # ------------------------------------------------------------------
    print("\n[Test 1] MetricsTracker lifecycle")
    tracker = MetricsTracker(window_size=5, verbose=False)
    
    rng = np.random.default_rng(42)
    for ep in range(20):
        tracker.start_episode(ep)
        for step in range(5):
            reward = rng.normal(0.3, 0.1) + 0.02 * ep
            info = {
                "sum_rate": 15.0 + 0.5 * ep + rng.normal(0, 0.5),
                "num_active_beams": 5 + ep % 3,
                "num_active_subarrays": 3 + ep % 3,
                "interference_power": 0.5 + 0.05 * ep,
            }
            tracker.add_step(reward, info)
        em = tracker.end_episode(
            high_level_loss=1.0 / (1 + ep),
            low_level_loss=2.0 / (1 + ep),
            epsilon=max(0.05, 1.0 - ep / 20),
        )
    
    print(f"  Episodes tracked: {len(tracker.episodes)}")
    print(f"  Rolling avg reward: {tracker.rolling_avg_reward():+.4f}")
    print(f"  Rolling avg rate:   {tracker.rolling_avg_rate():.3f}")
    assert len(tracker.episodes) == 20
    print("  ✓ Lifecycle works")
    
    # ------------------------------------------------------------------
    # Test 2: CSV export
    # ------------------------------------------------------------------
    print("\n[Test 2] CSV export")
    out_dir = "runs/test_metrics"
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "metrics.csv")
    tracker.save_csv(csv_path)
    assert os.path.exists(csv_path)
    with open(csv_path) as f:
        lines = f.readlines()
    print(f"  CSV lines: {len(lines)} (expect 1 header + 20 rows)")
    assert len(lines) == 21
    print(f"  Header: {lines[0].strip()[:80]}...")
    print("  ✓ CSV exported")
    
    # ------------------------------------------------------------------
    # Test 3: JSON export
    # ------------------------------------------------------------------
    print("\n[Test 3] JSON export")
    json_path = os.path.join(out_dir, "metrics.json")
    tracker.save_json(json_path)
    assert os.path.exists(json_path)
    with open(json_path) as f:
        data = json.load(f)
    assert data["num_episodes"] == 20
    print(f"  JSON keys: {sorted(data.keys())[:5]}...")
    print("  ✓ JSON exported")
    
    # ------------------------------------------------------------------
    # Test 4: Run summary JSON
    # ------------------------------------------------------------------
    print("\n[Test 4] Run summary JSON")
    summary_path = os.path.join(out_dir, "summary.json")
    save_run_summary(tracker, summary_path, extra={"experiment": "test"})
    with open(summary_path) as f:
        summary = json.load(f)
    print(f"  Summary keys: {sorted(summary.keys())}")
    assert "final_window_avg_reward" in summary
    assert "final_window_avg_rate" in summary
    print(f"  Final avg reward: {summary['final_window_avg_reward']:+.4f}")
    print(f"  Final avg rate:   {summary['final_window_avg_rate']:.3f}")
    print("  ✓ Run summary saved")
    
    # ------------------------------------------------------------------
    # Test 5: Plots
    # ------------------------------------------------------------------
    print("\n[Test 5] Plot generation")
    plot_dir = os.path.join(out_dir, "plots")
    paths = plot_training_curves(tracker, plot_dir, window=5, prefix="test_")
    print(f"  Generated {len(paths)} PNG files:")
    for p in paths:
        assert os.path.exists(p)
        print(f"    {os.path.basename(p)}")
    assert len(paths) == 5
    print("  ✓ Plots generated")
    
    # ------------------------------------------------------------------
    # Test 6: TensorBoard writer (optional, no crash)
    # ------------------------------------------------------------------
    print("\n[Test 6] TensorBoard writer (optional)")
    tb_dir = os.path.join(out_dir, "tb")
    tb = TensorBoardWriter(tb_dir, enabled=True)
    print(f"  TensorBoard available: {tb._writer is not None}")
    for em in tracker.episodes:
        tb.add_episode(em)
    tb.flush()
    tb.close()
    print("  ✓ TensorBoard writer callable (degrades gracefully if absent)")
    
    # ------------------------------------------------------------------
    # Test 7: Print summary
    # ------------------------------------------------------------------
    print("\n[Test 7] Print run summary")
    print_run_summary(tracker)
    
    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    import shutil
    shutil.rmtree(out_dir, ignore_errors=True)
    
    print("\n" + "=" * 60)
    print("✓ Metrics module validation successful!")
    print("=" * 60)

    