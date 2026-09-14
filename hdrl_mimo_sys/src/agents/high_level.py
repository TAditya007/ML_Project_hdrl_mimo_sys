"""
High-level DQN agent for subarray selection in XL-MIMO.

The agent takes the environment observation and outputs a binary
subarray activation mask. Action structure is Bernoulli: the network
produces K logits, one per subarray, and each bit is sampled
independently during exploration.

Training uses Double DQN:
    - Online network selects the greedy action
    - Target network evaluates that action
    - Reduces Q-value overestimation vs vanilla DQN

Usage:
    agent = HighLevelDQNAgent(obs_dim=26, num_subarrays=8)
    action = agent.select_action(obs)          # int8 array of shape (K,)
    agent.store(obs, action, reward, next_obs, done)
    loss = agent.train_step()                  # None if buffer not ready
    agent.update_epsilon(step)
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ============================================================================
# NETWORK
# ============================================================================

class HighLevelQNet(nn.Module):
    """
    Q-network for Bernoulli multi-binary subarray selection.
    
    Architecture:
        obs -> MLP -> K logits -> Q-values per bit
    
    Note on Q-values: for a Bernoulli action head, the "Q-value" for
    bit k is the expected return assuming bit k = 1. Standard DQN
    update rules apply per-bit (with shared replay).
    """
    
    def __init__(self,
                 obs_dim: int,
                 num_subarrays: int,
                 hidden_dim: int = 128,
                 num_layers: int = 2):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_subarrays = num_subarrays
        self.hidden_dim = hidden_dim
        
        layers: List[nn.Module] = []
        in_dim = obs_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, num_subarrays))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (batch, obs_dim) or (obs_dim,)
        Returns:
            (batch, K) or (K,) logits
        """
        return self.net(obs)


# ============================================================================
# REPLAY BUFFER
# ============================================================================

@dataclass
class Transition:
    """One (s, a, r, s', done) transition."""
    obs: np.ndarray
    action: np.ndarray       # int8, shape (K,)
    reward: float
    next_obs: np.ndarray
    done: bool


class ReplayBuffer:
    """Fixed-size experience replay buffer."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: deque[Transition] = deque(maxlen=capacity)
    
    def push(self,
             obs: np.ndarray,
             action: np.ndarray,
             reward: float,
             next_obs: np.ndarray,
             done: bool) -> None:
        self.buffer.append(Transition(
            obs=np.asarray(obs, dtype=np.float32).copy(),
            action=np.asarray(action, dtype=np.int8).copy(),
            reward=float(reward),
            next_obs=np.asarray(next_obs, dtype=np.float32).copy(),
            done=bool(done),
        ))
    
    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def ready(self, batch_size: int) -> bool:
        return len(self.buffer) >= batch_size


# ============================================================================
# AGENT
# ============================================================================

class HighLevelDQNAgent:
    """
    High-level DQN agent for subarray selection.
    
    Action structure: Bernoulli over K bits.
        - During training: sample each bit from Bernoulli(sigma(q_k))
        - During eval: threshold at 0.5 (greedy)
        - Epsilon: with prob eps, sample from Bernoulli(0.5)
    """
    
    def __init__(self,
                 obs_dim: int,
                 num_subarrays: int,
                 learning_rate: float = 1e-3,
                 gamma: float = 0.99,
                 hidden_dim: int = 128,
                 buffer_size: int = 10000,
                 batch_size: int = 64,
                 target_update_freq: int = 100,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.05,
                 epsilon_decay: int = 5000,
                 device: str = "cuda",
                 seed: Optional[int] = None):
        self.obs_dim = obs_dim
        self.num_subarrays = num_subarrays
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        
        self._step_count = 0
        
        # Device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        
        # Seeds
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
        
        # Networks
        self.online_net = HighLevelQNet(
            obs_dim=obs_dim,
            num_subarrays=num_subarrays,
            hidden_dim=hidden_dim,
        ).to(self.device)
        
        self.target_net = HighLevelQNet(
            obs_dim=obs_dim,
            num_subarrays=num_subarrays,
            hidden_dim=hidden_dim,
        ).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.online_net.parameters(), lr=learning_rate
        )
        
        # Replay
        self.replay = ReplayBuffer(capacity=buffer_size)
        
        # Diagnostics
        self.last_loss: float = 0.0
        self.updates_done: int = 0
    
    # ========================================================================
    # ACTION SELECTION
    # ========================================================================
    
    @torch.no_grad()
    def select_action(self,
                      obs: np.ndarray,
                      evaluate: bool = False
                      ) -> np.ndarray:
        """
        Select a binary subarray mask.
        
        Args:
            obs: (obs_dim,) observation
            evaluate: if True, use greedy (no exploration)
        
        Returns:
            int8 array of shape (K,) with values 0/1
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        logits = self.online_net(obs_t)                    # (K,)
        probs = torch.sigmoid(logits)                      # (K,) in (0, 1)
        
        if evaluate:
            action = (probs > 0.5).to(torch.int8)
        else:
            if random.random() < self.epsilon:
                action = torch.bernoulli(
                    torch.full_like(probs, 0.5)
                ).to(torch.int8)
            else:
                action = torch.bernoulli(probs).to(torch.int8)
        
        # Ensure at least one active subarray
        if action.sum().item() == 0:
            action[random.randrange(self.num_subarrays)] = 1
        
        return action.cpu().numpy().astype(np.int8)
    
    def update_epsilon(self, step: Optional[int] = None) -> None:
        """
        Update epsilon linearly from start -> end over epsilon_decay steps.
        
        Args:
            step: Global step. If None, uses internal counter.
        """
        # Local non-Optional variable for Pylance type narrowing
        current_step: int = self._step_count if step is None else int(step)
        
        if current_step >= self.epsilon_decay:
            self.epsilon = self.epsilon_end
        else:
            frac = current_step / max(1, self.epsilon_decay)
            self.epsilon = self.epsilon_start + frac * (
                self.epsilon_end - self.epsilon_start
            )
    
    # ========================================================================
    # REPLAY / TRAINING
    # ========================================================================
    
    def store(self,
              obs: np.ndarray,
              action: np.ndarray,
              reward: float,
              next_obs: np.ndarray,
              done: bool) -> None:
        """Push a transition to replay and increment step counter."""
        self.replay.push(obs, action, reward, next_obs, done)
        self._step_count += 1
        self.update_epsilon()
    
    def train_step(self) -> Optional[float]:
        """
        One gradient step of Double DQN.
        
        Returns:
            loss value (float), or None if replay not yet ready.
        """
        if not self.replay.ready(self.batch_size):
            return None
        
        batch = self.replay.sample(self.batch_size)
        
        obs = torch.as_tensor(
            np.stack([t.obs for t in batch]),
            dtype=torch.float32, device=self.device,
        )
        actions = torch.as_tensor(
            np.stack([t.action for t in batch]),
            dtype=torch.float32, device=self.device,
        )
        rewards = torch.as_tensor(
            np.array([t.reward for t in batch]),
            dtype=torch.float32, device=self.device,
        )
        next_obs = torch.as_tensor(
            np.stack([t.next_obs for t in batch]),
            dtype=torch.float32, device=self.device,
        )
        dones = torch.as_tensor(
            np.array([t.done for t in batch], dtype=np.float32),
            dtype=torch.float32, device=self.device,
        )
        
        # Current Q-values per bit
        q_all = self.online_net(obs)                        # (B, K)
        q_taken = (q_all * actions + (1.0 - q_all) * (1.0 - actions))
        q_sum = q_taken.sum(dim=1)                          # (B,)
        
        # Double DQN target
        with torch.no_grad():
            q_next_online = self.online_net(next_obs)
            probs_next = torch.sigmoid(q_next_online)
            greedy_next = (probs_next > 0.5).float()
            
            # Coerce: if all-zero row, set bit 0 to 1 (matches env behavior)
            all_zero = (greedy_next.sum(dim=1) == 0).float().unsqueeze(1)
            greedy_next = torch.where(
                all_zero > 0,
                torch.cat([
                    torch.ones_like(greedy_next[:, :1]),
                    greedy_next[:, 1:],
                ], dim=1),
                greedy_next,
            )
            
            q_target_next_all = self.target_net(next_obs)
            q_target_next = (
                q_target_next_all * greedy_next
                + (1.0 - q_target_next_all) * (1.0 - greedy_next)
            )
            q_target_next_sum = q_target_next.sum(dim=1)
            
            target = rewards + self.gamma * q_target_next_sum * (1.0 - dones)
        
        loss = F.mse_loss(q_sum, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.online_net.parameters(), max_norm=10.0
        )
        self.optimizer.step()
        
        # Target sync
        self.updates_done += 1
        if self.updates_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        
        loss_val = float(loss.item())
        self.last_loss = loss_val
        return loss_val
    
    # ========================================================================
    # CHECKPOINTS
    # ========================================================================
    
    def save(self, path: str) -> None:
        """Save model + optimizer + agent state."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "step_count": self._step_count,
            "updates_done": self.updates_done,
            "obs_dim": self.obs_dim,
            "num_subarrays": self.num_subarrays,
            "hidden_dim": self.hidden_dim,
        }, path)
    
    def load(self, path: str) -> None:
        """
        Load from a checkpoint.
        
        Raises:
            ValueError: if checkpoint architecture doesn't match this agent's.
        """
        ckpt = torch.load(path, map_location=self.device)
        
        # Sanity: architecture must match
        ckpt_obs_dim = ckpt.get("obs_dim")
        ckpt_num_sub = ckpt.get("num_subarrays")
        ckpt_hidden = ckpt.get("hidden_dim")
        
        if ckpt_obs_dim is not None and ckpt_obs_dim != self.obs_dim:
            raise ValueError(
                f"Checkpoint obs_dim={ckpt_obs_dim} != agent "
                f"obs_dim={self.obs_dim}"
            )
        if ckpt_num_sub is not None and ckpt_num_sub != self.num_subarrays:
            raise ValueError(
                f"Checkpoint num_subarrays={ckpt_num_sub} != agent "
                f"num_subarrays={self.num_subarrays}"
            )
        if ckpt_hidden is not None and ckpt_hidden != self.hidden_dim:
            raise ValueError(
                f"Checkpoint hidden_dim={ckpt_hidden} != agent "
                f"hidden_dim={self.hidden_dim}. Construct the agent with "
                f"hidden_dim={ckpt_hidden} to load this checkpoint."
            )
        
        try:
            self.online_net.load_state_dict(ckpt["online_net"])
            self.target_net.load_state_dict(ckpt["target_net"])
        except RuntimeError as e:
            raise ValueError(
                f"Checkpoint architecture mismatch (e.g., hidden_dim differs). "
                f"Create the agent with matching architecture. "
                f"Original error: {e}"
            ) from e
        
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt.get("epsilon", self.epsilon)
        self._step_count = ckpt.get("step_count", 0)
        self.updates_done = ckpt.get("updates_done", 0)
    
    # ========================================================================
    # INFO
    # ========================================================================
    
    def summary(self) -> str:
        return "\n".join([
            f"HighLevelDQNAgent",
            f"  obs_dim:          {self.obs_dim}",
            f"  num_subarrays:    {self.num_subarrays}",
            f"  hidden_dim:       {self.hidden_dim}",
            f"  device:           {self.device}",
            f"  gamma:            {self.gamma}",
            f"  batch_size:       {self.batch_size}",
            f"  replay_size:      {len(self.replay)} / {self.replay.capacity}",
            f"  epsilon:          {self.epsilon:.4f}",
            f"  step_count:       {self._step_count}",
            f"  updates_done:     {self.updates_done}",
            f"  last_loss:        {self.last_loss:.4f}",
        ])


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("High-Level DQN Agent Test")
    print("=" * 60)
    
    OBS_DIM = 26
    K = 8
    BATCH = 16
    HIDDEN = 64
    
    agent = HighLevelDQNAgent(
        obs_dim=OBS_DIM,
        num_subarrays=K,
        learning_rate=1e-3,
        gamma=0.99,
        hidden_dim=HIDDEN,
        buffer_size=500,
        batch_size=BATCH,
        target_update_freq=10,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=200,
        device="cpu",
        seed=42,
    )
    print("\n" + agent.summary())
    
    # ------------------------------------------------------------------
    # Test 1: Action selection
    # ------------------------------------------------------------------
    print("\n[Test 1] Action selection")
    obs = np.random.randn(OBS_DIM).astype(np.float32)
    a_train = agent.select_action(obs, evaluate=False)
    a_eval = agent.select_action(obs, evaluate=True)
    print(f"  train action shape: {a_train.shape}, dtype: {a_train.dtype}, "
          f"sum: {a_train.sum()}")
    print(f"  eval  action shape: {a_eval.shape}, dtype: {a_eval.dtype}, "
          f"sum: {a_eval.sum()}")
    assert a_train.shape == (K,)
    assert a_train.dtype == np.int8
    assert a_eval.shape == (K,)
    assert a_train.sum() >= 1
    assert a_eval.sum() >= 1
    assert set(np.unique(a_train)).issubset({0, 1})
    print("  OK: action selection works")
    
    # ------------------------------------------------------------------
    # Test 2: Store + replay ready
    # ------------------------------------------------------------------
    print("\n[Test 2] Store transitions")
    for _ in range(BATCH):
        o = np.random.randn(OBS_DIM).astype(np.float32)
        a = agent.select_action(o)
        r = float(np.random.randn())
        o_next = np.random.randn(OBS_DIM).astype(np.float32)
        d = bool(np.random.rand() < 0.1)
        agent.store(o, a, r, o_next, d)
    print(f"  Replay size: {len(agent.replay)} (expect {BATCH})")
    print(f"  Ready for training: {agent.replay.ready(BATCH)}")
    assert len(agent.replay) == BATCH
    assert agent.replay.ready(BATCH)
    print("  OK: buffer filled and ready")
    
    # ------------------------------------------------------------------
    # Test 3: train_step returns a loss
    # ------------------------------------------------------------------
    print("\n[Test 3] Training step")
    loss = agent.train_step()
    print(f"  Loss: {loss:.6f}")
    assert loss is not None
    assert np.isfinite(loss)
    assert loss >= 0
    print("  OK: loss computed")
    
    # ------------------------------------------------------------------
    # Test 4: Replay not ready -> None
    # ------------------------------------------------------------------
    print("\n[Test 4] Replay-not-ready case")
    agent_small = HighLevelDQNAgent(
        obs_dim=OBS_DIM, num_subarrays=K,
        hidden_dim=HIDDEN, batch_size=64, device="cpu", seed=0,
    )
    for _ in range(10):
        o = np.random.randn(OBS_DIM).astype(np.float32)
        agent_small.store(o, np.ones(K, dtype=np.int8), 0.5, o, False)
    loss_small = agent_small.train_step()
    print(f"  Loss with 10 transitions (batch=64): {loss_small}")
    assert loss_small is None
    print("  OK: returns None when replay too small")
    
    # ------------------------------------------------------------------
    # Test 5: Loss over 20 training steps
    # ------------------------------------------------------------------
    print("\n[Test 5] Loss over 20 training steps")
    losses = []
    for _ in range(20):
        o = np.random.randn(OBS_DIM).astype(np.float32)
        a = agent.select_action(o)
        r = 0.5
        o_next = np.random.randn(OBS_DIM).astype(np.float32)
        agent.store(o, a, r, o_next, False)
        l = agent.train_step()
        if l is not None:
            losses.append(l)
    print(f"  Collected {len(losses)} losses, range "
          f"[{min(losses):.4f}, {max(losses):.4f}]")
    assert all(np.isfinite(losses))
    print("  OK: training loss stable across steps")
    
    # ------------------------------------------------------------------
    # Test 6: Epsilon decay
    # ------------------------------------------------------------------
    print("\n[Test 6] Epsilon decay")
    for step in [0, 50, 100, 150, 200, 500]:
        agent.update_epsilon(step)
        print(f"  step={step:4d} -> eps={agent.epsilon:.4f}")
    agent.update_epsilon(500)
    assert agent.epsilon == agent.epsilon_end
    print("  OK: epsilon decays to epsilon_end")
    
    # ------------------------------------------------------------------
    # Test 7: Save / load round-trip  (FIXED: pass hidden_dim)
    # ------------------------------------------------------------------
    print("\n[Test 7] Checkpoint round-trip")
    ckpt_dir = "runs/test_agent"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "high_level.pt")
    agent.save(ckpt_path)
    print(f"  Saved to: {ckpt_path}")
    
    agent2 = HighLevelDQNAgent(
        obs_dim=OBS_DIM, num_subarrays=K,
        hidden_dim=HIDDEN,       # <-- FIX: match saved architecture
        batch_size=BATCH, device="cpu", seed=99,
    )
    agent2.load(ckpt_path)
    print(f"  Loaded epsilon: {agent2.epsilon:.4f} (expect {agent.epsilon:.4f})")
    print(f"  Loaded step_count: {agent2._step_count} "
          f"(expect {agent._step_count})")
    assert abs(agent2.epsilon - agent.epsilon) < 1e-6
    assert agent2._step_count == agent._step_count
    print("  OK: round-trip preserves state")
    
    # ------------------------------------------------------------------
    # Test 8: Device resolution
    # ------------------------------------------------------------------
    print("\n[Test 8] Device resolution")
    agent_gpu = HighLevelDQNAgent(
        obs_dim=OBS_DIM, num_subarrays=K,
        hidden_dim=HIDDEN, device="cuda", batch_size=BATCH, seed=1,
    )
    print(f"  Requested cuda -> got: {agent_gpu.device}")
    assert agent_gpu.device.type in ("cuda", "cpu")
    print("  OK: device resolved safely")
    
    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    import shutil
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    
    print("\n" + "=" * 60)
    print("High-level DQN agent validation successful!")
    print("=" * 60)