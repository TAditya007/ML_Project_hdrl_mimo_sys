"""
Low-level DQN agent for per-subarray beam selection in XL-MIMO.

The agent takes the environment observation PLUS the high-level agent's
subarray mask, and outputs one beam index per subarray. Only active
subarrays' outputs are used (masked loss).

Architecture (Option A):
    input = [obs (obs_dim) | subarray_mask (K)]
    shared torso:   Linear -> ReLU -> Linear -> ReLU -> hidden (hidden_dim)
    K independent heads: hidden -> num_beams  (one head per subarray)

Training:
    - Double DQN (online selects action, target evaluates)
    - Masked MSE loss over active subarray heads only
    - Target network synced every N steps

Usage:
    agent = LowLevelDQNAgent(obs_dim=26, num_subarrays=8, num_beams=61)
    beams = agent.select_action(obs, subarray_mask)   # int array (K,)
    agent.store(obs, mask, beams, reward, next_obs, next_mask, done)
    loss = agent.train_step()                          # None if buffer not ready
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ============================================================================
# NETWORK
# ============================================================================

class LowLevelQNet(nn.Module):
    """
    Shared torso + K independent beam-selection heads.
    
    Input:  concat(obs, subarray_mask)  →  (obs_dim + K,)
    Output: (K, num_beams) logits — one distribution per subarray
    """
    
    def __init__(self,
                 obs_dim: int,
                 num_subarrays: int,
                 num_beams: int,
                 hidden_dim: int = 128,
                 num_layers: int = 2):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_subarrays = num_subarrays
        self.num_beams = num_beams
        self.hidden_dim = hidden_dim
        
        # Shared torso
        in_dim = obs_dim + num_subarrays
        torso_layers: List[nn.Module] = []
        for _ in range(num_layers):
            torso_layers.append(nn.Linear(in_dim, hidden_dim))
            torso_layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.torso = nn.Sequential(*torso_layers)
        
        # K independent heads (one per subarray)
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, num_beams) for _ in range(num_subarrays)
        ])
    
    def forward(self, obs: torch.Tensor, mask: torch.Tensor
                ) -> torch.Tensor:
        """
        Args:
            obs:  (batch, obs_dim) or (obs_dim,)
            mask: (batch, K) or (K,) — binary float/int
        
        Returns:
            (batch, K, num_beams) or (K, num_beams) logits
        """
        # Handle 1D case by adding a batch dim
        squeeze_out = obs.dim() == 1
        if squeeze_out:
            obs = obs.unsqueeze(0)
            mask = mask.unsqueeze(0)
        
        x = torch.cat([obs, mask.float()], dim=1)   # (B, obs_dim + K)
        h = self.torso(x)                            # (B, hidden_dim)
        
        # Apply each head → stack along new dim
        logits = torch.stack(
            [head(h) for head in self.heads], dim=1
        )                                            # (B, K, num_beams)
        
        if squeeze_out:
            logits = logits.squeeze(0)               # (K, num_beams)
        
        return logits


# ============================================================================
# REPLAY BUFFER
# ============================================================================

@dataclass
class LLTransition:
    """One (s, mask, a, r, s', mask', done) transition."""
    obs: np.ndarray
    mask: np.ndarray          # int8, (K,)
    action: np.ndarray        # int,  (K,) — beam index per subarray
    reward: float
    next_obs: np.ndarray
    next_mask: np.ndarray     # int8, (K,)
    done: bool


class LLReplayBuffer:
    """Fixed-size replay buffer for low-level agent."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: deque[LLTransition] = deque(maxlen=capacity)
    
    def push(self,
             obs: np.ndarray,
             mask: np.ndarray,
             action: np.ndarray,
             reward: float,
             next_obs: np.ndarray,
             next_mask: np.ndarray,
             done: bool) -> None:
        self.buffer.append(LLTransition(
            obs=np.asarray(obs, dtype=np.float32).copy(),
            mask=np.asarray(mask, dtype=np.int8).copy(),
            action=np.asarray(action, dtype=np.int64).copy(),
            reward=float(reward),
            next_obs=np.asarray(next_obs, dtype=np.float32).copy(),
            next_mask=np.asarray(next_mask, dtype=np.int8).copy(),
            done=bool(done),
        ))
    
    def sample(self, batch_size: int) -> List[LLTransition]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def ready(self, batch_size: int) -> bool:
        return len(self.buffer) >= batch_size


# ============================================================================
# AGENT
# ============================================================================

class LowLevelDQNAgent:
    """
    Low-level DQN agent for beam selection per active subarray.
    
    Action: (K,) integer array of beam indices. Only entries where
    subarray_mask == 1 contribute to training loss.
    """
    
    def __init__(self,
                 obs_dim: int,
                 num_subarrays: int,
                 num_beams: int,
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
        self.num_beams = num_beams
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
        self.online_net = LowLevelQNet(
            obs_dim=obs_dim,
            num_subarrays=num_subarrays,
            num_beams=num_beams,
            hidden_dim=hidden_dim,
        ).to(self.device)
        
        self.target_net = LowLevelQNet(
            obs_dim=obs_dim,
            num_subarrays=num_subarrays,
            num_beams=num_beams,
            hidden_dim=hidden_dim,
        ).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.online_net.parameters(), lr=learning_rate
        )
        
        # Replay
        self.replay = LLReplayBuffer(capacity=buffer_size)
        
        # Diagnostics
        self.last_loss: float = 0.0
        self.updates_done: int = 0
    
    # ========================================================================
    # ACTION SELECTION
    # ========================================================================
    
    @torch.no_grad()
    def select_action(self,
                      obs: np.ndarray,
                      subarray_mask: np.ndarray,
                      evaluate: bool = False
                      ) -> np.ndarray:
        """
        Select a beam index per subarray.
        
        Args:
            obs: (obs_dim,) observation
            subarray_mask: (K,) binary — which subarrays are active
            evaluate: if True, use argmax (no exploration)
        
        Returns:
            int64 array of shape (K,) with beam indices.
            Inactive subarrays get arbitrary valid index (0).
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        mask_t = torch.as_tensor(
            subarray_mask, dtype=torch.float32, device=self.device
        )
        logits = self.online_net(obs_t, mask_t)     # (K, num_beams)
        
        mask_bool = np.asarray(subarray_mask, dtype=bool)
        action = np.zeros(self.num_subarrays, dtype=np.int64)
        
        for sid in range(self.num_subarrays):
            if not mask_bool[sid]:
                continue
            
            if evaluate or random.random() >= self.epsilon:
                action[sid] = int(torch.argmax(logits[sid]).item())
            else:
                action[sid] = random.randrange(self.num_beams)
        
        return action
    
    def update_epsilon(self, step: Optional[int] = None) -> None:
        """Linear epsilon decay."""
        current_step: int = self._step_count if step is None else int(step)
        
        if current_step >= self.epsilon_decay:
            self.epsilon = self.epsilon_end
        else:
            frac = current_step / max(1, self.epsilon_decay)
            self.epsilon = self.epsilon_start + frac * (
                self.epsilon_end - self.epsilon_start
            )
    
    # ========================================================================
    # TRAINING
    # ========================================================================
    
    def store(self,
              obs: np.ndarray,
              subarray_mask: np.ndarray,
              action: np.ndarray,
              reward: float,
              next_obs: np.ndarray,
              next_subarray_mask: np.ndarray,
              done: bool) -> None:
        """Push a transition to replay and increment step counter."""
        self.replay.push(
            obs, subarray_mask, action, reward,
            next_obs, next_subarray_mask, done,
        )
        self._step_count += 1
        self.update_epsilon()
    
    def train_step(self) -> Optional[float]:
        """
        One gradient step of Double DQN.
        
        Per-subarray loss (masked): each ACTIVE subarray's Q-value is
        updated against its OWN Bellman target:
        
            target_k = r + gamma * max_a' Q_target(s', a'_k, k) * (1 - done)
            loss_k   = MSE(Q_online(s, a_k, k), target_k)
        
        Loss = mean over active subarrays (weighted by mask).
        
        This avoids the pathological scale of a summed target across
        K subarrays, which caused exploding losses in earlier tests.
        """
        if not self.replay.ready(self.batch_size):
            return None
        
        batch = self.replay.sample(self.batch_size)
        
        # Stack tensors
        obs = torch.as_tensor(
            np.stack([t.obs for t in batch]),
            dtype=torch.float32, device=self.device,
        )                                          # (B, obs_dim)
        masks = torch.as_tensor(
            np.stack([t.mask for t in batch]).astype(np.float32),
            dtype=torch.float32, device=self.device,
        )                                          # (B, K)
        actions = torch.as_tensor(
            np.stack([t.action for t in batch]),
            dtype=torch.long, device=self.device,
        )                                          # (B, K)
        rewards = torch.as_tensor(
            np.array([t.reward for t in batch]),
            dtype=torch.float32, device=self.device,
        )                                          # (B,)
        next_obs = torch.as_tensor(
            np.stack([t.next_obs for t in batch]),
            dtype=torch.float32, device=self.device,
        )
        next_masks = torch.as_tensor(
            np.stack([t.next_mask for t in batch]).astype(np.float32),
            dtype=torch.float32, device=self.device,
        )
        dones = torch.as_tensor(
            np.array([t.done for t in batch], dtype=np.float32),
            dtype=torch.float32, device=self.device,
        )                                          # (B,)
        
        # ---- Online Q per (sample, subarray, beam) ----
        q_all = self.online_net(obs, masks)         # (B, K, num_beams)
        
        # Q at chosen action per subarray
        q_taken = q_all.gather(
            dim=2,
            index=actions.unsqueeze(-1),            # (B, K, 1)
        ).squeeze(-1)                               # (B, K)
        
        # ---- Double DQN target, PER SUBARRAY ----
        with torch.no_grad():
            # Online selects greedy next beam per subarray
            q_next_online = self.online_net(next_obs, next_masks)   # (B, K, Bn)
            greedy_next = torch.argmax(q_next_online, dim=2)        # (B, K)
            
            # Target evaluates those beams
            q_next_target = self.target_net(next_obs, next_masks)   # (B, K, Bn)
            q_next_taken = q_next_target.gather(
                dim=2,
                index=greedy_next.unsqueeze(-1),
            ).squeeze(-1)                                          # (B, K)
            
            # Per-subarray Bellman target: broadcast reward across subarrays
            # target[b, k] = r[b] + gamma * q_next_taken[b, k] * (1 - done[b])
            target_per_k = (
                rewards.unsqueeze(1)                                # (B, 1)
                + self.gamma * q_next_taken * (1.0 - dones.unsqueeze(1))
            )                                                       # (B, K)
        
        # ---- Masked per-subarray loss ----
        # Only active subarrays contribute
        squared_error = (q_taken - target_per_k) ** 2              # (B, K)
        masked_squared_error = squared_error * masks               # (B, K)
        
        num_active = masks.sum().clamp(min=1.0)                    # scalar
        loss = masked_squared_error.sum() / num_active
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.online_net.parameters(), max_norm=1.0
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
            "num_beams": self.num_beams,
            "hidden_dim": self.hidden_dim,
        }, path)
    
    def load(self, path: str) -> None:
        """Load from a checkpoint (validates architecture)."""
        ckpt = torch.load(path, map_location=self.device)
        
        # Validate architecture
        checks = [
            ("obs_dim", self.obs_dim),
            ("num_subarrays", self.num_subarrays),
            ("num_beams", self.num_beams),
            ("hidden_dim", self.hidden_dim),
        ]
        for key, expected in checks:
            saved = ckpt.get(key)
            if saved is not None and saved != expected:
                raise ValueError(
                    f"Checkpoint {key}={saved} != agent {key}={expected}. "
                    f"Construct the agent to match the checkpoint."
                )
        
        try:
            self.online_net.load_state_dict(ckpt["online_net"])
            self.target_net.load_state_dict(ckpt["target_net"])
        except RuntimeError as e:
            raise ValueError(
                f"Checkpoint architecture mismatch. Original error: {e}"
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
            f"LowLevelDQNAgent",
            f"  obs_dim:          {self.obs_dim}",
            f"  num_subarrays:    {self.num_subarrays}",
            f"  num_beams:        {self.num_beams}",
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
    print("Low-Level DQN Agent Test")
    print("=" * 60)
    
    OBS_DIM = 26
    K = 8
    B = 61
    HIDDEN = 64
    BATCH = 16
    
    agent = LowLevelDQNAgent(
        obs_dim=OBS_DIM,
        num_subarrays=K,
        num_beams=B,
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
    mask = np.array([1, 1, 0, 0, 1, 1, 0, 0], dtype=np.int8)
    
    a_train = agent.select_action(obs, mask, evaluate=False)
    a_eval = agent.select_action(obs, mask, evaluate=True)
    
    print(f"  mask:       {mask}")
    print(f"  train beams: {a_train}")
    print(f"  eval  beams: {a_eval}")
    assert a_train.shape == (K,)
    assert a_eval.shape == (K,)
    assert a_train.dtype == np.int64
    assert np.all(a_train >= 0) and np.all(a_train < B)
    assert np.all(a_eval >= 0) and np.all(a_eval < B)
    # Inactive subarrays should have beam = 0
    assert a_train[2] == 0 and a_train[3] == 0
    assert a_train[6] == 0 and a_train[7] == 0
    print("  OK: action selection works")
    
    # ------------------------------------------------------------------
    # Test 2: Store transitions
    # ------------------------------------------------------------------
    print("\n[Test 2] Store transitions")
    for _ in range(BATCH):
        o = np.random.randn(OBS_DIM).astype(np.float32)
        m = (np.random.rand(K) < 0.5).astype(np.int8)
        if m.sum() == 0:
            m[0] = 1
        a = agent.select_action(o, m)
        r = float(np.random.randn())
        o_next = np.random.randn(OBS_DIM).astype(np.float32)
        m_next = (np.random.rand(K) < 0.5).astype(np.int8)
        if m_next.sum() == 0:
            m_next[0] = 1
        agent.store(o, m, a, r, o_next, m_next, bool(np.random.rand() < 0.1))
    
    print(f"  Replay size: {len(agent.replay)} (expect {BATCH})")
    print(f"  Ready: {agent.replay.ready(BATCH)}")
    assert len(agent.replay) == BATCH
    assert agent.replay.ready(BATCH)
    print("  OK: buffer filled")
    
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
    # Test 4: Replay not ready
    # ------------------------------------------------------------------
    print("\n[Test 4] Replay-not-ready case")
    agent_small = LowLevelDQNAgent(
        obs_dim=OBS_DIM, num_subarrays=K, num_beams=B,
        hidden_dim=HIDDEN, batch_size=64, device="cpu", seed=0,
    )
    for _ in range(10):
        o = np.random.randn(OBS_DIM).astype(np.float32)
        m = np.ones(K, dtype=np.int8)
        agent_small.store(o, m, np.zeros(K, dtype=np.int64),
                          0.0, o, m, False)
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
        m = (np.random.rand(K) < 0.5).astype(np.int8)
        if m.sum() == 0:
            m[0] = 1
        a = agent.select_action(o, m)
        o_next = np.random.randn(OBS_DIM).astype(np.float32)
        m_next = (np.random.rand(K) < 0.5).astype(np.int8)
        if m_next.sum() == 0:
            m_next[0] = 1
        agent.store(o, m, a, 0.5, o_next, m_next, False)
        l = agent.train_step()
        if l is not None:
            losses.append(l)
    print(f"  Collected {len(losses)} losses, range "
          f"[{min(losses):.4f}, {max(losses):.4f}]")
    assert all(np.isfinite(losses))
    print("  OK: training loss stable")
    
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
    # Test 7: Checkpoint round-trip
    # ------------------------------------------------------------------
    print("\n[Test 7] Checkpoint round-trip")
    ckpt_dir = "runs/test_agent_ll"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "low_level.pt")
    agent.save(ckpt_path)
    print(f"  Saved to: {ckpt_path}")
    
    agent2 = LowLevelDQNAgent(
        obs_dim=OBS_DIM, num_subarrays=K, num_beams=B,
        hidden_dim=HIDDEN,
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
    # Test 8: Masked action behavior (inactive subarrays get beam 0)
    # ------------------------------------------------------------------
    print("\n[Test 8] Masked action behavior")
    test_mask = np.array([1, 0, 1, 0, 0, 1, 0, 0], dtype=np.int8)
    beams_masked = agent.select_action(obs, test_mask, evaluate=True)
    print(f"  mask:       {test_mask}")
    print(f"  beams:      {beams_masked}")
    for sid in range(K):
        if test_mask[sid] == 0:
            assert beams_masked[sid] == 0, \
                f"inactive subarray {sid} should have beam 0"
    print("  OK: inactive subarrays have beam index 0")
    
    # ------------------------------------------------------------------
    # Test 9: Architecture mismatch detection
    # ------------------------------------------------------------------
    print("\n[Test 9] Load rejects mismatched architecture")
    agent_wrong = LowLevelDQNAgent(
        obs_dim=OBS_DIM, num_subarrays=K, num_beams=B,
        hidden_dim=128,          # different from saved 64
        batch_size=BATCH, device="cpu", seed=1,
    )
    try:
        agent_wrong.load(ckpt_path)
        print("  ERROR: should have raised ValueError")
        assert False
    except ValueError as e:
        print(f"  Raised ValueError: {str(e)[:80]}...")
    print("  OK: architecture mismatch rejected")
    
    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    import shutil
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    
    print("\n" + "=" * 60)
    print("Low-level DQN agent validation successful!")
    print("=" * 60)