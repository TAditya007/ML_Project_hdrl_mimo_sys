"""
XL-MIMO hierarchical beam-management environment (Gymnasium).

This is the central environment for HDRL training. It exposes a
Gymnasium-compliant interface where the agent chooses:

    Action = (subarray_mask, beam_indices_per_subarray)

    subarray_mask:              binary vector, length K (num_subarrays)
    beam_indices_per_subarray:  int vector, length K, values in [0, B)
                                (only used for ACTIVE subarrays)

The environment internally:
    1. Samples a new multiuser channel via ChannelGenerator
    2. Applies the subarray mask to select active subarrays
    3. Looks up beam indices in the polar codebook
    4. Builds analog precoder F_RF
    5. Computes effective channel H_eff = H · F_RF
    6. Applies digital precoder (MMSE / ZF / MRT)
    7. Computes reward = weighted(rate, overhead, power, interference)

Observation (compact):
    [channel_power_per_user, snr_estimates, mask_state, codebook_stats]
    → 1D float32 vector

Design decisions (confirmed):
    - Hierarchical action (mask + beams)
    - Compact observation
    - Random channel per reset()
    
XL-MIMO hierarchical beam-management environment (Gymnasium).

Fix C: codebook is now built PER SUBARRAY (32 antennas each),
matching the physical aperture so near-field focusing is real.

Config:
    - 256 antennas total
    - 8 subarrays (32 antennas each)
    - Subarray aperture ~ 0.336 m -> Fraunhofer ~ 5.2 m (near-field real)

Action (Dict):
    "subarray_mask": MultiBinary(K)
    "beam_indices":  MultiDiscrete([B]*K)  where B = per-subarray codebook size
"""

from __future__ import annotations

import os
import sys
import warnings
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Any, Tuple, List

from src.utils.config import ExperimentConfig, SystemConfig, RLConfig
from src.channel.channel_generator import (
    ChannelGenerator, ChannelGeneratorConfig,
)
from src.codebook.subarray import (
    SubarrayConfig, SubarrayPartitioner, PartitionStrategy,
)
from src.codebook.polar_codebook import (
    PolarCodebook, PolarCodebookConfig,
)
from src.precoding.analog import (
    AnalogPrecoder, AnalogPrecoderConfig,
)
from src.precoding.digital import (
    DigitalPrecoder, DigitalPrecoderConfig, DigitalPrecoderType,
)
from src.environment.reward import (
    RewardFunction, RewardConfig,
)


# ============================================================================
# HELPERS
# ============================================================================

def _partition_strategy_from_str(name: str) -> PartitionStrategy:
    mapping = {
        "contiguous": PartitionStrategy.CONTIGUOUS,
        "interleaved": PartitionStrategy.INTERLEAVED,
        "overlapping": PartitionStrategy.OVERLAPPING,
    }
    if name not in mapping:
        raise ValueError(f"Unknown partition strategy: {name}")
    return mapping[name]


def _precoder_type_from_str(name: str) -> DigitalPrecoderType:
    mapping = {
        "zf": DigitalPrecoderType.ZF,
        "mmse": DigitalPrecoderType.MMSE,
        "mrt": DigitalPrecoderType.MRT,
        "none": DigitalPrecoderType.NONE,
    }
    if name not in mapping:
        raise ValueError(f"Unknown precoder type: {name}")
    return mapping[name]


# ============================================================================
# ENVIRONMENT
# ============================================================================

class XLMIMOEnv(gym.Env):
    """
    Hierarchical XL-MIMO beam-management environment.
    
    Fix C: each subarray has its own codebook sized to match its antenna
    count, so near-field focusing is physically meaningful.
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(self,
                 config: Optional[ExperimentConfig] = None,
                 render_mode: Optional[str] = None):
        super().__init__()
        
        self.cfg: ExperimentConfig = config or ExperimentConfig()
        self.sys_cfg: SystemConfig = self.cfg.system
        self.rl_cfg: RLConfig = self.cfg.rl
        self.render_mode = render_mode
        
        # ----- Fixed dimensions -----
        self.num_antennas: int = self.sys_cfg.num_antennas
        self.num_subarrays: int = self.sys_cfg.num_subarrays
        self.num_users: int = self.sys_cfg.num_users
        self.wavelength: float = 3e8 / self.sys_cfg.carrier_frequency
        
        # ----- Subarray partitioner (must come BEFORE codebook) -----
        sub_cfg = SubarrayConfig(
            num_antennas=self.num_antennas,
            num_subarrays=self.num_subarrays,
            strategy=_partition_strategy_from_str(self.sys_cfg.partition_strategy),
            overlap_ratio=self.sys_cfg.overlap_ratio,
            antenna_spacing=self.sys_cfg.antenna_spacing,
            wavelength=self.wavelength,
        )
        self.partitioner = SubarrayPartitioner(sub_cfg)
        self.antennas_per_subarray: int = self.partitioner.subarrays[0].num_antennas
        
        # ----- Per-subarray codebook (Fix C) -----
        cb_cfg = PolarCodebookConfig(
            num_antennas=self.antennas_per_subarray,
            antenna_spacing=self.sys_cfg.antenna_spacing,
            wavelength=self.wavelength,
            angle_min_deg=self.sys_cfg.user_angle_min_deg,
            angle_max_deg=self.sys_cfg.user_angle_max_deg,
            num_angles=self.sys_cfg.codebook_num_angles,
            range_min=self.sys_cfg.codebook_range_min,
            range_max=self.sys_cfg.codebook_range_max,
            num_ranges=self.sys_cfg.codebook_num_ranges,
            range_spacing=self.sys_cfg.codebook_range_spacing,
            correlation_threshold=self.sys_cfg.codebook_prune_threshold,
        )
        self.codebook = PolarCodebook(cb_cfg)
        self.codebook.prune(correlation_threshold=self.sys_cfg.codebook_prune_threshold)
        self.num_beams: int = len(self.codebook)
        
        # Sanity check: codebook must match subarray antenna count
        assert self.codebook.config.num_antennas == self.antennas_per_subarray, (
            f"Codebook antennas ({self.codebook.config.num_antennas}) != "
            f"subarray antennas ({self.antennas_per_subarray})"
        )
        
        # ----- Channel generator -----
        gen_cfg = ChannelGeneratorConfig(
            num_users=self.num_users,
            num_antennas=self.num_antennas,
            num_multipath=self.sys_cfg.num_multipath,
            multipath_spread=self.sys_cfg.multipath_spread_deg,
            spatial_correlation=self.sys_cfg.spatial_correlation,
            correlation_factor=self.sys_cfg.correlation_factor,
            normalize_channels=self.sys_cfg.normalize_channels,
            user_distance_range=(
                self.sys_cfg.user_distance_min, self.sys_cfg.user_distance_max,
            ),
            user_angle_range=(
                self.sys_cfg.user_angle_min_deg, self.sys_cfg.user_angle_max_deg,
            ),
            seed=self.sys_cfg.seed,
        )
        self.channel_gen = ChannelGenerator(gen_cfg)
        
        # ----- Analog precoder -----
        analog_cfg = AnalogPrecoderConfig(
            num_antennas=self.num_antennas,
            enforce_unit_modulus=True,
            normalize_columns=True,
        )
        self.analog_precoder = AnalogPrecoder(analog_cfg)
        
        # ----- Digital precoder -----
        digital_cfg = DigitalPrecoderConfig(
            precoder_type=_precoder_type_from_str(self.sys_cfg.precoder_type),
            noise_power=self.sys_cfg.precoder_noise_power,
            total_power=self.sys_cfg.precoder_total_power,
            regularization=self.sys_cfg.precoder_regularization,
            power_allocation=self.sys_cfg.power_allocation,
        )
        self.digital_precoder = DigitalPrecoder(digital_cfg)
        
        # ----- Reward -----
        reward_cfg = RewardConfig(
            w_rate=self.rl_cfg.reward_w_rate,
            w_overhead=self.rl_cfg.reward_w_overhead,
            w_power=self.rl_cfg.reward_w_power,
            w_interference=self.rl_cfg.reward_w_interference,
            max_sum_rate=self.rl_cfg.reward_max_sum_rate,
            max_beams=self.rl_cfg.reward_max_beams,
            max_subarrays=self.rl_cfg.reward_max_subarrays,
            max_interference=self.rl_cfg.reward_max_interference,
            clip_reward=True,
            clip_min=self.rl_cfg.reward_clip_min,
            clip_max=self.rl_cfg.reward_clip_max,
        )
        self.reward_fn = RewardFunction(reward_cfg)
        
        # ----- RNG -----
        self._rng = np.random.default_rng(self.sys_cfg.seed)
        
        # ----- Gymnasium spaces -----
        # Store Dict space in a typed attribute so `.spaces` access is Pylance-safe
        self._action_space_dict: spaces.Dict = spaces.Dict({
            "subarray_mask": spaces.MultiBinary(self.num_subarrays),
            "beam_indices":  spaces.MultiDiscrete([self.num_beams] * self.num_subarrays),
        })
        self.action_space = self._action_space_dict
        
        obs_dim = self._compute_obs_dim()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32,
        )
        
        # ----- Episode state -----
        self._current_channel: Optional[np.ndarray] = None
        self._current_positions: Optional[np.ndarray] = None
        self._step_count: int = 0
        self._last_info: Dict[str, Any] = {}
    
    # ========================================================================
    # GYMNASIUM API
    # ========================================================================
    
    def reset(self,
              seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None,
              **kwargs: Any,
              ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment: sample a new channel, all subarrays active.
        
        Note: `**kwargs` absorbs extra args from newer Gymnasium versions.
        """
        super().reset(seed=seed)
        
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        
        self.channel_gen.reset()
        self._current_channel = self.channel_gen.generate_channel_matrix()
        pos = self.channel_gen._user_positions
        if pos is None:
            pos = self.channel_gen.generate_user_positions()
        self._current_positions = pos
        
        self.partitioner.set_activation_mask(
            np.ones(self.num_subarrays, dtype=np.int8)
        )
        self._step_count = 0
        
        obs = self._build_observation()
        info = self._build_info(reward_components=None)
        self._last_info = info
        return obs, info
    
    def step(self, action: Dict[str, np.ndarray]
             ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Apply a hierarchical action.
        """
        mask = np.asarray(action["subarray_mask"], dtype=np.int8)
        beams = np.asarray(action["beam_indices"], dtype=int)
        
        if mask.shape != (self.num_subarrays,):
            raise ValueError(f"subarray_mask must have shape ({self.num_subarrays},)")
        if beams.shape != (self.num_subarrays,):
            raise ValueError(f"beam_indices must have shape ({self.num_subarrays},)")
        
        # Coerce: at least one active
        if mask.sum() == 0:
            mask = mask.copy()
            mask[0] = 1
        
        self.partitioner.set_activation_mask(mask)
        
        # Build selected_beams dict
        active_mask = mask.astype(bool)
        selected_beams: Dict[int, int] = {}
        for sid in range(self.num_subarrays):
            if active_mask[sid]:
                bid = int(beams[sid])
                if bid < 0 or bid >= self.num_beams:
                    bid = int(self._rng.integers(0, self.num_beams))
                selected_beams[sid] = bid
        
        # Analog precoder: codebook matches subarray size → no re-derivation
        F_RF = self.analog_precoder.build(
            subarray_partitioner=self.partitioner,
            codebook=self.codebook,
            selected_beams=selected_beams,
        )
        
        # Effective channel
        assert self._current_channel is not None
        H_full = self._current_channel
        H_eff = H_full @ F_RF
        
        # Digital precoder
        precoder_info = self.digital_precoder.design(H_eff)
        precoder_info["H_eff"] = H_eff
        sum_rate = float(precoder_info["sum_rate"])
        
        # Reward components
        num_active_beams = len(selected_beams)
        num_active_subarrays = int(mask.sum())
        
        W = precoder_info["W"]
        H_eff_W = H_eff @ W
        interference_matrix = np.abs(H_eff_W) ** 2
        interference_power = float(
            np.sum(interference_matrix) - np.sum(np.diag(interference_matrix))
        )
        
        reward, comp = self.reward_fn.compute(
            sum_rate=sum_rate,
            num_active_beams=num_active_beams,
            num_active_subarrays=num_active_subarrays,
            interference_power=interference_power,
        )
        
        self._step_count += 1
        terminated = False
        truncated = self._step_count >= self.cfg.training.max_steps_per_episode
        
        obs = self._build_observation()
        info = self._build_info(
            reward_components=comp,
            sum_rate=sum_rate,
            num_active_beams=num_active_beams,
            num_active_subarrays=num_active_subarrays,
            interference_power=interference_power,
            precoder_info=precoder_info,
            F_RF=F_RF,
            H_eff=H_eff,
        )
        self._last_info = info
        return obs, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode != "human":
            return
        print("-" * 50)
        print(f"Step {self._step_count}")
        print(f"Active subarrays: {int(self.partitioner.activation_mask.sum())}")
        print(f"Sum-rate:         {self._last_info.get('sum_rate', 0):.4f} b/s/Hz")
        print(f"Reward:           {self._last_info.get('reward', 0):.4f}")
        print("-" * 50)
    
    def close(self):
        pass
    
    # ========================================================================
    # OBSERVATION
    # ========================================================================
    
    def _compute_obs_dim(self) -> int:
        """2*K_users + 2*K_subarrays + 2"""
        return 2 * self.num_users + 2 * self.num_subarrays + 1 + 1
    
    def _build_observation(self) -> np.ndarray:
        obs = np.zeros(self._compute_obs_dim(), dtype=np.float32)
        idx = 0
        
        if self._current_channel is not None:
            H = self._current_channel
            user_powers = np.mean(np.abs(H) ** 2, axis=1)
            max_p = max(user_powers.max(), 1e-12)
            obs[idx:idx + self.num_users] = (user_powers / max_p).astype(np.float32)
            idx += self.num_users
            
            snr_db = 10 * np.log10(user_powers + 1e-12)
            snr_clip = np.clip(snr_db, 0, 30) / 30.0
            obs[idx:idx + self.num_users] = snr_clip.astype(np.float32)
            idx += self.num_users
        else:
            idx += 2 * self.num_users
        
        obs[idx:idx + self.num_subarrays] = (
            self.partitioner.activation_mask.astype(np.float32)
        )
        idx += self.num_subarrays
        
        sizes = np.array(
            [s.num_antennas for s in self.partitioner.subarrays],
            dtype=np.float32,
        )
        sizes_norm = sizes / max(1.0, float(sizes.max()))
        obs[idx:idx + self.num_subarrays] = sizes_norm
        idx += self.num_subarrays
        
        obs[idx] = self.partitioner.get_num_active_antennas() / max(1, self.num_antennas)
        idx += 1
        obs[idx] = self.num_beams / 500.0
        idx += 1
        
        return obs
    
    # ========================================================================
    # INFO
    # ========================================================================
    
    def _build_info(
        self,
        reward_components=None,
        sum_rate: Optional[float] = None,
        num_active_beams: Optional[int] = None,
        num_active_subarrays: Optional[int] = None,
        interference_power: Optional[float] = None,
        precoder_info: Optional[Dict[str, Any]] = None,
        F_RF: Optional[np.ndarray] = None,
        H_eff: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "step": self._step_count,
            "num_beams": self.num_beams,
            "num_subarrays": self.num_subarrays,
        }
        if reward_components is not None:
            info["reward"] = float(reward_components.total)
            info["reward_components"] = {
                "rate": reward_components.rate_contribution,
                "overhead": reward_components.overhead_contribution,
                "power": reward_components.power_contribution,
                "interference": reward_components.interference_contribution,
            }
        if sum_rate is not None:
            info["sum_rate"] = float(sum_rate)
        if num_active_beams is not None:
            info["num_active_beams"] = int(num_active_beams)
        if num_active_subarrays is not None:
            info["num_active_subarrays"] = int(num_active_subarrays)
        if interference_power is not None:
            info["interference_power"] = float(interference_power)
        if precoder_info is not None and "sinr" in precoder_info:
            info["sinr_per_user"] = precoder_info["sinr"].tolist()
        if F_RF is not None:
            info["F_RF_shape"] = F_RF.shape
        if H_eff is not None:
            info["H_eff_shape"] = H_eff.shape
        return info
    
    # ========================================================================
    # CONVENIENCE
    # ========================================================================
    
    def get_reference_action(self) -> Dict[str, np.ndarray]:
        """All subarrays active + oracle beam per subarray (matched filter)."""
        mask = np.ones(self.num_subarrays, dtype=np.int8)
        beams = np.zeros(self.num_subarrays, dtype=int)
        
        if self._current_channel is None:
            return {"subarray_mask": mask, "beam_indices": beams}
        
        H = self._current_channel
        user_powers = np.mean(np.abs(H) ** 2, axis=1)
        strongest_user = int(np.argmax(user_powers))
        h_u = H[strongest_user, :]
        
        for sid, sub in enumerate(self.partitioner.subarrays):
            h_sub = h_u[sub.antenna_indices]
            powers = np.zeros(self.num_beams)
            for b in range(self.num_beams):
                w = self.codebook[b].beam_vector
                if w.shape[0] == h_sub.shape[0]:
                    powers[b] = float(np.abs(np.vdot(w, h_sub)) ** 2)
            beams[sid] = int(np.argmax(powers))
        
        return {"subarray_mask": mask, "beam_indices": beams}
    
    def summary(self) -> str:
        """Human-readable summary."""
        action_keys = list(self._action_space_dict.spaces.keys())
        return "\n".join([
            f"XLMIMOEnv Summary (Fix C: per-subarray codebook)",
            f"  Antennas:            {self.num_antennas}",
            f"  Subarrays:           {self.num_subarrays}",
            f"  Antennas/subarray:   {self.antennas_per_subarray}",
            f"  Users:               {self.num_users}",
            f"  Beams/subarray:      {self.num_beams}",
            f"  Precoder:            {self.sys_cfg.precoder_type}",
            f"  Strategy:            {self.sys_cfg.partition_strategy}",
            f"  Observation dim:     {self.observation_space.shape}",
            f"  Action keys:         {action_keys}",
        ])


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("XL-MIMO Environment Test (Fix C)")
    print("=" * 60)
    
    cfg = ExperimentConfig.from_dict({
        "system": {
            "num_antennas": 256,
            "num_subarrays": 8,
            "num_users": 4,
            "codebook_num_angles": 61,
            "codebook_num_ranges": 5,
            "codebook_range_min": 2.0,
            "codebook_range_max": 15.0,
        },
        "rl": {"device": "cpu"},
    })
    env = XLMIMOEnv(config=cfg)
    print("\n" + env.summary())
    
    # ------------------------------------------------------------------
    # Test 1: reset
    # ------------------------------------------------------------------
    print("\n[Test 1] reset()")
    obs, info = env.reset(seed=42)
    print(f"  Observation shape: {obs.shape}")
    print(f"  Observation dtype: {obs.dtype}")
    print(f"  Finite: {bool(np.all(np.isfinite(obs)))}")
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32
    assert np.all(np.isfinite(obs))
    print("  ✓ reset() works")
    
    # ------------------------------------------------------------------
    # Test 2: step with random action
    # ------------------------------------------------------------------
    print("\n[Test 2] step() with random action")
    action = env.action_space.sample()
    obs2, reward, term, trunc, info = env.step(action)
    print(f"  Reward:      {reward:.4f}")
    print(f"  Sum-rate:    {info['sum_rate']:.4f} b/s/Hz")
    print(f"  Active subarrays: {info['num_active_subarrays']}")
    print(f"  Active beams:     {info['num_active_beams']}")
    assert np.isfinite(reward)
    print("  ✓ step() works")
    
    # ------------------------------------------------------------------
    # Test 3: reference action
    # ------------------------------------------------------------------
    print("\n[Test 3] reference action (oracle)")
    ref = env.get_reference_action()
    _, reward_ref, _, _, info_ref = env.step(ref)
    print(f"  Reference reward: {reward_ref:.4f}")
    print(f"  Reference rate:   {info_ref['sum_rate']:.4f} b/s/Hz")
    assert info_ref["num_active_subarrays"] == env.num_subarrays
    print("  ✓ Reference action activates all subarrays")
    
    # ------------------------------------------------------------------
    # Test 4: sparse action
    # ------------------------------------------------------------------
    print("\n[Test 4] sparse action (2/8 subarrays active)")
    sparse = {
        "subarray_mask": np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=np.int8),
        "beam_indices": np.array([10, 20, 0, 0, 0, 0, 0, 0], dtype=int),
    }
    _, reward_sp, _, _, info_sp = env.step(sparse)
    print(f"  Sparse reward:    {reward_sp:.4f}")
    print(f"  Sparse rate:      {info_sp['sum_rate']:.4f} b/s/Hz")
    assert info_sp["num_active_subarrays"] == 2
    print("  ✓ Sparse action reflected")
    
    # ------------------------------------------------------------------
    # Test 5: zero mask safe
    # ------------------------------------------------------------------
    print("\n[Test 5] zero mask coerced")
    _, _, _, _, info_z = env.step({
        "subarray_mask": np.zeros(8, dtype=np.int8),
        "beam_indices": np.zeros(8, dtype=int),
    })
    assert info_z["num_active_subarrays"] >= 1
    print(f"  Active after zero mask: {info_z['num_active_subarrays']}")
    print("  ✓ Safe")
    
    # ------------------------------------------------------------------
    # Test 6: truncation
    # ------------------------------------------------------------------
    print("\n[Test 6] truncation")
    env.reset(seed=1)
    truncated_at = -1
    for t in range(env.cfg.training.max_steps_per_episode + 5):
        _, _, _, trunc, _ = env.step(env.get_reference_action())
        if trunc:
            truncated_at = t + 1
            break
    print(f"  Truncated at step {truncated_at} "
          f"(expected {env.cfg.training.max_steps_per_episode})")
    assert truncated_at == env.cfg.training.max_steps_per_episode
    print("  ✓ Truncation works")
    
    # ------------------------------------------------------------------
    # Test 7: reward components
    # ------------------------------------------------------------------
    print("\n[Test 7] reward components")
    env.reset(seed=3)
    _, _, _, _, info_full = env.step(env.get_reference_action())
    for k, v in info_full["reward_components"].items():
        print(f"    {k}: {v:.4f}")
        assert np.isfinite(v)
    print("  ✓ Finite reward components")
    
    # ------------------------------------------------------------------
    # Test 8: determinism
    # ------------------------------------------------------------------
    print("\n[Test 8] determinism")
    env2 = XLMIMOEnv(config=cfg)
    obs_a, _ = env.reset(seed=7)
    obs_b, _ = env2.reset(seed=7)
    assert np.allclose(obs_a, obs_b)
    print("  ✓ Deterministic with same seed")
    
    # ------------------------------------------------------------------
    # Test 9: no re-derivation warnings
    # ------------------------------------------------------------------
    print("\n[Test 9] no per-subarray re-derivation warnings")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        env.reset(seed=5)
        env.step(env.get_reference_action())
        relevant = [
            x for x in w
            if "subarray" in str(x.message).lower()
            and "re-derive" in str(x.message).lower()
        ]
        print(f"  Warnings about re-derivation: {len(relevant)}")
        assert len(relevant) == 0, "Should no longer re-derive beams"
    print("  ✓ No re-derivation warnings — codebook/subarray sizes match")
    
    print("\n" + "=" * 60)
    print("✓ XL-MIMO environment (Fix C) validation successful!")
    print("=" * 60)