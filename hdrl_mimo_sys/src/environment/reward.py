"""
Multi-objective reward function for HDRL in near-field XL-MIMO systems.

The reward balances four objectives (as specified in the project abstract):

    1. Sum-rate (maximize)          → R_rate
    2. Beam training overhead       → penalty ∝ active beams
    3. Power consumption            → penalty ∝ active subarrays
    4. Multiuser interference       → penalty ∝ residual interference

The total reward is a weighted combination:

    r = w_rate · R_norm
      − w_overhead · overhead_norm
      − w_power · power_norm
      − w_interference · interference_norm
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass, field


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class RewardConfig:
    """Configuration for the multi-objective reward."""
    
    # Weights for each objective
    w_rate: float = 1.0
    w_overhead: float = 0.1
    w_power: float = 0.05
    w_interference: float = 0.2
    
    # Normalization references
    max_sum_rate: float = 100.0
    max_beams: int = 20
    max_subarrays: int = 8
    max_interference: float = 10.0
    
    # Baseline offsets
    min_sum_rate: float = 0.0
    min_overhead: float = 0.0
    min_power: float = 0.0
    min_interference: float = 0.0
    
    # Reward shaping
    clip_reward: bool = True
    clip_min: float = -10.0
    clip_max: float = 10.0
    
    # Optional success bonus
    rate_threshold: Optional[float] = None
    success_bonus: float = 0.0
    
    # Infeasible penalty
    infeasible_penalty: float = -100.0
    
    def __post_init__(self):
        if self.max_sum_rate <= 0:
            raise ValueError("max_sum_rate must be positive")
        if self.max_beams <= 0:
            raise ValueError("max_beams must be positive")
        if self.max_subarrays <= 0:
            raise ValueError("max_subarrays must be positive")
        if self.max_interference <= 0:
            raise ValueError("max_interference must be positive")


# ============================================================================
# REWARD COMPONENTS
# ============================================================================

@dataclass
class RewardComponents:
    """Container for the individual reward components."""
    
    sum_rate: float = 0.0
    num_active_beams: int = 0
    num_active_subarrays: int = 0
    interference_power: float = 0.0
    
    rate_norm: float = 0.0
    overhead_norm: float = 0.0
    power_norm: float = 0.0
    interference_norm: float = 0.0
    
    rate_contribution: float = 0.0
    overhead_contribution: float = 0.0
    power_contribution: float = 0.0
    interference_contribution: float = 0.0
    
    total: float = 0.0
    info: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# REWARD FUNCTION
# ============================================================================

class RewardFunction:
    """Multi-objective reward function for HDRL subarray/beam selection."""
    
    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config if config is not None else RewardConfig()
    
    # ------------------------------------------------------------------------
    # MAIN API
    # ------------------------------------------------------------------------
    
    def compute(self,
                sum_rate: float,
                num_active_beams: int,
                num_active_subarrays: int,
                interference_power: float,
                extra_info: Optional[Dict[str, Any]] = None
                ) -> Tuple[float, RewardComponents]:
        """Compute the total scalar reward and its components."""
        comp = RewardComponents()
        comp.sum_rate = float(sum_rate)
        comp.num_active_beams = int(num_active_beams)
        comp.num_active_subarrays = int(num_active_subarrays)
        comp.interference_power = float(interference_power)
        
        # Normalize
        comp.rate_norm = self._normalize(
            sum_rate, self.config.min_sum_rate, self.config.max_sum_rate
        )
        comp.overhead_norm = self._normalize(
            float(num_active_beams),
            self.config.min_overhead,
            float(self.config.max_beams),
        )
        comp.power_norm = self._normalize(
            float(num_active_subarrays),
            self.config.min_power,
            float(self.config.max_subarrays),
        )
        comp.interference_norm = self._normalize(
            interference_power,
            self.config.min_interference,
            self.config.max_interference,
        )
        
        # Weighted contributions
        comp.rate_contribution = self.config.w_rate * comp.rate_norm
        comp.overhead_contribution = -self.config.w_overhead * comp.overhead_norm
        comp.power_contribution = -self.config.w_power * comp.power_norm
        comp.interference_contribution = -self.config.w_interference * comp.interference_norm
        
        # Total
        total = (
            comp.rate_contribution
            + comp.overhead_contribution
            + comp.power_contribution
            + comp.interference_contribution
        )
        
        # ✅ FIX: capture threshold in local var so Pylance can narrow it
        threshold = self.config.rate_threshold
        if threshold is not None and sum_rate >= threshold:
            total += self.config.success_bonus
        
        # Clip
        if self.config.clip_reward:
            total = float(np.clip(total, self.config.clip_min, self.config.clip_max))
        
        comp.total = float(total)
        
        info: Dict[str, Any] = {
            'reward/total': comp.total,
            'reward/sum_rate': comp.sum_rate,
            'reward/num_active_beams': comp.num_active_beams,
            'reward/num_active_subarrays': comp.num_active_subarrays,
            'reward/interference_power': comp.interference_power,
            'reward/rate_norm': comp.rate_norm,
            'reward/overhead_norm': comp.overhead_norm,
            'reward/power_norm': comp.power_norm,
            'reward/interference_norm': comp.interference_norm,
            'reward/rate_contribution': comp.rate_contribution,
            'reward/overhead_contribution': comp.overhead_contribution,
            'reward/power_contribution': comp.power_contribution,
            'reward/interference_contribution': comp.interference_contribution,
        }
        if extra_info is not None:
            info.update(extra_info)
        comp.info = info
        
        return comp.total, comp
    
    # ------------------------------------------------------------------------
    # NORMALIZATION HELPER
    # ------------------------------------------------------------------------
    
    @staticmethod
    def _normalize(value: float, min_val: float, max_val: float) -> float:
        """Min-max normalize to [0, 1]."""
        if max_val - min_val <= 1e-12:
            return 0.0
        norm = (value - min_val) / (max_val - min_val)
        return float(np.clip(norm, 0.0, 1.0))
    
    # ------------------------------------------------------------------------
    # CONVENIENCE METHODS
    # ------------------------------------------------------------------------
    
    def rate_only(self, sum_rate: float) -> float:
        """Return only the rate contribution."""
        rate_norm = self._normalize(
            sum_rate, self.config.min_sum_rate, self.config.max_sum_rate
        )
        return self.config.w_rate * rate_norm
    
    def compute_from_precoder_info(self,
                                    precoder_info: Dict[str, Any],
                                    num_active_beams: int,
                                    num_active_subarrays: int,
                                    ) -> Tuple[float, RewardComponents]:
        """Compute reward from DigitalPrecoder.design() output."""
        sum_rate = float(precoder_info.get('sum_rate', 0.0))
        
        W = precoder_info.get('W', None)
        H_eff = precoder_info.get('H_eff', None)
        
        if W is not None and H_eff is not None:
            H_eff_W = H_eff @ W
            interference_matrix = np.abs(H_eff_W) ** 2
            total_interference = float(
                np.sum(interference_matrix) - np.sum(np.diag(interference_matrix))
            )
        else:
            total_interference = 0.0
        
        return self.compute(
            sum_rate=sum_rate,
            num_active_beams=num_active_beams,
            num_active_subarrays=num_active_subarrays,
            interference_power=total_interference,
        )
    
    # ------------------------------------------------------------------------
    # INFO
    # ------------------------------------------------------------------------
    
    def summary(self) -> str:
        """Human-readable summary."""
        return "\n".join([
            "RewardConfig Summary",
            f"  w_rate:         {self.config.w_rate}",
            f"  w_overhead:     {self.config.w_overhead}",
            f"  w_power:        {self.config.w_power}",
            f"  w_interference: {self.config.w_interference}",
            f"  max_sum_rate:   {self.config.max_sum_rate}",
            f"  max_beams:      {self.config.max_beams}",
            f"  max_subarrays:  {self.config.max_subarrays}",
            f"  max_interference: {self.config.max_interference}",
            f"  clip:           [{self.config.clip_min}, {self.config.clip_max}]",
        ])


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compute_reward(sum_rate: float,
                    num_active_beams: int,
                    num_active_subarrays: int,
                    interference_power: float,
                    config: Optional[RewardConfig] = None
                    ) -> float:
    """One-shot reward computation."""
    reward_fn = RewardFunction(config)
    reward, _ = reward_fn.compute(
        sum_rate=sum_rate,
        num_active_beams=num_active_beams,
        num_active_subarrays=num_active_subarrays,
        interference_power=interference_power,
    )
    return reward


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Reward Function Test")
    print("=" * 60)
    
    # ---------------------------------------------------------------
    # Test 1: Basic
    # ---------------------------------------------------------------
    print("\n[Test 1] Basic reward computation")
    config = RewardConfig(
        w_rate=1.0,
        w_overhead=0.1,
        w_power=0.05,
        w_interference=0.2,
        max_sum_rate=50.0,
        max_beams=20,
        max_subarrays=8,
        max_interference=10.0,
    )
    reward_fn = RewardFunction(config)
    print(reward_fn.summary())
    
    reward, comp = reward_fn.compute(
        sum_rate=25.0,
        num_active_beams=10,
        num_active_subarrays=4,
        interference_power=1.0,
    )
    print(f"\n  Inputs: sum_rate=25.0, beams=10, subarrays=4, interference=1.0")
    print(f"  rate_norm        = {comp.rate_norm:.4f}")
    print(f"  overhead_norm    = {comp.overhead_norm:.4f}")
    print(f"  power_norm       = {comp.power_norm:.4f}")
    print(f"  interference_norm= {comp.interference_norm:.4f}")
    print(f"  Total reward     = {comp.total:.4f}")
    
    # ---------------------------------------------------------------
    # Test 2: Trade-off  ✅ FIXED: explicit tuples with int types
    # ---------------------------------------------------------------
    print("\n[Test 2] Trade-off: more subarrays → higher rate but higher cost")
    scenarios: list[tuple[int, int, float, float]] = [
        (2, 4,  15.0, 0.5),
        (4, 8,  25.0, 1.0),
        (6, 12, 32.0, 1.5),
        (8, 16, 38.0, 2.0),
    ]
    print(f"  {'subarr':>6} {'beams':>6} {'rate':>6} {'reward':>10}")
    for sa, bm, rt, itf in scenarios:
        r, _ = reward_fn.compute(
            sum_rate=rt,
            num_active_beams=bm,
            num_active_subarrays=sa,
            interference_power=itf,
        )
        print(f"  {sa:>6} {bm:>6} {rt:>6.1f} {r:>10.4f}")
    
    # ---------------------------------------------------------------
    # Test 3: Rate-only
    # ---------------------------------------------------------------
    print("\n[Test 3] Rate-only reward (for ablation)")
    for rt in [0.0, 12.5, 25.0, 50.0]:
        r = reward_fn.rate_only(rt)
        print(f"  sum_rate={rt:>5.1f} → rate_only_reward = {r:.4f}")
    
    # ---------------------------------------------------------------
    # Test 4: Clipping
    # ---------------------------------------------------------------
    print("\n[Test 4] Reward clipping")
    extreme = RewardConfig(
        w_rate=1.0,
        w_overhead=100.0,
        max_sum_rate=50.0,
        max_beams=20,
        max_subarrays=8,
        max_interference=10.0,
        clip_reward=True,
        clip_min=-10.0,
        clip_max=10.0,
    )
    fn_extreme = RewardFunction(extreme)
    r, _ = fn_extreme.compute(
        sum_rate=5.0,
        num_active_beams=20,
        num_active_subarrays=8,
        interference_power=10.0,
    )
    print(f"  Extreme penalty case: reward = {r:.4f}")
    assert -10.0 <= r <= 10.0
    print("  ✓ Clipping works")
    
    # ---------------------------------------------------------------
    # Test 5: Success bonus
    # ---------------------------------------------------------------
    print("\n[Test 5] Success bonus (rate threshold)")
    config_bonus = RewardConfig(
        w_rate=1.0,
        w_overhead=0.1,
        max_sum_rate=50.0,
        rate_threshold=20.0,
        success_bonus=5.0,
    )
    fn_bonus = RewardFunction(config_bonus)
    
    for rt in [15.0, 25.0]:
        r, _ = fn_bonus.compute(
            sum_rate=rt,
            num_active_beams=5,
            num_active_subarrays=3,
            interference_power=0.5,
        )
        bonus_applied = rt >= 20.0
        print(f"  rate={rt:>5.1f}: reward={r:>7.4f} (bonus: {bonus_applied})")
    
    # ---------------------------------------------------------------
    # Test 6: Monotonicity
    # ---------------------------------------------------------------
    print("\n[Test 6] Monotonicity checks")
    r_low, _ = reward_fn.compute(sum_rate=10.0, num_active_beams=5,
                                  num_active_subarrays=3, interference_power=0.5)
    r_high, _ = reward_fn.compute(sum_rate=30.0, num_active_beams=5,
                                   num_active_subarrays=3, interference_power=0.5)
    print(f"  Higher rate → higher reward: {r_low:.4f} < {r_high:.4f}: {r_low < r_high}")
    assert r_low < r_high
    
    r_few, _ = reward_fn.compute(sum_rate=25.0, num_active_beams=3,
                                  num_active_subarrays=3, interference_power=0.5)
    r_many, _ = reward_fn.compute(sum_rate=25.0, num_active_beams=15,
                                   num_active_subarrays=3, interference_power=0.5)
    print(f"  Fewer beams → higher reward: {r_few:.4f} > {r_many:.4f}: {r_few > r_many}")
    assert r_few > r_many
    
    r_litf, _ = reward_fn.compute(sum_rate=25.0, num_active_beams=5,
                                   num_active_subarrays=3, interference_power=0.1)
    r_hitf, _ = reward_fn.compute(sum_rate=25.0, num_active_beams=5,
                                   num_active_subarrays=3, interference_power=5.0)
    print(f"  Less interference → higher reward: {r_litf:.4f} > {r_hitf:.4f}: {r_litf > r_hitf}")
    assert r_litf > r_hitf
    print("  ✓ All monotonicity checks passed")
    
    # ---------------------------------------------------------------
    # Test 7: Info dict
    # ---------------------------------------------------------------
    print("\n[Test 7] Info dict for logging")
    _, comp = reward_fn.compute(
        sum_rate=25.0,
        num_active_beams=10,
        num_active_subarrays=4,
        interference_power=1.0,
        extra_info={'episode': 42},
    )
    print(f"  Total info keys: {len(comp.info)}")
    print(f"  Sample keys: {sorted(list(comp.info.keys()))[:5]}")
    
    print("\n" + "=" * 60)
    print("✓ Reward function validation successful!")
    print("=" * 60)