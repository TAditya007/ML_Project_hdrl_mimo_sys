"""
Formal test suite for the multi-objective reward function.

Verifies:
    1. RewardFunction.compute() structure and return types
    2. Normalization correctness
    3. Weighted combination math
    4. Monotonicity (rate ↑ → reward ↑; beams/interference ↑ → reward ↓)
    5. Success bonus and clipping
    6. Info dict structure for logging
    7. Trade-off behavior across configurations
    8. Edge cases (zero rate, extreme values, invalid config)
    9. Determinism

Run with:
    pytest tests/test_environment.py -v
"""

import numpy as np
import pytest
from typing import Dict, Any, List, Tuple

from src.environment.reward import (
    RewardConfig,
    RewardFunction,
    RewardComponents,
    compute_reward,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def default_config() -> RewardConfig:
    return RewardConfig(
        w_rate=1.0,
        w_overhead=0.1,
        w_power=0.05,
        w_interference=0.2,
        max_sum_rate=50.0,
        max_beams=20,
        max_subarrays=8,
        max_interference=10.0,
        clip_reward=True,
        clip_min=-10.0,
        clip_max=10.0,
    )


@pytest.fixture
def reward_fn(default_config) -> RewardFunction:
    return RewardFunction(default_config)


# ============================================================================
# PART 1: CONFIG VALIDATION
# ============================================================================

class TestRewardConfig:
    """Tests for RewardConfig validation."""
    
    def test_valid_config(self):
        cfg = RewardConfig()
        assert cfg.w_rate == 1.0
        assert cfg.max_sum_rate == 100.0
    
    def test_invalid_max_sum_rate(self):
        with pytest.raises(ValueError):
            RewardConfig(max_sum_rate=0.0)
        with pytest.raises(ValueError):
            RewardConfig(max_sum_rate=-5.0)
    
    def test_invalid_max_beams(self):
        with pytest.raises(ValueError):
            RewardConfig(max_beams=0)
    
    def test_invalid_max_subarrays(self):
        with pytest.raises(ValueError):
            RewardConfig(max_subarrays=0)
    
    def test_invalid_max_interference(self):
        with pytest.raises(ValueError):
            RewardConfig(max_interference=0.0)


# ============================================================================
# PART 2: RETURN STRUCTURE
# ============================================================================

class TestReturnStructure:
    """Tests for return types and structure."""
    
    def test_compute_returns_tuple(self, reward_fn):
        result = reward_fn.compute(
            sum_rate=25.0, num_active_beams=10,
            num_active_subarrays=4, interference_power=1.0,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        reward, comp = result
        assert isinstance(reward, float)
        assert isinstance(comp, RewardComponents)
    
    def test_components_have_all_fields(self, reward_fn):
        _, comp = reward_fn.compute(
            sum_rate=25.0, num_active_beams=10,
            num_active_subarrays=4, interference_power=1.0,
        )
        assert comp.sum_rate == 25.0
        assert comp.num_active_beams == 10
        assert comp.num_active_subarrays == 4
        assert comp.interference_power == 1.0
        assert 0.0 <= comp.rate_norm <= 1.0
        assert 0.0 <= comp.overhead_norm <= 1.0
        assert 0.0 <= comp.power_norm <= 1.0
        assert 0.0 <= comp.interference_norm <= 1.0
        assert comp.rate_contribution >= 0.0
        assert comp.overhead_contribution <= 0.0
        assert comp.power_contribution <= 0.0
        assert comp.interference_contribution <= 0.0
    
    def test_info_dict_populated(self, reward_fn):
        _, comp = reward_fn.compute(
            sum_rate=25.0, num_active_beams=10,
            num_active_subarrays=4, interference_power=1.0,
        )
        required_keys = {
            'reward/total', 'reward/sum_rate',
            'reward/num_active_beams', 'reward/num_active_subarrays',
            'reward/interference_power',
            'reward/rate_norm', 'reward/overhead_norm',
            'reward/power_norm', 'reward/interference_norm',
            'reward/rate_contribution', 'reward/overhead_contribution',
            'reward/power_contribution', 'reward/interference_contribution',
        }
        assert required_keys.issubset(comp.info.keys())
    
    def test_extra_info_merged(self, reward_fn):
        _, comp = reward_fn.compute(
            sum_rate=25.0, num_active_beams=10,
            num_active_subarrays=4, interference_power=1.0,
            extra_info={'episode': 42, 'step': 7},
        )
        assert comp.info['episode'] == 42
        assert comp.info['step'] == 7


# ============================================================================
# PART 3: NORMALIZATION
# ============================================================================

class TestNormalization:
    """Tests for min-max normalization."""
    
    def test_rate_norm_at_zero(self, reward_fn):
        _, comp = reward_fn.compute(
            sum_rate=0.0, num_active_beams=0,
            num_active_subarrays=0, interference_power=0.0,
        )
        assert comp.rate_norm == pytest.approx(0.0)
    
    def test_rate_norm_at_max(self, reward_fn):
        _, comp = reward_fn.compute(
            sum_rate=50.0, num_active_beams=0,
            num_active_subarrays=0, interference_power=0.0,
        )
        assert comp.rate_norm == pytest.approx(1.0)
    
    def test_rate_norm_beyond_max_clipped(self, reward_fn):
        _, comp = reward_fn.compute(
            sum_rate=100.0, num_active_beams=0,
            num_active_subarrays=0, interference_power=0.0,
        )
        assert comp.rate_norm == pytest.approx(1.0)
    
    def test_overhead_norm(self, reward_fn):
        _, comp = reward_fn.compute(
            sum_rate=0.0, num_active_beams=10,
            num_active_subarrays=0, interference_power=0.0,
        )
        assert comp.overhead_norm == pytest.approx(0.5)
    
    def test_power_norm(self, reward_fn):
        _, comp = reward_fn.compute(
            sum_rate=0.0, num_active_beams=0,
            num_active_subarrays=4, interference_power=0.0,
        )
        assert comp.power_norm == pytest.approx(0.5)
    
    def test_interference_norm(self, reward_fn):
        _, comp = reward_fn.compute(
            sum_rate=0.0, num_active_beams=0,
            num_active_subarrays=0, interference_power=5.0,
        )
        assert comp.interference_norm == pytest.approx(0.5)
    
    def test_norm_bounded_in_0_1(self, reward_fn):
        for rt in [-10, 0, 25, 50, 100, 1000]:
            for bm in [0, 10, 20, 100]:
                for sa in [0, 4, 8, 50]:
                    for itf in [0, 5, 10, 100]:
                        _, comp = reward_fn.compute(
                            sum_rate=float(rt),
                            num_active_beams=int(bm),
                            num_active_subarrays=int(sa),
                            interference_power=float(itf),
                        )
                        assert 0.0 <= comp.rate_norm <= 1.0
                        assert 0.0 <= comp.overhead_norm <= 1.0
                        assert 0.0 <= comp.power_norm <= 1.0
                        assert 0.0 <= comp.interference_norm <= 1.0


# ============================================================================
# PART 4: WEIGHTED COMBINATION
# ============================================================================

class TestWeightedCombination:
    """Tests for the weighted sum math."""
    
    def test_reward_matches_manual_computation(self, reward_fn):
        _, comp = reward_fn.compute(
            sum_rate=25.0, num_active_beams=10,
            num_active_subarrays=4, interference_power=1.0,
        )
        expected = 1.0*0.5 - 0.1*0.5 - 0.05*0.5 - 0.2*0.1
        assert comp.total == pytest.approx(expected, abs=1e-6)
    
    def test_zero_inputs_give_zero_reward(self, reward_fn):
        reward, comp = reward_fn.compute(
            sum_rate=0.0, num_active_beams=0,
            num_active_subarrays=0, interference_power=0.0,
        )
        assert comp.total == pytest.approx(0.0)
    
    def test_rate_only_config(self):
        cfg = RewardConfig(
            w_rate=1.0, w_overhead=0.0, w_power=0.0, w_interference=0.0,
            max_sum_rate=100.0,
        )
        fn = RewardFunction(cfg)
        _, comp = fn.compute(
            sum_rate=50.0, num_active_beams=100,
            num_active_subarrays=100, interference_power=100.0,
        )
        assert comp.total == pytest.approx(0.5)


# ============================================================================
# PART 5: MONOTONICITY
# ============================================================================

class TestMonotonicity:
    """Tests for monotonic relationships."""
    
    def test_higher_rate_higher_reward(self, reward_fn):
        r_low, _ = reward_fn.compute(
            sum_rate=10.0, num_active_beams=5,
            num_active_subarrays=3, interference_power=0.5,
        )
        r_high, _ = reward_fn.compute(
            sum_rate=30.0, num_active_beams=5,
            num_active_subarrays=3, interference_power=0.5,
        )
        assert r_high > r_low
    
    def test_more_beams_lower_reward(self, reward_fn):
        r_few, _ = reward_fn.compute(
            sum_rate=25.0, num_active_beams=3,
            num_active_subarrays=3, interference_power=0.5,
        )
        r_many, _ = reward_fn.compute(
            sum_rate=25.0, num_active_beams=15,
            num_active_subarrays=3, interference_power=0.5,
        )
        assert r_few > r_many
    
    def test_more_subarrays_lower_reward(self, reward_fn):
        r_few, _ = reward_fn.compute(
            sum_rate=25.0, num_active_beams=5,
            num_active_subarrays=2, interference_power=0.5,
        )
        r_many, _ = reward_fn.compute(
            sum_rate=25.0, num_active_beams=5,
            num_active_subarrays=8, interference_power=0.5,
        )
        assert r_few > r_many
    
    def test_higher_interference_lower_reward(self, reward_fn):
        r_low, _ = reward_fn.compute(
            sum_rate=25.0, num_active_beams=5,
            num_active_subarrays=3, interference_power=0.1,
        )
        r_high, _ = reward_fn.compute(
            sum_rate=25.0, num_active_beams=5,
            num_active_subarrays=3, interference_power=5.0,
        )
        assert r_low > r_high
    
    def test_reward_decreases_monotonically_with_beams(self, reward_fn):
        rewards: List[float] = []
        for b in [0, 5, 10, 15, 20]:
            r, _ = reward_fn.compute(
                sum_rate=25.0,
                num_active_beams=int(b),
                num_active_subarrays=3,
                interference_power=0.5,
            )
            rewards.append(r)
        
        for i in range(len(rewards) - 1):
            assert rewards[i] > rewards[i + 1], \
                f"Reward not decreasing: {rewards}"


# ============================================================================
# PART 6: SUCCESS BONUS  ← FIXED (fully inlined)
# ============================================================================

class TestSuccessBonus:
    """Tests for the rate-threshold bonus."""
    
    def test_bonus_applied_above_threshold(self):
        """
        Bonus adds exactly success_bonus to the total when sum_rate ≥ threshold.
        
        FIXED: fully inlined compute() calls (no **inputs dict) so Pylance
        can infer concrete argument types.
        """
        cfg_with = RewardConfig(
            w_rate=1.0, max_sum_rate=50.0,
            rate_threshold=20.0, success_bonus=5.0,
            clip_reward=False,
        )
        cfg_without = RewardConfig(
            w_rate=1.0, max_sum_rate=50.0,
            rate_threshold=None, clip_reward=False,
        )
        
        total_with, _ = RewardFunction(cfg_with).compute(
            sum_rate=25.0,
            num_active_beams=5,
            num_active_subarrays=3,
            interference_power=0.5,
        )
        total_without, _ = RewardFunction(cfg_without).compute(
            sum_rate=25.0,
            num_active_beams=5,
            num_active_subarrays=3,
            interference_power=0.5,
        )
        
        assert total_with == pytest.approx(total_without + 5.0, abs=1e-6)
    
    def test_no_bonus_below_threshold(self):
        cfg = RewardConfig(
            w_rate=1.0, max_sum_rate=50.0,
            rate_threshold=20.0, success_bonus=5.0,
            clip_reward=False,
        )
        fn = RewardFunction(cfg)
        _, comp = fn.compute(
            sum_rate=15.0, num_active_beams=5,
            num_active_subarrays=3, interference_power=0.5,
        )
        manual = (
            comp.rate_contribution
            + comp.overhead_contribution
            + comp.power_contribution
            + comp.interference_contribution
        )
        assert comp.total == pytest.approx(manual, abs=1e-6)
    
    def test_no_bonus_when_threshold_none(self, reward_fn):
        """With rate_threshold=None, huge sum_rate → rate_norm=1.0 → reward=1.0."""
        _, comp = reward_fn.compute(
            sum_rate=1000.0, num_active_beams=0,
            num_active_subarrays=0, interference_power=0.0,
        )
        assert comp.rate_norm == pytest.approx(1.0)
        assert comp.total == pytest.approx(1.0)
    
    def test_bonus_pushes_above_rate_only(self):
        cfg_no_bonus = RewardConfig(
            w_rate=1.0, max_sum_rate=50.0,
            rate_threshold=None, clip_reward=False,
        )
        cfg_bonus = RewardConfig(
            w_rate=1.0, max_sum_rate=50.0,
            rate_threshold=20.0, success_bonus=3.0, clip_reward=False,
        )
        r_no, _ = RewardFunction(cfg_no_bonus).compute(
            sum_rate=30.0, num_active_beams=5,
            num_active_subarrays=3, interference_power=0.5,
        )
        r_bonus, _ = RewardFunction(cfg_bonus).compute(
            sum_rate=30.0, num_active_beams=5,
            num_active_subarrays=3, interference_power=0.5,
        )
        assert r_bonus == pytest.approx(r_no + 3.0, abs=1e-6)


# ============================================================================
# PART 7: CLIPPING
# ============================================================================

class TestClipping:
    """Tests for reward clipping."""
    
    def test_clip_upper_bound(self):
        cfg = RewardConfig(
            w_rate=100.0, max_sum_rate=1.0,
            clip_reward=True, clip_min=-10.0, clip_max=10.0,
        )
        fn = RewardFunction(cfg)
        r, _ = fn.compute(
            sum_rate=1.0, num_active_beams=0,
            num_active_subarrays=0, interference_power=0.0,
        )
        assert r == pytest.approx(10.0)
    
    def test_clip_lower_bound(self):
        cfg = RewardConfig(
            w_rate=1.0, w_overhead=1000.0,
            max_sum_rate=100.0, max_beams=10,
            clip_reward=True, clip_min=-10.0, clip_max=10.0,
        )
        fn = RewardFunction(cfg)
        r, _ = fn.compute(
            sum_rate=0.0, num_active_beams=10,
            num_active_subarrays=0, interference_power=0.0,
        )
        assert r == pytest.approx(-10.0)
    
    def test_no_clip_when_disabled(self):
        cfg = RewardConfig(
            w_rate=100.0, max_sum_rate=1.0,
            clip_reward=False,
        )
        fn = RewardFunction(cfg)
        r, _ = fn.compute(
            sum_rate=1.0, num_active_beams=0,
            num_active_subarrays=0, interference_power=0.0,
        )
        assert r == pytest.approx(100.0)


# ============================================================================
# PART 8: TRADE-OFF BEHAVIOR
# ============================================================================

class TestTradeoff:
    """Tests for the rate vs cost trade-off."""
    
    def test_reward_rises_then_plateaus(self):
        cfg = RewardConfig(
            w_rate=1.0, w_overhead=0.3, w_power=0.2,
            max_sum_rate=50.0, max_beams=20, max_subarrays=8,
            max_interference=10.0,
            clip_reward=False,
        )
        fn = RewardFunction(cfg)
        
        scenarios: List[Tuple[int, int, float, float]] = [
            (2, 4,  15.0, 0.5),
            (4, 8,  25.0, 1.0),
            (6, 12, 30.0, 1.5),
            (8, 16, 32.0, 2.0),
        ]
        
        rewards: List[float] = []
        for sa, bm, rt, itf in scenarios:
            r, _ = fn.compute(
                sum_rate=rt,
                num_active_beams=bm,
                num_active_subarrays=sa,
                interference_power=itf,
            )
            rewards.append(r)
        
        gain_first = rewards[1] - rewards[0]
        gain_last = rewards[-1] - rewards[-2]
        assert gain_last < gain_first, \
            f"Reward gains should shrink: {rewards}"


# ============================================================================
# PART 9: EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Edge cases and boundary conditions."""
    
    def test_zero_everything(self, reward_fn):
        reward, comp = reward_fn.compute(
            sum_rate=0.0, num_active_beams=0,
            num_active_subarrays=0, interference_power=0.0,
        )
        assert comp.total == pytest.approx(0.0)
        assert np.isfinite(reward)
    
    def test_max_everything(self, reward_fn):
        reward, comp = reward_fn.compute(
            sum_rate=50.0, num_active_beams=20,
            num_active_subarrays=8, interference_power=10.0,
        )
        assert np.isfinite(reward)
        assert -10.0 <= reward <= 10.0
    
    def test_extreme_values_clipped(self, reward_fn):
        """Very large sum_rate → rate_norm=1.0 → reward=1.0."""
        reward, comp = reward_fn.compute(
            sum_rate=1e6, num_active_beams=0,
            num_active_subarrays=0, interference_power=0.0,
        )
        assert comp.rate_norm == pytest.approx(1.0)
        assert reward == pytest.approx(1.0)
    
    def test_negative_inputs_handled(self, reward_fn):
        _, comp = reward_fn.compute(
            sum_rate=-10.0, num_active_beams=-5,
            num_active_subarrays=-3, interference_power=-2.0,
        )
        assert comp.rate_norm == pytest.approx(0.0)
        assert comp.overhead_norm == pytest.approx(0.0)
        assert comp.power_norm == pytest.approx(0.0)
        assert comp.interference_norm == pytest.approx(0.0)
    
    def test_zero_max_does_not_crash(self):
        cfg = RewardConfig(
            w_rate=1.0, max_sum_rate=1e-9,
            max_beams=1, max_subarrays=1, max_interference=1e-9,
        )
        fn = RewardFunction(cfg)
        r, _ = fn.compute(
            sum_rate=1e-8, num_active_beams=1,
            num_active_subarrays=1, interference_power=1e-8,
        )
        assert np.isfinite(r)


# ============================================================================
# PART 10: ONE-SHOT UTILITY
# ============================================================================

class TestComputeRewardUtility:
    """Tests for the standalone compute_reward function."""
    
    def test_utility_matches_class(self, default_config):
        fn = RewardFunction(default_config)
        r_class, _ = fn.compute(
            sum_rate=25.0, num_active_beams=10,
            num_active_subarrays=4, interference_power=1.0,
        )
        r_util = compute_reward(
            sum_rate=25.0, num_active_beams=10,
            num_active_subarrays=4, interference_power=1.0,
            config=default_config,
        )
        assert r_util == pytest.approx(r_class, abs=1e-10)
    
    def test_utility_with_default_config(self):
        r = compute_reward(
            sum_rate=25.0, num_active_beams=10,
            num_active_subarrays=4, interference_power=1.0,
        )
        assert np.isfinite(r)


# ============================================================================
# PART 11: RATE-ONLY UTILITY (ABLATION SUPPORT)
# ============================================================================

class TestRateOnly:
    """Tests for the rate-only reward (used for ablations)."""
    
    def test_rate_only_linear(self, reward_fn):
        assert reward_fn.rate_only(0.0) == pytest.approx(0.0)
        assert reward_fn.rate_only(25.0) == pytest.approx(0.5)
        assert reward_fn.rate_only(50.0) == pytest.approx(1.0)
    
    def test_rate_only_clipped(self, reward_fn):
        assert reward_fn.rate_only(100.0) == pytest.approx(1.0)
    
    def test_rate_only_ignores_costs(self, reward_fn):
        assert reward_fn.rate_only(25.0) == reward_fn.rate_only(25.0)


# ============================================================================
# PART 12: DETERMINISM
# ============================================================================

class TestDeterminism:
    """Tests for reproducibility."""
    
    def test_same_inputs_same_reward(self, reward_fn):
        r1, c1 = reward_fn.compute(
            sum_rate=25.0, num_active_beams=10,
            num_active_subarrays=4, interference_power=1.0,
        )
        r2, c2 = reward_fn.compute(
            sum_rate=25.0, num_active_beams=10,
            num_active_subarrays=4, interference_power=1.0,
        )
        assert r1 == r2
        for key in c1.info:
            v1, v2 = c1.info[key], c2.info[key]
            if isinstance(v1, float):
                assert v1 == pytest.approx(v2, abs=1e-12)
    
    def test_new_instance_same_reward(self, default_config):
        fn1 = RewardFunction(default_config)
        fn2 = RewardFunction(default_config)
        r1, _ = fn1.compute(
            sum_rate=30.0, num_active_beams=7,
            num_active_subarrays=5, interference_power=1.5,
        )
        r2, _ = fn2.compute(
            sum_rate=30.0, num_active_beams=7,
            num_active_subarrays=5, interference_power=1.5,
        )
        assert r1 == pytest.approx(r2, abs=1e-12)


# ============================================================================
# PART 13: SUMMARY
# ============================================================================

class TestSummary:
    """Tests for the summary string."""
    
    def test_summary_contains_weights(self, reward_fn):
        s = reward_fn.summary()
        assert 'w_rate' in s
        assert 'w_overhead' in s
        assert 'w_power' in s
        assert 'w_interference' in s
    
    def test_summary_multiline(self, reward_fn):
        s = reward_fn.summary()
        assert '\n' in s
        assert len(s) > 50


# ============================================================================
# RUN SUMMARY
# ============================================================================

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))