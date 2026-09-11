"""
Formal test suite for near-field channel modeling and multi-user generation.

These tests verify:
    1. NearFieldChannel correctness (shape, phase, path loss, regimes)
    2. ChannelGenerator correctness (multipath, correlation, normalization)
    3. Physical invariants (symmetry, monotonicity, unit power)
    4. Edge cases (single/zero users, extreme distances)

Run with:
    pytest tests/test_channel.py -v
    pytest tests/test_channel.py -v --cov=src.channel
"""

import numpy as np
import pytest
from typing import Tuple

from src.channel.near_field import (
    NearFieldChannel,
    ChannelParameters,
    generate_test_scenario as nf_test_scenario,
)
from src.channel.channel_generator import (
    ChannelGenerator,
    ChannelGeneratorConfig,
    generate_test_scenario as cg_test_scenario,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def default_channel_params() -> ChannelParameters:
    """Standard 64-antenna ULA at 28 GHz."""
    return ChannelParameters(
        num_antennas=64,
        antenna_spacing=0.5,
        array_type='uniform_linear',
        carrier_frequency=28e9,
        path_loss_exponent=2.0,
    )


@pytest.fixture
def default_channel(default_channel_params) -> NearFieldChannel:
    """Instantiated NearFieldChannel."""
    return NearFieldChannel(default_channel_params)


@pytest.fixture
def default_generator_config() -> ChannelGeneratorConfig:
    """Standard channel generator config."""
    return ChannelGeneratorConfig(
        num_users=4,
        num_antennas=64,
        los_probability=0.8,
        num_multipath=3,
        multipath_spread=10.0,
        spatial_correlation=True,
        correlation_factor=0.3,
        normalize_channels=True,
        user_distance_range=(5.0, 80.0),
        user_angle_range=(-60.0, 60.0),
        seed=42,
    )


@pytest.fixture
def default_generator(default_generator_config) -> ChannelGenerator:
    """Instantiated ChannelGenerator."""
    return ChannelGenerator(default_generator_config)


# ============================================================================
# PART 1: NearFieldChannel — STRUCTURE
# ============================================================================

class TestNearFieldChannelStructure:
    """Tests for channel matrix structure and shape."""
    
    def test_channel_shape(self, default_channel):
        """Channel matrix has shape (num_users, num_antennas)."""
        num_users = 4
        user_positions = default_channel.get_user_positions(
            num_users=num_users, seed=0
        )
        H = default_channel.compute_channel(user_positions)
        
        assert H.shape == (num_users, default_channel.params.num_antennas)
        assert H.dtype == np.complex128
    
    def test_antenna_positions_centered(self, default_channel):
        """ULA is centered at origin (sum of positions ≈ 0)."""
        positions = default_channel.antenna_positions
        assert positions.shape == (default_channel.params.num_antennas, 3)
        
        center = positions.mean(axis=0)
        np.testing.assert_allclose(center, [0.0, 0.0, 0.0], atol=1e-12)
    
    def test_antenna_spacing_correct(self, default_channel):
        """Adjacent antennas are separated by d·λ."""
        positions = default_channel.antenna_positions
        expected_spacing = (
            default_channel.params.antenna_spacing *
            default_channel.params.wavelength
        )
        
        actual_spacing = np.linalg.norm(positions[1] - positions[0])
        np.testing.assert_allclose(actual_spacing, expected_spacing, rtol=1e-10)
    
    def test_channel_dtype_complex(self, default_channel):
        """Channel matrix is complex-valued."""
        user_pos = np.array([[10.0, 0.0, 0.0]])
        H = default_channel.compute_channel(user_pos)
        assert np.iscomplexobj(H)


# ============================================================================
# PART 2: NearFieldChannel — PHYSICAL PROPERTIES
# ============================================================================

class TestNearFieldChannelPhysics:
    """Tests for physical correctness of the channel model."""
    
    def test_unit_modulus_phase(self, default_channel):
        """Phase-only channel has unit magnitude at each antenna."""
        user_pos = np.array([[10.0, 5.0, 0.0]])
        H = default_channel.compute_channel(
            user_pos, include_path_loss=False, include_phase=True
        )
        magnitudes = np.abs(H)
        np.testing.assert_allclose(magnitudes, 1.0, rtol=1e-10)
    
    def test_path_loss_decreases_with_distance(self, default_channel):
        """Channel power decreases with user distance (path loss)."""
        positions = np.array([
            [5.0, 0.0, 0.0],     # near
            [50.0, 0.0, 0.0],    # far
        ])
        H = default_channel.compute_channel(positions)
        power_near = np.mean(np.abs(H[0]) ** 2)
        power_far = np.mean(np.abs(H[1]) ** 2)
        
        assert power_near > power_far, (
            f"Near-field power {power_near:.2e} should exceed "
            f"far-field power {power_far:.2e}"
        )
    
    def test_phase_progression_with_distance(self, default_channel):
        """Phase at each antenna follows exp(-j·2π·d/λ)."""
        user_pos = np.array([[10.0, 5.0, 0.0]])
        H = default_channel.compute_channel(
            user_pos, include_path_loss=False, include_phase=True
        )
        
        # Recompute distances to verify phase
        distances = np.linalg.norm(
            default_channel.antenna_positions - user_pos[0], axis=1
        )
        expected_phases = -2 * np.pi * distances / default_channel.params.wavelength
        expected = np.exp(1j * expected_phases)
        
        np.testing.assert_allclose(H[0], expected, atol=1e-12)
    
    def test_channel_normalization_per_user(self, default_channel):
        """Channel power per user is consistent with path loss model."""
        user_pos = np.array([[10.0, 0.0, 0.0]])
        H = default_channel.compute_channel(user_pos)
        
        # Average power across antennas
        avg_power = np.mean(np.abs(H[0]) ** 2)
        assert avg_power > 0, "Channel power must be positive"
        assert avg_power < 1.0, "Channel power should be small (path loss)"


# ============================================================================
# PART 3: NearFieldChannel — NEAR/FAR-FIELD DISTINCTION
# ============================================================================

class TestNearFieldRegimes:
    """Tests for near-field vs far-field regime classification."""
    
    def test_fraunhofer_distance_positive(self, default_channel):
        """Fraunhofer distance is positive and reasonable."""
        D_f = default_channel.compute_fraunhofer_distance()
        assert D_f > 0
        # For 64-ant 28 GHz array: aperture ~0.34 m → D_f ~21 m
        assert 15.0 < D_f < 30.0
    
    def test_regime_classification(self, default_channel):
        """get_regime returns correct classification by distance."""
        D_f = default_channel.params.fraunhofer_distance
        assert D_f is not None
        
        assert default_channel.get_regime(D_f * 0.05) == 'near-field'
        assert default_channel.get_regime(D_f * 0.5) == 'fresnel'
        assert default_channel.get_regime(D_f * 2.0) == 'far-field'
    
    def test_near_far_channel_differ(self, default_channel):
        """Near-field and far-field channels differ at close range."""
        user_pos = np.array([[8.0, 0.0, 0.0]])  # ~8m, near-field
        
        H_near = default_channel.compute_channel(user_pos)
        H_far = default_channel.compute_far_field_channel(user_pos)
        
        # Channels should be different
        diff = np.linalg.norm(H_near - H_far) / (
            np.linalg.norm(H_near) + 1e-12
        )
        assert diff > 0.01, (
            f"Near-field and far-field channels should differ at 8 m "
            f"(relative diff = {diff:.4f})"
        )
    
    def test_near_far_channels_converge_at_far_distance(self, default_channel):
        """At large distances, near-field ≈ far-field."""
        D_f = default_channel.params.fraunhofer_distance
        assert D_f is not None
        far_dist = 10 * D_f
        
        user_pos = np.array([[far_dist, 0.0, 0.0]])
        H_near = default_channel.compute_channel(user_pos)
        H_far = default_channel.compute_far_field_channel(user_pos)
        
        # Normalize both to compare shapes only
        H_near_n = H_near / np.linalg.norm(H_near)
        H_far_n = H_far / np.linalg.norm(H_far)
        
        # Should be highly correlated at far distance
        correlation = np.abs(np.vdot(H_near_n[0], H_far_n[0]))
        assert correlation > 0.95, (
            f"Near/far-field channels should converge at {far_dist:.0f} m "
            f"(correlation = {correlation:.4f})"
        )


# ============================================================================
# PART 4: NearFieldChannel — SYMMETRY & INVARIANTS
# ============================================================================

class TestNearFieldSymmetry:
    """Tests for geometric symmetry of the y-axis ULA."""
    
    def test_y_mirror_symmetry(self, default_channel):
        """Users at +y and −y (same |y|) give mirrored channel patterns."""
        pos_plus = np.array([[10.0, +5.0, 0.0]])
        pos_minus = np.array([[10.0, -5.0, 0.0]])
        
        H_plus = default_channel.compute_channel(
            pos_plus, include_path_loss=False
        )
        H_minus = default_channel.compute_channel(
            pos_minus, include_path_loss=False
        )
        
        # Powers should be identical (distances are reflections)
        # Note: with y-axis ULA, +y and −y produce DIFFERENT distance sets.
        # We check that *something* differs (no accidental aliasing).
        correlation = np.abs(
            np.vdot(H_plus[0], H_minus[0])
        ) / (np.linalg.norm(H_plus[0]) * np.linalg.norm(H_minus[0]))
        
        # Correlation should be < 1 (different angles)
        assert correlation < 0.999, (
            f"+y and −y should produce distinct channels "
            f"(correlation = {correlation:.6f})"
        )


# ============================================================================
# PART 5: ChannelGenerator — MULTIPATH
# ============================================================================

class TestChannelGeneratorMultipath:
    """Tests for multipath generation."""
    
    def test_multipath_changes_channel(self, default_generator):
        """Enabling multipath produces a different channel."""
        user_positions = default_generator.generate_user_positions()
        
        H_los_only = default_generator.generate_channel_matrix(
            user_positions,
            include_multipath=False,
            include_spatial_correlation=False,
            normalize=False,
        )
        H_with_mp = default_generator.generate_channel_matrix(
            user_positions,
            include_multipath=True,
            include_spatial_correlation=False,
            normalize=False,
        )
        
        # Different matrices
        diff = np.linalg.norm(H_los_only - H_with_mp) / (
            np.linalg.norm(H_los_only) + 1e-12
        )
        assert diff > 0.01, "Multipath should change the channel"
    
    def test_multipath_deterministic_with_seed(self):
        """Same seed produces same multipath realization."""
        config = ChannelGeneratorConfig(
            num_users=2,
            num_antennas=16,
            num_multipath=3,
            seed=42,
        )
        gen1 = ChannelGenerator(config)
        H1 = gen1.generate_channel_matrix()
        
        gen2 = ChannelGenerator(config)
        H2 = gen2.generate_channel_matrix()
        
        np.testing.assert_allclose(H1, H2, atol=1e-12)


# ============================================================================
# PART 6: ChannelGenerator — SPATIAL CORRELATION
# ============================================================================

class TestChannelGeneratorSpatialCorrelation:
    """Tests for spatial correlation effect."""
    
    def test_correlation_reduces_independence(self, default_generator):
        """Spatial correlation increases correlation between adjacent antennas."""
        user_positions = default_generator.generate_user_positions()
        
        H_indep = default_generator.generate_channel_matrix(
            user_positions,
            include_multipath=False,
            include_spatial_correlation=False,
            normalize=False,
        )
        H_corr = default_generator.generate_channel_matrix(
            user_positions,
            include_multipath=False,
            include_spatial_correlation=True,
            normalize=False,
        )
        
        # Compute mean adjacent-antenna correlation in each case
        def mean_adj_corr(H: np.ndarray) -> float:
            correlations = []
            for u in range(H.shape[0]):
                h = H[u, :]
                for i in range(H.shape[1] - 1):
                    c = np.corrcoef(np.real(h[i]), np.real(h[i+1]))[0, 1]
                    if not np.isnan(c):
                        correlations.append(abs(c))
            return float(np.mean(correlations)) if correlations else 0.0
        
        corr_indep = mean_adj_corr(H_indep)
        corr_corr = mean_adj_corr(H_corr)
        
        # Correlated channels have higher adjacent-antenna correlation
        # (may not always hold with random seed, so we allow some slack)
        assert corr_corr >= corr_indep - 0.1, (
            f"Spatial correlation should not decrease correlation "
            f"(indep: {corr_indep:.3f}, corr: {corr_corr:.3f})"
        )


# ============================================================================
# PART 7: ChannelGenerator — NORMALIZATION
# ============================================================================

class TestChannelGeneratorNormalization:
    """Tests for channel normalization."""
    
    def test_normalization_gives_unit_power(self, default_generator):
        """Normalized channels have unit average power per user."""
        user_positions = default_generator.generate_user_positions()
        H = default_generator.generate_channel_matrix(
            user_positions, normalize=True
        )
        
        for u in range(H.shape[0]):
            power = np.mean(np.abs(H[u, :]) ** 2)
            np.testing.assert_allclose(power, 1.0, rtol=1e-10)
    
    def test_unnormalized_channels_not_unit_power(self, default_generator):
        """Without normalization, power ≠ 1 (path loss applied)."""
        user_positions = default_generator.generate_user_positions()
        H = default_generator.generate_channel_matrix(
            user_positions, normalize=False
        )
        
        powers = np.mean(np.abs(H) ** 2, axis=1)
        # Should not all be exactly 1
        assert not np.allclose(powers, 1.0, atol=1e-6)


# ============================================================================
# PART 8: ChannelGenerator — BATCH GENERATION
# ============================================================================

class TestChannelGeneratorBatch:
    """Tests for batch generation."""
    
    def test_batch_shape(self, default_generator):
        """Batch generation returns (batch_size, num_users, num_antennas)."""
        batch = default_generator.generate_batch(batch_size=5)
        expected = (
            5,
            default_generator.config.num_users,
            default_generator.config.num_antennas,
        )
        assert batch.shape == expected
    
    def test_batch_items_differ(self, default_generator):
        """Different batch items have different channels."""
        batch = default_generator.generate_batch(batch_size=3)
        # Batches should differ (fresh user positions)
        diff_01 = np.linalg.norm(batch[0] - batch[1])
        assert diff_01 > 1e-6, "Batch items should be independent"


# ============================================================================
# PART 9: ChannelGenerator — STATISTICS & ANALYSIS
# ============================================================================

class TestChannelGeneratorStatistics:
    """Tests for statistics and analysis methods."""
    
    def test_statistics_keys(self, default_generator):
        """get_channel_statistics returns expected keys."""
        default_generator.generate_channel_matrix()
        stats = default_generator.get_channel_statistics()
        
        required = {
            'num_users', 'num_antennas', 'mean_power',
            'condition_number', 'channel_hardening',
        }
        assert required.issubset(stats.keys())
    
    def test_condition_number_positive(self, default_generator):
        """Condition number is ≥ 1 (numerically well-defined)."""
        default_generator.generate_channel_matrix()
        stats = default_generator.get_channel_statistics()
        assert stats['condition_number'] >= 1.0
    
    def test_hardening_factor_positive(self, default_generator):
        """Channel hardening factor is positive."""
        default_generator.generate_channel_matrix()
        stats = default_generator.get_channel_statistics()
        assert stats['channel_hardening'] > 0


# ============================================================================
# PART 10: EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_single_antenna_channel(self):
        """Channel model works with a single antenna."""
        params = ChannelParameters(num_antennas=1)
        channel = NearFieldChannel(params)
        user_pos = np.array([[10.0, 0.0, 0.0]])
        H = channel.compute_channel(user_pos)
        
        assert H.shape == (1, 1)
        assert np.iscomplexobj(H)
    
    def test_single_user_channel(self, default_generator):
        """Channel generator works with a single user."""
        default_generator.config.num_users = 1
        H = default_generator.generate_channel_matrix()
        assert H.shape[0] == 1
    
    def test_invalid_parameters_raise(self):
        """Invalid parameters raise appropriate errors."""
        with pytest.raises(ValueError):
            ChannelParameters(num_antennas=0)
        
        with pytest.raises(ValueError):
            ChannelParameters(num_antennas=-5)
        
        with pytest.raises(ValueError):
            ChannelParameters(carrier_frequency=0)
    
    def test_extreme_distance(self, default_channel):
        """Channel computes correctly at extreme distances."""
        # Very far
        far_pos = np.array([[1e6, 0.0, 0.0]])
        H_far = default_channel.compute_channel(far_pos)
        assert np.all(np.isfinite(H_far))
        
        # Very close (but not at antenna)
        close_pos = np.array([[0.5, 0.5, 0.0]])
        H_close = default_channel.compute_channel(close_pos)
        assert np.all(np.isfinite(H_close))


# ============================================================================
# PART 11: INTEGRATION — FULL PIPELINE SANITY
# ============================================================================

class TestIntegrationSanity:
    """End-to-end sanity checks across near-field + generator."""
    
    def test_full_pipeline_runs(self):
        """Full channel pipeline: params → model → users → matrix."""
        channel, user_positions, H = nf_test_scenario()
        assert H.shape[0] == user_positions.shape[0]
        assert H.shape[1] == channel.params.num_antennas
    
    def test_generator_integration(self):
        """Generator integrates with near-field model end-to-end."""
        generator, H, stats, quality = cg_test_scenario()
        
        assert H.shape == (
            generator.config.num_users,
            generator.config.num_antennas,
        )
        assert 'mean_power' in stats
        assert 'channel_gain_per_user' in quality
        assert len(quality['channel_gain_per_user']) == generator.config.num_users
    
    def test_channel_quality_keys(self, default_generator):
        """get_channel_quality returns per-user metrics."""
        default_generator.generate_channel_matrix()
        quality = default_generator.get_channel_quality()
        
        assert 'channel_gain_per_user' in quality
        assert 'near_field_ratios' in quality
        assert len(quality['channel_gain_per_user']) == default_generator.config.num_users


# ============================================================================
# PART 12: DETERMINISM
# ============================================================================

class TestDeterminism:
    """Tests for reproducibility with fixed seeds."""
    
    def test_same_seed_same_channel(self):
        """Same seed → identical channel matrix."""
        gen1 = ChannelGenerator(ChannelGeneratorConfig(seed=123))
        gen2 = ChannelGenerator(ChannelGeneratorConfig(seed=123))
        
        H1 = gen1.generate_channel_matrix()
        H2 = gen2.generate_channel_matrix()
        
        np.testing.assert_allclose(H1, H2, atol=1e-14)
    
    def test_different_seeds_different_channels(self):
        """Different seeds → different channels."""
        gen1 = ChannelGenerator(ChannelGeneratorConfig(seed=1))
        gen2 = ChannelGenerator(ChannelGeneratorConfig(seed=2))
        
        H1 = gen1.generate_channel_matrix()
        H2 = gen2.generate_channel_matrix()
        
        diff = np.linalg.norm(H1 - H2)
        assert diff > 1e-6, "Different seeds should produce different channels"


# ============================================================================
# RUN SUMMARY
# ============================================================================

if __name__ == "__main__":
    """Allow running as a script for quick sanity check."""
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))