"""
Formal test suite for subarray partitioning and polar-domain codebook.

FIXED:
    - test_activate_deactivate now correctly expects subarrays to start
      ACTIVE (matches SubarrayPartitioner.__init__ design).
    
KNOWN FAILURE (intentional):
    - test_subarray_positions_match_codebook_axis is expected to FAIL
      until backlog item #4 (subarray axis alignment) is completed.
"""

import numpy as np
import pytest
from typing import Dict, Any

from src.codebook.subarray import (
    SubarrayConfig,
    SubarrayPartitioner,
    Subarray,
    PartitionStrategy,
    generate_test_scenario as sub_test_scenario,
)
from src.codebook.polar_codebook import (
    PolarCodebookConfig,
    PolarCodebook,
    PolarBeam,
    generate_test_scenario as cb_test_scenario,
    _channel_to_target,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def default_subarray_config() -> SubarrayConfig:
    return SubarrayConfig(
        num_antennas=64,
        num_subarrays=8,
        strategy=PartitionStrategy.CONTIGUOUS,
    )


@pytest.fixture
def contiguous_partitioner(default_subarray_config) -> SubarrayPartitioner:
    return SubarrayPartitioner(default_subarray_config)


@pytest.fixture
def default_codebook_config() -> PolarCodebookConfig:
    return PolarCodebookConfig(
        num_antennas=64,
        num_angles=121,
        num_ranges=10,
        range_min=5.0,
        range_max=40.0,
        range_spacing='log',
        correlation_threshold=0.95,
    )


@pytest.fixture
def dense_codebook(default_codebook_config) -> PolarCodebook:
    return PolarCodebook(default_codebook_config)


@pytest.fixture
def pruned_codebook(default_codebook_config) -> PolarCodebook:
    cb = PolarCodebook(default_codebook_config)
    cb.prune(correlation_threshold=0.95)
    return cb


# ============================================================================
# PART 1: SUBARRAY — CONFIG VALIDATION
# ============================================================================

class TestSubarrayConfig:
    """Tests for SubarrayConfig validation."""
    
    def test_valid_config(self):
        cfg = SubarrayConfig(num_antennas=64, num_subarrays=8)
        assert cfg.num_antennas == 64
        assert cfg.num_subarrays == 8
    
    def test_invalid_num_antennas(self):
        with pytest.raises(ValueError):
            SubarrayConfig(num_antennas=0)
        with pytest.raises(ValueError):
            SubarrayConfig(num_antennas=-5)
    
    def test_invalid_num_subarrays(self):
        with pytest.raises(ValueError):
            SubarrayConfig(num_antennas=64, num_subarrays=0)
    
    def test_subarrays_exceed_antennas(self):
        with pytest.raises(ValueError):
            SubarrayConfig(num_antennas=4, num_subarrays=10)
    
    def test_invalid_overlap_ratio(self):
        with pytest.raises(ValueError):
            SubarrayConfig(num_antennas=64, num_subarrays=8,
                           overlap_ratio=1.5)
        with pytest.raises(ValueError):
            SubarrayConfig(num_antennas=64, num_subarrays=8,
                           overlap_ratio=-0.1)


# ============================================================================
# PART 2: SUBARRAY — PARTITIONING STRATEGIES
# ============================================================================

class TestContiguousPartitioning:
    """Tests for contiguous partitioning."""
    
    def test_num_subarrays(self, contiguous_partitioner):
        assert len(contiguous_partitioner.subarrays) == 8
    
    def test_full_coverage(self, contiguous_partitioner):
        info = contiguous_partitioner.validate()
        assert info['all_antennas_covered']
        assert info['coverage'] == 1.0
        assert info['max_antenna_usage'] == 1
        assert info['min_antenna_usage'] == 1
    
    def test_even_split(self, contiguous_partitioner):
        sizes = [s.num_antennas for s in contiguous_partitioner.subarrays]
        assert sizes == [8] * 8
    
    def test_contiguous_indices(self, contiguous_partitioner):
        for s in contiguous_partitioner.subarrays:
            expected = np.arange(s.antenna_indices[0],
                                  s.antenna_indices[0] + s.num_antennas)
            np.testing.assert_array_equal(s.antenna_indices, expected)


class TestInterleavedPartitioning:
    """Tests for interleaved partitioning."""
    
    def test_full_coverage(self):
        cfg = SubarrayConfig(
            num_antennas=64, num_subarrays=8,
            strategy=PartitionStrategy.INTERLEAVED,
        )
        partitioner = SubarrayPartitioner(cfg)
        info = partitioner.validate()
        assert info['all_antennas_covered']
        assert info['coverage'] == 1.0
    
    def test_interleaved_indices(self):
        cfg = SubarrayConfig(
            num_antennas=64, num_subarrays=8,
            strategy=PartitionStrategy.INTERLEAVED,
        )
        partitioner = SubarrayPartitioner(cfg)
        for k, s in enumerate(partitioner.subarrays):
            expected = np.arange(k, 64, 8)
            np.testing.assert_array_equal(s.antenna_indices, expected)


class TestOverlappingPartitioning:
    """Tests for overlapping partitioning."""
    
    def test_full_coverage(self):
        cfg = SubarrayConfig(
            num_antennas=64, num_subarrays=8,
            strategy=PartitionStrategy.OVERLAPPING,
            overlap_ratio=0.25,
        )
        partitioner = SubarrayPartitioner(cfg)
        info = partitioner.validate()
        assert info['all_antennas_covered']
    
    def test_overlap_present(self):
        cfg = SubarrayConfig(
            num_antennas=64, num_subarrays=8,
            strategy=PartitionStrategy.OVERLAPPING,
            overlap_ratio=0.25,
        )
        partitioner = SubarrayPartitioner(cfg)
        info = partitioner.validate()
        assert info['max_antenna_usage'] >= 2
    
    def test_no_overlap_gives_usage_1(self):
        cfg = SubarrayConfig(
            num_antennas=64, num_subarrays=8,
            strategy=PartitionStrategy.OVERLAPPING,
            overlap_ratio=0.0,
        )
        partitioner = SubarrayPartitioner(cfg)
        info = partitioner.validate()
        assert info['max_antenna_usage'] == 1


# ============================================================================
# PART 3: SUBARRAY — ACTIVATION MASKS
# ============================================================================

class TestActivationMasks:
    """Tests for activation mask behavior."""
    
    def test_initial_mask_all_ones(self, contiguous_partitioner):
        mask = contiguous_partitioner.get_activation_mask()
        np.testing.assert_array_equal(mask, np.ones(8, dtype=np.int8))
    
    def test_set_activation_mask(self, contiguous_partitioner):
        mask = np.array([1, 1, 0, 0, 1, 1, 0, 0], dtype=np.int8)
        contiguous_partitioner.set_activation_mask(mask)
        active = contiguous_partitioner.get_active_subarrays()
        assert len(active) == 4
        assert [s.subarray_id for s in active] == [0, 1, 4, 5]
    
    def test_active_antennas_count(self, contiguous_partitioner):
        mask = np.array([1, 1, 0, 0, 1, 1, 0, 0], dtype=np.int8)
        contiguous_partitioner.set_activation_mask(mask)
        assert contiguous_partitioner.get_num_active_antennas() == 32
    
    def test_invalid_mask_shape(self, contiguous_partitioner):
        with pytest.raises(ValueError):
            contiguous_partitioner.set_activation_mask(np.array([1, 0]))
    
    def test_invalid_mask_values(self, contiguous_partitioner):
        with pytest.raises(ValueError):
            contiguous_partitioner.set_activation_mask(
                np.array([2, 1, 0, 0, 1, 1, 0, 0], dtype=np.int8)
            )
    
    def test_effective_channel_shape(self, contiguous_partitioner):
        mask = np.array([1, 1, 0, 0, 1, 1, 0, 0], dtype=np.int8)
        contiguous_partitioner.set_activation_mask(mask)
        full_channel = np.random.randn(4, 64) + 1j * np.random.randn(4, 64)
        effective = contiguous_partitioner.get_effective_channel(full_channel)
        assert effective.shape == (4, 32)
    
    def test_effective_channel_no_active(self, contiguous_partitioner):
        mask = np.zeros(8, dtype=np.int8)
        contiguous_partitioner.set_activation_mask(mask)
        full_channel = np.random.randn(4, 64) + 1j * np.random.randn(4, 64)
        with pytest.raises(ValueError):
            contiguous_partitioner.get_effective_channel(full_channel)


# ============================================================================
# PART 4: SUBARRAY — STRUCTURE  ← FIXED
# ============================================================================

class TestSubarrayStructure:
    """Tests for Subarray object properties."""
    
    def test_antenna_positions_shape(self, contiguous_partitioner):
        for s in contiguous_partitioner.subarrays:
            assert s.antenna_positions.shape == (s.num_antennas, 3)
    
    def test_center_position(self, contiguous_partitioner):
        s = contiguous_partitioner.subarrays[0]
        expected_center = np.mean(s.antenna_positions, axis=0)
        np.testing.assert_allclose(s.center_position, expected_center,
                                    atol=1e-12)
    
    def test_activate_deactivate(self, contiguous_partitioner):
        """
        activate/deactivate toggles is_active.
        
        FIXED: subarrays start ACTIVE (activation_mask = ones).
        Test now correctly verifies toggle from active → inactive → active.
        """
        s = contiguous_partitioner.subarrays[0]
        
        # Initial state: ACTIVE (default from partitioner)
        assert s.is_active, "Subarrays should start active by design"
        
        # Toggle OFF
        s.deactivate()
        assert not s.is_active
        
        # Toggle ON
        s.activate()
        assert s.is_active
    
    def test_aperture_positive(self, contiguous_partitioner):
        for s in contiguous_partitioner.subarrays:
            assert s.aperture >= 0


# ============================================================================
# PART 5: POLAR CODEBOOK — GENERATION
# ============================================================================

class TestPolarCodebookGeneration:
    """Tests for codebook generation."""
    
    def test_codebook_size(self, dense_codebook):
        expected = (dense_codebook.config.num_angles *
                    dense_codebook.config.num_ranges)
        assert len(dense_codebook) == expected
    
    def test_beam_matrix_shape(self, dense_codebook):
        B = dense_codebook.get_beam_matrix()
        assert B.shape == (64, len(dense_codebook))
    
    def test_beams_normalized(self, dense_codebook):
        for beam in dense_codebook.beams[:5]:
            norm = np.linalg.norm(beam.beam_vector)
            np.testing.assert_allclose(norm, 1.0, rtol=1e-10)
    
    def test_beam_angles_cover_range(self, dense_codebook):
        angles = [b.angle_deg for b in dense_codebook.beams]
        assert min(angles) == pytest.approx(-60.0)
        assert max(angles) == pytest.approx(60.0)
    
    def test_beam_ranges_cover_range(self, dense_codebook):
        ranges = [b.range_m for b in dense_codebook.beams]
        assert min(ranges) == pytest.approx(5.0, rel=1e-6)
        assert max(ranges) == pytest.approx(40.0, rel=1e-6)
    
    def test_beam_vector_shape(self, dense_codebook):
        for beam in dense_codebook.beams[:5]:
            assert beam.beam_vector.shape == (64,)


# ============================================================================
# PART 6: POLAR CODEBOOK — NEAR-FIELD FOCUSING
# ============================================================================

class TestNearFieldFocusing:
    """Tests for beam focusing."""
    
    def test_matched_beam_power_at_boresight(self, dense_codebook):
        channel = _channel_to_target(dense_codebook, 0.0, 15.0)
        powers = dense_codebook.project_channel(channel)
        best_idx = int(np.argmax(powers))
        best_beam = dense_codebook[best_idx]
        
        assert powers[best_idx] > 0.9
        assert abs(best_beam.angle_deg - 0.0) < 2.0
        assert 0.7 < best_beam.range_m / 15.0 < 1.4
    
    def test_matched_beam_at_various_angles(self, dense_codebook):
        targets = [(30.0, 15.0), (-45.0, 8.0), (15.0, 25.0),
                   (-25.0, 10.0), (50.0, 6.0)]
        
        for t_angle, t_range in targets:
            channel = _channel_to_target(dense_codebook, t_angle, t_range)
            powers = dense_codebook.project_channel(channel)
            best_idx = int(np.argmax(powers))
            best_beam = dense_codebook[best_idx]
            
            assert powers[best_idx] > 0.5
            angle_err = abs(best_beam.angle_deg - t_angle)
            assert angle_err <= 2.0
    
    def test_best_beam_beats_mean(self, dense_codebook):
        channel = _channel_to_target(dense_codebook, 30.0, 15.0)
        powers = dense_codebook.project_channel(channel)
        ratio = np.max(powers) / np.mean(powers)
        assert ratio > 10.0


# ============================================================================
# PART 7: POLAR CODEBOOK — SYMMETRY BREAK
# ============================================================================

class TestCodebookSymmetry:
    """Tests for +θ vs −θ distinction."""
    
    def test_positive_negative_angles_distinct(self, dense_codebook):
        for angle in [10.0, 30.0, 50.0]:
            ch_pos = _channel_to_target(dense_codebook, +angle, 15.0)
            ch_neg = _channel_to_target(dense_codebook, -angle, 15.0)
            best_pos = dense_codebook.best_beam_for_channel(ch_pos)
            best_neg = dense_codebook.best_beam_for_channel(ch_neg)
            assert best_pos != best_neg
    
    def test_positive_negative_angles_correct_sign(self, dense_codebook):
        for angle in [10.0, 30.0, 50.0]:
            ch_pos = _channel_to_target(dense_codebook, +angle, 15.0)
            ch_neg = _channel_to_target(dense_codebook, -angle, 15.0)
            beam_pos = dense_codebook[dense_codebook.best_beam_for_channel(ch_pos)]
            beam_neg = dense_codebook[dense_codebook.best_beam_for_channel(ch_neg)]
            assert abs(beam_pos.angle_deg - angle) < 2.0
            assert abs(beam_neg.angle_deg + angle) < 2.0


# ============================================================================
# PART 8: POLAR CODEBOOK — GRID RESOLUTION
# ============================================================================

class TestGridResolution:
    """Tests for grid resolution vs angular resolution."""
    
    def test_grid_step_finer_than_resolution(self, dense_codebook):
        check = dense_codebook.compute_effective_aperture_check()
        assert check['angle_step_deg'] <= check['angular_resolution_deg']
    
    def test_near_field_region_exists(self, dense_codebook):
        check = dense_codebook.compute_effective_aperture_check()
        assert check['has_near_field_region']


# ============================================================================
# PART 9: POLAR CODEBOOK — PRUNING
# ============================================================================

class TestPruning:
    """Tests for correlation-based pruning."""
    
    def test_pruning_reduces_size(self, dense_codebook):
        original_size = len(dense_codebook)
        dense_codebook.prune(correlation_threshold=0.95)
        assert len(dense_codebook) < original_size
    
    def test_pruned_preserves_performance(self, pruned_codebook):
        targets = [(30.0, 15.0), (-45.0, 8.0), (0.0, 12.0)]
        for t_angle, t_range in targets:
            channel = _channel_to_target(pruned_codebook, t_angle, t_range)
            powers = pruned_codebook.project_channel(channel)
            best_power = np.max(powers)
            assert best_power > 0.4
    
    def test_pruning_reassigns_ids(self, dense_codebook):
        dense_codebook.prune(correlation_threshold=0.9)
        ids = [b.beam_id for b in dense_codebook.beams]
        assert ids == list(range(len(dense_codebook)))
    
    def test_pruned_size_tractable_for_rl(self, pruned_codebook):
        assert len(pruned_codebook) < 500


# ============================================================================
# PART 10: POLAR CODEBOOK — CORRELATION
# ============================================================================

class TestCorrelationMatrix:
    """Tests for the correlation matrix."""
    
    def test_correlation_matrix_shape(self, dense_codebook):
        corr = dense_codebook.compute_correlation_matrix()
        n = len(dense_codebook)
        assert corr.shape == (n, n)
    
    def test_self_correlation_is_one(self, dense_codebook):
        corr = dense_codebook.compute_correlation_matrix()
        np.testing.assert_allclose(np.diag(corr), 1.0, rtol=1e-10)
    
    def test_correlation_bounded(self, dense_codebook):
        corr = dense_codebook.compute_correlation_matrix()
        assert np.all(corr >= -1e-10)
        assert np.all(corr <= 1.0 + 1e-10)
    
    def test_distinct_beams_have_low_correlation(self, dense_codebook):
        beam_a = dense_codebook[0].beam_vector
        beam_b = dense_codebook[-1].beam_vector
        corr = np.abs(np.vdot(beam_a, beam_b))
        assert corr < 0.5


# ============================================================================
# PART 11: INTEGRATION — SUBARRAY + CODEBOOK
# ============================================================================

class TestSubarrayCodebookIntegration:
    """Integration between subarray and codebook."""
    
    @pytest.mark.xfail(
        reason="Backlog #4: subarray.py is still on x-axis; "
               "will pass once axis alignment is applied",
        strict=True,
    )
    def test_subarray_positions_match_codebook_axis(self):
        """
        Subarray ULA is on y-axis, matching codebook geometry.
        
        KNOWN FAILURE — tracked as backlog item #4.
        This test is our canary: it will fail until subarray.py is
        updated to use y-axis ULA (matching polar_codebook.py and
        near_field.py).
        
        xfail(strict=True) means:
          - Test fails now → marked xfailed (expected)
          - Test passes → marked xpassed (surprise — remove xfail)
        """
        cfg = SubarrayConfig(num_antennas=64, num_subarrays=8)
        partitioner = SubarrayPartitioner(cfg)
        
        s = partitioner.subarrays[0]
        positions = s.antenna_positions
        
        np.testing.assert_allclose(positions[:, 0], 0.0, atol=1e-10)
        np.testing.assert_allclose(positions[:, 2], 0.0, atol=1e-10)
        assert np.std(positions[:, 1]) > 0
    
    def test_codebook_per_subarray_size(self):
        cfg = SubarrayConfig(num_antennas=64, num_subarrays=8)
        partitioner = SubarrayPartitioner(cfg)
        sub_size = partitioner.subarrays[0].num_antennas
        
        cb_cfg = PolarCodebookConfig(
            num_antennas=sub_size,
            num_angles=15,
            num_ranges=5,
        )
        cb = PolarCodebook(cb_cfg)
        
        for beam in cb.beams[:5]:
            assert beam.beam_vector.shape[0] == sub_size


# ============================================================================
# PART 12: EDGE CASES
# ============================================================================

class TestCodebookEdgeCases:
    """Edge cases for codebooks."""
    
    def test_single_antenna_codebook(self):
        cfg = PolarCodebookConfig(
            num_antennas=1, num_angles=5, num_ranges=3,
        )
        cb = PolarCodebook(cfg)
        assert len(cb) == 15
        for beam in cb.beams:
            assert beam.beam_vector.shape == (1,)
    
    def test_very_narrow_angle_range(self):
        cfg = PolarCodebookConfig(
            num_antennas=64,
            angle_min_deg=-5.0, angle_max_deg=5.0,
            num_angles=5, num_ranges=3,
        )
        cb = PolarCodebook(cfg)
        angles = [b.angle_deg for b in cb.beams]
        assert min(angles) == pytest.approx(-5.0)
        assert max(angles) == pytest.approx(5.0)
    
    def test_find_beam_returns_valid_index(self, dense_codebook):
        for angle, rng in [(0.0, 10.0), (45.0, 20.0), (-30.0, 5.0)]:
            idx = dense_codebook.find_beam(angle, rng)
            assert 0 <= idx < len(dense_codebook)
    
    def test_get_beam_out_of_range(self, dense_codebook):
        with pytest.raises(IndexError):
            dense_codebook.get_beam(len(dense_codebook))
        with pytest.raises(IndexError):
            dense_codebook.get_beam(-1)
    
    def test_state_representation_shape(self, pruned_codebook):
        state = pruned_codebook.get_state_representation()
        assert state.shape == (len(pruned_codebook), 3)


# ============================================================================
# PART 13: DETERMINISM
# ============================================================================

class TestDeterminism:
    """Reproducibility tests."""
    
    def test_same_config_same_codebook(self):
        cfg = PolarCodebookConfig(num_antennas=32, num_angles=7, num_ranges=3)
        cb1 = PolarCodebook(cfg)
        cb2 = PolarCodebook(cfg)
        for b1, b2 in zip(cb1.beams, cb2.beams):
            np.testing.assert_allclose(b1.beam_vector, b2.beam_vector,
                                        atol=1e-14)
    
    def test_subarray_deterministic(self):
        cfg = SubarrayConfig(num_antennas=64, num_subarrays=8)
        p1 = SubarrayPartitioner(cfg)
        p2 = SubarrayPartitioner(cfg)
        for s1, s2 in zip(p1.subarrays, p2.subarrays):
            np.testing.assert_array_equal(s1.antenna_indices, s2.antenna_indices)


# ============================================================================
# RUN SUMMARY
# ============================================================================

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))