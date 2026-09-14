"""
Subarray partitioning for XL-MIMO systems.

FIXED (backlog #4):
    The ULA is now placed along the Y-AXIS, matching the geometry used by
    `near_field.py` and `polar_codebook.py`. Previously it was on the x-axis,
    which created a geometric inconsistency across the pipeline.

Array geometry (now consistent across the project):
    ULA along y-axis: antenna i at (0, y_i, 0), y_i ∈ [−L/2, +L/2]
    Broadside direction: +x
    Angle θ measured from +x axis
    Target at (θ, r): (r·cos θ, r·sin θ, 0)

Partitioning strategies:
    CONTIGUOUS: [0,1,2,3] | [4,5,6,7]     (adjacent blocks)
    INTERLEAVED: [0,2,4,6] | [1,3,5,7]     (round-robin)
    OVERLAPPING: [0,1,2,3,4] | [3,4,5,6,7] (shared antennas)
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


class PartitionStrategy(Enum):
    """Enumeration of supported subarray partitioning strategies."""
    CONTIGUOUS = "contiguous"
    INTERLEAVED = "interleaved"
    OVERLAPPING = "overlapping"


@dataclass
class SubarrayConfig:
    """Configuration for subarray partitioning."""
    
    num_antennas: int = 64
    num_subarrays: int = 8
    strategy: PartitionStrategy = PartitionStrategy.CONTIGUOUS
    
    overlap_ratio: float = 0.0
    
    antenna_spacing: float = 0.5
    wavelength: float = 3e8 / 28e9
    
    def __post_init__(self):
        """Validate configuration."""
        if self.num_antennas <= 0:
            raise ValueError("num_antennas must be positive")
        if self.num_subarrays <= 0:
            raise ValueError("num_subarrays must be positive")
        if self.num_subarrays > self.num_antennas:
            raise ValueError(
                f"num_subarrays ({self.num_subarrays}) cannot exceed "
                f"num_antennas ({self.num_antennas})"
            )
        if not (0.0 <= self.overlap_ratio < 1.0):
            raise ValueError("overlap_ratio must be in [0, 1)")


class Subarray:
    """
    Represents a single subarray in an XL-MIMO system.
    
    A subarray is a collection of antennas treated as one unit for
    analog beamforming.
    """
    
    def __init__(self,
                 subarray_id: int,
                 antenna_indices: np.ndarray,
                 global_antenna_positions: np.ndarray):
        self.subarray_id = subarray_id
        self.antenna_indices = np.asarray(antenna_indices, dtype=int)
        self.antenna_positions = global_antenna_positions[self.antenna_indices]
        self.center_position = np.mean(self.antenna_positions, axis=0)
        
        if len(self.antenna_positions) > 1:
            self.aperture = np.max(
                np.linalg.norm(
                    self.antenna_positions - self.center_position, axis=1
                )
            ) * 2
        else:
            self.aperture = 0.0
        
        self.is_active = False
    
    @property
    def num_antennas(self) -> int:
        return len(self.antenna_indices)
    
    def activate(self):
        self.is_active = True
    
    def deactivate(self):
        self.is_active = False
    
    def compute_steering_vector(self,
                                direction: np.ndarray,
                                wavelength: float) -> np.ndarray:
        """Compute the steering vector for this subarray toward a direction."""
        projections = self.antenna_positions @ direction
        steering = np.exp(-1j * 2 * np.pi * projections / wavelength)
        return steering
    
    def get_info(self) -> Dict[str, Any]:
        """Return subarray information as dict."""
        return {
            'subarray_id': self.subarray_id,
            'num_antennas': self.num_antennas,
            'antenna_indices': self.antenna_indices.tolist(),
            'center_position': self.center_position.tolist(),
            'aperture': float(self.aperture),
            'is_active': self.is_active,
        }
    
    def __repr__(self) -> str:
        return (f"Subarray(id={self.subarray_id}, "
                f"num_antennas={self.num_antennas}, "
                f"active={self.is_active})")


class SubarrayPartitioner:
    """
    Partitions an XL-MIMO antenna array into subarrays.
    
    FIXED: array is now on the y-axis (was x-axis).
    """
    
    def __init__(self, config: SubarrayConfig):
        self.config = config
        self.antenna_positions = self._generate_antenna_positions()
        
        self.subarrays: List[Subarray] = []
        self._partition_antennas()
        
        self.activation_mask = np.ones(config.num_subarrays, dtype=np.int8)
        self._sync_subarray_states()
    
    def _generate_antenna_positions(self) -> np.ndarray:
        """
        Generate 3D positions of all antennas (ULA along Y-AXIS).
        
        FIXED: now uses `positions[:, 1]` (y-coordinate) instead of
        `positions[:, 0]` (x-coordinate) to match the rest of the project.
        """
        N = self.config.num_antennas
        spacing = self.config.antenna_spacing * self.config.wavelength
        
        positions = np.zeros((N, 3))
        positions[:, 1] = np.linspace(   # ← ✅ y-axis (was [:, 0])
            -(N - 1) * spacing / 2,
            (N - 1) * spacing / 2,
            N,
        )
        return positions
    
    def _partition_antennas(self):
        """Partition antennas into subarrays based on strategy."""
        strategy = self.config.strategy
        
        if strategy == PartitionStrategy.CONTIGUOUS:
            self._partition_contiguous()
        elif strategy == PartitionStrategy.INTERLEAVED:
            self._partition_interleaved()
        elif strategy == PartitionStrategy.OVERLAPPING:
            self._partition_overlapping()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _partition_contiguous(self):
        """Partition into contiguous blocks of antennas."""
        N = self.config.num_antennas
        K = self.config.num_subarrays
        
        base_size = N // K
        remainder = N % K
        
        idx = 0
        for k in range(K):
            size = base_size + (1 if k < remainder else 0)
            antenna_indices = np.arange(idx, idx + size)
            
            subarray = Subarray(
                subarray_id=k,
                antenna_indices=antenna_indices,
                global_antenna_positions=self.antenna_positions,
            )
            self.subarrays.append(subarray)
            idx += size
    
    def _partition_interleaved(self):
        """Partition into interleaved subarrays (round-robin)."""
        N = self.config.num_antennas
        K = self.config.num_subarrays
        
        for k in range(K):
            antenna_indices = np.arange(k, N, K)
            
            subarray = Subarray(
                subarray_id=k,
                antenna_indices=antenna_indices,
                global_antenna_positions=self.antenna_positions,
            )
            self.subarrays.append(subarray)
    
    def _partition_overlapping(self):
        """Partition into overlapping contiguous subarrays."""
        N = self.config.num_antennas
        K = self.config.num_subarrays
        overlap = self.config.overlap_ratio
        
        denom = K - overlap * (K - 1)
        if denom <= 0:
            raise ValueError(
                f"Overlap ratio too high for {K} subarrays. "
                f"Reduce overlap_ratio."
            )
        
        size = int(np.ceil(N / denom))
        step = int(round(size * (1 - overlap)))
        step = max(1, step)
        
        for k in range(K):
            start = k * step
            end = min(start + size, N)
            
            if start >= N:
                start = max(0, N - size)
                end = N
            
            antenna_indices = np.arange(start, end)
            
            subarray = Subarray(
                subarray_id=k,
                antenna_indices=antenna_indices,
                global_antenna_positions=self.antenna_positions,
            )
            self.subarrays.append(subarray)
    
    def _sync_subarray_states(self):
        """Sync subarray.is_active flags with activation_mask."""
        for k, subarray in enumerate(self.subarrays):
            subarray.is_active = bool(self.activation_mask[k])
    
    # ------------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------------
    
    def set_activation_mask(self, mask: np.ndarray):
        """Set the activation mask and update subarray states."""
        mask = np.asarray(mask, dtype=np.int8)
        
        if mask.shape != (self.config.num_subarrays,):
            raise ValueError(
                f"Activation mask must have shape "
                f"({self.config.num_subarrays},), got {mask.shape}"
            )
        
        if not np.all(np.isin(mask, [0, 1])):
            raise ValueError("Activation mask must contain only 0s and 1s")
        
        self.activation_mask = mask
        self._sync_subarray_states()
    
    def get_activation_mask(self) -> np.ndarray:
        return self.activation_mask.copy()
    
    def get_active_subarrays(self) -> List[Subarray]:
        return [s for s in self.subarrays if s.is_active]
    
    def get_active_antenna_indices(self) -> np.ndarray:
        active = self.get_active_subarrays()
        if not active:
            return np.array([], dtype=int)
        indices = np.concatenate([s.antenna_indices for s in active])
        return np.unique(indices)
    
    def get_num_active_antennas(self) -> int:
        return len(self.get_active_antenna_indices())
    
    def get_effective_channel(self, full_channel: np.ndarray) -> np.ndarray:
        active_indices = self.get_active_antenna_indices()
        if len(active_indices) == 0:
            raise ValueError("No active antennas. Activate at least one subarray.")
        return full_channel[:, active_indices]
    
    def get_subarray_centers(self) -> np.ndarray:
        centers = np.zeros((self.config.num_subarrays, 3))
        for k, s in enumerate(self.subarrays):
            centers[k, :] = s.center_position
        return centers
    
    def validate(self) -> Dict[str, Any]:
        """Validate the partitioner and return diagnostics."""
        N = self.config.num_antennas
        
        usage_count = np.zeros(N, dtype=int)
        for s in self.subarrays:
            usage_count[s.antenna_indices] += 1
        
        covered = np.sum(usage_count > 0)
        coverage = covered / N
        
        return {
            'num_antennas': N,
            'num_subarrays': self.config.num_subarrays,
            'strategy': self.config.strategy.value,
            'overlap_ratio': self.config.overlap_ratio,
            'subarray_sizes': [s.num_antennas for s in self.subarrays],
            'all_antennas_covered': bool(covered == N),
            'coverage': float(coverage),
            'num_unique_antennas': int(covered),
            'max_antenna_usage': int(np.max(usage_count)) if N > 0 else 0,
            'min_antenna_usage': int(np.min(usage_count)) if N > 0 else 0,
            'antenna_usage_count': usage_count.tolist(),
        }
    
    def get_state_representation(self) -> np.ndarray:
        sizes = np.array([s.num_antennas for s in self.subarrays], dtype=float)
        total_active = float(self.get_num_active_antennas())
        
        return np.concatenate([
            self.activation_mask.astype(float),
            sizes,
            [total_active],
        ])
    
    def summary(self) -> str:
        """Return a human-readable summary."""
        return "\n".join([
            f"SubarrayPartitioner Summary",
            f"  Array axis: Y (ULA along y-axis)",
            f"  Num antennas: {self.config.num_antennas}",
            f"  Num subarrays: {self.config.num_subarrays}",
            f"  Strategy: {self.config.strategy.value}",
            f"  Overlap ratio: {self.config.overlap_ratio}",
            f"  Subarray sizes: {[s.num_antennas for s in self.subarrays]}",
            f"  Activation mask: {self.activation_mask.tolist()}",
            f"  Active antennas: {self.get_num_active_antennas()}",
        ])


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def generate_test_scenario() -> Tuple[SubarrayPartitioner, Dict[str, Any]]:
    """Generate a test scenario for validation."""
    config = SubarrayConfig(
        num_antennas=64,
        num_subarrays=8,
        strategy=PartitionStrategy.CONTIGUOUS,
        overlap_ratio=0.0,
    )
    partitioner = SubarrayPartitioner(config)
    info = partitioner.validate()
    return partitioner, info


def test_all_strategies() -> Dict[str, Dict[str, Any]]:
    """Test all three partitioning strategies."""
    results = {}
    for strategy in PartitionStrategy:
        config = SubarrayConfig(
            num_antennas=64,
            num_subarrays=8,
            strategy=strategy,
            overlap_ratio=0.25 if strategy == PartitionStrategy.OVERLAPPING else 0.0,
        )
        partitioner = SubarrayPartitioner(config)
        results[strategy.value] = partitioner.validate()
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Subarray Partitioner Test (Y-Axis Fix)")
    print("=" * 60)
    
    print("\n[Test 1] Contiguous partitioning (64 antennas → 8 subarrays)")
    partitioner, info = generate_test_scenario()
    print(partitioner.summary())
    print(f"\nValidation:")
    print(f"  All antennas covered: {info['all_antennas_covered']}")
    print(f"  Coverage: {info['coverage']:.2%}")
    
    # Y-AXIS CHECK — the fix verification
    print(f"\n[Test 2] Y-Axis ULA verification (backlog #4 fix)")
    s0 = partitioner.subarrays[0]
    print(f"  Subarray 0 antenna positions (first 3):")
    for i in range(3):
        print(f"    {s0.antenna_positions[i]}")
    
    assert np.allclose(s0.antenna_positions[:, 0], 0.0, atol=1e-10), \
        "x-coords should be 0 (Y-axis ULA)"
    assert np.allclose(s0.antenna_positions[:, 2], 0.0, atol=1e-10), \
        "z-coords should be 0"
    assert np.std(s0.antenna_positions[:, 1]) > 0, \
        "y-coords should vary"
    print(f"  ✓ x=0, z=0, y varies → Y-axis ULA confirmed")
    
    print("\n[Test 3] Activation mask test")
    mask = np.array([1, 1, 0, 0, 1, 1, 0, 0], dtype=np.int8)
    partitioner.set_activation_mask(mask)
    print(f"  Set mask: {mask.tolist()}")
    print(f"  Active subarrays: {[s.subarray_id for s in partitioner.get_active_subarrays()]}")
    print(f"  Active antennas: {partitioner.get_num_active_antennas()}")
    
    print("\n[Test 4] All partitioning strategies")
    results = test_all_strategies()
    for strategy_name, result in results.items():
        print(f"  {strategy_name:12s}: sizes={result['subarray_sizes']}, "
              f"coverage={result['coverage']:.0%}, "
              f"max_usage={result['max_antenna_usage']}")
    
    print("\n" + "=" * 60)
    print("✓ Subarray partitioner validation successful!")
    print("=" * 60)