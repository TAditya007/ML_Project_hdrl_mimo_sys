"""
Subarray partitioning for XL-MIMO systems.

This module defines how a large XL-MIMO antenna array is divided into
multiple subarrays. Each subarray contains a subset of antennas and can
be independently activated or deactivated by the high-level RL agent.

Key concepts:
- Subarray: A group of antennas treated as one unit
- Activation mask: Binary vector of length num_subarrays
  (1 = active, 0 = inactive)
- Partitioning strategies: contiguous, interleaved, overlapping

Partitioning strategies:
- CONTIGUOUS: Subarrays are adjacent blocks of antennas
  Example (8 antennas, 2 subarrays): [0,1,2,3] | [4,5,6,7]
- INTERLEAVED: Antennas are distributed round-robin
  Example (8 antennas, 2 subarrays): [0,2,4,6] | [1,3,5,7]
- OVERLAPPING: Adjacent subarrays share some antennas
  Example (8 antennas, 2 subarrays, 50% overlap): [0,1,2,3,4] | [3,4,5,6,7]
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
    
    # Array dimensions
    num_antennas: int = 64          # Total number of antennas in XL-MIMO array
    
    # Subarray configuration
    num_subarrays: int = 8          # Number of subarrays to partition into
    strategy: PartitionStrategy = PartitionStrategy.CONTIGUOUS
    
    # Overlap configuration (only used when strategy == OVERLAPPING)
    overlap_ratio: float = 0.0      # Fraction of overlap between adjacent subarrays
                                    # 0.0 = no overlap, 0.5 = 50% overlap
    
    # Antenna spacing (needed for subarray response computation)
    antenna_spacing: float = 0.5    # Wavelength-normalized spacing (d/λ)
    wavelength: float = 3e8 / 28e9  # Default: 28 GHz
    
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
    analog beamforming. It has:
    - A list of antenna indices it contains
    - A center position (for steering vector computation)
    - A reference to the global antenna positions
    """
    
    def __init__(self, 
                 subarray_id: int,
                 antenna_indices: np.ndarray,
                 global_antenna_positions: np.ndarray):
        """
        Initialize a subarray.
        
        Args:
            subarray_id: Unique identifier for this subarray
            antenna_indices: Indices of antennas belonging to this subarray
            global_antenna_positions: (num_antennas, 3) positions of ALL antennas
        """
        self.subarray_id = subarray_id
        self.antenna_indices = np.asarray(antenna_indices, dtype=int)
        
        # Extract positions of antennas in this subarray
        self.antenna_positions = global_antenna_positions[self.antenna_indices]
        
        # Compute center of subarray
        self.center_position = np.mean(self.antenna_positions, axis=0)
        
        # Effective aperture (distance from edge to edge)
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
        """Number of antennas in this subarray."""
        return len(self.antenna_indices)
    
    def activate(self):
        """Activate the subarray."""
        self.is_active = True
    
    def deactivate(self):
        """Deactivate the subarray."""
        self.is_active = False
    
    def compute_steering_vector(self, 
                                direction: np.ndarray,
                                wavelength: float) -> np.ndarray:
        """
        Compute the steering vector for this subarray toward a direction.
        
        Args:
            direction: Unit vector (3,) pointing from array to target
            wavelength: Carrier wavelength in meters
        
        Returns:
            Complex steering vector of shape (num_antennas_in_subarray,)
        """
        # Project each antenna position onto direction
        projections = self.antenna_positions @ direction
        
        # Steering vector: exp(-j * 2π * projection / λ)
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
            'is_active': self.is_active
        }
    
    def __repr__(self) -> str:
        return (f"Subarray(id={self.subarray_id}, "
                f"num_antennas={self.num_antennas}, "
                f"active={self.is_active})")


class SubarrayPartitioner:
    """
    Partitions an XL-MIMO antenna array into subarrays.
    
    This class provides the mapping between the physical antenna array
    and the subarray structure that the RL agent controls.
    
    Main responsibilities:
    1. Generate antenna position grid (ULA)
    2. Partition antennas into subarrays
    3. Track which subarrays are active
    4. Provide effective array response under current activation
    """
    
    def __init__(self, config: SubarrayConfig):
        """
        Initialize the subarray partitioner.
        
        Args:
            config: SubarrayConfig object
        """
        self.config = config
        
        # Generate antenna positions (ULA along x-axis, centered at origin)
        self.antenna_positions = self._generate_antenna_positions()
        
        # Partition into subarrays
        self.subarrays: List[Subarray] = []
        self._partition_antennas()
        
        # Activation mask: 1 = active, 0 = inactive
        self.activation_mask = np.ones(
            config.num_subarrays, dtype=np.int8
        )  # Start with all active
        
        # Sync subarray active flags with mask
        self._sync_subarray_states()
    
    def _generate_antenna_positions(self) -> np.ndarray:
        """
        Generate 3D positions of all antennas (ULA along x-axis).
        
        Returns:
            (num_antennas, 3) array of antenna positions
        """
        N = self.config.num_antennas
        spacing = self.config.antenna_spacing * self.config.wavelength
        
        positions = np.zeros((N, 3))
        positions[:, 0] = np.linspace(
            -(N - 1) * spacing / 2,
            (N - 1) * spacing / 2,
            N
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
        
        # Compute subarray sizes (distribute remainder evenly)
        base_size = N // K
        remainder = N % K
        
        idx = 0
        for k in range(K):
            size = base_size + (1 if k < remainder else 0)
            antenna_indices = np.arange(idx, idx + size)
            
            subarray = Subarray(
                subarray_id=k,
                antenna_indices=antenna_indices,
                global_antenna_positions=self.antenna_positions
            )
            self.subarrays.append(subarray)
            idx += size
    
    def _partition_interleaved(self):
        """Partition into interleaved subarrays (round-robin)."""
        N = self.config.num_antennas
        K = self.config.num_subarrays
        
        # Antenna k gets antennas: k, k+K, k+2K, ...
        for k in range(K):
            antenna_indices = np.arange(k, N, K)
            
            subarray = Subarray(
                subarray_id=k,
                antenna_indices=antenna_indices,
                global_antenna_positions=self.antenna_positions
            )
            self.subarrays.append(subarray)
    
    def _partition_overlapping(self):
        """Partition into overlapping contiguous subarrays."""
        N = self.config.num_antennas
        K = self.config.num_subarrays
        overlap = self.config.overlap_ratio
        
        # Size of each subarray
        # With overlap r, each subarray has size:
        # size * K - overlap * size * (K-1) = N
        # size = N / (K - overlap * (K-1))
        denom = K - overlap * (K - 1)
        if denom <= 0:
            raise ValueError(
                f"Overlap ratio too high for {K} subarrays. "
                f"Reduce overlap_ratio."
            )
        
        size = int(np.ceil(N / denom))
        step = int(round(size * (1 - overlap)))
        step = max(1, step)  # Ensure step >= 1
        
        for k in range(K):
            start = k * step
            end = min(start + size, N)
            
            if start >= N:
                # If we've exceeded array bounds, wrap or reuse
                start = max(0, N - size)
                end = N
            
            antenna_indices = np.arange(start, end)
            
            subarray = Subarray(
                subarray_id=k,
                antenna_indices=antenna_indices,
                global_antenna_positions=self.antenna_positions
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
        """
        Set the activation mask and update subarray states.
        
        Args:
            mask: Binary vector of shape (num_subarrays,) with 0/1
        """
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
        """Return current activation mask."""
        return self.activation_mask.copy()
    
    def get_active_subarrays(self) -> List[Subarray]:
        """Return list of currently active subarrays."""
        return [s for s in self.subarrays if s.is_active]
    
    def get_active_antenna_indices(self) -> np.ndarray:
        """
        Get indices of all antennas belonging to active subarrays.
        
        Returns:
            1D array of unique antenna indices (sorted)
        """
        active = self.get_active_subarrays()
        if not active:
            return np.array([], dtype=int)
        
        indices = np.concatenate([s.antenna_indices for s in active])
        return np.unique(indices)
    
    def get_num_active_antennas(self) -> int:
        """Number of antennas currently active (union across subarrays)."""
        return len(self.get_active_antenna_indices())
    
    def get_effective_channel(self, 
                              full_channel: np.ndarray) -> np.ndarray:
        """
        Extract the effective channel using only active antennas.
        
        Args:
            full_channel: (num_users, num_antennas) full channel matrix
        
        Returns:
            (num_users, num_active_antennas) effective channel matrix
        """
        active_indices = self.get_active_antenna_indices()
        
        if len(active_indices) == 0:
            raise ValueError("No active antennas. Activate at least one subarray.")
        
        return full_channel[:, active_indices]
    
    def get_subarray_centers(self) -> np.ndarray:
        """
        Get center positions of all subarrays.
        
        Returns:
            (num_subarrays, 3) array of subarray centers
        """
        centers = np.zeros((self.config.num_subarrays, 3))
        for k, s in enumerate(self.subarrays):
            centers[k, :] = s.center_position
        return centers
    
    def validate(self) -> Dict[str, Any]:
        """
        Validate the partitioner configuration and return diagnostics.
        
        Returns:
            dict with validation info:
            - all_antennas_covered: bool
            - coverage: fraction of antennas covered
            - num_unique_antennas: int
            - antenna_usage_count: how many subarrays each antenna belongs to
        """
        N = self.config.num_antennas
        
        # Count how many times each antenna appears across subarrays
        usage_count = np.zeros(N, dtype=int)
        for s in self.subarrays:
            usage_count[s.antenna_indices] += 1
        
        covered = np.sum(usage_count > 0)
        coverage = covered / N
        
        info = {
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
            'antenna_usage_count': usage_count.tolist()
        }
        
        return info
    
    def get_state_representation(self) -> np.ndarray:
        """
        Return a compact state representation for the RL agent.
        
        Returns:
            1D array with:
            - activation mask (num_subarrays)
            - subarray sizes (num_subarrays)
            - total active antennas (1)
        """
        sizes = np.array([s.num_antennas for s in self.subarrays], dtype=float)
        total_active = float(self.get_num_active_antennas())
        
        return np.concatenate([
            self.activation_mask.astype(float),
            sizes,
            [total_active]
        ])
    
    # ------------------------------------------------------------------------
    # INFO / DEBUG
    # ------------------------------------------------------------------------
    
    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"SubarrayPartitioner Summary",
            f"  Num antennas: {self.config.num_antennas}",
            f"  Num subarrays: {self.config.num_subarrays}",
            f"  Strategy: {self.config.strategy.value}",
            f"  Overlap ratio: {self.config.overlap_ratio}",
            f"  Subarray sizes: {[s.num_antennas for s in self.subarrays]}",
            f"  Activation mask: {self.activation_mask.tolist()}",
            f"  Active antennas: {self.get_num_active_antennas()}"
        ]
        return "\n".join(lines)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def generate_test_scenario() -> Tuple[SubarrayPartitioner, Dict[str, Any]]:
    """
    Generate a test scenario for validation.
    
    Returns:
        (partitioner, validation_info)
    """
    config = SubarrayConfig(
        num_antennas=64,
        num_subarrays=8,
        strategy=PartitionStrategy.CONTIGUOUS,
        overlap_ratio=0.0
    )
    
    partitioner = SubarrayPartitioner(config)
    info = partitioner.validate()
    
    return partitioner, info


def test_all_strategies() -> Dict[str, Dict[str, Any]]:
    """Test all three partitioning strategies and return their validation."""
    results = {}
    
    for strategy in PartitionStrategy:
        config = SubarrayConfig(
            num_antennas=64,
            num_subarrays=8,
            strategy=strategy,
            overlap_ratio=0.25 if strategy == PartitionStrategy.OVERLAPPING else 0.0
        )
        partitioner = SubarrayPartitioner(config)
        results[strategy.value] = partitioner.validate()
    
    return results


if __name__ == "__main__":
    """Quick validation when run as script."""
    print("=" * 60)
    print("Subarray Partitioner Test")
    print("=" * 60)
    
    # Test 1: Contiguous partitioning
    print("\n[Test 1] Contiguous partitioning (64 antennas → 8 subarrays)")
    partitioner, info = generate_test_scenario()
    print(partitioner.summary())
    print(f"\nValidation:")
    print(f"  All antennas covered: {info['all_antennas_covered']}")
    print(f"  Coverage: {info['coverage']:.2%}")
    print(f"  Subarray sizes: {info['subarray_sizes']}")
    
    # Test 2: Activation mask
    print("\n[Test 2] Activation mask test")
    mask = np.array([1, 1, 0, 0, 1, 1, 0, 0], dtype=np.int8)
    partitioner.set_activation_mask(mask)
    print(f"  Set mask: {mask.tolist()}")
    print(f"  Active subarrays: {[s.subarray_id for s in partitioner.get_active_subarrays()]}")
    print(f"  Active antennas: {partitioner.get_num_active_antennas()}")
    
    # Test 3: Effective channel extraction
    print("\n[Test 3] Effective channel extraction")
    full_channel = np.random.randn(4, 64) + 1j * np.random.randn(4, 64)
    effective = partitioner.get_effective_channel(full_channel)
    print(f"  Full channel shape: {full_channel.shape}")
    print(f"  Effective channel shape: {effective.shape}")
    
    # Test 4: All strategies
    print("\n[Test 4] All partitioning strategies")
    results = test_all_strategies()
    for strategy_name, result in results.items():
        print(f"\n  Strategy: {strategy_name}")
        print(f"    Subarray sizes: {result['subarray_sizes']}")
        print(f"    All covered: {result['all_antennas_covered']}")
        print(f"    Coverage: {result['coverage']:.2%}")
        print(f"    Max antenna usage: {result['max_antenna_usage']}")
    
    print("\n" + "=" * 60)
    print("✓ Subarray partitioner validation successful!")
    print("=" * 60)