"""
Multi-user channel generation for XL-MIMO systems.

This module extends the near-field channel model to support:
- Multi-user channel matrix generation
- Spatial correlation modeling
- Multipath propagation (LOS + NLOS components)
- Channel hardening analysis
- Channel normalization and scaling
- Batch generation for training

The channel generator creates the complete channel matrix H of shape
(num_users, num_antennas) that will be used for precoding and beamforming.
"""
"""
Multi-user channel generation for XL-MIMO systems.

FIXED:
    - generate_user_positions() no longer reseeds on every call.
      The RNG is initialized ONCE at construction (if seed is set),
      so successive calls produce DIFFERENT user positions.
      This makes generate_batch() produce independent realizations.

    - Added reset() method to reseed on demand for reproducibility.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Union, Any
from dataclasses import dataclass, field
import warnings

from src.channel.near_field import NearFieldChannel, ChannelParameters


@dataclass
class ChannelGeneratorConfig:
    """Configuration for channel generator."""
    
    num_users: int = 4
    num_antennas: int = 64
    
    los_probability: float = 0.8
    num_multipath: int = 3
    multipath_spread: float = 10.0
    
    spatial_correlation: bool = True
    correlation_factor: float = 0.3
    
    normalize_channels: bool = True
    scale_factor: float = 1.0
    
    user_distance_range: Tuple[float, float] = (10.0, 100.0)
    user_angle_range: Tuple[float, float] = (-60.0, 60.0)
    
    seed: Optional[int] = None


class ChannelGenerator:
    """
    Multi-user channel generator for XL-MIMO systems.
    
    RNG handling:
        - If `seed` is set in config, RNG is initialized ONCE at construction.
        - Subsequent calls to generate_user_positions() ADVANCE the RNG state,
          giving different positions each call.
        - reset() re-seeds to reproduce a sequence.
    """
    
    def __init__(self, config: ChannelGeneratorConfig):
        self.config = config
        
        # Initialize near-field channel model
        channel_params = ChannelParameters(
            num_antennas=config.num_antennas,
            antenna_spacing=0.5,
            array_type='uniform_linear',
            carrier_frequency=28e9,
            path_loss_exponent=2.0,
        )
        self.near_field = NearFieldChannel(channel_params)
        
        # ✅ FIX: dedicated RNG object — seeded once at construction.
        # This avoids global np.random state pollution and gives
        # reproducible + advancing behavior.
        self._rng = np.random.default_rng(config.seed)
        
        self._user_positions: Optional[np.ndarray] = None
        self._channel_matrix: Optional[np.ndarray] = None
    
    # ------------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------------
    
    def generate_user_positions(
        self, num_users: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate random user positions (advancing the RNG each call).
        
        FIXED: uses self._rng instead of np.random with seed, so repeated
        calls give DIFFERENT positions.
        """
        if num_users is None:
            num_users = self.config.num_users
        
        d_min, d_max = self.config.user_distance_range
        a_min, a_max = self.config.user_angle_range
        
        distances = self._rng.uniform(d_min, d_max, num_users)
        angles = np.radians(self._rng.uniform(a_min, a_max, num_users))
        
        positions = np.zeros((num_users, 3))
        positions[:, 0] = distances * np.cos(angles)
        positions[:, 1] = distances * np.sin(angles)
        positions[:, 2] = 0.0
        
        self._user_positions = positions
        return positions
    
    def generate_channel_matrix(
        self,
        user_positions: Optional[np.ndarray] = None,
        include_multipath: bool = True,
        include_spatial_correlation: Optional[bool] = None,
        normalize: Optional[bool] = None,
    ) -> np.ndarray:
        """Generate the complete channel matrix for all users."""
        if user_positions is None:
            if self._user_positions is None:
                user_positions = self.generate_user_positions()
            else:
                user_positions = self._user_positions
        
        num_users = user_positions.shape[0]
        num_antennas = self.config.num_antennas
        
        H = np.zeros((num_users, num_antennas), dtype=np.complex128)
        
        for u in range(num_users):
            pos = user_positions[u, :].reshape(1, 3)
            h_los = self.near_field.compute_channel(pos).flatten()
            
            if include_multipath:
                h_nlos = self._generate_multipath_components(pos)
                h = np.sqrt(0.6) * h_los + np.sqrt(0.4) * h_nlos
            else:
                h = h_los
            
            H[u, :] = h
        
        if include_spatial_correlation is None:
            include_spatial_correlation = self.config.spatial_correlation
        
        if include_spatial_correlation:
            H = self._apply_spatial_correlation(H)
        
        if normalize is None:
            normalize = self.config.normalize_channels
        
        if normalize:
            H = self._normalize_channel_matrix(H)
        
        H = H * self.config.scale_factor
        self._channel_matrix = H
        return H
    
    def generate_batch(self, batch_size: int) -> np.ndarray:
        """
        Generate a batch of INDEPENDENT channel realizations.
        
        FIXED: now produces distinct channels per batch item because
        generate_user_positions() advances the RNG state.
        """
        batch = np.zeros(
            (batch_size, self.config.num_users, self.config.num_antennas),
            dtype=np.complex128,
        )
        
        for i in range(batch_size):
            positions = self.generate_user_positions()
            H = self.generate_channel_matrix(positions)
            batch[i, :, :] = H
        
        return batch
    
    def reset(self):
        """
        Reset the generator state (re-seed RNG, clear caches).
        
        Use this to reproduce a sequence of channels with the same seed.
        """
        if self.config.seed is not None:
            self._rng = np.random.default_rng(self.config.seed)
        self._user_positions = None
        self._channel_matrix = None
    
    # ------------------------------------------------------------------------
    # INTERNAL — MULTIPATH
    # ------------------------------------------------------------------------
    
    def _generate_multipath_components(self, user_pos: np.ndarray) -> np.ndarray:
        """Generate NLOS multipath components for a single user."""
        num_antennas = self.config.num_antennas
        num_multipath = self.config.num_multipath
        
        center_pos = np.mean(self.near_field.antenna_positions, axis=0)
        los_direction = user_pos.flatten() - center_pos
        los_distance = np.linalg.norm(los_direction)
        
        if los_distance > 0:
            los_direction = los_direction / los_distance
        
        h_multipath = np.zeros(num_antennas, dtype=np.complex128)
        
        for _ in range(num_multipath):
            angle_deviation = self._rng.uniform(
                -np.radians(self.config.multipath_spread),
                np.radians(self.config.multipath_spread),
            )
            distance_variation = self._rng.uniform(0.8, 1.2)
            
            new_direction = np.array([
                los_direction[0] * np.cos(angle_deviation)
                - los_direction[1] * np.sin(angle_deviation),
                los_direction[0] * np.sin(angle_deviation)
                + los_direction[1] * np.cos(angle_deviation),
                los_direction[2],
            ])
            
            multipath_pos = (
                center_pos + new_direction * los_distance * distance_variation
            )
            
            h_component = self.near_field.compute_channel(
                multipath_pos.reshape(1, 3)
            ).flatten()
            
            phase_offset = np.exp(1j * self._rng.uniform(0, 2 * np.pi))
            amplitude = self._rng.uniform(0.1, 0.5)
            
            h_multipath += amplitude * phase_offset * h_component
        
        h_multipath = h_multipath / np.sqrt(num_multipath)
        return h_multipath
    
    # ------------------------------------------------------------------------
    # INTERNAL — SPATIAL CORRELATION
    # ------------------------------------------------------------------------
    
    def _apply_spatial_correlation(self, H: np.ndarray) -> np.ndarray:
        """Apply spatial correlation to the channel matrix."""
        num_antennas = H.shape[1]
        rho = self.config.correlation_factor
        
        R = np.zeros((num_antennas, num_antennas), dtype=np.complex128)
        for i in range(num_antennas):
            for j in range(num_antennas):
                R[i, j] = rho ** abs(i - j)
        
        try:
            L = np.linalg.cholesky(R)
            H_corr = H @ L.conj().T
        except np.linalg.LinAlgError:
            warnings.warn(
                "Correlation matrix not positive definite. Using approximation."
            )
            H_corr = H.copy()
            for i in range(1, num_antennas - 1):
                H_corr[:, i] = (
                    (1 - rho) * H[:, i]
                    + rho * (H[:, i - 1] + H[:, i + 1]) / 2
                )
        
        return H_corr
    
    # ------------------------------------------------------------------------
    # INTERNAL — NORMALIZATION
    # ------------------------------------------------------------------------
    
    def _normalize_channel_matrix(self, H: np.ndarray) -> np.ndarray:
        """Normalize each user's channel to unit average power."""
        H_normalized = H.copy()
        for u in range(H.shape[0]):
            avg_power = np.mean(np.abs(H[u, :]) ** 2)
            if avg_power > 0:
                H_normalized[u, :] = H[u, :] / np.sqrt(avg_power)
        return H_normalized
    
    # ------------------------------------------------------------------------
    # ANALYSIS
    # ------------------------------------------------------------------------
    
    def get_channel_statistics(
        self, H: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Compute channel statistics for analysis."""
        if H is None:
            H = self._channel_matrix
        if H is None:
            raise ValueError("No channel matrix available. Generate one first.")
        
        num_users, num_antennas = H.shape
        
        stats: Dict[str, Any] = {
            'num_users': num_users,
            'num_antennas': num_antennas,
            'mean_power': float(np.mean(np.abs(H) ** 2)),
            'power_per_user': np.mean(np.abs(H) ** 2, axis=1).tolist(),
            'mean_magnitude': float(np.mean(np.abs(H))),
            'std_magnitude': float(np.std(np.abs(H))),
            'condition_number': float(np.linalg.cond(H)),
            'channel_hardening': float(self.compute_channel_hardening(H)),
            'spatial_correlation': float(self._compute_spatial_correlation(H)),
        }
        return stats
    
    def compute_channel_hardening(self, H: np.ndarray) -> float:
        """Compute the channel hardening factor."""
        variations = []
        for u in range(H.shape[0]):
            h = H[u, :]
            avg_power = np.mean(np.abs(h) ** 2)
            if avg_power > 0:
                normalized_h = h / np.sqrt(avg_power)
                variation = (
                    np.std(np.abs(normalized_h) ** 2)
                    / np.mean(np.abs(normalized_h) ** 2)
                )
                variations.append(variation)
        
        if len(variations) > 0 and np.mean(variations) > 0:
            return float(1.0 / np.mean(variations))
        return 1.0
    
    def _compute_spatial_correlation(self, H: np.ndarray) -> float:
        """Compute average spatial correlation between adjacent antennas."""
        num_users, num_antennas = H.shape
        correlations: List[float] = []
        
        for u in range(num_users):
            h = H[u, :]
            # Use full per-antenna power sequence rather than pairs
            powers = np.abs(h) ** 2
            if len(powers) >= 3:
                # Adjacent correlation of the power sequence
                c = np.corrcoef(powers[:-1], powers[1:])[0, 1]
                if not np.isnan(c):
                    correlations.append(float(c))
        
        if len(correlations) > 0:
            return float(np.mean(correlations))
        return 0.0
    
    def get_channel_quality(
        self, H: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Compute channel quality metrics for each user."""
        if H is None:
            H = self._channel_matrix
        if H is None:
            raise ValueError("No channel matrix available. Generate one first.")
        
        num_users = H.shape[0]
        quality: Dict[str, Any] = {
            'snr_per_user': [],
            'sinr_per_user': [],
            'channel_gain_per_user': [],
            'near_field_ratios': [],
        }
        
        for u in range(num_users):
            h = H[u, :]
            gain = np.mean(np.abs(h) ** 2)
            quality['channel_gain_per_user'].append(float(gain))
            
            if self._user_positions is not None:
                pos = self._user_positions[u, :].reshape(1, 3)
                ratio = self.near_field.compute_near_field_ratio(pos)[0]
                quality['near_field_ratios'].append(float(ratio))
            else:
                quality['near_field_ratios'].append(0.0)
        
        return quality
    
    def get_near_field_model(self) -> NearFieldChannel:
        """Return the underlying near-field channel model."""
        return self.near_field


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def generate_test_scenario() -> Tuple[ChannelGenerator, np.ndarray,
                                       Dict[str, Any], Dict[str, Any]]:
    """Generate a test scenario for quick validation."""
    config = ChannelGeneratorConfig(
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
    generator = ChannelGenerator(config)
    H = generator.generate_channel_matrix()
    stats = generator.get_channel_statistics(H)
    quality = generator.get_channel_quality(H)
    return generator, H, stats, quality


if __name__ == "__main__":
    print("=" * 60)
    print("Channel Generator Test (RNG Fixed)")
    print("=" * 60)
    
    generator, H, stats, quality = generate_test_scenario()
    
    print(f"\nConfiguration:")
    print(f"  - Users: {generator.config.num_users}")
    print(f"  - Antennas: {generator.config.num_antennas}")
    print(f"  - Seed: {generator.config.seed}")
    
    print(f"\nChannel Matrix:")
    print(f"  - Shape: {H.shape}")
    print(f"  - Mean power: {stats['mean_power']:.6f}")
    print(f"  - Condition number: {stats['condition_number']:.2f}")
    print(f"  - Hardening factor: {stats['channel_hardening']:.4f}")
    
    print(f"\nBatch Generation (NEW: independent realizations):")
    batch = generator.generate_batch(batch_size=3)
    print(f"  - Batch shape: {batch.shape}")
    print(f"  - ||batch[0] - batch[1]|| = "
          f"{np.linalg.norm(batch[0] - batch[1]):.6f}")
    print(f"  - ||batch[0] - batch[2]|| = "
          f"{np.linalg.norm(batch[0] - batch[2]):.6f}")
    assert np.linalg.norm(batch[0] - batch[1]) > 1e-6, \
        "Batch items should differ"
    print(f"  ✓ Batch items are independent")
    
    print(f"\nDeterminism Test (with reset):")
    generator.reset()
    H_first = generator.generate_channel_matrix()
    generator.reset()
    H_second = generator.generate_channel_matrix()
    diff = np.linalg.norm(H_first - H_second)
    print(f"  - After reset, ||H_first - H_second|| = {diff:.6e}")
    assert diff < 1e-12, "Reset should reproduce the same channel"
    print(f"  ✓ Reset reproduces the same channel")
    
    print("\n✓ Channel generator validation successful!")