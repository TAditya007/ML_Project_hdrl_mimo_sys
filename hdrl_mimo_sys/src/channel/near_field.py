"""
Near-field channel modeling for XL-MIMO systems.

This module implements the spherical wavefront propagation model that is
characteristic of near-field communications in XL-MIMO systems. Unlike
traditional far-field models that assume planar wavefronts, this model
accounts for both angle and distance dependence of the channel.

Key features:
- Spherical wavefront propagation
- Distance-dependent path loss
- Angle-of-arrival/departure (AoA/AoD) calculations
- Near-field vs far-field regime distinction
- Support for multi-user scenarios

References:
- Based on near-field XL-MIMO channel models from recent literature
- Spherical wave assumption for Fresnel region
"""

"""
Near-field channel modeling for XL-MIMO systems.

FIXED:
    1. ULA is now along the Y-AXIS (matching the polar codebook).
       Previously on x-axis, which made +θ and −θ indistinguishable.
    2. ChannelParameters validates on construction.
"""

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
import warnings


@dataclass
class ChannelParameters:
    """Container for channel configuration parameters."""
    
    num_antennas: int = 64
    antenna_spacing: float = 0.5
    array_type: str = 'uniform_linear'
    
    num_rows: Optional[int] = None
    num_cols: Optional[int] = None
    
    carrier_frequency: float = 28e9
    wavelength: Optional[float] = None
    fraunhofer_distance: Optional[float] = None
    path_loss_exponent: float = 2.0
    
    def __post_init__(self):
        """Compute derived parameters AND validate inputs."""
        # ✅ FIX: validate BEFORE computing derived values
        if self.num_antennas <= 0:
            raise ValueError(
                f"num_antennas must be positive, got {self.num_antennas}"
            )
        if self.antenna_spacing <= 0:
            raise ValueError(
                f"antenna_spacing must be positive, got {self.antenna_spacing}"
            )
        if self.carrier_frequency <= 0:
            raise ValueError(
                f"carrier_frequency must be positive, got {self.carrier_frequency}"
            )
        if self.array_type not in ('uniform_linear', 'uniform_planar'):
            raise ValueError(
                f"array_type must be 'uniform_linear' or 'uniform_planar', "
                f"got {self.array_type}"
            )
        
        # Compute wavelength
        if self.wavelength is None:
            self.wavelength = 3e8 / self.carrier_frequency
        
        # Compute Fraunhofer distance
        if self.fraunhofer_distance is None:
            wavelength = self.wavelength
            if self.array_type == 'uniform_linear':
                array_length = self.num_antennas * self.antenna_spacing * wavelength
            else:
                # UPA: aperture = diagonal
                if self.num_rows and self.num_cols:
                    array_length = np.sqrt(
                        (self.num_rows * self.antenna_spacing * wavelength) ** 2 +
                        (self.num_cols * self.antenna_spacing * wavelength) ** 2
                    )
                else:
                    array_length = self.num_antennas * self.antenna_spacing * wavelength
            self.fraunhofer_distance = 2 * (array_length ** 2) / wavelength


class NearFieldChannel:
    """
    Near-field XL-MIMO channel model with spherical wavefront propagation.
    
    Array geometry: ULA along the Y-AXIS (broadside = +x direction).
        Antenna i: (0, y_i, 0)
        Target at (θ, r): (r·cos θ, r·sin θ, 0)
    
    With y-axis ULA, +θ and −θ produce DISTINCT distance profiles.
    """
    
    def __init__(self, params: ChannelParameters):
        self.params = params
        
        if self.params.wavelength is None:
            self.params.wavelength = 3e8 / self.params.carrier_frequency
        
        if self.params.fraunhofer_distance is None:
            wavelength = self.params.wavelength
            array_length = (
                self.params.num_antennas * self.params.antenna_spacing * wavelength
            )
            self.params.fraunhofer_distance = 2 * (array_length ** 2) / wavelength
        
        self.antenna_positions = self._compute_antenna_positions()
        self._validate_parameters()
    
    def _compute_antenna_positions(self) -> np.ndarray:
        """
        Compute 3D positions of all antennas.
        
        ULA: along y-axis (broadside = +x direction).
        UPA: in x-y plane.
        """
        N = self.params.num_antennas
        wavelength = (
            self.params.wavelength
            if self.params.wavelength is not None
            else 3e8 / self.params.carrier_frequency
        )
        spacing = self.params.antenna_spacing * wavelength
        
        if self.params.array_type == 'uniform_linear':
            # ✅ FIXED: ULA now on y-axis (was x-axis)
            positions = np.zeros((N, 3))
            positions[:, 1] = np.linspace(
                -(N - 1) * spacing / 2,
                (N - 1) * spacing / 2,
                N,
            )
            return positions
        
        elif self.params.array_type == 'uniform_planar':
            if self.params.num_rows is None or self.params.num_cols is None:
                raise ValueError("num_rows and num_cols required for UPA")
            
            rows = self.params.num_rows
            cols = self.params.num_cols
            if rows * cols != N:
                raise ValueError(
                    f"rows*cols ({rows*cols}) != num_antennas ({N})"
                )
            
            x_pos = np.linspace(
                -(cols - 1) * spacing / 2, (cols - 1) * spacing / 2, cols
            )
            y_pos = np.linspace(
                -(rows - 1) * spacing / 2, (rows - 1) * spacing / 2, rows
            )
            xx, yy = np.meshgrid(x_pos, y_pos)
            positions = np.zeros((N, 3))
            positions[:, 0] = xx.flatten()
            positions[:, 1] = yy.flatten()
            return positions
        
        else:
            raise ValueError(f"Unsupported array type: {self.params.array_type}")
    
    def _validate_parameters(self):
        """Internal validation (defense in depth; params already validated)."""
        if self.params.num_antennas <= 0:
            raise ValueError("num_antennas must be positive")
        if self.params.antenna_spacing <= 0:
            raise ValueError("antenna_spacing must be positive")
        if self.params.carrier_frequency <= 0:
            raise ValueError("carrier_frequency must be positive")
    
    def get_user_positions(
        self,
        num_users: int,
        distance_range: Tuple[float, float] = (10.0, 100.0),
        angle_range: Tuple[float, float] = (-60.0, 60.0),
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Generate random user positions in 3D."""
        if seed is not None:
            np.random.seed(seed)
        
        distances = np.random.uniform(distance_range[0], distance_range[1], num_users)
        angles = np.random.uniform(
            np.radians(angle_range[0]), np.radians(angle_range[1]), num_users
        )
        
        positions = np.zeros((num_users, 3))
        positions[:, 0] = distances * np.cos(angles)
        positions[:, 1] = distances * np.sin(angles)
        positions[:, 2] = 0.0
        return positions
    
    def compute_channel(
        self,
        user_positions: np.ndarray,
        include_path_loss: bool = True,
        include_phase: bool = True,
    ) -> np.ndarray:
        """Compute the near-field channel matrix."""
        num_users = user_positions.shape[0]
        N = self.params.num_antennas
        wavelength = (
            self.params.wavelength
            if self.params.wavelength is not None
            else 3e8 / self.params.carrier_frequency
        )
        
        H = np.zeros((num_users, N), dtype=np.complex128)
        
        for u in range(num_users):
            user_pos = user_positions[u, :]
            distances = np.linalg.norm(
                self.antenna_positions - user_pos, axis=1
            )
            
            if include_phase:
                phase = np.exp(-1j * 2 * np.pi * distances / wavelength)
            else:
                phase = np.ones_like(distances, dtype=np.complex128)
            
            if include_path_loss:
                path_loss = (wavelength / (4 * np.pi * distances)) ** 2
                path_loss = path_loss * (1.0 / distances) ** self.params.path_loss_exponent
                H[u, :] = np.sqrt(path_loss) * phase
            else:
                H[u, :] = phase
        
        return H
    
    def compute_far_field_channel(self, user_positions: np.ndarray) -> np.ndarray:
        """Compute far-field (planar wavefront) channel for comparison."""
        num_users = user_positions.shape[0]
        N = self.params.num_antennas
        wavelength = (
            self.params.wavelength
            if self.params.wavelength is not None
            else 3e8 / self.params.carrier_frequency
        )
        
        H_far = np.zeros((num_users, N), dtype=np.complex128)
        
        for u in range(num_users):
            user_pos = user_positions[u, :]
            center_pos = np.mean(self.antenna_positions, axis=0)
            direction = user_pos - center_pos
            distance = np.linalg.norm(direction)
            if distance > 0:
                direction = direction / distance
            
            steering_vector = np.zeros(N, dtype=np.complex128)
            for i in range(N):
                projection = np.dot(self.antenna_positions[i, :], direction)
                steering_vector[i] = np.exp(
                    -1j * 2 * np.pi * projection / wavelength
                )
            
            path_loss = (wavelength / (4 * np.pi * distance)) ** 2
            H_far[u, :] = np.sqrt(path_loss) * steering_vector
        
        return H_far
    
    def compute_near_field_ratio(self, user_positions: np.ndarray) -> np.ndarray:
        """Compute near-field ratio for each user."""
        H_near = self.compute_channel(user_positions)
        H_far = self.compute_far_field_channel(user_positions)
        num_users = user_positions.shape[0]
        ratios = np.zeros(num_users)
        
        for u in range(num_users):
            denom = np.linalg.norm(H_near[u, :]) * np.linalg.norm(H_far[u, :])
            if denom > 1e-30:
                correlation = np.abs(np.vdot(H_near[u, :], H_far[u, :])) / denom
                ratios[u] = 1 - correlation
        
        return ratios
    
    def get_regime(self, distance: float) -> str:
        """Classify user distance as near-field / fresnel / far-field."""
        D_f = self.params.fraunhofer_distance
        if D_f is None:
            wavelength = (
                self.params.wavelength
                if self.params.wavelength is not None
                else 3e8 / self.params.carrier_frequency
            )
            array_length = (
                self.params.num_antennas * self.params.antenna_spacing * wavelength
            )
            D_f = 2 * (array_length ** 2) / wavelength
            self.params.fraunhofer_distance = D_f
        
        if distance < D_f / 10:
            return 'near-field'
        elif distance < D_f:
            return 'fresnel'
        else:
            return 'far-field'
    
    def compute_fraunhofer_distance(self) -> float:
        """Return the Fraunhofer distance."""
        if self.params.fraunhofer_distance is None:
            wavelength = (
                self.params.wavelength
                if self.params.wavelength is not None
                else 3e8 / self.params.carrier_frequency
            )
            array_length = (
                self.params.num_antennas * self.params.antenna_spacing * wavelength
            )
            self.params.fraunhofer_distance = (
                2 * (array_length ** 2) / wavelength
            )
        return self.params.fraunhofer_distance
    
    def get_antenna_positions(self) -> np.ndarray:
        """Return antenna positions."""
        return self.antenna_positions.copy()
    
    def validate_channel(
        self, H: np.ndarray, user_positions: np.ndarray
    ) -> Dict:
        """Validate channel properties and return statistics."""
        num_users, num_antennas = H.shape
        stats = {
            'shape': (num_users, num_antennas),
            'mean_magnitude': float(np.mean(np.abs(H))),
            'std_magnitude': float(np.std(np.abs(H))),
            'mean_phase': float(np.mean(np.angle(H))),
            'std_phase': float(np.std(np.angle(H))),
            'power_per_user': np.mean(np.abs(H) ** 2, axis=1),
            'condition_number': float(np.linalg.cond(H)),
            'near_field_users': [],
            'fresnel_users': [],
            'far_field_users': [],
        }
        
        for u in range(num_users):
            distance = np.linalg.norm(user_positions[u, :])
            regime = self.get_regime(distance)
            if regime == 'near-field':
                stats['near_field_users'].append(u)
            elif regime == 'fresnel':
                stats['fresnel_users'].append(u)
            else:
                stats['far_field_users'].append(u)
        
        return stats


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def generate_test_scenario() -> Tuple[NearFieldChannel, np.ndarray, np.ndarray]:
    """Generate a test scenario for quick validation."""
    params = ChannelParameters(
        num_antennas=64,
        antenna_spacing=0.5,
        array_type='uniform_linear',
        carrier_frequency=28e9,
        path_loss_exponent=2.0,
    )
    channel = NearFieldChannel(params)
    
    np.random.seed(42)
    user_positions = channel.get_user_positions(
        num_users=4,
        distance_range=(5.0, 80.0),
        angle_range=(-60.0, 60.0),
        seed=42,
    )
    H = channel.compute_channel(user_positions)
    return channel, user_positions, H


if __name__ == "__main__":
    print("=" * 60)
    print("Near-Field Channel Model Test (Y-Axis ULA)")
    print("=" * 60)
    
    channel, user_positions, H = generate_test_scenario()
    
    print(f"\nAntenna Configuration:")
    print(f"  - Num antennas: {channel.params.num_antennas}")
    print(f"  - Array type: {channel.params.array_type} (y-axis)")
    print(f"  - Carrier: {channel.params.carrier_frequency/1e9:.1f} GHz")
    print(f"  - Fraunhofer: {channel.params.fraunhofer_distance:.2f} m")
    
    print(f"\nAntenna positions (first 3):")
    for i in range(3):
        print(f"  Antenna {i}: {channel.antenna_positions[i]}")
    
    print(f"\nUser Positions:")
    for u, pos in enumerate(user_positions):
        distance = np.linalg.norm(pos)
        regime = channel.get_regime(distance)
        print(f"  User {u}: {pos}, distance={distance:.2f}m, regime={regime}")
    
    print(f"\nChannel Matrix:")
    print(f"  - Shape: {H.shape}")
    print(f"  - Mean magnitude: {np.mean(np.abs(H)):.6f}")
    print(f"  - Max magnitude: {np.max(np.abs(H)):.6f}")
    
    # Test y-axis symmetry break
    print(f"\nSymmetry Break Test (+θ vs −θ):")
    pos_plus = np.array([[10.0, 5.0, 0.0]])
    pos_minus = np.array([[10.0, -5.0, 0.0]])
    H_plus = channel.compute_channel(pos_plus, include_path_loss=False)
    H_minus = channel.compute_channel(pos_minus, include_path_loss=False)
    corr = np.abs(np.vdot(H_plus[0], H_minus[0])) / (
        np.linalg.norm(H_plus[0]) * np.linalg.norm(H_minus[0])
    )
    print(f"  Correlation +y vs −y: {corr:.4f}")
    print(f"  ✓ Distinct channels: {corr < 0.999}")
    
    # Test validation
    print(f"\nValidation Test:")
    try:
        ChannelParameters(num_antennas=0)
        print("  ✗ Should have raised ValueError")
    except ValueError as e:
        print(f"  ✓ Caught: {e}")
    
    print("\n✓ Near-field channel validation successful!")