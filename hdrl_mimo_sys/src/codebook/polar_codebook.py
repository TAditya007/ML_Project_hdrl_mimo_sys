"""
Polar-domain codebook generation for near-field XL-MIMO systems.

The polar-domain codebook enables near-field beamforming by parameterizing
beams by BOTH angle θ and range r.

PHYSICS — Key facts that shaped this design:

    Fraunhofer distance:   D_F = 2·D²/λ
    Angular resolution:    Δθ ≈ λ / D ≈ 1.8° (broadside, 64-element ULA)

    Therefore: angle grid step MUST be ≲ 1° to guarantee a beam lands
    inside the main lobe of any target.

ARRAY GEOMETRY (critical):
    The ULA is placed along the Y-AXIS:
        Antenna i position: (0, y_i, 0),  y_i ∈ [−L/2, +L/2]

    The target/beam angle θ is measured from the +x axis:
        Target at (θ, r):  (r·cos θ, r·sin θ, 0)

    With this geometry, +θ and −θ produce DISTINCT distance profiles
    to the antennas (array is symmetric about the x-axis / y=0),
    so the codebook can distinguish positive and negative angles.

    (If the ULA were along the x-axis, +θ and −θ would be indistinguishable,
    which we empirically discovered during testing.)

DESIGN STRATEGY:
    1. Generate a DENSE grid (1° angle step, log-spaced range).
       → ~1200 beams guaranteed to include near-matched beams.
    2. Prune by correlation (threshold 0.95) to ~50–150 distinct beams.
    3. The pruned set is the low-level RL agent's action space.

PHASE CONVENTION (verified):
    Channel:  h_i = exp(-j·2π·d_i/λ) / √N
    Beam:     w_i = exp(-j·2π·d_i/λ) / √N
    → w^H · h = 1 for a matched beam.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class PolarCodebookConfig:
    """Configuration for polar-domain codebook generation."""
    
    num_antennas: int = 64
    antenna_spacing: float = 0.5
    wavelength: float = 3e8 / 28e9
    
    angle_min_deg: float = -60.0
    angle_max_deg: float = 60.0
    num_angles: int = 121               # 1° step over [-60°, +60°]
    
    range_min: float = 5.0
    range_max: float = 40.0
    num_ranges: int = 10
    range_spacing: str = 'log'
    
    correlation_threshold: float = 0.95
    normalize_beams: bool = True
    
    elevation_deg: float = 0.0
    
    def __post_init__(self):
        if self.num_antennas <= 0:
            raise ValueError("num_antennas must be positive")
        if self.num_angles <= 0:
            raise ValueError("num_angles must be positive")
        if self.num_ranges <= 0:
            raise ValueError("num_ranges must be positive")
        if self.range_min >= self.range_max:
            raise ValueError("range_min must be < range_max")
        if self.range_spacing not in ('linear', 'log'):
            raise ValueError("range_spacing must be 'linear' or 'log'")
        if not (0.0 < self.correlation_threshold <= 1.0):
            raise ValueError("correlation_threshold must be in (0, 1]")


# ============================================================================
# BEAM CLASS
# ============================================================================

@dataclass
class PolarBeam:
    """A single near-field beam in polar domain."""
    beam_id: int
    angle_deg: float
    range_m: float
    beam_vector: np.ndarray
    position: np.ndarray
    
    def __repr__(self) -> str:
        return (f"PolarBeam(id={self.beam_id}, "
                f"θ={self.angle_deg:+.2f}°, r={self.range_m:.2f}m)")


# ============================================================================
# CODEBOOK CLASS
# ============================================================================

class PolarCodebook:
    """Polar-domain near-field codebook."""
    
    def __init__(self, config: PolarCodebookConfig,
                 antenna_positions: Optional[np.ndarray] = None):
        self.config = config
        
        if antenna_positions is None:
            self.antenna_positions = self._generate_ula_positions()
        else:
            self.antenna_positions = np.asarray(antenna_positions, dtype=float)
            if self.antenna_positions.shape != (config.num_antennas, 3):
                raise ValueError(
                    f"antenna_positions must have shape "
                    f"({config.num_antennas}, 3), got "
                    f"{self.antenna_positions.shape}"
                )
        
        self.beams: List[PolarBeam] = []
        self._build_codebook()
        self._beam_matrix: Optional[np.ndarray] = None
    
    # ------------------------------------------------------------------------
    # GRID CONSTRUCTION
    # ------------------------------------------------------------------------
    
    def _generate_ula_positions(self) -> np.ndarray:
        """
        Generate ULA positions along the Y-AXIS.
        
        FIXED: ULA is now along y-axis (not x-axis). This breaks the
        θ ↔ −θ symmetry that existed with an x-axis ULA.
        
        Array: (0, y_i, 0), y_i ∈ [−L/2, +L/2]
        Broadside direction is +x (θ=0).
        """
        N = self.config.num_antennas
        d = self.config.antenna_spacing * self.config.wavelength
        
        positions = np.zeros((N, 3))
        positions[:, 1] = np.linspace(-(N-1)*d/2, (N-1)*d/2, N)
        return positions
    
    def _build_codebook(self):
        angles_deg = np.linspace(
            self.config.angle_min_deg,
            self.config.angle_max_deg,
            self.config.num_angles
        )
        
        if self.config.range_spacing == 'log':
            ranges_m = np.logspace(
                np.log10(self.config.range_min),
                np.log10(self.config.range_max),
                self.config.num_ranges
            )
        else:
            ranges_m = np.linspace(
                self.config.range_min,
                self.config.range_max,
                self.config.num_ranges
            )
        
        beam_id = 0
        for angle_deg in angles_deg:
            for range_m in ranges_m:
                beam = self._create_beam(beam_id, angle_deg, range_m)
                self.beams.append(beam)
                beam_id += 1
    
    def _create_beam(self, beam_id: int,
                     angle_deg: float, range_m: float) -> PolarBeam:
        """
        Create a single near-field beam focused at (angle, range).
        
        Angle θ is measured from the +x axis.
        Target position: (r·cos θ, r·sin θ, 0)
        """
        angle_rad = np.radians(angle_deg)
        focal_point = np.array([
            range_m * np.cos(angle_rad),
            range_m * np.sin(angle_rad),
            0.0
        ])
        
        distances = np.linalg.norm(
            self.antenna_positions - focal_point, axis=1
        )
        
        # Beam matches channel phase: exp(-j·2π·d/λ)
        beam_vector = np.exp(
            -1j * 2 * np.pi * distances / self.config.wavelength
        )
        
        if self.config.normalize_beams:
            norm = np.linalg.norm(beam_vector)
            if norm > 0:
                beam_vector = beam_vector / norm
        
        return PolarBeam(
            beam_id=beam_id,
            angle_deg=float(angle_deg),
            range_m=float(range_m),
            beam_vector=beam_vector.astype(np.complex128),
            position=focal_point
        )
    
    # ------------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------------
    
    def __len__(self) -> int:
        return len(self.beams)
    
    def __getitem__(self, idx: int) -> PolarBeam:
        return self.beams[idx]
    
    def get_beam(self, idx: int) -> PolarBeam:
        if idx < 0 or idx >= len(self.beams):
            raise IndexError(f"Beam index {idx} out of range [0, {len(self.beams)})")
        return self.beams[idx]
    
    def get_beam_matrix(self) -> np.ndarray:
        """Return (num_antennas, num_beams) matrix; each column is a beam."""
        if self._beam_matrix is None:
            self._beam_matrix = np.column_stack(
                [b.beam_vector for b in self.beams]
            )
        return self._beam_matrix
    
    def get_angle_range(self) -> Tuple[float, float]:
        return (self.config.angle_min_deg, self.config.angle_max_deg)
    
    def get_range_range(self) -> Tuple[float, float]:
        return (self.config.range_min, self.config.range_max)
    
    def find_beam(self, angle_deg: float, range_m: float) -> int:
        """Find the closest beam index to (angle, range)."""
        best_idx = 0
        best_dist = np.inf
        for i, beam in enumerate(self.beams):
            d_angle = (beam.angle_deg - angle_deg) / max(
                1e-6, self.config.angle_max_deg - self.config.angle_min_deg
            )
            if range_m > 0 and beam.range_m > 0:
                d_range = np.log(beam.range_m / range_m) / np.log(
                    self.config.range_max / self.config.range_min
                )
            else:
                d_range = 0.0
            dist = d_angle ** 2 + d_range ** 2
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        return best_idx
    
    def compute_correlation_matrix(self) -> np.ndarray:
        B = self.get_beam_matrix()
        return np.abs(B.conj().T @ B)
    
    def compute_effective_aperture_check(self) -> Dict[str, float]:
        """Diagnostic: verify the codebook is in the near-field regime."""
        D = np.max(np.linalg.norm(
            self.antenna_positions - np.mean(self.antenna_positions, axis=0),
            axis=1
        )) * 2
        fraunhofer = 2 * D**2 / self.config.wavelength
        angular_resolution_deg = np.degrees(self.config.wavelength / D)
        
        return {
            'aperture_m': float(D),
            'fraunhofer_m': float(fraunhofer),
            'angular_resolution_deg': float(angular_resolution_deg),
            'angle_step_deg': float(
                (self.config.angle_max_deg - self.config.angle_min_deg) /
                max(1, self.config.num_angles - 1)
            ),
            'range_min': float(self.config.range_min),
            'range_max': float(self.config.range_max),
            'has_near_field_region': bool(self.config.range_min < fraunhofer),
            'near_field_fraction': float(
                min(1.0, max(0.0, (fraunhofer - self.config.range_min) /
                            (self.config.range_max - self.config.range_min)))
            )
        }
    
    def prune(self, correlation_threshold: Optional[float] = None,
              verbose: bool = False) -> 'PolarCodebook':
        """Greedy correlation-based pruning."""
        if correlation_threshold is None:
            correlation_threshold = self.config.correlation_threshold
        if len(self.beams) == 0:
            return self
        
        kept: List[PolarBeam] = [self.beams[0]]
        
        for beam in self.beams[1:]:
            max_corr = 0.0
            for kept_beam in kept:
                corr = np.abs(np.vdot(kept_beam.beam_vector, beam.beam_vector))
                if corr > max_corr:
                    max_corr = corr
            if max_corr < correlation_threshold:
                kept.append(beam)
        
        if verbose:
            print(f"Pruned {len(self.beams)} → {len(kept)} beams "
                  f"(threshold={correlation_threshold})")
        
        for i, beam in enumerate(kept):
            beam.beam_id = i
        
        self.beams = kept
        self._beam_matrix = None
        return self
    
    def project_channel(self, channel_vector: np.ndarray) -> np.ndarray:
        """Compute |w^H h|² for each beam."""
        if channel_vector.shape != (self.config.num_antennas,):
            raise ValueError(
                f"channel_vector must have shape ({self.config.num_antennas},), "
                f"got {channel_vector.shape}"
            )
        B = self.get_beam_matrix()
        inner = B.conj().T @ channel_vector
        return np.abs(inner) ** 2
    
    def best_beam_for_channel(self, channel_vector: np.ndarray) -> int:
        powers = self.project_channel(channel_vector)
        return int(np.argmax(powers))
    
    def get_state_representation(self) -> np.ndarray:
        """Compact (num_beams, 3) state representation for the RL agent."""
        rep = np.zeros((len(self.beams), 3))
        for i, beam in enumerate(self.beams):
            rep[i, 0] = beam.angle_deg / max(
                abs(self.config.angle_min_deg), abs(self.config.angle_max_deg)
            )
            rep[i, 1] = np.log10(max(beam.range_m, 1e-6))
            rep[i, 2] = i / max(1, len(self.beams) - 1)
        return rep
    
    def summary(self) -> str:
        angles = [b.angle_deg for b in self.beams]
        ranges = [b.range_m for b in self.beams]
        check = self.compute_effective_aperture_check()
        return "\n".join([
            f"PolarCodebook Summary",
            f"  Array axis: Y (ULA along y-axis)",
            f"  Num antennas: {self.config.num_antennas}",
            f"  Subarray aperture: {check['aperture_m']:.4f} m",
            f"  Fraunhofer distance: {check['fraunhofer_m']:.2f} m",
            f"  Angular resolution: {check['angular_resolution_deg']:.2f}°",
            f"  Angle grid step:    {check['angle_step_deg']:.2f}°",
            f"  Num beams: {len(self.beams)}",
            f"  Angle range: [{min(angles):.2f}°, {max(angles):.2f}°]",
            f"  Range range: [{min(ranges):.2f}m, {max(ranges):.2f}m]",
            f"  Near-field region: {check['near_field_fraction']:.0%}",
        ])


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def generate_test_scenario(dense: bool = False) -> PolarCodebook:
    """Generate a test codebook (dense or pruned)."""
    config = PolarCodebookConfig(
        num_antennas=64,
        num_angles=121,
        num_ranges=10,
        range_min=5.0,
        range_max=40.0,
        range_spacing='log',
        correlation_threshold=0.95,
    )
    cb = PolarCodebook(config)
    if not dense:
        cb.prune(correlation_threshold=0.95)
    return cb


def _channel_to_target(codebook: PolarCodebook,
                       target_angle: float,
                       target_range: float) -> np.ndarray:
    """Build a normalized channel from a target (θ, r)."""
    target_pos = np.array([
        target_range * np.cos(np.radians(target_angle)),
        target_range * np.sin(np.radians(target_angle)),
        0.0
    ])
    distances = np.linalg.norm(
        codebook.antenna_positions - target_pos, axis=1
    )
    channel = np.exp(
        -1j * 2 * np.pi * distances / codebook.config.wavelength
    )
    return channel / np.linalg.norm(channel)


if __name__ == "__main__":
    print("=" * 60)
    print("Polar Codebook Test — Y-Axis ULA Fix")
    print("=" * 60)
    
    # ---------------------------------------------------------------
    # Test 1: Dense codebook generation
    # ---------------------------------------------------------------
    print("\n[Test 1] Dense codebook generation")
    cb_dense = generate_test_scenario(dense=True)
    print(cb_dense.summary())
    print(f"  Dense grid beams: {len(cb_dense)}")
    
    check = cb_dense.compute_effective_aperture_check()
    print(f"\n  Angular resolution: {check['angular_resolution_deg']:.2f}°")
    print(f"  Angle grid step:    {check['angle_step_deg']:.2f}°")
    assert check['angle_step_deg'] <= check['angular_resolution_deg'], (
        f"Angle step exceeds angular resolution — codebook too coarse"
    )
    print("  ✓ Grid step ≤ angular resolution")
    
    # ---------------------------------------------------------------
    # Test 2: Symmetry break — +θ and −θ MUST be different
    # ---------------------------------------------------------------
    print("\n[Test 2] Symmetry break check (+θ vs −θ)")
    for angle in [10.0, 30.0, 50.0]:
        ch_pos = _channel_to_target(cb_dense, +angle, 15.0)
        ch_neg = _channel_to_target(cb_dense, -angle, 15.0)
        
        best_pos = cb_dense.best_beam_for_channel(ch_pos)
        best_neg = cb_dense.best_beam_for_channel(ch_neg)
        
        beam_pos = cb_dense[best_pos]
        beam_neg = cb_dense[best_neg]
        
        print(f"  θ = ±{angle}° (r=15m):")
        print(f"    +θ best: {beam_pos}")
        print(f"    −θ best: {beam_neg}")
        
        assert best_pos != best_neg, \
            f"BUG: +{angle}° and −{angle}° map to same beam!"
        assert abs(beam_pos.angle_deg - angle) < 2.0, \
            f"+{angle}°: best beam at {beam_pos.angle_deg}°"
        assert abs(beam_neg.angle_deg + angle) < 2.0, \
            f"−{angle}°: best beam at {beam_neg.angle_deg}°"
        print(f"    ✓ Distinct beams, both at correct angle")
    
    # ---------------------------------------------------------------
    # Test 3: Matched-beam power on DENSE grid (multi-target)
    # ---------------------------------------------------------------
    print("\n[Test 3] Matched-beam power (DENSE grid, 8 targets)")
    targets = [
        (30.0, 15.0), (-45.0, 8.0), (0.0, 12.0), (15.0, 25.0),
        (-10.0, 30.0), (50.0, 6.0), (45.0, 20.0), (-25.0, 10.0),
    ]
    
    for t_angle, t_range in targets:
        channel = _channel_to_target(cb_dense, t_angle, t_range)
        powers = cb_dense.project_channel(channel)
        best_idx = int(np.argmax(powers))
        best_beam = cb_dense[best_idx]
        
        angle_err = abs(best_beam.angle_deg - t_angle)
        range_ratio = best_beam.range_m / t_range
        
        print(f"  Target (θ={t_angle:+.1f}°, r={t_range:.1f}m): "
              f"{best_beam} | power={powers[best_idx]:.4f} "
              f"| err={angle_err:.2f}° | ratio={range_ratio:.2f}")
        
        assert angle_err <= check['angle_step_deg'] + 0.5, \
            f"Angle error {angle_err:.2f}° too large"
        assert 0.7 < range_ratio < 1.4, \
            f"Range ratio {range_ratio:.2f} outside [0.7, 1.4]"
        assert powers[best_idx] > 0.5, \
            f"Matched beam power {powers[best_idx]:.4f} < 0.5"
    
    print("  ✓ All 8 targets matched correctly")
    
    # ---------------------------------------------------------------
    # Test 4: Pruning
    # ---------------------------------------------------------------
    print("\n[Test 4] Correlation-based pruning (threshold=0.95)")
    cb_pruned = generate_test_scenario(dense=False)
    print(f"  Dense size:  {len(cb_dense)}")
    print(f"  Pruned size: {len(cb_pruned)}")
    print(f"  Retention:   {len(cb_pruned)/len(cb_dense):.1%}")
    
    print("\n  Verifying pruned codebook still works:")
    for t_angle, t_range in targets[:4]:
        channel = _channel_to_target(cb_pruned, t_angle, t_range)
        powers = cb_pruned.project_channel(channel)
        best_idx = int(np.argmax(powers))
        best_beam = cb_pruned[best_idx]
        print(f"    (θ={t_angle:+.1f}°, r={t_range:.1f}m) → "
              f"{best_beam}, power={powers[best_idx]:.4f}")
        assert powers[best_idx] > 0.4, \
            f"Pruned codebook power {powers[best_idx]:.4f} too low"
    print("  ✓ Pruned codebook preserves matched-beam performance")
    
    # ---------------------------------------------------------------
    # Test 5: RL action space
    # ---------------------------------------------------------------
    print("\n[Test 5] RL action space size")
    print(f"  Dense beams (unusable for DQN): {len(cb_dense)}")
    print(f"  Pruned beams (DQN action size): {len(cb_pruned)}")
    
    print("\n" + "=" * 60)
    print("✓ Polar codebook validation successful!")
    print("=" * 60)