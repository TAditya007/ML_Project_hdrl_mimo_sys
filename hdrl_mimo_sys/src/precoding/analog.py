"""
Analog precoder for hybrid beamforming in near-field XL-MIMO systems.

The analog precoder F_RF implements the analog (RF-chain) beamforming stage
of hybrid beamforming. Given:
    - A set of active subarrays (from the high-level RL agent)
    - One beam index per active subarray (from the low-level RL agent)
    - A polar-domain codebook

...this module builds the block-diagonal analog precoder matrix:

    F_RF ∈ ℂ^(N_total × K_active)

where:
    N_total   = total number of BS antennas
    K_active  = number of active subarrays (= number of RF chains used)

Structure:
    F_RF has one COLUMN per active subarray. Column k contains the
    beam weights for subarray k's antennas, and ZEROS elsewhere. This
    reflects the fact that each RF chain only drives the antennas of
    its own subarray.

Hardware constraint (phase shifters):
    In real hybrid beamforming, analog phase shifters have a unit-modulus
    constraint: |F_RF[i, j]| ∈ {0} ∪ {1/√N_per_sub} for antennas that belong
    to subarray j. We enforce this by construction — the codebook beams are
    already normalized, and zeros are placed for non-member antennas.

References:
    - Hybrid precoding for mmWave MIMO (Heath et al., 2016)
    - Near-field analog beamforming in XL-MIMO
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
import warnings

from src.codebook.subarray import SubarrayPartitioner, Subarray
from src.codebook.polar_codebook import PolarCodebook, PolarBeam


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class AnalogPrecoderConfig:
    """Configuration for analog precoder construction."""
    
    # Total number of BS antennas
    num_antennas: int = 64
    
    # Whether to enforce unit-modulus constraint
    enforce_unit_modulus: bool = True
    
    # Whether to apply per-column power normalization
    normalize_columns: bool = True
    
    # Small epsilon for division safety
    eps: float = 1e-12


# ============================================================================
# ANALOG PRECODER CLASS
# ============================================================================

class AnalogPrecoder:
    """
    Analog precoder builder for hybrid beamforming.
    
    Given the subarray structure, a codebook, and selected beam indices,
    this class builds the F_RF matrix that maps RF-chain signals to the
    antenna array.
    
    Usage:
        precoder = AnalogPrecoder(config)
        F_RF = precoder.build(
            subarray_partitioner=partitioner,
            codebook=codebook,
            selected_beams={0: 42, 1: 88, 4: 103},  # subarray_id -> beam_id
        )
    """
    
    def __init__(self, config: AnalogPrecoderConfig):
        self.config = config
        self._last_F_RF: Optional[np.ndarray] = None
        self._last_info: Optional[Dict[str, Any]] = None
    
    # ------------------------------------------------------------------------
    # MAIN BUILD METHOD
    # ------------------------------------------------------------------------
    
    def build(self,
              subarray_partitioner: SubarrayPartitioner,
              codebook: PolarCodebook,
              selected_beams: Dict[int, int]) -> np.ndarray:
        """
        Build the analog precoder matrix F_RF.
        
        Args:
            subarray_partitioner: Provides subarray-to-antenna mapping
            codebook: Polar codebook to look up beam vectors
            selected_beams: Dict mapping subarray_id → beam_id
                           Only active subarrays should be present.
        
        Returns:
            F_RF: Complex matrix of shape (N_total, K_active),
                  where K_active = len(selected_beams)
        
        Raises:
            ValueError: If subarray IDs invalid or beam indices out of range
        """
        N = self.config.num_antennas
        K = len(selected_beams)
        
        if K == 0:
            warnings.warn("No active subarrays; returning empty precoder")
            self._last_F_RF = np.zeros((N, 0), dtype=np.complex128)
            self._last_info = {'num_active': 0}
            return self._last_F_RF
        
        # Validate subarray IDs
        num_subarrays = subarray_partitioner.config.num_subarrays
        for sid in selected_beams.keys():
            if sid < 0 or sid >= num_subarrays:
                raise ValueError(
                    f"Subarray ID {sid} out of range [0, {num_subarrays})"
                )
        
        # Validate beam indices
        for sid, bid in selected_beams.items():
            if bid < 0 or bid >= len(codebook):
                raise ValueError(
                    f"Beam ID {bid} for subarray {sid} out of range "
                    f"[0, {len(codebook)})"
                )
        
        # Build F_RF column by column
        F_RF = np.zeros((N, K), dtype=np.complex128)
        
        # Sort subarrays for consistent column ordering
        sorted_sids = sorted(selected_beams.keys())
        
        column_info: List[Dict[str, Any]] = []
        for col_idx, sid in enumerate(sorted_sids):
            bid = selected_beams[sid]
            subarray = subarray_partitioner.subarrays[sid]
            beam = codebook[bid]
            
            # Sanity: beam vector length should match subarray size
            n_sub_antennas = subarray.num_antennas
            if beam.beam_vector.shape[0] != n_sub_antennas:
                # If codebook was generated for a different antenna count
                # than the subarray, we need to handle this gracefully.
                # In the standard setup they should match.
                if beam.beam_vector.shape[0] == codebook.config.num_antennas:
                    # Beam vector was built for the whole array; slice to
                    # this subarray's antennas (relative to subarray center)
                    warnings.warn(
                        f"Codebook was built for full array but subarray "
                        f"{sid} has {n_sub_antennas} antennas. "
                        f"Using per-subarray channel geometry to re-derive "
                        f"beam weights."
                    )
                    # Fallback: regenerate beam weights on the subarray geometry
                    beam_weights = self._regenerate_beam_for_subarray(
                        subarray, beam
                    )
                else:
                    raise ValueError(
                        f"Beam vector size {beam.beam_vector.shape[0]} "
                        f"does not match subarray {sid} size {n_sub_antennas}"
                    )
            else:
                beam_weights = beam.beam_vector.copy()
            
            # Enforce unit-modulus constraint (phase-shifter hardware)
            if self.config.enforce_unit_modulus:
                beam_weights = self._enforce_unit_modulus(beam_weights)
            
            # Place weights at the subarray's antenna indices
            antenna_indices = subarray.antenna_indices
            F_RF[antenna_indices, col_idx] = beam_weights
            
            column_info.append({
                'column': col_idx,
                'subarray_id': sid,
                'beam_id': bid,
                'beam_angle_deg': beam.angle_deg,
                'beam_range_m': beam.range_m,
                'num_antennas': n_sub_antennas,
                'antenna_indices': antenna_indices.tolist(),
            })
        
        # Per-column normalization
        if self.config.normalize_columns:
            F_RF = self._normalize_columns(F_RF)
        
        # Cache
        self._last_F_RF = F_RF
        self._last_info = {
            'num_active': K,
            'N_total': N,
            'columns': column_info,
            'shape': F_RF.shape,
        }
        
        return F_RF
    
    # ------------------------------------------------------------------------
    # HARDWARE CONSTRAINTS
    # ------------------------------------------------------------------------
    
    def _enforce_unit_modulus(self, weights: np.ndarray) -> np.ndarray:
        """
        Enforce unit-modulus constraint on beam weights.
        
        Real phase shifters can only change phase, not amplitude.
        We preserve phase and set magnitude to 1/√M (so column has unit norm).
        
        Args:
            weights: Complex weight vector of length M
        
        Returns:
            Unit-modulus weights of the same shape
        """
        M = len(weights)
        if M == 0:
            return weights
        
        # Preserve phase, set magnitude to 1/√M
        phases = np.angle(weights)
        target_magnitude = 1.0 / np.sqrt(M)
        
        return target_magnitude * np.exp(1j * phases)
    
    def _normalize_columns(self, F: np.ndarray) -> np.ndarray:
        """
        Normalize each column of F_RF to unit norm.
        
        This is standard in hybrid beamforming: F_RF^H F_RF should have
        unit diagonal to preserve total transmit power.
        """
        F_norm = F.copy()
        for j in range(F.shape[1]):
            col_norm = np.linalg.norm(F[:, j])
            if col_norm > self.config.eps:
                F_norm[:, j] = F[:, j] / col_norm
        return F_norm
    
    def _regenerate_beam_for_subarray(self,
                                       subarray: Subarray,
                                       beam: PolarBeam) -> np.ndarray:
        """
        Regenerate the beam weights for a specific subarray geometry.
        
        This is used when the codebook was built for the full array but
        we need per-subarray weights. We use the subarray's antenna
        positions and the beam's (angle, range) to recompute distances.
        """
        wavelength = 28e9  # Assume 28 GHz default (extend if needed)
        c = 3e8
        lam = c / wavelength if wavelength > 1e9 else wavelength
        
        # Get focal point
        angle_rad = np.radians(beam.angle_deg)
        focal_point = np.array([
            beam.range_m * np.cos(angle_rad),
            beam.range_m * np.sin(angle_rad),
            0.0
        ])
        
        # Compute distances from subarray antennas to focal point
        distances = np.linalg.norm(
            subarray.antenna_positions - focal_point, axis=1
        )
        
        weights = np.exp(-1j * 2 * np.pi * distances / lam)
        norm = np.linalg.norm(weights)
        if norm > 0:
            weights = weights / norm
        
        return weights
    
    # ------------------------------------------------------------------------
    # ANALYSIS HELPERS
    # ------------------------------------------------------------------------
    
    def get_last_F_RF(self) -> Optional[np.ndarray]:
        """Return the most recently built F_RF (or None)."""
        return self._last_F_RF
    
    def get_last_info(self) -> Optional[Dict[str, Any]]:
        """Return metadata about the most recent build."""
        return self._last_info
    
    def compute_effective_channel(self,
                                   channel: np.ndarray,
                                   F_RF: Optional[np.ndarray] = None
                                   ) -> np.ndarray:
        """
        Compute the effective channel H_eff = H · F_RF.
        
        Args:
            channel: (num_users, N_total) full channel matrix
            F_RF: Analog precoder (uses last built if None)
        
        Returns:
            H_eff: (num_users, K_active) effective channel
        """
        if F_RF is None:
            F_RF = self._last_F_RF
        if F_RF is None:
            raise ValueError("No F_RF available. Call build() first.")
        
        if channel.shape[1] != F_RF.shape[0]:
            raise ValueError(
                f"Channel has {channel.shape[1]} antennas but F_RF expects "
                f"{F_RF.shape[0]}"
            )
        
        return channel @ F_RF
    
    def analyze_power_allocation(self,
                                  F_RF: Optional[np.ndarray] = None
                                  ) -> Dict[str, Any]:
        """
        Analyze how transmit power is distributed across antennas and subarrays.
        
        Returns:
            dict with per-antenna power, per-column power, and constraints check.
        """
        if F_RF is None:
            F_RF = self._last_F_RF
        if F_RF is None:
            raise ValueError("No F_RF available. Call build() first.")
        
        N, K = F_RF.shape
        
        # Per-antenna total power
        per_antenna_power = np.sum(np.abs(F_RF) ** 2, axis=1)
        
        # Per-column power (should each be 1 if normalized)
        per_column_power = np.sum(np.abs(F_RF) ** 2, axis=0)
        
        # Unit modulus check: each nonzero entry should have magnitude
        # equal to the target (1/√M for its column's M)
        unit_modulus_ok = True
        for j in range(K):
            nonzero = np.abs(F_RF[:, j]) > 1e-9
            mags = np.abs(F_RF[nonzero, j])
            if len(mags) > 0 and np.std(mags) > 1e-6:
                unit_modulus_ok = False
                break
        
        return {
            'shape': (N, K),
            'per_antenna_power': per_antenna_power.tolist(),
            'per_column_power': per_column_power.tolist(),
            'total_power': float(np.sum(per_antenna_power)),
            'unit_modulus_satisfied': bool(unit_modulus_ok),
            'num_antennas_used': int(np.sum(per_antenna_power > 1e-9)),
        }
    
    def summary(self) -> str:
        """Human-readable summary of the last build."""
        if self._last_info is None:
            return "AnalogPrecoder: no build performed yet"
        
        info = self._last_info
        lines = [
            f"AnalogPrecoder Summary",
            f"  Shape: {info['shape']}",
            f"  Active subarrays: {info['num_active']}",
            f"  Total antennas: {info['N_total']}",
            f"  Columns:",
        ]
        for col in info['columns']:
            lines.append(
                f"    [{col['column']}] subarray={col['subarray_id']}, "
                f"beam={col['beam_id']} "
                f"(θ={col['beam_angle_deg']:+.1f}°, r={col['beam_range_m']:.1f}m), "
                f"{col['num_antennas']} antennas"
            )
        return "\n".join(lines)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def build_analog_precoder(subarray_partitioner: SubarrayPartitioner,
                           codebook: PolarCodebook,
                           selected_beams: Dict[int, int],
                           config: Optional[AnalogPrecoderConfig] = None
                           ) -> Tuple[np.ndarray, AnalogPrecoder]:
    """Convenience function to build an analog precoder."""
    if config is None:
        config = AnalogPrecoderConfig(
            num_antennas=subarray_partitioner.config.num_antennas
        )
    precoder = AnalogPrecoder(config)
    F_RF = precoder.build(subarray_partitioner, codebook, selected_beams)
    return F_RF, precoder


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    """Quick validation when run as script."""
    from src.codebook.subarray import (
        SubarrayConfig, SubarrayPartitioner, PartitionStrategy
    )
    from src.codebook.polar_codebook import PolarCodebookConfig, PolarCodebook
    
    print("=" * 60)
    print("Analog Precoder Test")
    print("=" * 60)
    
    # ------------------------------------------------------------------
    # Setup: subarray partitioner + codebook (matched to subarray size)
    # ------------------------------------------------------------------
    N_TOTAL = 64
    N_SUB = 8
    
    print(f"\n[Setup] {N_TOTAL} antennas → {N_SUB} subarrays")
    
    sub_config = SubarrayConfig(
        num_antennas=N_TOTAL,
        num_subarrays=N_SUB,
        strategy=PartitionStrategy.CONTIGUOUS,
    )
    partitioner = SubarrayPartitioner(sub_config)
    
    print(f"  Subarray sizes: {[s.num_antennas for s in partitioner.subarrays]}")
    print(f"  Each subarray has {N_SUB//1} antennas")
    
    # Build a codebook per subarray (same geometry as subarray)
    subarray_size = partitioner.subarrays[0].num_antennas  # typically 8
    
    # NOTE: In this project we'll eventually use ONE codebook per subarray
    # For this test, we use a shared codebook matched to the subarray size
    print(f"\n  Building codebook for {subarray_size}-antenna subarray geometry...")
    
    # Build codebook matched to subarray antenna count
    subarray_positions = partitioner.subarrays[0].antenna_positions
    codebook_config = PolarCodebookConfig(
        num_antennas=subarray_size,
        antenna_spacing=0.5,
        wavelength=3e8 / 28e9,
        angle_min_deg=-60.0,
        angle_max_deg=60.0,
        num_angles=13,       # reduce for speed
        range_min=5.0,
        range_max=20.0,      # smaller range for 8-antenna subarray
        num_ranges=5,
        range_spacing='log',
    )
    codebook = PolarCodebook(
        codebook_config,
        antenna_positions=None,  # auto-generate 8-ant ULA
    )
    print(f"  Codebook size: {len(codebook)} beams")
    
    # ------------------------------------------------------------------
    # Build F_RF
    # ------------------------------------------------------------------
    print("\n[Test 1] Build analog precoder")
    selected_beams = {
        0: 5,    # subarray 0 uses beam 5
        1: 8,    # subarray 1 uses beam 8
        4: 3,    # subarray 4 uses beam 3
        5: 10,   # subarray 5 uses beam 10
    }
    print(f"  Selected beams: {selected_beams}")
    
    precoder_config = AnalogPrecoderConfig(
        num_antennas=N_TOTAL,
        enforce_unit_modulus=True,
        normalize_columns=True,
    )
    precoder = AnalogPrecoder(precoder_config)
    F_RF = precoder.build(partitioner, codebook, selected_beams)
    
    print(f"\n  F_RF shape: {F_RF.shape}")
    print(f"  Expected: ({N_TOTAL}, {len(selected_beams)})")
    assert F_RF.shape == (N_TOTAL, len(selected_beams))
    print("  ✓ Shape correct")
    
    print("\n" + precoder.summary())
    
    # ------------------------------------------------------------------
    # Test 2: Unit modulus constraint
    # ------------------------------------------------------------------
    print("\n[Test 2] Unit modulus constraint")
    analysis = precoder.analyze_power_allocation()
    print(f"  Unit modulus satisfied: {analysis['unit_modulus_satisfied']}")
    print(f"  Num antennas used: {analysis['num_antennas_used']}")
    print(f"  Total power: {analysis['total_power']:.4f}")
    assert analysis['unit_modulus_satisfied']
    print("  ✓ All nonzero entries have equal magnitude (phase-shifter constraint)")
    
    # ------------------------------------------------------------------
    # Test 3: Column sparsity (each column only touches its subarray's antennas)
    # ------------------------------------------------------------------
    print("\n[Test 3] Column sparsity")
    for j, (sid, bid) in enumerate(sorted(selected_beams.items())):
        nonzero_rows = np.where(np.abs(F_RF[:, j]) > 1e-9)[0]
        expected_rows = partitioner.subarrays[sid].antenna_indices
        
        print(f"  Col {j} (subarray {sid}): {len(nonzero_rows)} nonzeros, "
              f"expected {len(expected_rows)}")
        assert len(nonzero_rows) == len(expected_rows)
        assert np.all(np.isin(nonzero_rows, expected_rows))
    print("  ✓ Each column only touches its own subarray's antennas")
    
    # ------------------------------------------------------------------
    # Test 4: Effective channel
    # ------------------------------------------------------------------
    print("\n[Test 4] Effective channel H_eff = H · F_RF")
    num_users = 4
    H_full = (np.random.randn(num_users, N_TOTAL) + 
              1j * np.random.randn(num_users, N_TOTAL))
    H_eff = precoder.compute_effective_channel(H_full)
    print(f"  H_full shape:  {H_full.shape}")
    print(f"  F_RF shape:    {F_RF.shape}")
    print(f"  H_eff shape:   {H_eff.shape}")
    assert H_eff.shape == (num_users, len(selected_beams))
    print("  ✓ Effective channel computed correctly")
    
    # ------------------------------------------------------------------
    # Test 5: Per-column power distribution
    # ------------------------------------------------------------------
    print("\n[Test 5] Column power normalization")
    col_powers = analysis['per_column_power']
    for j, p in enumerate(col_powers):
        print(f"  Col {j}: power = {p:.6f}")
        assert abs(p - 1.0) < 1e-6, f"Column {j} not normalized"
    print("  ✓ All columns have unit power")
    
    # ------------------------------------------------------------------
    # Test 6: Empty case
    # ------------------------------------------------------------------
    print("\n[Test 6] Empty case (no active subarrays)")
    F_empty = precoder.build(partitioner, codebook, {})
    print(f"  Empty F_RF shape: {F_empty.shape}")
    assert F_empty.shape == (N_TOTAL, 0)
    print("  ✓ Handles zero active subarrays")
    
    print("\n" + "=" * 60)
    print("✓ Analog precoder validation successful!")
    print("=" * 60)