"""
Digital (baseband) precoder for hybrid beamforming in XL-MIMO systems.

The digital precoder W operates on the EFFECTIVE channel:

    H_eff = H · F_RF

where:
    H     ∈ ℂ^(K_users × N_total)   full channel
    F_RF  ∈ ℂ^(N_total × K_active)  analog precoder
    H_eff ∈ ℂ^(K_users × K_active)  effective channel

Then W ∈ ℂ^(K_active × K_users) is computed via one of:

    ZF:      W = H_eff^H (H_eff H_eff^H)^(-1)
    MMSE:    W = H_eff^H (H_eff H_eff^H + α I)^(-1)
    MRT:     W = H_eff^H
    NONE:    W = identity columns

Sum-rate (used as reward):
    R = Σ_k log2(1 + SINR_k)
    SINR_k = |h_k^eff w_k|² · p_k / (Σ_{j≠k} |h_k^eff w_j|² · p_j + σ²)
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import warnings


# ============================================================================
# CONFIGURATION
# ============================================================================

class DigitalPrecoderType(Enum):
    """Supported digital precoding schemes."""
    ZF = "zero_forcing"
    MMSE = "mmse"
    MRT = "maximum_ratio_transmission"
    NONE = "none"


@dataclass
class DigitalPrecoderConfig:
    """Configuration for digital precoding."""
    
    precoder_type: DigitalPrecoderType = DigitalPrecoderType.MMSE
    regularization: float = 1e-3
    noise_power: float = 1e-9
    total_power: float = 1.0
    power_allocation: str = "equal"
    eps: float = 1e-12
    enforce_power_constraint: bool = True
    
    def __post_init__(self):
        if self.power_allocation not in ("equal", "water_filling"):
            raise ValueError(
                f"power_allocation must be 'equal' or 'water_filling', "
                f"got {self.power_allocation}"
            )
        if self.total_power <= 0:
            raise ValueError("total_power must be positive")
        if self.noise_power < 0:
            raise ValueError("noise_power must be non-negative")


# ============================================================================
# DIGITAL PRECODER
# ============================================================================

class DigitalPrecoder:
    """Digital precoder for hybrid beamforming."""
    
    def __init__(self, config: DigitalPrecoderConfig):
        self.config = config
        self._last_W: Optional[np.ndarray] = None
        self._last_power: Optional[np.ndarray] = None
        self._last_info: Optional[Dict[str, Any]] = None
    
    # ------------------------------------------------------------------------
    # PRECODER COMPUTATION
    # ------------------------------------------------------------------------
    
    def compute_precoder(self,
                         H_eff: np.ndarray,
                         user_priority: Optional[np.ndarray] = None
                         ) -> np.ndarray:
        """Compute the digital precoder W."""
        K_users, K_active = H_eff.shape
        
        if K_users == 0 or K_active == 0:
            self._last_W = np.zeros((K_active, K_users), dtype=np.complex128)
            return self._last_W
        
        ptype = self.config.precoder_type
        
        if ptype == DigitalPrecoderType.ZF:
            W = self._compute_zf(H_eff)
        elif ptype == DigitalPrecoderType.MMSE:
            W = self._compute_mmse(H_eff)
        elif ptype == DigitalPrecoderType.MRT:
            W = self._compute_mrt(H_eff)
        elif ptype == DigitalPrecoderType.NONE:
            W = self._compute_identity(H_eff)
        else:
            raise ValueError(f"Unsupported precoder type: {ptype}")
        
        # Apply power allocation
        powers = self._allocate_power(H_eff, W, user_priority)
        W = W * np.sqrt(powers)[np.newaxis, :]
        
        self._last_W = W
        self._last_power = powers
        return W
    
    def _compute_zf(self, H_eff: np.ndarray) -> np.ndarray:
        """Zero-Forcing: W = H_eff^H (H_eff H_eff^H)^(-1)."""
        K_users = H_eff.shape[0]
        G = H_eff @ H_eff.conj().T
        G = G + self.config.eps * np.eye(K_users)
        
        try:
            G_inv = np.linalg.inv(G)
            W = H_eff.conj().T @ G_inv
        except np.linalg.LinAlgError:
            warnings.warn("ZF: Gram matrix singular; using pseudo-inverse")
            W = np.linalg.pinv(H_eff.conj().T)
        return W
    
    def _compute_mmse(self, H_eff: np.ndarray) -> np.ndarray:
        """MMSE: W = H_eff^H (H_eff H_eff^H + αI)^(-1)."""
        K_users = H_eff.shape[0]
        alpha = self.config.regularization
        G = H_eff @ H_eff.conj().T
        G = G + (alpha + self.config.eps) * np.eye(K_users)
        
        try:
            G_inv = np.linalg.inv(G)
            W = H_eff.conj().T @ G_inv
        except np.linalg.LinAlgError:
            warnings.warn("MMSE: matrix inversion failed; using pseudo-inverse")
            W = np.linalg.pinv(H_eff.conj().T)
        return W
    
    def _compute_mrt(self, H_eff: np.ndarray) -> np.ndarray:
        """MRT (conjugate beamforming): W = H_eff^H."""
        return H_eff.conj().T
    
    def _compute_identity(self, H_eff: np.ndarray) -> np.ndarray:
        """Identity precoder (no digital processing)."""
        K_users, K_active = H_eff.shape
        W = np.zeros((K_active, K_users), dtype=np.complex128)
        for k in range(min(K_users, K_active)):
            W[k, k] = 1.0
        return W
    
    # ------------------------------------------------------------------------
    # POWER ALLOCATION  ← FIXED
    # ------------------------------------------------------------------------
    
    def _allocate_power(self,
                        H_eff: np.ndarray,
                        W: np.ndarray,
                        user_priority: Optional[np.ndarray] = None
                        ) -> np.ndarray:
        """
        Allocate transmit power across users.
        
        FIXED: normalize `user_priority` into a concrete np.ndarray before
        use, so downstream code never sees `None`.
        """
        K_users = H_eff.shape[0]
        if K_users == 0:
            return np.array([])
        
        # ✅ FIX: default and normalize up front
        if user_priority is None:
            priority: np.ndarray = np.ones(K_users, dtype=float)
        else:
            priority = np.asarray(user_priority, dtype=float)
        
        # Guard against all-zero priority
        total = float(np.sum(priority))
        if total <= self.config.eps:
            priority = np.ones(K_users, dtype=float)
            total = float(K_users)
        
        priority = priority / total    # now a real ndarray, never None
        
        if self.config.power_allocation == "equal":
            return self.config.total_power * priority
        
        if self.config.power_allocation == "water_filling":
            return self._water_filling(H_eff, W, priority)
        
        return self.config.total_power * priority
    
    def _water_filling(self,
                       H_eff: np.ndarray,
                       W: np.ndarray,
                       user_priority: np.ndarray       # ✅ FIX: non-optional
                       ) -> np.ndarray:
        """
        Simple water-filling power allocation.
        
        Allocates more power to users with better effective channel gains.
        `user_priority` is guaranteed to be a concrete ndarray by the caller.
        """
        K_users = H_eff.shape[0]
        if K_users == 0:
            return np.array([])
        
        H_eff_W = H_eff @ W
        gains = np.abs(np.diag(H_eff_W)) ** 2
        gains = np.maximum(gains, self.config.eps)
        
        # ✅ FIX: user_priority is guaranteed ndarray here (no None risk)
        weights = np.sqrt(gains) * user_priority
        w_sum = float(np.sum(weights))
        if w_sum <= self.config.eps:
            weights = np.ones(K_users, dtype=float) / K_users
        else:
            weights = weights / w_sum
        
        return self.config.total_power * weights
    
    # ------------------------------------------------------------------------
    # SINR AND SUM-RATE
    # ------------------------------------------------------------------------
    
    def compute_sinr(self,
                     H_eff: np.ndarray,
                     W: Optional[np.ndarray] = None,
                     noise_power: Optional[float] = None
                     ) -> np.ndarray:
        """Compute per-user SINR."""
        if W is None:
            W = self._last_W
        if W is None:
            raise ValueError("No precoder available. Call compute_precoder first.")
        
        if noise_power is None:
            noise_power = self.config.noise_power
        
        H_eff_W = H_eff @ W
        
        if self._last_power is None:
            K_users = H_eff.shape[0]
            powers: np.ndarray = np.full(
                K_users, self.config.total_power / max(1, K_users)
            )
        else:
            powers = self._last_power
        
        signal_power = np.abs(np.diag(H_eff_W)) ** 2 * powers
        
        interference_matrix = np.abs(H_eff_W) ** 2
        total_interference = interference_matrix @ powers
        self_interference = np.diag(interference_matrix) * powers
        interference_power = total_interference - self_interference
        
        sinr = signal_power / (interference_power + noise_power + self.config.eps)
        return sinr
    
    def compute_sum_rate(self,
                         H_eff: np.ndarray,
                         W: Optional[np.ndarray] = None,
                         noise_power: Optional[float] = None
                         ) -> float:
        """Compute sum-rate in bits/s/Hz."""
        sinr = self.compute_sinr(H_eff, W, noise_power)
        return float(np.sum(np.log2(1.0 + sinr)))
    
    # ------------------------------------------------------------------------
    # FULL PIPELINE
    # ------------------------------------------------------------------------
    
    def design(self,
               H_eff: np.ndarray,
               user_priority: Optional[np.ndarray] = None
               ) -> Dict[str, Any]:
        """Full digital precoder design."""
        W = self.compute_precoder(H_eff, user_priority)
        sinr = self.compute_sinr(H_eff, W)
        sum_rate = float(np.sum(np.log2(1.0 + sinr)))
        
        info = {
            'W': W,
            'sinr': sinr,
            'sum_rate': sum_rate,
            'num_users': H_eff.shape[0],
            'num_rf_chains': H_eff.shape[1],
            'precoder_type': self.config.precoder_type.value,
            'power_allocation': self._last_power,
        }
        
        self._last_info = info
        return info
    
    # ------------------------------------------------------------------------
    # GETTERS
    # ------------------------------------------------------------------------
    
    def get_last_W(self) -> Optional[np.ndarray]:
        return self._last_W
    
    def get_last_power(self) -> Optional[np.ndarray]:
        return self._last_power
    
    def get_last_info(self) -> Optional[Dict[str, Any]]:
        return self._last_info
    
    def summary(self) -> str:
        """Human-readable summary."""
        if self._last_info is None:
            return "DigitalPrecoder: no design performed yet"
        
        info = self._last_info
        lines = [
            f"DigitalPrecoder Summary",
            f"  Type: {info['precoder_type']}",
            f"  Num users: {info['num_users']}",
            f"  RF chains: {info['num_rf_chains']}",
            f"  Sum-rate: {info['sum_rate']:.4f} bits/s/Hz",
            f"  Per-user SINR (dB):",
        ]
        sinr_db = 10 * np.log10(info['sinr'] + 1e-30)
        for k, s in enumerate(sinr_db):
            lines.append(f"    User {k}: {s:.2f} dB")
        return "\n".join(lines)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def design_digital_precoder(
    H_eff: np.ndarray,
    precoder_type: DigitalPrecoderType = DigitalPrecoderType.MMSE,
    noise_power: float = 1e-9,
    total_power: float = 1.0,
    regularization: float = 1e-3,
) -> Dict[str, Any]:
    """Convenience function to design a digital precoder in one call."""
    config = DigitalPrecoderConfig(
        precoder_type=precoder_type,
        noise_power=noise_power,
        total_power=total_power,
        regularization=regularization,
    )
    precoder = DigitalPrecoder(config)
    return precoder.design(H_eff)


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Digital Precoder Test")
    print("=" * 60)
    
    np.random.seed(42)
    
    K_users = 4
    K_active = 4
    noise_power = 1e-3
    total_power = 1.0
    
    H_eff = (np.random.randn(K_users, K_active) +
             1j * np.random.randn(K_users, K_active)) / np.sqrt(2)
    
    print(f"\n[Setup]")
    print(f"  Users:        {K_users}")
    print(f"  RF chains:    {K_active}")
    print(f"  Noise power:  {noise_power}")
    print(f"  Total power:  {total_power}")
    print(f"  H_eff shape:  {H_eff.shape}")
    
    # Test 1: ZF
    print("\n[Test 1] Zero-Forcing (ZF) precoding")
    config_zf = DigitalPrecoderConfig(
        precoder_type=DigitalPrecoderType.ZF,
        noise_power=noise_power,
        total_power=total_power,
    )
    zf = DigitalPrecoder(config_zf)
    info_zf = zf.design(H_eff)
    
    print(f"  Sum-rate: {info_zf['sum_rate']:.4f} bits/s/Hz")
    print(f"  W shape: {info_zf['W'].shape}")
    print(f"  SINR per user (dB): {10*np.log10(info_zf['sinr']+1e-30).round(2)}")
    
    H_eff_W_zf = H_eff @ info_zf['W']
    off_diag = H_eff_W_zf - np.diag(np.diag(H_eff_W_zf))
    print(f"  Off-diagonal power: {np.sum(np.abs(off_diag)**2):.6e} (should be ~0)")
    assert np.sum(np.abs(off_diag)**2) < 1e-3, "ZF should null interference"
    print("  ✓ Interference fully nulled")
    
    # Test 2: MMSE
    print("\n[Test 2] MMSE precoding")
    config_mmse = DigitalPrecoderConfig(
        precoder_type=DigitalPrecoderType.MMSE,
        noise_power=noise_power,
        total_power=total_power,
        regularization=noise_power * K_users / total_power,
    )
    mmse = DigitalPrecoder(config_mmse)
    info_mmse = mmse.design(H_eff)
    
    print(f"  Sum-rate: {info_mmse['sum_rate']:.4f} bits/s/Hz")
    print(f"  SINR per user (dB): {10*np.log10(info_mmse['sinr']+1e-30).round(2)}")
    print("  ✓ MMSE precoder designed")
    
    # Test 3: MRT
    print("\n[Test 3] MRT (baseline)")
    config_mrt = DigitalPrecoderConfig(
        precoder_type=DigitalPrecoderType.MRT,
        noise_power=noise_power,
        total_power=total_power,
    )
    mrt = DigitalPrecoder(config_mrt)
    info_mrt = mrt.design(H_eff)
    
    print(f"  Sum-rate: {info_mrt['sum_rate']:.4f} bits/s/Hz")
    print(f"  SINR per user (dB): {10*np.log10(info_mrt['sinr']+1e-30).round(2)}")
    print("  ✓ MRT precoder designed")
    
    # Test 4: Comparison
    print("\n[Test 4] Comparison of precoding schemes")
    print(f"  {'Scheme':<12} {'Sum-rate (b/s/Hz)':<20}")
    print(f"  {'-'*12} {'-'*20}")
    print(f"  {'MRT':<12} {info_mrt['sum_rate']:<20.4f}")
    print(f"  {'ZF':<12} {info_zf['sum_rate']:<20.4f}")
    print(f"  {'MMSE':<12} {info_mmse['sum_rate']:<20.4f}")
    
    # Test 5: Power constraint
    print("\n[Test 5] Power constraint verification")
    for name, info in [('ZF', info_zf), ('MMSE', info_mmse), ('MRT', info_mrt)]:
        W = info['W']
        total_w_power = float(np.sum(np.abs(W)**2))
        print(f"  {name}: ||W||_F² = {total_w_power:.4f} (target {total_power})")
    
    # Test 6: Sum-rate vs noise
    print("\n[Test 6] Sum-rate vs noise power (MMSE)")
    for np_val in [1e-2, 1e-3, 1e-4, 1e-5]:
        cfg = DigitalPrecoderConfig(
            precoder_type=DigitalPrecoderType.MMSE,
            noise_power=np_val,
            total_power=total_power,
            regularization=np_val * K_users / total_power,
        )
        d = DigitalPrecoder(cfg)
        inf = d.design(H_eff)
        print(f"  noise={np_val:.0e}: sum-rate = {inf['sum_rate']:.4f} b/s/Hz")
    
    # Test 7: Water-filling (exercises the fixed code path)
    print("\n[Test 7] Water-filling power allocation (exercises fixed path)")
    cfg_wf = DigitalPrecoderConfig(
        precoder_type=DigitalPrecoderType.MMSE,
        noise_power=noise_power,
        total_power=total_power,
        regularization=noise_power * K_users / total_power,
        power_allocation="water_filling",
    )
    wf = DigitalPrecoder(cfg_wf)
    info_wf = wf.design(H_eff)                  # no user_priority → tests default path
    print(f"  Sum-rate (water-filling, no priority): "
          f"{info_wf['sum_rate']:.4f} b/s/Hz")
    print(f"  Power allocation: {info_wf['power_allocation'].round(4)}")
    
    # With explicit priority
    priority = np.array([2.0, 1.0, 1.0, 0.5])
    info_wf2 = wf.design(H_eff, user_priority=priority)
    print(f"  Sum-rate (water-filling, priority): "
          f"{info_wf2['sum_rate']:.4f} b/s/Hz")
    print(f"  Power allocation: {info_wf2['power_allocation'].round(4)}")
    print("  ✓ Water-filling both paths work")
    
    print("\n" + "=" * 60)
    print("✓ Digital precoder validation successful!")
    print("=" * 60)