"""
Typed configuration management for the HDRL XL-MIMO project.

Provides:
    - Dataclass schemas for system, RL, and training configs
    - YAML loading with nested merge (default <- experiment overrides)
    - Save/load for reproducibility
    - Validation on construction

Design philosophy:
    - Configs are DATACLASSES -> type hints, autocomplete, validation
    - YAML files contain OVERRIDES only; defaults live in Python
    - Merge order: built-in defaults <- default.yaml <- experiment.yaml

Usage:
    from src.utils.config import ExperimentConfig

    # Load an experiment (merges default.yaml + exp1.yaml)
    cfg = ExperimentConfig.from_yaml(
        default_path="configs/default.yaml",
        experiment_path="configs/experiments/exp1_subarray.yaml",
    )

    # Access typed fields
    print(cfg.system.num_antennas)          # int
    print(cfg.rl.high_level_learning_rate)  # float
    print(cfg.training.num_episodes)        # int

    # Save resolved config for reproducibility
    cfg.save("runs/exp1/config_resolved.yaml")
"""

import os
import yaml
from dataclasses import dataclass, field, asdict, fields
from typing import Optional, Dict, Any, List, Union
from pathlib import Path


# ============================================================================
# SECTION 1: SYSTEM CONFIG (channel, codebook, precoding)
# ============================================================================

@dataclass
class SystemConfig:
    """Physical-layer system configuration."""
    
    # ----- Antenna array -----
    num_antennas: int = 256
    num_subarrays: int = 8
    antenna_spacing: float = 0.5           # in wavelengths
    carrier_frequency: float = 28e9        # Hz
    array_type: str = "uniform_linear"     # only ULA supported for now
    
    # ----- Subarray partitioning -----
    partition_strategy: str = "contiguous"  # contiguous | interleaved | overlapping
    overlap_ratio: float = 0.0              # only used if overlapping
    
    # ----- Channel -----
    path_loss_exponent: float = 2.0
    num_multipath: int = 3
    multipath_spread_deg: float = 10.0
    spatial_correlation: bool = True
    correlation_factor: float = 0.3
    normalize_channels: bool = True
    
    # ----- Users -----
    num_users: int = 4
    user_distance_min: float = 5.0          # meters
    user_distance_max: float = 40.0         # meters
    user_angle_min_deg: float = -60.0
    user_angle_max_deg: float = 60.0
    
    # ----- Codebook -----
    codebook_num_angles: int = 121           # 1 degree step (must be <= angular resolution)
    codebook_num_ranges: int = 10
    codebook_range_min: float = 5.0          # meters
    codebook_range_max: float = 40.0         # meters
    codebook_range_spacing: str = "log"      # log | linear
    codebook_prune_threshold: float = 0.95   # correlation threshold
    
    # ----- Precoding -----
    precoder_type: str = "mmse"              # zf | mmse | mrt | none
    precoder_regularization: float = 1e-3
    precoder_total_power: float = 1.0
    precoder_noise_power: float = 1e-3
    power_allocation: str = "equal"          # equal | water_filling
    
    # ----- Random seed -----
    seed: int = 42
    
    def __post_init__(self):
        """Validate configuration."""
        if self.num_antennas <= 0:
            raise ValueError("num_antennas must be positive")
        if self.num_subarrays <= 0:
            raise ValueError("num_subarrays must be positive")
        if self.num_subarrays > self.num_antennas:
            raise ValueError("num_subarrays cannot exceed num_antennas")
        if self.carrier_frequency <= 0:
            raise ValueError("carrier_frequency must be positive")
        if self.num_users <= 0:
            raise ValueError("num_users must be positive")
        if self.partition_strategy not in ("contiguous", "interleaved", "overlapping"):
            raise ValueError(
                f"partition_strategy must be one of "
                f"contiguous/interleaved/overlapping, got {self.partition_strategy}"
            )
        if self.precoder_type not in ("zf", "mmse", "mrt", "none"):
            raise ValueError(
                f"precoder_type must be one of zf/mmse/mrt/none, "
                f"got {self.precoder_type}"
            )
        if self.power_allocation not in ("equal", "water_filling"):
            raise ValueError(
                f"power_allocation must be equal or water_filling, "
                f"got {self.power_allocation}"
            )
        if self.codebook_range_spacing not in ("log", "linear"):
            raise ValueError(
                f"codebook_range_spacing must be log or linear, "
                f"got {self.codebook_range_spacing}"
            )


# ============================================================================
# SECTION 2: RL CONFIG (agents, action space, reward)
# ============================================================================

@dataclass
class RLConfig:
    """Reinforcement learning configuration."""
    
    # ----- High-level agent (subarray selection) -----
    high_level_learning_rate: float = 1e-3
    high_level_gamma: float = 0.99
    high_level_hidden_dim: int = 128
    high_level_replay_buffer_size: int = 10000
    high_level_batch_size: int = 64
    high_level_target_update_freq: int = 100
    high_level_epsilon_start: float = 1.0
    high_level_epsilon_end: float = 0.05
    high_level_epsilon_decay: int = 5000    # steps
    
    # ----- Low-level agent (beam selection) -----
    low_level_learning_rate: float = 1e-3
    low_level_gamma: float = 0.99
    low_level_hidden_dim: int = 128
    low_level_replay_buffer_size: int = 10000
    low_level_batch_size: int = 64
    low_level_target_update_freq: int = 100
    low_level_epsilon_start: float = 1.0
    low_level_epsilon_end: float = 0.05
    low_level_epsilon_decay: int = 5000
    
    # ----- Shared -----
    device: str = "cuda"                    # cuda | cpu
    
    # ----- Reward weights -----
    reward_w_rate: float = 1.0
    reward_w_overhead: float = 0.1
    reward_w_power: float = 0.05
    reward_w_interference: float = 0.2
    reward_max_sum_rate: float = 50.0
    reward_max_beams: int = 20
    reward_max_subarrays: int = 8
    reward_max_interference: float = 10.0
    reward_clip_min: float = -10.0
    reward_clip_max: float = 10.0
    
    def __post_init__(self):
        if self.high_level_learning_rate <= 0:
            raise ValueError("high_level_learning_rate must be positive")
        if self.low_level_learning_rate <= 0:
            raise ValueError("low_level_learning_rate must be positive")
        if not (0.0 < self.high_level_gamma <= 1.0):
            raise ValueError("high_level_gamma must be in (0, 1]")
        if not (0.0 < self.low_level_gamma <= 1.0):
            raise ValueError("low_level_gamma must be in (0, 1]")
        if self.device not in ("cuda", "cpu"):
            raise ValueError(f"device must be cuda or cpu, got {self.device}")


# ============================================================================
# SECTION 3: TRAINING CONFIG (episodes, logging, checkpoints)
# ============================================================================

@dataclass
class TrainingConfig:
    """Training loop configuration."""
    
    num_episodes: int = 500
    max_steps_per_episode: int = 50
    eval_frequency: int = 25                # episodes between evals
    eval_num_episodes: int = 10
    log_frequency: int = 10                 # episodes between logs
    
    save_frequency: int = 50                # episodes between checkpoints
    checkpoint_dir: str = "runs/checkpoints"
    log_dir: str = "runs/logs"
    
    # Early stopping
    early_stop: bool = False
    early_stop_threshold: float = 0.0       # reward plateau threshold
    early_stop_patience: int = 100          # episodes
    
    def __post_init__(self):
        if self.num_episodes <= 0:
            raise ValueError("num_episodes must be positive")
        if self.max_steps_per_episode <= 0:
            raise ValueError("max_steps_per_episode must be positive")


# ============================================================================
# SECTION 4: FULL EXPERIMENT CONFIG
# ============================================================================

@dataclass
class ExperimentConfig:
    """
    Full experiment configuration.
    
    Combines SystemConfig, RLConfig, TrainingConfig with metadata.
    """
    
    # Sub-configs
    system: SystemConfig = field(default_factory=SystemConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Metadata (for report)
    experiment_name: str = "default"
    experiment_description: str = ""
    author: str = ""
    
    # ========================================================================
    # LOADING / SAVING
    # ========================================================================
    
    @classmethod
    def from_yaml(cls,
                  default_path: Optional[str] = None,
                  experiment_path: Optional[str] = None) -> "ExperimentConfig":
        """
        Load config from YAML files.
        
        Merge order:
            1. Built-in defaults (from dataclass)
            2. default_path (e.g., configs/default.yaml)
            3. experiment_path (e.g., configs/experiments/exp1.yaml)
        
        Path resolution:
            Relative paths are resolved against the project root
            (the directory containing `src/`), not the current working
            directory. This allows notebooks in `notebooks/` to load
            configs from `configs/` without changing directories.
        
        Args:
            default_path: Path to default YAML (optional)
            experiment_path: Path to experiment YAML (optional)
        
        Returns:
            ExperimentConfig with merged settings
        """
        merged: Dict[str, Any] = {}
        
        # Load default.yaml if provided
        if default_path is not None:
            default_path = _resolve_config_path(default_path)
            if not os.path.exists(default_path):
                raise FileNotFoundError(f"Config file not found: {default_path}")
            with open(default_path, "r", encoding="utf-8") as f:
                default_dict = yaml.safe_load(f) or {}
            merged = _deep_merge(merged, default_dict)
        
        # Load experiment.yaml if provided (overrides defaults)
        if experiment_path is not None:
            experiment_path = _resolve_config_path(experiment_path)
            if not os.path.exists(experiment_path):
                raise FileNotFoundError(f"Config file not found: {experiment_path}")
            with open(experiment_path, "r", encoding="utf-8") as f:
                exp_dict = yaml.safe_load(f) or {}
            merged = _deep_merge(merged, exp_dict)
        
        return cls.from_dict(merged)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """
        Build an ExperimentConfig from a nested dict (partial allowed).
        
        Any key not present in the dict uses the dataclass default.
        Nested dicts (system, rl, training) are merged with their defaults.
        """
        data = data or {}
        
        system = _build_subconfig(SystemConfig, data.get("system", {}))
        rl = _build_subconfig(RLConfig, data.get("rl", {}))
        training = _build_subconfig(TrainingConfig, data.get("training", {}))
        
        return cls(
            system=system,
            rl=rl,
            training=training,
            experiment_name=data.get("experiment_name", "default"),
            experiment_description=data.get("experiment_description", ""),
            author=data.get("author", ""),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to nested dict (JSON/YAML-friendly)."""
        return {
            "experiment_name": self.experiment_name,
            "experiment_description": self.experiment_description,
            "author": self.author,
            "system": asdict(self.system),
            "rl": asdict(self.rl),
            "training": asdict(self.training),
        }
    
    def save(self, path: str) -> None:
        """
        Save the resolved config to a YAML file.
        
        Use this to log the EXACT config used for each run — essential
        for reproducibility in reports.
        
        Note: uses encoding="utf-8" so any non-ASCII character in the
        config (e.g., unicode in metadata) is written correctly on
        Windows.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, default_flow_style=False)
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "=" * 60,
            f"ExperimentConfig: {self.experiment_name}",
            "=" * 60,
            f"  Description: {self.experiment_description or '(none)'}",
            f"  Author:      {self.author or '(none)'}",
            "",
            "  System:",
            f"    Antennas:        {self.system.num_antennas}",
            f"    Subarrays:       {self.system.num_subarrays}",
            f"    Users:           {self.system.num_users}",
            f"    Carrier:         {self.system.carrier_frequency/1e9:.1f} GHz",
            f"    Strategy:        {self.system.partition_strategy}",
            f"    Precoder:        {self.system.precoder_type}",
            f"    Codebook beams:  {self.system.codebook_num_angles * self.system.codebook_num_ranges}",
            "",
            "  RL:",
            f"    Device:          {self.rl.device}",
            f"    HL LR:           {self.rl.high_level_learning_rate}",
            f"    LL LR:           {self.rl.low_level_learning_rate}",
            f"    HL batch:        {self.rl.high_level_batch_size}",
            f"    LL batch:        {self.rl.low_level_batch_size}",
            f"    Reward weights:  rate={self.rl.reward_w_rate}, "
            f"overhead={self.rl.reward_w_overhead}, "
            f"power={self.rl.reward_w_power}, "
            f"interference={self.rl.reward_w_interference}",
            "",
            "  Training:",
            f"    Episodes:        {self.training.num_episodes}",
            f"    Max steps/ep:    {self.training.max_steps_per_episode}",
            f"    Eval freq:       {self.training.eval_frequency}",
            f"    Log dir:         {self.training.log_dir}",
            "=" * 60,
        ]
        return "\n".join(lines)


# ============================================================================
# SECTION 5: HELPERS
# ============================================================================

def _deep_merge(base: Dict[str, Any],
                override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge `override` into `base`.
    
    For keys present in both:
        - If both values are dicts -> recurse
        - Otherwise -> override wins
    """
    result = dict(base)
    for key, value in override.items():
        if (key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _build_subconfig(cls, data: Dict[str, Any]):
    """
    Build a subconfig (SystemConfig, RLConfig, TrainingConfig) from a dict,
    keeping defaults for any keys not present in `data`.
    
    Silently ignores unknown keys (with a warning).
    """
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict for subconfig, got {type(data)}")
    
    known_fields = {f.name for f in fields(cls)}
    unknown = set(data.keys()) - known_fields
    if unknown:
        import warnings
        warnings.warn(
            f"Unknown fields in {cls.__name__}: {sorted(unknown)}. "
            f"These will be ignored."
        )
    
    filtered = {k: v for k, v in data.items() if k in known_fields}
    return cls(**filtered)


def _resolve_config_path(path: str) -> str:
    """
    Resolve a config path relative to the project root.
    
    Handles the common notebook scenario where the current working
    directory is `notebooks/` but the config file lives at
    `<project_root>/configs/...`.
    
    Resolution order:
        1. Absolute path as-is.
        2. Path relative to the current working directory.
        3. Path relative to the nearest ancestor directory that
           contains a `src/` folder (the project root).
    
    Returns the first existing path, or the input unchanged if none
    of the candidates exist (the caller will raise FileNotFoundError
    with the original path for a clear error message).
    """
    # 1. Absolute path -> return as-is
    if os.path.isabs(path):
        return path
    
    # 2. Relative to current working directory
    if os.path.exists(path):
        return os.path.abspath(path)
    
    # 3. Walk upward looking for the project root (contains `src/`)
    current = os.path.abspath(os.getcwd())
    for _ in range(10):    # bounded upward walk
        if os.path.isdir(os.path.join(current, "src")):
            candidate = os.path.join(current, path)
            if os.path.exists(candidate):
                return candidate
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    
    # 4. Fallback: return input unchanged (caller raises clear error)
    return path


# ============================================================================
# SECTION 6: QUICK ACCESS HELPERS
# ============================================================================

def load_default_config() -> ExperimentConfig:
    """Load the default config (used when no YAML is present)."""
    return ExperimentConfig()


def get_default_config_path() -> str:
    """Return the canonical path to configs/default.yaml."""
    return os.path.join(os.getcwd(), "configs", "default.yaml")


def get_experiment_config_path(name: str) -> str:
    """Return the canonical path to configs/experiments/<name>.yaml."""
    return os.path.join(os.getcwd(), "configs", "experiments", f"{name}.yaml")


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Config Module Test")
    print("=" * 60)
    
    # ------------------------------------------------------------------
    # Test 1: Default config
    # ------------------------------------------------------------------
    print("\n[Test 1] Default config")
    cfg = ExperimentConfig()
    print(f"  num_antennas: {cfg.system.num_antennas}")
    print(f"  num_users:    {cfg.system.num_users}")
    print(f"  device:       {cfg.rl.device}")
    print(f"  episodes:     {cfg.training.num_episodes}")
    
    # ------------------------------------------------------------------
    # Test 2: from_dict with partial overrides
    # ------------------------------------------------------------------
    print("\n[Test 2] from_dict (partial override)")
    partial = {
        "experiment_name": "test_partial",
        "system": {"num_antennas": 128, "num_users": 8},
        "rl": {"device": "cpu"},
    }
    cfg2 = ExperimentConfig.from_dict(partial)
    print(f"  num_antennas (overridden): {cfg2.system.num_antennas}  (expected 128)")
    print(f"  num_users (overridden):    {cfg2.system.num_users}     (expected 8)")
    print(f"  num_subarrays (default):   {cfg2.system.num_subarrays} (expected 8)")
    print(f"  device (overridden):       {cfg2.rl.device}            (expected cpu)")
    print(f"  num_episodes (default):    {cfg2.training.num_episodes} (expected 500)")
    
    assert cfg2.system.num_antennas == 128
    assert cfg2.system.num_users == 8
    assert cfg2.rl.device == "cpu"
    assert cfg2.training.num_episodes == 500
    print("  OK: partial override works")
    
    # ------------------------------------------------------------------
    # Test 3: Round-trip (save -> load)
    # ------------------------------------------------------------------
    print("\n[Test 3] Save -> load round-trip")
    test_dir = os.path.join(os.getcwd(), "runs", "test_config")
    os.makedirs(test_dir, exist_ok=True)
    test_path = os.path.join(test_dir, "resolved.yaml")
    
    cfg3 = ExperimentConfig.from_dict({
        "experiment_name": "roundtrip_test",
        "system": {"num_antennas": 32, "num_users": 2},
        "rl": {"device": "cpu", "high_level_learning_rate": 5e-4},
    })
    cfg3.save(test_path)
    print(f"  Saved to: {test_path}")
    
    loaded = ExperimentConfig.from_yaml(default_path=test_path)
    print(f"  Loaded experiment_name: {loaded.experiment_name}")
    print(f"  Loaded num_antennas:    {loaded.system.num_antennas}")
    print(f"  Loaded device:          {loaded.rl.device}")
    print(f"  Loaded HL LR:           {loaded.rl.high_level_learning_rate}")
    
    assert loaded.experiment_name == "roundtrip_test"
    assert loaded.system.num_antennas == 32
    assert loaded.rl.device == "cpu"
    assert abs(loaded.rl.high_level_learning_rate - 5e-4) < 1e-12
    print("  OK: round-trip preserves values")
    
    # ------------------------------------------------------------------
    # Test 4: Validation
    # ------------------------------------------------------------------
    print("\n[Test 4] Validation")
    bad_cases = [
        ({"num_antennas": -1}, "num_antennas"),
        ({"num_antennas": 8, "num_subarrays": 100}, "num_subarrays > num_antennas"),
        ({"carrier_frequency": 0}, "carrier_frequency"),
        ({"partition_strategy": "unknown"}, "partition_strategy"),
        ({"precoder_type": "unknown"}, "precoder_type"),
    ]
    for overrides, label in bad_cases:
        try:
            SystemConfig(**overrides)
            print(f"  FAIL {label}: should have raised")
            raise AssertionError(f"{label} did not raise")
        except ValueError:
            print(f"  OK {label}: raised ValueError as expected")
    
    # ------------------------------------------------------------------
    # Test 5: Summary string
    # ------------------------------------------------------------------
    print("\n[Test 5] Summary string")
    cfg4 = ExperimentConfig.from_dict({
        "experiment_name": "demo",
        "experiment_description": "Demo experiment for summary test",
        "author": "Student",
    })
    print(cfg4.summary())
    
    # ------------------------------------------------------------------
    # Test 6: Unknown key warning
    # ------------------------------------------------------------------
    print("\n[Test 6] Unknown key warning")
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = ExperimentConfig.from_dict({
            "system": {"unknown_field": 42, "num_antennas": 32},
        })
        assert len(w) >= 1
        print(f"  OK warning raised: {str(w[0].message)[:80]}...")
    
    # ------------------------------------------------------------------
    # Test 7: Path resolution (NEW)
    # ------------------------------------------------------------------
    print("\n[Test 7] Path resolution from non-root CWD")
    original_cwd = os.getcwd()
    try:
        # Change to notebooks/ and try to load configs/default.yaml
        ntbk_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "notebooks")
        if os.path.isdir(ntbk_dir):
            os.chdir(ntbk_dir)
            try:
                cfg5 = ExperimentConfig.from_yaml(default_path="configs/default.yaml")
                print(f"  OK loaded from notebooks/ CWD: {cfg5.experiment_name}")
            except FileNotFoundError as e:
                print(f"  FAIL path resolution: {e}")
                raise
        else:
            print(f"  (skipped: {ntbk_dir} not found)")
    finally:
        os.chdir(original_cwd)
    
    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    import shutil
    shutil.rmtree(test_dir, ignore_errors=True)
    
    print("\n" + "=" * 60)
    print("Config module validation successful!")
    print("=" * 60)