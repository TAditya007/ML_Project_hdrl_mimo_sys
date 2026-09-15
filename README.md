# Hierarchical Deep Reinforcement Learning for Subarray and Beam Selection in Near-Field XL-MIMO Systems

A simulation-driven **Hierarchical Deep Reinforcement Learning (HDRL)** framework for beam management in near-field XL-MIMO downlink systems. The system learns to (1) select which antenna subarrays to activate and (2) select near-field polar-domain beams per active subarray, using hybrid analog-digital precoding.

---

## Notice

**Running any module:** Always use `python -m src.<package>.<module>` from the project root to ensure `src` is importable. Direct file execution (`python src/...`) will fail due to `sys.path` semantics.

**Running notebooks:** Always select the **`hdrl_mimo`** kernel in VS Code. The notebook kernels start in the `notebooks/` directory, so path-agnostic loading is handled via `_find_project_root()` inside each notebook.

---

## Overview

The project implements the full pipeline described in the abstract:

- **Near-field channel model** with spherical wavefront effects
- **Polar-domain codebook** with joint angle + range beam focusing
- **Subarray partitioning** (contiguous / interleaved / overlapping)
- **Hybrid analog-digital precoding** (ZF / MMSE / MRT)
- **Custom Gymnasium environment** for multiuser near-field beam management
- **Two cooperative DQN agents** (hierarchical RL)
- **Multi-objective reward** balancing sum-rate, beam training overhead, power, and interference

---

## Problem Setup

| Parameter | Value |
| ----------- | ------- |
| Total antennas (BS) | 256 |
| Subarrays | 8 (32 antennas each) |
| Carrier frequency | 28 GHz (mmWave) |
| Wavelength | 10.71 mm |
| Subarray aperture | 0.3375 m |
| Subarray Fraunhofer distance | ~21.3 m |
| Number of users | 4 (or 3 for exp1) |
| User distribution | Near-field region (2-15 m) |
| Codebook (dense -> pruned) | 1210 -> ~90 beams |
| Precoder | MMSE (hybrid) |

The array uses a Y-axis ULA (broadside = +x direction). Users are located in the near-field region `r < 2*D^2/lambda`, where `D` is the subarray aperture.

---

## Directory Structure

hdrl_mimo_sys/
├── src/
│ ├── channel/ # Near-field channel modeling
│ │ ├── near_field.py
│ │ └── channel_generator.py
│ ├── codebook/ # Subarray + polar codebook
│ │ ├── subarray.py
│ │ └── polar_codebook.py
│ ├── precoding/ # Hybrid precoding
│ │ ├── analog.py
│ │ └── digital.py
│ ├── environment/ # Gymnasium env + reward
│ │ ├── reward.py
│ │ └── xl_mimo_env.py
│ ├── agents/ # RL agents
│ │ ├── high_level.py
│ │ └── low_level.py
│ ├── training/ # Training loop
│ │ └── loop.py
│ └── utils/ # Config + metrics
│ ├── config.py
│ └── metrics.py
├── tests/ # Pytest test suites
│ ├── test_channel.py
│ ├── test_codebook.py
│ └── test_environment.py
├── configs/ # YAML configurations
│ ├── default.yaml
│ └── experiments/
│ ├── exp1_subarray.yaml
│ └── exp2_beam_selection.yaml
├── notebooks/ # Analysis notebooks
│ ├── 01_channel_visualization.ipynb
│ ├── 02_codebook_generation.ipynb
│ ├── 03_env_testing.ipynb
│ ├── 04_training_analysis.ipynb
│ └── figures/ # Generated PNGs
├── runs/ # Training outputs
│ ├── default/
│ ├── exp1_subarray/
│ ├── exp2_beam_selection/
│ └── checkpoints/
├── PROJECT_TIMELINE.txt # Full development log
└── README.md

text

---

## Installation

### Prerequisites

- Windows / Linux / macOS
- Anaconda or Miniconda
- Python 3.11
- (Optional) NVIDIA GPU with CUDA for faster training

### Setup

```powershell
# Create and activate conda environment
conda create -n hdrl_mimo python=3.11
conda activate hdrl_mimo

# Install dependencies
pip install numpy scipy matplotlib torch gymnasium pyyaml tensorboard pandas tabulate pytest pytest-cov
Verify
powershell
cd "path/to/hdrl_mimo_sys"
python -m pytest tests/ -q
Expected: 133 tests passing in ~2 seconds.

Quick Start
1. Verify core modules
powershell
# Channel model
python -m src.channel.near_field

# Codebook
python -m src.codebook.polar_codebook

# Environment
python -m src.environment.xl_mimo_env

# RL agents
python -m src.agents.high_level
python -m src.agents.low_level
2. Run tests
powershell
python -m pytest tests/ -v
3. Train the HDRL system
powershell
# Default config (500 episodes, ~4 min)
python -m src.training.loop --config configs/default.yaml

# Exp1: sparse subarray selection (300 episodes)
python -m src.training.loop --config configs/default.yaml --experiment configs/experiments/exp1_subarray.yaml

# Exp2: beam discrimination (400 episodes)
python -m src.training.loop --config configs/default.yaml --experiment configs/experiments/exp2_beam_selection.yaml

# Quick smoke test (10 episodes)
python -m src.training.loop --config configs/default.yaml --episodes 10
4. Analyze results
Open notebooks/04_training_analysis.ipynb in VS Code, select the hdrl_mimo kernel, run all cells.

Produces 8 comparison figures + runs/comparison_table.csv.

Key Results
Training Outcomes (last 10% of episodes)
Experiment Episodes Reward (init -> final) Sum-rate (b/s/Hz) Active Subarrays
default 500 +0.327 -> +0.401 (+22.5%) 18.78 -> 23.45 6.07
exp1_subarray 300 +0.169 -> +0.166 (-2.3%) 18.32 -> 19.36 4.66 (sparse)
exp2_beam_selection 400 +0.338 -> +0.416 (+23.2%) 18.78 -> 23.79 6.83
Interpretation
Default: Both agents converge - reward and rate improve steadily.

Exp1 (sparse subarrays): Stronger power penalty (w_power=0.30) forces the agent to use only ~4.7 subarrays on average, demonstrating learned sparsity.

Exp2 (tight user cluster): Users clustered in +/-10 deg sector - the agent must select discriminating beams, achieving the highest sum-rate (23.79 b/s/Hz).

Reproducing the Results
Full pipeline (~10 min)
powershell
# 1. Verify environment
python -m pytest tests/ -q

# 2. Train all three experiments (clean runs/ first)
Remove-Item -Recurse -Force runs -ErrorAction SilentlyContinue
python -m src.training.loop --config configs/default.yaml
python -m src.training.loop --config configs/default.yaml --experiment configs/experiments/exp1_subarray.yaml
python -m src.training.loop --config configs/default.yaml --experiment configs/experiments/exp2_beam_selection.yaml

# 3. Analyze results
# (open notebooks/04_training_analysis.ipynb, run all cells)
Regenerate all figures
powershell
# Phase 1 figures
# (open notebooks/01_channel_visualization.ipynb, run all cells)
# (open notebooks/02_codebook_generation.ipynb, run all cells)

# Phase 2 environment figures
# (open notebooks/03_env_testing.ipynb, run all cells)

# Phase 2 training comparison figures
# (open notebooks/04_training_analysis.ipynb, run all cells)
Configuration
Configs live in configs/. The loading order is:

text
Built-in defaults -> configs/default.yaml -> configs/experiments/<exp>.yaml
Later entries override earlier ones. See configs/default.yaml for full documentation of every field.

Key config sections
Section Purpose
system Physical layer: antennas, codebook, precoding, users
rl Agents: learning rates, batch sizes, reward weights
training Episodes, logging, checkpointing
experiment_name / experiment_description / author Metadata
CLI overrides
powershell
python -m src.training.loop `
    --config configs/default.yaml `
    --experiment configs/experiments/exp1_subarray.yaml `
    --episodes 50 `
    --tensorboard
Module Overview
Channel (src/channel/)
near_field.py - spherical wavefront propagation, Fraunhofer distance, Y-axis ULA

channel_generator.py - multiuser channels with multipath and spatial correlation

Codebook (src/codebook/)
subarray.py - 3 partitioning strategies (contiguous/interleaved/overlapping)

polar_codebook.py - near-field beam grid over (theta, r) with correlation-based pruning

Precoding (src/precoding/)
analog.py - block-diagonal F_RF with unit-modulus constraint

digital.py - ZF / MMSE / MRT / water-filling

Environment (src/environment/)
reward.py - 4-objective weighted reward (rate, overhead, power, interference)

xl_mimo_env.py - Gymnasium environment with hierarchical Dict action space

Agents (src/agents/)
high_level.py - Bernoulli-action DQN for subarray selection

low_level.py - K-head DQN for per-subarray beam selection

Both use Double DQN with experience replay and target networks

Training (src/training/)
loop.py - orchestrates HL + LL training, logs metrics, saves checkpoints

Utils (src/utils/)
config.py - typed YAML loader with validation

metrics.py - CSV/TensorBoard/plot generation

Testing
powershell
# All tests
python -m pytest tests/ -v

# Specific suite
python -m pytest tests/test_channel.py -v
python -m pytest tests/test_codebook.py -v
python -m pytest tests/test_environment.py -v

# With coverage
python -m pytest tests/ --cov=src --cov-report=term-missing
Expected output: 133 passed in ~2s.

Outputs
Per training run (runs/<experiment>/)
text
metrics.csv          - per-episode table (reward, rate, losses, epsilon)
summary.json         - initial vs final window comparison
plots/               - 5 training-curve PNGs
Checkpoints (runs/checkpoints/)
text
<exp>_ep<NNNN>.pt.hl - high-level agent
<exp>_ep<NNNN>.pt.ll - low-level agent
Both agents save/load with architecture validation (obs_dim, num_subarrays, hidden_dim).

Cross-experiment (runs/)
text
comparison_table.csv - metrics across all experiments
comparison_table.md  - same, in markdown
Known Limitations & Future Work
Backlog items
# Item File Priority
1 MRT per-column normalization src/precoding/digital.py Medium
2 np.corrcoef warning suppression src/channel/channel_generator.py Low
3 Near-field ratio diagnostic refinement src/channel/channel_generator.py Low
4 HL summed-target loss scale issue src/agents/high_level.py Medium
Future extensions
Dataset validation: Cross-validate the near-field channel model against public ray-traced datasets (e.g., DeepMIMO-NF).

Parameter sweep: Vary antennas, subarrays, codebook size, user count.

Alternative RL algorithms: Compare DQN against PPO / SAC.

Beam tracking: Extend to time-varying channels.

Hardware-aware constraints: Add realistic phase-shifter quantization.

References
Near-field XL-MIMO channel modeling and polar-domain codebooks (recent IEEE literature)

Hybrid precoding for mmWave massive MIMO (Heath et al., 2016)

Double DQN (van Hasselt et al., 2016)

Gymnasium API: https://gymnasium.farama.org

Development Log
See PROJECT_TIMELINE.txt for the full chronological record of every file, milestone, error resolved, and metric.

Author
Student - academic project, 2026.

License
Academic use only.

text

---

**Save this as `README.md` in your project root** (`hdrl_mimo_sys/README.md`).

That completes the project. 🎉
