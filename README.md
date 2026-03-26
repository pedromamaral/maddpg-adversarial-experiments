# MADDPG Adversarial Robustness Framework

**Complete self-contained MADDPG adversarial robustness evaluation with Docker deployment**

---

## Quick Start

### Method 1: Docker (Recommended – Zero Setup Issues)

```bash
git clone https://github.com/pedromamaral/maddpg-adversarial-experiments.git
cd maddpg-adversarial-experiments
./docker_setup.sh         # Build environment (one-time, ~5 min)
docker run --gpus all -d --name maddpg-quick maddpg-adversarial:latest ./test_quick.sh            # Quick test (2 min)
docker run --gpus all -d --name maddpg-full  maddpg-adversarial:latest ./run_full_experiment.sh   # Complete experiments (6-12 hours)
```

### Method 2: Editable Install (Advanced Users)

```bash
git clone https://github.com/pedromamaral/maddpg-adversarial-experiments.git
cd maddpg-adversarial-experiments
./setup_no_conda.sh          # Python venv setup
source activate_env.sh
pip install -e .             # install src/ as importable package
python standalone_experiment_runner.py --quick --gpu 0
```

---

## Docker Scripts

| Script | Purpose | Time | Usage |
|--------|---------|------|-------|
| `docker_setup.sh` | Build Docker environment | 5 min | One-time setup |
| `docker run -d --name maddpg-quick maddpg-adversarial:latest ./test_quick.sh` | Quick functionality test | 2 min | Background validation |
| `docker run -d --name maddpg-full maddpg-adversarial:latest ./run_full_experiment.sh` | Complete experiments | 6-12 hrs | Full experiment run |
| `shell.sh` | Interactive development | – | Development/debugging |

---

## What You Get

### 6 MADDPG Variants Evaluated
- Central Critic + Simple Q-Network
- Central Critic + Duelling Q-Network
- Local Critic + Duelling Q-Network
- All three variants + Graph Neural Network enhancement

### 3 Attack Types
- **Packet Loss Attack**: Targets bandwidth perception
- **Reward Minimization**: Maximises action entropy
- **Confusion Attack**: Pushes toward random actions

### Publication-Quality Results
- Thesis-ready plots (300 DPI)
- Statistical significance testing
- Comprehensive robustness rankings
- Complete experimental methodology

---

## For Students

### Zero-Setup Workflow
```bash
# On any GPU server with Docker:
git clone https://github.com/pedromamaral/maddpg-adversarial-experiments.git
cd maddpg-adversarial-experiments
./docker_setup.sh && ./run_full_experiment.sh

# Results automatically saved to host_data/results/
# Thesis plots in: host_data/results/thesis_graphs/

# Check logs:
docker logs -f maddpg-full

# Download results from a remote server:
scp -r user@your-server:~/maddpg-adversarial-experiments/host_data/results ./results_local_copy
```

### Complete Documentation
- **THESIS_GUIDANCE.md**: Academic writing guide (formal equations, methodology)
- **Generated results**: Statistical analysis, comparison tables, discussion points

---

## Performance Reference

| Hardware | Quick Test | Single Variant | All 6 Variants |
|----------|-----------|----------------|----------------|
| RTX 3080   | 2 minutes | 1-2 hours   | 6-12 hours  |
| RTX 2080 Ti | 3 minutes | 2-3 hours  | 8-15 hours  |
| Tesla V100 | 2 minutes | 45-90 min   | 4-8 hours   |

---

## Framework Features

- **Self-contained** – No external dependencies beyond `requirements.txt`
- **GPU optimised** – Automatic CUDA detection
- **Reproducible** – Centralised random seed control (`set_global_seeds()`)
- **Professional** – Industry-standard Docker deployment
- **Installable** – `pip install -e .` via `pyproject.toml`
- **Academic ready** – Formal methodology and documentation

---

## Repository Structure

```
maddpg-adversarial-experiments/
├── standalone_experiment_runner.py    # Experiment orchestrator
├── experiment_config.json             # Configuration
├── pyproject.toml                     # Installable package definition
├── requirements.txt                   # Pinned dependencies
├── Dockerfile                         # Complete environment
├── docker_setup.sh                    # Build & setup utilities
├── test_quick.sh                      # Quick validation
├── run_full_experiment.sh             # Complete experiments
├── shell.sh                           # Interactive development
├── setup_no_conda.sh                  # Python venv (no Docker)
├── README.md                          # This file
├── THESIS_GUIDANCE.md                 # Academic writing guide
└── src/
    ├── maddpg_clean/                  # Clean MADDPG implementation
    │   ├── maddpg_implementation.py
    │   └── network_environment.py
    └── attack_framework/              # FGSM adversarial attack framework
        └── improved_fgsm_attack.py
```

---

## Key Improvements in This Version

- **Fixed** Duelling critic forward pass (incorrect Q-value collapse via `torch.max`)
- **Fixed** `torch.load()` security warning (`weights_only=True`)
- **Fixed** Soft target-network update (idiomatic `lerp_()` instead of dict manipulation)
- **Fixed** Heterogeneous observation spaces in `ReplayBuffer`
- **Fixed** Hardcoded absolute path in `ThesisVisualizationSuite`
- **Added** `set_global_seeds()` for full reproducibility
- **Added** `pyproject.toml` for clean `pip install -e .` workflow
- **Added** `logging` module throughout (replaces raw `print` profiling)
- **Added** Pinned `requirements.txt` versions
- **Moved** optimisers from network classes to `Agent` for clean separation

---

**Ready to deploy? Start with `./docker_setup.sh`!**
