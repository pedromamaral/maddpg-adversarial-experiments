# MADDPG Adversarial Robustness Framework

Self-contained MADDPG adversarial robustness evaluation framework, designed to run on a remote GPU server via Docker in detached mode.

---

## Prerequisites

- Remote host with Docker + NVIDIA Container Toolkit
- Git

---

## Workflow

### 1. Clone & build

```bash
git clone https://github.com/pedromamaral/maddpg-adversarial-experiments.git
cd maddpg-adversarial-experiments
docker build -t maddpg-adversarial:latest .
```

The build takes ~5 minutes and installs all dependencies inside the image.

### 2. Validate (optional but recommended)

```bash
./test_quick.sh
```

Launches a quick smoke-test in the background (~2 min). Check its output with:

```bash
./check_progress_quick.sh
```

Press **Ctrl-C** at any time to stop following the logs — the container keeps running.

### 3. Run the full experiment

```bash
./run_full_experiment.sh
```

Launches all 6 MADDPG variants × 3 attack types in detached mode. Monitor progress with:

```bash
./check_progress_full.sh
```

### 4. Collect results

Results are written to the host machine in real time via volume mounts:

```
host_data/results/          # JSON stats, CSV tables
host_data/results/thesis_graphs/   # 300 DPI plots
host_data/models/           # Trained model checkpoints
```

Download from a remote server with:

```bash
scp -r user@your-server:~/maddpg-adversarial-experiments/host_data/results ./results
```

---

## Scripts

| Script | What it does |
|--------|--------------|
| `test_quick.sh` | Launch quick smoke-test in detached mode |
| `run_full_experiment.sh` | Launch full experiment in detached mode |
| `check_progress_quick.sh` | Follow logs of the quick-test container |
| `check_progress_full.sh` | Follow logs of the full-experiment container |

---

## What the experiment evaluates

**6 MADDPG architectural variants:**
- Central Critic + Simple Q-Network
- Central Critic + Duelling Q-Network
- Local Critic + Duelling Q-Network
- All three variants repeated with Graph Neural Network state pre-processing

**3 FGSM attack types** at 5 epsilon values (0.01 → 0.20):
- Packet Loss Attack — targets bandwidth perception
- Reward Minimization — maximises action entropy
- Confusion Attack — pushes towards random actions

**Outputs:** reward degradation, packet-loss increase, attack success rate, robustness score, and publication-quality plots (300 DPI).

---

## Performance reference

| Hardware | Quick test | Full experiment |
|----------|------------|-----------------|
| RTX 3080 | ~2 min | 6–12 h |
| RTX 2080 Ti | ~3 min | 8–15 h |
| Tesla V100 | ~2 min | 4–8 h |

---

## Repository structure

```
maddpg-adversarial-experiments/
├── test_quick.sh                      # Launch quick test (detached)
├── run_full_experiment.sh             # Launch full experiment (detached)
├── check_progress_quick.sh            # Monitor quick test logs
├── check_progress_full.sh             # Monitor full experiment logs
├── Dockerfile                         # Complete runtime environment
├── standalone_experiment_runner.py    # Experiment orchestrator
├── experiment_config.json             # Hyperparameters & topology config
├── pyproject.toml                     # Installable package definition
├── requirements.txt                   # Pinned dependencies
├── README.md
├── THESIS_GUIDANCE.md                 # Academic writing guide
└── src/
    ├── maddpg_clean/
    │   ├── maddpg_implementation.py   # MADDPG + agents + replay buffer
    │   └── network_environment.py     # Network simulation environment
    └── attack_framework/
        └── improved_fgsm_attack.py    # FGSM attacks + evaluation + plots
```

---

## Documentation

- **THESIS_GUIDANCE.md** — formal equations, methodology narrative, and discussion points for academic writing.
- Code is fully documented with docstrings. Enable debug logging at runtime:
  ```bash
  PYTHONPATH=src python -c "
  import logging; logging.basicConfig(level=logging.DEBUG)
  from maddpg_clean.maddpg_implementation import MADDPG
  "
  ```
