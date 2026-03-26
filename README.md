# 🎯 MADDPG Adversarial Robustness Framework

**Complete self-contained MADDPG adversarial robustness evaluation with Docker deployment**

---

## 🚀 Quick Start

### **Method 1: Docker (Recommended - Zero Setup Issues)**

```bash
git clone https://github.com/YOUR_USERNAME/maddpg-adversarial-experiments.git
cd maddpg-adversarial-experiments
./docker_setup.sh         # Build environment (one-time, ~5 min)
docker run --gpus all -d --name maddpg-quick maddpg-adversarial:latest ./test_quick.sh            # Quick test (2 min)
docker run --gpus all -d --name maddpg-full maddpg-adversarial:latest ./run_full_experiment.sh   # Complete experiments (6-12 hours)
```

### **Method 2: Direct Installation (Advanced Users)**

```bash
git clone https://github.com/YOUR_USERNAME/maddpg-adversarial-experiments.git
cd maddpg-adversarial-experiments
./setup_no_conda.sh       # Python venv setup
source activate_env.sh    
python standalone_experiment_runner.py --quick --gpu 0
```

---

## 🐳 Docker Scripts

| Script | Purpose | Time | Usage |
|--------|---------|------|--------|
| `docker_setup.sh` | Build Docker environment | 5 min | One-time setup |
| `docker run -d --name maddpg-quick maddpg-adversarial:latest ./test_quick.sh` | Quick functionality test (detached) | 2 min | Starts quick test in background |
| `docker run -d --name maddpg-full maddpg-adversarial:latest ./run_full_experiment.sh` | Complete experiments (detached) | 6-12 hrs | Starts full experiment in background |
| `shell.sh` | Interactive development | - | Development/debugging |

---

## 📊 What You Get

### **🧠 6 MADDPG Variants Evaluated**
- Central Critic + Simple Q-Network
- Central Critic + Duelling Q-Network  
- Local Critic + Duelling Q-Network
- All variants + Graph Neural Network enhancement

### **🔥 3 Attack Types**
- **Packet Loss Attack**: Targets bandwidth perception
- **Reward Minimization**: Maximizes action entropy
- **Confusion Attack**: Pushes toward random actions

### **📈 Publication-Quality Results**
- Thesis-ready plots (300 DPI)
- Statistical significance testing
- Comprehensive robustness rankings
- Complete experimental methodology

---

## 🎓 For Students

### **Zero-Setup Workflow**
```bash
# On any GPU server with Docker:
git clone [repo-url]
cd maddpg-adversarial-experiments
./docker_setup.sh && ./run_full_experiment.sh

# Results automatically saved to host_data/results/
# Thesis plots in: host_data/results/thesis_graphs/

# Run in the background (detached):
docker run --gpus all -d --name maddpg-quick maddpg-adversarial:latest ./test_quick.sh
# To see logs:
docker logs -f maddpg-quick
# For full experiment (detached):
docker run --gpus all -d --name maddpg-full maddpg-adversarial:latest ./run_full_experiment.sh
# See logs for full:
docker logs -f maddpg-full
```

### **Complete Documentation**
- **THESIS_GUIDANCE.md**: Academic writing guide (formal equations, methodology)
- **Generated results**: Statistical analysis, comparison tables, discussion points

---

## ⚡ Performance

| Hardware | Quick Test | Single Variant | All 6 Variants |
|----------|-----------|----------------|----------------|
| RTX 3080 | 2 minutes | 1-2 hours | 6-12 hours |
| RTX 2080 Ti | 3 minutes | 2-3 hours | 8-15 hours |
| Tesla V100 | 2 minutes | 45-90 min | 4-8 hours |

---

## 🔧 Framework Features

- ✅ **Self-contained** - No external dependencies
- ✅ **GPU optimized** - Automatic CUDA detection
- ✅ **Reproducible** - Controlled random seeds
- ✅ **Professional** - Industry-standard Docker deployment
- ✅ **Academic ready** - Formal methodology and documentation

---

## 📁 Repository Structure

```
maddpg-adversarial-experiments/
├── 🚀 Main Scripts
│   ├── standalone_experiment_runner.py    # Experiment orchestrator
│   └── experiment_config.json             # Configuration
├── 🐳 Docker Deployment  
│   ├── Dockerfile                         # Complete environment
│   ├── docker_setup.sh                   # Build & setup utilities
│   ├── test_quick.sh                      # Quick validation
│   ├── run_full_experiment.sh            # Complete experiments
│   └── shell.sh                          # Interactive development
├── 🔧 Alternative Setup
│   └── setup_no_conda.sh                 # Python venv (no Docker)
├── 📚 Documentation
│   ├── README.md                         # This file
│   └── THESIS_GUIDANCE.md                # Academic writing guide
└── 📊 Framework Implementation
    ├── src/maddpg_clean/                 # Clean MADDPG implementation  
    ├── src/attack_framework/             # Corrected attack framework
    └── data/ (generated)                 # Results and models
```

---

## 🎯 Key Improvements Over Original

### **✅ What We Fixed**
- **Mathematical errors** in FGSM attack implementation
- **Circular dependencies** in attack objectives  
- **Environment compatibility** issues (conda/pip conflicts)
- **Missing academic documentation** for thesis writing

### **✅ What We Added**
- **Complete Docker deployment** (eliminates setup issues)
- **6 MADDPG architectural variants** with proper evaluation
- **Comprehensive attack framework** with statistical validation
- **Publication-quality visualization** and analysis tools
- **Formal academic guidance** for thesis writing

---

**Ready to deploy? Start with `./docker_setup.sh`!** 🚀