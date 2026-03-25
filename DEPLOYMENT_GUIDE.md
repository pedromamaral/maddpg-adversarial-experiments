# MADDPG Adversarial Robustness Experiments - Deployment Guide

## 🚀 Deployment Options

### Option 1: GitHub Repository + VSCode Remote SSH (Recommended)

#### **Setup Steps:**

1. **Create GitHub Repository**
```bash
# On your local machine
git init maddpg-adversarial-experiments
cd maddpg-adversarial-experiments
git remote add origin https://github.com/YOUR_USERNAME/maddpg-adversarial-experiments.git
```

2. **VSCode Remote SSH Setup**
- Install "Remote - SSH" extension in VSCode
- Configure SSH connection to your GPU server
- Open remote folder directly in VSCode

3. **GPU Server Environment Setup**
```bash
# On GPU server
conda create -n maddpg-attacks python=3.9
conda activate maddpg-attacks
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib seaborn numpy pandas networkx torch-geometric
```

### Option 2: Jupyter Lab on GPU Server
```bash
# On GPU server  
pip install jupyterlab
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
# Then access via SSH tunnel: ssh -L 8888:localhost:8888 user@gpu-server
```

### Option 3: Docker Container
```bash
# Use official PyTorch GPU container
docker run -it --gpus all pytorch/pytorch:latest
```

## 📁 **Repository Structure**

```
maddpg-adversarial-experiments/
├── README.md
├── requirements.txt
├── environment.yml
├── .gitignore
├── config/
│   ├── experiment_configs.py
│   └── topology_configs.py
├── src/
│   ├── __init__.py
│   ├── attack_framework/
│   │   ├── __init__.py
│   │   ├── fgsm_attack.py           # Improved attack framework
│   │   └── evaluation.py           # Robustness evaluation
│   ├── maddpg_original/
│   │   ├── MADDPG.py               # Student's original code  
│   │   ├── Agent.py
│   │   ├── NetworkEngine.py
│   │   ├── NetworkEnv.py
│   │   └── environmental_variables.py
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── experiment_runner.py    # Main experiment orchestrator
│   │   └── data_collector.py       # Results collection
│   └── visualization/
│       ├── __init__.py
│       ├── thesis_plots.py         # Thesis graph generation
│       └── analysis_reports.py     # Statistical analysis
├── experiments/
│   ├── baseline_training.py        # Train clean models
│   ├── attack_evaluation.py        # Evaluate robustness
│   └── comparative_analysis.py     # Generate comparisons
├── data/
│   ├── topologies/                 # Network topology files
│   ├── models/                     # Saved model weights
│   └── results/                    # Experimental results
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   ├── attack_visualization.ipynb
│   └── thesis_plots.ipynb
└── scripts/
    ├── setup_environment.sh
    ├── run_full_experiment.sh
    └── generate_thesis_plots.sh
```

## 🔧 **Integration Requirements**

### **Code Integration Checklist:**

- [ ] Clone student's original MADDPG repository
- [ ] Integrate improved attack framework  
- [ ] Replace mock classes with real implementations
- [ ] Configure GPU training settings
- [ ] Set up experiment tracking (wandb/tensorboard)
- [ ] Create automated result collection
- [ ] Implement thesis graph generation

### **Hardware Requirements:**

- **GPU**: CUDA-capable (Tesla V100, RTX 3080+, or A100)
- **RAM**: 16GB+ (for large network topologies)  
- **Storage**: 50GB+ (for models, data, results)
- **Network**: Fast internet for remote access

## 🎯 **Experiment Pipeline**

### **Phase 1: Baseline Training** (2-4 hours on GPU)
```bash
python experiments/baseline_training.py \
  --topology service_provider \
  --variants CC-Simple,CC-Duelling,LC-Duelling \
  --epochs 200 \
  --episodes 100 \
  --gpu 0
```

### **Phase 2: Attack Evaluation** (4-6 hours on GPU)
```bash
python experiments/attack_evaluation.py \
  --models data/models/baseline_* \
  --attack_types packet_loss,reward_minimize,confusion \
  --epsilon_values 0.01,0.05,0.1,0.15,0.2 \
  --episodes 100 \
  --gpu 0
```

### **Phase 3: Analysis & Visualization** (30 minutes on CPU)
```bash
python experiments/comparative_analysis.py \
  --results data/results/attack_evaluation_* \
  --output thesis_graphs/
```

## 📊 **Data Collection Strategy**

### **Metrics to Track:**
- Episode rewards (clean vs attacked)
- Packet loss rates  
- Link utilization distributions
- Action selection patterns
- Training convergence curves
- Attack success rates
- Performance degradation curves

### **Storage Format:**
```python
{
    "experiment_id": "exp_001_cc_simple_clean",
    "variant": "CC-Simple", 
    "attack_type": null,
    "epsilon": null,
    "topology": "service_provider_65_nodes",
    "results": {
        "episode_rewards": [1420.3, 1398.7, ...],
        "packet_losses": [0.8, 1.2, ...],
        "utilization_stats": {...},
        "training_time": 3.2  # hours
    }
}
```

## 🚀 **Quick Start Commands**

### **1. Setup Repository:**
```bash
git clone https://github.com/YOUR_USERNAME/maddpg-adversarial-experiments.git
cd maddpg-adversarial-experiments
chmod +x scripts/setup_environment.sh
./scripts/setup_environment.sh
```

### **2. Run Full Experiment:**
```bash
# Train all variants with and without attacks
./scripts/run_full_experiment.sh --gpu 0 --variants all --epochs 200
```

### **3. Generate Thesis Plots:**
```bash
# Generate publication-ready graphs
./scripts/generate_thesis_plots.sh --output thesis_graphs/
```

## 💾 **Remote Development Workflow**

### **VSCode Remote SSH:**
1. Connect to GPU server via SSH
2. Open project folder remotely  
3. Use integrated terminal for training
4. Edit code locally, run remotely
5. Use VSCode's GPU monitoring extensions

### **Development Cycle:**
```bash
# 1. Pull latest changes
git pull origin main

# 2. Run experiment
python experiments/baseline_training.py --config config/experiment_1.yml

# 3. Monitor progress (in separate terminal)
watch -n 10 nvidia-smi

# 4. Collect results
python src/visualization/thesis_plots.py --results data/results/latest/

# 5. Commit results
git add data/results/exp_$(date +%Y%m%d_%H%M%S)/
git commit -m "Experiment results: baseline training complete"
git push origin main
```

## 🔍 **Monitoring & Debugging**

### **GPU Monitoring:**
```bash
# Monitor GPU usage
watch -n 2 nvidia-smi

# Monitor training progress
tail -f logs/training.log

# Check experiment status
python scripts/check_experiment_status.py
```

### **Remote Visualization:**
```bash
# Forward Jupyter port for interactive analysis
ssh -L 8888:localhost:8888 user@gpu-server
jupyter lab --ip localhost --port 8888
```

## 📋 **Collaboration Workflow**

### **For Student:**
1. Fork the repository
2. Clone to local machine and GPU server
3. Run experiments using provided scripts
4. Commit results to personal branch
5. Create pull request with findings

### **For Supervisor:**
1. Review experiment configurations
2. Monitor progress via Git commits
3. Analyze results remotely
4. Provide feedback via GitHub issues

## ⚡ **Performance Optimization**

### **GPU Training Tips:**
- Use mixed precision training (`--fp16`)
- Batch multiple episodes together
- Leverage PyTorch DataParallel for multi-GPU
- Monitor memory usage to prevent OOM errors

### **Data Management:**
- Use Git LFS for large model files
- Compress result files (JSON → gzipped)
- Archive old experiments periodically
- Use rsync for efficient file transfers

---

**Next Step:** Choose deployment option and I'll create the complete repository structure with all integration code!