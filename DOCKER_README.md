# MADDPG Adversarial Robustness - Docker Edition

🐳 **GPU-Accelerated Docker Setup for Clean, Reproducible Experiments**

## 🚀 **Why Docker is Perfect for This:**

- ✅ **No environment conflicts** - Clean CUDA PyTorch environment
- ✅ **Reproducible across servers** - Same environment everywhere  
- ✅ **GPU access included** - nvidia-docker integration
- ✅ **Data persistence** - Results saved to host filesystem
- ✅ **Easy deployment** - Single command setup

## 🎯 **Quick Start (GPU Server)**

### **Step 1: Clone and Setup**
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/maddpg-adversarial-experiments.git
cd maddpg-adversarial-experiments

# Build Docker environment  
./docker_setup.sh
```

### **Step 2: Test Setup**
```bash
# Quick test (2 minutes)
./test_container.sh
```

### **Step 3: Run Experiments**
```bash
# Full experiments (6-12 hours)
./run_experiments.sh

# OR interactive mode
./run_container.sh
```

## 🐳 **Docker Scripts Created**

| Script | Purpose | Usage |
|--------|---------|--------|
| `docker_setup.sh` | Build Docker image | One-time setup |
| `test_container.sh` | Quick functionality test | Verify setup works |
| `run_experiments.sh` | Run complete experiments | Batch execution |
| `run_container.sh` | Interactive container | Development/debugging |
| `run_jupyter.sh` | Jupyter Lab interface | Data analysis |
| `shell.sh` | Interactive bash shell | Direct access |
| `monitor.sh` | Monitor running container | Check GPU usage |

## 📊 **Data Persistence**

Results are automatically saved to host filesystem:

```
host_data/
├── results/          # All experiment results
│   ├── complete_experiment_results.json
│   ├── comprehensive_summary.json
│   └── thesis_graphs/
└── models/           # Trained MADDPG models
    ├── CC-Simple/
    ├── CC-Duelling/
    └── ...

host_logs/            # Training logs and monitoring
```

## 🔧 **Container Features**

### **🎯 Base Image**
- **pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel**
- Pre-configured CUDA environment
- No dependency conflicts

### **📦 Included Packages**
- ✅ PyTorch with CUDA 11.8
- ✅ torch-geometric for GNN support
- ✅ All framework dependencies
- ✅ Jupyter Lab for interactive analysis
- ✅ TensorBoard and Wandb for monitoring

### **⚡ GPU Configuration**
- Automatic CUDA detection
- nvidia-smi integration
- GPU memory monitoring
- Multi-GPU ready

## 🧪 **Experiment Workflow**

### **Quick Test (2 minutes)**
```bash
./test_container.sh
# Output: ✅ All imports successful, GPU detected, basic functionality verified
```

### **Interactive Development**
```bash
./run_container.sh
# Drops you into container shell with all tools available
> python standalone_experiment_runner.py --quick
> python analyze_topology.py
> jupyter lab --ip=0.0.0.0 --allow-root
```

### **Batch Experiments**
```bash
./run_experiments.sh
# Runs complete experiment pipeline
# Results automatically saved to host_data/
```

### **Monitoring**
```bash
# In another terminal
./monitor.sh
# Shows GPU usage, memory consumption, container stats
```

## 📈 **Expected Output**

Container startup shows:
```
🚀 MADDPG Adversarial Robustness Container
========================================
🐍 Python: Python 3.9.16
🔥 PyTorch: 2.1.0+cu118
🎯 CUDA available: True
✅ GPU: GeForce RTX 2080 Ti
💾 GPU Memory: 11.0GB
📁 Workspace: /workspace
📊 Results: /workspace/data/results

🎯 Available commands:
  python test_standalone.sh           # Test framework
  python standalone_experiment_runner.py --quick  # Quick test
  python standalone_experiment_runner.py --gpu 0  # Full experiment
  jupyter lab --ip=0.0.0.0 --allow-root         # Start Jupyter
```

## 🎓 **Advantages for Thesis Work**

### **🔬 Reproducibility**
- **Exact same environment** on any CUDA-enabled server
- **Dockerhub deployment** for sharing with reviewers
- **Version-controlled dependencies** 

### **🚀 Portability**
- Works on **any GPU server** with Docker + nvidia-docker
- **No manual environment setup** required
- **Instant deployment** for collaborators

### **🛡️ Isolation**
- **No conflicts** with system Python/packages
- **Clean slate** for each experiment run
- **Rollback capability** if issues arise

### **📊 Data Management**
- **Persistent results** on host filesystem
- **Easy backup/transfer** of experimental data
- **Jupyter integration** for analysis

## 🔧 **Advanced Usage**

### **Custom Configuration**
```bash
# Edit experiment parameters
vim experiment_config.json

# Rebuild container with changes
docker build -t maddpg-adversarial:latest .

# Run with custom config
./run_experiments.sh
```

### **Multi-GPU Experiments**
```bash
# Container supports multi-GPU automatically
docker run --gpus all ...  # Uses all available GPUs
docker run --gpus '"device=0,1"' ...  # Specific GPUs
```

### **Remote Access**
```bash
# Start Jupyter on server
./run_jupyter.sh

# Access from local machine (SSH tunnel)
ssh -L 8888:localhost:8888 user@gpu-server
# Open: http://localhost:8888
```

## 🎯 **Perfect for Student Use**

The student gets:
- ✅ **Zero setup complexity** - just run Docker commands
- ✅ **Guaranteed working environment** - no dependency issues
- ✅ **Professional deployment** - industry-standard containerization
- ✅ **Easy result sharing** - data persisted on host
- ✅ **Scalable to multiple servers** - same image everywhere

**This Docker setup eliminates ALL the environment issues and provides a clean, professional research platform!** 🚀🐳