#!/bin/bash
# Setup script for GPU server environment

echo "🚀 Setting up MADDPG Adversarial Robustness Experiment Environment"
echo "=================================================================="

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "✅ CUDA detected:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
else
    echo "⚠️  Warning: CUDA not detected. Will use CPU (much slower)."
fi

# Create conda environment
echo "📦 Creating conda environment..."
conda create -n maddpg-attacks python=3.9 -y
source activate maddpg-attacks

# Install PyTorch with CUDA support
echo "🔥 Installing PyTorch with CUDA..."
if command -v nvidia-smi &> /dev/null; then
    # Install CUDA version
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
else
    # Install CPU version
    conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
fi

# Install additional requirements
echo "📚 Installing additional packages..."
pip install torch-geometric
pip install matplotlib seaborn
pip install numpy pandas
pip install networkx
pip install tensorboard
pip install wandb
pip install jupyter jupyterlab
pip install tqdm
pip install psutil

# Create project directories
echo "📁 Creating project structure..."
mkdir -p data/models
mkdir -p data/results
mkdir -p data/topologies
mkdir -p logs
mkdir -p notebooks
mkdir -p scripts

# Set up Git LFS for large files (optional)
if command -v git-lfs &> /dev/null; then
    echo "🗂️  Setting up Git LFS for large files..."
    git lfs track "*.pth"
    git lfs track "*.pkl"
    git lfs track "*.h5"
fi

# Create environment activation script
cat > activate_env.sh << 'EOF'
#!/bin/bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate maddpg-attacks
export CUDA_VISIBLE_DEVICES=0
echo "🐍 Environment activated: maddpg-attacks"
echo "🎮 GPU: $CUDA_VISIBLE_DEVICES"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
EOF
chmod +x activate_env.sh

# Create quick test script
cat > test_setup.py << 'EOF'
#!/usr/bin/env python3
"""Quick test to verify installation"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

def test_pytorch():
    print("🔥 Testing PyTorch...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    
    # Test tensor operations
    x = torch.randn(3, 3)
    if torch.cuda.is_available():
        x = x.cuda()
        print(f"Tensor device: {x.device}")
    print("✅ PyTorch working correctly")

def test_imports():
    print("📦 Testing imports...")
    try:
        import matplotlib
        import seaborn
        import pandas
        import networkx
        print("✅ All packages imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        sys.exit(1)

def test_visualization():
    print("📊 Testing visualization...")
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    
    # Create simple test plot
    plt.figure(figsize=(6, 4))
    plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
    plt.title("Test Plot")
    plt.savefig("test_plot.png")
    plt.close()
    print("✅ Visualization working correctly")

if __name__ == "__main__":
    print("🧪 Testing environment setup...")
    test_pytorch()
    test_imports()
    test_visualization()
    print("✅ All tests passed! Environment ready for experiments.")
EOF

chmod +x test_setup.py

# Create experiment runner script
cat > run_experiment.sh << 'EOF'
#!/bin/bash
# Main experiment runner script

# Activate environment
source activate_env.sh

# Default configuration
CONFIG="experiment_config.json"
MODE="full"
GPU=0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --mode)
            MODE="$2" 
            shift 2
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "OPTIONS:"
            echo "  --config FILE    Configuration file (default: experiment_config.json)"
            echo "  --mode MODE      Experiment mode: baseline, attack, full (default: full)"
            echo "  --gpu ID         GPU device ID (default: 0)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set GPU
export CUDA_VISIBLE_DEVICES=$GPU

# Check if config file exists
if [[ ! -f "$CONFIG" ]]; then
    echo "❌ Configuration file not found: $CONFIG"
    exit 1
fi

echo "🚀 Running MADDPG Adversarial Robustness Experiment"
echo "=================================================="
echo "Config: $CONFIG"
echo "Mode: $MODE"
echo "GPU: $GPU"
echo ""

# Create log file
LOG_FILE="logs/experiment_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

# Run experiment
python experiment_runner.py \
    --config "$CONFIG" \
    --mode "$MODE" \
    --gpu "$GPU" \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "📋 Experiment complete!"
echo "📄 Log file: $LOG_FILE"
echo "📁 Results directory: data/results/"
EOF

chmod +x run_experiment.sh

# Create monitoring script
cat > monitor_gpu.sh << 'EOF'
#!/bin/bash
# GPU monitoring script

echo "🎮 Monitoring GPU usage (Press Ctrl+C to stop)"
echo "=============================================="

while true; do
    clear
    echo "$(date)"
    echo "=============================================="
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
        echo ""
        echo "Running processes:"
        nvidia-smi --query-compute-apps=pid,process_name,gpu_instance_id,used_memory --format=csv,noheader
    else
        echo "NVIDIA GPU not available"
    fi
    echo ""
    echo "System memory:"
    free -h
    echo ""
    sleep 5
done
EOF

chmod +x monitor_gpu.sh

# Test the installation
echo "🧪 Running setup tests..."
python test_setup.py

echo ""
echo "✅ Setup complete!"
echo "=================="
echo "📝 Next steps:"
echo "   1. Activate environment: source activate_env.sh"
echo "   2. Test installation: python test_setup.py" 
echo "   3. Run experiments: ./run_experiment.sh"
echo "   4. Monitor GPU: ./monitor_gpu.sh"
echo ""
echo "📁 Project structure created in current directory"
echo "🔗 Remember to clone the student's MADDPG code and update import paths"