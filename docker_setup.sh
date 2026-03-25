#!/bin/bash
# Docker-based setup for MADDPG experiments on GPU servers

set -e

echo "🐳 MADDPG Docker Setup for GPU Servers"
echo "======================================"

# Check if docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

# Check if nvidia-docker/nvidia-container-runtime is available
if ! docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA Docker not working. Check nvidia-container-runtime installation."
    echo "🛠️  Install with: sudo apt install nvidia-container-runtime"
    exit 1
fi

echo "✅ Docker with NVIDIA GPU support detected"

# Build the Docker image
echo "🔨 Building MADDPG Docker image..."
docker build -t maddpg-adversarial:latest .

echo "✅ Docker image built successfully!"

# Create convenience scripts
echo "📝 Creating convenience scripts..."

# Run container script
cat > run_container.sh << 'EOF'
#!/bin/bash
# Run MADDPG container with GPU access

echo "🚀 Starting MADDPG container..."

# Create host directories for data persistence
mkdir -p ./host_data/results ./host_data/models ./host_logs

# Run container with GPU access and volume mounts
docker run -it --rm \
    --gpus all \
    --name maddpg-experiment \
    -v $(pwd)/host_data:/workspace/data \
    -v $(pwd)/host_logs:/workspace/logs \
    -p 8888:8888 \
    maddpg-adversarial:latest

EOF
chmod +x run_container.sh

# Run experiments script  
cat > run_experiments.sh << 'EOF'
#!/bin/bash
# Run experiments inside container

echo "🧪 Running MADDPG experiments in container..."

docker run --rm \
    --gpus all \
    --name maddpg-experiment-batch \
    -v $(pwd)/host_data:/workspace/data \
    -v $(pwd)/host_logs:/workspace/logs \
    maddpg-adversarial:latest \
    python standalone_experiment_runner.py --gpu 0

echo "✅ Experiments complete! Check host_data/results/"

EOF
chmod +x run_experiments.sh

# Quick test script
cat > test_container.sh << 'EOF'
#!/bin/bash
# Quick test of container setup

echo "🧪 Testing MADDPG container..."

docker run --rm \
    --gpus all \
    maddpg-adversarial:latest \
    python test_standalone.sh

EOF
chmod +x test_container.sh

# Jupyter script
cat > run_jupyter.sh << 'EOF'
#!/bin/bash
# Run Jupyter Lab in container

echo "📊 Starting Jupyter Lab in container..."
echo "🔗 Access at: http://localhost:8888"

docker run -it --rm \
    --gpus all \
    --name maddpg-jupyter \
    -v $(pwd)/host_data:/workspace/data \
    -v $(pwd)/host_logs:/workspace/logs \
    -p 8888:8888 \
    maddpg-adversarial:latest \
    jupyter lab --ip=0.0.0.0 --allow-root --no-browser

EOF
chmod +x run_jupyter.sh

# Interactive shell script
cat > shell.sh << 'EOF'
#!/bin/bash
# Interactive shell in container

echo "🐚 Starting interactive shell in container..."

docker run -it --rm \
    --gpus all \
    --name maddpg-shell \
    -v $(pwd)/host_data:/workspace/data \
    -v $(pwd)/host_logs:/workspace/logs \
    maddpg-adversarial:latest \
    bash

EOF
chmod +x shell.sh

# Monitor script
cat > monitor.sh << 'EOF'
#!/bin/bash
# Monitor running container

echo "📊 Monitoring container..."

if docker ps | grep -q maddpg; then
    echo "✅ Container is running"
    echo ""
    echo "📈 GPU Usage:"
    docker exec -it $(docker ps --filter "name=maddpg" --format "{{.Names}}" | head -1) nvidia-smi
    echo ""
    echo "💾 Container Stats:"
    docker stats --no-stream $(docker ps --filter "name=maddpg" --format "{{.Names}}" | head -1)
else
    echo "❌ No MADDPG container running"
fi

EOF
chmod +x monitor.sh

echo ""
echo "✅ Docker setup complete!"
echo "========================"
echo ""
echo "📁 Created scripts:"
echo "   🐳 run_container.sh     - Interactive container"
echo "   🧪 run_experiments.sh   - Run full experiments" 
echo "   ⚡ test_container.sh     - Quick test"
echo "   📊 run_jupyter.sh       - Jupyter Lab"
echo "   🐚 shell.sh             - Interactive shell"
echo "   📈 monitor.sh           - Monitor running container"
echo ""
echo "🎯 Quick start:"
echo "   ./test_container.sh              # Test setup"
echo "   ./run_container.sh               # Interactive mode"
echo "   ./run_experiments.sh             # Run experiments"
echo ""
echo "📊 Data persistence:"
echo "   ./host_data/results/             # Experiment results"
echo "   ./host_data/models/              # Trained models"  
echo "   ./host_logs/                     # Log files"
echo ""
echo "🚀 Container includes:"
echo "   ✅ CUDA-enabled PyTorch"
echo "   ✅ All dependencies installed"
echo "   ✅ Framework code ready"
echo "   ✅ GPU access configured"
echo "   ✅ Jupyter Lab available"