#!/bin/bash
# Quick test of MADDPG Docker container setup

echo "🧪 Quick Docker Container Test"
echo "============================="

# Check if Docker image exists
if ! docker image inspect maddpg-adversarial:latest >/dev/null 2>&1; then
    echo "❌ Docker image not found. Run ./docker_setup.sh first."
    exit 1
fi

echo "🔧 Testing container functionality..."

# Test basic container startup and dependencies
docker run --rm --gpus all \
    -v $(pwd)/host_data:/workspace/data \
    -v $(pwd)/host_logs:/workspace/logs \
    maddpg-adversarial:latest \
    python standalone_experiment_runner.py --quick --gpu 0

echo ""
echo "✅ Quick test complete!"
echo "🎯 If successful, run: ./run_full_experiment.sh"