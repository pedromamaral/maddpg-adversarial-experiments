#!/bin/bash
# Run complete MADDPG adversarial robustness experiments

echo "🚀 MADDPG Complete Experiment Runner"
echo "===================================="

# Check if Docker image exists
if ! docker image inspect maddpg-adversarial:latest >/dev/null 2>&1; then
    echo "❌ Docker image not found. Run ./docker_setup.sh first."
    exit 1
fi

# Create data directories
mkdir -p ./host_data/results ./host_data/models ./host_logs

echo "🧪 Starting complete adversarial robustness evaluation..."
echo "⏱️  Estimated time: 6-12 hours on GPU"
echo "📁 Results will be saved to: host_data/results/"
echo ""

# Run complete experiments in container
docker run --rm \
    --gpus all \
    --name maddpg-experiment-full \
    -v $(pwd)/host_data:/workspace/data \
    -v $(pwd)/host_logs:/workspace/logs \
    maddpg-adversarial:latest \
    python standalone_experiment_runner.py --gpu 0

echo ""
echo "✅ Complete experiments finished!"
echo "📊 Results location: host_data/results/"
echo "📈 Thesis graphs: host_data/results/thesis_graphs/"
echo "💾 Trained models: host_data/models/"