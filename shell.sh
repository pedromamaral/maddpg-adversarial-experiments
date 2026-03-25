#!/bin/bash
# Interactive shell in MADDPG container for development

echo "🐚 Starting Interactive MADDPG Container"
echo "========================================"

# Check if Docker image exists
if ! docker image inspect maddpg-adversarial:latest >/dev/null 2>&1; then
    echo "❌ Docker image not found. Run ./docker_setup.sh first."
    exit 1
fi

# Create data directories
mkdir -p ./host_data/results ./host_data/models ./host_logs

echo "🔧 Starting interactive container..."
echo "📁 Data mounted to: /workspace/data"
echo "📝 Logs mounted to: /workspace/logs"
echo ""

# Start interactive container
docker run -it --rm \
    --gpus all \
    --name maddpg-interactive \
    -v $(pwd)/host_data:/workspace/data \
    -v $(pwd)/host_logs:/workspace/logs \
    maddpg-adversarial:latest \
    bash

echo "Container exited."