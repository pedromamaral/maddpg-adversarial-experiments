#!/usr/bin/env bash
# Launch the quick validation test in detached Docker mode.
# Usage: ./test_quick.sh

set -euo pipefail

IMAGE="maddpg-adversarial:latest"
CONTAINER="maddpg-quick"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "[ERROR] Docker image '$IMAGE' not found."
    echo "        Build it first: docker build -t $IMAGE ."
    exit 1
fi

# Remove any stale container with the same name
docker rm -f "$CONTAINER" 2>/dev/null || true

# Ensure host_data directory exists and is writable
if [ -d "./host_data" ] && [ ! -w "./host_data" ]; then
    echo "Fixing permissions for host_data directory..."
    sudo chown -R $USER:$USER ./host_data 2>/dev/null || true
fi

mkdir -p ./host_data/results ./host_data/models ./host_logs

# Detect GPU availability
GPU_FLAGS=""
GPU_ARG="--gpu -1"
if docker run --rm --gpus all "$IMAGE" nvidia-smi >/dev/null 2>&1; then
    echo "GPU detected — running with CUDA acceleration"
    GPU_FLAGS="--gpus all"
    GPU_ARG="--gpu 0"
else
    echo "No GPU / NVIDIA runtime detected — running on CPU"
fi

docker run -d \
    $GPU_FLAGS \
    --name "$CONTAINER" \
    -v "$(pwd)/host_data:/workspace/data" \
    -v "$(pwd)/host_logs:/workspace/logs" \
    "$IMAGE" \
    python standalone_experiment_runner.py --quick $GPU_ARG

echo "Quick test started in detached mode  (container: $CONTAINER)"
echo "Monitor progress with: ./check_progress_quick.sh"
