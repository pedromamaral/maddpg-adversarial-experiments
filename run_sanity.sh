#!/usr/bin/env bash
set -euo pipefail

IMAGE="maddpg-adversarial:latest"
CONTAINER="maddpg-sanity"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "[ERROR] Docker image '$IMAGE' not found."
  exit 1
fi

docker rm -f "$CONTAINER" 2>/dev/null || true
mkdir -p ./host_data/results ./host_data/models ./host_logs

GPU_FLAGS=""
GPU_ARG="--gpu -1"
if docker run --rm --gpus all "$IMAGE" nvidia-smi >/dev/null 2>&1; then
  echo "GPU detected — running with CUDA acceleration"
  GPU_FLAGS="--gpus all"
  GPU_ARG="--gpu 0"
fi

docker run -d \
  $GPU_FLAGS \
  --name "$CONTAINER" \
  -v "$(pwd)/host_data:/workspace/data" \
  -v "$(pwd)/host_logs:/workspace/logs" \
  -v "$(pwd)/sanity_config.json:/workspace/sanity_config.json" \
  "$IMAGE" \
  python standalone_experiment_runner.py --config sanity_config.json $GPU_ARG

echo "✅ CC-Simple sanity run started (container: $CONTAINER)"
echo "Monitor with: ./check_progress_sanity.sh"
echo "Results in:   host_data/results/"
