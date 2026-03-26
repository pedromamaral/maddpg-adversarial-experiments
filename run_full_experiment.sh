#!/usr/bin/env bash
# Launch the full adversarial robustness experiment in detached Docker mode.
# Usage: ./run_full_experiment.sh

set -euo pipefail

IMAGE="maddpg-adversarial:latest"
CONTAINER="maddpg-full"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "[ERROR] Docker image '$IMAGE' not found."
    echo "        Build it first: docker build -t $IMAGE ."
    exit 1
fi

# Remove any stale container with the same name
docker rm -f "$CONTAINER" 2>/dev/null || true

mkdir -p ./host_data/results ./host_data/models ./host_logs

docker run -d \
    --gpus all \
    --name "$CONTAINER" \
    -v "$(pwd)/host_data:/workspace/data" \
    -v "$(pwd)/host_logs:/workspace/logs" \
    "$IMAGE" \
    python standalone_experiment_runner.py --gpu 0

echo "Full experiment started in detached mode  (container: $CONTAINER)"
echo "Estimated runtime: 6-12 hours on a modern GPU."
echo "Monitor progress with: ./check_progress_full.sh"
echo "Results will appear in: host_data/results/"
