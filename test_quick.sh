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

mkdir -p ./host_data/results ./host_data/models ./host_logs

docker run -d \
    --gpus all \
    --name "$CONTAINER" \
    -v "$(pwd)/host_data:/workspace/data" \
    -v "$(pwd)/host_logs:/workspace/logs" \
    "$IMAGE" \
    python standalone_experiment_runner.py --quick --gpu 0

echo "Quick test started in detached mode  (container: $CONTAINER)"
echo "Monitor progress with: ./check_progress_quick.sh"
