#!/usr/bin/env bash
# Tail the logs of the running full-experiment container.
# Usage: ./check_progress_full.sh
# Press Ctrl-C to stop tailing; the container keeps running.

CONTAINER="maddpg-full"

if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "[INFO] Container '$CONTAINER' is not currently running."
    echo "       Showing last 50 log lines (if the container has run before):"
    docker logs --tail 50 "$CONTAINER" 2>/dev/null || \
        echo "       No logs found. Start the experiment with: ./run_full_experiment.sh"
    exit 0
fi

echo "Tailing logs for '$CONTAINER'  (Ctrl-C to stop following):"
echo "-----------------------------------------------------------"
docker logs -f "$CONTAINER"
