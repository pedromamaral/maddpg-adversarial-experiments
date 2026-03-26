#!/usr/bin/env bash
# Tail the logs of the running quick-test container.
# Usage: ./check_progress_quick.sh
# Press Ctrl-C to stop tailing; the container keeps running.

CONTAINER="maddpg-quick"

if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "[INFO] Container '$CONTAINER' is not currently running."
    echo "       Showing last 50 log lines (if the container has run before):"
    docker logs --tail 50 "$CONTAINER" 2>/dev/null || \
        echo "       No logs found. Start the test with: ./test_quick.sh"
    exit 0
fi

echo "Tailing logs for '$CONTAINER'  (Ctrl-C to stop following):"
echo "-----------------------------------------------------------"
docker logs -f "$CONTAINER"
