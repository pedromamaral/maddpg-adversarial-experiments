#!/usr/bin/env bash
CONTAINER="maddpg-sanity"

if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
  echo "[INFO] Container '$CONTAINER' is not running."
  docker logs --tail 50 "$CONTAINER" 2>/dev/null || echo "No logs found."
  exit 0
fi

echo "Tailing logs for '$CONTAINER' (Ctrl-C to stop):"
echo "--------------------------------------------------"
docker logs -f "$CONTAINER"
