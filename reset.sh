#!/usr/bin/env bash
set -euo pipefail

# -------------------------------------------------------
# reset.sh — Stop containers and wipe previous results
# Use before re-running experiments from a clean state.
# -------------------------------------------------------

echo "=== MADDPG Experiment Reset ==="
echo ""

# 1. Stop and remove experiment containers
for CONTAINER in maddpg-quick maddpg-full; do
  if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "Removing container: ${CONTAINER}"
    docker rm -f "${CONTAINER}" 2>/dev/null || true
  else
    echo "Container not found (skipping): ${CONTAINER}"
  fi
done

echo ""

# 2. Wipe previous results and logs
echo "Clearing previous results and logs..."
rm -rf host_data/results host_logs

# 3. Re-create clean output directories
mkdir -p host_data/results host_data/models host_logs

echo ""
echo "=== Reset complete ==="
echo "  Removed containers : maddpg-quick, maddpg-full"
echo "  Cleared            : host_data/results, host_logs"
echo "  Preserved          : host_data/models, Docker image"
echo ""
echo "You can now re-run:"
echo "  ./test_quick.sh          # quick validation"
echo "  ./run_full_experiment.sh # full experiment"
