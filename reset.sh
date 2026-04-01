#!/usr/bin/env bash
set -euo pipefail
docker stop maddpg_training maddpg_paper1 maddpg_fgsm 2>/dev/null || true
docker rm   maddpg_training maddpg_paper1 maddpg_fgsm 2>/dev/null || true
rm -rf host_data host_logs
mkdir -p host_data/results host_data/models host_logs
echo "Full reset complete."