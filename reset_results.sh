#!/usr/bin/env bash
set -euo pipefail
docker stop maddpg_training maddpg_paper1 maddpg_paper2 2>/dev/null || true
docker rm   maddpg_training maddpg_paper1 maddpg_paper2 2>/dev/null || true
rm -rf host_data/results host_logs/*.log
mkdir -p host_data/results host_logs
echo "Results wiped. Models preserved in host_data/models/"