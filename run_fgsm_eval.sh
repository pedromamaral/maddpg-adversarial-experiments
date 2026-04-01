#!/usr/bin/env bash
set -euo pipefail
mkdir -p host_logs
docker rm -f maddpg_fgsm 2>/dev/null || true
docker run -d --gpus all --name maddpg_fgsm \
  -v "$(pwd)/host_data":/workspace/data \
  -v "$(pwd)/host_logs":/workspace/logs \
  -v "$(pwd)/experiment_config.json":/workspace/experiment_config.json \
  maddpg-exp:latest \
  python src/standalone_experiment_runner.py \
    --config experiment_config.json \
    --gpu 0 \
    --phase paper2 \
    --results-dir data/results/main_run
echo "Paper-2 eval started. Follow with: ./check_progress_fgsm.sh"