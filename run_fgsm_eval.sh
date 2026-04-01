#!/usr/bin/env bash
set -euo pipefail
mkdir -p host_logs
docker run -d --gpus all --name maddpg_fgsm \
  -v "$(pwd)/host_data":/app/data \
  -v "$(pwd)/host_logs":/app/logs \
  maddpg-exp:latest \
  python src/standalone_experiment_runner.py \
    --config experiment_config.json \
    --gpu 0 \
    --phase paper2 \
    --results-dir data/results/main_run
echo "Paper-2 eval started. Follow with: ./check_progress_fgsm.sh"