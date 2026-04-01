#!/usr/bin/env bash
set -euo pipefail
mkdir -p host_data/results host_data/models host_logs
docker run -d --gpus all --name maddpg_training \
  -v "$(pwd)/host_data":/workspace/data \
  -v "$(pwd)/host_logs":/workspace/logs \
  maddpg-exp:latest \
  python src/standalone_experiment_runner.py \
    --config experiment_config.json \
    --gpu 0 \
    --phase train \
    --results-dir data/results/main_run
echo "Training started. Follow with: ./check_progress_training.sh"