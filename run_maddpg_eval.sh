#!/usr/bin/env bash
set -euo pipefail
mkdir -p host_logs
docker rm -f maddpg_paper1 2>/dev/null || true
docker run -d --gpus all --name maddpg_paper1 \
  -v "$(pwd)/host_data":/workspace/data \
  -v "$(pwd)/host_logs":/workspace/logs \
  -v "$(pwd)/experiment_config.json":/workspace/experiment_config.json \
  maddpg-exp:latest \
  python src/standalone_experiment_runner.py \
    --config experiment_config.json \
    --gpu 0 \
    --phase paper1 \
    --results-dir data/results/main_run
echo "Paper-1 eval started. Follow with: ./check_progress_maddpg.sh"