#!/usr/bin/env bash
# Train the 4 remaining variants (CC-Duelling, CC-Simple-GNN, CC-Duelling-GNN, LC-Duelling-GNN).
# Uses experiment_config_remaining.json — does NOT retrain CC-Simple or LC-Duelling.
# Run this on server 10.26.110.14 while 10.26.110.15 runs the FGSM evaluation.
set -euo pipefail
mkdir -p host_data/results host_data/models host_logs
docker rm -f maddpg_training 2>/dev/null || true
docker run -d --gpus all --name maddpg_training \
  -v "$(pwd)/host_data":/workspace/data \
  -v "$(pwd)/host_logs":/workspace/logs \
  -v "$(pwd)/experiment_config_remaining.json":/workspace/experiment_config.json \
  -v "$(pwd)/src":/workspace/src \
  maddpg-exp:latest \
  python src/standalone_experiment_runner.py \
    --config experiment_config.json \
    --gpu 0 \
    --phase train \
    --results-dir data/results/main_run
echo "Training (remaining variants) started. Follow with: ./check_progress_training.sh"
