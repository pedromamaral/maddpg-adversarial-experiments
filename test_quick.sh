#!/usr/bin/env bash
set -euo pipefail
mkdir -p host_data/results host_data/models host_logs
docker rm -f maddpg_quick 2>/dev/null || true
docker run -d --gpus all --name maddpg_quick \
  -v "$(pwd)/host_data":/workspace/data \
  -v "$(pwd)/host_logs":/workspace/logs \
  -v "$(pwd)/experiment_config.json":/workspace/experiment_config.json \
  maddpg-exp:latest \
  python src/standalone_experiment_runner.py \
    --config experiment_config.json \
    --gpu 0 \
    --phase all \
    --quick \
    --results-dir data/results/quick_test
echo "Smoke-test started. Follow with: ./check_progress_quick.sh"