#!/usr/bin/env bash
# load_weights.sh — Pull trained weights from a remote server into local host_data/.
#
# Usage: bash load_weights.sh <source_server>
# Example: bash load_weights.sh pedroamaral@10.26.110.14
#
# Use this to pull weights from whichever server has the fully trained models
# so you can run Phase 2 or Phase 3 locally or on another server.
set -euo pipefail

SRC_HOST="${1:-}"
if [[ -z "$SRC_HOST" ]]; then
  echo "Usage: bash load_weights.sh <user@host>"
  echo "Example: bash load_weights.sh pedroamaral@10.26.110.14"
  exit 1
fi

DST="host_data/results/main_run"
SRC="$SRC_HOST:~/maddpg-adversarial-experiments/host_data/results/main_run"

echo "Pulling models from $SRC_HOST ..."
mkdir -p "$DST/models"
rsync -av --info=progress2 \
  "$SRC/models/" \
  "$DST/models/"

echo "Pulling phase1_training_results.json"
rsync -av \
  "$SRC/phase1_training_results.json" \
  "$DST/"

echo ""
echo "Done. You can now run:"
echo "  bash run_maddpg_eval.sh   # Phase 2 — MADDPG evaluation"
echo "  bash run_fgsm_eval.sh     # Phase 3 — FGSM adversarial evaluation"
