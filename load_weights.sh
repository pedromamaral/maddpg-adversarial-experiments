#!/usr/bin/env bash
# load_weights.sh — Restore pretrained weights from the repository into host_data/
# so that Phase 2 (Paper 1 eval) or Phase 3 (FGSM) can run without retraining.
#
# Usage: bash load_weights.sh
# After this, run Phase 2 with:  bash run_maddpg_eval.sh
# Or Phase 3 with:               bash run_fgsm_eval.sh
set -euo pipefail

SRC="pretrained_weights"
DST="host_data/results/main_run"

if [[ ! -f "$SRC/phase1_training_results.json" ]]; then
  echo "ERROR: $SRC/phase1_training_results.json not found."
  echo "Have you pulled the repository (including LFS objects)?"
  echo "  git lfs pull"
  exit 1
fi

echo "Restoring weights: $SRC/models/ → $DST/models/"
mkdir -p "$DST/models"
rsync -a --info=progress2 "$SRC/models/" "$DST/models/"

echo "Restoring phase1_training_results.json"
mkdir -p "$DST"
cp "$SRC/phase1_training_results.json" "$DST/phase1_training_results.json"

echo ""
echo "Weights restored. You can now run Phase 2 or Phase 3 directly:"
echo "  bash run_maddpg_eval.sh   # Phase 2 — Paper 1 MADDPG evaluation"
echo "  bash run_fgsm_eval.sh     # Phase 3 — FGSM adversarial evaluation"
