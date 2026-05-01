#!/usr/bin/env bash
# save_weights.sh — Copy trained model weights from host_data into pretrained_weights/
# so they can be committed to the repository via Git LFS.
#
# Run this after a successful training phase completes.
# Usage: bash save_weights.sh
set -euo pipefail

SRC="host_data/results/main_run"
DST="pretrained_weights"

if [[ ! -f "$SRC/phase1_training_results.json" ]]; then
  echo "ERROR: $SRC/phase1_training_results.json not found. Has training completed?"
  exit 1
fi

echo "Saving weights: $SRC/models/ → $DST/models/"
mkdir -p "$DST/models"
rsync -a --info=progress2 "$SRC/models/" "$DST/models/"

echo "Saving phase1_training_results.json"
cp "$SRC/phase1_training_results.json" "$DST/phase1_training_results.json"

echo ""
echo "Done. Commit with:"
echo "  git add pretrained_weights/"
echo "  git commit -m 'chore: save pretrained weights for all variants'"
echo "  git push"
