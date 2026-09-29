#!/bin/bash
# Run the fgsm_probe phase for several victims in parallel, one container each.
#
#   tools/run_probe_grid.sh RUN OVERRIDES [MODELS] [VARIANTS]
#
#   RUN        results go to host_data/results/RUN/<variant>/, log to ~/RUN.log
#   OVERRIDES  JSON file whose "attack_eval" keys replace those of the canonical
#              config (reward_fix_full_config.json), e.g. configs/probe_*.json
#   MODELS     victim weights, relative to the results/RUN/<variant> dir
#              (default ../../reward_fix/models, the canonical victims)
#   VARIANTS   space-separated list (default: all seven)
#
# A variant whose fgsm_probe_results.json exists is skipped, so a rerun resumes.
# Run on the server from the repo root, e.g. with nohup ... &.
set -u
RUN=$1; OVR=$2
MODELS=${3:-../../reward_fix/models}
VARIANTS=${4:-"CC-Simple CC-Duelling LC-Simple LC-Duelling CC-Simple-GNN CC-Duelling-GNN LC-Duelling-GNN"}
R=$(pwd)
LOG=$HOME/$RUN.log
IMG=maddpg-exp:latest
[ -f reward_fix_full_config.json ] || { echo "run from the repo root"; exit 1; }
[ -f "$OVR" ] || { echo "no overrides file $OVR"; exit 1; }
say() { echo "$(date -u '+%F %T UTC') $*" >> "$LOG"; }

one() {  # $1 = variant
  V=$1
  OUT="host_data/results/$RUN/$V"
  RES="$OUT/fgsm_probe_results.json"
  [ -f "$RES" ] && { say "$V already done"; return; }
  mkdir -p "$OUT"
  if [ -e "$OUT/models" ] && [ ! -L "$OUT/models" ]; then say "GUARD ABORT: $OUT/models is a real dir"; return; fi
  ln -sfn "$MODELS" "$OUT/models"
  [ -d "$(readlink -f "$OUT/models")/$V" ] || { say "GUARD: no weights for $V under $MODELS"; return; }
  CFG=/tmp/${RUN}_${V}.json
  python3 - "$V" "$OVR" "$CFG" <<'PY'
import json, sys
v, ovr, out = sys.argv[1:]
cfg = json.load(open("reward_fix_full_config.json"))
cfg["variants"] = [x for x in cfg["variants"] if x["name"] == v]
assert cfg["variants"], v
cfg.setdefault("attack_eval", {}).update(json.load(open(ovr))["attack_eval"])
json.dump(cfg, open(out, "w"), indent=1)
PY
  for ATTEMPT in 1 2 3; do
    say "--- $V attempt $ATTEMPT ---"
    docker run --rm --name "probe_${RUN}_${V}" --gpus all \
      -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      -v "$R/host_data:/workspace/data" -v "$R/host_logs:/workspace/logs" \
      -v "$CFG:/workspace/experiment_config.json" -v "$R/src:/workspace/src" "$IMG" \
      python src/standalone_experiment_runner.py --config experiment_config.json \
        --gpu 0 --phase fgsm_probe --results-dir "data/results/$RUN/$V" > "$HOME/${RUN}_${V}.log" 2>&1
    rc=$?
    say "--- $V attempt $ATTEMPT exited rc=$rc ---"
    [ -f "$RES" ] && { say "$V complete"; return; }
    sleep 30
  done
  say "$V GAVE UP"
}

say "=== $RUN START ($OVR; models $MODELS) ==="
for V in $VARIANTS; do one "$V" & done
wait
say "=== $RUN END ==="
