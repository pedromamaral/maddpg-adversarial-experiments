#!/bin/bash
# Paper 1 failure-severity sweep: the damage-ceiling rollouts (policy, greedy, sp,
# random, worst) of one victim at 2x hotspot under n random link failures, one
# container per n, in parallel.
#
#   tools/run_failsev.sh RUN [VARIANT] [N_LIST]
#
# Results: host_data/results/RUN/n_<k>/damage_ceiling.json; log ~/RUN.log.
# Needs the failure-accumulation fix (commit 8b4a4f9): before it, each rule after
# the policy inherited the failures of the rules before it.
set -u
RUN=$1; V=${2:-CC-Simple}; NS=${3:-"1 2 4 6 8"}
R=$(pwd); LOG=$HOME/$RUN.log
[ -f reward_fix_full_config.json ] || { echo "run from the repo root"; exit 1; }
say() { echo "$(date -u '+%F %T UTC') $*" >> "$LOG"; }

one() {
  K=$1; OUT="host_data/results/$RUN/n_$K"
  [ -f "$OUT/damage_ceiling.json" ] && { say "n=$K done"; return; }
  mkdir -p "$OUT"; ln -sfn ../../reward_fix/models "$OUT/models"
  [ -d "$(readlink -f "$OUT/models")/$V" ] || { say "GUARD: no weights for $V"; return; }
  CFG=/tmp/${RUN}_n$K.json
  python3 - "$K" "$V" "$CFG" <<'PY'
import json, sys
k, v, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]
c = json.load(open("reward_fix_full_config.json"))
c["variants"] = [x for x in c["variants"] if x["name"] == v]
ae = c.setdefault("attack_eval", {})
ae.update({"offered_load_factor": 2.0, "n_link_failures": k, "ceiling_episodes": 20})
ae.pop("target_links", None)
json.dump(c, open(out, "w"), indent=1)
PY
  say "--- n=$K ---"
  docker run --rm --name "fsev_${RUN}_$K" --gpus all \
    -v "$R/host_data:/workspace/data" -v "$R/host_logs:/workspace/logs" \
    -v "$CFG:/workspace/experiment_config.json" -v "$R/src:/workspace/src" maddpg-exp:latest \
    python src/standalone_experiment_runner.py --config experiment_config.json --gpu 0 \
      --phase ceiling --results-dir "data/results/$RUN/n_$K" > "$HOME/${RUN}_n$K.log" 2>&1
  say "n=$K exited rc=$?"
}

say "=== $RUN START ($V, n = $NS) ==="
for K in $NS; do one "$K" & done
wait
say "=== $RUN END ==="
