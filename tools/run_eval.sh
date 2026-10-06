#!/bin/bash
# Evaluate trained policies: does each one route, and how does it compare with
# the rule baselines on identical traffic?
#
#   tools/run_eval.sh RUN CONFIG VARIANT [N_FAILURES]
#
#   RUN         results dir under host_data/results holding models/<VARIANT>/
#   CONFIG      the config the run was trained with (repo-relative): it carries the
#               actor/critic heads, without which the checkpoints do not load
#   N_FAILURES  space-separated link-failure counts (default "0 2 4 6 8")
#
# Writes, under host_data/results/RUN/eval/:
#   n_<k>/damage_ceiling.json   policy + greedy / sp / random / worst on the same 20
#                               episodes at 2x hotspot with k random link failures;
#                               the rule rows must equal ceil2x / failsev_fixed
#   <VARIANT>_attribution.json  static share, greedy agreement, saturation, feature
#                               importance (tools/policy_attribution.py)
# Optional environment:
#   RULES      rule labels to roll out instead of the standard five, e.g.
#              "policy greedy random random+flow greedy_stale4" (see
#              StandaloneExperimentRunner._parse_rule_spec)
#   EVAL_NAME  output subdirectory instead of eval/ (keeps the standard evaluation intact)
#   NO_ATTR=1  skip the attribution run
#   LOAD       offered load factor (default 2.0, the Paper 1 stress point)
# Log: ~/<EVAL_NAME>_<RUN>_<VARIANT>.log. Run on the server from the repo root.
set -u
RUN=$1; CFG=$2; V=$3; NS=${4:-"0 2 4 6 8"}
EV=${EVAL_NAME:-eval}; export RULES=${RULES:-} LOAD=${LOAD:-}
R=$(pwd); LOG=$HOME/${EV}_${RUN}_${V}.log
[ -f "$CFG" ] || { echo "no config $CFG (run from the repo root)"; exit 1; }
[ -d "host_data/results/$RUN/models/$V" ] || { echo "no models for $V in $RUN"; exit 1; }
say() { echo "$(date -u '+%F %T UTC') $*" >> "$LOG"; }
IMG=maddpg-exp:latest
MOUNTS="-v $R/host_data:/workspace/data -v $R/src:/workspace/src -v $R/tools:/workspace/tools -v $R/configs:/workspace/configs"

ceiling() {  # $1 = number of failures
  K=$1; OUT="host_data/results/$RUN/$EV/n_$K"
  [ -f "$OUT/damage_ceiling.json" ] && { say "n=$K done"; return; }
  mkdir -p "$OUT" && ln -sfn ../../models "$OUT/models"
  # Without the link the container would create an empty models/ of its own and
  # the checkpoint guard would (rightly) refuse to evaluate untrained weights.
  [ -L "$OUT/models" ] || { say "GUARD: could not link models into $OUT"; return; }
  C=/tmp/${EV}_${RUN}_${V}_n$K.json
  python3 - "$CFG" "$V" "$K" "$C" <<'PY'
import json, os, sys
cfg, v, k, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
c = json.load(open(cfg))
c["variants"] = [x for x in c["variants"] if x["name"] == v]
ae = c.setdefault("attack_eval", {})
ae.update({"offered_load_factor": float(os.environ.get("LOAD") or 2.0), "n_link_failures": k, "ceiling_episodes": 20})
ae.pop("target_links", None)
if os.environ.get("RULES"):
    ae["ceiling_rules"] = os.environ["RULES"].split()
json.dump(c, open(out, "w"), indent=1)
PY
  say "--- ceiling n=$K ---"
  docker run --rm --name "${EV}_${RUN}_${V}_$K" --gpus all $MOUNTS -v "$C:/workspace/experiment_config.json" "$IMG" \
    python src/standalone_experiment_runner.py --config experiment_config.json --gpu 0 \
      --phase ceiling --results-dir "data/results/$RUN/$EV/n_$K" >> "$LOG" 2>&1
  say "ceiling n=$K exited rc=$?"; rm -f "$C"
}

say "=== eval $RUN $V ($CFG; n = $NS) ==="
# Training containers write as root; take the run over so the eval dirs and model
# links can be created from here.
docker run --rm -v "$R/host_data/results/$RUN:/r" "$IMG" chown -R "$(id -u):$(id -g)" /r > /dev/null 2>&1
mkdir -p "host_data/results/$RUN/$EV" || { say "GUARD: cannot write host_data/results/$RUN/$EV"; exit 1; }
for K in $NS; do ceiling "$K" & done
ATTR="host_data/results/$RUN/eval/${V}_attribution.json"
if [ -n "${NO_ATTR:-}" ] || [ -f "$ATTR" ] || docker ps --format '{{.Names}}' | grep -qx "attr_${RUN}_${V}"; then
  say "attribution skipped, done or running"
else
  docker run --rm --name "attr_${RUN}_${V}" --gpus all $MOUNTS -v "$R/$CFG:/workspace/eval_config.json" "$IMG" \
    python tools/policy_attribution.py --config eval_config.json --results "data/results/$RUN" \
      --variants "$V" --failures 0,2,4 --out "data/results/$RUN/eval/${V}_attribution.json" >> "$LOG" 2>&1 &
fi
wait
say "=== eval $RUN $V END ==="
