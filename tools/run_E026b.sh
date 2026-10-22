#!/bin/bash
# E026b: held-out eval of the E026 system tables (H1) + per-change own-flow vs system credit (H2). CPU only.
# Run from the repo root on the server. Chain A: credit dc -> credit sp (8 workers). Chain B: eval dc -> eval sp (2 cpus per n).
cd ~/maddpg-adversarial-experiments; R=$(pwd); O=host_data/results/E026b; mkdir -p $O
docker run --rm -v $R/host_data/results/E026b:/r maddpg-exp:latest chown -R $(id -u):$(id -g) /r >/dev/null 2>&1
credit() { c=$1; ld=$2; tag=$3   # config, load, dc|sp
  docker run --rm --name maddpg-E026b-credit-$tag -e TQDM_DISABLE=1 --cpus 8 -v $R/host_data:/workspace/data -v $R/src:/workspace/src -v $R/configs:/workspace/configs -v $R/tools:/workspace/tools -w /workspace --entrypoint python3 maddpg-exp:latest tools/table_change_credit.py --config configs/$c.json --load $ld --search-log data/results/E026_$tag/search_log.json --table-dir data/results/E026_$tag --episodes 20 --workers 8 --out data/results/E026b/credit_$tag.json > ~/idqn/maddpg-E026b-credit-$tag.log 2>&1
  echo "$(date -u +%m-%d\ %H:%M) E026b credit $tag done" >> ~/idqn/run_E026b.done; }
ev() { c=$1; ld=$2; tag=$3; run=$4
  TABLE_FILE=$R/host_data/results/E026_$tag/table_best.json DOCKER_EXTRA="--cpus 2" LOAD=$ld EVAL_NAME=E026b_$tag NO_ATTR=1 RULES="table random+flow greedy greedy_stale2 sp" TQDM_DISABLE=1 \
    bash tools/run_eval.sh $run configs/$c.json LC-Simple "0 4 8" > ~/idqn/maddpg-E026b-eval-$tag.log 2>&1
  echo "$(date -u +%m-%d\ %H:%M) E026b eval $tag done" >> ~/idqn/run_E026b.done; }
( credit v2_deltacom 8 dc; credit v2_pilot4_lc 3 sp ) &
( ev v2_deltacom 8 dc v2_dc_lc; ev v2_pilot4_lc 3 sp v2_p4_lc ) &
wait
