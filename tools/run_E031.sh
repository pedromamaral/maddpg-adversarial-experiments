#!/bin/bash
# E031 (A037): failure-aware static table search, J over n in {0,4}, restarted from the E026 table_best. CPU only, no src change.
# Chain: search dc -> search sp (5 workers) -> held-out eval of both new tables (n=0/4/8, seed 20240601). Run from the repo root on the server.
cd ~/maddpg-adversarial-experiments; R=$(pwd); O=host_data/results/E031; mkdir -p $O
docker run --rm -v $R/$O:/r maddpg-exp:latest chown -R $(id -u):$(id -g) /r >/dev/null 2>&1
search() { c=$1; ld=$2; tag=$3
  docker run --rm --name maddpg-E031-search-$tag -e TQDM_DISABLE=1 --cpus 5 -v $R/host_data:/workspace/data -v $R/src:/workspace/src -v $R/configs:/workspace/configs -v $R/tools:/workspace/tools -w /workspace --entrypoint python3 maddpg-exp:latest tools/static_table_search.py --src /workspace/src --config configs/$c.json --load $ld --episodes 16 --eps 0.5 --workers 5 --failures 0,4 --start data/results/E026_$tag/table_best.json --out-dir data/results/E031_$tag > ~/idqn/maddpg-E031-search-$tag.log 2>&1
  docker run --rm -v $R/host_data/results/E031_$tag:/r maddpg-exp:latest chown -R $(id -u):$(id -g) /r >/dev/null 2>&1
  echo "$(date -u +%m-%d\ %H:%M) E031 search $tag done" >> ~/idqn/run_E031.done; }
ev() { c=$1; ld=$2; tag=$3; run=$4
  TABLE_FILE=$R/host_data/results/E031_$tag/table_best.json DOCKER_EXTRA="--cpus 2" LOAD=$ld EVAL_NAME=E031_$tag NO_ATTR=1 RULES="table random+flow greedy" TQDM_DISABLE=1 \
    bash tools/run_eval.sh $run configs/$c.json LC-Simple "0 4 8" > ~/idqn/maddpg-E031-eval-$tag.log 2>&1
  echo "$(date -u +%m-%d\ %H:%M) E031 eval $tag done" >> ~/idqn/run_E031.done; }
search v2_deltacom 8 dc; ev v2_deltacom 8 dc v2_dc_lc
search v2_pilot4_lc 3 sp; ev v2_pilot4_lc 3 sp v2_p4_lc
